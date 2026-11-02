"""Thread-safe Blackboard for the deploy control stack (3-thread refactor, docs/28).

The ONLY shared state between the Vision / Robot-IO / IK-FK-Control / Dashboard threads.
No thread calls another thread's methods -- all coupling is a blackboard read/write. This
mirrors the two copy-in/copy-out patterns already proven in-tree (AsyncPerception.snapshot,
DebugDashboard._lock).

Design (docs/28):
  - ONE coarse lock (rates are <=30Hz; per-section locks are a later optimization if profiling
    shows IO-write vs control-telemetry contention -- it won't at these rates).
  - publish(**kw): writer merges keys under the lock.
  - snapshot(): reader returns a DEEP copy (so a nested dict like perception's `tm`, or a numpy
    array, can't be mutated under the writer -- the shallow-copy bug noted in the dashboard map).
  - Every producer section carries `ts` (freshness) + optional `dead` (liveness). The control
    thread's stale/dead fail-stop is a SAFETY contract, so freshness lives in the blackboard.

Sections (writer -> readers):
  proprio        IO      -> Control            arm/grip qpos/qvel/torque, is_grasped, tcp_io, ts
  perception     Vision  -> Control            the detector `snap` dict + captured cube_world/raw + ts/hz/dead
  perception_ctx Control -> Vision             tcp_prior, need_align, cap_req, ov_queries (control->vision back-channel)
  command        Control -> IO                 raw IK target arm[6] (UNSHAPED), grip, grip_mode, grip_torque_target, arm_snap, seq
  mode           Control -> IO (some)          mode enum, estop, io_task/io_task_state, cube_target (debug)
  debug          all     -> Dashboard          per-producer telemetry sub-dicts (gates.io/perc/ctrl)

Values MUST be plain (float/int/bool/str/list/tuple/dict/np.ndarray) -- NOT torch tensors and
NOT camera frames (frames go through DebugDashboard.update_frame, never the blackboard), so the
deep copy stays cheap.
"""

import copy
import threading


class _Section:
    """One named region of the blackboard. publish() merges; snapshot() deep-copies out."""

    __slots__ = ("_lock", "_d", "_name")

    def __init__(self, name, lock):
        self._name = name
        self._lock = lock
        self._d = {}

    def publish(self, **kw):
        """Merge keys into the section (writer). Overwrites existing keys, keeps the rest."""
        with self._lock:
            self._d.update(kw)

    def set(self, mapping):
        """Merge a dict (same as publish but for a pre-built dict, e.g. the perception `snap`)."""
        with self._lock:
            self._d.update(mapping)

    def snapshot(self):
        """Return a DEEP copy of the whole section (reader). Safe to mutate; won't touch the source."""
        with self._lock:
            return copy.deepcopy(self._d)

    def get(self, key, default=None):
        """Deep-copied single value (reader) -- avoids snapshotting the whole section for one field."""
        with self._lock:
            return copy.deepcopy(self._d.get(key, default))

    def __repr__(self):
        with self._lock:
            return f"<Section {self._name}: {sorted(self._d)}>"


class Blackboard:
    """The single shared-state object. One coarse RLock across all sections (docs/28)."""

    def __init__(self):
        self._lock = threading.RLock()
        # producer -> reader (see module docstring)
        self.proprio = _Section("proprio", self._lock)          # IO      -> Control
        self.perception = _Section("perception", self._lock)     # Vision  -> Control
        self.perception_ctx = _Section("perception_ctx", self._lock)  # Control -> Vision
        self.command = _Section("command", self._lock)           # Control -> IO
        self.mode = _Section("mode", self._lock)                 # Control -> IO(some)
        self.debug = _Section("debug", self._lock)               # all     -> Dashboard

    def sections(self):
        return {"proprio": self.proprio, "perception": self.perception,
                "perception_ctx": self.perception_ctx, "command": self.command,
                "mode": self.mode, "debug": self.debug}


def _self_test():
    """Validate the copy semantics the refactor depends on (run: python deploy/blackboard.py)."""
    import numpy as np
    bb = Blackboard()

    # publish + snapshot round-trip
    bb.proprio.publish(arm_qpos=np.array([1.0, 2.0, 3.0]), ts=123.0, is_grasped=False)
    snap = bb.proprio.snapshot()
    assert snap["ts"] == 123.0 and snap["is_grasped"] is False
    assert list(snap["arm_qpos"]) == [1.0, 2.0, 3.0]

    # mutating a snapshot must NOT affect the blackboard (deep-copy contract)
    snap["arm_qpos"][0] = 999.0
    snap["ts"] = 0.0
    assert bb.proprio.get("arm_qpos")[0] == 1.0, "snapshot mutation leaked into the blackboard!"
    assert bb.proprio.get("ts") == 123.0

    # nested dict (mirrors perception `tm`) must be deep-copied, not shared
    bb.perception.set({"tm": {"det_w": 31, "front": 44}, "hz": 14.2})
    s2 = bb.perception.snapshot()
    s2["tm"]["det_w"] = -1
    assert bb.perception.get("tm")["det_w"] == 31, "nested dict was shallow-copied (the known bug)!"

    # partial publish merges (does not clobber other keys)
    bb.command.publish(arm=[0.0] * 6, grip=0.03, seq=1)
    bb.command.publish(seq=2)                     # only seq changes
    c = bb.command.snapshot()
    assert c["seq"] == 2 and c["grip"] == 0.03 and len(c["arm"]) == 6

    print("blackboard self-test: OK (deep-copy snapshot + merge semantics verified)")


if __name__ == "__main__":
    _self_test()
