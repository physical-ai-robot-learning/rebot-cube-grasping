"""Canonical scripted ~28-deg top-down pick trajectory (single source of truth).

The arm joint poses below are the proven ~28-deg compromise pick (poses found by
scripts/probes/probe_vertical.py + probe_graspxy.py; 100% scripted at the aligned
table_center). ~28 deg slides down the cube SIDE (no vertical plunge-knock) and is far
more top-down than the old ~41-deg lean.

These are shared by scripts/collect_demos.py (RLPD demo collection) and
scripts/render_demo.py (demo visualization) so the two stay in lockstep. APPROACH MUST
match ``rest_arm`` in configs/train.yaml (the pre-grasp hover) — keep them in sync.

Poses are plain python lists (arm joints joint1..joint6, rad); callers build their own
torch tensors. Phase step-counts are the default scripted timing.
"""
from __future__ import annotations

# arm joint poses (joint1..joint6, rad)
APPROACH = [0.0, -1.65, -1.15, 0.60, 0.0, 0.0]   # mirrors config rest_arm (the pre-grasp hover, ~27 deg)
GRASP    = [0.0, -1.85, -1.00, 0.40, 0.0, 0.0]   # ~25 deg, tcp at the cube
LIFT     = [0.0, -1.55, -1.30, 0.80, 0.0, 0.0]   # ~30 deg, +13cm

# default scripted phase step-counts
APPROACH_STEPS = 12   # settle at the lifted approach/hover pose
DESCEND_STEPS = 24    # open fingers descend onto the cube (toward GRASP)
CLOSE_STEPS = 18      # close the gripper on the cube
LIFT_STEPS = 70       # lift the held cube

# ---- scripted RESET brackets (shared by deploy/run_pick.py + the online-RLPD reset harness) ----
# These bracket the LEARNED policy at both ends so the real loop is deterministic + autonomous:
#   episode entry  : initial lift-up  ZERO -> APPROACH (open gripper, raise to the pre-grasp hover)
#   episode/park end: comeback to zero  current -> APPROACH -> ZERO (lift clear, then park at home)
# They are SCRIPTED position ramps (NOT learned) -> identical in sim and on the real arm, so the same
# Zero<->Rest trajectory also serves as a free system-ID probe of the sim2real dynamics gap.
ZERO = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # URDF/mechanical home (== LeRobot calibrated zero); safe park target
LIFTUP_STEPS = 30     # entry: current -> APPROACH at the 15 Hz control rate (~2 s)
COMEBACK_STEPS = 36   # return: current -> APPROACH -> ZERO (~2.4 s)


def cosine_ease(n: int):
    """n global ease-in/out weights s_k in (0,1] -> bounded joint velocity at both endpoints (smooth
    accel/decel, gentle on the real motors). Pure python so sim and deploy stay byte-for-byte in sync."""
    import math
    return [0.5 - 0.5 * math.cos(math.pi * (k + 1) / n) for k in range(n)]


def scripted_joint_path(vias, n_steps: int):
    """Piecewise-linear joint-space path through `vias` (list of arm poses joint1..6, incl. start & end),
    re-sampled at `n_steps` cosine-eased points along cumulative arclength. Returns a list of arm-pose
    lists. Same waypoints drive sim (set_joint_position_targets) and deploy (io.command)."""
    import math
    P = [[float(x) for x in v] for v in vias]
    d = len(P[0])
    seg = [math.sqrt(sum((P[i + 1][j] - P[i][j]) ** 2 for j in range(d))) for i in range(len(P) - 1)]
    total = sum(seg) or 1.0
    cum = [0.0]
    for s in seg:
        cum.append(cum[-1] + s)
    out = []
    for s in cosine_ease(n_steps):
        dist = s * total
        i = 0
        while i < len(seg) - 1 and dist > cum[i + 1]:
            i += 1
        f = 0.0 if seg[i] < 1e-9 else (dist - cum[i]) / seg[i]
        out.append([P[i][j] + (P[i + 1][j] - P[i][j]) * f for j in range(d)])
    return out
