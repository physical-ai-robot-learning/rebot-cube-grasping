"""Hand-computed unit test for NStepReplayBuffer (docs/06 §12⑧).

Verifies the subtle per-env truncation/bootstrap logic with exact float arithmetic:
  - R_n = sum_k gamma^k r_k (alive-masked), disc = gamma**n_eff, done_n = terminal at boundary
  - true termination within the window  -> done_n=1, TD target == ret (bootstrap killed)
  - TIMEOUT truncation within the window -> done_n=0 (bootstrap SURVIVES) AND obs_n frozen at the
    boundary's terminal obs (NOT a post-reset obs)  [the docs §12②/⑧ correctness fix]
  - obs_n / next_priv frozen at the last alive step
  - n_step=1 is behavior-equivalent to the flat ReplayBuffer
  - priv (asymmetric critic block) carried + frozen alongside obs_n

torch only, no Isaac. Run: uv run --no-sync python tests/test_nstep_buffer.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from rebot_sim2real.rl.nstep_buffer import NStepReplayBuffer
from rebot_sim2real.rl.sac import ReplayBuffer

DEV = "cpu"
TOL = 1e-6


def _t(x):
    return torch.tensor(x, dtype=torch.float32, device=DEV)


def _row(vals):
    return torch.tensor([vals], dtype=torch.float32, device=DEV)  # [1, D]


def _close(a, b, msg):
    a = float(a); b = float(b)
    assert abs(a - b) < TOL, f"{msg}: {a} != {b}"


def add(buf, o, a, r, no, done, term, priv=None, npriv=None):
    buf.add_batch(_row(o), _row(a), _t([r]), _row(no), _t([done]), terminal=_t([term]),
                  priv=(_row(priv) if priv is not None else None),
                  next_priv=(_row(npriv) if npriv is not None else None))


def get_one(buf):
    """Read the single most-recently pushed transition (ptr-1)."""
    i = (buf.ptr - 1) % buf.cap
    return dict(obs=buf.obs[i], priv=buf.priv[i], act=buf.act[i], ret=float(buf.ret[i]),
                next_obs=buf.next_obs[i], next_priv=buf.next_priv[i],
                disc=float(buf.disc[i]), done=float(buf.done[i]))


def test_no_termination():
    g = 0.5
    buf = NStepReplayBuffer(64, obs_dim=2, act_dim=1, n_step=3, gamma=g, device=DEV, priv_dim=2)
    # steps 0..3 (step0 emits when window length exceeds n=3). obs_k=[k,k], no_k=[k+1,k+1], priv_k=[10k,..]
    rews = [1.0, 2.0, 3.0, 4.0]
    for k in range(4):
        add(buf, [k, k], [k], rews[k], [k + 1, k + 1], done=0, term=0,
            priv=[10 * k, 10 * k], npriv=[10 * k + 1, 10 * k + 1])
    tr = get_one(buf)  # transition for step0 (last popped on step3) -> but get_one reads ptr-1 = step0's push
    # step0 emitted first (ptr advanced once), step1 not yet (needs step4). So ptr==1, get_one -> step0.
    assert buf.ptr == 1, f"expected 1 push, got {buf.ptr}"
    exp_ret = rews[0] + g * rews[1] + g ** 2 * rews[2]   # 1 + 1 + 0.75 = 2.75
    _close(tr["ret"], exp_ret, "no-term ret")
    _close(tr["disc"], g ** 3, "no-term disc (n_eff=3)")
    _close(tr["done"], 0.0, "no-term done_n")
    assert torch.allclose(tr["next_obs"], _t([3.0, 3.0])), f"obs_n should be no[2]=[3,3], got {tr['next_obs']}"
    assert torch.allclose(tr["obs"], _t([0.0, 0.0])), "obs0"
    assert torch.allclose(tr["priv"], _t([0.0, 0.0])), "priv0"
    assert torch.allclose(tr["next_priv"], _t([21.0, 21.0])), f"priv_n=npriv[2]=[21,21], got {tr['next_priv']}"
    print("[ok] no_termination: ret=%.4f disc=%.4f done_n=%.0f obs_n frozen at step2" % (tr["ret"], tr["disc"], tr["done"]))


def test_true_termination():
    g = 0.5
    buf = NStepReplayBuffer(64, obs_dim=2, act_dim=1, n_step=3, gamma=g, device=DEV, priv_dim=0)
    # boundary (true terminal) at the SECOND window step (index 1). done1=1, terminal1=1.
    seq = [(1.0, 0, 0), (2.0, 1, 1), (3.0, 0, 0), (4.0, 0, 0)]
    for k, (r, d, term) in enumerate(seq):
        add(buf, [k, k], [k], r, [k + 1, k + 1], done=d, term=term)
    tr = get_one(buf)
    exp_ret = 1.0 + g * 2.0          # r0 + g*r1 (alive zeroes r2)
    _close(tr["ret"], exp_ret, "true-term ret")
    _close(tr["disc"], g ** 2, "true-term disc (n_eff=2)")
    _close(tr["done"], 1.0, "true-term done_n")
    assert torch.allclose(tr["next_obs"], _t([2.0, 2.0])), f"obs_n frozen at no[1]=[2,2], got {tr['next_obs']}"
    # TD target with done_n=1 must equal ret regardless of Q(obs_n)
    Q = 999.0
    target = tr["ret"] + tr["disc"] * (1.0 - tr["done"]) * Q
    _close(target, exp_ret, "true-term target==ret (bootstrap killed)")
    print("[ok] true_termination: ret=%.4f disc=%.4f done_n=1 target==ret" % (tr["ret"], tr["disc"]))


def test_timeout_truncation():
    g = 0.5
    buf = NStepReplayBuffer(64, obs_dim=2, act_dim=1, n_step=3, gamma=g, device=DEV, priv_dim=0)
    # TIMEOUT at window step index 1: done1=1 (episode boundary) but terminal1=0 (NOT a true terminal).
    seq = [(1.0, 0, 0), (2.0, 1, 0), (3.0, 0, 0), (4.0, 0, 0)]
    for k, (r, d, term) in enumerate(seq):
        add(buf, [k, k], [k], r, [k + 1, k + 1], done=d, term=term)
    tr = get_one(buf)
    exp_ret = 1.0 + g * 2.0
    _close(tr["ret"], exp_ret, "timeout ret")
    _close(tr["disc"], g ** 2, "timeout disc (n_eff=2)")
    _close(tr["done"], 0.0, "timeout done_n (bootstrap survives)")
    # CRITICAL: obs_n is the boundary's terminal obs (no[1]), NOT a post-boundary obs (no[2]/no[3])
    assert torch.allclose(tr["next_obs"], _t([2.0, 2.0])), f"obs_n must be the timeout obs no[1]=[2,2], got {tr['next_obs']}"
    Q = 7.0
    target = tr["ret"] + tr["disc"] * (1.0 - tr["done"]) * Q   # bootstraps off Q(obs_n)
    _close(target, exp_ret + g ** 2 * Q, "timeout target bootstraps off obs_n")
    print("[ok] timeout_truncation: done_n=0, obs_n frozen at boundary, target bootstraps")


def test_nstep1_equiv_flat():
    g = 0.97
    nb = NStepReplayBuffer(64, obs_dim=2, act_dim=1, n_step=1, gamma=g, device=DEV, priv_dim=0)
    fb = ReplayBuffer(64, obs_dim=2, act_dim=1, device=DEV, priv_dim=0, gamma=g)
    seq = [(0, 1.5, 0, 0), (1, -2.0, 1, 1), (2, 3.0, 0, 0)]
    for (k, r, d, term) in seq:
        add(nb, [k, k], [k], r, [k + 1, k + 1], done=d, term=term)
        fb.add_batch(_row([k, k]), _row([k]), _t([r]), _row([k + 1, k + 1]), _t([d]), terminal=_t([term]))
    # nstep(1) emits step0 and step1 (needs len>1). flat stores all 3. Compare the first two.
    assert nb.ptr == 2, f"nstep(1) should have pushed 2, got {nb.ptr}"
    for i in range(2):
        ns = (nb.obs[i], nb.act[i], float(nb.ret[i]), nb.next_obs[i], float(nb.disc[i]), float(nb.done[i]))
        fs = (fb.obs[i], fb.act[i], float(fb.rew[i]), fb.next_obs[i], g, float(fb.done[i]))
        assert torch.allclose(ns[0], fs[0]) and torch.allclose(ns[3], fs[3]), f"obs/next mismatch @ {i}"
        _close(ns[2], fs[2], f"ret==rew @ {i}")
        _close(ns[4], fs[4], f"disc==gamma @ {i}")
        _close(ns[5], fs[5], f"done @ {i}")
    print("[ok] nstep1_equiv_flat: NStep(n=1) == flat ReplayBuffer (ret/disc/done/obs)")


def test_legacy_terminal_eq_done():
    """terminal defaulting to done reproduces the OLD gamma**n behavior: the only window with
    n_eff<n has done_n=1 (masked), so gamma**n_eff vs gamma**n is indistinguishable in the target."""
    g = 0.9
    buf = NStepReplayBuffer(64, obs_dim=1, act_dim=1, n_step=3, gamma=g, device=DEV, priv_dim=0)
    # done at step1, NO explicit terminal (defaults to done) -> done_n=1
    seq = [(1.0, 0), (2.0, 1), (3.0, 0), (4.0, 0)]
    for k, (r, d) in enumerate(seq):
        buf.add_batch(_row([k]), _row([k]), _t([r]), _row([k + 1]), _t([d]))  # no terminal kwarg
    tr = get_one(buf)
    _close(tr["done"], 1.0, "legacy done_n==done")
    Q = 123.0
    target = tr["ret"] + tr["disc"] * (1.0 - tr["done"]) * Q
    _close(target, 1.0 + g * 2.0, "legacy target==ret (masked, gamma**n_eff irrelevant)")
    print("[ok] legacy_terminal_eq_done: terminal=done reproduces old masked behavior")


if __name__ == "__main__":
    test_no_termination()
    test_true_termination()
    test_timeout_truncation()
    test_nstep1_equiv_flat()
    test_legacy_terminal_eq_done()
    print("\nALL NSTEP BUFFER TESTS PASSED")
