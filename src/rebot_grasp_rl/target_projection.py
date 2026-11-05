"""tp_net: C51 distributional-target projection for the two-phase SAC trainer.

`project_target` is extracted verbatim from train_sac_det_xattn.py. It is a pure
function of its arguments (the target critic `tc` is passed in), so it has no
module-global coupling. Used by the live TWO_PHASE agent update (`_upd`).

The C51-SAC actor/critic classes (DetActor, DetCriticC51) live in det_xattn_nets.py
and are imported by the entry file directly.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def project_target(tc, nfd, nwd, nori, ns, na, rewards, bootstrap, gamma):
    logits = tc.logits(nfd, nwd, nori, ns, na)
    probs = F.softmax(logits, dim=-1); support = tc.support
    v_min, v_max, A = tc.v_min, tc.v_max, tc.num_atoms
    Q, Nn, _ = probs.shape
    dz = (v_max - v_min) / (A - 1)
    tz = rewards.view(1, Nn, 1) + bootstrap.view(1, Nn, 1) * gamma * support.view(1, 1, A)
    tz = tz.clamp(v_min, v_max).expand(Q, Nn, A)
    b = (tz - v_min) / dz; l = b.floor().long(); u = b.ceil().long()
    is_int = (u == l)
    l = torch.where((l > 0) & is_int, l - 1, l); u = torch.where((l == 0) & is_int, u + 1, u)
    l = l.clamp(0, A - 1); u = u.clamp(0, A - 1)
    m = torch.zeros_like(probs)
    m.scatter_add_(2, l, probs * (u.float() - b)); m.scatter_add_(2, u, probs * (b - l.float()))
    return m
