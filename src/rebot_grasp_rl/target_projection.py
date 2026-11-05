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
def projectTarget(tc, nfd, nwd, nori, ns, na, rewards, bootstrap, gamma):
    tLogits = tc.logits(nfd, nwd, nori, ns, na)
    tProbs = F.softmax(tLogits, dim=-1); tSupport = tc.support
    tVMin, tVMax, tA = tc.v_min, tc.v_max, tc.num_atoms
    tQ, tNn, _ = tProbs.shape
    tDz = (tVMax - tVMin) / (tA - 1)
    tTz = rewards.view(1, tNn, 1) + bootstrap.view(1, tNn, 1) * gamma * tSupport.view(1, 1, tA)
    tTz = tTz.clamp(tVMin, tVMax).expand(tQ, tNn, tA)
    tB = (tTz - tVMin) / tDz; tL = tB.floor().long(); tU = tB.ceil().long()
    tIsInt = (tU == tL)
    tL = torch.where((tL > 0) & tIsInt, tL - 1, tL); tU = torch.where((tL == 0) & tIsInt, tU + 1, tU)
    tL = tL.clamp(0, tA - 1); tU = tU.clamp(0, tA - 1)
    tM = torch.zeros_like(tProbs)
    tM.scatter_add_(2, tL, tProbs * (tU.float() - tB)); tM.scatter_add_(2, tU, tProbs * (tB - tL.float()))
    return tM
