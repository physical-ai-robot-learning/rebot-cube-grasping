"""C51 distributional Bellman target, projected onto the critic's fixed value support (was src/det_xattn/tp_net.py).

Pure function of its arguments: the target critic is passed in, so there is no module state. Used by the two-phase
agent update of rebot_grasp_rl.train. Input: target critic `tc` (logits(), support, v_min, v_max, num_atoms), next
observations (front / wrist detections, cube orientation, state), next actions, rewards (N,), bootstrap mask (N,) and
the discount. Output: projected target distribution, shape (num_q, N, num_atoms).
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def projectTarget(tc, nfd, nwd, nori, ns, na, rewards, bootstrap, gamma):
    tLogits = tc.logits(nfd, nwd, nori, ns, na)
    tProbs = F.softmax(tLogits, dim=-1)
    tSupport = tc.support
    tVMin, tVMax, tA = tc.v_min, tc.v_max, tc.num_atoms
    tQ, tNn, _ = tProbs.shape
    tDz = (tVMax - tVMin) / (tA - 1)
    tTz = rewards.view(1, tNn, 1) + bootstrap.view(1, tNn, 1) * gamma * tSupport.view(1, 1, tA)
    tTz = tTz.clamp(tVMin, tVMax).expand(tQ, tNn, tA)
    tB = (tTz - tVMin) / tDz
    tL = tB.floor().long()
    tU = tB.ceil().long()
    tIsInt = (tU == tL)
    # an atom that lands exactly on a support point would get zero mass from both neighbours; shift one index
    tL = torch.where((tL > 0) & tIsInt, tL - 1, tL)
    tU = torch.where((tL == 0) & tIsInt, tU + 1, tU)
    tL = tL.clamp(0, tA - 1)
    tU = tU.clamp(0, tA - 1)
    tM = torch.zeros_like(tProbs)
    tM.scatter_add_(2, tL, tProbs * (tU.float() - tB))
    tM.scatter_add_(2, tU, tProbs * (tB - tL.float()))
    return tM
