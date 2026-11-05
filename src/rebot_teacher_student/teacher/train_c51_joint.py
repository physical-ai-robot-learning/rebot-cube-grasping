"""JOINT 3-stage hierarchical C51 teacher (state-only): THREE actor-critics trained SIMULTANEOUSLY.

Stage: June teacher (jointly trained stages; scripts/teacher/train_joint_teacher.sh).
Was src/rebot_sim2real/teacher/train_c51_joint.py (moved and renamed to the OA style on 2026-11-05; docs/refactor_2026-11.md).
Run: python -m rebot_teacher_student.teacher.train_c51_joint
Needs Isaac Sim 5.1 / Isaac Lab 2.3 + the LeIsaac reBot fork; not executable in CI (compile-checked only).

Each env has a stage pointer (0 approach -> 1 grasp -> 2 place); the ACTIVE stage's actor acts, the env
advances on that stage's done-condition (approach=align-hold latch, grasp=_is_grasped, place=is_fold_
success), and each stage's transitions go into ITS OWN replay buffer (full env reward + a bonus when that
stage's done is first reached). All 3 critics+actors update every step -> the stages CO-ADAPT (the grasp
policy learns from the approach policy's live end-states, place from grasp's). End-to-end success = an
env that reaches stage 2 and hits place-done. At deploy the 3 actors run in sequence by stage.

Run (LIVE-watch in the GUI on :1 -> omit --headless):
  PYTHONPATH=src DISPLAY=:1 python -m rebot_teacher_student.teacher.train_c51_joint --num-envs 1024 --save-dir rl_output/rl_c51_joint
"""
import argparse, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rebot_grasp_common.paths import LEISAAC_ASSETS

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT", LEISAAC_ASSETS)

ap = argparse.ArgumentParser()
ap.add_argument("--num-envs", type=int, default=1024)
ap.add_argument("--total-timesteps", type=int, default=1_000_000_000)
ap.add_argument("--http-port", type=int, default=8890)
ap.add_argument("--save-dir", type=str, required=True)
ap.add_argument("--target-cube", type=str, default="cube")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--success-rate-target", type=float, default=0.9)
ap.add_argument("--min-sr-episodes", type=int, default=300)
ap.add_argument("--min-sr-hits", type=int, default=3)
ap.add_argument("--bonus", type=float, default=5.0)
ap.add_argument("--warm", nargs="*", default=[], help="warm-start ckpt(s): 1 path=all 3 stages, or 3 paths=s0 s1 s2 (loads actor+critic+log_alpha)")
ap.add_argument("--config", type=str, default="configs/reward_config.yaml")
ap.add_argument("--buffer-size", type=int, default=600_000)
ap.add_argument("--batch-size", type=int, default=512)
ap.add_argument("--num-updates", type=int, default=16)
ap.add_argument("--learning-starts", type=int, default=5000)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--gamma", type=float, default=0.9)
ap.add_argument("--tau", type=float, default=0.01)
ap.add_argument("--policy-frequency", type=int, default=4)
ap.add_argument("--num-atoms", type=int, default=101)
ap.add_argument("--v-min", type=float, default=-20.0)
ap.add_argument("--v-max", type=float, default=20.0)
ap.add_argument("--num-q", type=int, default=2)
args, _ = ap.parse_known_args()

import yaml
CFG = {}
if os.path.exists(args.config):
    with open(args.config) as f:
        CFG = yaml.safe_load(f) or {}

from isaaclab.app import AppLauncher
simulation_app = AppLauncher({"headless": args.headless, "enable_cameras": False}).app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import SceneEntityCfg, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
import leisaac.tasks.lift_cube  # noqa: F401
from leisaac.tasks.lift_cube import mdp
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from rebot_teacher_student.env.env_config import configureEnv, applyMotorLimits
import rebot_teacher_student.env.rewards as MR
from rebot_teacher_student import dashboard as dash

DEVICE = "cuda:0"; TARGET = args.target_cube; N = args.num_envs; NS = 4   # reaching -> align -> grasp -> lift
env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=N)
env_cfg = configureEnv(env_cfg, CFG, mdp, RewTerm, DoneTerm, SceneEntityCfg, target_cube=TARGET)
_obj = getattr(env_cfg.scene, TARGET); _cp = _obj.prim_path
env_cfg.scene.gripper_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_left",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
env_cfg.scene.jaw_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_right",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
_obj.spawn.activate_contact_sensors = True
env_cfg.scene.robot.spawn.activate_contact_sensors = True
for cam in ("front", "side", "wrist"):
    if hasattr(env_cfg.scene, cam):
        setattr(env_cfg.scene, cam, None)
for attr in list(vars(env_cfg.observations.policy)):
    if attr in ("front", "wrist", "side") or "image" in attr.lower():
        setattr(env_cfg.observations.policy, attr, None)
try:
    env_cfg.viewer.origin_type = "env"; env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (1.2, 1.0, 0.8); env_cfg.viewer.lookat = (0.3, -0.3, 0.15)
except Exception:
    pass

env = ManagerBasedRLEnv(cfg=env_cfg)
applyMotorLimits(env, CFG)
sb3_env = Sb3VecEnvWrapper(env)
obs_dim = int(np.prod(sb3_env.observation_space.shape)); act_dim = int(np.prod(sb3_env.action_space.shape))
print(f"[JOINT] obs={obs_dim} act={act_dim} envs={N} stages={NS}", flush=True)
dash.start(args.http_port, targetCube=TARGET)
os.makedirs(args.save_dir, exist_ok=True)


def orthogonalInit(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Actor(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, ns, na, h=256):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(ns, 256), nn.LayerNorm(256), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(256, h), nn.LayerNorm(h), nn.ReLU(),
            nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(), nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU())
        self.mean = nn.Linear(h, na); self.logstd = nn.Linear(h, na); self.apply(orthogonalInit)

    def _ml(self, s):
        tX = self.fc(self.proj(s)); tM = self.mean(tX)
        tLs = torch.tanh(self.logstd(tX)); tLs = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (tLs + 1)
        return tM, tLs

    def getAction(self, s):
        tM, tLs = self._ml(s); tStd = tLs.exp(); tNrm = torch.distributions.Normal(tM, tStd)
        tX = tNrm.rsample(); tY = torch.tanh(tX)
        tLogp = (tNrm.log_prob(tX) - torch.log(1 - tY.pow(2) + 1e-6)).sum(1, keepdim=True)
        return tY, tLogp

    @torch.no_grad()
    def actDeterministic(self, s):
        tM, _ = self._ml(s); return torch.tanh(tM)


class CriticC51(nn.Module):
    def __init__(self, ns, na, atoms, vmin, vmax, nq=2, h=512):
        super().__init__()
        self.num_atoms, self.num_q, self.v_min, self.v_max = atoms, nq, vmin, vmax
        self.register_buffer("support", torch.linspace(vmin, vmax, atoms))
        self.proj = nn.Sequential(nn.Linear(ns, 256), nn.LayerNorm(256), nn.ReLU())
        def qn():
            return nn.Sequential(nn.Linear(256 + na, h), nn.LayerNorm(h), nn.ReLU(),
                nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(), nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(),
                nn.Linear(h, atoms))
        self.qs = nn.ModuleList([qn() for _ in range(nq)]); self.apply(orthogonalInit)

    def logits(self, s, a):
        tX = torch.cat([self.proj(s), a], dim=-1); return torch.stack([tQ2(tX) for tQ2 in self.qs], dim=0)

    def qValues(self, s, a):
        return (F.softmax(self.logits(s, a), dim=-1) * self.support).sum(-1)


@torch.no_grad()
def projectTarget(tc, ns_, na_, rew, boot, gamma):
    tProbs = F.softmax(tc.logits(ns_, na_), dim=-1); tSup = tc.support
    tVmin, tVmax, tA = tc.v_min, tc.v_max, tc.num_atoms; tQ, tNn, _ = tProbs.shape; tDz = (tVmax - tVmin) / (tA - 1)
    tTz = (rew.view(1, tNn, 1) + boot.view(1, tNn, 1) * gamma * tSup.view(1, 1, tA)).clamp(tVmin, tVmax).expand(tQ, tNn, tA)
    tB = (tTz - tVmin) / tDz; tL = tB.floor().long(); tU = tB.ceil().long(); tIi = (tU == tL)
    tL = torch.where((tL > 0) & tIi, tL - 1, tL); tU = torch.where((tL == 0) & tIi, tU + 1, tU)
    tL = tL.clamp(0, tA - 1); tU = tU.clamp(0, tA - 1); tM = torch.zeros_like(tProbs)
    tM.scatter_add_(2, tL, tProbs * (tU.float() - tB)); tM.scatter_add_(2, tU, tProbs * (tB - tL.float())); return tM


class Buffer:
    def __init__(self, cap, od, ad, dev):
        self.cap, self.device, self.ptr, self.full = cap, dev, 0, False
        self.obs = torch.zeros(cap, od, device=dev); self.act = torch.zeros(cap, ad, device=dev)
        self.rew = torch.zeros(cap, device=dev); self.nxt = torch.zeros(cap, od, device=dev)
        self.done = torch.zeros(cap, device=dev)

    def add(self, o, a, r, n, d):
        tB2 = o.shape[0]
        if tB2 == 0:
            return
        tIdx = (torch.arange(tB2, device=self.device) + self.ptr) % self.cap
        self.obs[tIdx] = o; self.act[tIdx] = a; self.rew[tIdx] = r; self.nxt[tIdx] = n; self.done[tIdx] = d
        tNp = self.ptr + tB2
        self.full = self.full or tNp >= self.cap; self.ptr = tNp % self.cap

    def ready(self, bs):
        return self.full or self.ptr >= bs

    def sample(self, bs):
        tHi = self.cap if self.full else self.ptr; tIdx = torch.randint(0, tHi, (bs,), device=self.device)
        return self.obs[tIdx], self.act[tIdx], self.rew[tIdx], self.nxt[tIdx], self.done[tIdx]


# 3 actor-critics
actors = [Actor(obs_dim, act_dim).to(DEVICE) for _ in range(NS)]
critics = [CriticC51(obs_dim, act_dim, args.num_atoms, args.v_min, args.v_max, args.num_q).to(DEVICE) for _ in range(NS)]
targets = [CriticC51(obs_dim, act_dim, args.num_atoms, args.v_min, args.v_max, args.num_q).to(DEVICE) for _ in range(NS)]
for s in range(NS):
    targets[s].load_state_dict(critics[s].state_dict())
    for p in targets[s].parameters():
        p.requires_grad_(False)
a_opt = [torch.optim.Adam(actors[tS].parameters(), lr=args.lr) for tS in range(NS)]
c_opt = [torch.optim.Adam(critics[tS].parameters(), lr=args.lr) for tS in range(NS)]
log_alpha = [torch.zeros(1, requires_grad=True, device=DEVICE) for _ in range(NS)]
al_opt = [torch.optim.Adam([log_alpha[tS]], lr=args.lr) for tS in range(NS)]
alpha = [log_alpha[tS].exp().detach() for tS in range(NS)]
target_entropy = -float(act_dim)
bufs = [Buffer(args.buffer_size, obs_dim, act_dim, DEVICE) for _ in range(NS)]

# warm-start each stage's actor+critic (+ converged alpha so the warm policy isn't wrecked by high-entropy exploration)
if args.warm:
    wpaths = args.warm if len(args.warm) == NS else [args.warm[0]] * NS
    for s in range(NS):
        ck = torch.load(wpaths[s], map_location=DEVICE)
        actors[s].load_state_dict(ck["actor"])
        try:
            critics[s].load_state_dict(ck["critic"]); targets[s].load_state_dict(ck["critic"])
        except Exception as e:
            print(f"[JOINT] warm s{s}: critic load SKIPPED ({e})", flush=True)
        if "log_alpha" in ck:
            with torch.no_grad():
                log_alpha[s].copy_(torch.as_tensor(ck["log_alpha"], device=DEVICE).reshape(1))
            alpha[s] = log_alpha[s].exp().detach()
        print(f"[JOINT] warm s{s} <- {wpaths[s]} (alpha={float(alpha[s]):.3f})", flush=True)


def updateStage(s):
    tO, tA2, tR, tN, tD = bufs[s].sample(args.batch_size)
    with torch.no_grad():
        tNa, tNlogp = actors[s].getAction(tN); tBoot = (1.0 - tD)
        tREnt = tR - tBoot * args.gamma * (alpha[s].squeeze() * tNlogp.squeeze(-1))
        tTgt = projectTarget(targets[s], tN, tNa, tREnt, tBoot, args.gamma)
    tLogp = F.log_softmax(critics[s].logits(tO, tA2), dim=-1)
    tCloss = (-(tTgt * tLogp).sum(-1).mean(-1)).sum()
    c_opt[s].zero_grad(set_to_none=True); tCloss.backward(); c_opt[s].step()
    return tO


def updateActorStage(s, o):
    tApi, tLogp = actors[s].getAction(o); tQv = critics[s].qValues(o, tApi).mean(0)
    tAloss = (alpha[s].squeeze() * tLogp.squeeze(-1) - tQv).mean()
    a_opt[s].zero_grad(set_to_none=True); tAloss.backward(); a_opt[s].step()
    with torch.no_grad():
        _, tLp2 = actors[s].getAction(o)
    tAlloss = -(log_alpha[s].exp() * (tLp2 + target_entropy)).mean()
    al_opt[s].zero_grad(set_to_none=True); tAlloss.backward(); al_opt[s].step()


_ee = SceneEntityCfg("ee_frame"); _oc = SceneEntityCfg(TARGET)
def _dist():
    return torch.norm(env.scene[TARGET].data.root_pos_w - MR._tcp(env, _ee), dim=1)
def stageCond(s):
    # 4 stages: 0 reaching -> 1 align -> 2 grasp -> 3 lift. Each returns the env-wise done-condition.
    if s == 0:   # reaching done: gripper reached the cube vicinity (slightly looser than the align band)
        return _dist() < (MR.ALIGN_DIST + 0.05)
    if s == 1:   # align done: open jaws aligned-and-HELD over the cube (the align-hold latch)
        return MR._alignHeld(env, _dist())
    if s == 2:   # grasp done: contact-grasp (requires the align-hold latch -> grasp-after-align)
        return MR._isGrasped(env, _oc)
    return MR.isFoldSuccess(env, _oc)   # lift done: lifted + grasped + returned to REST = success


def toT(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=DEVICE)


print(f"[JOINT] start. sr-target={args.success_rate_target} hits={args.min_sr_hits} bonus={args.bonus}", flush=True)
obs = np.nan_to_num(np.asarray(sb3_env.reset(), dtype=np.float32), nan=0.0, posinf=100.0, neginf=-100.0)
obs_t = toT(obs)
stage = torch.zeros(N, dtype=torch.long, device=DEVICE)
bonus_given = torch.zeros(N, dtype=torch.bool, device=DEVICE)   # got this stage's bonus already
ep_succ = torch.zeros(N, dtype=torch.bool, device=DEVICE)       # reached place-done this episode
reached2 = torch.zeros(N, dtype=torch.bool, device=DEVICE)      # reached stage 2 this episode
gstep = 0; nit = 0; _sr_hits = 0; _last_hit_ep = -10**9; _rec = []; _ep = 0

while gstep < args.total_timesteps:
    pre_stage = stage.clone()
    action = torch.zeros(N, act_dim, device=DEVICE)
    for s in range(NS):
        m = (pre_stage == s)
        if m.any():
            if gstep < args.learning_starts:
                action[m] = torch.rand(int(m.sum()), act_dim, device=DEVICE) * 2 - 1
            else:
                with torch.no_grad():
                    a_s, _ = actors[s].getAction(obs_t[m])
                action[m] = a_s
    next_obs, rew, done, infos = sb3_env.step(action.detach().cpu().numpy().astype(np.float32))
    next_obs = np.nan_to_num(np.asarray(next_obs, dtype=np.float32), nan=0.0, posinf=100.0, neginf=-100.0)
    rew = np.nan_to_num(np.asarray(rew, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    next_t = toT(next_obs); rew_t = toT(rew)
    conds = [stageCond(0), stageCond(1), stageCond(2), stageCond(3)]
    # per-stage collect (full reward + this-stage done bonus, once)
    for s in range(NS):
        m = (pre_stage == s)
        if m.any():
            r_s = rew_t[m] + args.bonus * (conds[s][m] & ~bonus_given[m]).float()
            bufs[s].add(obs_t[m], action[m].detach(), r_s, next_t[m], torch.zeros(int(m.sum()), device=DEVICE))
    bonus_given |= ((conds[0] & (pre_stage == 0)) | (conds[1] & (pre_stage == 1)) |
                    (conds[2] & (pre_stage == 2)) | (conds[3] & (pre_stage == 3)))
    # advance stages 0->1->2->3 (reset that env's bonus flag on entering a fresh stage)
    adv0 = (stage == 0) & conds[0]; stage = torch.where(adv0, torch.ones_like(stage), stage)
    adv1 = (stage == 1) & conds[1]; stage = torch.where(adv1, torch.full_like(stage, 2), stage)
    adv2 = (stage == 2) & conds[2]; stage = torch.where(adv2, torch.full_like(stage, 3), stage)
    bonus_given = torch.where(adv0 | adv1 | adv2, torch.zeros_like(bonus_given), bonus_given)
    reached2 |= (stage == 3); ep_succ |= ((pre_stage == 3) & conds[3])   # end2end = reached lift-stage & lift-done

    dmask = torch.as_tensor(done, device=DEVICE).bool()
    if dmask.any():
        for i in torch.nonzero(dmask).flatten().tolist():
            _rec.append(1.0 if bool(ep_succ[i]) else 0.0); _ep += 1
        _rec = _rec[-500:]
        stage[dmask] = 0; bonus_given[dmask] = False; ep_succ[dmask] = False; reached2[dmask] = False

    obs = next_obs; obs_t = next_t; gstep += N; nit += 1

    if gstep >= args.learning_starts:
        for s in range(NS):
            if bufs[s].ready(args.batch_size):
                for j in range(args.num_updates):
                    o = updateStage(s)
                    if j % args.policy_frequency == 0:
                        updateActorStage(s, o)
                    with torch.no_grad():
                        torch._foreach_lerp_(list(targets[s].parameters()), list(critics[s].parameters()), args.tau)
                alpha[s] = log_alpha[s].exp().detach()

    sr = (sum(_rec) / len(_rec)) if _rec else 0.0
    hit = len(_rec) >= args.min_sr_episodes and sr >= args.success_rate_target
    if hit and (_ep - _last_hit_ep) >= args.min_sr_episodes:
        _sr_hits += 1; _last_hit_ep = _ep
        print(f"[JOINT] sr-hit {_sr_hits}/{args.min_sr_hits} (end2end sr={sr:.3f}, {_ep} eps) t={gstep}", flush=True)
    if hit and _sr_hits >= args.min_sr_hits:
        print(f"[DONE] joint end2end sr={sr:.3f} x{_sr_hits} at t={gstep}; saving stage actors.", flush=True)
        for s in range(NS):
            torch.save({"actor": actors[s].state_dict(), "critic": critics[s].state_dict(),
                        "log_alpha": log_alpha[s], "step": gstep, "stage": s},
                       os.path.join(args.save_dir, f"s{s}_final.pt"))
        break

    if nit % 20 == 0:
        c = [int((stage == tS).sum()) for tS in range(NS)]
        print(f"[JOINT] t={gstep} end2end_sr={sr:.3f} ({_ep} eps) envs@[reach{c[0]},align{c[1]},grasp{c[2]},lift{c[3]}] "
              f"alpha=[{float(alpha[0]):.2f},{float(alpha[1]):.2f},{float(alpha[2]):.2f},{float(alpha[3]):.2f}]", flush=True)
        try:
            dash.pushStats(step=gstep, episode=_ep, current_reward=0.0, episode_reward=0.0, cumulative_reward=0.0,
                            best_reward=0.0, success_count=int(sum(_rec)), success_rate=sr,
                            reward_terms={"end2end_sr": round(sr, 3), "reach": c[0], "align": c[1], "grasp": c[2], "lift": c[3]},
                            cumulative_terms={}, dopamine={})
        except Exception:
            pass
    if gstep % 204800 == 0:
        for s in range(NS):
            torch.save({"actor": actors[s].state_dict(), "step": gstep, "stage": s},
                       os.path.join(args.save_dir, f"s{s}_ckpt_{gstep}.pt"))

print("[JOINT] loop finished.", flush=True)
try:
    env.close(); simulation_app.close()
except Exception:
    pass
