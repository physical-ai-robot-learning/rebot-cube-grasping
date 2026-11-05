"""3-STAGE hierarchical C51 teacher (state-only) — train one actor-critic per stage, SEQUENTIALLY:
  stage 0 APPROACH : bring the OPEN gripper aligned over the cube. done = align-hold latch.
  stage 1 GRASP    : from aligned, close + grasp.                  done = _is_grasped.
  stage 2 PLACE    : from grasped, lift + return arm to REST.      done = is_fold_success.

Sequential curriculum: train --stage 0 first; then --stage 1 with --prior rl_output/<s0>/c51_final.pt
(the FROZEN s0 rolls each env to approach-done, then the trainee learns the grasp segment); then
--stage 2 with --prior <s0> <s1>. At deploy the 3 actors run in sequence (s0 until approach-done -> s1
until grasp-done -> s2). Reuses train_c51's C51 (Actor/CriticC51/project_target/updates).

  PYTHONPATH=src DISPLAY=:1 python -m rebot_teacher_student.teacher.train_c51_staged --stage 0 --headless --num-envs 1024
  ... --stage 1 --prior rl_output/rl_c51_s0/c51_final.pt
  ... --stage 2 --prior rl_output/rl_c51_s0/c51_final.pt rl_output/rl_c51_s1/c51_final.pt
"""
import argparse, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT", os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl")) + "/leisaac/assets")

ap = argparse.ArgumentParser()
ap.add_argument("--stage", type=int, required=True, choices=[0, 1, 2])
ap.add_argument("--prior", type=str, nargs="*", default=[], help="frozen prior-stage actor checkpoints (s0[,s1])")
ap.add_argument("--num-envs", type=int, default=1024)
ap.add_argument("--total-timesteps", type=int, default=1_000_000_000)
ap.add_argument("--http-port", type=int, default=8890)
ap.add_argument("--save-dir", type=str, required=True)
ap.add_argument("--target-cube", type=str, default="cube")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--cameras", action="store_true", help="add front+wrist cameras for a LIVE dashboard view")
ap.add_argument("--success-rate-target", type=float, default=0.95)
ap.add_argument("--min-sr-episodes", type=int, default=300)
ap.add_argument("--min-sr-hits", type=int, default=3)
ap.add_argument("--bonus", type=float, default=5.0, help="reward bonus when the stage's done is first reached")
ap.add_argument("--config", type=str, default="configs/reward_config.yaml")
ap.add_argument("--buffer-size", type=int, default=1_000_000)
ap.add_argument("--batch-size", type=int, default=512)
ap.add_argument("--num-updates", type=int, default=32)
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
simulation_app = AppLauncher({"headless": args.headless, "enable_cameras": args.cameras}).app

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

DEVICE = "cuda:0"; TARGET = args.target_cube; N_ENVS = args.num_envs
TRAIN_STAGE = args.stage
STAGE_NAME = {0: "approach", 1: "grasp", 2: "place"}[TRAIN_STAGE]
# reward terms ACTIVE per stage (align_hold ALWAYS kept — it maintains the align-hold latch used by the
# approach/grasp done-conditions). others zeroed so each actor optimises only its own segment.
STAGE_KEEP = {
    0: {"reaching", "align_hold", "open_approach", "table_penalty"},
    1: {"reaching", "align_hold", "close_align", "grasped", "table_penalty"},
    2: {"reaching", "align_hold", "place", "not_lifted", "grasped", "table_penalty"},
}[TRAIN_STAGE]

env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=N_ENVS)
env_cfg = configureEnv(env_cfg, CFG, mdp, RewTerm, DoneTerm, SceneEntityCfg, target_cube=TARGET)
_obj = getattr(env_cfg.scene, TARGET); _cp = _obj.prim_path
env_cfg.scene.gripper_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_left",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
env_cfg.scene.jaw_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_right",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
_obj.spawn.activate_contact_sensors = True
env_cfg.scene.robot.spawn.activate_contact_sensors = True
if args.cameras:
    # add a head-on FRONT camera (env-level, baked look-at) + keep env_config's wrist POV, for a LIVE
    # dashboard view of the staged training. (state-only obs is unaffected — these feed only the monitor.)
    import isaaclab.sim as _sc
    from isaaclab.sensors import TiledCameraCfg as _TC
    from rebot_teacher_student.env.visual_obs import _lookatQuatOpengl as _laq, FRONT_EYE, FRONT_TARGET
    env_cfg.scene.front = _TC(prim_path="{ENV_REGEX_NS}/front_camera",
        offset=_TC.OffsetCfg(pos=FRONT_EYE, rot=_laq(FRONT_EYE, FRONT_TARGET), convention="opengl"),
        data_types=["rgb"], spawn=_sc.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.01, 50.0), lock_camera=True),
        width=240, height=240, update_period=0)
    if hasattr(env_cfg.scene, "side"):
        env_cfg.scene.side = None
else:
    for cam in ("front", "side", "wrist"):
        if hasattr(env_cfg.scene, cam):
            setattr(env_cfg.scene, cam, None)
for attr in list(vars(env_cfg.observations.policy)):
    if attr in ("front", "wrist", "side") or "image" in attr.lower():
        setattr(env_cfg.observations.policy, attr, None)
# stage reward subset: zero the weight of every reward term NOT in this stage's keep-set
for _rn in list(vars(env_cfg.rewards)):
    if _rn.startswith("_"):
        continue
    if _rn not in STAGE_KEEP:
        _rt = getattr(env_cfg.rewards, _rn, None)
        if _rt is not None and hasattr(_rt, "weight"):
            _rt.weight = 0.0

# frame the GUI viewer on env_0 (live-watch the staged training when run WITHOUT --headless)
try:
    env_cfg.viewer.origin_type = "env"; env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (1.2, 1.0, 0.8); env_cfg.viewer.lookat = (0.3, -0.3, 0.15)
except Exception:
    pass

env = ManagerBasedRLEnv(cfg=env_cfg)
applyMotorLimits(env, CFG)
sb3_env = Sb3VecEnvWrapper(env)
obs_dim = int(np.prod(sb3_env.observation_space.shape)); act_dim = int(np.prod(sb3_env.action_space.shape))
print(f"[STAGED s{TRAIN_STAGE}={STAGE_NAME}] obs={obs_dim} act={act_dim} envs={N_ENVS} keep={sorted(STAGE_KEEP)}", flush=True)
dash.start(args.http_port, targetCube=TARGET)
os.makedirs(args.save_dir, exist_ok=True)
_cam_map = {}
if args.cameras:
    for _nm in ("front", "wrist"):
        try:
            _cam_map[_nm] = env.scene[_nm]
        except Exception:
            pass
    print(f"[STAGED] live dashboard cameras: {list(_cam_map.keys())}", flush=True)

# ── nets (inline copy of train_c51's; train_c51 is a script so it can't be imported) ──
def orthogonalInit(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Actor(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, n_state, n_act, hidden=256):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(n_state, 256), nn.LayerNorm(256), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(256, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, n_act); self.logstd = nn.Linear(hidden, n_act)
        self.apply(orthogonalInit)

    def _meanLogstd(self, s):
        tX = self.fc(self.proj(s)); tMean = self.mean(tX)
        tLogstd = torch.tanh(self.logstd(tX))
        tLogstd = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (tLogstd + 1)
        return tMean, tLogstd

    def getAction(self, s):
        tMean, tLogstd = self._meanLogstd(s); tStd = tLogstd.exp()
        tNormal = torch.distributions.Normal(tMean, tStd); tX = tNormal.rsample(); tY = torch.tanh(tX)
        tLogp = (tNormal.log_prob(tX) - torch.log(1 - tY.pow(2) + 1e-6)).sum(1, keepdim=True)
        return tY, tLogp

    @torch.no_grad()
    def actDeterministic(self, s):
        tMean, _ = self._meanLogstd(s); return torch.tanh(tMean)


class CriticC51(nn.Module):
    def __init__(self, n_state, n_act, num_atoms, v_min, v_max, num_q=2, hidden=512):
        super().__init__()
        self.num_atoms, self.num_q, self.v_min, self.v_max = num_atoms, num_q, v_min, v_max
        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))
        self.proj = nn.Sequential(nn.Linear(n_state, 256), nn.LayerNorm(256), nn.ReLU())

        def qnet():
            return nn.Sequential(
                nn.Linear(256 + n_act, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                nn.Linear(hidden, num_atoms))
        self.qs = nn.ModuleList([qnet() for _ in range(num_q)])
        self.apply(orthogonalInit)

    def logits(self, s, a):
        tX = torch.cat([self.proj(s), a], dim=-1)
        return torch.stack([tQ2(tX) for tQ2 in self.qs], dim=0)

    def qValues(self, s, a):
        return (F.softmax(self.logits(s, a), dim=-1) * self.support).sum(-1)


@torch.no_grad()
def projectTarget(tc, ns, na, rewards, bootstrap, gamma):
    tProbs = F.softmax(tc.logits(ns, na), dim=-1); tSupport = tc.support
    tVMin, tVMax, tA = tc.v_min, tc.v_max, tc.num_atoms; tQ, tNn, _ = tProbs.shape
    tDz = (tVMax - tVMin) / (tA - 1)
    tTz = (rewards.view(1, tNn, 1) + bootstrap.view(1, tNn, 1) * gamma * tSupport.view(1, 1, tA)).clamp(tVMin, tVMax).expand(tQ, tNn, tA)
    tB = (tTz - tVMin) / tDz; tL = tB.floor().long(); tU = tB.ceil().long(); tIsInt = (tU == tL)
    tL = torch.where((tL > 0) & tIsInt, tL - 1, tL); tU = torch.where((tL == 0) & tIsInt, tU + 1, tU)
    tL = tL.clamp(0, tA - 1); tU = tU.clamp(0, tA - 1); tM = torch.zeros_like(tProbs)
    tM.scatter_add_(2, tL, tProbs * (tU.float() - tB)); tM.scatter_add_(2, tU, tProbs * (tB - tL.float()))
    return tM


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
        if tNp >= self.cap:
            self.full = True
        self.ptr = tNp % self.cap

    def sample(self, bs):
        tHi = self.cap if self.full else self.ptr
        tIdx = torch.randint(0, tHi, (bs,), device=self.device)
        return self.obs[tIdx], self.act[tIdx], self.rew[tIdx], self.nxt[tIdx], self.done[tIdx]


actor = Actor(obs_dim, act_dim).to(DEVICE)
critic = CriticC51(obs_dim, act_dim, args.num_atoms, args.v_min, args.v_max, args.num_q).to(DEVICE)
critic_target = CriticC51(obs_dim, act_dim, args.num_atoms, args.v_min, args.v_max, args.num_q).to(DEVICE)
critic_target.load_state_dict(critic.state_dict())
for p in critic_target.parameters():
    p.requires_grad_(False)
actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
critic_opt = torch.optim.Adam(critic.parameters(), lr=args.lr)
target_entropy = -float(act_dim)
log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE); alpha_opt = torch.optim.Adam([log_alpha], lr=args.lr)
alpha = log_alpha.exp().detach()

# frozen prior-stage actors (deterministic rollout up to this stage)
assert len(args.prior) >= TRAIN_STAGE, f"stage {TRAIN_STAGE} needs {TRAIN_STAGE} prior checkpoint(s) via --prior"
priors = []
for i in range(TRAIN_STAGE):
    pa = Actor(obs_dim, act_dim).to(DEVICE)
    pc = torch.load(args.prior[i], map_location=DEVICE)
    pa.load_state_dict(pc["actor"] if "actor" in pc else pc); pa.eval()
    for p in pa.parameters():
        p.requires_grad_(False)
    priors.append(pa)
    print(f"[STAGED] prior s{i} <- {args.prior[i]}", flush=True)

buf = Buffer(args.buffer_size, obs_dim, act_dim, DEVICE)
crit_params = list(critic.parameters()); targ_params = list(critic_target.parameters())


def updateCritic(o, a, r, n, d):
    with torch.no_grad():
        tNa, tNlogp = actor.getAction(n); tBootstrap = (1.0 - d)
        tREnt = r - tBootstrap * args.gamma * (alpha.squeeze() * tNlogp.squeeze(-1))
        tTarget = projectTarget(critic_target, n, tNa, tREnt, tBootstrap, args.gamma)
    tLogp = F.log_softmax(critic.logits(o, a), dim=-1)
    tLoss = (-(tTarget * tLogp).sum(-1).mean(-1)).sum()
    critic_opt.zero_grad(set_to_none=True); tLoss.backward(); critic_opt.step()


def updateActor(o):
    tAPi, tLogp = actor.getAction(o); tQv = critic.qValues(o, tAPi).mean(0)
    (alpha.squeeze() * tLogp.squeeze(-1) - tQv).mean().backward()
    actor_opt.step(); actor_opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        _, tLogp2 = actor.getAction(o)
    tAloss = -(log_alpha.exp() * (tLogp2 + target_entropy)).mean()
    alpha_opt.zero_grad(set_to_none=True); tAloss.backward(); alpha_opt.step()


def softUpdate():
    with torch.no_grad():
        torch._foreach_lerp_(targ_params, crit_params, args.tau)


# ── stage done-conditions (from the raw env) ──
_ee = SceneEntityCfg("ee_frame"); _oc = SceneEntityCfg(TARGET)
def _dist():
    return torch.norm(env.scene[TARGET].data.root_pos_w - MR._tcp(env, _ee), dim=1)
def stageDone(s):
    if s == 0:
        return MR._alignHeld(env, _dist())
    if s == 1:
        return MR._isGrasped(env, _oc)
    return MR.isFoldSuccess(env, _oc)


def toT(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=DEVICE)


print(f"[STAGED s{TRAIN_STAGE}] start. sr-target={args.success_rate_target} hits={args.min_sr_hits} bonus={args.bonus}", flush=True)
obs = np.nan_to_num(np.asarray(sb3_env.reset(), dtype=np.float32), nan=0.0, posinf=100.0, neginf=-100.0)
obs_t = toT(obs)
stage_ptr = torch.zeros(N_ENVS, dtype=torch.long, device=DEVICE)
done_latch = torch.zeros(N_ENVS, dtype=torch.bool, device=DEVICE)   # train-stage done reached this episode
reached = torch.zeros(N_ENVS, dtype=torch.bool, device=DEVICE)      # episode reached the train stage
global_step = 0; n_iter = 0; _sr_hits = 0; _last_hit_ep = -10**9
_succ_recent = []; _ep_total = 0

while global_step < args.total_timesteps:
    m_tr = (stage_ptr == TRAIN_STAGE)
    reached |= m_tr
    action = torch.zeros(N_ENVS, act_dim, device=DEVICE)
    for s in range(TRAIN_STAGE):
        m = (stage_ptr == s)
        if m.any():
            action[m] = priors[s].actDeterministic(obs_t[m])
    if m_tr.any():
        if global_step < args.learning_starts:
            action[m_tr] = torch.rand(int(m_tr.sum()), act_dim, device=DEVICE) * 2 - 1
        else:
            with torch.no_grad():
                a_tr, _ = actor.getAction(obs_t[m_tr])
            action[m_tr] = a_tr

    next_obs, rew, done, infos = sb3_env.step(action.detach().cpu().numpy().astype(np.float32))
    next_obs = np.nan_to_num(np.asarray(next_obs, dtype=np.float32), nan=0.0, posinf=100.0, neginf=-100.0)
    rew = np.nan_to_num(np.asarray(rew, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    done = np.asarray(done); next_t = toT(next_obs)

    # stage conditions AFTER the step
    appr = stageDone(0); grsp = stageDone(1); trd = stageDone(TRAIN_STAGE)
    # collect training-stage transitions (only envs that ACTED with the trainee), with a done-bonus
    if m_tr.any():
        r_tr = toT(rew)[m_tr] + args.bonus * (trd[m_tr] & ~done_latch[m_tr]).float()
        buf.add(obs_t[m_tr], action[m_tr].detach(), r_tr, next_t[m_tr], torch.zeros(int(m_tr.sum()), device=DEVICE))
    done_latch |= (trd & m_tr)
    # advance prior stages on their done-conditions (never past the train stage)
    if TRAIN_STAGE >= 1:
        stage_ptr = torch.where((stage_ptr == 0) & appr, torch.ones_like(stage_ptr), stage_ptr)
    if TRAIN_STAGE >= 2:
        stage_ptr = torch.where((stage_ptr == 1) & grsp, torch.full_like(stage_ptr, 2), stage_ptr)

    # env reset (time_out): tally stage success among episodes that reached the train stage, then clear
    dmask = torch.as_tensor(done, device=DEVICE).bool()
    if dmask.any():
        for i in torch.nonzero(dmask & reached).flatten().tolist():
            _succ_recent.append(1.0 if bool(done_latch[i]) else 0.0); _ep_total += 1
        _succ_recent = _succ_recent[-500:]
        stage_ptr[dmask] = 0; done_latch[dmask] = False; reached[dmask] = False

    obs = next_obs; obs_t = next_t; global_step += N_ENVS; n_iter += 1

    if global_step >= args.learning_starts and (buf.full or buf.ptr >= args.batch_size):
        for j in range(args.num_updates):
            o, a, r, n, d = buf.sample(args.batch_size)
            updateCritic(o, a, r, n, d)
            if j % args.policy_frequency == 0:
                updateActor(o)
            softUpdate()
        alpha = log_alpha.exp().detach()

    sr = (sum(_succ_recent) / len(_succ_recent)) if _succ_recent else 0.0
    hit = len(_succ_recent) >= args.min_sr_episodes and sr >= args.success_rate_target
    if hit and (_ep_total - _last_hit_ep) >= args.min_sr_episodes:
        _sr_hits += 1; _last_hit_ep = _ep_total
        print(f"[STAGED s{TRAIN_STAGE}] sr-hit {_sr_hits}/{args.min_sr_hits} (sr={sr:.3f}, {_ep_total} eps) t={global_step}", flush=True)
    if hit and _sr_hits >= args.min_sr_hits:
        print(f"[DONE] stage {TRAIN_STAGE} ({STAGE_NAME}) sr={sr:.3f} x{_sr_hits} at t={global_step}; saving c51_final.pt", flush=True)
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "log_alpha": log_alpha,
                    "step": global_step, "stage": TRAIN_STAGE}, os.path.join(args.save_dir, "c51_final.pt"))
        break

    if n_iter % 20 == 0:
        _c0 = int((stage_ptr == 0).sum()); _c1 = int((stage_ptr == 1).sum()); _c2 = int((stage_ptr == 2).sum())
        print(f"[STAGED s{TRAIN_STAGE}={STAGE_NAME}] t={global_step} stage_sr={sr:.3f} ({_ep_total} eps, "
              f"reached={float(reached.float().mean()):.2f}, envs@[{_c0},{_c1},{_c2}]) alpha={float(alpha):.3f}", flush=True)
        try:
            dash.pushStats(step=global_step, episode=_ep_total, current_reward=0.0, episode_reward=0.0,
                            cumulative_reward=0.0, best_reward=0.0, success_count=int(sum(_succ_recent)),
                            success_rate=sr,
                            reward_terms={f"s{TRAIN_STAGE}_{STAGE_NAME}_sr": round(sr, 3), "train_stage": TRAIN_STAGE,
                                          "envs@s0": _c0, "envs@s1": _c1, "envs@s2": _c2},
                            cumulative_terms={}, dopamine={})
            if _cam_map:
                dash.updateCameras(_cam_map)
        except Exception:
            pass

    if global_step % 204800 == 0:
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "log_alpha": log_alpha,
                    "step": global_step, "stage": TRAIN_STAGE}, os.path.join(args.save_dir, f"c51_ckpt_{global_step}.pt"))

print(f"[STAGED s{TRAIN_STAGE}] loop finished.", flush=True)
try:
    env.close(); simulation_app.close()
except Exception:
    pass

