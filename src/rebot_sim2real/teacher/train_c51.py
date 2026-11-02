"""Squint-matched distributional-C51 SAC (state-only) for the reBot LiftCube task on Isaac Lab.

Faithful port of _sources/squint/train_squint.py MINUS the RGB CNN (we train on the state vector):
  - Actor : state -> Projection(Linear256+LayerNorm+ReLU) -> 3x[Linear256+LN+ReLU] -> mean+logstd
            tanh-squashed Gaussian, logstd in [-5,2], orthogonal init. action_scale=1 (env clamps).
  - Critic: distributional C51 (num_atoms=101, support [-20,20]); shared state Projection then
            cat(action) -> 3x[Linear512+LN+ReLU] -> num_atoms logits; ENSEMBLE of 2 Q-nets.
            actor uses the MEAN of the 2 Q (No-CDQ), exactly like Squint.
  - autotune entropy (target=-dim(action)), gamma 0.9, tau 0.01, LR 3e-4 (policy/q/alpha)
  - num_updates (UTD) per env-step, actor delayed (policy_frequency 4), soft target every grad step
  - bootstrap_at_done='always' (every done is a time_out; bootstrap on next_obs, done stored 0)

Uses the SAME Squint-matched env as train_sac.py (configure_env): 5-term reward, delta gripper,
success=is_fold_success, 50-step episodes. Fleet success counting + dashboard are reused.
Run: PYTHONPATH=src DISPLAY=:1 .venv/bin/python -m rebot_sim2real.teacher.train_c51 --num-envs 1024 ...
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT", os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl")) + "/leisaac/assets")

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--total-timesteps", type=int, default=1_000_000_000)
parser.add_argument("--http-port", type=int, default=8890)
parser.add_argument("--save-dir", type=str, default="rl_output/rl_c51")
parser.add_argument("--target-cube", type=str, default="cube", choices=["cube", "cube_blue", "cube_purple"])
parser.add_argument("--headless", action="store_true")
parser.add_argument("--success-target", type=int, default=5000, help="stop at this many FLEET successes; 0 disables")
parser.add_argument("--success-rate-target", type=float, default=0.0,
                    help="stop when recent success_rate >= this (e.g. 0.95); 0 disables. Needs --min-sr-episodes samples.")
parser.add_argument("--min-sr-episodes", type=int, default=300,
                    help="min finished episodes in the recent window before the sr-target stop can fire")
parser.add_argument("--min-sr-hits", type=int, default=1,
                    help="stop only after the recent-window sr has reached the target on this many SEPARATE "
                         "windows (>= min-sr-episodes fresh episodes between counted hits) — robustness vs a "
                         "single lucky window. 1 = old behaviour.")
parser.add_argument("--resume", type=str, default="",
                    help="warm-start: load actor/critic/log_alpha from this checkpoint (e.g. c51_final.pt)")
parser.add_argument("--config", type=str, default="configs/reward_config.yaml")
# C51 / SAC hyperparameters (Squint train_squint.py defaults)
parser.add_argument("--buffer-size", type=int, default=1_000_000)
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--num-updates", type=int, default=32,
                    help="gradient updates per env-step (UTD). Squint uses 256 with compile+cudagraphs; "
                         "at 1024 envs that is ratio 128 and far too slow without compile. 32 -> ratio 16 "
                         "(still strong sample efficiency), ~8x faster wall-clock. Raise if GPU is idle.")
parser.add_argument("--learning-starts", type=int, default=5000)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--gamma", type=float, default=0.9)
parser.add_argument("--tau", type=float, default=0.01)
parser.add_argument("--policy-frequency", type=int, default=4)
parser.add_argument("--num-atoms", type=int, default=101)
parser.add_argument("--v-min", type=float, default=-20.0)
parser.add_argument("--v-max", type=float, default=20.0)
parser.add_argument("--num-q", type=int, default=2)
parser.add_argument("--cameras", action="store_true",
                    help="enable front/iso/wrist cameras + LIVE views on the dashboard (slower; use fewer envs)")
args, _ = parser.parse_known_args()

import yaml
CFG = {}
if os.path.exists(args.config):
    with open(args.config) as f:
        CFG = yaml.safe_load(f)
if args.cameras:
    CFG["enable_cameras"] = True

# ── Isaac Lab init ──
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": args.headless, "enable_cameras": CFG.get("enable_cameras", False)})
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import SceneEntityCfg, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
import leisaac.tasks.lift_cube  # noqa: F401  (registers the task)
from leisaac.tasks.lift_cube import mdp
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from rebot_sim2real.env.env_config import configure_env, apply_motor_limits
import rebot_sim2real.env.maniskill_rewards as MR
from rebot_sim2real.tools import rebot_dashboard as dash

DEVICE = "cuda:0"

# ── Env setup (Squint-matched env via configure_env) ──
env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=args.num_envs)
env_cfg = configure_env(env_cfg, CFG, mdp, RewTerm, DoneTerm, SceneEntityCfg, target_cube=args.target_cube)

# Contact sensors on the reBot fingers, pair-wise filtered to the TARGET cube (grasp detection).
_target_obj = getattr(env_cfg.scene, args.target_cube)
_cube_prim = _target_obj.prim_path
env_cfg.scene.gripper_contact = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/gripper_left", update_period=0.0, history_length=4,
    filter_prim_paths_expr=[_cube_prim])
env_cfg.scene.jaw_contact = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/gripper_right", update_period=0.0, history_length=4,
    filter_prim_paths_expr=[_cube_prim])
_target_obj.spawn.activate_contact_sensors = True
env_cfg.scene.robot.spawn.activate_contact_sensors = True

# cameras: state-based policy NEVER feeds on images (obs is state-only), but with --cameras we keep
# front/iso/wrist sensors purely for the LIVE dashboard monitor. Their world poses are set by a look-at
# after env creation (below) so they actually frame the workspace.
if args.cameras:
    from isaaclab.sensors import TiledCameraCfg as _TCam
    import isaaclab.sim as _simcam
    _cspawn = _simcam.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.01, 50.0), lock_camera=True)
    env_cfg.scene.front = _TCam(prim_path="{ENV_REGEX_NS}/Robot/base_link/front_camera",
        offset=_TCam.OffsetCfg(pos=(0.0, 0.0, 0.5), convention="ros"),
        data_types=["rgb"], spawn=_cspawn, width=256, height=192, update_period=0)
    env_cfg.scene.iso = _TCam(prim_path="{ENV_REGEX_NS}/Robot/base_link/iso_camera",
        offset=_TCam.OffsetCfg(pos=(0.0, 0.0, 0.5), convention="ros"),
        data_types=["rgb"], spawn=_cspawn, width=256, height=192, update_period=0)
    # (env_config already put 'wrist' on gripper_link/fp_camera -- keep it)
else:
    for cam in ["front", "side", "wrist"]:
        if hasattr(env_cfg.scene, cam):
            setattr(env_cfg.scene, cam, None)
# policy obs is state-only either way; drop any stray image obs terms.
for attr in list(vars(env_cfg.observations.policy)):
    if attr in ('front', 'wrist', 'side') or 'image' in attr.lower():
        setattr(env_cfg.observations.policy, attr, None)

# Frame the GUI on env_0
try:
    env_cfg.viewer.origin_type = "env"; env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (1.2, 1.0, 0.8); env_cfg.viewer.lookat = (0.3, -0.3, 0.15)
except Exception:
    pass

env = ManagerBasedRLEnv(cfg=env_cfg)
apply_motor_limits(env, CFG)
sb3_env = Sb3VecEnvWrapper(env)
N_ENVS = args.num_envs
obs_dim = int(np.prod(sb3_env.observation_space.shape))
act_dim = int(np.prod(sb3_env.action_space.shape))
print(f"[C51] obs_dim={obs_dim} act_dim={act_dim} num_envs={N_ENVS}", flush=True)

dash.start(args.http_port, target_cube=args.target_cube)

# Live monitor cameras (front/iso/wrist) for the dashboard. front/iso are re-pointed at the workspace
# via look-at each push (robust to the base_link +90deg frame and to resets). wrist is the gripper POV.
_cam_map = {}
_cam_eyes = None
_cam_tgt = None
if args.cameras:
    import torch as _tcam
    _eo = env.scene.env_origins                                   # (N,3) per-env world origin
    _cam_tgt = _eo + _tcam.tensor([0.30, -0.35, 0.08], device=_eo.device)   # workspace (cube/gripper)
    _cam_eyes = {"front": _eo + _tcam.tensor([0.30, 0.40, 0.45], device=_eo.device),  # in front (+y), up
                 "iso":   _eo + _tcam.tensor([0.95, 0.20, 0.55], device=_eo.device)}  # iso from +x side
    for _nm in ["front", "iso", "wrist"]:
        try:
            _cam_map[_nm] = env.scene[_nm]
        except Exception:
            pass
    print(f"[C51] live dashboard cameras: {list(_cam_map.keys())}", flush=True)
os.makedirs(args.save_dir, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────────
#  Networks
# ────────────────────────────────────────────────────────────────────────────────
def orthogonal_init(m):
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
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, n_act)
        self.logstd = nn.Linear(hidden, n_act)
        self.apply(orthogonal_init)

    def _mean_logstd(self, s):
        x = self.fc(self.proj(s))
        mean = self.mean(x)
        logstd = torch.tanh(self.logstd(x))
        logstd = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (logstd + 1)
        return mean, logstd

    def get_action(self, s):
        mean, logstd = self._mean_logstd(s)
        std = logstd.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        y = torch.tanh(x)
        logp = normal.log_prob(x) - torch.log(1 - y.pow(2) + 1e-6)
        logp = logp.sum(1, keepdim=True)
        return y, logp                      # action in [-1,1] (env action terms clamp to [-1,1])

    @torch.no_grad()
    def act_deterministic(self, s):
        mean, _ = self._mean_logstd(s)
        return torch.tanh(mean)


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
                nn.Linear(hidden, num_atoms),
            )
        self.qs = nn.ModuleList([qnet() for _ in range(num_q)])
        self.apply(orthogonal_init)

    def logits(self, s, a):                  # -> (num_q, N, atoms)
        x = torch.cat([self.proj(s), a], dim=-1)
        return torch.stack([q(x) for q in self.qs], dim=0)

    def q_values(self, s, a):                # expected Q -> (num_q, N)
        probs = F.softmax(self.logits(s, a), dim=-1)
        return (probs * self.support).sum(-1)


@torch.no_grad()
def project_target(target_critic, next_s, next_a, rewards, bootstrap, gamma):
    """C51 categorical projection (Squint envs categorical). Returns target dists (num_q, N, atoms)."""
    logits = target_critic.logits(next_s, next_a)            # (Q, N, A)
    probs = F.softmax(logits, dim=-1)
    support = target_critic.support
    v_min, v_max, A = target_critic.v_min, target_critic.v_max, target_critic.num_atoms
    Q, Nn, _ = probs.shape
    dz = (v_max - v_min) / (A - 1)
    tz = rewards.view(1, Nn, 1) + bootstrap.view(1, Nn, 1) * gamma * support.view(1, 1, A)
    tz = tz.clamp(v_min, v_max).expand(Q, Nn, A)
    b = (tz - v_min) / dz
    l = b.floor().long()
    u = b.ceil().long()
    is_int = (u == l)
    l = torch.where((l > 0) & is_int, l - 1, l)
    u = torch.where((l == 0) & is_int, u + 1, u)
    l = l.clamp(0, A - 1)
    u = u.clamp(0, A - 1)
    m = torch.zeros_like(probs)
    m.scatter_add_(2, l, probs * (u.float() - b))
    m.scatter_add_(2, u, probs * (b - l.float()))
    return m


# ────────────────────────────────────────────────────────────────────────────────
#  Replay buffer (GPU, circular)
# ────────────────────────────────────────────────────────────────────────────────
class Buffer:
    def __init__(self, cap, obs_dim, act_dim, device):
        self.cap, self.device, self.ptr, self.full = cap, device, 0, False
        self.obs = torch.zeros(cap, obs_dim, device=device)
        self.act = torch.zeros(cap, act_dim, device=device)
        self.rew = torch.zeros(cap, device=device)
        self.nxt = torch.zeros(cap, obs_dim, device=device)
        self.done = torch.zeros(cap, device=device)

    def add(self, o, a, r, n, d):
        B = o.shape[0]
        idx = (torch.arange(B, device=self.device) + self.ptr) % self.cap
        self.obs[idx] = o; self.act[idx] = a; self.rew[idx] = r; self.nxt[idx] = n; self.done[idx] = d
        newptr = self.ptr + B
        if newptr >= self.cap:
            self.full = True
        self.ptr = newptr % self.cap

    def sample(self, bs):
        hi = self.cap if self.full else self.ptr
        idx = torch.randint(0, hi, (bs,), device=self.device)
        return self.obs[idx], self.act[idx], self.rew[idx], self.nxt[idx], self.done[idx]


# ────────────────────────────────────────────────────────────────────────────────
#  Instantiate
# ────────────────────────────────────────────────────────────────────────────────
actor = Actor(obs_dim, act_dim).to(DEVICE)
critic = CriticC51(obs_dim, act_dim, args.num_atoms, args.v_min, args.v_max, args.num_q).to(DEVICE)
critic_target = CriticC51(obs_dim, act_dim, args.num_atoms, args.v_min, args.v_max, args.num_q).to(DEVICE)
critic_target.load_state_dict(critic.state_dict())
for p in critic_target.parameters():
    p.requires_grad_(False)

actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
critic_opt = torch.optim.Adam(critic.parameters(), lr=args.lr)

target_entropy = -float(act_dim)
log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
alpha_opt = torch.optim.Adam([log_alpha], lr=args.lr)
alpha = log_alpha.exp().detach()

# warm-start (continue training a previous teacher toward a higher success rate)
if args.resume:
    _ck = torch.load(args.resume, map_location=DEVICE)
    actor.load_state_dict(_ck["actor"])
    if "critic" in _ck:
        critic.load_state_dict(_ck["critic"]); critic_target.load_state_dict(_ck["critic"])
    if "log_alpha" in _ck:
        _la = _ck["log_alpha"]
        log_alpha.data.copy_((_la.detach() if torch.is_tensor(_la) else torch.tensor([float(_la)])).to(DEVICE).reshape(1))
        alpha = log_alpha.exp().detach()
    print(f"[C51] RESUMED from {args.resume} (step {_ck.get('step','?')})", flush=True)

buf = Buffer(args.buffer_size, obs_dim, act_dim, DEVICE)
crit_params = list(critic.parameters())
targ_params = list(critic_target.parameters())

print(f"[C51] networks ready. UTD={args.num_updates} batch={args.batch_size} gamma={args.gamma} "
      f"atoms={args.num_atoms} v=[{args.v_min},{args.v_max}] tau={args.tau} lr={args.lr}", flush=True)


def update_critic(o, a, r, n, d):
    with torch.no_grad():
        na, nlogp = actor.get_action(n)
        bootstrap = (1.0 - d)                                   # bootstrap_at_done='always' -> d is 0
        # entropy-augmented reward (soft) folded into the distributional target support shift
        r_ent = r - bootstrap * args.gamma * (alpha.squeeze() * nlogp.squeeze(-1))
        target_dist = project_target(critic_target, n, na, r_ent, bootstrap, args.gamma)  # (Q,N,A)
    logits = critic.logits(o, a)                                # (Q,N,A)
    logp = F.log_softmax(logits, dim=-1)
    q_losses = -(target_dist * logp).sum(-1).mean(-1)           # (Q,)
    critic_loss = q_losses.sum()
    critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_opt.step()
    return critic_loss.detach()


def update_actor(o):
    a_pi, logp = actor.get_action(o)
    qv = critic.q_values(o, a_pi).mean(0)                       # MEAN over ensemble (No-CDQ) -> (N,)
    actor_loss = (alpha.squeeze() * logp.squeeze(-1) - qv).mean()
    actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_opt.step()
    # alpha (autotune)
    with torch.no_grad():
        _, logp2 = actor.get_action(o)
    alpha_loss = -(log_alpha.exp() * (logp2 + target_entropy)).mean()
    alpha_opt.zero_grad(set_to_none=True)
    alpha_loss.backward()
    alpha_opt.step()
    return actor_loss.detach()


def soft_update():
    with torch.no_grad():
        torch._foreach_lerp_(targ_params, crit_params, args.tau)


# ────────────────────────────────────────────────────────────────────────────────
#  Training loop
# ────────────────────────────────────────────────────────────────────────────────
def to_t(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=DEVICE)


print(f"[C51] Starting training: target={args.total_timesteps} steps, success_target={args.success_target}", flush=True)
obs = sb3_env.reset()
obs = np.nan_to_num(np.asarray(obs, dtype=np.float32), nan=0.0, posinf=100.0, neginf=-100.0)
obs_t = to_t(obs)

global_step = 0
n_iter = 0
_sr_hits = 0           # how many SEPARATE recent-windows have hit the sr target (for --min-sr-hits)
_last_hit_ep = -10**9  # episode count at the last counted hit (gate hits by >= min_sr_episodes fresh eps)
ep_rew = np.zeros(N_ENVS, dtype=np.float64)
ep_hist = []
best_reward = float("-inf")
cum_reward = 0.0
cum_terms = {}

while global_step < args.total_timesteps:
    # ── collect 1 env-step (all envs) ──
    if global_step < args.learning_starts:
        action = (torch.rand(N_ENVS, act_dim, device=DEVICE) * 2.0 - 1.0)
    else:
        with torch.no_grad():
            action, _ = actor.get_action(obs_t)
    a_np = action.detach().cpu().numpy().astype(np.float32)
    next_obs, rew, done, infos = sb3_env.step(a_np)
    next_obs = np.nan_to_num(np.asarray(next_obs, dtype=np.float32), nan=0.0, posinf=100.0, neginf=-100.0)
    rew = np.nan_to_num(np.asarray(rew, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    done = np.asarray(done)

    # bootstrap_at_done='always': every done here is a time_out (success_latch never terminates), so we
    # bootstrap on next_obs and store done=0. (Sb3VecEnvWrapper's terminal_observation is the POST-reset
    # obs, so there is no genuine pre-reset terminal obs to substitute -- the next-obs approximation is
    # fine for this fixed-horizon, no-terminal-state task; only ~1/50 transitions are boundaries.)
    buf.add(obs_t, action.detach(), to_t(rew), to_t(next_obs), torch.zeros(N_ENVS, device=DEVICE))

    # ── episode bookkeeping (all envs) ──
    ep_rew += rew
    cum_reward += float(rew.mean())
    for i in np.nonzero(done)[0]:
        ep_hist.append(float(ep_rew[i]))
        if ep_rew[i] > best_reward:
            best_reward = float(ep_rew[i])
        ep_rew[i] = 0.0
    ep_hist = ep_hist[-200:]

    obs = next_obs
    obs_t = to_t(obs)
    global_step += N_ENVS
    n_iter += 1

    # ── gradient updates (UTD) ──
    if global_step >= args.learning_starts:
        for j in range(args.num_updates):
            o, a, r, n, d = buf.sample(args.batch_size)
            update_critic(o, a, r, n, d)
            if j % args.policy_frequency == 0:
                update_actor(o)
            soft_update()
        alpha = log_alpha.exp().detach()

    # ── early stop: count target OR success-rate target ──
    succ = MR._succ_total[0]
    _recent = MR._succ_recent
    _sr = (sum(_recent) / len(_recent)) if _recent else 0.0
    _hit_count = bool(args.success_target) and succ >= args.success_target
    _hit_sr = bool(args.success_rate_target) and len(_recent) >= args.min_sr_episodes and _sr >= args.success_rate_target
    # count a "hit" at most once per FRESH window (>= min_sr_episodes new episodes since the last hit),
    # so --min-sr-hits N requires the sr target to hold on N separate windows (robust vs a lucky window).
    if _hit_sr and (MR._ep_total[0] - _last_hit_ep) >= args.min_sr_episodes:
        _sr_hits += 1
        _last_hit_ep = MR._ep_total[0]
        print(f"[C51] sr-hit {_sr_hits}/{args.min_sr_hits} (sr={_sr:.3f}, {MR._ep_total[0]} eps) at t={global_step}", flush=True)
    _stop_sr = _hit_sr and _sr_hits >= args.min_sr_hits
    if _hit_count or _stop_sr:
        why = (f"sr={_sr:.3f}>={args.success_rate_target} x{_sr_hits} hits ({len(_recent)} eps)"
               if _stop_sr else f"{succ} successes")
        print(f"[DONE] stop ({why}) at t={global_step}; saving c51_final.pt.", flush=True)
        try:
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                        "log_alpha": log_alpha, "step": global_step},
                       os.path.join(args.save_dir, "c51_final.pt"))
        except Exception as e:
            print(f"[DONE] save error: {e}", flush=True)
        break

    # ── dashboard ──
    if n_iter % 5 == 0:
        rt = {}
        try:
            rm = env.reward_manager
            for name in rm.active_terms:
                cfg = rm.get_term_cfg(name)
                try:
                    raw = cfg.func(env, **cfg.params)
                    val = float(raw.float().mean().item()) * float(cfg.weight) * float(env.step_dt)
                except Exception:
                    val = 0.0
                rt[name] = round(val, 4)
                cum_terms[name] = cum_terms.get(name, 0.0) + val
        except Exception:
            pass
        recent = MR._succ_recent
        sr = (sum(recent) / len(recent)) if recent else 0.0
        _best = best_reward if best_reward != float("-inf") else 0.0
        _mean_ep = float(np.mean(ep_hist[-20:])) if ep_hist else 0.0   # completed-episode mean
        dash.push_stats(
            step=global_step, episode=MR._ep_total[0],
            current_reward=float(rew.mean()), episode_reward=_mean_ep,
            cumulative_reward=cum_reward, best_reward=_best,
            success_count=succ, success_rate=sr,
            reward_terms=rt, cumulative_terms=dict(cum_terms),
            dopamine={},
        )
        if _cam_map:
            try:
                if _cam_eyes is not None:
                    for _nm, _eye in _cam_eyes.items():
                        _cam_map[_nm].set_world_poses_from_view(_eye, _cam_tgt)
                dash.update_cameras(_cam_map)
            except Exception as _e:
                pass

    if n_iter % 50 == 0:
        mean_r = float(np.mean(ep_hist[-20:])) if ep_hist else 0.0
        recent = MR._succ_recent
        sr = (sum(recent) / len(recent)) if recent else 0.0
        print(f"[C51] t={global_step} eps={MR._ep_total[0]} mean_r={mean_r:.2f} best={best_reward:.2f} "
              f"success={succ} sr={sr:.3f} alpha={float(alpha):.3f}", flush=True)

    if n_iter % 200 == 0:
        try:
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                        "log_alpha": log_alpha, "step": global_step},
                       os.path.join(args.save_dir, f"c51_ckpt_{global_step}.pt"))
        except Exception:
            pass

print("[C51] training loop finished.", flush=True)
try:
    simulation_app.close()
except Exception:
    pass
