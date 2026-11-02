"""SQUINT-style VISUAL distributional-C51 SAC for reBot LiftCube (wrist eye-in-hand + fixed front).

Faithful visual port of _sources/squint/train_squint.py:
  - obs   : rgb_wrist(16x16) + rgb_front(64x64) + proprio state(21) -- NO privileged obj/grasp.
  - encode: two CNN encoders (SQUINT conv; 16 and 64 variants, each -> 1024 features).
  - proj  : Projection(wrist 1024->50 +LN+Tanh, front 1024->50 +LN+Tanh, state 21->256 +LN+ReLU) -> 356.
  - actor : proj -> 3x[256+LN+ReLU] -> mean/logstd (tanh Gaussian, logstd [-5,2]).
  - critic: distributional C51 (101 atoms [-20,20]); proj+action -> 3x[512+LN+ReLU] -> atoms; 2-Q.
  - encoders trained by the CRITIC optimizer ONLY (DrQ-style); actor uses DETACHED features +
    detached critic; rollout uses no_grad encode. autotune entropy, gamma .9, tau .01, bootstrap-always.

Live dashboard shows the front+wrist views the policy actually consumes.
Run: PYTHONPATH=src DISPLAY=:1 python -m rebot_sim2real.student.train_c51_visual --num-envs 128 ...
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
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--total-timesteps", type=int, default=1_000_000_000)
parser.add_argument("--http-port", type=int, default=8890)
parser.add_argument("--save-dir", type=str,
                    default="rl_output/rl_c51_visual")
parser.add_argument("--target-cube", type=str, default="cube", choices=["cube", "cube_blue", "cube_purple"])
parser.add_argument("--headless", action="store_true")
parser.add_argument("--success-target", type=int, default=5000, help="stop at this many FLEET successes; 0 disables")
parser.add_argument("--config", type=str, default="configs/reward_config.yaml")
parser.add_argument("--buffer-size", type=int, default=200_000)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--num-updates", type=int, default=8,
                    help="grad updates per env-step (UTD). Lower than state-only (CNN encoders are heavy).")
parser.add_argument("--learning-starts", type=int, default=5000)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--gamma", type=float, default=0.9)
parser.add_argument("--tau", type=float, default=0.01)
parser.add_argument("--policy-frequency", type=int, default=4)
parser.add_argument("--num-atoms", type=int, default=101)
parser.add_argument("--v-min", type=float, default=-20.0)
parser.add_argument("--v-max", type=float, default=20.0)
parser.add_argument("--num-q", type=int, default=2)
parser.add_argument("--buffer-device", type=str, default="cuda:0", help="cuda:0 (fast) or cpu (saves VRAM)")
args, _ = parser.parse_known_args()

import yaml
CFG = {}
if os.path.exists(args.config):
    with open(args.config) as f:
        CFG = yaml.safe_load(f) or {}
CFG["enable_cameras"] = True  # visual policy: cameras are the inputs

from isaaclab.app import AppLauncher
simulation_app = AppLauncher({"headless": args.headless, "enable_cameras": True}).app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import SceneEntityCfg, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
import leisaac.tasks.lift_cube  # noqa: F401
from leisaac.tasks.lift_cube import mdp

from rebot_sim2real.env.env_config import configure_env, apply_motor_limits
from rebot_sim2real.env.visual_obs import (
    configure_visual, downsample, STATE_DIM, WRIST_SIZE, FRONT_SIZE)
import rebot_sim2real.env.maniskill_rewards as MR
from rebot_sim2real.tools import rebot_dashboard as dash

DEVICE = "cuda:0"
BUF_DEV = torch.device(args.buffer_device)

# ── Env ──
env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=args.num_envs)
env_cfg = configure_env(env_cfg, CFG, mdp, RewTerm, DoneTerm, SceneEntityCfg, target_cube=args.target_cube)
_obj = getattr(env_cfg.scene, args.target_cube); _cp = _obj.prim_path
env_cfg.scene.gripper_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_left",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
env_cfg.scene.jaw_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_right",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
_obj.spawn.activate_contact_sensors = True
env_cfg.scene.robot.spawn.activate_contact_sensors = True
env_cfg = configure_visual(env_cfg, target_cube=args.target_cube)

env = ManagerBasedRLEnv(cfg=env_cfg)
apply_motor_limits(env, CFG)
N_ENVS = args.num_envs
act_dim = env.action_manager.total_action_dim
print(f"[VC51] num_envs={N_ENVS} act_dim={act_dim} state={STATE_DIM} wrist={WRIST_SIZE} front={FRONT_SIZE}", flush=True)

dash.start(args.http_port, target_cube=args.target_cube)
_cam_map = {}
for nm in ("front", "wrist"):
    try:
        _cam_map[nm] = env.scene[nm]
    except Exception:
        pass
print(f"[VC51] live dashboard cameras: {list(_cam_map.keys())}", flush=True)
os.makedirs(args.save_dir, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────────
#  Networks (SQUINT-exact)
# ────────────────────────────────────────────────────────────────────────────────
def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class CNNEncoder(nn.Module):
    """SQUINT CNN encoder. Input (N,H,W,3) uint8. Output (N,1024). Supports image_size 16/32/64."""
    REPR_DIM = 1024

    def __init__(self, image_size):
        super().__init__()
        if image_size == 16:
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=1), nn.ReLU(), nn.Flatten())
        elif image_size == 32:
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten())
        elif image_size == 64:
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten())
        else:
            raise ValueError(f"unsupported image size {image_size}")
        self.apply(orthogonal_init)

    def forward(self, img_uint8):
        x = img_uint8.permute(0, 3, 1, 2).float() / 255.0 - 0.5
        return self.conv(x)


class Projection(nn.Module):
    """wrist_feat(1024)->50, front_feat(1024)->50, state(21)->256, concat -> 356."""
    REPR_DIM = 50 + 50 + 256

    def __init__(self, n_state):
        super().__init__()
        self.wrist_proj = nn.Sequential(nn.Linear(CNNEncoder.REPR_DIM, 50), nn.LayerNorm(50), nn.Tanh())
        self.front_proj = nn.Sequential(nn.Linear(CNNEncoder.REPR_DIM, 50), nn.LayerNorm(50), nn.Tanh())
        self.state_proj = nn.Sequential(nn.Linear(n_state, 256), nn.LayerNorm(256), nn.ReLU())

    def forward(self, wf, ff, state):
        return torch.cat([self.wrist_proj(wf), self.front_proj(ff), self.state_proj(state)], dim=-1)


class Actor(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, n_state, n_act, hidden=256):
        super().__init__()
        self.proj = Projection(n_state)
        self.fc = nn.Sequential(
            nn.Linear(Projection.REPR_DIM, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, n_act)
        self.logstd = nn.Linear(hidden, n_act)
        self.apply(orthogonal_init)

    def _mean_logstd(self, wf, ff, s):
        x = self.fc(self.proj(wf, ff, s))
        mean = self.mean(x)
        logstd = torch.tanh(self.logstd(x))
        logstd = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (logstd + 1)
        return mean, logstd

    def get_action(self, wf, ff, s):
        mean, logstd = self._mean_logstd(wf, ff, s)
        std = logstd.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        y = torch.tanh(x)
        logp = (normal.log_prob(x) - torch.log(1 - y.pow(2) + 1e-6)).sum(1, keepdim=True)
        return y, logp

    @torch.no_grad()
    def act_deterministic(self, wf, ff, s):
        mean, _ = self._mean_logstd(wf, ff, s)
        return torch.tanh(mean)


class CriticC51(nn.Module):
    def __init__(self, n_state, n_act, num_atoms, v_min, v_max, num_q=2, hidden=512):
        super().__init__()
        self.num_atoms, self.num_q, self.v_min, self.v_max = num_atoms, num_q, v_min, v_max
        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))
        self.proj = Projection(n_state)

        def qnet():
            return nn.Sequential(
                nn.Linear(Projection.REPR_DIM + n_act, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                nn.Linear(hidden, num_atoms))
        self.qs = nn.ModuleList([qnet() for _ in range(num_q)])
        self.apply(orthogonal_init)

    def logits(self, wf, ff, s, a):
        x = torch.cat([self.proj(wf, ff, s), a], dim=-1)
        return torch.stack([q(x) for q in self.qs], dim=0)         # (Q,N,A)

    def q_values(self, wf, ff, s, a):
        probs = F.softmax(self.logits(wf, ff, s, a), dim=-1)
        return (probs * self.support).sum(-1)                      # (Q,N)


@torch.no_grad()
def project_target(target_critic, nwf, nff, ns, na, rewards, bootstrap, gamma):
    logits = target_critic.logits(nwf, nff, ns, na)
    probs = F.softmax(logits, dim=-1)
    support = target_critic.support
    v_min, v_max, A = target_critic.v_min, target_critic.v_max, target_critic.num_atoms
    Q, Nn, _ = probs.shape
    dz = (v_max - v_min) / (A - 1)
    tz = rewards.view(1, Nn, 1) + bootstrap.view(1, Nn, 1) * gamma * support.view(1, 1, A)
    tz = tz.clamp(v_min, v_max).expand(Q, Nn, A)
    b = (tz - v_min) / dz
    l = b.floor().long(); u = b.ceil().long()
    is_int = (u == l)
    l = torch.where((l > 0) & is_int, l - 1, l)
    u = torch.where((l == 0) & is_int, u + 1, u)
    l = l.clamp(0, A - 1); u = u.clamp(0, A - 1)
    m = torch.zeros_like(probs)
    m.scatter_add_(2, l, probs * (u.float() - b))
    m.scatter_add_(2, u, probs * (b - l.float()))
    return m


# ────────────────────────────────────────────────────────────────────────────────
#  Replay buffer (images uint8 + state float). Circular.
# ────────────────────────────────────────────────────────────────────────────────
class Buffer:
    def __init__(self, cap, dev):
        self.cap, self.dev, self.ptr, self.full = cap, dev, 0, False
        z = lambda *s, dt: torch.zeros(*s, dtype=dt, device=dev)
        self.s = z(cap, STATE_DIM, dt=torch.float32)
        self.w = z(cap, WRIST_SIZE, WRIST_SIZE, 3, dt=torch.uint8)
        self.f = z(cap, FRONT_SIZE, FRONT_SIZE, 3, dt=torch.uint8)
        self.a = z(cap, act_dim, dt=torch.float32)
        self.r = z(cap, dt=torch.float32)
        self.ns = z(cap, STATE_DIM, dt=torch.float32)
        self.nw = z(cap, WRIST_SIZE, WRIST_SIZE, 3, dt=torch.uint8)
        self.nf = z(cap, FRONT_SIZE, FRONT_SIZE, 3, dt=torch.uint8)
        self.d = z(cap, dt=torch.float32)

    def add(self, s, w, f, a, r, ns, nw, nf, d):
        B = s.shape[0]
        idx = (torch.arange(B, device=self.dev) + self.ptr) % self.cap
        self.s[idx] = s.to(self.dev); self.w[idx] = w.to(self.dev); self.f[idx] = f.to(self.dev)
        self.a[idx] = a.to(self.dev); self.r[idx] = r.to(self.dev)
        self.ns[idx] = ns.to(self.dev); self.nw[idx] = nw.to(self.dev); self.nf[idx] = nf.to(self.dev)
        self.d[idx] = d.to(self.dev)
        newptr = self.ptr + B
        if newptr >= self.cap:
            self.full = True
        self.ptr = newptr % self.cap

    def sample(self, bs):
        hi = self.cap if self.full else self.ptr
        idx = torch.randint(0, hi, (bs,), device=self.dev)
        g = lambda t: t[idx].to(DEVICE, non_blocking=True)
        return g(self.s), g(self.w), g(self.f), g(self.a), g(self.r), g(self.ns), g(self.nw), g(self.nf), g(self.d)


# ── instantiate ──
enc_w = CNNEncoder(WRIST_SIZE).to(DEVICE)
enc_f = CNNEncoder(FRONT_SIZE).to(DEVICE)
actor = Actor(STATE_DIM, act_dim).to(DEVICE)
critic = CriticC51(STATE_DIM, act_dim, args.num_atoms, args.v_min, args.v_max, args.num_q).to(DEVICE)
critic_target = CriticC51(STATE_DIM, act_dim, args.num_atoms, args.v_min, args.v_max, args.num_q).to(DEVICE)
critic_target.load_state_dict(critic.state_dict())
for p in critic_target.parameters():
    p.requires_grad_(False)

# encoders trained by the CRITIC optimizer (DrQ-style); actor only updates actor params
critic_opt = torch.optim.Adam(list(critic.parameters()) + list(enc_w.parameters()) + list(enc_f.parameters()), lr=args.lr)
actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
target_entropy = -float(act_dim)
log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
alpha_opt = torch.optim.Adam([log_alpha], lr=args.lr)
alpha = log_alpha.exp().detach()

buf = Buffer(args.buffer_size, BUF_DEV)
crit_params = list(critic.parameters())
targ_params = list(critic_target.parameters())
print(f"[VC51] nets ready. UTD={args.num_updates} batch={args.batch_size} buffer={args.buffer_size}@{BUF_DEV} "
      f"gamma={args.gamma} atoms={args.num_atoms}", flush=True)


def update_critic(s, w, f, a, r, ns, nw, nf, d):
    wf_o = enc_w(w); ff_o = enc_f(f)                              # WITH grad (critic trains encoders)
    with torch.no_grad():
        wf_n = enc_w(nw); ff_n = enc_f(nf)
        na, nlogp = actor.get_action(wf_n, ff_n, ns)
        bootstrap = (1.0 - d)
        r_ent = r - bootstrap * args.gamma * (alpha.squeeze() * nlogp.squeeze(-1))
        target_dist = project_target(critic_target, wf_n, ff_n, ns, na, r_ent, bootstrap, args.gamma)
    logits = critic.logits(wf_o, ff_o, s, a)
    logp = F.log_softmax(logits, dim=-1)
    critic_loss = (-(target_dist * logp).sum(-1).mean(-1)).sum()
    critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_opt.step()
    return wf_o.detach(), ff_o.detach(), critic_loss.detach()


def update_actor(s, wf_det, ff_det):
    a_pi, logp = actor.get_action(wf_det, ff_det, s)              # detached features -> no encoder grad
    qv = critic.q_values(wf_det, ff_det, s, a_pi).mean(0)         # grads to critic are wasted (not stepped)
    actor_loss = (alpha.squeeze() * logp.squeeze(-1) - qv).mean()
    actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_opt.step()
    with torch.no_grad():
        _, logp2 = actor.get_action(wf_det, ff_det, s)
    alpha_loss = -(log_alpha.exp() * (logp2 + target_entropy)).mean()
    alpha_opt.zero_grad(set_to_none=True)
    alpha_loss.backward()
    alpha_opt.step()


def soft_update():
    with torch.no_grad():
        torch._foreach_lerp_(targ_params, crit_params, args.tau)


# ────────────────────────────────────────────────────────────────────────────────
#  Rollout helpers: read state + cameras, downsample
# ────────────────────────────────────────────────────────────────────────────────
def read_imgs():
    """current wrist(16) + front(64) downsampled uint8, on DEVICE."""
    w = downsample(env.scene["wrist"].data.output["rgb"], WRIST_SIZE)
    f = downsample(env.scene["front"].data.output["rgb"], FRONT_SIZE)
    return w, f


def sanitize_state(t):
    return torch.nan_to_num(t, nan=0.0, posinf=100.0, neginf=-100.0)


# ── training loop ──
print(f"[VC51] start. target={args.total_timesteps} success_target={args.success_target}", flush=True)
obs_dict, _ = env.reset()
for _ in range(3):  # warmup so camera buffers are populated
    obs_dict, *_ = env.step(torch.zeros((N_ENVS, act_dim), device=DEVICE))
state = sanitize_state(obs_dict["policy"])
w_img, f_img = read_imgs()

global_step, n_iter = 0, 0
ep_rew = np.zeros(N_ENVS, dtype=np.float64)
ep_hist = []
best_reward = float("-inf")
cum_reward, cum_terms = 0.0, {}

while global_step < args.total_timesteps:
    if global_step < args.learning_starts:
        action = torch.rand(N_ENVS, act_dim, device=DEVICE) * 2.0 - 1.0
    else:
        with torch.no_grad():
            wf = enc_w(w_img); ff = enc_f(f_img)
            action, _ = actor.get_action(wf, ff, state)

    obs_dict, rew, terminated, truncated, infos = env.step(action.detach())
    done = (terminated | truncated)
    next_state = sanitize_state(obs_dict["policy"])
    nw_img, nf_img = read_imgs()
    rew_t = torch.nan_to_num(rew.float(), nan=0.0, posinf=0.0, neginf=0.0)

    # bootstrap_at_done='always': all dones are time_outs -> store done=0, bootstrap on next obs
    buf.add(state, w_img, f_img, action.detach(), rew_t,
            next_state, nw_img, nf_img, torch.zeros(N_ENVS, device=DEVICE))

    rew_np = rew_t.detach().cpu().numpy()
    ep_rew += rew_np
    cum_reward += float(rew_np.mean())
    done_np = done.detach().cpu().numpy()
    for i in np.nonzero(done_np)[0]:
        ep_hist.append(float(ep_rew[i]))
        if ep_rew[i] > best_reward:
            best_reward = float(ep_rew[i])
        ep_rew[i] = 0.0
    ep_hist = ep_hist[-200:]

    state, w_img, f_img = next_state, nw_img, nf_img
    global_step += N_ENVS
    n_iter += 1

    if global_step >= args.learning_starts:
        for j in range(args.num_updates):
            batch = buf.sample(args.batch_size)
            wf_det, ff_det, _ = update_critic(*batch)
            if j % args.policy_frequency == 0:
                update_actor(batch[0], wf_det, ff_det)
            soft_update()
        alpha = log_alpha.exp().detach()

    succ = MR._succ_total[0]
    if args.success_target and succ >= args.success_target:
        print(f"[DONE] fleet reached {succ} successes at t={global_step}; stopping.", flush=True)
        torch.save({"enc_w": enc_w.state_dict(), "enc_f": enc_f.state_dict(),
                    "actor": actor.state_dict(), "critic": critic.state_dict(),
                    "log_alpha": log_alpha, "step": global_step},
                   os.path.join(args.save_dir, "vc51_final.pt"))
        break

    if n_iter % 5 == 0:
        rt = {}
        try:
            rm = env.reward_manager
            for name in rm.active_terms:
                cfg = rm.get_term_cfg(name)
                try:
                    val = float(cfg.func(env, **cfg.params).float().mean().item()) * float(cfg.weight) * float(env.step_dt)
                except Exception:
                    val = 0.0
                rt[name] = round(val, 4)
                cum_terms[name] = cum_terms.get(name, 0.0) + val
        except Exception:
            pass
        recent = MR._succ_recent
        sr = (sum(recent) / len(recent)) if recent else 0.0
        dash.push_stats(
            step=global_step, episode=MR._ep_total[0],
            current_reward=float(rew_np.mean()), episode_reward=float(np.mean(ep_hist[-20:])) if ep_hist else 0.0,
            cumulative_reward=cum_reward, best_reward=best_reward if best_reward != float("-inf") else 0.0,
            success_count=succ, success_rate=sr, reward_terms=rt, cumulative_terms=dict(cum_terms), dopamine={})
        if _cam_map:
            try:
                dash.update_cameras(_cam_map)
            except Exception:
                pass

    if n_iter % 50 == 0:
        recent = MR._succ_recent
        sr = (sum(recent) / len(recent)) if recent else 0.0
        print(f"[VC51] t={global_step} eps={MR._ep_total[0]} mean_r={(np.mean(ep_hist[-20:]) if ep_hist else 0):.2f} "
              f"best={best_reward:.2f} success={succ} sr={sr:.3f} alpha={float(alpha):.3f}", flush=True)

    if n_iter % 200 == 0:
        try:
            torch.save({"enc_w": enc_w.state_dict(), "enc_f": enc_f.state_dict(),
                        "actor": actor.state_dict(), "critic": critic.state_dict(),
                        "log_alpha": log_alpha, "step": global_step},
                       os.path.join(args.save_dir, f"vc51_ckpt_{global_step}.pt"))
        except Exception:
            pass

print("[VC51] training loop finished.", flush=True)
try:
    simulation_app.close()
except Exception:
    pass
