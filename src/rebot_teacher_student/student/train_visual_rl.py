"""VISUAL RL fine-tune: warm-start the DISTILLED visual student, then improve it with C51-SAC on the
TRUE env reward (breaks the imitation ceiling -- distillation caps where the student can't reproduce the
teacher's privileged-info actions; RL on the env reward lets the student find its OWN vision-feasible policy).

PIPELINE
  warm-start  : load enc_w/enc_f/actor from a dagger checkpoint (the distilled student).
  add critic  : CriticC51 (visual) + target critic + target ENCODERS (soft-updated; image-RL stability).
  C51-SAC     : collect with the stochastic actor on the SAME DR'd visual domain as distillation, store
                (wrist,front,state21, action, reward, next*), train critic (encoder learns from critic loss
                only -- DrQ; actor update DETACHES the encoder features so it can't corrupt the representation).
  critic warmup: freeze the actor for the first --critic-warmup env-steps so the fresh critic learns Q for the
                warm policy before any actor gradient moves it (else garbage Q wrecks the warm policy).
  reward      : the env's 8-term lift reward (identical config to the single state-teacher). done=0 always
                (continuing formulation -- mirrors the proven joint teacher).

Run (needs a free GPU + a distilled checkpoint to --warm):
  PYTHONPATH=src DISPLAY=:1 python -m rebot_teacher_student.student.train_visual_rl --headless --dr \
     --warm rl_output/rl_dagger_single_real/dagger_final.pt --bg-dir bg_images/office_real \
     --num-envs 96 --buffer-device cpu --http-port 8891 --save-dir rl_output/rl_visual_rl_single
"""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT", os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl")) + "/leisaac/assets")

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=96)
parser.add_argument("--warm", type=str, required=True,
                    help="distilled student checkpoint (loads enc_f/enc_w/actor) to fine-tune")
parser.add_argument("--total-timesteps", type=int, default=1_000_000_000)
parser.add_argument("--http-port", type=int, default=8891)
parser.add_argument("--save-dir", type=str,
                    default="rl_output/rl_visual_rl_single")
parser.add_argument("--target-cube", type=str, default="cube", choices=["cube", "cube_blue", "cube_purple"])
parser.add_argument("--headless", action="store_true")
parser.add_argument("--success-target", type=int, default=0,
                    help="stop at this many fleet successes; 0 disables")
parser.add_argument("--success-rate-target", type=float, default=0.0,
                    help="stop if recent sr reaches this (e.g. the teacher's sr); 0 disables")
parser.add_argument("--min-sr-recent", type=int, default=300,
                    help="require >= this many recent finished episodes before the sr-stop can fire")
parser.add_argument("--config", type=str, default="configs/reward_config.yaml")
# RL (C51-SAC) hyperparameters -- mirror the joint teacher
parser.add_argument("--buffer-size", type=int, default=60_000)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--num-updates", type=int, default=2, help="grad updates per env-step (UTD)")
parser.add_argument("--policy-frequency", type=int, default=2, help="actor update every N critic updates")
parser.add_argument("--learning-starts", type=int, default=2000)
parser.add_argument("--critic-warmup", type=int, default=20_000,
                    help="env-steps to train ONLY the critic (actor frozen) before fine-tuning the warm policy")
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--gamma", type=float, default=0.9)
parser.add_argument("--tau", type=float, default=0.01)
parser.add_argument("--init-alpha", type=float, default=0.1, help="initial SAC temperature (low: gentle on the warm policy)")
parser.add_argument("--num-atoms", type=int, default=101)
parser.add_argument("--v-min", type=float, default=-20.0)
parser.add_argument("--v-max", type=float, default=20.0)
parser.add_argument("--num-q", type=int, default=2)
parser.add_argument("--buffer-device", type=str, default="cpu")
parser.add_argument("--dr", action="store_true", help="visual domain randomization (same as distillation)")
parser.add_argument("--front-only", action="store_true", help="drop the wrist camera (front + state only)")
parser.add_argument("--bg-dir", type=str, default=os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl")) + "/bg_images/office_real")
parser.add_argument("--bg-max", type=int, default=1500)
parser.add_argument("--table-tex-dir", type=str,
                    default=os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl")) + "/leisaac/assets/scenes/table_with_cube/textures")
args, _ = parser.parse_known_args()

import yaml
CFG = {}
if os.path.exists(args.config):
    with open(args.config) as f:
        CFG = yaml.safe_load(f) or {}
CFG["enable_cameras"] = True

from isaaclab.app import AppLauncher
simulation_app = AppLauncher({"headless": args.headless, "enable_cameras": True}).app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import SceneEntityCfg, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
import leisaac.tasks.lift_cube  # noqa: F401
from leisaac.tasks.lift_cube import mdp

from rebot_teacher_student.env.env_config import configure_env, apply_motor_limits
from rebot_teacher_student.env.visual_obs import (
    configure_visual, downsample, student_state, composite_bg, add_proprio_noise, wrist_aug, image_dr,
    recolor_seg_region, tag_gripper, WRIST_SIZE, FRONT_SIZE, IMG_RENDER, STATE_DIM)
from rebot_teacher_student.student.visual_nets import CNNEncoder, Actor, CriticC51, project_target
import rebot_teacher_student.env.rewards as MR
from rebot_teacher_student import dashboard as dash

DEVICE = "cuda:0"
BUF_DEV = torch.device(args.buffer_device)

# ── Env (cameras for the student; privileged obs kept but unused by the visual policy) ──
env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=args.num_envs)
env_cfg = configure_env(env_cfg, CFG, mdp, RewTerm, DoneTerm, SceneEntityCfg, target_cube=args.target_cube)
_obj = getattr(env_cfg.scene, args.target_cube); _cp = _obj.prim_path
env_cfg.scene.gripper_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_left",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
env_cfg.scene.jaw_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_right",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
_obj.spawn.activate_contact_sensors = True
env_cfg.scene.robot.spawn.activate_contact_sensors = True
env_cfg = configure_visual(env_cfg, target_cube=args.target_cube, keep_privileged=True, dr=args.dr,
                           front_only=args.front_only)

env = ManagerBasedRLEnv(cfg=env_cfg)
apply_motor_limits(env, CFG)
N_ENVS = args.num_envs
act_dim = env.action_manager.total_action_dim
FRONT_ONLY = args.front_only
print(f"[VRL] num_envs={N_ENVS} act_dim={act_dim} student_state={STATE_DIM} wrist={WRIST_SIZE} front={FRONT_SIZE} "
      f"front_only={FRONT_ONLY}", flush=True)

# ── Student nets: encoders + actor (warm) + critic (fresh) + targets ──
enc_w = None if FRONT_ONLY else CNNEncoder(WRIST_SIZE).to(DEVICE)
enc_f = CNNEncoder(FRONT_SIZE).to(DEVICE)
actor = Actor(STATE_DIM, act_dim, use_wrist=not FRONT_ONLY).to(DEVICE)
critic = CriticC51(STATE_DIM, act_dim, args.num_atoms, args.v_min, args.v_max, args.num_q,
                   use_wrist=not FRONT_ONLY).to(DEVICE)
target_critic = CriticC51(STATE_DIM, act_dim, args.num_atoms, args.v_min, args.v_max, args.num_q,
                          use_wrist=not FRONT_ONLY).to(DEVICE)
target_critic.load_state_dict(critic.state_dict())
for p in target_critic.parameters():
    p.requires_grad_(False)
# target ENCODERS (soft-updated) for stable next-state features in image RL
enc_w_t = None if FRONT_ONLY else CNNEncoder(WRIST_SIZE).to(DEVICE)
enc_f_t = CNNEncoder(FRONT_SIZE).to(DEVICE)
if enc_w_t is not None:
    enc_w_t.load_state_dict(enc_w.state_dict())
    for p in enc_w_t.parameters():
        p.requires_grad_(False)
enc_f_t.load_state_dict(enc_f.state_dict())
for p in enc_f_t.parameters():
    p.requires_grad_(False)

# warm-start the encoders + actor from the distilled student
_wck = torch.load(args.warm, map_location=DEVICE)
enc_f.load_state_dict(_wck["enc_f"]); actor.load_state_dict(_wck["actor"])
if enc_w is not None and "enc_w" in _wck:
    enc_w.load_state_dict(_wck["enc_w"])
if enc_w_t is not None and "enc_w" in _wck:
    enc_w_t.load_state_dict(_wck["enc_w"])
enc_f_t.load_state_dict(_wck["enc_f"])
_warm_step = int(_wck.get("step", 0))
print(f"[VRL] warm student <- {args.warm} (distill step {_warm_step})", flush=True)

# optimizers: critic loss trains critic + ENCODERS (DrQ); actor loss trains the actor only (encoder detached)
_enc_params = list(enc_f.parameters()) + (list(enc_w.parameters()) if enc_w else [])
c_opt = torch.optim.Adam(list(critic.parameters()) + _enc_params, lr=args.lr)
a_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
log_alpha = torch.tensor([float(np.log(args.init_alpha))], requires_grad=True, device=DEVICE)
al_opt = torch.optim.Adam([log_alpha], lr=args.lr)
alpha = log_alpha.exp().detach()
target_entropy = -float(act_dim)


def WF(w):
    return None if FRONT_ONLY else enc_w(w)


def WF_t(w):
    return None if FRONT_ONLY else enc_w_t(w)


dash.start(args.http_port, target_cube=args.target_cube)
os.makedirs(args.save_dir, exist_ok=True)


# ── Replay buffer: (wrist, front, state21, action, reward, next_wrist, next_front, next_state21) ──
# done is NOT stored: continuing formulation (boot=1 always), mirroring the proven joint teacher.
class Buffer:
    def __init__(self, cap, dev, store_wrist=True):
        self.cap, self.dev, self.ptr, self.full = cap, dev, 0, False
        self.store_wrist = store_wrist
        z_img = lambda: torch.zeros(cap, WRIST_SIZE, WRIST_SIZE, 3, dtype=torch.uint8, device=dev)
        zf_img = lambda: torch.zeros(cap, FRONT_SIZE, FRONT_SIZE, 3, dtype=torch.uint8, device=dev)
        self.w = z_img() if store_wrist else None
        self.nw = z_img() if store_wrist else None
        self.f = zf_img(); self.nf = zf_img()
        self.s = torch.zeros(cap, STATE_DIM, dtype=torch.float32, device=dev)
        self.ns = torch.zeros(cap, STATE_DIM, dtype=torch.float32, device=dev)
        self.a = torch.zeros(cap, act_dim, dtype=torch.float32, device=dev)
        self.r = torch.zeros(cap, dtype=torch.float32, device=dev)

    def add(self, w, f, s, a, r, nw, nf, ns):
        B = f.shape[0]
        idx = (torch.arange(B, device=self.dev) + self.ptr) % self.cap
        if self.store_wrist:
            self.w[idx] = w.to(self.dev); self.nw[idx] = nw.to(self.dev)
        self.f[idx] = f.to(self.dev); self.nf[idx] = nf.to(self.dev)
        self.s[idx] = s.to(self.dev); self.ns[idx] = ns.to(self.dev)
        self.a[idx] = a.to(self.dev); self.r[idx] = r.to(self.dev)
        newptr = self.ptr + B
        if newptr >= self.cap:
            self.full = True
        self.ptr = newptr % self.cap

    def ready(self, bs):
        return self.full or self.ptr >= bs

    def sample(self, bs):
        hi = self.cap if self.full else self.ptr
        idx = torch.randint(0, hi, (bs,), device=self.dev)
        g = lambda t: t[idx].to(DEVICE, non_blocking=True)
        w = g(self.w) if self.store_wrist else None
        nw = g(self.nw) if self.store_wrist else None
        return w, g(self.f), g(self.s), g(self.a), g(self.r), nw, g(self.nf), g(self.ns)


buf = Buffer(args.buffer_size, BUF_DEV, store_wrist=not FRONT_ONLY)
print(f"[VRL] buffer={args.buffer_size}@{BUF_DEV} batch={args.batch_size} UTD={args.num_updates} "
      f"gamma={args.gamma} tau={args.tau} init_alpha={args.init_alpha} critic_warmup={args.critic_warmup}", flush=True)


# ── DR pipeline (IDENTICAL to train_dagger_visual: the student must see the same domain it was distilled on) ──
_BG_BANK = None
_TABLE_TEX = None
def _load_bg_bank():
    global _BG_BANK, _TABLE_TEX
    import glob
    from PIL import Image
    R = IMG_RENDER
    def _load(dir_, mx, gray):
        if not dir_ or not os.path.isdir(dir_):
            return None
        fs = sorted(glob.glob(os.path.join(dir_, "*.jpg")) + glob.glob(os.path.join(dir_, "*.jpeg")) +
                    glob.glob(os.path.join(dir_, "*.png")))[:mx]
        out = []
        for f in fs:
            try:
                im = Image.open(f).convert("L" if gray else "RGB").resize((R, R))
                out.append(torch.from_numpy(np.asarray(im, dtype=np.uint8)))
            except Exception:
                pass
        return torch.stack(out) if out else None
    bb = _load(args.bg_dir, args.bg_max, False)
    if bb is not None:
        _BG_BANK = bb.to(DEVICE)
        print(f"[VRL] bg-bank: {_BG_BANK.shape[0]} photo backgrounds from {args.bg_dir}", flush=True)
    else:
        print(f"[VRL] bg-bank: no photos ({args.bg_dir}) -> synthetic world only", flush=True)
    tt = _load(args.table_tex_dir, 64, True)
    if tt is not None:
        _TABLE_TEX = (tt.float() / 255.0).clamp(0.40, 1.0).unsqueeze(-1).to(DEVICE)


def _rand_bg(n):
    R = IMG_RENDER
    g = torch.randint(3, 9, (1,)).item()
    blobs = torch.rand(n, g, g, 3, device=DEVICE)
    blobs = torch.nn.functional.interpolate(blobs.permute(0, 3, 1, 2), size=(R, R), mode="bilinear",
                                            align_corners=False).permute(0, 2, 3, 1)
    blobs = (blobs * 255.0 + torch.randn(n, R, R, 3, device=DEVICE) * 18.0).clamp(0, 255)
    solid = (torch.rand(n, 1, 1, 3, device=DEVICE) * 255.0).expand(n, R, R, 3)
    wmode = torch.rand(n, 1, 1, 1, device=DEVICE)
    if _BG_BANK is not None:                                   # REAL frames ONLY (no synthetic blobs/solids)
        photo = _BG_BANK[torch.randint(0, _BG_BANK.shape[0], (n,), device=DEVICE)].float()
        return torch.where(torch.rand(n, 1, 1, 1, device=DEVICE) < 0.5, photo.flip(2), photo)
    return torch.where(wmode < 0.6, blobs, solid)


def _cam_ds(cam, size, is_front=True):
    if args.dr:
        full = composite_bg(cam.data.output["rgb"], cam.data.output["semantic_segmentation"],
                            cam.data.info["semantic_segmentation"], _rand_bg(N_ENVS), keep_lower=0.10)
        _seg = cam.data.output["semantic_segmentation"]; _info = cam.data.info["semantic_segmentation"]
        full = recolor_seg_region(full, _seg, _info, ("table",), 0.95, 1.08, 0.0, grain=False, base=0.68)
        full = recolor_seg_region(full, _seg, _info, ("gripper",), 0.02, 0.12, 0.0, grain=False, base=0.0)
        full = image_dr(full, scale=is_front)
        return downsample(full, size)
    return downsample(cam.data.output["rgb"], size)


WRIST_ROLL_MAX, WRIST_TRANS_MAX = 0.35, 0.12
def _draw_wrist_aug(mask=None):
    if mask is None:
        _wa["roll"].uniform_(-WRIST_ROLL_MAX, WRIST_ROLL_MAX)
        _wa["tx"].uniform_(-WRIST_TRANS_MAX, WRIST_TRANS_MAX)
        _wa["ty"].uniform_(-WRIST_TRANS_MAX, WRIST_TRANS_MAX)
    else:
        k = int(mask.sum())
        if k:
            _wa["roll"][mask] = torch.empty(k, device=DEVICE).uniform_(-WRIST_ROLL_MAX, WRIST_ROLL_MAX)
            _wa["tx"][mask] = torch.empty(k, device=DEVICE).uniform_(-WRIST_TRANS_MAX, WRIST_TRANS_MAX)
            _wa["ty"][mask] = torch.empty(k, device=DEVICE).uniform_(-WRIST_TRANS_MAX, WRIST_TRANS_MAX)


_wa = {"roll": torch.zeros(N_ENVS, device=DEVICE), "tx": torch.zeros(N_ENVS, device=DEVICE),
       "ty": torch.zeros(N_ENVS, device=DEVICE)}
if args.dr:
    _load_bg_bank()
    tag_gripper(env)
if args.dr and not FRONT_ONLY:
    _draw_wrist_aug()


def read_imgs():
    f = _cam_ds(env.scene["front"], FRONT_SIZE, is_front=True)
    if FRONT_ONLY:
        return None, f
    w = _cam_ds(env.scene["wrist"], WRIST_SIZE, is_front=False)
    if args.dr:
        w = wrist_aug(w, _wa["roll"], _wa["tx"], _wa["ty"])
    return w, f


def sanitize(t):
    return torch.nan_to_num(t, nan=0.0, posinf=100.0, neginf=-100.0)


def _save_ckpt(path):
    d = {"enc_f": enc_f.state_dict(), "actor": actor.state_dict(), "critic": critic.state_dict(),
         "log_alpha": log_alpha.detach(), "step": global_step, "front_only": FRONT_ONLY}
    if enc_w is not None:
        d["enc_w"] = enc_w.state_dict()
    torch.save(d, path)


# ── C51-SAC update (visual). Encoder learns from the CRITIC loss; actor sees DETACHED features. ──
def update(do_actor):
    w, f, s, a, r, nw, nf, ns = buf.sample(args.batch_size)
    # critic
    with torch.no_grad():
        nwf = WF_t(nw); nff = enc_f_t(nf)
        na, nlogp = actor.get_action(nwf, nff, ns)
        boot = torch.ones_like(r)                                  # continuing (done=0), mirrors joint teacher
        r_ent = r - boot * args.gamma * (alpha.squeeze() * nlogp.squeeze(-1))
        tgt = project_target(target_critic, nwf, nff, ns, na, r_ent, boot, args.gamma)
    wf = WF(w); ff = enc_f(f)
    logp = F.log_softmax(critic.logits(wf, ff, s, a), dim=-1)
    closs = (-(tgt * logp).sum(-1).mean(-1)).sum()
    c_opt.zero_grad(set_to_none=True); closs.backward(); c_opt.step()
    # actor (encoder features DETACHED -> only the actor is updated)
    if do_actor:
        wf_d = wf.detach() if wf is not None else None
        ff_d = ff.detach()
        api, alogp = actor.get_action(wf_d, ff_d, s)
        qv = critic.q_values(wf_d, ff_d, s, api).mean(0)
        aloss = (alpha.squeeze() * alogp.squeeze(-1) - qv).mean()
        a_opt.zero_grad(set_to_none=True); aloss.backward(); a_opt.step()
        with torch.no_grad():
            _, lp2 = actor.get_action(wf_d, ff_d, s)
        alloss = -(log_alpha.exp() * (lp2 + target_entropy)).mean()
        al_opt.zero_grad(set_to_none=True); alloss.backward(); al_opt.step()
    return float(closs.detach())


# ── RL loop ──
print(f"[VRL] start. total={args.total_timesteps} success_target={args.success_target} "
      f"sr_target={args.success_rate_target}", flush=True)
obs_dict, _ = env.reset()
for _ in range(3):  # warmup camera buffers
    obs_dict, *_ = env.step(torch.zeros((N_ENVS, act_dim), device=DEVICE))
full_obs = sanitize(obs_dict["policy"])
w_img, f_img = read_imgs()

global_step, n_iter = 0, 0
closs_ema = None

while global_step < args.total_timesteps:
    s21 = add_proprio_noise(student_state(full_obs)) if args.dr else student_state(full_obs)
    with torch.no_grad():
        act, _ = actor.get_action(WF(w_img), enc_f(f_img), s21)    # stochastic (exploration)

    next_dict, rew, terminated, truncated, infos = env.step(act.detach())
    if args.dr and not FRONT_ONLY:
        _draw_wrist_aug(mask=(terminated | truncated))
    next_obs = sanitize(next_dict["policy"])
    ns21 = add_proprio_noise(student_state(next_obs)) if args.dr else student_state(next_obs)
    nw_img, nf_img = read_imgs()
    rew_t = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0).float()

    buf.add(w_img, f_img, s21, act.detach(), rew_t, nw_img, nf_img, ns21)

    full_obs = next_obs; w_img, f_img = nw_img, nf_img
    global_step += N_ENVS; n_iter += 1

    # ── updates ──
    last_closs = 0.0
    if global_step >= args.learning_starts and buf.ready(args.batch_size):
        do_actor_phase = global_step >= args.critic_warmup     # freeze actor during critic warmup
        for j in range(args.num_updates):
            do_actor = do_actor_phase and (j % args.policy_frequency == 0)
            last_closs = update(do_actor)
            with torch.no_grad():
                torch._foreach_lerp_(list(target_critic.parameters()), list(critic.parameters()), args.tau)
                torch._foreach_lerp_(list(enc_f_t.parameters()), list(enc_f.parameters()), args.tau)
                if enc_w_t is not None:
                    torch._foreach_lerp_(list(enc_w_t.parameters()), list(enc_w.parameters()), args.tau)
        alpha = log_alpha.exp().detach()
        closs_ema = last_closs if closs_ema is None else 0.98 * closs_ema + 0.02 * last_closs

    succ = MR._succ_total[0]
    _rec = getattr(MR, "_succ_recent", [])
    _sr_now = (sum(_rec) / len(_rec)) if _rec else 0.0
    _hit_succ = bool(args.success_target) and succ >= args.success_target
    _hit_sr = bool(args.success_rate_target) and len(_rec) >= args.min_sr_recent and _sr_now >= args.success_rate_target
    if _hit_succ or _hit_sr:
        why = f"{succ} successes (target {args.success_target})" if _hit_succ else \
              f"sr={_sr_now:.3f} >= {args.success_rate_target} over {len(_rec)} eps"
        print(f"[DONE] visual-RL stop: {why} at t={global_step}; saving.", flush=True)
        _save_ckpt(os.path.join(args.save_dir, "visual_rl_final.pt"))
        break

    # ── dashboard ──
    if n_iter % 5 == 0:
        sr = _sr_now
        warm = "critic-warmup" if global_step < args.critic_warmup else "fine-tune"
        dash.push_stats(
            step=global_step, episode=MR._ep_total[0],
            current_reward=float(rew_t.mean()), episode_reward=float(rew_t.mean()),
            cumulative_reward=0.0, best_reward=0.0,
            success_count=succ, success_rate=sr,
            reward_terms={"critic_loss": round(closs_ema or 0.0, 4), "alpha": round(float(alpha), 4),
                          "phase": warm, "mean_rew": round(float(rew_t.mean()), 4)},
            cumulative_terms={}, dopamine={})
        try:
            dash.set_camera_image("front", f_img[0].detach().cpu().numpy())
            if w_img is not None:
                dash.set_camera_image("wrist", w_img[0].detach().cpu().numpy())
        except Exception:
            pass

    if n_iter % 50 == 0:
        phase = "warmup" if global_step < args.critic_warmup else "tune"
        print(f"[VRL] t={global_step} {phase} closs={closs_ema if closs_ema else 0:.4f} "
              f"alpha={float(alpha):.4f} succ={succ} sr={_sr_now:.3f} rew={float(rew_t.mean()):.4f}", flush=True)

    if n_iter % 200 == 0:
        try:
            _save_ckpt(os.path.join(args.save_dir, f"visual_rl_ckpt_{global_step}.pt"))
        except Exception:
            pass

print("[VRL] loop finished.", flush=True)
try:
    simulation_app.close()
except Exception:
    pass
