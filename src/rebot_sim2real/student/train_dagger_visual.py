"""DAgger teacher->student distillation: state-only privileged TEACHER -> VISUAL student (reBot LiftCube).

TEACHER  : the trained state-only C51 actor (rl_c51_smooth/c51_final.pt), input = full 43D privileged
           obs (incl. obj_pos/tcp/grasp). Solves the task ~52%. Frozen; provides the LABEL action.
STUDENT  : visual policy -- CNN encoders (wrist 16x16 + front 64x64) + Projection + Actor; input =
           images + 21D proprio state (deployable from cameras + joint encoders only).

DAgger loop (per env-step, all envs):
  - teacher_act = teacher(full_obs43)                      # the expert label at the visited state
  - student_act = student(images, state21)                 # deterministic tanh(mean)
  - beta = max(0, 1 - step/beta_decay) ; per-env Bernoulli(beta) picks teacher vs student to DRIVE
  - store (wrist16, front64, state21, teacher_act) ; step env with the chosen (mixed) action
  - supervised update: minimise MSE(student tanh(mean), teacher_act) through encoders + actor
Both actions feed the SAME env action term (scale .05 + joint clamps + elbow gate), so imitating the
teacher's raw [-1,1] action reproduces the same constrained motion.

The dashboard shows the live front+wrist camera RENDERS (native 128x128; the student consumes 16x16/64x64
area-downsamples of these) plus imitation loss, beta, and success rate.
Run: PYTHONPATH=src DISPLAY=:1 python -m rebot_sim2real.student.train_dagger_visual --num-envs 128 \
        --teacher-checkpoint rl_output/rl_c51_smooth/c51_final.pt
"""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT", os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl")) + "/leisaac/assets")

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--teacher-checkpoint", type=str,
                    default="rl_output/rl_c51_smooth/c51_final.pt")
parser.add_argument("--total-timesteps", type=int, default=1_000_000_000)
parser.add_argument("--http-port", type=int, default=8890)
parser.add_argument("--save-dir", type=str,
                    default="rl_output/rl_dagger_visual")
parser.add_argument("--target-cube", type=str, default="cube", choices=["cube", "cube_blue", "cube_purple"])
parser.add_argument("--headless", action="store_true")
parser.add_argument("--success-target", type=int, default=5000,
                    help="stop at this many STUDENT-driven (beta=0) fleet successes; 0 disables")
parser.add_argument("--success-rate-target", type=float, default=0.0,
                    help="ALSO stop if the student's recent sr reaches this (e.g. the teacher's sr); 0 disables")
parser.add_argument("--min-sr-recent", type=int, default=300,
                    help="require >= this many recent finished episodes before the sr-stop can fire")
parser.add_argument("--resume", type=str, default=None,
                    help="resume the student from a dagger ckpt (loads enc_f/enc_w/actor + step)")
parser.add_argument("--config", type=str, default="configs/reward_config.yaml")
parser.add_argument("--buffer-size", type=int, default=300_000)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--num-updates", type=int, default=8, help="supervised grad updates per env-step")
parser.add_argument("--learning-starts", type=int, default=2000)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--beta-decay-steps", type=int, default=150_000,
                    help="env-steps over which DAgger beta anneals 1->0 (teacher-driven -> student-driven)")
parser.add_argument("--buffer-device", type=str, default="cuda:0")
parser.add_argument("--dr", action="store_true",
                    help="visual domain randomization (greenscreen bg + lighting + qpos noise) for sim2real")
parser.add_argument("--front-only", action="store_true",
                    help="drop the wrist camera entirely; student = front(64) + state only (sim2real: "
                         "the real wrist mount/framing is OOD). Deploy then needs no wrist camera.")
parser.add_argument("--bg-dir", type=str, default=os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl")) + "/bg_images/val2017",
                    help="folder of real-scene photos (people/office/nature, e.g. COCO val2017) MIXED into the DR bg bank")
parser.add_argument("--bg-max", type=int, default=1500, help="max photos preloaded into the GPU bg bank")
parser.add_argument("--table-tex-dir", type=str,
                    default=os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl")) + "/leisaac/assets/scenes/table_with_cube/textures",
                    help="folder of surface textures (e.g. wood grain) for the TABLE region; color is randomized per-env")
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

from rebot_sim2real.env.env_config import configure_env, apply_motor_limits
from rebot_sim2real.env.visual_obs import (
    configure_visual, downsample, student_state, composite_bg, add_proprio_noise, wrist_aug, image_dr,
    recolor_seg_region, tag_gripper, WRIST_SIZE, FRONT_SIZE, IMG_RENDER, STATE_DIM, TEACHER_STATE_DIM)
from rebot_sim2real.student.visual_nets import CNNEncoder, Actor, StateOnlyActor
import rebot_sim2real.env.maniskill_rewards as MR
from rebot_sim2real.tools import rebot_dashboard as dash

DEVICE = "cuda:0"
BUF_DEV = torch.device(args.buffer_device)

# ── Env (FULL 43D obs for the teacher + front/wrist cameras for the student) ──
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
print(f"[DAG] num_envs={N_ENVS} act_dim={act_dim} student_state={STATE_DIM} wrist={WRIST_SIZE} front={FRONT_SIZE}", flush=True)

# ── Teacher (frozen, privileged 43D) ──
teacher = StateOnlyActor(TEACHER_STATE_DIM, act_dim).to(DEVICE)
ck = torch.load(args.teacher_checkpoint, map_location=DEVICE)
teacher.load_state_dict(ck["actor"] if "actor" in ck else ck)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)
print(f"[DAG] teacher loaded: {args.teacher_checkpoint} (step {ck.get('step','?')})", flush=True)

# ── Student (visual) ──
FRONT_ONLY = args.front_only            # drop the wrist input (sim2real: real wrist mount is OOD)
enc_w = None if FRONT_ONLY else CNNEncoder(WRIST_SIZE).to(DEVICE)
enc_f = CNNEncoder(FRONT_SIZE).to(DEVICE)
actor = Actor(STATE_DIM, act_dim, use_wrist=not FRONT_ONLY).to(DEVICE)
_params = list(enc_f.parameters()) + list(actor.parameters()) + (list(enc_w.parameters()) if enc_w else [])
opt = torch.optim.Adam(_params, lr=args.lr)

# resume the student (preserve a run past the beta-anneal): load nets + step. Optimizer state is NOT
# checkpointed (Adam re-warms quickly) and the buffer refills from scratch.
_resume_step = 0
if args.resume:
    _rck = torch.load(args.resume, map_location=DEVICE)
    enc_f.load_state_dict(_rck["enc_f"]); actor.load_state_dict(_rck["actor"])
    if enc_w is not None and "enc_w" in _rck:
        enc_w.load_state_dict(_rck["enc_w"])
    _resume_step = int(_rck.get("step", 0))
    print(f"[DAG] RESUMED student from {args.resume} (step {_resume_step}; beta will be 0 if past anneal)", flush=True)


def WF(w):
    """wrist features (None in front-only mode; the Projection ignores it then)."""
    return None if FRONT_ONLY else enc_w(w)

dash.start(args.http_port, target_cube=args.target_cube)
_cam_map = {}
for nm in ("front", "wrist"):
    try:
        _cam_map[nm] = env.scene[nm]
    except Exception:
        pass
print(f"[DAG] live dashboard cameras: {list(_cam_map.keys())}", flush=True)
os.makedirs(args.save_dir, exist_ok=True)


# ── Replay buffer: (wrist16, front64, state21, teacher_action). No next/reward (supervised). ──
class Buffer:
    def __init__(self, cap, dev, store_wrist=True):
        self.cap, self.dev, self.ptr, self.full = cap, dev, 0, False
        self.store_wrist = store_wrist
        self.w = torch.zeros(cap, WRIST_SIZE, WRIST_SIZE, 3, dtype=torch.uint8, device=dev) if store_wrist else None
        self.f = torch.zeros(cap, FRONT_SIZE, FRONT_SIZE, 3, dtype=torch.uint8, device=dev)
        self.s = torch.zeros(cap, STATE_DIM, dtype=torch.float32, device=dev)
        self.a = torch.zeros(cap, act_dim, dtype=torch.float32, device=dev)

    def add(self, w, f, s, a):
        B = f.shape[0]
        idx = (torch.arange(B, device=self.dev) + self.ptr) % self.cap
        if self.store_wrist:
            self.w[idx] = w.to(self.dev)
        self.f[idx] = f.to(self.dev); self.s[idx] = s.to(self.dev); self.a[idx] = a.to(self.dev)
        newptr = self.ptr + B
        if newptr >= self.cap:
            self.full = True
        self.ptr = newptr % self.cap

    def sample(self, bs):
        hi = self.cap if self.full else self.ptr
        idx = torch.randint(0, hi, (bs,), device=self.dev)
        g = lambda t: t[idx].to(DEVICE, non_blocking=True)
        w = g(self.w) if self.store_wrist else None
        return w, g(self.f), g(self.s), g(self.a)


buf = Buffer(args.buffer_size, BUF_DEV, store_wrist=not FRONT_ONLY)
print(f"[DAG] buffer={args.buffer_size}@{BUF_DEV} batch={args.batch_size} UTD={args.num_updates} "
      f"beta_decay={args.beta_decay_steps} lr={args.lr}", flush=True)


_BG_BANK = None      # (M,R,R,3) uint8 — real-scene photos for the WORLD (upper background) region
_TABLE_TEX = None    # (T,R,R,1) float[~0.4,1] — surface-texture luminance for the TABLE region (recolored per-env)
def _load_bg_bank():
    """preload (1) real-scene PHOTOS (--bg-dir: people/office/nature) -> _BG_BANK for the WORLD region, and
    (2) surface TEXTURES (--table-tex-dir: wood grain) -> _TABLE_TEX luminance so the TABLE region stays a
    TEXTURED surface whose COLOR is randomized per-env (texture + color), instead of being deleted."""
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
        print(f"[DAG] bg-bank: {_BG_BANK.shape[0]} photo backgrounds (world) from {args.bg_dir}", flush=True)
    else:
        print(f"[DAG] bg-bank: no photos ({args.bg_dir}) -> synthetic world only", flush=True)
    tt = _load(args.table_tex_dir, 64, True)
    if tt is not None:
        _TABLE_TEX = (tt.float() / 255.0).clamp(0.40, 1.0).unsqueeze(-1).to(DEVICE)   # (T,R,R,1) grain luminance
        print(f"[DAG] table-tex: {_TABLE_TEX.shape[0]} surface textures from {args.table_tex_dir}", flush=True)
    else:
        print(f"[DAG] table-tex: none ({args.table_tex_dir}) -> flat-color table", flush=True)


def _rand_bg(n):
    """FAR-BACKGROUND = OFFICE-like (matches the real OFFICE inference env): mostly real-scene PHOTOS
    (_BG_BANK = COCO indoor/office/scenes) + some blob clutter + solid color for robustness to the office
    variation. The TABLE is the rendered table (kept by composite_bg) so the cube/arm don't float.
    (If a photo of the ACTUAL real office is provided, point --bg-dir at it for an even closer match.)"""
    R = IMG_RENDER
    g = torch.randint(3, 9, (1,)).item()
    blobs = torch.rand(n, g, g, 3, device=DEVICE)
    blobs = torch.nn.functional.interpolate(blobs.permute(0, 3, 1, 2), size=(R, R), mode="bilinear",
                                            align_corners=False).permute(0, 2, 3, 1)
    blobs = (blobs * 255.0 + torch.randn(n, R, R, 3, device=DEVICE) * 18.0).clamp(0, 255)
    solid = (torch.rand(n, 1, 1, 3, device=DEVICE) * 255.0).expand(n, R, R, 3)
    wmode = torch.rand(n, 1, 1, 1, device=DEVICE)
    if _BG_BANK is not None:                                   # REAL reBot frames ONLY (no synthetic blobs/solids)
        photo = _BG_BANK[torch.randint(0, _BG_BANK.shape[0], (n,), device=DEVICE)].float()
        return torch.where(torch.rand(n, 1, 1, 1, device=DEVICE) < 0.5, photo.flip(2), photo)
    return torch.where(wmode < 0.6, blobs, solid)             # fallback only if no real bg loaded


def _cam_ds(cam, size, is_front=True):
    """downsampled image for one camera. --dr: KEEP the rendered foreground (robot+cube) + the lower TABLE
    region, replace only the upper FAR-background with _rand_bg (so the arm/cube sit on the real rendered
    table, no floating), then image_dr (ISP color + glare + front-only scale/aspect). keep_lower: front 0.62
    (table+near kept, upper far-bg replaced); wrist 1.0 (down-view = table fills the frame, keep all)."""
    if args.dr:
        # table is semantic-tagged -> composite_bg keeps {robot,cube,table} by SILHOUETTE (no hard horizon);
        # only the true far-background is replaced. keep_lower=0.10 is a thin safety floor if seg lags.
        full = composite_bg(cam.data.output["rgb"], cam.data.output["semantic_segmentation"],
                            cam.data.info["semantic_segmentation"], _rand_bg(N_ENVS), keep_lower=0.10)
        _seg = cam.data.output["semantic_segmentation"]; _info = cam.data.info["semantic_segmentation"]
        full = recolor_seg_region(full, _seg, _info, ("table",), 0.95, 1.08, 0.0, grain=False, base=0.68)  # WHITE table (gloss via image_dr glare)
        full = recolor_seg_region(full, _seg, _info, ("gripper",), 0.02, 0.12, 0.0, grain=False, base=0.0)  # BLACK gripper (fixed)
        # NOTE: robot ARM is NOT recolored (left as rendered) per user; cubes never recolored (task cue)
        full = image_dr(full, scale=is_front)
        return downsample(full, size)
    return downsample(cam.data.output["rgb"], size)


# per-episode wrist-cam roll + translate DR (redrawn on reset) — robust to real wrist mount roll/offset.
WRIST_ROLL_MAX, WRIST_TRANS_MAX = 0.35, 0.12     # ±20° roll, ±12% translate
def _draw_wrist_aug(mask=None):
    """(re)draw per-env wrist roll/translate; mask=None -> all envs, else only the masked (reset) envs."""
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
    tag_gripper(env)            # separate 'gripper' semantic so only the gripper is painted black
if args.dr and not FRONT_ONLY:
    _draw_wrist_aug()


def read_imgs():
    f = _cam_ds(env.scene["front"], FRONT_SIZE, is_front=True)
    if FRONT_ONLY:
        return None, f
    w = _cam_ds(env.scene["wrist"], WRIST_SIZE, is_front=False)
    if args.dr:
        w = wrist_aug(w, _wa["roll"], _wa["tx"], _wa["ty"])   # per-episode roll/pos jitter
    return w, f


def sanitize(t):
    return torch.nan_to_num(t, nan=0.0, posinf=100.0, neginf=-100.0)


def _save_ckpt(path):
    d = {"enc_f": enc_f.state_dict(), "actor": actor.state_dict(),
         "step": global_step, "front_only": FRONT_ONLY}
    if enc_w is not None:
        d["enc_w"] = enc_w.state_dict()
    torch.save(d, path)


# ── DAgger loop ──
print(f"[DAG] start. total={args.total_timesteps} success_target={args.success_target}", flush=True)
obs_dict, _ = env.reset()
for _ in range(3):  # warmup camera buffers
    obs_dict, *_ = env.step(torch.zeros((N_ENVS, act_dim), device=DEVICE))
full_obs = sanitize(obs_dict["policy"])
w_img, f_img = read_imgs()

global_step, n_iter = _resume_step, 0
loss_ema = None
beta0_reset_done = False
ep_succ_hist = []

while global_step < args.total_timesteps:
    beta = max(0.0, 1.0 - global_step / float(args.beta_decay_steps))
    # student sees NOISY proprio (dr); the teacher (below) always acts on the CLEAN 43D obs.
    s21 = add_proprio_noise(student_state(full_obs)) if args.dr else student_state(full_obs)

    teacher_act = teacher.act_deterministic(full_obs)                  # (N,7) label — CLEAN obs
    with torch.no_grad():
        student_act = actor.act_deterministic(WF(w_img), enc_f(f_img), s21)
    # per-env Bernoulli(beta): True -> teacher drives this env this step
    use_teacher = (torch.rand(N_ENVS, device=DEVICE) < beta).unsqueeze(-1)
    rollout_act = torch.where(use_teacher, teacher_act, student_act)

    # store the expert label at the visited (student) observation
    buf.add(w_img, f_img, s21, teacher_act)

    obs_dict, rew, terminated, truncated, infos = env.step(rollout_act.detach())
    if args.dr and not FRONT_ONLY:                       # reset envs -> redraw per-episode wrist roll/pos
        _draw_wrist_aug(mask=(terminated | truncated))
    full_obs = sanitize(obs_dict["policy"])
    w_img, f_img = read_imgs()
    global_step += N_ENVS
    n_iter += 1

    # ── supervised update (regress student tanh(mean) -> teacher action) ──
    last_loss = 0.0
    if global_step >= args.learning_starts:
        for _ in range(args.num_updates):
            bw, bf, bs, ba = buf.sample(args.batch_size)
            pred = actor.mean_action(WF(bw), enc_f(bf), bs)            # tanh(mean), WITH grad
            loss = F.mse_loss(pred, ba)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach())
        loss_ema = last_loss if loss_ema is None else 0.98 * loss_ema + 0.02 * last_loss

    # once the student fully drives (beta hit 0), reset the env-side success counters so the
    # success_target / reported rate reflect the STUDENT only (not teacher-driven successes).
    if beta <= 0.0 and not beta0_reset_done:
        try:
            MR._succ_total[0] = 0; MR._ep_total[0] = 0
            MR._succ_recent.clear()
            # also clear the IN-FLIGHT per-env success latch + last-ep flag, else teacher-era successes
            # latched mid-episode get tallied into the freshly-zeroed student-only metric on next reset.
            if getattr(MR, "_succ_flag", [None])[0] is not None:
                MR._succ_flag[0].zero_()
            if getattr(MR, "_last_ep_succ", [None])[0] is not None:
                MR._last_ep_succ[0].zero_()
        except Exception:
            pass
        beta0_reset_done = True
        print(f"[DAG] beta=0 reached at t={global_step}; success counters reset (student-only metric).", flush=True)

    succ = MR._succ_total[0]
    _rec = getattr(MR, "_succ_recent", [])
    _sr_now = (sum(_rec) / len(_rec)) if _rec else 0.0
    _hit_succ = bool(args.success_target) and beta0_reset_done and succ >= args.success_target
    _hit_sr = bool(args.success_rate_target) and beta0_reset_done and len(_rec) >= args.min_sr_recent and _sr_now >= args.success_rate_target
    if _hit_succ or _hit_sr:
        why = f"{succ} successes (target {args.success_target})" if _hit_succ else f"sr={_sr_now:.3f} >= teacher sr {args.success_rate_target} over {len(_rec)} eps"
        print(f"[DONE] student stop: {why} at t={global_step}; stopping.", flush=True)
        _save_ckpt(os.path.join(args.save_dir, "dagger_final.pt"))
        break

    # ── dashboard ──
    if n_iter % 5 == 0:
        recent = getattr(MR, "_succ_recent", [])
        sr = (sum(recent) / len(recent)) if recent else 0.0
        dash.push_stats(
            step=global_step, episode=MR._ep_total[0],
            current_reward=float(-last_loss), episode_reward=float(-(loss_ema or 0.0)),
            cumulative_reward=0.0, best_reward=0.0,
            success_count=succ, success_rate=sr,
            reward_terms={"imitation_mse": round(loss_ema or 0.0, 5), "beta": round(beta, 3),
                          "last_mse": round(last_loss, 5)},
            cumulative_terms={}, dopamine={})
        # show the DR'd STUDENT INPUT (128, greenscreen/photo-bg/glare/color all applied) on the dashboard,
        # not the raw render — so the live cam reflects the actual domain-randomized input the student sees.
        try:
            dash.set_camera_image("front", f_img[0].detach().cpu().numpy())
            if w_img is not None:
                dash.set_camera_image("wrist", w_img[0].detach().cpu().numpy())
        except Exception:
            pass

    if n_iter % 50 == 0:
        recent = getattr(MR, "_succ_recent", [])
        sr = (sum(recent) / len(recent)) if recent else 0.0
        print(f"[DAG] t={global_step} beta={beta:.3f} mse={loss_ema if loss_ema else 0:.5f} "
              f"succ={succ} sr={sr:.3f} (student={beta0_reset_done})", flush=True)

    if n_iter % 200 == 0:
        try:
            _save_ckpt(os.path.join(args.save_dir, f"dagger_ckpt_{global_step}.pt"))
        except Exception:
            pass

print("[DAG] loop finished.", flush=True)
try:
    simulation_app.close()
except Exception:
    pass
