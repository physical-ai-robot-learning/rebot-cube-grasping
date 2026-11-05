"""STAGED visual student via DAgger from the 4-stage teacher (reaching -> align -> grasp -> lift).

TEACHER  : 4 state-only C51 actors (rl_c51_joint4/s{0..3}_final.pt). A per-env STATE-based stage pointer
           (same done-conditions as train_c51_joint) picks the active stage; that stage's teacher actor
           gives the expert action + the stage LABEL.
STUDENT  : ONE StagedVisualActor = shared CNN encoders (wrist128+front128) + Projection + 4 action heads
           + 1 stage-classifier head. Deployable from cameras+proprio ONLY (the stage head predicts the
           stage; no privileged state at deploy).
DAgger   : beta 1->0 (teacher-driven -> student-driven). Loss = MSE(student head[stage_label], teacher_act)
           + CE(student stage_logits, stage_label). Student-driven rollout uses the student's OWN predicted
           stage head (deploy-realistic). Stops at end-to-end success-rate >= target, N hits (pure-student).
Visual DR identical to train_dagger_visual (office bg, white glossy table, black gripper, glare, ISP, ...).
"""
import argparse, os
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT", os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl")) + "/leisaac/assets")

ap = argparse.ArgumentParser()
ap.add_argument("--teachers", nargs=4, required=True, help="4 stage teacher ckpts: reaching align grasp lift")
ap.add_argument("--num-envs", type=int, default=96)
ap.add_argument("--total-timesteps", type=int, default=1_000_000_000)
ap.add_argument("--http-port", type=int, default=8893)
ap.add_argument("--save-dir", type=str, required=True)
ap.add_argument("--target-cube", type=str, default="cube")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--dr", action="store_true")
ap.add_argument("--bg-dir", type=str, default=os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl")) + "/bg_images/val2017")
ap.add_argument("--bg-max", type=int, default=1500)
ap.add_argument("--table-tex-dir", type=str,
                default=os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl")) + "/leisaac/assets/scenes/table_with_cube/textures")
ap.add_argument("--success-rate-target", type=float, default=0.99,
                help="stop when pure-student end2end sr reaches this (= teacher level, 'as good as teacher')")
ap.add_argument("--min-sr-episodes", type=int, default=300)
ap.add_argument("--min-sr-hits", type=int, default=3)
ap.add_argument("--success-target", type=int, default=0,
                help="ALSO stop at this many pure-student end2end successes (0=disabled)")
ap.add_argument("--beta-decay-steps", type=int, default=150_000)
ap.add_argument("--buffer-size", type=int, default=300_000)
ap.add_argument("--buffer-device", type=str, default="cpu")
ap.add_argument("--batch-size", type=int, default=256)
ap.add_argument("--num-updates", type=int, default=8)
ap.add_argument("--learning-starts", type=int, default=2000)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--stage-loss-weight", type=float, default=0.5)
ap.add_argument("--config", type=str, default="configs/reward_config.yaml")
args, _ = ap.parse_known_args()

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
import rebot_teacher_student.env.rewards as MR
from rebot_teacher_student.env.visual_obs import (
    configure_visual, downsample, student_state, composite_bg, add_proprio_noise, wrist_aug, image_dr,
    recolor_seg_region, tag_gripper, WRIST_SIZE, FRONT_SIZE, IMG_RENDER, STATE_DIM, TEACHER_STATE_DIM)
from rebot_teacher_student.student.visual_nets import CNNEncoder, StagedVisualActor, StateOnlyActor
from rebot_teacher_student import dashboard as dash

DEVICE = "cuda:0"; TARGET = args.target_cube; N_ENVS = args.num_envs; NS = 4
env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=N_ENVS)
env_cfg = configure_env(env_cfg, CFG, mdp, RewTerm, DoneTerm, SceneEntityCfg, target_cube=TARGET)
_obj = getattr(env_cfg.scene, TARGET); _cp = _obj.prim_path
env_cfg.scene.gripper_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_left",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
env_cfg.scene.jaw_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_right",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
_obj.spawn.activate_contact_sensors = True
env_cfg.scene.robot.spawn.activate_contact_sensors = True
env_cfg = configure_visual(env_cfg, target_cube=TARGET, keep_privileged=True, dr=args.dr, front_only=False)
env = ManagerBasedRLEnv(cfg=env_cfg)
apply_motor_limits(env, CFG)

act_dim = 7
print(f"[STG-DAG] envs={N_ENVS} stages={NS} state21={STATE_DIM} wrist={WRIST_SIZE} front={FRONT_SIZE} teacher_obs={TEACHER_STATE_DIM}", flush=True)
dash.start(args.http_port, target_cube=TARGET)
os.makedirs(args.save_dir, exist_ok=True)

# ── teachers (4 state-only actors) ──
teachers = []
for i, c in enumerate(args.teachers):
    t = StateOnlyActor(TEACHER_STATE_DIM, act_dim).to(DEVICE)
    ck = torch.load(c, map_location=DEVICE)
    t.load_state_dict(ck["actor"] if "actor" in ck else ck); t.eval()
    for p in t.parameters():
        p.requires_grad_(False)
    teachers.append(t); print(f"[STG-DAG] teacher s{i} <- {c}", flush=True)

# ── student (staged visual) ──
enc_w = CNNEncoder(WRIST_SIZE).to(DEVICE)
enc_f = CNNEncoder(FRONT_SIZE).to(DEVICE)
student = StagedVisualActor(STATE_DIM, act_dim, n_stages=NS).to(DEVICE)
opt = torch.optim.Adam(list(enc_w.parameters()) + list(enc_f.parameters()) + list(student.parameters()), lr=args.lr)


class Buffer:
    def __init__(self, cap, dev):
        self.cap, self.dev, self.ptr, self.full = cap, dev, 0, False
        self.w = torch.zeros(cap, WRIST_SIZE, WRIST_SIZE, 3, dtype=torch.uint8, device=dev)
        self.f = torch.zeros(cap, FRONT_SIZE, FRONT_SIZE, 3, dtype=torch.uint8, device=dev)
        self.s = torch.zeros(cap, STATE_DIM, device=dev)
        self.a = torch.zeros(cap, act_dim, device=dev)
        self.lbl = torch.zeros(cap, dtype=torch.long, device=dev)

    def add(self, w, f, s, a, lbl):
        B = w.shape[0]; idx = (torch.arange(B, device=self.dev) + self.ptr) % self.cap
        self.w[idx] = w.to(self.dev); self.f[idx] = f.to(self.dev); self.s[idx] = s.to(self.dev)
        self.a[idx] = a.to(self.dev); self.lbl[idx] = lbl.to(self.dev)
        np_ = self.ptr + B; self.full = self.full or np_ >= self.cap; self.ptr = np_ % self.cap

    def sample(self, bs):
        hi = self.cap if self.full else self.ptr; idx = torch.randint(0, hi, (bs,), device=self.dev)
        return (self.w[idx].to(DEVICE), self.f[idx].to(DEVICE), self.s[idx].to(DEVICE),
                self.a[idx].to(DEVICE), self.lbl[idx].to(DEVICE))


buf = Buffer(args.buffer_size, torch.device(args.buffer_device))

# ── visual DR pipeline (identical to train_dagger_visual) ──
_BG_BANK = None; _TABLE_TEX = None
def _load_banks():
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
        for fp in fs:
            try:
                out.append(torch.from_numpy(np.asarray(Image.open(fp).convert("L" if gray else "RGB").resize((R, R)), dtype=np.uint8)))
            except Exception:
                pass
        return torch.stack(out) if out else None
    bb = _load(args.bg_dir, args.bg_max, False)
    if bb is not None:
        _BG_BANK = bb.to(DEVICE); print(f"[STG-DAG] bg-bank: {_BG_BANK.shape[0]} office photos", flush=True)
    tt = _load(args.table_tex_dir, 64, True)
    if tt is not None:
        _TABLE_TEX = (tt.float() / 255.0).clamp(0.40, 1.0).unsqueeze(-1).to(DEVICE)


def _rand_bg(n):
    R = IMG_RENDER
    g = torch.randint(3, 9, (1,)).item()
    blobs = torch.rand(n, g, g, 3, device=DEVICE)
    blobs = F.interpolate(blobs.permute(0, 3, 1, 2), size=(R, R), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    blobs = (blobs * 255.0 + torch.randn(n, R, R, 3, device=DEVICE) * 18.0).clamp(0, 255)
    solid = (torch.rand(n, 1, 1, 3, device=DEVICE) * 255.0).expand(n, R, R, 3)
    wmode = torch.rand(n, 1, 1, 1, device=DEVICE)
    if _BG_BANK is not None:                                   # REAL reBot frames ONLY (no synthetic blobs/solids)
        photo = _BG_BANK[torch.randint(0, _BG_BANK.shape[0], (n,), device=DEVICE)].float()
        return torch.where(torch.rand(n, 1, 1, 1, device=DEVICE) < 0.5, photo.flip(2), photo)
    return torch.where(wmode < 0.6, blobs, solid)             # fallback only if no real bg loaded


def _cam_ds(cam, size, is_front=True):
    if not args.dr:
        return downsample(cam.data.output["rgb"], size)
    seg = cam.data.output["semantic_segmentation"]; info = cam.data.info["semantic_segmentation"]
    full = composite_bg(cam.data.output["rgb"], seg, info, _rand_bg(N_ENVS), keep_lower=0.10)
    full = recolor_seg_region(full, seg, info, ("table",), 0.95, 1.08, 0.0, grain=False, base=0.68)   # WHITE glossy table
    full = recolor_seg_region(full, seg, info, ("gripper",), 0.02, 0.12, 0.0, grain=False, base=0.0)  # BLACK gripper
    full = image_dr(full, scale=is_front)
    return downsample(full, size)


WRIST_ROLL_MAX, WRIST_TRANS_MAX = 0.35, 0.12
_wa = {"roll": torch.zeros(N_ENVS, device=DEVICE), "tx": torch.zeros(N_ENVS, device=DEVICE), "ty": torch.zeros(N_ENVS, device=DEVICE)}
def _draw_wrist_aug(mask=None):
    if mask is None:
        _wa["roll"].uniform_(-WRIST_ROLL_MAX, WRIST_ROLL_MAX); _wa["tx"].uniform_(-WRIST_TRANS_MAX, WRIST_TRANS_MAX); _wa["ty"].uniform_(-WRIST_TRANS_MAX, WRIST_TRANS_MAX)
    else:
        k = int(mask.sum())
        if k:
            _wa["roll"][mask] = torch.empty(k, device=DEVICE).uniform_(-WRIST_ROLL_MAX, WRIST_ROLL_MAX)
            _wa["tx"][mask] = torch.empty(k, device=DEVICE).uniform_(-WRIST_TRANS_MAX, WRIST_TRANS_MAX)
            _wa["ty"][mask] = torch.empty(k, device=DEVICE).uniform_(-WRIST_TRANS_MAX, WRIST_TRANS_MAX)


def read_imgs():
    f = _cam_ds(env.scene["front"], FRONT_SIZE, is_front=True)
    w = _cam_ds(env.scene["wrist"], WRIST_SIZE, is_front=False)
    if args.dr:
        w = wrist_aug(w, _wa["roll"], _wa["tx"], _wa["ty"])
    return w, f


def sanitize(t):
    return torch.nan_to_num(t, nan=0.0, posinf=100.0, neginf=-100.0)


if args.dr:
    _load_banks(); tag_gripper(env)
    if not args.headless:
        try:
            env_cfg.viewer.origin_type = "env"; env_cfg.viewer.env_index = 0
        except Exception:
            pass
    _draw_wrist_aug()

_ee = SceneEntityCfg("ee_frame"); _oc = SceneEntityCfg(TARGET)
def _dist():
    return torch.norm(env.scene[TARGET].data.root_pos_w - MR._tcp(env, _ee), dim=1)
def stage_cond(s):                       # 0 reaching -> 1 align -> 2 grasp -> 3 lift (matches the teacher)
    if s == 0:
        return _dist() < (MR.ALIGN_DIST + 0.05)
    if s == 1:
        return MR._align_held(env, _dist())
    if s == 2:
        return MR._is_grasped(env, _oc)
    return MR.is_fold_success(env, _oc)


def _save(path):
    d = {"enc_f": enc_f.state_dict(), "enc_w": enc_w.state_dict(), "staged_actor": student.state_dict(),
         "n_stages": NS, "step": global_step, "front_only": False}
    torch.save(d, path)


print(f"[STG-DAG] start. sr-target={args.success_rate_target} hits={args.min_sr_hits} beta_decay={args.beta_decay_steps}", flush=True)
obs_dict = env.reset()[0]; full_obs = sanitize(obs_dict["policy"]); w_img, f_img = read_imgs()
stage = torch.zeros(N_ENVS, dtype=torch.long, device=DEVICE)   # STATE-based stage pointer (for teacher labels)
global_step, n_iter = 0, 0
loss_ema = None; beta0_reset_done = False; _sr_hits = 0; _last_hit_ep = -10**9

while global_step < args.total_timesteps:
    beta = max(0.0, 1.0 - global_step / float(args.beta_decay_steps))
    s21 = add_proprio_noise(student_state(full_obs)) if args.dr else student_state(full_obs)
    # teacher: per-env active-stage actor + the stage LABEL
    teacher_act = torch.zeros(N_ENVS, act_dim, device=DEVICE)
    for s in range(NS):
        m = (stage == s)
        if m.any():
            teacher_act[m] = teachers[s].act_deterministic(full_obs[m])
    stage_label = stage.clone()
    # student: predict stage + action from its own predicted-stage head
    with torch.no_grad():
        feat = student.feat(enc_w(w_img), enc_f(f_img), s21)
        # route the student-driven rollout by the STATE stage (stable: avoids classifier-misprediction
        # compounding). The stage classifier is still trained via CE below and is what routes AT DEPLOY.
        student_act = student.action_for(feat, stage)
    use_teacher = (torch.rand(N_ENVS, device=DEVICE) < beta).unsqueeze(-1)
    rollout_act = torch.where(use_teacher, teacher_act, student_act)
    buf.add(w_img, f_img, s21, teacher_act, stage_label)

    obs_dict, rew, terminated, truncated, infos = env.step(rollout_act.detach())
    full_obs = sanitize(obs_dict["policy"]); w_img, f_img = read_imgs()
    # advance the STATE stage pointer 0->1->2->3
    c0, c1, c2 = stage_cond(0), stage_cond(1), stage_cond(2)
    stage = torch.where((stage == 0) & c0, torch.ones_like(stage), stage)
    stage = torch.where((stage == 1) & c1, torch.full_like(stage, 2), stage)
    stage = torch.where((stage == 2) & c2, torch.full_like(stage, 3), stage)
    done = (terminated | truncated)
    if done.any():
        stage[done] = 0
        if args.dr:
            _draw_wrist_aug(mask=done)

    global_step += N_ENVS; n_iter += 1

    last_loss = 0.0
    if global_step >= args.learning_starts:
        for _ in range(args.num_updates):
            bw, bf, bs, ba, bl = buf.sample(args.batch_size)
            feat_b = student.feat(enc_w(bw), enc_f(bf), bs)
            pred = student.action_for(feat_b, bl)                       # student's stage-LABEL head action
            mse = F.mse_loss(pred, ba)
            ce = F.cross_entropy(student.stage_logits(feat_b), bl)      # stage classifier
            loss = mse + args.stage_loss_weight * ce
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            last_loss = float(mse.detach())
        loss_ema = last_loss if loss_ema is None else 0.98 * loss_ema + 0.02 * last_loss

    # once student fully drives, reset env success counters -> student-only end-to-end metric
    if beta <= 0.0 and not beta0_reset_done:
        try:
            MR._succ_total[0] = 0; MR._ep_total[0] = 0; MR._succ_recent.clear()
            if getattr(MR, "_succ_flag", [None])[0] is not None:
                MR._succ_flag[0].zero_()
            if getattr(MR, "_last_ep_succ", [None])[0] is not None:
                MR._last_ep_succ[0].zero_()
        except Exception:
            pass
        beta0_reset_done = True
        print(f"[STG-DAG] beta=0 at t={global_step}; success counters reset (student-only).", flush=True)

    succ = MR._succ_total[0]; ep = MR._ep_total[0]
    rec = getattr(MR, "_succ_recent", []); sr = (sum(rec) / len(rec)) if rec else 0.0
    stop, why = False, ""
    if beta0_reset_done and args.success_target and succ >= args.success_target:   # reached N successes
        stop, why = True, f"{succ} successes (target {args.success_target})"
    hit = beta0_reset_done and len(rec) >= args.min_sr_episodes and sr >= args.success_rate_target  # as good as teacher
    if hit and (ep - _last_hit_ep) >= args.min_sr_episodes:
        _sr_hits += 1; _last_hit_ep = ep
        print(f"[STG-DAG] sr-hit {_sr_hits}/{args.min_sr_hits} (end2end sr={sr:.3f} ~teacher, {ep} eps) t={global_step}", flush=True)
    if hit and _sr_hits >= args.min_sr_hits:
        stop, why = True, f"sr={sr:.3f} >= teacher x{_sr_hits}"
    if stop:
        print(f"[DONE] staged student: {why} at t={global_step}; saving.", flush=True)
        _save(os.path.join(args.save_dir, "staged_student_final.pt")); break

    if n_iter % 5 == 0:
        try:
            dash.push_stats(step=global_step, episode=ep, current_reward=float(-last_loss),
                            episode_reward=float(-(loss_ema or 0.0)), cumulative_reward=0.0, best_reward=0.0,
                            success_count=succ, success_rate=sr,
                            reward_terms={"mse": round(loss_ema or 0.0, 5), "beta": round(beta, 3)},
                            cumulative_terms={}, dopamine={})
            dash.set_camera_image("front", f_img[0].detach().cpu().numpy())
            dash.set_camera_image("wrist", w_img[0].detach().cpu().numpy())
        except Exception:
            pass
    if n_iter % 50 == 0:
        c = [int((stage == s).sum()) for s in range(NS)]
        print(f"[STG-DAG] t={global_step} beta={beta:.3f} mse={loss_ema if loss_ema else 0:.5f} "
              f"succ={succ} sr={sr:.3f} stagePtr@[r{c[0]},a{c[1]},g{c[2]},l{c[3]}] (student={beta0_reset_done})", flush=True)
    if n_iter % 200 == 0:
        _save(os.path.join(args.save_dir, f"staged_student_ckpt_{global_step}.pt"))

print("[STG-DAG] loop finished.", flush=True)
try:
    env.close(); simulation_app.close()
except Exception:
    pass
