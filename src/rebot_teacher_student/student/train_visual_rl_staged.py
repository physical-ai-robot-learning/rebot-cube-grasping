"""STAGED visual RL fine-tune: warm-start the DISTILLED staged visual student, then improve it with a
4-stage C51-SAC on the env reward (the visual analogue of train_c51_joint -- 4 actor-critics co-trained
through a SHARED CNN encoder, each env routed by its STATE stage pointer reaching->align->grasp->lift).

STUDENT  : shared enc_w/enc_f (CNNEncoder) + StagedVisualActor (4 stochastic action heads + 1 stage
           classifier). Warm-started from the staged distillation (logstd heads start fresh -> strict=False).
CRITICS  : 4 visual CriticC51 (one per stage) + 4 target critics + 1 shared target-encoder pair.
LOOP     : per-env STATE stage pointer (same done-conditions as the teacher). The active stage's head acts
           (stochastic); transitions (+ a stage-done bonus, once) go into THAT stage's image buffer. Each
           stage's critic trains on the env reward; its actor head maximises that critic; the SHARED encoder
           learns from the critic loss only (DrQ: actor sees DETACHED encoder features). The stage classifier
           is co-trained by CE on the STATE-stage label (for deploy routing). alpha auto-tuned per stage.
critic-warmup: actor+classifier frozen for the first --critic-warmup env-steps so the fresh critics learn Q
           for the warm policy before any gradient moves it. done=0 always (continuing; mirrors joint teacher).

Run (needs a free GPU + a distilled staged checkpoint to --warm):
  PYTHONPATH=src DISPLAY=:1 python -m rebot_teacher_student.student.train_visual_rl_staged --headless --dr \
     --warm rl_output/rl_dagger_staged/staged_student_final.pt --bg-dir bg_images/office_real \
     --num-envs 96 --buffer-device cpu --http-port 8894 --save-dir rl_output/rl_visual_rl_staged
"""
import argparse, os
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT", os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl")) + "/leisaac/assets")

ap = argparse.ArgumentParser()
ap.add_argument("--warm", type=str, required=True, help="distilled staged student ckpt (enc_f/enc_w/staged_actor)")
ap.add_argument("--num-envs", type=int, default=96)
ap.add_argument("--total-timesteps", type=int, default=1_000_000_000)
ap.add_argument("--http-port", type=int, default=8894)
ap.add_argument("--save-dir", type=str, required=True)
ap.add_argument("--target-cube", type=str, default="cube")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--dr", action="store_true")
ap.add_argument("--bg-dir", type=str, default=os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl")) + "/bg_images/office_real")
ap.add_argument("--bg-max", type=int, default=1500)
ap.add_argument("--table-tex-dir", type=str,
                default=os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl")) + "/leisaac/assets/scenes/table_with_cube/textures")
ap.add_argument("--success-rate-target", type=float, default=0.0, help="stop when end2end sr reaches this; 0 disables")
ap.add_argument("--min-sr-episodes", type=int, default=300)
ap.add_argument("--min-sr-hits", type=int, default=3)
ap.add_argument("--success-target", type=int, default=0, help="ALSO stop at this many end2end successes; 0 disables")
ap.add_argument("--bonus", type=float, default=5.0, help="one-time reward bonus when a stage's done-condition is first met")
ap.add_argument("--buffer-size", type=int, default=20_000, help="PER-STAGE image replay capacity")
ap.add_argument("--buffer-device", type=str, default="cpu")
ap.add_argument("--batch-size", type=int, default=256)
ap.add_argument("--num-updates", type=int, default=2, help="grad updates per env-step (UTD)")
ap.add_argument("--policy-frequency", type=int, default=2, help="actor update every N critic updates")
ap.add_argument("--learning-starts", type=int, default=2000)
ap.add_argument("--critic-warmup", type=int, default=20_000, help="env-steps of critic-only (actor frozen) warmup")
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--gamma", type=float, default=0.9)
ap.add_argument("--tau", type=float, default=0.01)
ap.add_argument("--init-alpha", type=float, default=0.1)
ap.add_argument("--stage-loss-weight", type=float, default=0.5)
ap.add_argument("--num-atoms", type=int, default=101)
ap.add_argument("--v-min", type=float, default=-20.0)
ap.add_argument("--v-max", type=float, default=20.0)
ap.add_argument("--num-q", type=int, default=2)
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
from rebot_teacher_student.env.env_config import configureEnv, applyMotorLimits
import rebot_teacher_student.env.rewards as MR
from rebot_teacher_student.env.visual_obs import (
    configureVisual, downsample, studentState, compositeBg, addProprioNoise, wristAug, imageDr,
    recolorSegRegion, tagGripper, WRIST_SIZE, FRONT_SIZE, IMG_RENDER, STATE_DIM)
from rebot_teacher_student.student.visual_nets import CNNEncoder, StagedVisualActor, CriticC51, projectTarget
from rebot_teacher_student import dashboard as dash

DEVICE = "cuda:0"; TARGET = args.target_cube; N_ENVS = args.num_envs; NS = 4
BUF_DEV = torch.device(args.buffer_device)

# ── Env (cameras for the student; privileged obs kept for the state-stage pointer) ──
env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=N_ENVS)
env_cfg = configureEnv(env_cfg, CFG, mdp, RewTerm, DoneTerm, SceneEntityCfg, target_cube=TARGET)
_obj = getattr(env_cfg.scene, TARGET); _cp = _obj.prim_path
env_cfg.scene.gripper_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_left",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
env_cfg.scene.jaw_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_right",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
_obj.spawn.activate_contact_sensors = True
env_cfg.scene.robot.spawn.activate_contact_sensors = True
env_cfg = configureVisual(env_cfg, target_cube=TARGET, keep_privileged=True, dr=args.dr, front_only=False)
env = ManagerBasedRLEnv(cfg=env_cfg)
applyMotorLimits(env, CFG)

act_dim = 7
print(f"[SVRL] envs={N_ENVS} stages={NS} act={act_dim} state21={STATE_DIM} wrist={WRIST_SIZE} front={FRONT_SIZE}", flush=True)
dash.start(args.http_port, targetCube=TARGET)
os.makedirs(args.save_dir, exist_ok=True)

# ── student: shared encoders + staged actor (warm) + 4 visual critics (fresh) + targets ──
enc_w = CNNEncoder(WRIST_SIZE).to(DEVICE)
enc_f = CNNEncoder(FRONT_SIZE).to(DEVICE)
enc_w_t = CNNEncoder(WRIST_SIZE).to(DEVICE)
enc_f_t = CNNEncoder(FRONT_SIZE).to(DEVICE)
student = StagedVisualActor(STATE_DIM, act_dim, n_stages=NS).to(DEVICE)
critics = [CriticC51(STATE_DIM, act_dim, args.num_atoms, args.v_min, args.v_max, args.num_q).to(DEVICE) for _ in range(NS)]
targets = [CriticC51(STATE_DIM, act_dim, args.num_atoms, args.v_min, args.v_max, args.num_q).to(DEVICE) for _ in range(NS)]
for s in range(NS):
    targets[s].load_state_dict(critics[s].state_dict())
    for p in targets[s].parameters():
        p.requires_grad_(False)

# warm-start shared encoders + staged actor (logstd heads are NEW -> strict=False)
_wck = torch.load(args.warm, map_location=DEVICE)
enc_f.load_state_dict(_wck["enc_f"]); enc_w.load_state_dict(_wck["enc_w"])
enc_f_t.load_state_dict(_wck["enc_f"]); enc_w_t.load_state_dict(_wck["enc_w"])
_missing = student.load_state_dict(_wck["staged_actor"], strict=False)
for p in list(enc_w_t.parameters()) + list(enc_f_t.parameters()):
    p.requires_grad_(False)
_warm_step = int(_wck.get("step", 0))
print(f"[SVRL] warm staged student <- {args.warm} (distill step {_warm_step}; fresh logstd heads: "
      f"{[tK for tK in _missing.missing_keys if 'logstd' in tK][:2]}...)", flush=True)

# optimizers: critic loss trains the 4 critics + the SHARED encoder (per-stage zero_grad isolates each critic);
# actor loss trains the student (proj + heads + classifier) only, on DETACHED encoder features.
_enc_params = list(enc_f.parameters()) + list(enc_w.parameters())
c_opt = torch.optim.Adam(sum([list(tC.parameters()) for tC in critics], []) + _enc_params, lr=args.lr)
a_opt = torch.optim.Adam(student.parameters(), lr=args.lr)
log_alpha = [torch.tensor([float(np.log(args.init_alpha))], requires_grad=True, device=DEVICE) for _ in range(NS)]
al_opt = [torch.optim.Adam([log_alpha[tS]], lr=args.lr) for tS in range(NS)]
alpha = [log_alpha[tS].exp().detach() for tS in range(NS)]
target_entropy = -float(act_dim)


# ── per-stage image replay buffer: (wrist, front, state21, action, reward, next_wrist, next_front, next_state21) ──
class Buffer:
    def __init__(self, cap, dev):
        self.cap, self.dev, self.ptr, self.full = cap, dev, 0, False
        tZi = lambda S: torch.zeros(cap, S, S, 3, dtype=torch.uint8, device=dev)
        self.w = tZi(WRIST_SIZE); self.nw = tZi(WRIST_SIZE)
        self.f = tZi(FRONT_SIZE); self.nf = tZi(FRONT_SIZE)
        self.s = torch.zeros(cap, STATE_DIM, device=dev); self.ns = torch.zeros(cap, STATE_DIM, device=dev)
        self.a = torch.zeros(cap, act_dim, device=dev); self.r = torch.zeros(cap, device=dev)

    def add(self, w, f, s, a, r, nw, nf, ns):
        tB = w.shape[0]
        if tB == 0:
            return
        tIdx = (torch.arange(tB, device=self.dev) + self.ptr) % self.cap
        self.w[tIdx] = w.to(self.dev); self.nw[tIdx] = nw.to(self.dev)
        self.f[tIdx] = f.to(self.dev); self.nf[tIdx] = nf.to(self.dev)
        self.s[tIdx] = s.to(self.dev); self.ns[tIdx] = ns.to(self.dev)
        self.a[tIdx] = a.to(self.dev); self.r[tIdx] = r.to(self.dev)
        tNp = self.ptr + tB; self.full = self.full or tNp >= self.cap; self.ptr = tNp % self.cap

    def ready(self, bs):
        return self.full or self.ptr >= bs

    def sample(self, bs):
        tHi = self.cap if self.full else self.ptr; tIdx = torch.randint(0, tHi, (bs,), device=self.dev)
        tG = lambda t: t[tIdx].to(DEVICE, non_blocking=True)
        return tG(self.w), tG(self.f), tG(self.s), tG(self.a), tG(self.r), tG(self.nw), tG(self.nf), tG(self.ns)


bufs = [Buffer(args.buffer_size, BUF_DEV) for _ in range(NS)]
print(f"[SVRL] per-stage buffer={args.buffer_size}@{BUF_DEV} batch={args.batch_size} UTD={args.num_updates} "
      f"gamma={args.gamma} tau={args.tau} bonus={args.bonus} critic_warmup={args.critic_warmup}", flush=True)

# ── visual DR pipeline (identical to train_dagger_staged) ──
_BG_BANK = None; _TABLE_TEX = None
def _loadBanks():
    global _BG_BANK, _TABLE_TEX
    import glob
    from PIL import Image
    tR = IMG_RENDER
    def _load(dir_, mx, gray):
        if not dir_ or not os.path.isdir(dir_):
            return None
        tFs = sorted(glob.glob(os.path.join(dir_, "*.jpg")) + glob.glob(os.path.join(dir_, "*.jpeg")) +
                    glob.glob(os.path.join(dir_, "*.png")))[:mx]
        tOut = []
        for tFp in tFs:
            try:
                tOut.append(torch.from_numpy(np.asarray(Image.open(tFp).convert("L" if gray else "RGB").resize((tR, tR)), dtype=np.uint8)))
            except Exception:
                pass
        return torch.stack(tOut) if tOut else None
    tBb = _load(args.bg_dir, args.bg_max, False)
    if tBb is not None:
        _BG_BANK = tBb.to(DEVICE); print(f"[SVRL] bg-bank: {_BG_BANK.shape[0]} office photos", flush=True)
    tTt = _load(args.table_tex_dir, 64, True)
    if tTt is not None:
        _TABLE_TEX = (tTt.float() / 255.0).clamp(0.40, 1.0).unsqueeze(-1).to(DEVICE)


def _randBg(n):
    tR = IMG_RENDER
    tG = torch.randint(3, 9, (1,)).item()
    tBlobs = torch.rand(n, tG, tG, 3, device=DEVICE)
    tBlobs = F.interpolate(tBlobs.permute(0, 3, 1, 2), size=(tR, tR), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    tBlobs = (tBlobs * 255.0 + torch.randn(n, tR, tR, 3, device=DEVICE) * 18.0).clamp(0, 255)
    tSolid = (torch.rand(n, 1, 1, 3, device=DEVICE) * 255.0).expand(n, tR, tR, 3)
    tWmode = torch.rand(n, 1, 1, 1, device=DEVICE)
    if _BG_BANK is not None:                                   # REAL frames ONLY (no synthetic blobs/solids)
        tPhoto = _BG_BANK[torch.randint(0, _BG_BANK.shape[0], (n,), device=DEVICE)].float()
        return torch.where(torch.rand(n, 1, 1, 1, device=DEVICE) < 0.5, tPhoto.flip(2), tPhoto)
    return torch.where(tWmode < 0.6, tBlobs, tSolid)


def _camDs(cam, size, isFront=True):
    if not args.dr:
        return downsample(cam.data.output["rgb"], size)
    tSeg = cam.data.output["semantic_segmentation"]; tInfo = cam.data.info["semantic_segmentation"]
    tFull = compositeBg(cam.data.output["rgb"], tSeg, tInfo, _randBg(N_ENVS), keep_lower=0.10)
    tFull = recolorSegRegion(tFull, tSeg, tInfo, ("table",), 0.95, 1.08, 0.0, grain=False, base=0.68)   # WHITE glossy table
    tFull = recolorSegRegion(tFull, tSeg, tInfo, ("gripper",), 0.02, 0.12, 0.0, grain=False, base=0.0)  # BLACK gripper
    tFull = imageDr(tFull, scale=isFront)
    return downsample(tFull, size)


WRIST_ROLL_MAX, WRIST_TRANS_MAX = 0.35, 0.12
_wa = {"roll": torch.zeros(N_ENVS, device=DEVICE), "tx": torch.zeros(N_ENVS, device=DEVICE), "ty": torch.zeros(N_ENVS, device=DEVICE)}
def _drawWristAug(mask=None):
    if mask is None:
        _wa["roll"].uniform_(-WRIST_ROLL_MAX, WRIST_ROLL_MAX); _wa["tx"].uniform_(-WRIST_TRANS_MAX, WRIST_TRANS_MAX); _wa["ty"].uniform_(-WRIST_TRANS_MAX, WRIST_TRANS_MAX)
    else:
        tK = int(mask.sum())
        if tK:
            _wa["roll"][mask] = torch.empty(tK, device=DEVICE).uniform_(-WRIST_ROLL_MAX, WRIST_ROLL_MAX)
            _wa["tx"][mask] = torch.empty(tK, device=DEVICE).uniform_(-WRIST_TRANS_MAX, WRIST_TRANS_MAX)
            _wa["ty"][mask] = torch.empty(tK, device=DEVICE).uniform_(-WRIST_TRANS_MAX, WRIST_TRANS_MAX)


def readImgs():
    tF = _camDs(env.scene["front"], FRONT_SIZE, isFront=True)
    tW = _camDs(env.scene["wrist"], WRIST_SIZE, isFront=False)
    if args.dr:
        tW = wristAug(tW, _wa["roll"], _wa["tx"], _wa["ty"])
    return tW, tF


def sanitize(t):
    return torch.nan_to_num(t, nan=0.0, posinf=100.0, neginf=-100.0)


if args.dr:
    _loadBanks(); tagGripper(env); _drawWristAug()

_ee = SceneEntityCfg("ee_frame"); _oc = SceneEntityCfg(TARGET)
def _dist():
    return torch.norm(env.scene[TARGET].data.root_pos_w - MR._tcp(env, _ee), dim=1)
def stageCond(s):                       # 0 reaching -> 1 align -> 2 grasp -> 3 lift (matches the teacher)
    if s == 0:
        return _dist() < (MR.ALIGN_DIST + 0.05)
    if s == 1:
        return MR._alignHeld(env, _dist())
    if s == 2:
        return MR._isGrasped(env, _oc)
    return MR.isFoldSuccess(env, _oc)


def _save(path):
    torch.save({"enc_f": enc_f.state_dict(), "enc_w": enc_w.state_dict(), "staged_actor": student.state_dict(),
                "critics": [tC.state_dict() for tC in critics], "log_alpha": [tLa.detach() for tLa in log_alpha],
                "n_stages": NS, "step": global_step, "front_only": False}, path)


# ── C51-SAC update for one stage. Encoder learns from the critic loss; actor sees DETACHED features. ──
def updateStage(s, doActor):
    tW, tF, tSt, tA, tR2, tNw, tNf, tNs = bufs[s].sample(args.batch_size)
    with torch.no_grad():
        tNwf = enc_w_t(tNw); tNff = enc_f_t(tNf)
        tNfeat = student.feat(tNwf, tNff, tNs)
        tNa, tNlogp = student.getActionStage(tNfeat, s)
        tBoot = torch.ones_like(tR2)                                  # continuing (done=0), mirrors joint teacher
        tREnt = tR2 - tBoot * args.gamma * (alpha[s].squeeze() * tNlogp.squeeze(-1))
        tTgt = projectTarget(targets[s], tNwf, tNff, tNs, tNa, tREnt, tBoot, args.gamma)
    tWf = enc_w(tW); tFf = enc_f(tF)
    tLogp = F.log_softmax(critics[s].logits(tWf, tFf, tSt, tA), dim=-1)
    tCloss = (-(tTgt * tLogp).sum(-1).mean(-1)).sum()
    c_opt.zero_grad(set_to_none=True); tCloss.backward(); c_opt.step()   # only critics[s] + encoder have grad
    if doActor:
        tWfD = tWf.detach(); tFfD = tFf.detach()
        tFeatD = student.feat(tWfD, tFfD, tSt)
        tApi, tAlogp = student.getActionStage(tFeatD, s)
        tQv = critics[s].qValues(tWfD, tFfD, tSt, tApi).mean(0)
        tLbl = torch.full((tSt.shape[0],), s, device=DEVICE, dtype=torch.long)
        tCe = F.cross_entropy(student.stageLogits(tFeatD), tLbl)         # stage classifier (deploy routing)
        tAloss = (alpha[s].squeeze() * tAlogp.squeeze(-1) - tQv).mean() + args.stage_loss_weight * tCe
        a_opt.zero_grad(set_to_none=True); tAloss.backward(); a_opt.step()
        with torch.no_grad():
            _, tLp2 = student.getActionStage(student.feat(tWfD, tFfD, tSt), s)
        tAlloss = -(log_alpha[s].exp() * (tLp2 + target_entropy)).mean()
        al_opt[s].zero_grad(set_to_none=True); tAlloss.backward(); al_opt[s].step()
    return float(tCloss.detach())


def softUpdate(tgt, src):
    with torch.no_grad():
        torch._foreach_lerp_(list(tgt.parameters()), list(src.parameters()), args.tau)


# ── RL loop ──
print(f"[SVRL] start. total={args.total_timesteps} sr_target={args.success_rate_target} success_target={args.success_target}", flush=True)
obs_dict = env.reset()[0]
for _ in range(3):  # warmup camera buffers
    obs_dict, *_ = env.step(torch.zeros((N_ENVS, act_dim), device=DEVICE))
full_obs = sanitize(obs_dict["policy"]); w_img, f_img = readImgs()
stage = torch.zeros(N_ENVS, dtype=torch.long, device=DEVICE)        # STATE-based stage pointer
bonus_given = torch.zeros(N_ENVS, dtype=torch.bool, device=DEVICE)
global_step, n_iter = 0, 0
closs_ema = None; _sr_hits = 0; _last_hit_ep = -10**9

while global_step < args.total_timesteps:
    pre_stage = stage.clone()
    s21 = addProprioNoise(studentState(full_obs)) if args.dr else studentState(full_obs)
    with torch.no_grad():
        feat = student.feat(enc_w(w_img), enc_f(f_img), s21)
        action = torch.zeros(N_ENVS, act_dim, device=DEVICE)
        for s in range(NS):
            m = (pre_stage == s)
            if m.any():
                a_s, _ = student.getActionStage(feat[m], s)
                action[m] = a_s

    next_dict, rew, terminated, truncated, infos = env.step(action.detach())
    if args.dr:
        _drawWristAug(mask=(terminated | truncated))
    next_obs = sanitize(next_dict["policy"])
    ns21 = addProprioNoise(studentState(next_obs)) if args.dr else studentState(next_obs)
    nw_img, nf_img = readImgs()
    rew_t = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0).float()

    conds = [stageCond(0), stageCond(1), stageCond(2), stageCond(3)]
    # per-stage store (full reward + this-stage done bonus, once)
    for s in range(NS):
        m = (pre_stage == s)
        if m.any():
            r_s = rew_t[m] + args.bonus * (conds[s][m] & ~bonus_given[m]).float()
            bufs[s].add(w_img[m], f_img[m], s21[m], action[m].detach(), r_s, nw_img[m], nf_img[m], ns21[m])
    bonus_given |= ((conds[0] & (pre_stage == 0)) | (conds[1] & (pre_stage == 1)) |
                    (conds[2] & (pre_stage == 2)) | (conds[3] & (pre_stage == 3)))
    # advance the STATE stage pointer 0->1->2->3 (reset the bonus flag on entering a fresh stage)
    adv0 = (stage == 0) & conds[0]; stage = torch.where(adv0, torch.ones_like(stage), stage)
    adv1 = (stage == 1) & conds[1]; stage = torch.where(adv1, torch.full_like(stage, 2), stage)
    adv2 = (stage == 2) & conds[2]; stage = torch.where(adv2, torch.full_like(stage, 3), stage)
    bonus_given = torch.where(adv0 | adv1 | adv2, torch.zeros_like(bonus_given), bonus_given)

    done = (terminated | truncated)
    if done.any():
        stage[done] = 0; bonus_given[done] = False

    full_obs = next_obs; w_img, f_img = nw_img, nf_img
    global_step += N_ENVS; n_iter += 1

    # ── updates ──
    if global_step >= args.learning_starts:
        do_actor_phase = global_step >= args.critic_warmup
        for j in range(args.num_updates):
            do_actor = do_actor_phase and (j % args.policy_frequency == 0)
            for s in range(NS):
                if bufs[s].ready(args.batch_size):
                    closs = updateStage(s, do_actor)
                    closs_ema = closs if closs_ema is None else 0.98 * closs_ema + 0.02 * closs
                    softUpdate(targets[s], critics[s])
            softUpdate(enc_f_t, enc_f); softUpdate(enc_w_t, enc_w)
        for s in range(NS):
            alpha[s] = log_alpha[s].exp().detach()

    succ = MR._succ_total[0]; ep = MR._ep_total[0]
    rec = getattr(MR, "_succ_recent", []); sr = (sum(rec) / len(rec)) if rec else 0.0
    stop, why = False, ""
    if args.success_target and succ >= args.success_target:
        stop, why = True, f"{succ} successes (target {args.success_target})"
    hit = bool(args.success_rate_target) and len(rec) >= args.min_sr_episodes and sr >= args.success_rate_target
    if hit and (ep - _last_hit_ep) >= args.min_sr_episodes:
        _sr_hits += 1; _last_hit_ep = ep
        print(f"[SVRL] sr-hit {_sr_hits}/{args.min_sr_hits} (end2end sr={sr:.3f}, {ep} eps) t={global_step}", flush=True)
    if hit and _sr_hits >= args.min_sr_hits:
        stop, why = True, f"sr={sr:.3f} x{_sr_hits}"
    if stop:
        print(f"[DONE] staged visual-RL: {why} at t={global_step}; saving.", flush=True)
        _save(os.path.join(args.save_dir, "staged_visual_rl_final.pt")); break

    if n_iter % 5 == 0:
        warm = "critic-warmup" if global_step < args.critic_warmup else "fine-tune"
        try:
            dash.pushStats(step=global_step, episode=ep, current_reward=float(rew_t.mean()),
                            episode_reward=float(rew_t.mean()), cumulative_reward=0.0, best_reward=0.0,
                            success_count=succ, success_rate=sr,
                            reward_terms={"critic_loss": round(closs_ema or 0.0, 4), "phase": warm,
                                          "a0": round(float(alpha[0]), 3), "a3": round(float(alpha[3]), 3),
                                          "mean_rew": round(float(rew_t.mean()), 4)},
                            cumulative_terms={}, dopamine={})
            dash.setCameraImage("front", f_img[0].detach().cpu().numpy())
            dash.setCameraImage("wrist", w_img[0].detach().cpu().numpy())
        except Exception:
            pass
    if n_iter % 50 == 0:
        c = [int((stage == tS).sum()) for tS in range(NS)]
        phase = "warmup" if global_step < args.critic_warmup else "tune"
        print(f"[SVRL] t={global_step} {phase} closs={closs_ema if closs_ema else 0:.4f} succ={succ} sr={sr:.3f} "
              f"stagePtr@[r{c[0]},a{c[1]},g{c[2]},l{c[3]}] alpha=[{float(alpha[0]):.2f},{float(alpha[3]):.2f}]", flush=True)
    if n_iter % 200 == 0:
        try:
            _save(os.path.join(args.save_dir, f"staged_visual_rl_ckpt_{global_step}.pt"))
        except Exception:
            pass

print("[SVRL] loop finished.", flush=True)
try:
    env.close(); simulation_app.close()
except Exception:
    pass
