"""Headless, seeded evaluation of the shipped teacher / student checkpoints in the reBot LiftCube env.

Written for this repository (it did not exist in the original project). It builds the environment exactly the way
the training scripts do (configure_env + the same finger contact sensors + apply_motor_limits; for the student also
configure_visual without domain randomisation) and runs the policy deterministically (tanh of the actor mean). It does
not change any algorithm, reward or success definition.

Success of an episode = the environment's own success latch (maniskill_rewards.success_latch / count_successes), i.e.
is_fold_success (cube lifted >= 1 mm, grasped by both fingers, commanded arm within 0.2 rad of REST) at any control
step of the episode. This is the quantity the training scripts counted as "successes".

Modes
  single   one state-only C51 actor (43-D privileged state)                      --ckpts c51_final.pt
  staged3  the 3-stage joint teacher of commit ca032b5 (approach -> grasp -> place); the stage advances on
           the same done-conditions as that commit's train_c51_joint.py / infer_staged.py
                                                                                  --ckpts s0 s1 s2
  student  visual DAgger student (front + wrist RGB 128x128 + 21-D proprio), clean images (no DR)
                                                                                  --ckpts dagger_final.pt

Outputs (in --out-dir): <tag>_episodes.csv (one row per episode) and <tag>_summary.json.

Run (Isaac Sim 5.1 / Isaac Lab 2.3 venv with LeIsaac + the reBot assets; from the repository root):
  PYTHONPATH=src python scripts/eval/eval_policies.py --mode single --ckpts <ckpt> --tag teacher_v5g \
      --num-envs 64 --episodes 128 --seed 0 --out-dir results/eval
"""
import argparse
import csv
import hashlib
import json
import os
import time

import numpy as np
import torch

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT",
                      os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl")) + "/leisaac/assets")

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["single", "staged3", "student"], required=True)
ap.add_argument("--ckpts", nargs="+", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--num-envs", type=int, default=64)
ap.add_argument("--episodes", type=int, default=128, help="evaluate at least this many episodes")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--target-cube", default="cube")
ap.add_argument("--config", default="configs/reward_config.yaml")
ap.add_argument("--out-dir", default="results/eval")
args, _ = ap.parse_known_args()

import yaml  # noqa: E402

CFG = {}
if os.path.exists(args.config):
    with open(args.config) as f:
        CFG = yaml.safe_load(f) or {}
STUDENT = args.mode == "student"
if STUDENT:
    CFG["enable_cameras"] = True

from isaaclab.app import AppLauncher  # noqa: E402

simulation_app = AppLauncher({"headless": True, "enable_cameras": STUDENT}).app

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
import leisaac.tasks.lift_cube  # noqa: E402,F401
from leisaac.tasks.lift_cube import mdp  # noqa: E402

from rebot_sim2real.env.env_config import configure_env, apply_motor_limits  # noqa: E402
import rebot_sim2real.env.maniskill_rewards as MR  # noqa: E402
from rebot_sim2real.student.visual_nets import StateOnlyActor  # noqa: E402

DEVICE = "cuda:0"
N = args.num_envs
TARGET = args.target_cube
torch.manual_seed(args.seed)
np.random.seed(args.seed)

env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=N)
env_cfg = configure_env(env_cfg, CFG, mdp, RewTerm, DoneTerm, SceneEntityCfg, target_cube=TARGET)
_obj = getattr(env_cfg.scene, TARGET)
_cp = _obj.prim_path
env_cfg.scene.gripper_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_left",
                                                 update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
env_cfg.scene.jaw_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_right",
                                             update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
_obj.spawn.activate_contact_sensors = True
env_cfg.scene.robot.spawn.activate_contact_sensors = True
if STUDENT:
    from rebot_sim2real.env.visual_obs import configure_visual, downsample, student_state, WRIST_SIZE, FRONT_SIZE, STATE_DIM
    env_cfg = configure_visual(env_cfg, target_cube=TARGET, keep_privileged=True, dr=False, front_only=False)
else:  # state-only policies never use images (same as train_c51.py without --cameras)
    for cam in ["front", "side", "wrist"]:
        if hasattr(env_cfg.scene, cam):
            setattr(env_cfg.scene, cam, None)
    for attr in list(vars(env_cfg.observations.policy)):
        if attr in ("front", "wrist", "side") or "image" in attr.lower():
            setattr(env_cfg.observations.policy, attr, None)
env_cfg.seed = args.seed

env = ManagerBasedRLEnv(cfg=env_cfg)
apply_motor_limits(env, CFG)
ACT = env.action_manager.total_action_dim
STEPS_PER_EP = int(env.max_episode_length)


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def load(p):
    return torch.load(p, map_location=DEVICE)


# ── policies ──
if args.mode in ("single", "staged3"):
    actors = []
    for c in args.ckpts:
        a = StateOnlyActor(43, ACT).to(DEVICE)
        ck = load(c)
        a.load_state_dict(ck["actor"] if "actor" in ck else ck)
        a.eval()
        actors.append(a)
    if args.mode == "staged3" and len(actors) != 3:
        raise SystemExit("staged3 needs exactly 3 checkpoints (s0 s1 s2)")
else:
    from rebot_sim2real.student.visual_nets import CNNEncoder, Actor
    ck = load(args.ckpts[0])
    if ck.get("front_only", False):
        raise SystemExit("front-only students are not supported by this evaluator")
    enc_w = CNNEncoder(WRIST_SIZE).to(DEVICE)
    enc_f = CNNEncoder(FRONT_SIZE).to(DEVICE)
    s_actor = Actor(STATE_DIM, ACT, use_wrist=True).to(DEVICE)
    enc_w.load_state_dict(ck["enc_w"])
    enc_f.load_state_dict(ck["enc_f"])
    s_actor.load_state_dict(ck["actor"])
    for m in (enc_w, enc_f, s_actor):
        m.eval()

_ee = SceneEntityCfg("ee_frame")
_oc = SceneEntityCfg(TARGET)


def tcp_dist():
    return torch.norm(env.scene[TARGET].data.root_pos_w - MR._tcp(env, _ee), dim=1)


def stage_done_v5(s):  # ca032b5 infer_staged.py: approach=align-hold latch, grasp=_is_grasped, place=is_fold_success
    if s == 0:
        return MR._align_held(env, tcp_dist())
    if s == 1:
        return MR._is_grasped(env, _oc)
    return MR.is_fold_success(env, _oc)


def sanitize(t):
    return torch.nan_to_num(t, nan=0.0, posinf=100.0, neginf=-100.0)


obs_dict, _ = env.reset()
if STUDENT:  # camera buffers need a few rendered frames; then start from a fresh reset
    for _ in range(3):
        obs_dict, *_ = env.step(torch.zeros((N, ACT), device=DEVICE))
    obs_dict, _ = env.reset()
obs = sanitize(obs_dict["policy"])


def init_pose():
    o = env.scene[TARGET].data.root_pos_w - env.scene.env_origins
    q = env.scene[TARGET].data.root_quat_w  # wxyz
    yaw = 2.0 * torch.atan2(q[:, 3], q[:, 0])
    return o[:, 0].tolist(), o[:, 1].tolist(), yaw.tolist()


ix, iy, iyaw = init_pose()
stage = torch.zeros(N, dtype=torch.long, device=DEVICE)
staged_succ = torch.zeros(N, dtype=torch.bool, device=DEVICE)
grasp_ever = torch.zeros(N, dtype=torch.bool, device=DEVICE)
max_lift = torch.zeros(N, device=DEVICE)
min_dist = torch.full((N,), 1e9, device=DEVICE)
first_succ = torch.full((N,), -1, dtype=torch.long, device=DEVICE)
step_in_ep = torch.zeros(N, dtype=torch.long, device=DEVICE)
rows = []
t0 = time.time()
max_steps = (args.episodes // N + 2) * (STEPS_PER_EP + 2)
for _ in range(max_steps):
    with torch.no_grad():
        if args.mode == "single":
            action = actors[0].act_deterministic(obs)
        elif args.mode == "staged3":
            action = torch.zeros(N, ACT, device=DEVICE)
            for s in range(3):
                m = stage == s
                if m.any():
                    action[m] = actors[s].act_deterministic(obs[m])
        else:
            f_img = downsample(env.scene["front"].data.output["rgb"], FRONT_SIZE)
            w_img = downsample(env.scene["wrist"].data.output["rgb"], WRIST_SIZE)
            action = s_actor.act_deterministic(enc_w(w_img), enc_f(f_img), student_state(obs))
    obs_dict, _, term, trunc, _ = env.step(action)
    obs = sanitize(obs_dict["policy"])
    dones = term | trunc
    live = ~dones  # on a done step the state is already the next episode's reset state
    step_in_ep += 1
    if live.any():
        d = tcp_dist()
        min_dist = torch.where(live, torch.minimum(min_dist, d), min_dist)
        g = MR._is_grasped(env, _oc)
        grasp_ever |= g & live
        z = env.scene[TARGET].data.root_pos_w[:, 2] - MR.CUBE_REST_Z
        max_lift = torch.where(live, torch.maximum(max_lift, z), max_lift)
        succ_now = MR.is_fold_success(env, _oc) & live
        first_succ = torch.where((first_succ < 0) & succ_now, step_in_ep, first_succ)
        if args.mode == "staged3":
            for s in range(2):
                adv = (stage == s) & stage_done_v5(s) & live
                stage = torch.where(adv, stage + 1, stage)
            staged_succ |= (stage == 2) & stage_done_v5(2) & live
    if dones.any():
        idx = torch.nonzero(dones).flatten().tolist()
        latch = MR._last_ep_succ[0]
        for i in idx:
            rows.append({
                "episode": len(rows), "env": i, "seed": args.seed,
                "cube_x": round(ix[i], 4), "cube_y": round(iy[i], 4), "cube_yaw": round(iyaw[i], 4),
                "success": int(bool(latch[i])),
                "first_success_step": int(first_succ[i]),
                "grasped_ever": int(bool(grasp_ever[i])),
                "max_lift_m": round(float(max_lift[i]), 4),
                "min_tcp_cube_dist_m": round(float(min_dist[i]), 4),
                "final_stage": int(stage[i]) if args.mode == "staged3" else "",
                "staged_end_to_end": int(bool(staged_succ[i])) if args.mode == "staged3" else "",
                "steps": int(step_in_ep[i]),
            })
        nx, ny, nyaw = init_pose()
        for i in idx:
            ix[i], iy[i], iyaw[i] = nx[i], ny[i], nyaw[i]
        dm = dones
        stage[dm] = 0
        staged_succ[dm] = False
        grasp_ever[dm] = False
        max_lift[dm] = 0.0
        min_dist[dm] = 1e9
        first_succ[dm] = -1
        step_in_ep[dm] = 0
    if len(rows) >= args.episodes:
        break

rows = rows[: args.episodes] if len(rows) >= args.episodes else rows
os.makedirs(args.out_dir, exist_ok=True)
csv_p = os.path.join(args.out_dir, f"{args.tag}_episodes.csv")
with open(csv_p, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
n = len(rows)
k = sum(r["success"] for r in rows)
p = k / n
z = 1.96  # Wilson 95 % interval
den = 1 + z * z / n
ctr = (p + z * z / (2 * n)) / den
half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
summary = {
    "tag": args.tag, "mode": args.mode, "seed": args.seed, "num_envs": N, "episodes": n,
    "successes": k, "success_rate": round(p, 4), "wilson95": [round(ctr - half, 4), round(ctr + half, 4)],
    "grasped_ever_rate": round(sum(r["grasped_ever"] for r in rows) / n, 4),
    "steps_per_episode": STEPS_PER_EP, "deterministic": True, "domain_randomization": False,
    "checkpoints": [{"path": os.path.basename(c), "md5": md5(c)} for c in args.ckpts],
    "wall_s": round(time.time() - t0, 1),
}
if args.mode == "staged3":
    summary["staged_end_to_end_rate"] = round(sum(r["staged_end_to_end"] for r in rows) / n, 4)
with open(os.path.join(args.out_dir, f"{args.tag}_summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
print("[eval] " + json.dumps(summary), flush=True)
try:
    env.close()
    simulation_app.close()
except Exception:
    pass
