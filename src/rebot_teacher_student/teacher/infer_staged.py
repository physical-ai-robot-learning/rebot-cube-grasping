"""STAGED inference / eval for the 3-stage teacher: classify the current stage from STATE each step,

Stage: June teacher (staged inference).
Was src/rebot_sim2real/teacher/infer_staged.py (moved and renamed to the OA style on 2026-11-05; docs/refactor_2026-11.md).
Run: python -m rebot_teacher_student.teacher.infer_staged
Needs Isaac Sim 5.1 / Isaac Lab 2.3 + the LeIsaac reBot fork; not executable in CI (compile-checked only).

run the MATCHING stage actor, advance on that stage's done-condition (approach=align-hold latch,
grasp=_is_grasped, place=is_fold_success). Reports per-env stage progression + end-to-end success.

Works with however many stage checkpoints you pass (1, 2, or 3) — runs the chain up to the last given.
  PYTHONPATH=src python -m rebot_teacher_student.teacher.infer_staged \
      --ckpts rl_output/rl_c51_s0/c51_final.pt rl_output/rl_c51_s1/c51_final.pt rl_output/rl_c51_s2/c51_final.pt
"""
import argparse, os
import numpy as np
import torch
import torch.nn as nn
from rebot_grasp_common.paths import LEISAAC_ASSETS

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT", LEISAAC_ASSETS)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="+", required=True, help="stage checkpoints in order: s0 [s1 [s2]]")
ap.add_argument("--num-envs", type=int, default=64)
ap.add_argument("--episodes", type=int, default=4)
ap.add_argument("--target-cube", type=str, default="cube")
args, _ = ap.parse_known_args()

from isaaclab.app import AppLauncher
simulation_app = AppLauncher({"headless": True, "enable_cameras": True}).app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import SceneEntityCfg, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
import leisaac.tasks.lift_cube  # noqa: F401
from leisaac.tasks.lift_cube import mdp
from rebot_teacher_student.env.env_config import configureEnv, applyMotorLimits
import rebot_teacher_student.env.rewards as MR

DEVICE = "cuda:0"; N = args.num_envs; TARGET = args.target_cube; NS = len(args.ckpts)
env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=N)
env_cfg = configureEnv(env_cfg, {}, mdp, RewTerm, DoneTerm, SceneEntityCfg, target_cube=TARGET)
_obj = getattr(env_cfg.scene, TARGET); _cp = _obj.prim_path
env_cfg.scene.gripper_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_left",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
env_cfg.scene.jaw_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_right",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
_obj.spawn.activate_contact_sensors = True
env_cfg.scene.robot.spawn.activate_contact_sensors = True
env = ManagerBasedRLEnv(cfg=env_cfg)
applyMotorLimits(env, {})


class Actor(nn.Module):
    def __init__(self, n_state=43, n_act=7, hidden=256):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(n_state, 256), nn.LayerNorm(256), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(256, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, n_act); self.logstd = nn.Linear(hidden, n_act)
    @torch.no_grad()
    def act(self, o): return torch.tanh(self.mean(self.fc(self.proj(o))))


actors = []
for i, c in enumerate(args.ckpts):
    a = Actor().to(DEVICE); ck = torch.load(c, map_location=DEVICE)
    a.load_state_dict(ck["actor"] if "actor" in ck else ck); a.eval(); actors.append(a)
    print(f"[stg-inf] stage {i} actor <- {c}", flush=True)

obs_mgr = env.observation_manager
_ee = SceneEntityCfg("ee_frame"); _oc = SceneEntityCfg(TARGET)
def _dist():
    return torch.norm(env.scene[TARGET].data.root_pos_w - MR._tcp(env, _ee), dim=1)
def stageDone(s):                     # done condition that ADVANCES out of stage s
    # 4-stage chain: 0 reaching -> 1 align -> 2 grasp -> 3 lift (matches train_c51_joint)
    if s == 0:                         # reaching: reached the cube vicinity
        return _dist() < (MR.ALIGN_DIST + 0.05)
    if s == 1:                         # align: open jaws aligned-and-held (latch)
        return MR._alignHeld(env, _dist())
    if s == 2:                         # grasp: contact-grasp after align
        return MR._isGrasped(env, _oc)
    return MR.isFoldSuccess(env, _oc)   # lift: lifted + returned = success


obs = env.reset()[0]["policy"]
stage = torch.zeros(N, dtype=torch.long, device=DEVICE)
ep_succ = torch.zeros(N, dtype=torch.bool, device=DEVICE)        # end-to-end success this episode
ever_stage = torch.zeros(N, NS, dtype=torch.bool, device=DEVICE)  # reached each stage this episode
done_eps = 0; succ_tally = 0; reached_tally = np.zeros(NS)
for step in range(args.episodes * 110):
    action = torch.zeros(N, 7, device=DEVICE)
    for s in range(NS):
        m = (stage == s)
        if m.any():
            action[m] = actors[s].act(obs[m])
    obs_dict, _, term, trunc, _ = env.step(action)
    obs = obs_dict["policy"]
    for s in range(NS):
        ever_stage[:, s] |= (stage == s)
    # advance stages (last stage doesn't advance; its done = end-to-end success)
    for s in range(NS - 1):
        adv = (stage == s) & stageDone(s)
        stage = torch.where(adv, stage + 1, stage)
    # end-to-end success = the LAST stage's done reached (place_done if 3 stages, else that stage's done)
    ep_succ |= (stage == NS - 1) & stageDone(NS - 1)
    dones = term | trunc
    if dones.any():
        idx = torch.nonzero(dones).flatten()
        done_eps += int(idx.numel()); succ_tally += int(ep_succ[idx].sum())
        for s in range(NS):
            reached_tally[s] += int(ever_stage[idx, s].sum())
        stage[idx] = 0; ep_succ[idx] = False; ever_stage[idx] = False

print(f"\n[stg-inf] over {done_eps} episodes ({NS}-stage chain):", flush=True)
for s in range(NS):
    print(f"   reached stage {s}: {100*reached_tally[s]/max(done_eps,1):5.1f}%", flush=True)
print(f"   END-TO-END success: {100*succ_tally/max(done_eps,1):5.1f}%", flush=True)
print("[stg-inf] DONE", flush=True)
try:
    env.close(); simulation_app.close()
except Exception:
    pass
