"""Auto-generate RELIABLE pick demos (joint-space, env-spawned near-center cube).

Uses the PROVEN setup: env spawns the cube near the table center, the arm rests at the grasp
config ([0,-2.2,-0.8]), then settle->close->lift (the same motion that lifts a center cube to
~13cm). Commits only successful episodes' (obs,act,rew,next_obs,done) for RLPD symmetric sampling.
(A driven approach->descend was tried but the open gripper collides with the cube before reaching
this side-grasp pose -> see scripts/probe_descent.py; teleport-to-grasp is the reliable demo.)

The sim gripper's reliable grasp region is narrow (~+/-2cm), so demos are near-center; train with
a matching small spawn radius (curriculum stage 1), then widen. Reach is learned by SAC's dense
reward; the demos supply the grasp+lift behavior single-SAC could not discover.

    OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/collect_demos.py \
        --num-envs 64 --rounds 30 --spawn-rad 0.02 --out demos/pick_demos.pt
"""
from __future__ import annotations

import argparse, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rebot_sim2real.grasp_traj import (
    APPROACH as APPROACH_Q, GRASP as GRASP_Q, LIFT as LIFT_Q,
    APPROACH_STEPS, DESCEND_STEPS, CLOSE_STEPS, LIFT_STEPS,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--spawn-rad", type=float, default=0.02, help="cube spawn half-range around center (m)")
    ap.add_argument("--out", default="demos/pick_demos.pt")
    ap.add_argument("--approach", type=int, default=APPROACH_STEPS)   # settle at the vertical hover pose
    ap.add_argument("--untilt", type=int, default=0)      # DEPRECATED (vertical APPROACH needs no untilt)
    ap.add_argument("--descend", type=int, default=DESCEND_STEPS)     # open fingers descend straight onto the cube
    ap.add_argument("--close", type=int, default=CLOSE_STEPS)
    ap.add_argument("--lift", type=int, default=LIFT_STEPS)
    return ap.parse_args()


args = parse_args()
from rebot_sim2real.rl.config import load_train_config   # pure Python -> load before the app
_cfg0 = load_train_config(args.config)
from isaacsim import SimulationApp
app = SimulationApp({"headless": True,
                     "enable_cameras": bool(_cfg0.env.obs_wrist or _cfg0.env.obs_top_uv)})   # wrist/top render
import torch
from rebot_sim2real.rl.demo import save_demos


def main():
    cfg = load_train_config(args.config)
    cfg.env.num_envs = args.num_envs
    cfg.env.pos_rand_xy = args.spawn_rad          # near-center -> within the reliable grasp region
    cfg.env.dr_enable = False                     # CLEAN scripted demos: execution DR (action latency/
    #   noise, gain jitter) corrupts the open-loop expert -> few successful grasps. Online training keeps
    #   full DR; the demos just need to be high-quality successful trajectories (obs_torque tail stays).
    device = torch.device(cfg.env.device)
    from rebot_sim2real.envs.pick_env import RebotPickEnv
    env = RebotPickEnv(cfg.env, cfg.reward)
    N = env.num_envs
    arm_idx = env.arm_idx
    # ~28-deg top-down pick (canonical poses in rebot_sim2real.grasp_traj). The cube spawns at the
    # 28-deg JAW landing point.
    #   APPROACH = hover ~6.8cm above the cube (~27 deg, = rest_arm)
    #   GRASP    = tcp AT the cube (~25 deg);  LIFT = +13cm while held (~30 deg, < the 35-deg gate)
    APPROACH = torch.tensor([APPROACH_Q], device=device).expand(N, 6)
    UNTILT = APPROACH                                    # deprecated stage (untilt steps default 0)
    GRASP = torch.tensor([GRASP_Q], device=device).expand(N, 6)
    LIFT = torch.tensor([LIFT_Q], device=device).expand(N, 6)
    GO = cfg.env.grip_open_qpos
    b1, b2, b3, b4 = args.approach, args.approach + args.untilt, \
        args.approach + args.untilt + args.descend, args.approach + args.untilt + args.descend + args.close

    def expert_action(t):
        cur = env.robots.get_joint_positions()[:, arm_idx]
        if t < b1:      tgt, closing = APPROACH, False
        elif t < b2:    tgt, closing = UNTILT, False
        elif t < b3:    tgt, closing = GRASP, False
        elif t < b4:    tgt, closing = GRASP, True
        else:           tgt, closing = LIFT, True
        a_arm = ((tgt - cur) / cfg.env.arm_step).clamp(-1.0, 1.0)
        a_grip = torch.ones(N, 1, device=device) if closing else -torch.ones(N, 1, device=device)
        return torch.cat([a_arm, a_grip], dim=-1)

    buf = {k: [] for k in ("obs", "act", "rew", "next_obs", "done")}
    n_succ = n_ep = 0
    fails = {"table_hit": 0, "y_back_hit": 0, "fallen": 0, "other": 0}   # DIAG: why non-success demos fail
    for _ in range(args.rounds):
        env.reset()
        # start each demo at the lifted APPROACH pose (no cube overlap -> no fling)
        q = env.robots.get_joint_positions().clone(); q[:, arm_idx] = APPROACH; q[:, env.grip_idx] = GO
        env.robots.set_joint_positions(q); env.arm_target[:] = APPROACH; env.grip_target[:] = GO
        env.robots.set_joint_position_targets(env._full_targets())
        for _ in range(8):
            env.world.step(render=False)
        obs = env.get_obs()
        traj = [[] for _ in range(N)]
        alive = torch.ones(N, dtype=torch.bool, device=device)
        # capture success WHEN it fires (per-step): success -> done -> env auto-resets, so reading
        # ep_success at the END misses it (it's cleared by the post-reset episode).
        round_success = torch.zeros(N, dtype=torch.bool, device=device)
        ever_t = torch.zeros(N, dtype=torch.bool, device=device)   # DIAG: termination reasons seen this round
        ever_y = torch.zeros(N, dtype=torch.bool, device=device)
        ever_f = torch.zeros(N, dtype=torch.bool, device=device)
        for t in range(b4 + args.lift):
            a = expert_action(t)
            next_obs, rew, done, info = env.step(a)
            round_success |= info["success"].bool()
            ever_t |= info["table_hit"].bool(); ever_y |= info["y_back_hit"].bool(); ever_f |= info["fallen"].bool()
            for i in range(N):
                if alive[i]:
                    traj[i].append((obs[i].clone(), a[i].clone(), rew[i].clone(), next_obs[i].clone(), done[i].clone()))
            obs = next_obs
            alive &= ~done.bool()
        for i in range(N):
            n_ep += 1
            if bool(round_success[i]) and traj[i]:
                n_succ += 1
                for (o, ac, r, no, d) in traj[i]:
                    buf["obs"].append(o); buf["act"].append(ac); buf["rew"].append(r)
                    buf["next_obs"].append(no); buf["done"].append(d)
            elif bool(ever_t[i]):   fails["table_hit"] += 1
            elif bool(ever_y[i]):   fails["y_back_hit"] += 1
            elif bool(ever_f[i]):   fails["fallen"] += 1
            else:                   fails["other"] += 1   # never terminated -> grasp/lift just didn't form
    sr = n_succ / max(1, n_ep)
    print(f"[demos] expert success rate = {sr:.2%} ({n_succ}/{n_ep}), {len(buf['obs'])} transitions", flush=True)
    print(f"[demos] FAILURE breakdown (of {n_ep-n_succ} fails): {fails}", flush=True)
    if buf["obs"]:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        save_demos(args.out, torch.stack(buf["obs"]), torch.stack(buf["act"]), torch.stack(buf["rew"]),
                   torch.stack(buf["next_obs"]), torch.stack(buf["done"]))
        print(f"[demos] saved -> {args.out}", flush=True)
    with open("/tmp/collect_demos_result.txt", "w") as f:
        f.write(f"success_rate={sr:.4f} n_succ={n_succ} n_ep={n_ep} transitions={len(buf['obs'])} "
                f"out={args.out if buf['obs'] else 'NONE'}\n")
        f.flush(); os.fsync(f.fileno())
    app.close()


if __name__ == "__main__":
    main()
