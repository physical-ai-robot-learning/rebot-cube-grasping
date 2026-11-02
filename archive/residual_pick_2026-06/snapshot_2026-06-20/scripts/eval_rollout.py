"""Is the learned pick_rlpd3 'success' GENUINE or a reward hack? Load the checkpoint, run a
deterministic rollout (1 env, center cube), and log the REAL per-step state (cube_z, tcp_z, dist3,
is_grasped, table_pen) BEFORE the env auto-resets. Reports whether the cube truly reaches the 12cm
success height while the gripper holds it ABOVE the table. Writes /tmp/eval_rollout.txt.
"""
import os, sys, argparse, glob, re
ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/train.yaml")
ap.add_argument("--exp", default="pick_rlpd3"); ap.add_argument("--episodes", type=int, default=3)
args = ap.parse_args()

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from rebot_sim2real.rl.config import load_train_config
from rebot_sim2real.rl.sac import SAC

tcfg = load_train_config(args.config); tcfg.env.num_envs = 1; tcfg.env.pos_rand_xy = 0.0
tcfg.env.dr_enable = False
dev = torch.device(tcfg.env.device)
from rebot_sim2real.envs.pick_env import RebotPickEnv
env = RebotPickEnv(tcfg.env, tcfg.reward)
agent = SAC(env.obs_dim, env.act_dim, tcfg.sac, dev)
ckpt = os.path.join(tcfg.ckpt_dir, f"{args.exp}_latest.pt")
agent.load_state_dict(torch.load(ckpt, map_location=dev))
succ_z = tcfg.env.table_z + tcfg.env.success_lift_height

lines = [f"ckpt={ckpt}  success needs cube_z>={succ_z:.3f} & near(<{tcfg.env.grasp_radius}) & lifted"]
obs = env.reset()
ep = 0; step = 0; max_cube_z = 0.0; succ_logged = 0
peak = {"cube_z": 0.0, "tcp_z": 0.0, "dist3": 9.0, "grasped": False}
while ep < args.episodes and step < tcfg.env.episode_len * args.episodes + 50:
    a = agent.act(obs, deterministic=True)
    obs, _, done, info = env.step(a)
    _, s, _ = env._compute_state(env.prev_action)     # post-step (post-reset if done) — use info for pre-reset
    cz = float(s["cube_pos"][0, 2]); tz = float(s["tcp_pos"][0, 2])
    d3 = float(torch.linalg.norm(s["cube_pos"][0] - s["tcp_pos"][0]))
    gr = bool(s["is_grasped"][0]); tpen = float(s["table_pen"][0])
    fired = float(info["success"][0]) > 0.5
    if cz > peak["cube_z"] and not done:
        peak = {"cube_z": cz, "tcp_z": tz, "dist3": d3, "grasped": gr}
    if fired and succ_logged < 8:
        lines.append(f"  SUCCESS@step{step}: cube_z={cz:.3f} tcp_z={tz:.3f} dist3={d3:.3f} grasped={gr} table_pen={tpen:.3f}")
        succ_logged += 1
    if int(info["num_done"]) > 0:
        es = float(info["ep_success"][0])
        lines.append(f"  [ep {ep} end @step{step}] ep_success={es:.0f}  peak_cube_z={peak['cube_z']:.3f} "
                     f"@tcp_z={peak['tcp_z']:.3f} dist3={peak['dist3']:.3f} grasped={peak['grasped']}")
        peak = {"cube_z": 0.0, "tcp_z": 0.0, "dist3": 9.0, "grasped": False}
        ep += 1
    step += 1

lines.append(f"\nVERDICT inputs: success fires only if a step shows cube_z>={succ_z:.3f} with tcp within "
             f"{tcfg.env.grasp_radius} of it (=> tcp_z high, above table {tcfg.env.table_z}).")
out = "\n".join(lines)
with open("/tmp/eval_rollout.txt", "w") as f:
    f.write(out + "\n"); f.flush(); os.fsync(f.fileno())
print(out, flush=True)
app.close()
