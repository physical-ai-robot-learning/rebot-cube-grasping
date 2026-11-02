"""VERIFY the learned pick_rlpd5 success is genuine (not fling/knock). Load the checkpoint, run the
DETERMINISTIC policy on 16 envs, and at each env's FIRST success step capture the real mechanics:
cube_z (lifted?), dist3 = |cube - tcp| (cube AT the gripper, not flung away?), finger gap (closed ON
the cube?), is_grasped, table_pen. A genuine pick: cube_z>=0.11, dist3 small, gap~=cube width, no pen.
Writes /tmp/eval_verify.txt.
"""
import os, sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from rebot_sim2real.rl.config import load_train_config
from rebot_sim2real.rl.sac import SAC

N = 16
tcfg = load_train_config("configs/train.yaml"); tcfg.env.num_envs = N; tcfg.env.dr_enable = False
dev = torch.device(tcfg.env.device)
from rebot_sim2real.envs.pick_env import RebotPickEnv
env = RebotPickEnv(tcfg.env, tcfg.reward)
agent = SAC(env.obs_dim, env.act_dim, tcfg.sac, dev)
CKPT = sys.argv[1] if len(sys.argv) > 1 else "pick_rlpd7_latest.pt"
agent.load_state_dict(torch.load(os.path.join(tcfg.ckpt_dir, CKPT), map_location=dev))
succ_z = tcfg.env.table_z + tcfg.env.success_lift_height

# eval at the FULL trained difficulty: widest spawn + the reset jitter (deployment-realistic, not center-only)
env.pos_rand = float(getattr(tcfg.env, "pos_rand_max", 0.0)) or float(tcfg.env.pos_rand_xy)
print(f"[eval] pos_rand={env.pos_rand} rest_rand={tcfg.env.rest_rand}", flush=True)
obs = env.reset()
# track, per env, the MAX cube_z reached while GENUINELY GRIPPED (is_grasped & cube at the gripper),
# measured ONLY on non-reset steps (so the success->auto-reset frame can't corrupt it). dist3/gap at
# that peak show the cube is really in the jaw, not flung.
peak_held = torch.full((N,), -1.0, device=dev)
peak_dist3 = torch.zeros(N, device=dev); peak_gap = torch.zeros(N, device=dev); peak_tilt = torch.zeros(N, device=dev)
ever_succ = torch.zeros(N, dtype=torch.bool, device=dev)
max_pen = torch.zeros(N, device=dev)
rest_cube_z = tcfg.env.table_z + tcfg.env.cube_size / 2.0
min_cubez_near = torch.full((N,), 9.0, device=dev)   # lowest cube_z while the gripper is at it (push-DOWN check)
for t in range(tcfg.env.episode_len):
    a = agent.act(obs, deterministic=True)
    obs, _, done, info = env.step(a)
    _, s, _ = env._compute_state(env.prev_action)
    ever_succ |= info["success"].bool()
    cube = s["cube_pos"]; tcp = s["tcp_pos"]
    d3 = torch.linalg.norm(cube - tcp, dim=-1)
    gap = 2.0 * env.robots.get_joint_positions()[:, env.grip_idx].mean(dim=-1)
    nd = ~done.bool()
    near_gr = (d3 < 0.06) & nd
    min_cubez_near = torch.where(near_gr & (cube[:, 2] < min_cubez_near), cube[:, 2], min_cubez_near)
    held = s["is_grasped"].bool() & (d3 < 0.06) & nd   # genuinely gripped, this is NOT a reset frame
    max_pen = torch.where(nd, torch.maximum(max_pen, s["table_pen"]), max_pen)
    better = held & (cube[:, 2] > peak_held)
    peak_held = torch.where(better, cube[:, 2], peak_held)
    peak_dist3 = torch.where(better, d3, peak_dist3)
    peak_gap = torch.where(better, gap, peak_gap)
    peak_tilt = torch.where(better, s["tilt_deg"], peak_tilt)

genuine = (peak_held >= succ_z)
m = peak_held >= 0
def avg(x, msk): return float(x[msk].mean()) if msk.any() else float('nan')
pushed = (min_cubez_near < rest_cube_z - 0.005) & (min_cubez_near < 8.0)   # cube pressed DOWN below rest while near
lines = [f"ckpt={CKPT}, {N} envs deterministic. success_z={succ_z:.3f} rest_cube_z={rest_cube_z:.3f} tilt_gate={tcfg.env.tilt_gate_deg} table_center={tcfg.env.table_center}",
         f"episode success (info): {int(ever_succ.sum())}/{N}",
         f"GENUINE held-lift (cube reached >= {succ_z:.3f} WHILE gripped & at the jaw): {int(genuine.sum())}/{N}",
         f"at the peak held-lift (mean over envs that gripped):",
         f"  peak cube_z while gripped = {avg(peak_held, m):.3f}  (>= {succ_z:.3f} => genuinely lifted while held)",
         f"  dist3 there               = {avg(peak_dist3, m):.3f}  (small => cube IN the jaw, not flung)",
         f"  gap there                 = {avg(peak_gap, m):.3f}  (~{tcfg.env.grip_grasp_gap} => fingers closed ON the cube)",
         f"  tilt there                = {avg(peak_tilt, m):.1f} deg  (< gate {tcfg.env.tilt_gate_deg} => not an edge grasp)",
         f"  max table_pen (any step)  = {avg(max_pen, torch.ones(N, dtype=torch.bool, device=dev)):.3f}  (0 => never below the table)",
         f"PUSH-DOWN (cube pressed below rest while gripper near): {int(pushed.sum())}/{N}  min cube_z near = {avg(min_cubez_near, min_cubez_near<8.0):.3f}",
         f"per-env peak held cube_z: {[round(float(x),3) for x in peak_held.tolist()]}"]
out = "\n".join(lines)
with open("/tmp/eval_verify.txt", "w") as f:
    f.write(out + "\n"); f.flush(); os.fsync(f.fileno())
print(out, flush=True)
app.close()
