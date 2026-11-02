"""SIM2REAL VERIFY V1: how robust is the trained state policy to PERCEPTION error?

On a real robot, cube_pos comes from perception (color-seg + homography/depth), not the sim's
ground truth -- it carries ~1-2cm error. This sweeps the observation position noise (the obs-only
Gaussian the env injects on cube_pos/tcp_pos) and measures GENUINE held-lift at each level, in ONE
Isaac session, at the FULL trained difficulty (pos_rand=pos_rand_max + the reset jitter). If
genuine-pick holds out to ~1-2cm, the policy tolerates real perception error; if it collapses,
either improve perception (depth cam / calibration) or retrain with larger obs-noise DR.

    OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/verify_perception_noise.py [ckpt]
Writes /tmp/verify_perception_noise.txt.
"""
import os, sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from rebot_sim2real.rl.config import load_train_config
from rebot_sim2real.rl.sac import SAC

N = 16
CKPT = sys.argv[1] if len(sys.argv) > 1 else "pick_rlpd10_latest.pt"
NOISES = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]   # std (m) of obs position noise = emulated perception error
tcfg = load_train_config("configs/train.yaml")
tcfg.env.num_envs = N
dev = torch.device(tcfg.env.device)
from rebot_sim2real.envs.pick_env import RebotPickEnv
env = RebotPickEnv(tcfg.env, tcfg.reward)
env.pos_rand = float(getattr(tcfg.env, "pos_rand_max", 0.0)) or float(tcfg.env.pos_rand_xy)
agent = SAC(env.obs_dim, env.act_dim, tcfg.sac, dev)
agent.load_state_dict(torch.load(os.path.join(tcfg.ckpt_dir, CKPT), map_location=dev))
succ_z = tcfg.env.table_z + tcfg.env.success_lift_height
# isolate perception noise: keep mass nominal, only the obs position noise varies
env.cfg.dr_enable = True
env.cfg.dr_mass_min = env.cfg.dr_mass_max = float(tcfg.env.cube_mass)


def run_at_noise(noise):
    env.cfg.dr_obs_pos_noise = float(noise)
    obs = env.reset()
    peak_held = torch.full((N,), -1.0, device=dev)
    ever_succ = torch.zeros(N, dtype=torch.bool, device=dev)
    for _ in range(tcfg.env.episode_len):
        a = agent.act(obs, deterministic=True)
        obs, _, done, info = env.step(a)
        _, s, _ = env._compute_state(env.prev_action)
        ever_succ |= info["success"].bool()
        cube = s["cube_pos"]; tcp = s["tcp_pos"]
        d3 = torch.linalg.norm(cube - tcp, dim=-1)
        held = s["is_grasped"].bool() & (d3 < 0.06) & (~done.bool())
        better = held & (cube[:, 2] > peak_held)
        peak_held = torch.where(better, cube[:, 2], peak_held)
    genuine = int((peak_held >= succ_z).sum())
    m = peak_held >= 0
    avg_peak = float(peak_held[m].mean()) if m.any() else float("nan")
    return genuine, int(ever_succ.sum()), avg_peak


rows = []
for nz in NOISES:
    g, info_s, peak = run_at_noise(nz)
    rows.append((nz, g, info_s, peak))
    print(f"[verify] noise={nz*1000:.0f}mm  genuine={g}/{N}  info_succ={info_s}/{N}  avg_peak_z={peak:.3f}", flush=True)

lines = [f"PERCEPTION-NOISE ROBUSTNESS  ckpt={CKPT}  N={N}  pos_rand={env.pos_rand}  rest_rand={tcfg.env.rest_rand}  succ_z={succ_z:.3f}",
         "obs cube/tcp position noise (= emulated perception error) vs GENUINE held-lift:",
         f"{'noise(mm)':>10} {'genuine':>10} {'info_succ':>10} {'avg_peak_z':>11}"]
for nz, g, info_s, peak in rows:
    lines.append(f"{nz*1000:>10.0f} {f'{g}/{N}':>10} {f'{info_s}/{N}':>10} {peak:>11.3f}")
lines.append("")
lines.append("read: genuine holding to ~10-20mm => policy tolerates real perception error; collapse => improve perception or add obs-noise DR + retrain.")
out = "\n".join(lines)
with open("/tmp/verify_perception_noise.txt", "w") as f:
    f.write(out + "\n"); f.flush(); os.fsync(f.fileno())
print(out, flush=True)
app.close()
