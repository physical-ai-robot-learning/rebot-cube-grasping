"""Validate the obs_wrist env path before the long retrain: build the env (per-env wrist cams + dome
+ render), step it, confirm obs_dim, that the wrist detection populates (visible>0), and measure fps.
    OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/wrist_env_check.py
"""
import os, sys, time, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/train.yaml")
ap.add_argument("--steps", type=int, default=20); args = ap.parse_args()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rebot_sim2real.rl.config import load_train_config
tcfg = load_train_config(args.config)
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "enable_cameras": bool(tcfg.env.obs_wrist)})
import numpy as np, torch
from rebot_sim2real.envs.pick_env import RebotPickEnv


def main():
    env = RebotPickEnv(tcfg.env, tcfg.reward)
    obs = env.reset()
    line1 = f"obs.shape={tuple(obs.shape)} env.obs_dim={env.obs_dim} num_envs={env.num_envs} obs_wrist={tcfg.env.obs_wrist}"
    print("[chk]", line1, flush=True)
    t0 = time.time()
    for k in range(args.steps):
        obs, r, d, info = env.step(torch.zeros(env.num_envs, env.act_dim, device=env.device))
    dt = time.time() - t0
    sps = args.steps / dt
    # DIAG: dump one wrist frame to see if it's dark / red detectable
    import imageio.v2 as imageio
    from rebot_sim2real import perception as P
    cam0 = env.wrist_cams[0]; rgba = cam0.get_rgba()
    if rgba is not None and getattr(rgba, "size", 0) > 0:
        fr = np.asarray(rgba)[..., :3].astype("uint8")
        os.makedirs("runs", exist_ok=True); imageio.imwrite("runs/wrist_env_frame.png", fr)
        m = P.red_mask(fr.astype(np.float32))
        print(f"[chk] cam0 frame shape={fr.shape} min/max/mean={fr.min()}/{fr.max()}/{fr.mean():.1f} "
              f"red_area={int(m.sum())}  -> runs/wrist_env_frame.png", flush=True)
    wf = obs[:, -3:].detach().cpu().numpy()                 # (u_norm, v_norm, visible)
    vis = int((wf[:, 2] > 0.5).sum())
    samp = wf[wf[:, 2] > 0.5][:3].round(3).tolist()
    line2 = (f"steps/s={sps:.2f} env-steps/s={sps*env.num_envs:.0f}  wrist visible={vis}/{env.num_envs}  "
             f"sample(u,v,vis)={samp}")
    print("[chk]", line2, flush=True)
    # extrapolate the retrain time for a few transition budgets
    eps = sps * env.num_envs
    for M in (1_000_000, 2_000_000):
        print(f"[chk] {M/1e6:.0f}M transitions @ {eps:.0f} env-steps/s -> {M/max(eps,1e-6)/3600:.1f} h", flush=True)
    with open("/tmp/wrist_env_check.txt", "w") as fh:
        fh.write(line1 + "\n" + line2 + "\n"); fh.flush(); os.fsync(fh.fileno())
    app.close()


if __name__ == "__main__":
    main()
