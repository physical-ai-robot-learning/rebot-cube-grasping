"""Render a TRAINED policy rollout (1 env) as a 3-view GIF + a success PNG. Loads a ckpt, runs the
deterministic policy in the obs_wrist env, grabs quarter/top/wrist each step. Use to SEE the trained
grasp. Safe to run alongside training (1 env, fits the spare GPU memory).

    OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/render_policy.py --ckpt ckpt/pick_d_wrist_best.pt
"""
import os, sys, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="ckpt/pick_d_wrist_best.pt")
ap.add_argument("--config", default="configs/train.yaml")
ap.add_argument("--scene-config", default="configs/scene.yaml")
ap.add_argument("--max-steps", type=int, default=160)
ap.add_argument("--pos-rand", type=float, default=0.04)     # spawn jitter for a non-trivial grasp
ap.add_argument("--episodes", type=int, default=3)
ap.add_argument("--mlflow-run-id", default=None,
                help="attach the rollout GIF/PNG as artifacts to this MLflow run; default = the latest "
                     "run of this exp (resolved by name). Enabled when this OR $MLFLOW_TRACKING_URI is set.")
args = ap.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rebot_sim2real.rl.config import load_train_config
tcfg = load_train_config(args.config)
tcfg.env.num_envs = 1; tcfg.env.dr_enable = False; tcfg.env.pos_rand_xy = args.pos_rand; tcfg.env.rest_rand = 0.0
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "enable_cameras": True})
import numpy as np, torch, cv2
import imageio.v2 as imageio
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.stage import get_current_stage
from rebot_sim2real.constants import load_scene_config
from rebot_sim2real.scene.cameras import look_at_quat, _apply_c922, make_quarter_camera
from rebot_sim2real.rl.sac import SAC
from rebot_sim2real.envs.pick_env import RebotPickEnv


def grab(cam, h=360):
    rgba = cam.get_rgba()
    if rgba is None or getattr(rgba, "size", 0) == 0:
        return None
    im = np.asarray(rgba)[..., :3].astype("uint8")
    w = int(im.shape[1] * h / im.shape[0])
    return cv2.resize(im, (w, h))


def main():
    dev = torch.device(tcfg.env.device)
    env = RebotPickEnv(tcfg.env, tcfg.reward); env.set_render(True)
    stage = get_current_stage(); origin = env.env_origins[0].cpu().numpy()
    scfg = load_scene_config(args.scene_config)
    top = Camera(prim_path="/World/r_top", name="r_top", resolution=scfg.cam.resolution)
    top.initialize(); _apply_c922(top, scfg.cam)
    top.set_world_pose(position=origin + np.array(scfg.top_cam_pos),
                       orientation=look_at_quat(origin + np.array(scfg.top_cam_pos),
                                                origin + np.array(scfg.top_cam_target)), camera_axes="usd")
    quarter = make_quarter_camera(scfg, origin=origin, prim_path="/World/r_quarter")
    wrist = env.wrist_cams[0]                                     # the env's obs wrist cam (reuse for viz)

    agent = SAC(env.obs_dim, env.act_dim, tcfg.sac, dev)
    agent.load_state_dict(torch.load(args.ckpt, map_location=dev))
    print(f"[render_policy] loaded {args.ckpt} (obs_dim={env.obs_dim})", flush=True)

    frames, n_succ, best_succ_frame = [], 0, None
    for ep in range(args.episodes):
        obs = env.reset()
        for _ in range(8):
            env.world.step(render=True)
        ep_done = False
        for t in range(args.max_steps):
            a = agent.act(obs, deterministic=True).clamp(-1, 1)
            obs, r, d, info = env.step(a)
            f = [grab(quarter), grab(top), grab(wrist)]
            if not any(x is None for x in f):
                hh = min(x.shape[0] for x in f)
                fr = np.concatenate([x[:hh] for x in f], axis=1)
                frames.append(fr)
                if float(info["success"][0]) > 0.5 and best_succ_frame is None:
                    best_succ_frame = fr
            if bool(d[0]):
                if float(info["ep_success"][0]) > 0.5:
                    n_succ += 1
                ep_done = True
                break
        if not ep_done and float(info["ep_success"][0]) > 0.5:
            n_succ += 1
    os.makedirs(tcfg.log_dir, exist_ok=True)
    sp = os.path.join(tcfg.log_dir, "policy_success.png")
    if os.path.exists(sp):
        os.remove(sp)                       # clear STALE: success PNG is only present if THIS run hit a success
    if frames:
        clip = np.stack(frames)[::2, ::2, ::2, :]   # downsample frames+resolution -> lighter GIF (was ~90MB full-res)
        imageio.mimsave(os.path.join(tcfg.log_dir, "policy_rollout.gif"), list(clip), format="GIF", fps=10, loop=0)
        imageio.imwrite(os.path.join(tcfg.log_dir, "policy_rollout_last.png"), frames[-1])
        if best_succ_frame is not None:
            imageio.imwrite(sp, best_succ_frame)
    msg = f"[render_policy] episodes={args.episodes} success={n_succ}/{args.episodes} frames={len(frames)} -> runs/policy_rollout.gif"
    print(msg, flush=True)
    with open("/tmp/render_policy.txt", "w") as fh:
        fh.write(msg + "\n"); fh.flush(); os.fsync(fh.fileno())

    # best-effort: attach the rollout media to the MLflow run so the success GIF/PNG show in MLflow artifacts
    if os.environ.get("MLFLOW_TRACKING_URI") or args.mlflow_run_id:
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://<ray-head>:5000"))
            cl = MlflowClient()
            rid = args.mlflow_run_id
            if not rid:
                ex = cl.get_experiment_by_name(tcfg.exp_name)
                runs = (cl.search_runs([ex.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
                        if ex else [])
                rid = runs[0].info.run_id if runs else None
            if rid:
                for fn in ("policy_success.png", "policy_rollout.gif", "policy_rollout_last.png"):
                    p = os.path.join(tcfg.log_dir, fn)
                    if os.path.exists(p):
                        cl.log_artifact(rid, p, artifact_path="rollouts")
                print(f"[render_policy] logged rollout media -> MLflow run {rid[:8]} (artifacts/rollouts)", flush=True)
            else:
                print(f"[render_policy] MLflow: no run found for exp '{tcfg.exp_name}'", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[render_policy] MLflow artifact logging skipped ({type(e).__name__}: {e})", flush=True)
    app.close()


if __name__ == "__main__":
    main()
