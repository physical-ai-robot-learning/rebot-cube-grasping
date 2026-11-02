"""Live training monitor (separate process — trainer untouched, throughput unaffected).

Builds ONE rebot pick env with cameras enabled, polls the latest checkpoint, and on every
new checkpoint runs a deterministic episode while capturing THREE views — 3rd-person QUARTER,
TOP, and WRIST (eye-in-hand) — assembled side-by-side and logged to TensorBoard as a video
(``monitor/three_view``). Optionally also writes mp4 and/or opens a live Isaac GUI window.

Why a separate process: ``enable_cameras`` can only be set at SimulationApp launch, so the
trainer stays camera-free (headless, render=False) and fast; rendering is fully isolated here.

Also hosts a live web dashboard (sim frames + trainer metrics, no tensorboard needed): the
3 views are pushed per frame and the trainer's ``<exp>_live.json`` is relayed to it, so one
URL shows the rollout next to the SAC curves. Disable with ``--http-port 0``.

Run (alongside the trainer):
    OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/monitor.py \
        --config configs/train.yaml --exp pick_v1 --monitor-every 30 --http-port 8890
    # then open http://localhost:8890  (or tensorboard --logdir runs for the 'monitor/three_view' video)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time


def parse_args():
    ap = argparse.ArgumentParser(description="reBot training monitor: 3-view rollout -> TensorBoard video.")
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--scene-config", default="configs/scene.yaml")
    ap.add_argument("--exp", default=None, help="experiment name (defaults to cfg.exp_name); polls <exp>_latest.pt")
    ap.add_argument("--ckpt", default=None, help="explicit checkpoint path to poll (overrides --exp)")
    ap.add_argument("--monitor-every", type=float, default=30.0, help="wall-clock seconds between checkpoint polls")
    ap.add_argument("--fps", type=int, default=15, help="clip fps (policy rate = 120/decimation)")
    ap.add_argument("--downsample", type=int, default=2, help="integer downscale of the assembled frame for lighter video")
    ap.add_argument("--gui", action="store_true", help="open a live Isaac viewport window")
    ap.add_argument("--mp4", action="store_true", help="also write an mp4 per clip")
    ap.add_argument("--once", action="store_true", help="render a single clip and exit (for testing)")
    ap.add_argument("--http-port", type=int, default=8890,
                    help="live web dashboard port (sim frames + trainer metrics); 0 to disable")
    return ap.parse_args()


args = parse_args()

# 1) launch Kit with cameras ENABLED (must be at launch time)
from isaacsim import SimulationApp
app = SimulationApp({"headless": not args.gui, "enable_cameras": True})

# 2) imports after Kit exists
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.stage import get_current_stage

from rebot_sim2real.rl.config import load_train_config
from rebot_sim2real.constants import load_scene_config
from rebot_sim2real.scene.cameras import look_at_quat, _apply_c922, make_wrist_camera, make_quarter_camera
from rebot_sim2real.rl.sac import SAC
from rebot_sim2real.viz import dashboard as dash


def grab(cam):
    """One camera RGB as HxWx3 uint8, or None if the buffer isn't populated yet."""
    rgba = cam.get_rgba()
    if rgba is None or getattr(rgba, "size", 0) == 0:
        return None
    return np.asarray(rgba)[..., :3].astype("uint8")


def latest_ckpt_and_step(ckpt_path, exp, ckpt_dir):
    """Resolve the checkpoint to load + a best-effort training step for the TB x-axis."""
    path = ckpt_path or os.path.join(ckpt_dir, f"{exp}_latest.pt")
    step = 0
    for f in glob.glob(os.path.join(ckpt_dir, f"{exp}_*.pt")):     # numbered ckpts encode env-steps
        m = re.search(rf"{re.escape(exp)}_(\d+)\.pt$", os.path.basename(f))
        if m:
            step = max(step, int(m.group(1)))
    return path, step


def main():
    tcfg = load_train_config(args.config)
    exp = args.exp or tcfg.exp_name
    tcfg.env.num_envs = 1                     # single monitored env
    device = torch.device(tcfg.env.device)

    from rebot_sim2real.envs.pick_env import RebotPickEnv
    env = RebotPickEnv(tcfg.env, tcfg.reward)
    env.set_render(True)                      # RTX pass on (env.step renders each physics step)
    stage = get_current_stage()
    origin = env.env_origins[0].detach().cpu().numpy()    # cloned env grid offset (≈0 for env_0)

    # camera geometry from the viz scene config; bind the wrist cam to the CLONED robot link
    scfg = load_scene_config(args.scene_config)
    scfg.wrist_link_path = f"/World/envs/env_0/{tcfg.env.robot_prim_name}/{tcfg.env.wrist_link_name}"

    # green-screen background (match the viz scene) so the monitor views are green-backed too
    if getattr(scfg, "green_bg_enabled", False):
        from isaacsim.core.api.objects import VisualCuboid
        gz = scfg.table_z - scfg.table_size[2]
        VisualCuboid(prim_path="/World/monitor_greenfloor", name="mon_greenfloor",
                     position=origin + np.array([scfg.table_center[0], scfg.table_center[1], gz + 0.003]),
                     scale=np.array([scfg.green_bg_size, scfg.green_bg_size, 0.004], dtype=float),
                     color=np.array(scfg.green_bg_color, dtype=float))
        if getattr(scfg, "green_bg_dome", False):
            from pxr import UsdLux, Gf
            dome = UsdLux.DomeLight.Define(stage, "/World/monitor_greendome")
            dome.CreateColorAttr(Gf.Vec3f(*[float(x) for x in scfg.green_bg_color]))
            dome.CreateIntensityAttr(300.0)

    top = Camera(prim_path="/World/monitor_top", name="monitor_top", resolution=scfg.cam.resolution)
    top.initialize(); _apply_c922(top, scfg.cam)
    top.set_world_pose(position=origin + np.array(scfg.top_cam_pos),
                       orientation=look_at_quat(origin + np.array(scfg.top_cam_pos),
                                                origin + np.array(scfg.top_cam_target)),
                       camera_axes="usd")
    quarter = make_quarter_camera(scfg, origin=origin, prim_path="/World/monitor_quarter")
    wrist = make_wrist_camera(scfg, stage, robot_loaded=True)
    cams = {"quarter": quarter, "top": top, "wrist": wrist}

    agent = SAC(env.obs_dim, env.act_dim, tcfg.sac, device)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(os.path.join(tcfg.log_dir, f"{exp}_monitor"))
    print(f"[monitor] exp={exp} | logging to {tcfg.log_dir}/{exp}_monitor | "
          f"poll every {args.monitor_every}s | gui={args.gui}", flush=True)

    # live web dashboard: this (cameras-enabled) process hosts the server, pushes the 3-view
    # frames, and relays the trainer's <exp>_live.json metrics -> one combined URL.
    live_json = os.path.join(tcfg.log_dir, f"{exp}_live.json")
    if args.http_port:
        dash.start(args.http_port)

    # warm up the render pipeline so the first capture isn't empty
    env.reset()
    for _ in range(int(getattr(env.cfg, "decimation", 8)) * 3):
        env.world.step(render=True)

    @torch.no_grad()
    def run_clip(gstep, log_tb=True):
        obs = env.reset()
        frames = []
        for _ in range(tcfg.env.episode_len + 5):
            a = agent.act(obs, deterministic=True)
            obs, _, done, info = env.step(a)
            views = {k: grab(cams[k]) for k in ("quarter", "top", "wrist")}
            if any(v is None for v in views.values()):
                continue                                 # still warming up
            if args.http_port:                           # live feed: push each view per frame
                for k, v in views.items():
                    try:
                        dash.set_camera_image(k, v)
                    except Exception:  # noqa: BLE001
                        pass
            frames.append(np.concatenate([views["quarter"], views["top"], views["wrist"]], axis=1))  # H x 3W x 3
            if info["num_done"] > 0:
                break
        if not frames:
            print("[monitor] no frames captured (cameras empty?)", flush=True)
            return
        ds = max(1, args.downsample)
        clip = np.stack(frames)[:, ::ds, ::ds, :]        # T x H' x 3W' x 3
        import imageio.v2 as imageio
        imageio.imwrite(os.path.join(tcfg.log_dir, f"{exp}_monitor_latest.png"), clip[-1])  # quick-look (always)
        # lightweight GIF of the rollout (no ffmpeg); keep a SUCCESS-only copy when the cube was lifted
        gclip = clip[::2, ::2, ::2, :]                    # coarser -> small/shareable
        succeeded = info.get("num_done", 0) > 0 and float(info["ep_success"][0]) > 0.5
        try:
            imageio.mimsave(os.path.join(tcfg.log_dir, f"{exp}_monitor_latest.gif"), list(gclip), format="GIF", loop=0)
            if succeeded:
                imageio.mimsave(os.path.join(tcfg.log_dir, f"{exp}_success.gif"), list(gclip), format="GIF", loop=0)
                imageio.imwrite(os.path.join(tcfg.log_dir, f"{exp}_success.png"), clip[-1])
                print(f"[monitor] SUCCESS clip saved (step {gstep})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[monitor] gif skipped: {e}", flush=True)
        if not log_tb:                                   # cams-only refresh (no new ckpt) -> skip TB video
            return
        vid = torch.from_numpy(clip).permute(0, 3, 1, 2).unsqueeze(0)   # 1 x T x 3 x H' x 3W'
        writer.add_video("monitor/three_view", vid, global_step=gstep, fps=args.fps)
        writer.flush()
        print(f"[monitor] logged clip: step={gstep} frames={len(frames)} shape={tuple(clip.shape)}", flush=True)
        if args.mp4:
            try:
                out = os.path.join(tcfg.log_dir, f"{exp}_monitor_{gstep}.mp4")
                imageio.mimwrite(out, list(clip), fps=args.fps)
                print(f"[monitor] wrote {out}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[monitor] mp4 skipped (pip install imageio-ffmpeg): {e}", flush=True)

    last_mtime = None
    warned_no_ckpt = False
    while app.is_running():
        ckpt, step = latest_ckpt_and_step(args.ckpt, exp, tcfg.ckpt_dir)
        new_ckpt = False
        if os.path.exists(ckpt):
            mt = os.path.getmtime(ckpt)
            if mt != last_mtime:
                last_mtime, new_ckpt = mt, True
                try:
                    agent.load_state_dict(torch.load(ckpt, map_location=device))
                    print(f"[monitor] loaded {ckpt} (step~{step})", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[monitor] load failed: {e}", flush=True)
        elif not warned_no_ckpt:
            warned_no_ckpt = True
            print(f"[monitor] no checkpoint yet ({ckpt}); showing live scene with the current policy", flush=True)
        # render every cycle so the dashboard cams stay live; log a TB video only on a new checkpoint
        try:
            run_clip(step, log_tb=new_ckpt)
        except Exception as e:  # noqa: BLE001
            print(f"[monitor] clip failed: {e}", flush=True)
        if args.once:
            break
        t_end = time.time() + args.monitor_every          # idle without freezing the app/GUI
        while time.time() < t_end and app.is_running():
            app.update()
            if args.http_port and os.path.exists(live_json):   # relay trainer metrics -> dashboard
                try:
                    with open(live_json) as fh:
                        dash.push_stats(**json.load(fh))
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(0.1)

    writer.close()
    app.close()


if __name__ == "__main__":
    main()
