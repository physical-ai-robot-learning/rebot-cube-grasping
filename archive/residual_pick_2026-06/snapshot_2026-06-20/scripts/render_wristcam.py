"""Eye-in-hand approach test: arm lifted (J2/J3 raise the wrist above the table), J5 fixed, sweep
J4 (wrist pitch). For each J4 render the wrist-cam + quarter view and project the cube into the
wrist image -> shows whether J4 alone can aim the wrist cam at the table/cube during approach.
Saves runs/wristcam_approach.png (rows = J4) + /tmp/render_wristcam.txt.
"""
import os, sys, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/train.yaml")
ap.add_argument("--scene-config", default="configs/scene.yaml"); args = ap.parse_args()

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "enable_cameras": True})
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from isaacsim.core.utils.stage import get_current_stage
from rebot_sim2real.rl.config import load_train_config
from rebot_sim2real.constants import load_scene_config
from rebot_sim2real.scene.cameras import make_wrist_camera, make_quarter_camera

J4S = [-1.2, -0.8, -0.4, 0.0, 0.4]          # wrist-pitch sweep (J5 fixed at 0)
APPROACH = [0.0, -1.30, -0.60]              # J1,J2,J3: arm lifted -> wrist hovers above the table


def grab(cam):
    r = cam.get_rgba()
    return None if r is None or getattr(r, "size", 0) == 0 else np.asarray(r)[..., :3].astype("uint8")


def main():
    tcfg = load_train_config(args.config); tcfg.env.num_envs = 1
    tcfg.env.pos_rand_xy = 0.0; tcfg.env.dr_enable = False
    dev = torch.device(tcfg.env.device)
    from rebot_sim2real.envs.pick_env import RebotPickEnv
    env = RebotPickEnv(tcfg.env, tcfg.reward); env.set_render(True)
    stage = get_current_stage(); origin = env.env_origins[0].detach().cpu().numpy()
    scfg = load_scene_config(args.scene_config)
    scfg.wrist_link_path = f"/World/envs/env_0/{tcfg.env.robot_prim_name}/{tcfg.env.wrist_link_name}"
    quarter = make_quarter_camera(scfg, origin=origin, prim_path="/World/wc_quarter")
    wrist = make_wrist_camera(scfg, stage, robot_loaded=True)
    arm_idx, grip_idx = env.arm_idx, env.grip_idx
    env.reset()

    rows, report = [], []
    for j4 in J4S:
        pose = torch.tensor([[APPROACH[0], APPROACH[1], APPROACH[2], j4, 0.0, 0.0]], device=dev)
        q = env.robots.get_joint_positions().clone(); q[:, arm_idx] = pose
        env.robots.set_joint_positions(q); env.arm_target[:] = pose
        env.robots.set_joint_position_targets(env._full_targets())
        for _ in range(30):
            env.world.step(render=True)
        _, s, _ = env._compute_state(env.prev_action)
        cube_w = (s["cube_pos"][0].cpu().numpy() + origin)
        tcp = s["tcp_pos"][0].cpu().numpy()
        # project the cube into the wrist image
        px = wrist.get_image_coords_from_world_points(torch.as_tensor(cube_w[None], device=dev, dtype=torch.float32))
        px = np.asarray(px.detach().cpu().numpy() if hasattr(px, "detach") else px)[0]
        W, H = scfg.cam.resolution
        in_frame = (0 <= px[0] <= W) and (0 <= px[1] <= H)
        rows.append((j4, grab(quarter), grab(wrist), px, in_frame))
        report.append(f"J4={j4:+.2f}  tcp_y={tcp[1]:+.3f} tcp_z={tcp[2]:+.3f}  cube_px=({px[0]:.0f},{px[1]:.0f}) "
                      f"in_wrist_view={in_frame}")

    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(9, 3.0 * n))
    for i, (j4, qv, wv, px, inf) in enumerate(rows):
        for j, (img, ttl) in enumerate(((qv, "quarter"), (wv, "wrist cam"))):
            ax = axes[i, j]; ax.axis("off")
            if img is not None:
                ax.imshow(img)
                if j == 1 and inf:
                    ax.plot(px[0], px[1], "o", ms=12, mec="red", mfc="none", mew=2)
            ax.set_title(f"J4={j4:+.2f}  {ttl}" + (" [cube visible]" if (j == 1 and inf) else ("" if j == 0 else " [cube OUT]")),
                         fontsize=9)
    fig.suptitle("Eye-in-hand approach: arm lifted (J2=-1.3,J3=-0.6), J5=0, sweep J4 -> wrist cam aim", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(tcfg.log_dir, "wristcam_approach.png"), dpi=95, bbox_inches="tight")
    out = "approach pose J1,J2,J3 = " + str(APPROACH) + "  (J5=0)\n" + "\n".join(report)
    with open("/tmp/render_wristcam.txt", "w") as f:
        f.write(out + "\n"); f.flush(); os.fsync(f.fileno())
    print(out, flush=True)
    app.close()


if __name__ == "__main__":
    main()
