"""Render the reBot arm with joint numbers (J1..J6) labeled + report the lowest gripper point.

Camera-enabled, 1 env, green background. Sets a clear arm pose, renders the quarter view,
projects each arm link's world position into the image and labels J1..J6 (+ gripper). Also
reads the finger links' world z (lowest gripper point) to inform a finger-tip table rule.
Saves runs/joint_labels.png.
"""
import os, sys, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/train.yaml")
ap.add_argument("--scene-config", default="configs/scene.yaml"); args = ap.parse_args()

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "enable_cameras": True})
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from isaacsim.sensors.camera import Camera
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.stage import get_current_stage
from rebot_sim2real.rl.config import load_train_config
from rebot_sim2real.constants import load_scene_config
from rebot_sim2real.scene.cameras import look_at_quat, _apply_c922


def main():
    tcfg = load_train_config(args.config); tcfg.env.num_envs = 1
    tcfg.env.pos_rand_xy = 0.0; tcfg.env.dr_enable = False
    dev = torch.device(tcfg.env.device)
    from rebot_sim2real.envs.pick_env import RebotPickEnv
    env = RebotPickEnv(tcfg.env, tcfg.reward); env.set_render(True)
    stage = get_current_stage(); origin = env.env_origins[0].detach().cpu().numpy()
    scfg = load_scene_config(args.scene_config)
    if getattr(scfg, "green_bg_enabled", False):
        from isaacsim.core.api.objects import VisualCuboid
        gz = scfg.table_z - scfg.table_size[2]
        VisualCuboid(prim_path="/World/jl_greenfloor", name="jl_greenfloor",
                     position=origin + np.array([scfg.table_center[0], scfg.table_center[1], gz + 0.003]),
                     scale=np.array([scfg.green_bg_size, scfg.green_bg_size, 0.004], dtype=float),
                     color=np.array(scfg.green_bg_color, dtype=float))
        if getattr(scfg, "green_bg_dome", False):
            from pxr import UsdLux, Gf
            d = UsdLux.DomeLight.Define(stage, "/World/jl_greendome")
            d.CreateColorAttr(Gf.Vec3f(*[float(x) for x in scfg.green_bg_color])); d.CreateIntensityAttr(300.0)

    # clear pose: slight bend so joints are visually separated
    pose = torch.tensor([[0.0, -0.7, -0.9, 0.5, 0.0, 0.0]], device=dev)
    q = env.robots.get_joint_positions().clone(); q[:, env.arm_idx] = pose
    env.robots.set_joint_positions(q); env.arm_target[:] = pose
    env.robots.set_joint_position_targets(env._full_targets())

    cam = Camera(prim_path="/World/jl_cam", name="jl_cam", resolution=(1280, 720)); cam.initialize(); _apply_c922(cam, scfg.cam)
    eye = origin + np.array([scfg.table_center[0] + 0.7, scfg.table_center[1] - 0.7, scfg.table_z + 0.5])
    tgt = origin + np.array([scfg.table_center[0], scfg.table_center[1] - 0.15, scfg.table_z + 0.2])
    cam.set_world_pose(position=eye, orientation=look_at_quat(eye, tgt), camera_axes="usd")
    for _ in range(40):
        env.world.step(render=True)

    # link world positions (joint i ~ at link i origin); fingers for the lowest-point check
    names = [f"link{i}" for i in range(1, 7)] + ["gripper_link", "gripper_left", "gripper_right"]
    wpos = {}
    for nm in names:
        try:
            rp = RigidPrim(f"/World/envs/env_0/Robot/{nm}", name=f"jl_{nm}"); rp.initialize()
            p, _ = rp.get_world_poses(); wpos[nm] = p[0].cpu().numpy()
        except Exception as e:
            print(f"[jl] {nm} skipped: {e}", flush=True)

    rgb = np.asarray(cam.get_rgba())[..., :3].astype(np.uint8)
    labels = {f"J{i}": wpos.get(f"link{i}") for i in range(1, 7)}
    labels["gripper"] = wpos.get("gripper_link")
    pts = np.array([v for v in labels.values() if v is not None], dtype=np.float32)
    if len(pts):                                  # camera backend is torch -> feed a tensor, not ndarray
        px = cam.get_image_coords_from_world_points(torch.as_tensor(pts, device=dev, dtype=torch.float32))
        px = np.asarray(px.detach().cpu().numpy() if hasattr(px, "detach") else px)
    else:
        px = np.zeros((0, 2))

    fig, axp = plt.subplots(figsize=(11, 6.5)); axp.imshow(rgb); axp.axis("off")
    if len(px) >= 2:                              # kinematic chain: J1-J2-J3-J4-J5-J6-gripper
        axp.plot(px[:, 0], px[:, 1], "-", color="deepskyblue", lw=3, alpha=0.65, zorder=3)
    keys = [k for k in labels if labels[k] is not None]
    # staggered label offsets + leader lines so the tight J4/J5/J6 wrist cluster stays legible
    OFF = {"J1": (40, 25), "J2": (50, -5), "J3": (-70, 10), "J4": (-55, -45),
           "J5": (5, -60), "J6": (60, -35), "gripper": (70, 20)}
    for k, (u, v) in zip(keys, px):
        dx, dy = OFF.get(k, (10, -10))
        axp.plot(u, v, "o", color="yellow", ms=9, mec="black", zorder=5)
        axp.annotate(k, (u, v), color="white", fontsize=15, fontweight="bold",
                     xytext=(dx, dy), textcoords="offset points", zorder=6,
                     bbox=dict(boxstyle="round", fc="red", ec="white", lw=1.0, alpha=0.9),
                     arrowprops=dict(arrowstyle="-", color="white", lw=1.2))
    axp.set_title("reBot B601-DM joints — J1/J2 base, J3 elbow, J4/J5/J6 wrist cluster", fontsize=12)
    fig.savefig(os.path.join(tcfg.log_dir, "joint_labels.png"), dpi=110, bbox_inches="tight")

    # lowest gripper point (finger-tip table-rule basis)
    fz = [wpos[n][2] for n in ("gripper_left", "gripper_right", "gripper_link") if n in wpos]
    lowest = min(fz) if fz else float("nan")
    lines = [f"table_z={tcfg.env.table_z}",
             "link world z (m): " + ", ".join(f"{n}={wpos[n][2]:.3f}" for n in names if n in wpos),
             f"lowest gripper-link/finger z = {lowest:.3f}  (table top = {tcfg.env.table_z})"]
    with open("/tmp/render_joints_result.txt", "w") as f:
        f.write("\n".join(lines) + "\n"); f.flush(); os.fsync(f.fileno())
    print("\n".join(lines), flush=True)
    app.close()


if __name__ == "__main__":
    main()
