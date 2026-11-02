"""Stage-1 feasibility: how fast can we render a per-env WRIST cam (low-res) + detect the cube, for the
D+render plan? Creates M envs, one wrist cam per env (parented to each gripper_link), renders K steps,
times it, and reports fps + env-steps/s + a sample detection. Use the number to pick num_envs/res/steps
for the rendered retrain. NO training; just measures the render+detect bottleneck.

    OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/wrist_render_perf.py --num-envs 64 --res 160x90
"""
import os, sys, time, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/train.yaml")
ap.add_argument("--scene-config", default="configs/scene.yaml")
ap.add_argument("--num-envs", type=int, default=64)
ap.add_argument("--res", default="160x90")                 # low-res wrist render (WxH)
ap.add_argument("--steps", type=int, default=30)
args = ap.parse_args()
W, H = (int(x) for x in args.res.lower().split("x"))

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "enable_cameras": True})
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.stage import get_current_stage
from pxr import UsdGeom, Gf
from rebot_sim2real.rl.config import load_train_config
from rebot_sim2real.constants import load_scene_config
from rebot_sim2real.scene.cameras import look_at_quat
from rebot_sim2real import perception as P


def main():
    tcfg = load_train_config(args.config); tcfg.env.num_envs = args.num_envs
    tcfg.env.dr_enable = False; tcfg.env.pos_rand_xy = 0.0
    from rebot_sim2real.envs.pick_env import RebotPickEnv
    env = RebotPickEnv(tcfg.env, tcfg.reward); env.set_render(True)
    stage = get_current_stage()
    scfg = load_scene_config(args.scene_config)
    mount = Gf.Vec3d(*scfg.wrist_mount_offset); aim_l = Gf.Vec3d(*scfg.wrist_aim_local)

    # one wrist cam per env, parented to that env's gripper_link (USD child)
    cams = []
    t_build = time.time()
    for p in env.env_paths:
        link = f"{p}/Robot/{tcfg.env.wrist_link_name}"
        if not stage.GetPrimAtPath(link).IsValid():
            continue
        path = link + "/wrist_cam"
        UsdGeom.Camera.Define(stage, path)
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(path)); xf.ClearXformOpOrder()
        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(mount)
        pw = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(link))
        cw = pw.Transform(mount); aw = pw.Transform(aim_l)
        w, x, y, z = look_at_quat((cw[0], cw[1], cw[2]), (aw[0], aw[1], aw[2]))
        qw = Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z)))
        ql = pw.ExtractRotation().GetQuat().GetInverse() * qw
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(ql)
        cam = Camera(prim_path=path, name=f"w{len(cams)}", resolution=(W, H)); cam.initialize()
        cams.append(cam)
    print(f"[perf] built {len(cams)} wrist cams ({W}x{H}) in {time.time()-t_build:.1f}s", flush=True)

    env.reset()
    for _ in range(15):                                    # warm up the render pipeline
        env.world.step(render=True)

    # timed loop: physics+render step, grab all cams, HSV-detect each
    narm = env.act_dim
    t0 = time.time(); ndet = 0; sample = None
    for k in range(args.steps):
        env.step(torch.zeros(env.num_envs, narm, device=env.device))
        for ci, cam in enumerate(cams):
            rgba = cam.get_rgba()
            if rgba is None or getattr(rgba, "size", 0) == 0:
                continue
            rgb = np.asarray(rgba)[..., :3].astype("uint8")
            m = P.red_mask(rgb); c = P._largest_blob_centroid(m)
            if c is not None:
                ndet += 1
                if ci == 0 and sample is None:
                    sample = (round(float(c[0])), round(float(c[1])), int((m > 0).sum()))
    dt = time.time() - t0
    sps = args.steps / dt
    print(f"[perf] {len(cams)} cams x {args.steps} steps in {dt:.1f}s -> {sps:.1f} policy-steps/s "
          f"({sps*env.num_envs:.0f} env-steps/s)", flush=True)
    print(f"[perf] detections: {ndet}/{args.steps*len(cams)} cam-frames had the cube; sample cam0 uv/area={sample}", flush=True)
    base = 550.0  # state-based env-steps/s reference (from the prior run log)
    print(f"[perf] slowdown vs state-based(~{base:.0f} steps/s): ~{base/max(sps,1e-6):.1f}x", flush=True)
    with open("/tmp/wrist_render_perf.txt", "w") as fh:
        fh.write(f"envs={env.num_envs} res={W}x{H} steps/s={sps:.1f} env-steps/s={sps*env.num_envs:.0f} "
                 f"det={ndet}/{args.steps*len(cams)} sample={sample}\n"); fh.flush(); os.fsync(fh.fileno())
    app.close()


if __name__ == "__main__":
    main()
