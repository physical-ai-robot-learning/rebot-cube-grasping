"""NO-MOTION dry run: read the arm + BOTH cameras, build the 53-D obs, run the policy FORWARD and
PRINT the action it WOULD command (but never send it). Validates the full deploy perception -> obs ->
policy pipeline (cameras/AE, detections, obs dim, ckpt load, sane action) before any arm motion.
Saves annotated front/wrist frames -> deploy/dry_front.png, deploy/dry_wrist.png.

** Reads the arm (connect holds torque at the CURRENT pose -> no commanded motion). E-stop ready.
   RUN YOURSELF: **
    cd ~/workspace/rebot_sim2real && .venv-deploy/bin/python deploy/dry_run.py --ckpt ckpt/pick_d_wrist_latest.pt
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np, torch, cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from rebot_sim2real.rl.config import load_train_config
from rebot_sim2real.rl.sac import SAC
from rebot_sim2real import perception as P
from build_obs import build_obs
from robot_io import DamiaoRobotIO
from run_pick import read_camera_rgb, wrist_feature, _read_rgb, WRIST_DEV

URDF = "assets/robot_description_gripper/reBot_B601_DM_with_gripper.urdf"


def _annotate(rgb, c, tag):
    bgr = rgb[..., ::-1].copy()
    h, w = bgr.shape[:2]
    m = P.red_mask(rgb.astype(np.float32)); bgr[m > 0] = (0, 255, 0)
    if c is not None:
        cv2.drawMarker(bgr, (int(c[0]), int(c[1])), (255, 0, 0), cv2.MARKER_CROSS, 40, 3)
    cv2.circle(bgr, (w // 2, h // 2), 6, (0, 0, 255), 2)
    cv2.imwrite(f"deploy/dry_{tag}.png", bgr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--ckpt", default="ckpt/pick_d_wrist_latest.pt")
    ap.add_argument("--calib", default="deploy/calib.npz")
    ap.add_argument("--steps", type=int, default=12)
    args = ap.parse_args()

    tc = load_train_config(args.config); cfg = tc.env
    dev = torch.device("cpu")
    obs_dim = 43 + (7 if getattr(cfg, "obs_torque", False) else 0) + (3 if getattr(cfg, "obs_wrist", False) else 0)
    agent = SAC(obs_dim, 7, tc.sac, dev); agent.load_state_dict(torch.load(args.ckpt, map_location=dev))
    cal = np.load(args.calib); K = cal["K"].astype(float); T_cam2world = cal["T_cam2world"].astype(float)
    plane_z = cfg.table_z + cfg.cube_size / 2.0
    rest_t = torch.tensor(np.array(cfg.rest_arm, float), dtype=torch.float32).unsqueeze(0)
    arm_target = rest_t.clone(); grip_target = torch.tensor([[cfg.grip_open_qpos]], dtype=torch.float32)
    print(f"[dry] obs_dim={obs_dim} (torque={cfg.obs_torque}, wrist={cfg.obs_wrist})  ckpt={args.ckpt}", flush=True)

    io = DamiaoRobotIO(cfg, URDF); io.connect()
    try:
        for step in range(args.steps):
            pr = io.read_proprio()
            arm_qpos = torch.tensor(pr.arm_qpos, dtype=torch.float32).unsqueeze(0)
            tcp_np = io.fk_tcp(pr.arm_qpos); tcp = torch.tensor(tcp_np, dtype=torch.float32).unsqueeze(0)
            grasped = io.is_grasped(pr.grip_qpos)
            front = read_camera_rgb()
            est = P.estimate_red_cube_world(front, K, T_cam2world, plane_z) if front is not None else None
            cube = torch.tensor(est, dtype=torch.float32).unsqueeze(0) if est is not None else tcp.clone()
            wf = wrist_feature(cfg)
            obs = build_obs(arm_qpos, torch.tensor(pr.grip_qpos, dtype=torch.float32).unsqueeze(0),
                            torch.tensor(pr.arm_qvel, dtype=torch.float32).unsqueeze(0),
                            torch.tensor(pr.grip_qvel, dtype=torch.float32).unsqueeze(0),
                            arm_target, grip_target, rest_t, cube, tcp,
                            torch.tensor([[1.0 if grasped else 0.0]]),
                            arm_torque=(torch.tensor(pr.arm_torque, dtype=torch.float32).unsqueeze(0)
                                        if cfg.obs_torque else None),
                            grip_torque=(torch.tensor([[pr.grip_torque]], dtype=torch.float32) if cfg.obs_torque else None),
                            torque_obs_scale=getattr(cfg, "torque_obs_scale", 0.2),
                            wrist_feat=wf)
            a = agent.act(obs, deterministic=True).clamp(-1, 1)[0].numpy()
            ok_dim = obs.shape[-1] == obs_dim
            wfp = None if wf is None else np.round(wf[0].numpy(), 2).tolist()
            print(f"  t={step:2d} obs={tuple(obs.shape)}{'' if ok_dim else ' !!DIM MISMATCH'} "
                  f"qpos={np.round(pr.arm_qpos,2).tolist()} tcp={np.round(tcp_np,3).tolist()} "
                  f"cube={'DET '+str(np.round(cube[0].numpy(),3).tolist()) if est is not None else 'none'} "
                  f"wrist={wfp} grasped={int(grasped)} | act={np.round(a,2).tolist()}", flush=True)
            if step == 0:                       # save annotated frames for a visual check
                if front is not None:
                    _annotate(front, P._largest_blob_centroid(P.red_mask(front.astype(np.float32))), "front")
                wrgb = _read_rgb(WRIST_DEV)
                if wrgb is not None:
                    _annotate(wrgb, P._largest_blob_centroid(P.red_mask(wrgb.astype(np.float32))), "wrist")
            time.sleep(0.1)
        print("[dry] PIPELINE OK if: obs=53 every step, action in [-1,1] (no NaN), cube DET, wrist vis=1 when "
              "gripper sees the cube. Check deploy/dry_front.png + dry_wrist.png.", flush=True)
    finally:
        io.estop()    # disconnect -> torque off
    print("[dry] done (no motion was commanded).", flush=True)


if __name__ == "__main__":
    main()
