"""Deploy the trained "full" SAC pick policy on the real reBot. 15 Hz closed loop:
  read proprio -> FK tcp -> perception cube_pos (or tcp while grasped) -> build_obs -> policy
  -> delta-on-target + one-way grip gate -> command motors -> safety watchdog.
Control logic MIRRORS pick_env._apply_action (validated in sim by verify_closed_loop_sim.py 15/16).

** MOVES THE ARM under policy control. Hand on E-STOP. Cube in the workspace (under the rest hover).
   First test: short --max-steps. RUN YOURSELF: **
    cd ~/workspace/rebot_sim2real && .venv-deploy/bin/python deploy/run_pick.py --max-steps 200
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from rebot_sim2real.rl.config import load_train_config
from rebot_sim2real.rl.sac import SAC
from rebot_sim2real import perception as P
from build_obs import build_obs
from robot_io import DamiaoRobotIO

URDF = "assets/robot_description_gripper/reBot_B601_DM_with_gripper.urdf"
_CAP = {}               # device index -> cv2.VideoCapture
FRONT_DEV = 2           # front/top cam (perception cube_pos)
WRIST_DEV = 0           # eye-in-hand wrist cam (obs_wrist detection)
# COMMIT latch: real gripper torque needs a SUSTAINED close, but the policy (never seeing grasped=1 on
# real, since torque only rises after the cube is squeezed) bobs the arm away before torque builds. So
# once genuinely on the cube, freeze the arm at the contact pose + force-close until grasped, then hand
# back to the policy (now grasped=1) to lift. Timeout -> reopen and re-approach.
COMMIT_DIST = 0.04      # latch when tcp within this of the perceived cube (xy already aligned)
COMMIT_TIMEOUT = 25     # steps to hold the forced close before giving up and re-approaching


def _read_rgb(idx):
    """Open (once) + read a camera as RGB. Enables AUTO-EXPOSURE: cv2 opens UVC cams with AE OFF by
    default -> near-black frames -> no detection (the bug that made the first wrist/front reads dark)."""
    import cv2
    if idx not in _CAP:
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3); cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        for _ in range(10):
            cap.read()                                  # warm up auto-exposure
        _CAP[idx] = cap
    ok, fr = _CAP[idx].read()
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if ok else None


def read_camera_rgb():
    return _read_rgb(FRONT_DEV)


def wrist_feature(cfg):
    """obs_wrist deploy feature: detect the red cube in the WRIST image -> normalized (u,v,visible),
    matching pick_env._wrist_obs. Not-visible (no detection / occluded while grasped) -> (0,0,0)."""
    if not getattr(cfg, "obs_wrist", False):
        return None
    rgb = _read_rgb(WRIST_DEV)
    if rgb is None:
        return torch.zeros(1, 3)
    H, W = rgb.shape[:2]
    c = P._largest_blob_centroid(P.red_mask(rgb.astype(np.float32)))
    if c is None or c[2] < 8:                           # c = (u, v, area)
        return torch.zeros(1, 3)
    un = (c[0] - W / 2.0) / (W / 2.0); vn = (c[1] - H / 2.0) / (H / 2.0)   # normalized [-1,1], FOV-relative
    return torch.tensor([[un, vn, 1.0]], dtype=torch.float32)


def top_feature(cfg, det):
    """project (b) TOP token: the sim-trained CNN's (u,v,visible) from the top/front cam. LIVE each step;
    when the arm occludes the cube the CNN returns visible<thresh -> (0,0,0), matching the sim occlusion
    proxy (top_occlusion_radius). No cube_pos / no extrinsic calib needed for the obs (= why (b) transfers)."""
    if det is None or not getattr(cfg, "obs_top_uv", False):
        return None
    rgb = read_camera_rgb()
    if rgb is None:
        return torch.zeros(1, 3)
    return det.uv_normalized(rgb)                        # [1,3] (un, vn, visible) in [-1,1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--ckpt", default="ckpt/pick_d_wrist_best.pt")
    ap.add_argument("--calib", default="deploy/calib.npz")
    ap.add_argument("--rate", type=float, default=15.0)
    ap.add_argument("--max-steps", type=int, default=200, help="stop after N policy steps (first-test safety)")
    ap.add_argument("--no-wrist", action="store_true",
                    help="force the wrist feature to [0,0,0] (visible=0) — use if the real wrist cam mount "
                         "doesn't match sim; the policy falls back to front-cam+state (dropout-DR regime)")
    args = ap.parse_args()

    tc = load_train_config(args.config); cfg = tc.env
    dev = torch.device("cpu")
    # AUTO-DETECT the obs layout from the ckpt's actor input dim, and override the cfg flags so the
    # torque tail / wrist tail are built to MATCH the loaded policy (any ckpt deploys w/o editing config):
    #   43 = state only | 46 = +wrist | 50 = +torque | 53 = +torque +wrist
    sd = torch.load(args.ckpt, map_location=dev)
    ckpt_dim = int(sd["actor"]["net.0.weight"].shape[1])
    _DIM2FLAGS = {43: (False, False), 46: (False, True), 50: (True, False), 53: (True, True)}
    keypoint = (ckpt_dim == 39)                          # project (b): top_uv + wrist (u,v), NO privileged cube_pos
    if keypoint:
        cfg.obs_top_uv, cfg.obs_wrist, cfg.obs_torque = True, True, False
        print(f"[deploy] {os.path.basename(args.ckpt)}: obs_dim=39 KEYPOINT (b) -> top_uv(CNN)+wrist(u,v), no cube_pos in obs")
    elif ckpt_dim in _DIM2FLAGS:
        cfg.obs_torque, cfg.obs_wrist = _DIM2FLAGS[ckpt_dim]
        print(f"[deploy] {os.path.basename(args.ckpt)}: obs_dim={ckpt_dim} -> "
              f"obs_torque={cfg.obs_torque} obs_wrist={cfg.obs_wrist}")
    else:
        raise SystemExit(f"[deploy] ckpt obs_dim={ckpt_dim} unrecognized (expected 39/43/46/50/53)")
    obs_dim = ckpt_dim
    top_det = None
    if keypoint:                                         # CNN top-cam detector (occlusion-aware -> matches sim occlusion proxy)
        from cube_detector_io import CubeDetectorIO
        top_det = CubeDetectorIO(device="cpu")
    agent = SAC(obs_dim, 7, tc.sac, dev); agent.load_state_dict(sd)
    cal = np.load(args.calib); K = cal["K"].astype(float); T_cam2world = cal["T_cam2world"].astype(float)
    plane_z = cfg.table_z + cfg.cube_size / 2.0
    rest = np.array(cfg.rest_arm, float)
    rest_t = torch.tensor(rest, dtype=torch.float32).unsqueeze(0)
    arm_lo = torch.tensor(cfg.arm_limit_lo); arm_hi = torch.tensor(cfg.arm_limit_hi)
    dt = 1.0 / args.rate

    io = DamiaoRobotIO(cfg, URDF)
    io.connect()
    try:
        # CAPTURE the cube ONCE now, while the arm is AWAY from the table (top cam unoccluded). The arm
        # occludes the top-cam cube view once it homes over the table -> live perception goes garbage
        # (the failed run: cube jumped to y=-2.3). The cube is static, so a one-shot capture suffices.
        input("\n[deploy] Cube in the workspace + arm AWAY from the table (top cam clear). ENTER -> capture cube...")
        est0 = P.estimate_red_cube_world(read_camera_rgb(), K, T_cam2world, plane_z)
        wx, wy = cfg.workspace_x, cfg.workspace_y
        if est0 is None or not (wx[0] - 0.12 < est0[0] < wx[1] + 0.12 and wy[0] - 0.12 < est0[1] < wy[1] + 0.12):
            print(f"[deploy] cube capture invalid / out of workspace: {est0} -> ABORT (re-place cube / re-calib)"); return
        cube_fixed = torch.tensor(est0, dtype=torch.float32).unsqueeze(0)
        print(f"[deploy] captured cube_fixed = {np.round(est0, 3).tolist()} (will approach this static point)")

        input("[deploy] ENTER -> initial LIFT-UP (open gripper + raise to REST) + START policy (arm WILL move). E-stop ready...")
        io.initial_liftup(rest, cfg.grip_open_qpos)   # scripted ZERO/current -> REST hover (open gripper)

        arm_target = rest_t.clone()
        grip_target = torch.tensor([[cfg.grip_open_qpos]], dtype=torch.float32)
        grip_mid = 0.5 * (cfg.grip_open_qpos + cfg.grip_close_qpos)
        mode = "approach"; commit_pose = None; commit_count = 0
        for step in range(args.max_steps):
            t0 = time.time()
            pr = io.read_proprio()
            arm_qpos = torch.tensor(pr.arm_qpos, dtype=torch.float32).unsqueeze(0)
            tcp_np = io.fk_tcp(pr.arm_qpos); tcp = torch.tensor(tcp_np, dtype=torch.float32).unsqueeze(0)
            # is_grasped only when the gripper is COMMANDED to close — else the over-open gripper's
            # limit-strain torque false-fires grasped=1 at t=0 (the failed run started stuck in 'hold').
            grasped = io.is_grasped(pr.grip_qpos) and float(grip_target[0, 0]) <= grip_mid
            # SAFETY watchdog (relaxed table thresh vs sim, since FK z has a ~cm offset)
            if tcp_np[1] < cfg.y_back_term or tcp_np[2] < cfg.table_z - 0.08:
                print(f"[deploy] SAFETY halt: tcp={np.round(tcp_np,3)}"); break

            # approach the STATIC captured cube (top cam is occluded by the arm now); once grasped, cube=tcp
            cube = tcp.clone() if grasped else cube_fixed
            dist3 = float(torch.linalg.norm(cube - tcp))

            obs = build_obs(arm_qpos, torch.tensor(pr.grip_qpos, dtype=torch.float32).unsqueeze(0),
                            torch.tensor(pr.arm_qvel, dtype=torch.float32).unsqueeze(0),
                            torch.tensor(pr.grip_qvel, dtype=torch.float32).unsqueeze(0),
                            arm_target, grip_target, rest_t, cube, tcp,
                            torch.tensor([[1.0 if grasped else 0.0]]),
                            arm_torque=(torch.tensor(pr.arm_torque, dtype=torch.float32).unsqueeze(0)
                                        if getattr(cfg, "obs_torque", False) else None),
                            grip_torque=(torch.tensor([[pr.grip_torque]], dtype=torch.float32)
                                         if getattr(cfg, "obs_torque", False) else None),
                            torque_obs_scale=getattr(cfg, "torque_obs_scale", 0.2),
                            wrist_feat=(torch.zeros(1, 3) if (args.no_wrist and getattr(cfg, "obs_wrist", False))
                                        else wrist_feature(cfg)),
                            top_feat=top_feature(cfg, top_det))   # keypoint (b): live CNN top (u,v); None for project (a)
            a = agent.act(obs, deterministic=True).clamp(-1, 1)
            gate_near = dist3 < cfg.grip_gate_dist
            close = torch.ones_like(a[:, 6:7]); openg = torch.full_like(a[:, 6:7], -1.0)
            if mode == "hold":                                   # grasped: stay closed, policy lifts
                a_grip = close
                arm_target = torch.clamp(arm_target + a[:, :6] * cfg.arm_step, arm_lo, arm_hi)
            elif mode == "commit":                               # on the cube: freeze + force-close
                if grasped:
                    mode = "hold"; a_grip = close
                    arm_target = torch.clamp(arm_target + a[:, :6] * cfg.arm_step, arm_lo, arm_hi)
                else:
                    commit_count += 1; a_grip = close; arm_target = commit_pose
                    if commit_count > COMMIT_TIMEOUT:            # never gripped -> reopen, re-approach
                        mode = "approach"; a_grip = openg
            else:                                                # approach
                arm_target = torch.clamp(arm_target + a[:, :6] * cfg.arm_step, arm_lo, arm_hi)
                if grasped:
                    mode = "hold"; a_grip = close
                elif dist3 < COMMIT_DIST:
                    mode = "commit"; commit_pose = arm_target.clone(); commit_count = 0; a_grip = close
                else:
                    a_grip = a[:, 6:7] if gate_near else openg   # policy grip near / force-open far
            grip_target = torch.clamp(grip_target - a_grip * cfg.grip_step, cfg.grip_close_qpos, cfg.grip_open_qpos)
            io.command(arm_target[0].numpy(), float(grip_target[0, 0]))

            if step % 5 == 0 or mode != "approach":
                cmd_tcp = io.fk_tcp(arm_target[0].numpy())   # where the policy COMMANDED the tcp (vs measured)
                track = float(np.linalg.norm(cmd_tcp - tcp_np))   # cmd-vs-measured gap: large => motors too soft
                print(f"  t={step:3d} tcp={np.round(tcp_np,3)} cmd={np.round(cmd_tcp,3)} trk={track:.3f} "
                      f"cube={np.round(cube[0].numpy(),3)} dist={dist3:.3f} grasped={int(grasped)} "
                      f"grip={float(grip_target[0,0]):.3f} {mode}")
            time.sleep(max(0.0, dt - (time.time() - t0)))
        print("[deploy] episode done -> comeback to ZERO (lift clear + park at home)")
        io.comeback_to_zero(cfg.grip_open_qpos)   # scripted current -> APPROACH -> ZERO (open gripper)
    finally:
        io.estop()
    print("[deploy] done (torque off).")


if __name__ == "__main__":
    main()
