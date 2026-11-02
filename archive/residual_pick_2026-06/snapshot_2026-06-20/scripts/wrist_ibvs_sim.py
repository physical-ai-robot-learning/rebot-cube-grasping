"""SIM proof of eye-in-hand IBVS grasp: spawn the cube OFF-center, then use the WRIST cam only to
ALIGN (servo the red centroid to a target pixel via a numerically-estimated 2x2 image Jacobian over
[J1,J2]) -> feedforward descend+close+lift. Validates against ground truth (did tcp align over the
cube? did it grasp?). IBVS servos to a PIXEL, so the ~2cm PBVS parallax bias is cancelled.

    OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/wrist_ibvs_sim.py --dx 0.05 --dy 0.0
"""
import os, sys, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/train.yaml")
ap.add_argument("--scene-config", default="configs/scene.yaml")
ap.add_argument("--dx", type=float, default=0.05)        # cube tangential offset from center (m)
ap.add_argument("--dy", type=float, default=0.0)         # cube forward offset (m)
ap.add_argument("--target-u", type=float, default=634)   # grasp-ready-approach target pixel (from wrist_servo_sim)
ap.add_argument("--target-v", type=float, default=648)
ap.add_argument("--iters", type=int, default=18)
ap.add_argument("--gain", type=float, default=0.6)
args = ap.parse_args()

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "enable_cameras": True})
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from isaacsim.core.utils.stage import get_current_stage
from rebot_sim2real.rl.config import load_train_config
from rebot_sim2real.constants import load_scene_config
from rebot_sim2real.scene.cameras import make_wrist_camera
from rebot_sim2real import perception as P
from rebot_sim2real.grasp_traj import APPROACH as APPROACH_Q, GRASP as GRASP_Q, LIFT as LIFT_Q


def main():
    tcfg = load_train_config(args.config)
    tcfg.env.num_envs = 1; tcfg.env.pos_rand_xy = 0.0; tcfg.env.dr_enable = False
    tcfg.env.rest_rand = 0.0; tcfg.env.yaw_rand = 0.0
    dev = torch.device(tcfg.env.device)
    from rebot_sim2real.envs.pick_env import RebotPickEnv
    env = RebotPickEnv(tcfg.env, tcfg.reward); env.set_render(True)
    stage = get_current_stage()
    scfg = load_scene_config(args.scene_config)
    scfg.wrist_link_path = f"/World/envs/env_0/{tcfg.env.robot_prim_name}/{tcfg.env.wrist_link_name}"
    wrist = make_wrist_camera(scfg, stage, robot_loaded=True)
    ai, gi = env.arm_idx, env.grip_idx
    origin = env.env_origins[0].cpu().numpy()
    tc = tcfg.env.table_center
    plane_z = tcfg.env.table_z + tcfg.env.cube_size / 2.0
    APPROACH = torch.tensor([list(APPROACH_Q)], device=dev)
    dGRASP = torch.tensor([list(GRASP_Q)], device=dev) - APPROACH      # relative descend delta
    dLIFT = torch.tensor([list(LIFT_Q)], device=dev) - torch.tensor([list(GRASP_Q)], device=dev)
    GO, GC = tcfg.env.grip_open_qpos, tcfg.env.grip_close_qpos
    uv_t = np.array([args.target_u, args.target_v])

    def settle(n=3):
        for _ in range(n):
            env.world.step(render=True)

    def hold(arm, grip):
        q = env.robots.get_joint_positions().clone(); q[:, ai] = arm; q[:, gi] = grip
        env.robots.set_joint_positions(q); env.arm_target[:] = arm; env.grip_target[:] = grip
        env.robots.set_joint_position_targets(env._full_targets())

    def uv():
        rgba = wrist.get_rgba()
        if rgba is None or getattr(rgba, "size", 0) == 0:
            return None, 0
        rgb = np.asarray(rgba)[..., :3].astype("uint8")
        m = P.red_mask(rgb); c = P._largest_blob_centroid(m)
        return (None if c is None else np.array([c[0], c[1]], float)), int((m > 0).sum())

    def true_err():
        _, st, _ = env._compute_state(env.prev_action)
        tcp = st["tcp_pos"][0].cpu().numpy(); cube = st["cube_pos"][0].cpu().numpy()
        return float(np.linalg.norm(tcp[:2] - cube[:2])), tcp, cube

    # reset + place cube OFF-center
    env.reset()
    cube_w = torch.tensor([origin[0] + tc[0] + args.dx, origin[1] + tc[1] + args.dy, origin[2] + plane_z],
                          device=dev)
    cp = env.cubes.get_world_poses()[0].clone(); cp[0] = cube_w
    env.cubes.set_world_poses(positions=cp[:1]); env.cubes.set_velocities(torch.zeros(1, 6, device=dev))
    arm = APPROACH.clone(); hold(arm, GO); settle(20)

    uv0, area0 = uv(); e0, tcp0, cube0 = true_err()
    print(f"[ibvs] cube offset=({args.dx},{args.dy})  start uv={None if uv0 is None else uv0.round()} "
          f"area={area0}  tcp-cube xy err={e0:.3f}m", flush=True)
    if uv0 is None:
        print("[ibvs] cube not visible at start -> abort"); app.close(); return

    # --- numerical 2x2 image Jacobian J = d(u,v)/d(J1,J2) ---
    d = 0.04
    J = np.zeros((2, 2))
    for k, j in enumerate((0, 1)):                     # control joints J1, J2
        aq = arm.clone(); aq[0, j] += d; hold(aq, GO); settle(3)
        uvp, _ = uv()
        aq2 = arm.clone(); aq2[0, j] -= d; hold(aq2, GO); settle(3)
        uvm, _ = uv()
        J[:, k] = (uvp - uvm) / (2 * d)
    hold(arm, GO); settle(3)
    print(f"[ibvs] image Jacobian d(u,v)/d(J1,J2)=\n{np.round(J,1)}", flush=True)

    # --- IBVS servo: drive centroid -> target pixel ---
    for it in range(args.iters):
        cur, area = uv()
        if cur is None:
            print(f"[ibvs] it{it}: lost sight -> stop servo"); break
        err = cur - uv_t
        if np.linalg.norm(err) < 8:
            print(f"[ibvs] it{it}: aligned (|err|={np.linalg.norm(err):.1f}px)"); break
        try:
            dq = -args.gain * np.linalg.solve(J, err)
        except np.linalg.LinAlgError:
            dq = -args.gain * (np.linalg.pinv(J) @ err)
        dq = np.clip(dq, -0.06, 0.06)
        arm[0, 0] += float(dq[0]); arm[0, 1] += float(dq[1])
        hold(arm, GO); settle(3)
        if it % 3 == 0:
            print(f"  it{it:2d} uv={cur.round()} err={err.round()} dq={np.round(dq,3)} area={area}", flush=True)

    ea, _, _ = true_err()
    curf, _ = uv()
    print(f"[ibvs] after align: uv={None if curf is None else curf.round()} (target {uv_t})  "
          f"tcp-cube xy err {e0:.3f} -> {ea:.3f} m", flush=True)

    # --- feedforward descend + close + lift (cube occludes during close; open-loop). Keep the SERVOED
    #     J1 (tangential alignment) but use the scripted grasp posture for J2-J6 (descend onto the cube
    #     along the aligned heading); a fixed relative joint delta from the servoed J2 misses. ---
    j1 = float(arm[0, 0])
    grasp_pose = torch.tensor([[j1] + list(GRASP_Q[1:])], device=dev)
    lift_pose = torch.tensor([[j1] + list(LIFT_Q[1:])], device=dev)
    for f in np.linspace(0, 1, 12):
        hold(arm * (1 - f) + grasp_pose * f, GO); settle(2)
    for _ in range(10):
        hold(grasp_pose, GC); settle(2)
    for f in np.linspace(0, 1, 12):
        hold(grasp_pose * (1 - f) + lift_pose * f, GC); settle(2)
    settle(6)
    _, st, _ = env._compute_state(env.prev_action)
    cube_z = float(st["cube_pos"][0, 2]); grasped = bool(st["is_grasped"][0])
    success = cube_z >= tcfg.env.table_z + tcfg.env.success_lift_height
    print(f"[ibvs] RESULT offset=({args.dx},{args.dy}): align_err={ea:.3f}m  grasped={grasped}  "
          f"cube_z={cube_z:.3f}  lifted={'YES' if success else 'no'}", flush=True)
    with open("/tmp/wrist_ibvs_sim.txt", "w") as fh:
        fh.write(f"offset=({args.dx},{args.dy}) start_err={e0:.3f} align_err={ea:.3f} grasped={grasped} "
                 f"cube_z={cube_z:.3f} lifted={success}\n"); fh.flush(); os.fsync(fh.fileno())
    app.close()


if __name__ == "__main__":
    main()
