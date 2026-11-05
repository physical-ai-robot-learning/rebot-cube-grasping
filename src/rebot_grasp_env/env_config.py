"""Environment configuration — Squint-exact for SO101 LiftCube.

Stage: Project 2 environment.
Was src/det_xattn/packages/sim/env_setup/env_config.py (moved and renamed to the OA style on 2026-11-05; docs/refactor_2026-11.md).
Run: python -m (imported by rebot_grasp_rl.train)
Needs Isaac Sim 5.1 / Isaac Lab 2.3 + the LeIsaac reBot fork; not executable in CI (compile-checked only).


Matches Squint (https://github.com/aalmuzairee/squint) exactly:
  - Delta joint position control: arm 0.1 rad, gripper 0.2 rad
  - Observations: qpos, qvel, target_qpos, dist_to_rest, obj_pose, tcp_pos, tcp_to_obj, is_grasped
  - Rewards: reaching + grasped + place*grasped - 3*table - 1*not_lifted, /3
  - Episode: 5.0s (50 steps @ 10Hz)
  - rest_qpos = SO101 start keyframe [0,0,0,pi/2,-pi/2,60deg]
"""

import torch
import numpy as np


# ── Squint SO101 rest_qpos (start keyframe) ──
# reBot arm 6 joints [pan, lift, elbow, wrist_flex, wrist_yaw, wrist_roll] (from leisaac REBOT_REST_POSE)
_REST_QPOS = torch.tensor([-1.5708, -1.30, -1.10, 0.75, 0.0, -0.10])  # CALIB e110 2026-06-19


# ── Custom observation functions ──

def tcpPosW(env, ee_frame_cfg=None):
    """TCP position in world frame. Shape: (N, 3)."""
    from isaaclab.managers import SceneEntityCfg
    tCfg = ee_frame_cfg or SceneEntityCfg("ee_frame")
    tEe = env.scene[tCfg.name]
    tJaw0 = tEe.data.target_pos_w[:, 0, :]
    tJaw1 = tEe.data.target_pos_w[:, 1, :]
    return (tJaw0 + tJaw1) * 0.5


def tcpToObjPos(env, ee_frame_cfg=None, object_cfg=None):
    """Relative position from TCP to object. Shape: (N, 3)."""
    from isaaclab.managers import SceneEntityCfg
    tEeCfg = ee_frame_cfg or SceneEntityCfg("ee_frame")
    tObjCfg = object_cfg or SceneEntityCfg("cube")  # module-level fallback (overridden by configure_env)
    tEe = env.scene[tEeCfg.name]
    tObj = env.scene[tObjCfg.name]
    tTcp = (tEe.data.target_pos_w[:, 0, :] + tEe.data.target_pos_w[:, 1, :]) * 0.5
    return tObj.data.root_pos_w - tTcp


def isGraspedObs(env, robot_cfg=None, object_cfg=None, ee_frame_cfg=None):
    """Binary grasp signal (geometric: cube at TCP + gripper closed). Shape: (N, 1)."""
    from rebot_grasp_env.rewards import _isGrasped
    tOc = object_cfg or SceneEntityCfg("cube")
    return _isGrasped(env, tOc, min_force=0.3).float().unsqueeze(-1)


def contactForceObs(env, object_cfg=None):
    """PRIVILEGED critic input (2D): per-jaw contact-force magnitude (tanh). Appended at the TAIL of the 43D
    privileged state -> critic-only; NEVER in the student/deploy slice (STUDENT_STATE_IDX). Gives the value fn
    finer grasp-progress than binary is_grasped."""
    import torch as _t
    tGrip = env.scene["gripper_contact"].data.force_matrix_w[:, 0, 0, :].norm(dim=-1)
    tJaw = env.scene["jaw_contact"].data.force_matrix_w[:, 0, 0, :].norm(dim=-1)
    return _t.stack([_t.tanh(tGrip), _t.tanh(tJaw)], dim=-1)


def jointVelObs(env, robot_cfg=None):
    """Joint velocities. Shape: (N, 6). Squint includes qvel in obs."""
    from isaaclab.managers import SceneEntityCfg
    tCfg = robot_cfg or SceneEntityCfg("robot")
    return env.scene[tCfg.name].data.joint_vel


def resetCubesMixed(env, env_ids, cube_names=("cube", "cube_blue", "cube_purple"),
                      x_range=(0.15, 0.55), y_range=(-0.40, -0.20),
                      yaw_range=(-3.14159, 3.14159), min_sep=0.08, max_tries=12):
    """(reset event) Place ALL cubes at fully-random, NON-OVERLAPPING positions in a box centered on
    the robot's reach: x in [0.15,0.55] is SYMMETRIC about the arm base (x=0.35), y in [-0.40,-0.20]
    is the reachable depth. Each color is sampled INDEPENDENTLY from the SAME box, so the red target
    ("cube") lands left / center / right uniformly — fully mixed, not pinned to one side as the old
    fixed per-cube base positions (red 0.25 / purple 0.35 / blue 0.45) did. Vectorized over env_ids;
    a cube too close to an already-placed one is resampled (rejection) up to max_tries."""
    import torch as _t
    from isaaclab.utils.math import quat_from_euler_xyz
    env_ids = _t.as_tensor(env_ids, device=env.device) if not _t.is_tensor(env_ids) else env_ids
    tN = int(env_ids.numel())
    if tN == 0:
        return
    tOrigins = env.scene.env_origins[env_ids]                 # (n,3) world env offsets
    tDev = tOrigins.device
    tPlaced = []                                              # already-placed cube xy (local), (n,2) each
    for tName in cube_names:
        tAsset = env.scene.get(tName) if hasattr(env.scene, "get") else env.scene[tName]
        if tAsset is None:
            continue
        tXy = _t.stack([_t.empty(tN, device=tDev).uniform_(*x_range),
                       _t.empty(tN, device=tDev).uniform_(*y_range)], dim=-1)
        for _ in range(max_tries):                           # resample only the colliding envs
            if not tPlaced:
                break
            tBad = _t.zeros(tN, dtype=_t.bool, device=tDev)
            for tPrev in tPlaced:
                tBad |= (_t.norm(tXy - tPrev, dim=-1) < min_sep)
            if not bool(tBad.any()):
                break
            tM = int(tBad.sum())
            tXy[tBad, 0] = _t.empty(tM, device=tDev).uniform_(*x_range)
            tXy[tBad, 1] = _t.empty(tM, device=tDev).uniform_(*y_range)
        tPlaced.append(tXy.clone())
        tRoot = tAsset.data.default_root_state[env_ids].clone()   # [:, 2] = default spawn height
        tPos = _t.stack([tOrigins[:, 0] + tXy[:, 0], tOrigins[:, 1] + tXy[:, 1], tOrigins[:, 2] + tRoot[:, 2]], dim=-1)
        tYaw = _t.empty(tN, device=tDev).uniform_(*yaw_range)
        tQuat = quat_from_euler_xyz(_t.zeros(tN, device=tDev), _t.zeros(tN, device=tDev), tYaw)
        tAsset.write_root_pose_to_sim(_t.cat([tPos, tQuat], dim=-1), env_ids=env_ids)
        tAsset.write_root_velocity_to_sim(_t.zeros(tN, 6, device=tDev), env_ids=env_ids)


def configureEnv(env_cfg, cfg, mdp, RewTerm, DoneTerm, SceneEntityCfg, target_cube="cube"):
    """Configure env for reBot LiftCube. target_cube selects which cube to pick
    (cube=red, cube_blue, cube_purple); the other two stay as randomized distractors."""
    from rebot_grasp_env.rewards import (
        sqReaching, sqGrasped, sqNotLifted, sqLift,
        liftReward, successReward,   # graded lift + outcome bonus (fold removed; task = grasp + lift high)
        tableCollisionPenalty, successLatch, countSuccesses,
        graspCloseReward, graspContactReward, graspHoldReward,
    )

    # ══════════════════════════════════════════
    #  EPISODE & SIMULATION (Squint-exact)
    # ══════════════════════════════════════════
    # Squint: max_episode_steps=50, sim_freq=100, control_freq=10 => 5.0s
    env_cfg.episode_length_s = 10.0  # 100 steps at 10Hz -- MATCH reference (was 5.0/50; too short to lift+fold)
                                     # from the 10s we had raised only for the (now-removed) align-hold gate.
    env_cfg.sim.dt = 0.01           # 100Hz sim
    env_cfg.decimation = 10         # 100Hz / 10 = 10Hz control
    env_cfg.sim.physx.solver_position_iteration_count = 15
    env_cfg.sim.physx.solver_velocity_iteration_count = 1
    env_cfg.sim.physx.contact_offset = 0.02
    env_cfg.sim.physx.rest_offset = 0.0
    env_cfg.sim.physx.solve_articulation_contact_last = True  # gripper penetration fix
    env_cfg.sim.render_interval = cfg.get("sim_render_interval", 2)
    env_cfg.recorders = None
    env_cfg.scene.robot.spawn.activate_contact_sensors = True

    # ══════════════════════════════════════════
    #  ee_frame offsets — Squint URDF exact
    # ══════════════════════════════════════════
    from isaaclab.sensors import OffsetCfg
    # Jaw INNER contact surface (from STL mesh analysis):
    # Fixed jaw inner face: y shifted 10mm inward from Squint tip
    # Moving jaw inner face: y shifted ~10mm toward fixed jaw
    # This prevents false reaching reward from jaw closing
    # reBot grasp center, MEASURED from the finger collision AABBs (diag_tcp.py): the grasp center
    # sits ~13cm forward of the finger body origins. Both frames -> gripper_link at that grasp
    # center, so the TCP (their midpoint) lands exactly between the reBot fingertips.
    # (Old values were SO101 jaw geometry -> TCP ~9-13cm off -> reaching/grasp never reached the cube.)
    env_cfg.scene.ee_frame.target_frames[1].prim_path = env_cfg.scene.ee_frame.target_frames[0].prim_path
    env_cfg.scene.ee_frame.target_frames[0].offset = OffsetCfg(pos=(-0.086, 0.0, 0.103))
    env_cfg.scene.ee_frame.target_frames[1].offset = OffsetCfg(pos=(-0.086, 0.0, 0.103))

    # ══════════════════════════════════════════
    #  Camera — wrist only (Squint-exact, 142 v28 config)
    # ══════════════════════════════════════════
    from isaaclab.sensors import TiledCameraCfg
    import isaaclab.sim as sim_utils_cam

    # Disable front/side to save VRAM
    if hasattr(env_cfg.scene, 'front'):
        env_cfg.scene.front = None
    if hasattr(env_cfg.scene, 'side'):
        env_cfg.scene.side = None

    env_cfg.scene.wrist = TiledCameraCfg(
        # reBot: gripper cam on gripper_link looking toward the grasp (TCP ~ -0.13 x). SO101 used
        # "/Robot/gripper/fp_camera"; reBot's EE link is gripper_link. [TUNE in viewer] for exact pose.
        prim_path="{ENV_REGEX_NS}/Robot/gripper_link/fp_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.04, 0.0, 0.04), rot=(1.0, 0.0, 0.0, 0.0), convention="opengl"),
        data_types=["rgb"], spawn=sim_utils_cam.PinholeCameraCfg(focal_length=2.5, focus_distance=200.0, horizontal_aperture=24.0, vertical_aperture=18.0, clipping_range=(0.003, 50.0), lock_camera=True),
        width=128, height=128, update_period=0)  # 128x128 for learning, downsampled to 16x16

    # ══════════════════════════════════════════
    #  CUBE SPAWN — fully-random, mixed (target appears left/center/right)
    # ══════════════════════════════════════════
    # Replace the per-cube fixed-base uniform resets (which pinned red to x in [0.15,0.35] = one side)
    # with ONE custom event that samples all 3 cubes from a single box centered on the arm (x in
    # [0.15,0.55] symmetric about base 0.35) -> the red target is mixed across the workspace.
    from isaaclab.managers import EventTermCfg as _EventTerm
    tCUBENAMES = ("cube", "cube_blue", "cube_purple")
    try:
        for tEn in list(vars(env_cfg.events)):
            if tEn.startswith('_'):
                continue
            tEv = getattr(env_cfg.events, tEn, None)
            tAc = (getattr(tEv, 'params', {}) or {}).get('asset_cfg', None)
            if tAc is not None and getattr(tAc, 'name', '') in tCUBENAMES:
                setattr(env_cfg.events, tEn, None)    # drop the old fixed-base per-cube reset
    except Exception:
        pass
    # WRIST-ONLY (Squint LiftCube): size the spawn box to the WRIST eye-in-hand FOV at the REST pose so the
    # cube is ALWAYS in view from step 0 (verified via diag_rest_wrist.py: x[0.22,0.48] y[-0.40,-0.24] all
    # in-frame with margin; cx~0.12-0.88, cy~0.13-0.60). The previous WIDE box (x[0.10,0.60] y[-0.55,-0.13])
    # put many cubes OUTSIDE the wrist view at rest -> the policy had to blindly hunt for the cube (= the
    # coordination failure the user saw). Widen again only after grasp works.
    env_cfg.events.reset_cubes_mixed = _EventTerm(
        func=resetCubesMixed, mode="reset",
        params={"cube_names": tCUBENAMES, "x_range": (0.15, 0.48), "y_range": (-0.44, -0.16),  # NEAR-BASE 2026-06-24 (user): EXTEND near edge 0.22->0.15 toward base, KEEP far reach 0.48 (was SQUINT (0.22,0.48)). arm couldnt reach the under-gripper cube
                "yaw_range": (-3.14159, 3.14159), "min_sep": 0.08})

    # Disable any domain-randomization event that targets a removed camera (state-mode = no cameras).
    if hasattr(env_cfg, 'events'):
        for tEvName in list(vars(env_cfg.events)):
            if tEvName.startswith('_'):
                continue
            tEv = getattr(env_cfg.events, tEvName, None)
            tAc = (getattr(tEv, 'params', {}) or {}).get('asset_cfg', None)
            if tAc is not None and getattr(tAc, 'name', '') in ('front', 'side', 'wrist'):
                setattr(env_cfg.events, tEvName, None)

    if hasattr(env_cfg.observations, 'subtask_terms'):
        env_cfg.observations.subtask_terms = None
    env_cfg.observations.policy.concatenate_terms = True
    env_cfg.observations.policy.concatenate_dim = -1

    # ══════════════════════════════════════════
    #  OBSERVATIONS (Squint state-mode exact)
    # ══════════════════════════════════════════
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.envs.mdp import joint_pos, root_pos_w, root_quat_w

    # Clear ALL existing obs terms
    for tAttr in list(vars(env_cfg.observations.policy)):
        if not tAttr.startswith('_') and tAttr not in ('concatenate_terms', 'concatenate_dim', 'enable_corruption', 'corruption_cfg'):
            setattr(env_cfg.observations.policy, tAttr, None)

    # 1. Joint positions (6D)
    env_cfg.observations.policy.qpos = ObsTerm(
        func=joint_pos,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 2. Joint velocities (6D) — Squint includes qvel
    env_cfg.observations.policy.qvel = ObsTerm(
        func=jointVelObs,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )

    # 3. Controller target_qpos (6D)
    def _targetQpos(env, robot_cfg=None):
        import torch as _t
        tAm = env.action_manager
        tTargets = []
        for tTerm in tAm._terms.values():
            if hasattr(tTerm, '_target_qpos'):
                tTargets.append(tTerm._target_qpos)
        if tTargets:
            return _t.cat(tTargets, dim=-1)
        return env.scene["robot"].data.joint_pos
    env_cfg.observations.policy.target_qpos = ObsTerm(
        func=_targetQpos,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )

    # 4. dist_to_rest_qpos (5D, exclude gripper) — Squint exact
    def _distToRestQpos(env, robot_cfg=None):
        # reBot: 6 arm joints by name vs rest pose (robust to action-term internals)
        tRobot = env.scene["robot"]
        tArmIds, _ = tRobot.find_joints(["shoulder_pan", "shoulder_lift", "elbow_flex",
                                        "wrist_flex", "wrist_yaw", "wrist_roll"])
        tQ = tRobot.data.joint_pos[:, tArmIds]
        return tQ - _REST_QPOS.to(tQ.device)
    env_cfg.observations.policy.dist_to_rest = ObsTerm(
        func=_distToRestQpos,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )

    # 5. Object position (3D)
    env_cfg.observations.policy.obj_pos = ObsTerm(
        func=root_pos_w,
        params={"asset_cfg": SceneEntityCfg(target_cube)},
    )
    # 6. Object orientation (4D)
    env_cfg.observations.policy.obj_quat = ObsTerm(
        func=root_quat_w,
        params={"asset_cfg": SceneEntityCfg(target_cube)},
    )
    # 7. TCP position (3D)
    env_cfg.observations.policy.tcp_pos = ObsTerm(
        func=tcpPosW,
        params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )
    # 8. TCP-to-object (3D)
    env_cfg.observations.policy.tcp_to_obj = ObsTerm(
        func=tcpToObjPos,
        params={"ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg(target_cube)},
    )
    # 9. Is grasped (1D)
    env_cfg.observations.policy.is_grasped = ObsTerm(
        func=isGraspedObs,
        params={"robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg(target_cube),
                "ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )
    # PRIVILEGED critic-only (tail of the 43->45D state; not in STUDENT_STATE_IDX) 2026-06-23
    env_cfg.observations.policy.contact_force = ObsTerm(
        func=contactForceObs,
        params={"object_cfg": SceneEntityCfg(target_cube)},
    )

    # ══════════════════════════════════════════
    #  ACTIONS — Squint-exact delta scale
    # ══════════════════════════════════════════
    from rebot_grasp_env.target_delta_action import TargetDeltaJointPositionActionCfg
    from leisaac.devices.mirrored_gripper_action import MirroredTargetDeltaGripperActionCfg
    # arm: reBot 6 joints (adds wrist_yaw), delta position control
    env_cfg.actions.arm_action = TargetDeltaJointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll"],
        scale=0.05,   # delta rad/step (halved from 0.1 -> ~0.5 rad/s arm motion, calmer for sim2real)
        # JOINT CAPS DISABLED (user request 2026-06-16): no target_lo/hi -> the accumulated joint target is
        # bounded only by the USD physical joint limits (no artificial position cap). Elbow gate kept (below).
        # Coupled elbow gate KEPT ON (real-robot feedback v2 2026-06-11): the real robot REACHES BETTER
        # with the gate (= package4); removing it (v3) made reaching worse. So train WITH the gate, and
        # the deploy ActionController (which still has the gate) matches the policy. Elbow may extend only
        # once shoulder_pan >= couple_pan_forward (~88deg); below that it's held in the compact tuck.
        couple_object=None, couple_pan_local=0, couple_elbow_local=2,   # GATE OFF 2026-06-20 (no pan/lift/elbow restriction)
        couple_pan_forward=1.54, couple_elbow_lo=-1.4, couple_elbow_hi=-0.7,
    )
    # gripper: 1-DOF prismatic (meters), Squint-style DELTA-on-target (action=0 -> HOLD, rate-limited).
    # MirroredTargetDeltaGripperAction drives BOTH fingers (gripper + gripper_joint2) together.
    env_cfg.actions.gripper_action = MirroredTargetDeltaGripperActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        mirror_joint_names=["gripper_joint2"],
        scale=0.0143,            # delta m/step
        # CONFIRMED by render (diag_gripper_openclose) + measured (diag_gripper_stroke): qpos UP = OPEN
        # (jaws apart), qpos DOWN = CLOSED (jaws together). mouth gap: q0->0cm(shut) | q0.030->~3.5cm
        # (grips 4cm cube) | q0.035->4.0cm | q0.045->~5.5cm | q0.0715->10.4cm (mechanical MAX = over-open).
        # MATCH THE REAL GRIPPER SAFE RANGE (feedback v2: real closed=+320deg, over-open=0deg, moderate-
        # open=+40deg): sim CLOSED q0 <-> real +320deg(grip); sim max-OPEN q0.045(~5.5cm, clears 4cm cube)
        # <-> real +40deg(moderate open). The dangerous over-open (real 0deg / sim q>0.045) is EXCLUDED
        # from the training range so the policy can never command it.
        # NB: stroke_lo MUST stay 0.0 — the grip force comes from the PD pushing the close TARGET well below
        # the cube-contact qpos(~0.035); raising stroke_lo to ~0.03 kills the push -> joint can't grip
        # (diag_gripper_stroke: force@q0.03=7N IF reached, but PD can't reach it from a 0.03 target). Limit
        # real grip TORQUE via current/torque limits, NOT via the position target.
        stroke_lo=0.0, stroke_hi=0.045,
    )

    # ══════════════════════════════════════════
    #  TERMINATIONS — time_out only
    # ══════════════════════════════════════════
    if hasattr(env_cfg, 'terminations'):
        for tAttr in list(vars(env_cfg.terminations)):
            if not tAttr.startswith('_') and tAttr != 'time_out':
                setattr(env_cfg.terminations, tAttr, None)

    # ── Fleet-wide success counter (ALL envs, not just env_0) ──────────────────────
    # success_latch: termination term that ALWAYS returns False; runs every step before
    #   reset and latches per-env success while the terminal state is still valid.
    # success_counter: reset event that tallies + clears the latch for the done env_ids.
    # Counts feed the dashboard via maniskill_rewards._succ_total / _ep_total / _succ_recent.
    from isaaclab.managers import EventTermCfg as _EventTerm
    env_cfg.terminations.success_latch = DoneTerm(
        func=successLatch,
        params={"object_cfg": SceneEntityCfg(target_cube)})
    env_cfg.events.success_counter = _EventTerm(func=countSuccesses, mode="reset", params={})

    # ══════════════════════════════════════════
    #  REWARDS — Squint-exact (5 terms)
    #  Isaac Lab multiplies reward * weight * dt internally.
    #  dt = decimation * sim_dt = 10 * 0.01 = 0.1
    #  Squint does NOT multiply by dt. To match: weight_isaaclab = weight_squint / dt
    #  Squint normalized reward = raw / 3, so weight_squint = 1/3
    #  => weight_isaaclab = (1/3) / 0.1 = 10/3
    # ══════════════════════════════════════════
    tDTCOMP = 10.0  # 1 / dt = 1 / 0.1

    # SQUINT-EXACT 5 terms (lift.py:385-398), all /3 (normalized_dense) -> weight = coef/3*_DT_COMP.
    # No dopamine, no chained gates, no 15cm bar, no FOLDED pose. Stationary + ungated.

    # 1. Reaching: 1 - tanh(5 * dist(tcp, cube))   [coef +1]
    env_cfg.rewards.reaching = RewTerm(
        func=sqReaching,
        params={"object_cfg": SceneEntityCfg(target_cube),
                "ee_frame_cfg": SceneEntityCfg("ee_frame")},
        weight=1.0 / 3 * tDTCOMP)

    # 2. Grasped: binary +1 when grasped   [coef +1]
    env_cfg.rewards.grasped = RewTerm(
        func=sqGrasped,
        params={"object_cfg": SceneEntityCfg(target_cube),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "robot_cfg": SceneEntityCfg("robot")},
        weight=1.0 / 3 * tDTCOMP)

    # 3. (Place REMOVED 2026-06-25 — the old single fold-toward-REST term, superseded by the lift_hold+fold+
    #    fold_hold curriculum below. It was the stale "old reward" still showing in the current breakdown.)

    # 4. Table collision penalty   [coef -3]
    env_cfg.rewards.table_penalty = RewTerm(
        func=tableCollisionPenalty,
        params={"table_height": 0.04,
                "ee_frame_cfg": SceneEntityCfg("ee_frame")},
        weight=-3.0 / 3 * tDTCOMP)  # SQUINT-MATCH 2026-06-23: table coef -3 restored

    # 5. Not-lifted penalty (full Squint weight, ~5mm threshold)   [coef -1]
    env_cfg.rewards.not_lifted = RewTerm(
        func=sqNotLifted,
        params={"object_cfg": SceneEntityCfg(target_cube)},
        weight=-1.0 / 3 * tDTCOMP)

    # 6. LIFT (2026-06-25, user): FOLD REMOVED — task = "grasp + lift high". Graded lift reward grows with cube
    #    height (×grasped), full at +15cm (LIFT_FULL). Main shaping term -> weight 2/3.
    env_cfg.rewards.lift = RewTerm(
        func=liftReward,
        params={"object_cfg": SceneEntityCfg(target_cube)},
        weight=2.0 / 3 * tDTCOMP)
    # 7. SUCCESS bonus (user 2026-06-25): +1.0/step while in the success state (cube>=+7cm & grasped) — the
    #    OUTCOME reward (success had none before, only counted). weight 1.0 so the goal is strongly reinforced.
    env_cfg.rewards.success = RewTerm(
        func=successReward,
        params={"object_cfg": SceneEntityCfg(target_cube)},
        weight=1.0 * tDTCOMP)

    # (action_rate smoothness penalty DROPPED -- the HARD joint limits + elbow gate in arm_action already
    #  prevent the gross twisting/backward motion; residual fine jitter, if any, is handled at deploy by
    #  the EMA action filter in rebot_deploy.py. pan/wrist limits live in the action term, not here.)

    # REVERTED to the ORIGINAL Squint 5-term reward (lift.py compute_dense_reward): reaching + grasped +
    # place + table_penalty + not_lifted. The pre-grasp shaping (open_approach/align_hold/close_align) was
    # our feedback-v2 overlay; Squint succeeds on the 5 alone, so the added stateful align-latch machinery
    # is dropped. Grasp now uses a STATELESS alignment-distance gate only (see _is_grasped).


    tKeep = {"reaching", "grasped", "table_penalty", "not_lifted",            # base 4
            "lift", "success"}    # + graded lift + success outcome bonus (fold removed 2026-06-25;
    #  WAS the silent bug: this keep-filter stripped lift_hold/fold/fold_hold right after registration -> fold
    #  reward never active -> teacher couldn't learn the fold -> success=0. `place` dropped (superseded).
    for tName in list(vars(env_cfg.rewards)):
        if not tName.startswith("_") and tName not in tKeep:
            setattr(env_cfg.rewards, tName, None)

    return env_cfg


def applyMotorLimits(env, cfg):
    """Set PD gains — STS3215 real servo matching."""
    try:
        import omni.usd
        from pxr import UsdPhysics
        tStage = omni.usd.get_context().get_stage()
        tCount = 0
        # reBot: keep the verified Damiao actuator gains from REBOT_FOLLOWER_CFG (do NOT override here).
        # (Previously forced STS3215 servo gains 20/2/3.5 on EVERY angular drive -- wrong for reBot.)
        try:
            print("[Motor] velocity: unlimited", flush=True)
        except Exception as tVe:
            print(f"[Motor] velocity limit failed: {tVe}", flush=True)
        if tCount > 0:
            print(f"[Motor] PD: stiffness=20 damping=2 force=3.5Nm STS3215-real ({tCount} joints)", flush=True)
    except Exception as tE:
        print(f"[Motor] Error: {tE}", flush=True)

    # Gripper friction = 2.0 (Squint-exact: SO101.py urdf_config)
    try:
        import omni.usd
        from pxr import UsdPhysics
        tStage = omni.usd.get_context().get_stage()
        tGripCount = 0
        for tPrim in tStage.Traverse():
            tName = tPrim.GetName().lower()
            if tName in ("gripper_left", "gripper_right", "gripper_link"):
                tMat = UsdPhysics.MaterialAPI.Get(tStage, tPrim.GetPath())
                if not tMat:
                    tMat = UsdPhysics.MaterialAPI.Apply(tPrim)
                if tMat:
                    tMat.GetStaticFrictionAttr().Set(2.0)
                    tMat.GetDynamicFrictionAttr().Set(2.0)
                    tMat.GetRestitutionAttr().Set(0.0)
                    tGripCount += 1
        print(f"[Gripper] friction=2.0 on {tGripCount} prims (Squint-exact)", flush=True)
    except Exception as tE:
        print(f"[Gripper] Friction error: {tE}", flush=True)

    # Robot link mass readout
    try:
        import omni.usd
        from pxr import UsdPhysics
        tStage = omni.usd.get_context().get_stage()
        tTotalRobotMass = 0.0
        for tPrim in tStage.Traverse():
            tPath = str(tPrim.GetPath())
            if "/env_0/" not in tPath or "cube" in tPrim.GetName().lower():
                continue
            tMassApi = UsdPhysics.MassAPI.Get(tStage, tPrim.GetPath())
            if tMassApi:
                tM = tMassApi.GetMassAttr().Get()
                if tM and tM > 0:
                    tTotalRobotMass += tM
                    print(f"[Robot] {tPrim.GetName():25s} mass={tM:.4f} kg", flush=True)
        print(f"[Robot] Total robot mass (env_0): {tTotalRobotMass:.4f} kg", flush=True)
    except Exception as tE:
        print(f"[Robot] Mass readout error: {tE}", flush=True)

    # Cube physics: set mass=20g + friction (Squint ~density-200 scale; was 50g/18x heavier)
    try:
        import omni.usd
        from pxr import UsdPhysics
        tStage = omni.usd.get_context().get_stage()
        tCubeCount = 0
        for tPrim in tStage.Traverse():
            if "cube" in tPrim.GetName().lower():
                tMassApi = UsdPhysics.MassAPI.Get(tStage, tPrim.GetPath())
                if not tMassApi:
                    tMassApi = UsdPhysics.MassAPI.Apply(tPrim)
                if tMassApi:
                    tMassApi.GetMassAttr().Set(0.02)  # 20g
                    tCubeCount += 1
                tMat = UsdPhysics.MaterialAPI.Get(tStage, tPrim.GetPath())
                if tMat:
                    tMat.GetStaticFrictionAttr().Set(0.3)
                    tMat.GetDynamicFrictionAttr().Set(0.3)
                    tMat.GetRestitutionAttr().Set(0.0)
        # Verify
        for tPrim in tStage.Traverse():
            if "cube" in tPrim.GetName().lower():
                tMassApi = UsdPhysics.MassAPI.Get(tStage, tPrim.GetPath())
                if tMassApi:
                    tM = tMassApi.GetMassAttr().Get()
                    if "/env_0/" in str(tPrim.GetPath()):
                        print(f"[Cube] mass={tM:.4f} kg ({tM*1000:.1f}g) friction=0.3", flush=True)
                break
        print(f"[Cube] Set {tCubeCount} cubes to 20g", flush=True)
    except Exception as tE:
        print(f"[Cube] Physics error: {tE}", flush=True)


def addDynamicsDr(env_cfg, SceneEntityCfg, target_cube="cube"):
    """Track B (sim2real study): add DYNAMICS domain randomization EventTerms — cube/gripper friction + mass +
    arm actuator PD gains, re-rolled at reset. The reBot pipeline previously randomized VISUAL only; every
    real-world-successful hard-contact sim2real transfer also randomizes dynamics (+ latency, done in the trainer).
    Opt-in: call this from the trainer AFTER configure_env. Each term is wrapped so an API mismatch degrades to a
    warning instead of crashing."""
    from isaaclab.managers import EventTermCfg as _EventTerm
    from isaaclab.envs import mdp as _imdp
    tCubes = ("cube", "cube_blue", "cube_purple")
    tAdded = []
    # (1) cube friction + restitution (physics material), reset-mode
    try:
        for tC in tCubes:
            setattr(env_cfg.events, f"dr_mat_{tC}", _EventTerm(
                func=_imdp.randomize_rigid_body_material, mode="reset",
                params={"asset_cfg": SceneEntityCfg(tC), "static_friction_range": (0.6, 1.4),
                        "dynamic_friction_range": (0.6, 1.4), "restitution_range": (0.0, 0.1), "num_buckets": 64}))
        tAdded.append("cube-material")
    except Exception as tE:
        print(f"[dyn-DR] cube material skipped: {tE}", flush=True)
    # (2) cube mass scale, reset-mode
    try:
        for tC in tCubes:
            setattr(env_cfg.events, f"dr_mass_{tC}", _EventTerm(
                func=_imdp.randomize_rigid_body_mass, mode="reset",
                params={"asset_cfg": SceneEntityCfg(tC), "mass_distribution_params": (0.7, 1.3), "operation": "scale"}))
        tAdded.append("cube-mass")
    except Exception as tE:
        print(f"[dyn-DR] cube mass skipped: {tE}", flush=True)
    # (3) gripper-finger friction (grasp robustness), reset-mode
    try:
        setattr(env_cfg.events, "dr_mat_gripper", _EventTerm(
            func=_imdp.randomize_rigid_body_material, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*gripper.*"]),
                    "static_friction_range": (1.2, 2.8), "dynamic_friction_range": (1.2, 2.8),
                    "restitution_range": (0.0, 0.1), "num_buckets": 64}))
        tAdded.append("gripper-material")
    except Exception as tE:
        print(f"[dyn-DR] gripper material skipped: {tE}", flush=True)
    # (4) arm actuator PD gains (sim STS3215 mismatch), startup-mode scale
    try:
        setattr(env_cfg.events, "dr_actuator_gains", _EventTerm(
            func=_imdp.randomize_actuator_gains, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                    "stiffness_distribution_params": (0.8, 1.2), "damping_distribution_params": (0.8, 1.2),
                    "operation": "scale"}))
        tAdded.append("actuator-PD")
    except Exception as tE:
        print(f"[dyn-DR] actuator gains skipped: {tE}", flush=True)
    print(f"[dyn-DR] dynamics randomization added: {tAdded}", flush=True)
    return env_cfg