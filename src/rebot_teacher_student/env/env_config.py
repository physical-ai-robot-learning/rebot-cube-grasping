"""Environment configuration — Squint-exact for SO101 LiftCube.

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
_REST_QPOS = torch.tensor([0.0, -1.15, -1.00, 0.55, 0.0, 0.0])


# ── Custom observation functions ──

def tcp_pos_w(env, ee_frame_cfg=None):
    """TCP position in world frame. Shape: (N, 3)."""
    from isaaclab.managers import SceneEntityCfg
    cfg = ee_frame_cfg or SceneEntityCfg("ee_frame")
    ee = env.scene[cfg.name]
    jaw0 = ee.data.target_pos_w[:, 0, :]
    jaw1 = ee.data.target_pos_w[:, 1, :]
    return (jaw0 + jaw1) * 0.5


def tcp_to_obj_pos(env, ee_frame_cfg=None, object_cfg=None):
    """Relative position from TCP to object. Shape: (N, 3)."""
    from isaaclab.managers import SceneEntityCfg
    ee_cfg = ee_frame_cfg or SceneEntityCfg("ee_frame")
    obj_cfg = object_cfg or SceneEntityCfg("cube")  # module-level fallback (overridden by configure_env)
    ee = env.scene[ee_cfg.name]
    obj = env.scene[obj_cfg.name]
    tcp = (ee.data.target_pos_w[:, 0, :] + ee.data.target_pos_w[:, 1, :]) * 0.5
    return obj.data.root_pos_w - tcp


def is_grasped_obs(env, robot_cfg=None, object_cfg=None, ee_frame_cfg=None):
    """Binary grasp signal (geometric: cube at TCP + gripper closed). Shape: (N, 1)."""
    from rebot_teacher_student.env.rewards import _is_grasped
    _oc = object_cfg or SceneEntityCfg("cube")
    return _is_grasped(env, _oc, min_force=0.3).float().unsqueeze(-1)


def joint_vel_obs(env, robot_cfg=None):
    """Joint velocities. Shape: (N, 6). Squint includes qvel in obs."""
    from isaaclab.managers import SceneEntityCfg
    cfg = robot_cfg or SceneEntityCfg("robot")
    return env.scene[cfg.name].data.joint_vel


def reset_cubes_mixed(env, env_ids, cube_names=("cube", "cube_blue", "cube_purple"),
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
    n = int(env_ids.numel())
    if n == 0:
        return
    origins = env.scene.env_origins[env_ids]                 # (n,3) world env offsets
    dev = origins.device
    placed = []                                              # already-placed cube xy (local), (n,2) each
    for name in cube_names:
        asset = env.scene.get(name) if hasattr(env.scene, "get") else env.scene[name]
        if asset is None:
            continue
        xy = _t.stack([_t.empty(n, device=dev).uniform_(*x_range),
                       _t.empty(n, device=dev).uniform_(*y_range)], dim=-1)
        for _ in range(max_tries):                           # resample only the colliding envs
            if not placed:
                break
            bad = _t.zeros(n, dtype=_t.bool, device=dev)
            for prev in placed:
                bad |= (_t.norm(xy - prev, dim=-1) < min_sep)
            if not bool(bad.any()):
                break
            m = int(bad.sum())
            xy[bad, 0] = _t.empty(m, device=dev).uniform_(*x_range)
            xy[bad, 1] = _t.empty(m, device=dev).uniform_(*y_range)
        placed.append(xy.clone())
        root = asset.data.default_root_state[env_ids].clone()   # [:, 2] = default spawn height
        pos = _t.stack([origins[:, 0] + xy[:, 0], origins[:, 1] + xy[:, 1], origins[:, 2] + root[:, 2]], dim=-1)
        yaw = _t.empty(n, device=dev).uniform_(*yaw_range)
        quat = quat_from_euler_xyz(_t.zeros(n, device=dev), _t.zeros(n, device=dev), yaw)
        asset.write_root_pose_to_sim(_t.cat([pos, quat], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(_t.zeros(n, 6, device=dev), env_ids=env_ids)


def configure_env(env_cfg, cfg, mdp, RewTerm, DoneTerm, SceneEntityCfg, target_cube="cube"):
    """Configure env for reBot LiftCube. target_cube selects which cube to pick
    (cube=red, cube_blue, cube_purple); the other two stay as randomized distractors."""
    from rebot_teacher_student.env.rewards import (
        sq_reaching, sq_grasped, sq_place, sq_not_lifted, sq_open_approach, sq_align_hold,
        sq_close_after_align, table_collision_penalty, success_latch, count_successes,
    )

    # ══════════════════════════════════════════
    #  EPISODE & SIMULATION (Squint-exact)
    # ══════════════════════════════════════════
    # Squint: max_episode_steps=50, sim_freq=100, control_freq=10 => 5.0s
    env_cfg.episode_length_s = 10.0  # 100 steps at 10Hz. Raised 5->10s (feedback v2): the hard align-hold
                                     # gate adds a ~1s open-aligned hold before grasping, so 5s left no time
                                     # for the final grasp->lift->REST-return (success needs all at once);
                                     # 10s fits the full reach->align-hold->close->grasp->lift->return.
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
    _CUBE_NAMES = ("cube", "cube_blue", "cube_purple")
    try:
        for _en in list(vars(env_cfg.events)):
            if _en.startswith('_'):
                continue
            _ev = getattr(env_cfg.events, _en, None)
            _ac = (getattr(_ev, 'params', {}) or {}).get('asset_cfg', None)
            if _ac is not None and getattr(_ac, 'name', '') in _CUBE_NAMES:
                setattr(env_cfg.events, _en, None)    # drop the old fixed-base per-cube reset
    except Exception:
        pass
    # WIDENED (feedback v2): bigger box + reaches the gripper-rest projection (~0.35,-0.41) so the cube
    # also spawns RIGHT UNDER the gripper (that near case was rare before and failed). y now extends back
    # to -0.47 (toward the base, easy reach) and the box is wider; still within arm reach.
    env_cfg.events.reset_cubes_mixed = _EventTerm(
        func=reset_cubes_mixed, mode="reset",
        params={"cube_names": _CUBE_NAMES, "x_range": (0.13, 0.57), "y_range": (-0.47, -0.18),
                "yaw_range": (-3.14159, 3.14159), "min_sep": 0.08})

    # Disable any domain-randomization event that targets a removed camera (state-mode = no cameras).
    if hasattr(env_cfg, 'events'):
        for _ev_name in list(vars(env_cfg.events)):
            if _ev_name.startswith('_'):
                continue
            _ev = getattr(env_cfg.events, _ev_name, None)
            _ac = (getattr(_ev, 'params', {}) or {}).get('asset_cfg', None)
            if _ac is not None and getattr(_ac, 'name', '') in ('front', 'side', 'wrist'):
                setattr(env_cfg.events, _ev_name, None)

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
    for attr in list(vars(env_cfg.observations.policy)):
        if not attr.startswith('_') and attr not in ('concatenate_terms', 'concatenate_dim', 'enable_corruption', 'corruption_cfg'):
            setattr(env_cfg.observations.policy, attr, None)

    # 1. Joint positions (6D)
    env_cfg.observations.policy.qpos = ObsTerm(
        func=joint_pos,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 2. Joint velocities (6D) — Squint includes qvel
    env_cfg.observations.policy.qvel = ObsTerm(
        func=joint_vel_obs,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )

    # 3. Controller target_qpos (6D)
    def _target_qpos(env, robot_cfg=None):
        import torch as _t
        am = env.action_manager
        targets = []
        for term in am._terms.values():
            if hasattr(term, '_target_qpos'):
                targets.append(term._target_qpos)
        if targets:
            return _t.cat(targets, dim=-1)
        return env.scene["robot"].data.joint_pos
    env_cfg.observations.policy.target_qpos = ObsTerm(
        func=_target_qpos,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )

    # 4. dist_to_rest_qpos (5D, exclude gripper) — Squint exact
    def _dist_to_rest_qpos(env, robot_cfg=None):
        # reBot: 6 arm joints by name vs rest pose (robust to action-term internals)
        robot = env.scene["robot"]
        arm_ids, _ = robot.find_joints(["shoulder_pan", "shoulder_lift", "elbow_flex",
                                        "wrist_flex", "wrist_yaw", "wrist_roll"])
        q = robot.data.joint_pos[:, arm_ids]
        return q - _REST_QPOS.to(q.device)
    env_cfg.observations.policy.dist_to_rest = ObsTerm(
        func=_dist_to_rest_qpos,
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
        func=tcp_pos_w,
        params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )
    # 8. TCP-to-object (3D)
    env_cfg.observations.policy.tcp_to_obj = ObsTerm(
        func=tcp_to_obj_pos,
        params={"ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg(target_cube)},
    )
    # 9. Is grasped (1D)
    env_cfg.observations.policy.is_grasped = ObsTerm(
        func=is_grasped_obs,
        params={"robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg(target_cube),
                "ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )

    # ══════════════════════════════════════════
    #  ACTIONS — Squint-exact delta scale
    # ══════════════════════════════════════════
    from rebot_teacher_student.env.target_delta_action import TargetDeltaJointPositionActionCfg
    from leisaac.devices.mirrored_gripper_action import MirroredTargetDeltaGripperActionCfg
    # arm: reBot 6 joints (adds wrist_yaw), delta position control
    env_cfg.actions.arm_action = TargetDeltaJointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll"],
        scale=0.05,   # delta rad/step (halved from 0.1 -> ~0.5 rad/s arm motion, calmer for sim2real)
        # HARD target limits (real-robot safety; order = joint_names). pan -> FRONT +-90deg
        # (cube always spawns in front, required_pan ~0-40deg); wrist_roll -> +-90deg to stop
        # over-spinning (cable/servo damage). others kept at their USD joint limits.
        # order: [pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll]
        #  pan        : front +-90deg
        #  wrist_flex : UP(neg) capped -10deg; DOWN(pos) +90deg for grasping
        #  wrist_yaw  : +-105deg (left-right total 210deg)
        #  wrist_roll : +-180deg per side (no more than one full half-turn each way)
        # (effective range is min(this clamp, USD joint limit); if USD is tighter the joint stays tighter.)
        target_lo=[-1.5708, -3.14, -3.14, -0.1745, -1.8326, -3.1416],
        target_hi=[ 1.5708,  0.00,  0.00,  1.57,    1.8326,  3.1416],
        # Coupled elbow gate KEPT ON (real-robot feedback v2 2026-06-11): the real robot REACHES BETTER
        # with the gate (= package4); removing it (v3) made reaching worse. So train WITH the gate, and
        # the deploy ActionController (which still has the gate) matches the policy. Elbow may extend only
        # once shoulder_pan >= couple_pan_forward (~88deg); below that it's held in the compact tuck.
        couple_object=target_cube, couple_pan_local=0, couple_elbow_local=2,
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
        for attr in list(vars(env_cfg.terminations)):
            if not attr.startswith('_') and attr != 'time_out':
                setattr(env_cfg.terminations, attr, None)

    # ── Fleet-wide success counter (ALL envs, not just env_0) ──────────────────────
    # success_latch: termination term that ALWAYS returns False; runs every step before
    #   reset and latches per-env success while the terminal state is still valid.
    # success_counter: reset event that tallies + clears the latch for the done env_ids.
    # Counts feed the dashboard via maniskill_rewards._succ_total / _ep_total / _succ_recent.
    from isaaclab.managers import EventTermCfg as _EventTerm
    env_cfg.terminations.success_latch = DoneTerm(
        func=success_latch,
        params={"object_cfg": SceneEntityCfg(target_cube)})
    env_cfg.events.success_counter = _EventTerm(func=count_successes, mode="reset", params={})

    # ══════════════════════════════════════════
    #  REWARDS — Squint-exact (5 terms)
    #  Isaac Lab multiplies reward * weight * dt internally.
    #  dt = decimation * sim_dt = 10 * 0.01 = 0.1
    #  Squint does NOT multiply by dt. To match: weight_isaaclab = weight_squint / dt
    #  Squint normalized reward = raw / 3, so weight_squint = 1/3
    #  => weight_isaaclab = (1/3) / 0.1 = 10/3
    # ══════════════════════════════════════════
    _DT_COMP = 10.0  # 1 / dt = 1 / 0.1

    # SQUINT-EXACT 5 terms (lift.py:385-398), all /3 (normalized_dense) -> weight = coef/3*_DT_COMP.
    # No dopamine, no chained gates, no 15cm bar, no FOLDED pose. Stationary + ungated.

    # 1. Reaching: 1 - tanh(5 * dist(tcp, cube))   [coef +1]
    env_cfg.rewards.reaching = RewTerm(
        func=sq_reaching,
        params={"object_cfg": SceneEntityCfg(target_cube),
                "ee_frame_cfg": SceneEntityCfg("ee_frame")},
        weight=1.0 / 3 * _DT_COMP)

    # 2. Grasped: binary +1 when grasped   [coef +1]
    env_cfg.rewards.grasped = RewTerm(
        func=sq_grasped,
        params={"object_cfg": SceneEntityCfg(target_cube),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "robot_cfg": SceneEntityCfg("robot")},
        weight=1.0 / 3 * _DT_COMP)

    # 3. Place: exp(-2*dist_to_rest) * grasped -- return arm to REST while holding   [coef +1]
    env_cfg.rewards.place = RewTerm(
        func=sq_place,
        params={"object_cfg": SceneEntityCfg(target_cube),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "robot_cfg": SceneEntityCfg("robot")},
        weight=1.0 / 3 * _DT_COMP)

    # 4. Table collision penalty   [coef -3]
    env_cfg.rewards.table_penalty = RewTerm(
        func=table_collision_penalty,
        params={"table_height": 0.04,
                "ee_frame_cfg": SceneEntityCfg("ee_frame")},
        weight=-3.0 / 3 * _DT_COMP)

    # 5. Not-lifted penalty (full Squint weight, ~5mm threshold)   [coef -1]
    env_cfg.rewards.not_lifted = RewTerm(
        func=sq_not_lifted,
        params={"object_cfg": SceneEntityCfg(target_cube)},
        weight=-1.0 / 3 * _DT_COMP)

    # 6. Pre-grasp wide-open shaping (gated ~grasped so it never fights the grasp). Small weight.
    env_cfg.rewards.open_approach = RewTerm(
        func=sq_open_approach,
        params={"object_cfg": SceneEntityCfg(target_cube),
                "ee_frame_cfg": SceneEntityCfg("ee_frame")},
        weight=0.15 / 3 * _DT_COMP)  # ~0.5 -- modest vs the +1 main terms

    # 7. Pre-grasp ALIGNMENT-HOLD (feedback v2: reach/align is poor). Reward GROWS the longer the OPEN
    #    jaws stay aligned ON the cube (up to ~3s = 30 steps), teaching the policy to seat the open mouth
    #    on the cube and settle before closing. Gated ~grasped + jaws-open so it never fights the grasp.
    #    Below the +1 main terms (grasp/place stay dominant -> it still closes, doesn't just hover).
    env_cfg.rewards.align_hold = RewTerm(
        func=sq_align_hold,
        params={"object_cfg": SceneEntityCfg(target_cube),
                "ee_frame_cfg": SceneEntityCfg("ee_frame")},
        weight=0.3 / 3 * _DT_COMP)   # modest: a non-farmable pre-grasp nudge; grasp/place (+1) dominate

    # 8. CLOSE-after-align (user idea): after the align-hold, reward closing the gripper -> bridges
    #    hold -> close -> grasp (the missing link). Gated latch + near + ~grasped (stops at grasp).
    env_cfg.rewards.close_align = RewTerm(
        func=sq_close_after_align,
        params={"object_cfg": SceneEntityCfg(target_cube),
                "ee_frame_cfg": SceneEntityCfg("ee_frame")},
        weight=0.4 / 3 * _DT_COMP)

    # (action_rate smoothness penalty DROPPED -- the HARD joint limits + elbow gate in arm_action already
    #  prevent the gross twisting/backward motion; residual fine jitter, if any, is handled at deploy by
    #  the EMA action filter in rebot_deploy.py. pan/wrist limits live in the action term, not here.)

    # Keep Squint's 5 + pre-grasp open (all bounded). Motion shaping is via the action-term hard limits.
    keep = {"reaching", "grasped", "place", "table_penalty", "not_lifted", "open_approach",
            "align_hold", "close_align"}
    for name in list(vars(env_cfg.rewards)):
        if not name.startswith("_") and name not in keep:
            setattr(env_cfg.rewards, name, None)

    return env_cfg


def apply_motor_limits(env, cfg):
    """Set PD gains — STS3215 real servo matching."""
    try:
        import omni.usd
        from pxr import UsdPhysics
        stage = omni.usd.get_context().get_stage()
        count = 0
        # reBot: keep the verified Damiao actuator gains from REBOT_FOLLOWER_CFG (do NOT override here).
        # (Previously forced STS3215 servo gains 20/2/3.5 on EVERY angular drive -- wrong for reBot.)
        try:
            print("[Motor] velocity: unlimited", flush=True)
        except Exception as ve:
            print(f"[Motor] velocity limit failed: {ve}", flush=True)
        if count > 0:
            print(f"[Motor] PD: stiffness=20 damping=2 force=3.5Nm STS3215-real ({count} joints)", flush=True)
    except Exception as e:
        print(f"[Motor] Error: {e}", flush=True)

    # Gripper friction = 2.0 (Squint-exact: SO101.py urdf_config)
    try:
        import omni.usd
        from pxr import UsdPhysics
        stage = omni.usd.get_context().get_stage()
        grip_count = 0
        for prim in stage.Traverse():
            name = prim.GetName().lower()
            if name in ("gripper_left", "gripper_right", "gripper_link"):
                mat = UsdPhysics.MaterialAPI.Get(stage, prim.GetPath())
                if not mat:
                    mat = UsdPhysics.MaterialAPI.Apply(prim)
                if mat:
                    mat.GetStaticFrictionAttr().Set(2.0)
                    mat.GetDynamicFrictionAttr().Set(2.0)
                    mat.GetRestitutionAttr().Set(0.0)
                    grip_count += 1
        print(f"[Gripper] friction=2.0 on {grip_count} prims (Squint-exact)", flush=True)
    except Exception as e:
        print(f"[Gripper] Friction error: {e}", flush=True)

    # Robot link mass readout
    try:
        import omni.usd
        from pxr import UsdPhysics
        stage = omni.usd.get_context().get_stage()
        total_robot_mass = 0.0
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if "/env_0/" not in path or "cube" in prim.GetName().lower():
                continue
            mass_api = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
            if mass_api:
                m = mass_api.GetMassAttr().Get()
                if m and m > 0:
                    total_robot_mass += m
                    print(f"[Robot] {prim.GetName():25s} mass={m:.4f} kg", flush=True)
        print(f"[Robot] Total robot mass (env_0): {total_robot_mass:.4f} kg", flush=True)
    except Exception as e:
        print(f"[Robot] Mass readout error: {e}", flush=True)

    # Cube physics: set mass=20g + friction (Squint ~density-200 scale; was 50g/18x heavier)
    try:
        import omni.usd
        from pxr import UsdPhysics
        stage = omni.usd.get_context().get_stage()
        cube_count = 0
        for prim in stage.Traverse():
            if "cube" in prim.GetName().lower():
                mass_api = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
                if not mass_api:
                    mass_api = UsdPhysics.MassAPI.Apply(prim)
                if mass_api:
                    mass_api.GetMassAttr().Set(0.02)  # 20g
                    cube_count += 1
                mat = UsdPhysics.MaterialAPI.Get(stage, prim.GetPath())
                if mat:
                    mat.GetStaticFrictionAttr().Set(0.3)
                    mat.GetDynamicFrictionAttr().Set(0.3)
                    mat.GetRestitutionAttr().Set(0.0)
        # Verify
        for prim in stage.Traverse():
            if "cube" in prim.GetName().lower():
                mass_api = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
                if mass_api:
                    m = mass_api.GetMassAttr().Get()
                    if "/env_0/" in str(prim.GetPath()):
                        print(f"[Cube] mass={m:.4f} kg ({m*1000:.1f}g) friction=0.3", flush=True)
                break
        print(f"[Cube] Set {cube_count} cubes to 20g", flush=True)
    except Exception as e:
        print(f"[Cube] Physics error: {e}", flush=True)
