"""Squint-exact reward for SO101 LiftCube."""

import os
import torch
import numpy as np
from isaaclab.managers import SceneEntityCfg

# Squint SO101 start keyframe
# reBot arm 6 joints [pan, lift, elbow, wrist_flex, wrist_yaw, wrist_roll]
REST_QPOS = torch.tensor([-1.5708, -1.30, -1.10, 0.75, 0.0, -0.10])  # CALIB e110 2026-06-22 (was old rest -> success impossible)
# Fold poses -- FK-VERIFIED on reBot (fk_sweep.py). The OLD values folded the arm DOWN
# (gripper z 0.22 -> -0.05, below the floor!) which fought the lift reward. Correct direction:
# shoulder_lift toward 0 + elbow MORE bent -> gripper rises and retracts OVER the base.
#   FOLD   (lift -1.00, elbow -1.60): gripper ~z 0.59  (intermediate, rising)
#   FOLDED (lift -0.80, elbow -2.10): gripper ~z 0.76, horiz 0.04 over base (lifted + folded in)
FOLD_QPOS = torch.tensor([0.0, -1.00, -1.60, 0.55, 0.0, 0.0])
FOLDED_QPOS = torch.tensor([0.0, -0.80, -2.10, 0.55, 0.0, 0.0])

# Cube resting height: 4cm table top (~0.04) + cube half-extent (0.02) => center rests at ~0.06 m.
# (The old not_lifted threshold assumed a table at z=0, so it never fired -- penalty was dead.)
CUBE_REST_Z = 0.06
GRASP_ZONE = 0.08   # PHASE3 (user): max ||cube-TCP|| for contact/grasp to COUNT -> inner jaw mouth + small margin (cube half=0.02)


def _get_target_qpos(env):
    """Retrieve the target joint positions from the action manager terms."""
    am = env.action_manager
    targets = []
    for term in am._terms.values():
        if hasattr(term, "_target_qpos"):
            targets.append(term._target_qpos)
    if targets:
        return torch.cat(targets, dim=-1)
    return env.scene["robot"].data.joint_pos


def _compute_angle_between(a, b):
    """Compute angle between two vector batches."""
    a_n = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_n = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return torch.acos((a_n * b_n).sum(dim=-1).clamp(-1, 1))


def _quat_to_y_axis(q):
    """Extract y-axis direction from quaternion (w,x,y,z format).

    Given a unit quaternion q = (w, x, y, z), the y-column of the
    corresponding rotation matrix is:
        [2(xy + wz),  1 - 2(x^2 + z^2),  2(yz - wx)]
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack(
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        dim=-1,
    )


def _get_link_body_idx(robot, link_name_substr):
    """Find body index whose name contains *link_name_substr* (case-insensitive)."""
    for i, name in enumerate(robot.data.body_names):
        if link_name_substr in name.lower():
            return i
    return None


def _cube_between_jaws(env, max_tcp_dist=0.05):
    """Geometric check: cube center is BETWEEN the two jaw tips.

    Uses dot product of (fixed_tip→cube) and (moving_tip→cube).
    If cube is between jaws, these vectors point in opposite directions → dot < 0.
    If robot is pressing from one side (tip pushing), both vectors point same way → dot > 0.
    """
    ee = env.scene["ee_frame"]
    fixed_tip = ee.data.target_pos_w[:, 0, :]   # (N, 3)
    moving_tip = ee.data.target_pos_w[:, 1, :]   # (N, 3)
    cube_pos = env.scene["cube"].data.root_pos_w  # (N, 3)

    vec_to_fixed = fixed_tip - cube_pos    # cube → fixed jaw
    vec_to_moving = moving_tip - cube_pos  # cube → moving jaw

    # Dot product < 0 means cube is between the two tips
    dot = (vec_to_fixed * vec_to_moving).sum(dim=-1)
    between = dot < 0

    # Also check cube is near TCP (not far away)
    tcp = (fixed_tip + moving_tip) * 0.5
    tcp_dist = (cube_pos - tcp).norm(dim=-1)
    near = tcp_dist < max_tcp_dist

    return between & near


def _is_grasped(env, object_cfg=SceneEntityCfg("cube"), min_force=0.3, max_angle_deg=110, align_tol=0.14,
                require_align=True):
    """Contact-based grasp, NOW ALSO REQUIRING ALIGNMENT (feedback v2 2026-06-11): BOTH fingers press the
    cube (force >= min_force) AND the cube is CENTERED at the grasp center (||cube - TCP|| < align_tol).
    So 'grasped' = a properly ALIGNED grasp, not an off-center edge-catch. This propagates to the grasp
    reward, the place/open gates, AND is_fold_success -> the policy must ALIGN to earn grasp credit and
    success, so it learns alignment (was the weak point). The gripper_contact/jaw_contact sensors are
    pre-filtered to the target cube's rigid body."""
    grip_force = env.scene["gripper_contact"].data.force_matrix_w[:, 0, 0, :]
    jaw_force = env.scene["jaw_contact"].data.force_matrix_w[:, 0, 0, :]
    grip_ok = grip_force.norm(dim=-1) >= min_force
    jaw_ok = jaw_force.norm(dim=-1) >= min_force
    return grip_ok & jaw_ok   # SQUINT-exact: BOTH jaws contact the cube => cube is BETWEEN them = inner grasp (no separate inner-pad/distance gate needed)
    if not require_align:
        return grip_ok & jaw_ok
    obj = env.scene[object_cfg.name]
    tcp = _tcp(env, SceneEntityCfg("ee_frame"))
    dist = torch.norm(obj.data.root_pos_w - tcp, dim=1)
    aligned = dist < align_tol                       # cube in the grasp zone (no off-center edge-catch)
    return grip_ok & jaw_ok & aligned                # STATELESS alignment gate (align-HOLD latch dropped --
    # reverted to Squint's pure-contact grasp + a one-line align-distance gate. The align_hold/close_align
    # reward terms and their per-env latch were our feedback-v2 overlay (now removed); align_tol=0.14 matches
    # the real ~12cm grasp distance (NOT the old impossible 0.035 that once caused 0 success).


# ======================================================================
# Dopamine RPE — reward decays as hit rate increases
# ======================================================================
_hit_rates = {}
_DOPAMINE_ALPHA = 0.3   # decay exponent (moderate)
_DOPAMINE_LAMBDA = 0.3  # floor — reward never drops below 30%

def _dopamine(key, raw_reward):
    """Apply dopamine: R_eff = R_raw × [λ + (1-λ)(1-h)^α]."""
    if key not in _hit_rates:
        _hit_rates[key] = 0.0
    h_now = (raw_reward > 0.001).float().mean().item()
    _hit_rates[key] = 0.99 * _hit_rates[key] + 0.01 * h_now  # EMA
    h = _hit_rates[key]
    mult = _DOPAMINE_LAMBDA + (1 - _DOPAMINE_LAMBDA) * (1 - h) ** _DOPAMINE_ALPHA
    return raw_reward   # DOPAMINE DISABLED on ALL stages 2026-06-23 (Squint has none; decay shrank grasp reward as it was learned -> grasp vanished)


# ======================================================================
# Reward terms
# ======================================================================


# Retry tracking: detect close→open→close cycle (grasp attempt)
_was_close_to_cube = {}
_was_jaw_closed = {}
_retry_count = {}

def grasp_retry_reward(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """One-time reward each time agent re-opens jaw after failed grasp attempt near cube.
    Encourages: approach → close (fail) → open → approach again → close."""
    n = env.num_envs
    device = env.device
    key = "retry"
    if key not in _retry_count or _retry_count[key].shape[0] != n:
        _was_close_to_cube[key] = torch.zeros(n, dtype=torch.bool, device=device)
        _was_jaw_closed[key] = torch.zeros(n, dtype=torch.bool, device=device)
        _retry_count[key] = torch.zeros(n, dtype=torch.long, device=device)

    # Reset on new episodes
    just_reset = env.episode_length_buf <= 1
    _was_close_to_cube[key][just_reset] = False
    _was_jaw_closed[key][just_reset] = False
    _retry_count[key][just_reset] = 0

    # Current state
    obj = env.scene[object_cfg.name]
    ee = env.scene[ee_frame_cfg.name]
    tcp = (ee.data.target_pos_w[:, 0, :] + ee.data.target_pos_w[:, 1, :]) * 0.5
    dist = torch.norm(obj.data.root_pos_w - tcp, dim=1)
    close = dist < 0.10
    _r = env.scene["robot"]; _gi = _r.data.joint_names.index("gripper")
    gripper_qpos = _r.data.joint_pos[:, _gi]   # prismatic, meters (stroke 0~0.0715)
    # reBot pads CONVERGE at HIGH qpos (joint value goes UP to close the fingers), so the
    # open/closed sense is INVERTED vs a naive "small=closed": gripping = qpos>0.05, released = qpos<0.035.
    jaw_closed = gripper_qpos > 0.05    # pads gripping
    jaw_open = gripper_qpos < 0.035     # pads apart (released)

    # Detect retry: was close+closed (attempt), now open again (retry)
    was_attempting = _was_close_to_cube[key] & _was_jaw_closed[key]
    retrying = was_attempting & jaw_open & close
    _retry_count[key][retrying] += 1

    # Update state
    _was_close_to_cube[key] = close
    _was_jaw_closed[key] = jaw_closed

    return retrying.float()


def reaching_reward(
    env,
    object_cfg=SceneEntityCfg("cube"),
    ee_frame_cfg=SceneEntityCfg("ee_frame"),
):
    """Dense reaching reward: 1 - tanh(5 * dist(tcp, cube))."""
    obj = env.scene[object_cfg.name]
    ee = env.scene[ee_frame_cfg.name]
    tcp = (ee.data.target_pos_w[:, 0, :] + ee.data.target_pos_w[:, 1, :]) * 0.5
    dist = torch.norm(obj.data.root_pos_w - tcp, dim=1)
    base = 1 - torch.tanh(5 * dist)
    # Only reward reaching when the gripper is NOT flipped -- its mouth must point DOWN, not up.
    # gripper_link local +z points toward the grasp; its world z-component < 0 means pointing down.
    robot = env.scene["robot"]
    q = robot.data.body_quat_w[:, robot.data.body_names.index("gripper_link"), :]
    zc = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)    # gripper_link local +z, world z-component
    upright = torch.clamp(-zc, min=0.0, max=1.0)       # 1=straight down, 0=horizontal/flipped up
    gate = _pan_face_ok(env, object_cfg).float()       # GATE: only reward reaching when pan-aligned
    return _dopamine("reaching", base * upright * gate)


# ── Stage-gate helpers: a stage's reward is given ONLY when the PREVIOUS stage's condition holds ──
def _tcp(env, ee_frame_cfg):
    ee = env.scene[ee_frame_cfg.name]
    return (ee.data.target_pos_w[:, 0, :] + ee.data.target_pos_w[:, 1, :]) * 0.5

def _gq(env):
    r = env.scene["robot"]
    return r.data.joint_pos[:, r.data.joint_names.index("gripper")]

def _gripper_down(env):
    r = env.scene["robot"]
    q = r.data.body_quat_w[:, r.data.body_names.index("gripper_link"), :]
    return (1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)) < 0.0       # gripper mouth points DOWN (not flipped)

def _pan_error(env, object_cfg=SceneEntityCfg("cube")):
    """|shoulder_pan - required_pan| (wrapped to [0,π]). required_pan points the arm (pan=0 => +Y
    world) at the cube. JOINT-based, so it is NOT fooled by a folded/flipped arm whose gripper happens
    to land at the cube's azimuth (which the old base->gripper measure mistook for 'aligned')."""
    robot = env.scene["robot"]; obj = env.scene[object_cfg.name]
    base = robot.data.body_pos_w[:, robot.data.body_names.index("base_link"), :2]
    dx = obj.data.root_pos_w[:, 0] - base[:, 0]
    dy = obj.data.root_pos_w[:, 1] - base[:, 1]
    req = torch.atan2(-dx, dy)
    pan = robot.data.joint_pos[:, robot.data.joint_names.index("shoulder_pan")]
    e = pan - req
    return torch.atan2(torch.sin(e), torch.cos(e)).abs()

def _pan_face_ok(env, object_cfg=SceneEntityCfg("cube")):
    """STAGE 0 condition: pan faces the cube (joint-based) AND finger axis matches a cube face."""
    robot = env.scene["robot"]; obj = env.scene[object_cfg.name]; bn = robot.data.body_names
    facing = _pan_error(env, object_cfg) < 0.26
    gl = robot.data.body_pos_w[:, bn.index("gripper_left"), :2]
    gr = robot.data.body_pos_w[:, bn.index("gripper_right"), :2]
    faxis = gr - gl; faxis = faxis / (faxis.norm(dim=-1, keepdim=True) + 1e-6)
    q = obj.data.root_quat_w
    cx = torch.stack([1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2), 2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])], dim=-1)
    cx = cx / (cx.norm(dim=-1, keepdim=True) + 1e-6)
    dot = (faxis * cx).sum(-1).abs().clamp(max=1.0)
    s2 = 2.0 * dot * torch.sqrt((1.0 - dot * dot).clamp(min=0.0))
    return facing & (s2 < 0.40)

def _reach_ok(env, object_cfg, ee_frame_cfg):       # STAGE 1: tcp near cube + pointing down
    d = torch.norm(env.scene[object_cfg.name].data.root_pos_w - _tcp(env, ee_frame_cfg), dim=1)
    return (d < 0.12) & _gripper_down(env)

def _approach_ok(env, object_cfg, ee_frame_cfg):    # STAGE 2: closer + jaw open
    d = torch.norm(env.scene[object_cfg.name].data.root_pos_w - _tcp(env, ee_frame_cfg), dim=1)
    return (d < 0.08) & (_gq(env) < 0.05)

def _near_mouth(env, object_cfg, ee_frame_cfg):     # STAGE 3: cube at the grasp center (gripper-agnostic)
    d = torch.norm(env.scene[object_cfg.name].data.root_pos_w - _tcp(env, ee_frame_cfg), dim=1)
    return d < 0.05


_pan_held = [None]

def pan_aligned_reward(env, object_cfg=SceneEntityCfg("cube")):
    """STAGE 0 (before reaching): shoulder_pan turns the arm to FACE the target cube -- the
    base->gripper azimuth is aligned with the base->cube azimuth (within ~15deg). Reward GROWS the
    longer it's held facing (up to ~10 s = the full episode) and RESETS when alignment is lost."""
    n = env.num_envs
    device = env.device
    if _pan_held[0] is None or _pan_held[0].shape[0] != n:
        _pan_held[0] = torch.zeros(n, dtype=torch.long, device=device)
    just_reset = env.episode_length_buf <= 1
    _pan_held[0][just_reset] = 0

    aligned = _pan_face_ok(env, object_cfg)        # JOINT-based pan facing + finger-axis matches cube face

    _pan_held[0][aligned] += 1
    _pan_held[0][~aligned] = 0
    reward = aligned.float() * (_pan_held[0].float() / 50.0).clamp(max=2.0)  # 5s->1.0, 10s->2.0
    return _dopamine("pan_aligned", reward)


def pan_flip_penalty(env, object_cfg=SceneEntityCfg("cube")):
    """Penalty when the base (pan) is rotated AWAY from the cube -- discourages over-rotating /
    flipping the base around. 0 while roughly facing the cube, ramps to 1 when facing backward."""
    angle = _pan_error(env, object_cfg)            # JOINT-based pan error (robust to fold/flip)
    return torch.clamp((angle - 1.0) / 1.5, min=0.0, max=1.0)   # 0 below ~57deg, ->1 at ~143deg away


_prev_phi = [None]

def toward_cube_reward(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """Dense PROGRESS reward (potential-based): reward pan rotating toward facing the cube AND the
    gripper getting closer. phi = pan_error + 3*tcp_dist (lower=better); reward = prev_phi - cur_phi
    (toward cube = +, away = -; telescopes so it can't be farmed). UNGATED -> guides pan/lift from
    the very first step (fixes 'pan/lift go the opposite way' during early exploration)."""
    n = env.num_envs
    pe = _pan_error(env, object_cfg)
    d = torch.norm(env.scene[object_cfg.name].data.root_pos_w - _tcp(env, ee_frame_cfg), dim=1)
    phi = pe + 3.0 * d
    if _prev_phi[0] is None or _prev_phi[0].shape[0] != n:
        _prev_phi[0] = phi.clone()
    just_reset = env.episode_length_buf <= 1
    _prev_phi[0][just_reset] = phi[just_reset]
    reward = (_prev_phi[0] - phi).clamp(-1.0, 1.0)
    _prev_phi[0] = phi.clone()
    return reward


_aligned_held = [None]
_grasp_held_r = [None]

def grasp_hold_reward(env, object_cfg=SceneEntityCfg("cube")):
    """HOLD-PROPORTIONAL grasp bonus (2026-06-23): reward GROWS with consecutive grasped steps (0 -> 1.0 over 20
    steps) and RESETS the moment the grasp breaks -> incentivizes MAINTAINING the grasp through lift + return-to-rest,
    not just a brief touch. (not_lifted penalty still forces it to LIFT, so this can't be farmed on the table.)"""
    import torch as _t
    n = env.num_envs; device = env.device
    if _grasp_held_r[0] is None or _grasp_held_r[0].shape[0] != n:
        _grasp_held_r[0] = _t.zeros(n, dtype=_t.long, device=device)
    just_reset = env.episode_length_buf <= 1
    _grasp_held_r[0][just_reset] = 0
    g = _is_grasped(env, object_cfg, min_force=0.3, require_align=False)
    _grasp_held_r[0][g] += 1
    _grasp_held_r[0][~g] = 0
    return (_grasp_held_r[0].float() / 20.0).clamp(max=1.0)


def aligned_reward(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """Pre-grasp ALIGNED stage: the target cube sits INSIDE the open gripper mouth (at the TCP /
    grasp center). Reward GROWS the longer it is held aligned -- 3 s (=30 steps) -> 1.0, up to 2.0 --
    and RESETS to 0 the moment the cube leaves the mouth or the gripper closes. Bridges reaching->grasp."""
    n = env.num_envs
    device = env.device
    if _aligned_held[0] is None or _aligned_held[0].shape[0] != n:
        _aligned_held[0] = torch.zeros(n, dtype=torch.long, device=device)
    just_reset = env.episode_length_buf <= 1
    _aligned_held[0][just_reset] = 0

    obj = env.scene[object_cfg.name]
    ee = env.scene[ee_frame_cfg.name]
    tcp = (ee.data.target_pos_w[:, 0, :] + ee.data.target_pos_w[:, 1, :]) * 0.5
    dist = torch.norm(obj.data.root_pos_w - tcp, dim=1)
    _r = env.scene["robot"]; _gi = _r.data.joint_names.index("gripper")
    gripper_open = _r.data.joint_pos[:, _gi] < 0.05            # pads apart -> cube fits in the mouth
    aligned = (dist < 0.04) & gripper_open & _approach_ok(env, object_cfg, ee_frame_cfg)  # gated by approach

    _aligned_held[0][aligned] += 1
    _aligned_held[0][~aligned] = 0                              # reset when it leaves the mouth
    reward = aligned.float() * (_aligned_held[0].float() / 30.0).clamp(max=2.0)  # 1.0 @3s, 2.0 @6s+
    return _dopamine("aligned", reward)


_approach_open_fired = {}

def approach_open_reward(
    env,
    object_cfg=SceneEntityCfg("cube"),
    ee_frame_cfg=SceneEntityCfg("ee_frame"),
):
    """One-time reward for first reaching cube with jaw open."""
    n = env.num_envs
    device = env.device
    key = "approach_open"
    if key not in _approach_open_fired or _approach_open_fired[key].shape[0] != n:
        _approach_open_fired[key] = torch.zeros(n, dtype=torch.bool, device=device)

    just_reset = env.episode_length_buf <= 1
    _approach_open_fired[key][just_reset] = False

    obj = env.scene[object_cfg.name]
    ee = env.scene[ee_frame_cfg.name]
    tcp = (ee.data.target_pos_w[:, 0, :] + ee.data.target_pos_w[:, 1, :]) * 0.5
    dist = torch.norm(obj.data.root_pos_w - tcp, dim=1)

    close = dist < 0.10
    _r = env.scene["robot"]; _gi = _r.data.joint_names.index("gripper")
    gripper_qpos = _r.data.joint_pos[:, _gi]   # prismatic, meters
    jaw_open = gripper_qpos < 0.035  # pads APART (reBot pads converge at HIGH qpos -> "open" is LOW qpos)

    condition = close & jaw_open & _reach_ok(env, object_cfg, ee_frame_cfg) & ~_approach_open_fired[key]
    _approach_open_fired[key] |= condition

    return condition.float()


def grasped_reward(
    env,
    object_cfg=SceneEntityCfg("cube"),
    ee_frame_cfg=SceneEntityCfg("ee_frame"),
    robot_cfg=SceneEntityCfg("robot"),
):
    """Binary: 1 when grasping, 0 otherwise."""
    # GATE: grasp only counts when the cube is at the grasp center (came through the aligned stage)
    g = _is_grasped(env, object_cfg, min_force=0.3) & _near_mouth(env, object_cfg, ee_frame_cfg)
    return _dopamine("grasped", g.float())


def place_reward(
    env,
    object_cfg=SceneEntityCfg("cube"),
    ee_frame_cfg=SceneEntityCfg("ee_frame"),
    robot_cfg=SceneEntityCfg("robot"),
):
    """DISABLED — replaced by fold + fold_hold."""
    return torch.zeros(env.num_envs, device=env.device)


def not_lifted_penalty(env, object_cfg=SceneEntityCfg("cube")):
    """Penalty while the cube stays on the table (i.e. near its resting height ~0.06 m).
    Old threshold (0.016) assumed table at z=0, so it was always 'lifted' -> penalty never fired."""
    obj = env.scene[object_cfg.name]
    cube_z = obj.data.root_pos_w[:, 2]
    item_lifted = cube_z >= (CUBE_REST_Z + 0.02)   # clearly off the table (>2cm above rest)
    return (~item_lifted).float()


def lift_hold_reward(env, object_cfg=SceneEntityCfg("cube")):
    """Reward for lifting cube 15cm + grasped (no fold requirement)."""
    obj = env.scene[object_cfg.name]
    cube_z = obj.data.root_pos_w[:, 2]
    lifted = (cube_z >= 0.15).float()
    grasped = _is_grasped(env, object_cfg, min_force=0.3).float()
    return _dopamine("lift_hold", lifted * grasped)


def _fold_3joint(env):
    """3-joint fold errors: pan(0), shoulder_lift(1), elbow(2)."""
    tgt = _get_target_qpos(env)[:, :-1]  # (N, 5)
    folded = FOLDED_QPOS[:-1].to(tgt.device)
    errs = (tgt - folded).abs()
    return errs[:, 0], errs[:, 1], errs[:, 2]  # pan, lift, elbow


def fold_reward(env, object_cfg=SceneEntityCfg("cube")):
    """3-joint continuous fold: pan + shoulder_lift + elbow. All always-on."""
    obj = env.scene[object_cfg.name]
    cube_z = obj.data.root_pos_w[:, 2]
    lifted = (cube_z >= 0.15).float()
    grasped = _is_grasped(env, object_cfg, min_force=0.3).float()
    pan_err, lift_err, elbow_err = _fold_3joint(env)

    s_pan = torch.exp(-2.0 * pan_err)        # pan (0° target)
    s_lift = torch.exp(-0.5 * lift_err)       # shoulder_lift (-90° target, big range)
    s_elbow = torch.exp(-0.5 * elbow_err)     # elbow (150° target, big range)

    return _dopamine("fold", lifted * grasped * (s_pan + s_lift + s_elbow) / 3.0)


_fh_first = [None]   # step when fold first achieved per env
_fh_held = [None]    # cumulative steps held since first fold

def fold_hold_reward(env, object_cfg=SceneEntityCfg("cube")):
    """Fold hold: pan<5° AND lift<5° AND elbow<5°, held 70%→save, 95%→big reward."""
    n = env.num_envs
    device = env.device
    if _fh_first[0] is None or _fh_first[0].shape[0] != n:
        _fh_first[0] = torch.full((n,), -1, dtype=torch.long, device=device)
        _fh_held[0] = torch.zeros(n, dtype=torch.long, device=device)

    just_reset = env.episode_length_buf <= 1
    _fh_first[0][just_reset] = -1
    _fh_held[0][just_reset] = 0

    obj = env.scene[object_cfg.name]
    cube_z = obj.data.root_pos_w[:, 2]
    lifted = (cube_z >= 0.15)
    grasped = _is_grasped(env, object_cfg, min_force=0.3)
    pan_err = _fold_3joint(env)[0]
    holding = lifted & grasped & (pan_err < 0.262)  # pan<15°

    first_time = holding & (_fh_first[0] < 0)
    _fh_first[0][first_time] = env.episode_length_buf[first_time]
    _fh_held[0][holding] += 1

    started = _fh_first[0] >= 0
    elapsed = (env.episode_length_buf - _fh_first[0]).clamp(min=1)
    ratio = (_fh_held[0].float() / elapsed.float()).clamp(max=1.0)

    above_50 = started & (ratio >= 0.50) & (_fh_held[0] >= 5)
    reward = torch.zeros(n, device=device)
    held_frac = (_fh_held[0].float() / env.max_episode_length).clamp(max=1.0)
    reward[above_50] = held_frac[above_50] ** 2
    reward[started & ~above_50] = ratio[started & ~above_50] * 1.0

    return _dopamine("fold_hold", reward)


# ════════════════════════════════════════════════════════════════════════════════
#  SQUINT-MATCHED reward + success (envs/lift.py compute_dense_reward + evaluate).
#  Squint solves SO101 LiftCube to ~100% in ~0.2-0.3M samples with just these:
#    reward = (1-tanh(5d)) + is_grasped + exp(-2*dist_to_rest)*is_grasped
#             - 3*touching_table - 1*not_lifted,   all /3 (normalized_dense).
#  success = lifted(~off table) & grasped & arm-target back near REST (<0.2).
#  No dopamine, no chained gates, no 15cm bar, no FOLDED pose. Stationary + ungated.
# ════════════════════════════════════════════════════════════════════════════════
def _dist_to_rest(env):
    """||commanded arm target qpos - REST_QPOS|| over the 6 arm joints (Squint reached_rest_qpos).
    Uses the COMMANDED target (free at spawn), exactly like Squint's controller._target_qpos."""
    # SQUINT-EXACT (per user): reached_rest uses the COMMANDED arm target (controller._target_qpos), NOT actual qpos
    # -> the setpoint near REST is achievable (the reBot actual-qpos change made success too strict = a likely cause of success=0).
    q = env.action_manager.get_term("arm_action")._target_qpos
    return torch.norm(q - REST_QPOS.to(q.device), dim=1)


# DENSE-reward DECAY (neuro: dopamine fades as a skill is mastered). The trainer sets _DENSE_SCALE[0] from the
# success rate (high sr -> ~0 dense -> only outcome reward; sr drops -> dense returns = homeostatic, self-correcting).
# Default 1.0 -> NO effect for other trainers/teachers. Multiplies the FARMABLE dense terms (reaching + align only).
_DENSE_SCALE = [1.0]

# REAL-SENSOR reward toggle (per-process; the trainer sets MR._REAL_SENSOR[0] = args.real_sensor_reward).
# False (default) = ORIGINAL HYBRID reward (privileged physics + vision/current/position).
# True = real-robot-measurable only (vision / current / joint qpos); physics removed (the asym critic carries it).
_REAL_SENSOR = [False]

def sq_reaching(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """Squint reaching: 1 - tanh(5 * dist(tcp, cube)). Ungated, stationary. Scaled by _DENSE_SCALE (decay)."""
    obj = env.scene[object_cfg.name]
    tcp = _tcp(env, ee_frame_cfg)
    dist = torch.norm(obj.data.root_pos_w - tcp, dim=1)
    return (1.0 - torch.tanh(5.0 * dist)) * _DENSE_SCALE[0]


# ── OBSERVABLE align reward (sim-real-consistent) ─────────────────────────────
# The cube renders RED; reward the red blob being CENTERED in the eye-in-hand WRIST view (= lateral grasp
# alignment), computed from the SAME raw rendered image the policy consumes -> reward and perception are
# coupled (visual servoing). Measurable IDENTICALLY on the real robot (HSV-red centroid in the real wrist
# frame), so it does not exploit privileged sim ground-truth. (DR is applied downstream in the trainer's
# read_imgs, NOT to env.scene[..].output, so the cube is its true red here -- matches real.)
WRIST_ALIGN_PX = (0.5, 0.80)   # target (x,y): grasp point = bottom-center of upright wrist view (between the fingers)

def _red_centroid(rgb):
    """rgb (N,H,W,3) uint8 -> (cx,cy in [0,1], visible[N]). Lighting-robust red mask: R high & R>>G,B."""
    r = rgb[..., 0].float(); g = rgb[..., 1].float(); b = rgb[..., 2].float()
    red = (r > 100) & (r > 1.5 * g) & (r > 1.5 * b)
    N, H, W = red.shape
    cnt = red.flatten(1).sum(1)
    ys = torch.arange(H, device=rgb.device).view(1, H, 1).float()
    xs = torch.arange(W, device=rgb.device).view(1, 1, W).float()
    denom = cnt.clamp(min=1.0)
    cx = (red * xs).flatten(1).sum(1) / denom / (W - 1)
    cy = (red * ys).flatten(1).sum(1) / denom / (H - 1)
    return cx, cy, (cnt >= 5).float(), cnt / float(H * W)   # + red-area fraction (proximity proxy)

def align_wrist_centroid(env, cam_name="wrist", object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """align. _REAL_SENSOR -> VISION ONLY (cube centered on the grasp pixel x size in the WRIST view). else HYBRID
    (privileged physics 1-tanh(12d) x vision size = the ORIGINAL). x _DENSE_SCALE. (Name kept.)"""
    if _REAL_SENSOR[0]:
        try:
            cx, cy, vis, frac = _red_centroid(env.scene[cam_name].data.output["rgb"])
        except Exception:
            return torch.zeros(env.num_envs, device=env.device)
        tx, ty = WRIST_ALIGN_PX
        d = torch.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)
        centered = (1.0 - 2.0 * d).clamp(min=0.0)            # 1 at the grasp pixel, 0 half-frame away
        size = torch.tanh(frac * 15.0)                       # bigger red blob = closer = more grasp-ready
        return torch.nan_to_num(vis * centered * size * _DENSE_SCALE[0], nan=0.0, posinf=1.0, neginf=0.0)
    obj = env.scene[object_cfg.name]
    tcp = _tcp(env, ee_frame_cfg)
    dist = torch.norm(obj.data.root_pos_w - tcp, dim=1)
    phys = 1.0 - torch.tanh(12.0 * dist)                     # PHYSICS: grasp-readiness (privileged)
    try:
        _, _, vis, frac = _red_centroid(env.scene[cam_name].data.output["rgb"])
        size = vis * (0.3 + 0.7 * torch.tanh(frac * 15.0))   # VISION: red-area size
    except Exception:
        size = torch.ones_like(dist)
    return torch.nan_to_num(phys * size * _DENSE_SCALE[0], nan=0.0, posinf=1.0, neginf=0.0)


def reach_sharp(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """reach. _REAL_SENSOR -> VISION ONLY (wrist red-area, real-measurable). else HYBRID (privileged physics
    1-tanh(8d) + near-bonus + observable wrist red-area = the ORIGINAL setup). x _DENSE_SCALE (decay)."""
    if _REAL_SENSOR[0]:
        try:
            _, _, _, frac = _red_centroid(env.scene["wrist"].data.output["rgb"])
            wrist_prox = torch.tanh(frac * 15.0)
        except Exception:
            wrist_prox = torch.zeros(env.num_envs, device=env.device)
        return torch.nan_to_num(wrist_prox * _DENSE_SCALE[0], nan=0.0, posinf=1.0, neginf=0.0)
    obj = env.scene[object_cfg.name]; tcp = _tcp(env, ee_frame_cfg)
    dist = torch.norm(obj.data.root_pos_w - tcp, dim=1)
    coarse = 1.0 - torch.tanh(8.0 * dist)
    near_bonus = (dist < 0.06).float() * 0.5
    try:
        _, _, _, frac = _red_centroid(env.scene["wrist"].data.output["rgb"])
        wrist_prox = torch.tanh(frac * 15.0)
    except Exception:
        wrist_prox = torch.zeros_like(dist)
    out = (coarse + near_bonus + 0.5 * wrist_prox) * _DENSE_SCALE[0]
    return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)


GATE_W = 1.0   # multiplicative align-gating strength: reach * exp(GATE_W * align)
def reach_align_gated(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """Frontal/OFC GATING (alternative to additive reach+align): the SHARP reach BOOSTED multiplicatively by the
    wrist-align (visual servoing). reach_sharp * exp(GATE_W*align). align=0 (cube not in wrist view) -> exp(0)=1 ->
    reach SURVIVES (no dead zone); align high -> big boost -> alignment-first, can't farm reach alone."""
    base = reach_sharp(env, object_cfg, ee_frame_cfg)
    al = align_wrist_centroid(env)
    return base * torch.exp(GATE_W * al)


def sq_grasped(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame"),
               robot_cfg=SceneEntityCfg("robot")):
    """grasp. _REAL_SENSOR -> CURRENT x POSITION (real). else HYBRID = CURRENT x PHYSICS x POSITION (the ORIGINAL;
    physics 1-tanh(8d) = cube actually at the grasp center). CURRENT=both-finger force (real=motor current),
    POSITION=gripper qpos not fully closed (>=0.02, real=encoder)."""
    return _is_grasped(env, object_cfg, min_force=0.3, require_align=False).float()   # +dopamine PURE contact grasp (no align gate -> matches Squint is_grasping + our success metric)


HOLD_STEPS = 5      # SQUINT-MATCH 2026-06-23: place/lift credit only a grasp MAINTAINED this many consecutive steps
_grasp_hold = [None]
_held_cache = [None, -1]
def _held(env, object_cfg=SceneEntityCfg("cube")):
    """Squint HOLD_STEPS gate: consecutive _is_grasped steps (resets the moment the grasp breaks). Memoized per
    env step (common_step_counter) so place & lift share ONE counter update regardless of reward-term order/count."""
    sid = int(getattr(env, "common_step_counter", -1))
    if _held_cache[1] == sid and _held_cache[0] is not None and _held_cache[0].shape[0] == env.num_envs:
        return _held_cache[0]
    g = _is_grasped(env, object_cfg, min_force=0.3, require_align=False)
    if _grasp_hold[0] is None or _grasp_hold[0].shape[0] != env.num_envs:
        _grasp_hold[0] = torch.zeros(env.num_envs, device=g.device)
    _grasp_hold[0] = torch.where(g, _grasp_hold[0] + 1.0, torch.zeros_like(_grasp_hold[0]))
    held = (_grasp_hold[0] >= HOLD_STEPS).float()
    _held_cache[0], _held_cache[1] = held, sid
    return held


def sq_place(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame"),
             robot_cfg=SceneEntityCfg("robot")):
    """place = JOINT x grasp. _REAL_SENSOR -> grasp = CURRENT+POSITION (real). else HYBRID grasp = _is_grasped
    (force + privileged physics align) = the ORIGINAL. exp(-0.5*dist_to_rest) * grasp. (dist_to_rest = commanded
    arm target qpos = real-measurable; gentler decay so the ~4rad return is learnable.)"""
    # SQUINT-EXACT (per user): exp(-0.5*dist_to_rest) * is_grasped (instantaneous). Dropped reBot held+item_lifted gates.
    grasped = _is_grasped(env, object_cfg, min_force=0.3, require_align=False).float()
    return torch.exp(-0.5 * _dist_to_rest(env)) * grasped


# ── FOLD CURRICULUM — ported from the author's earlier SO-101 project (the WORKING reference) 2026-06-25.
#    The reference taught the fold via lift_hold + a 3-joint graded fold + a sustained fold_hold; this delivery
#    had collapsed all that into a single weak `place` term -> teacher never learned the fold -> success=0.
#    Adapted to THIS delivery's REST_QPOS as the fold target (success here = reached_rest toward REST).
LIFT_GATE = 0.065      # cube z(m) counted as 'lifted' for the fold reward (2cm above the 0.06 rest; achievable).
FOLD_FRAMES = 3       # consecutive in-band frames before the fold bonus starts paying (user: "fold at 3 frames").
FOLD_TOL = 0.2        # rad: each of the 3 fold joints (pan/shoulder_lift/elbow) within this of REST = "folded".
#  USER 2026-06-25: success/fold use ONLY these 3 joints (wrists FREE) — the full-6-joint reached_rest was too
#  hard (wrists rarely return to REST). "when pan, shoulder_lift and elbow are all folded" = folded.

def _fold3_err(env):
    """(N,3) abs error of the 3 fold joints [pan(0), shoulder_lift(1), elbow(2)] vs REST (commanded target)."""
    q = env.action_manager.get_term("arm_action")._target_qpos
    return (q[:, :3] - REST_QPOS[:3].to(q.device)).abs()

_fold_streak = [None]
def fold_reward(env, object_cfg=SceneEntityCfg("cube")):
    """FOLD = return the LIFTED+GRASPED arm to REST over ALL 6 joints (== the success reached_rest metric).
    FIX 2026-06-25: the old fold shaped only 3 joints (pan/shoulder/elbow) while success checks all 6 -> the
    wrists never folded -> reached_rest stuck 0 -> success 0. Now uses the full 6-joint dist (== reached_rest).
    Dense graded pull toward REST + a STREAK bonus that needs >=FOLD_FRAMES consecutive in-band frames and GROWS
    the longer it's sustained (replaces fold_hold; user design). lift_hold removed (was farmable hover)."""
    n, device = env.num_envs, env.device
    if _fold_streak[0] is None or _fold_streak[0].shape[0] != n:
        _fold_streak[0] = torch.zeros(n, dtype=torch.long, device=device)
    _fold_streak[0][env.episode_length_buf <= 1] = 0
    cube_z = env.scene[object_cfg.name].data.root_pos_w[:, 2]
    held = (cube_z >= LIFT_GATE) & _is_grasped(env, object_cfg, min_force=0.3, require_align=False)
    e3 = _fold3_err(env)                                      # (N,3) pan/shoulder_lift/elbow err vs REST
    # GENTLE decay (ref-style) so there's gradient even when FAR from REST (at the grasp pose) -> pulls the arm
    # all the way to the fold. exp(-3) was too steep (~0 far away -> no fold signal -> policy stuck at lift+grasp).
    closeness = (torch.exp(-2.0 * e3[:, 0]) + torch.exp(-0.5 * e3[:, 1]) + torch.exp(-0.5 * e3[:, 2])) / 3.0
    in_band = held & (e3 < FOLD_TOL).all(dim=1)              # all 3 fold joints folded (wrists FREE) = success band
    _fold_streak[0] = torch.where(in_band, _fold_streak[0] + 1, torch.zeros_like(_fold_streak[0]))
    grow = (_fold_streak[0].float() / 20.0).clamp(max=1.0)    # 0->1 over ~20 held frames -> "the longer it is held, the more it grows"
    bonus = torch.where(_fold_streak[0] >= FOLD_FRAMES, grow, torch.zeros(n, device=device))
    return held.float() * closeness + bonus                  # dense approach pull (all 6 joints) + growing fold bonus


LIFT_FULL = 0.15      # lift reward saturates(=1.0) at +15cm above rest = cube_z 0.21 (user: "15 is full"; ~max reach).
SUCC_LIFT = 0.13      # SUCCESS = cube_z >= 0.13 (+7cm above rest) & grasped, held FOLD_FRAMES (user: "success is 7").
def lift_reward(env, object_cfg=SceneEntityCfg("cube")):
    """GRADED lift (user 2026-06-25, re-added): reward GROWS with cube HEIGHT above rest (higher = more, capped
    at LIFT_FULL), gated by grasped -> 'the higher it lifts, the higher the score'. Unlike the old flat lift_hold, this pays more the
    higher the held cube goes, so the policy actually lifts (not just clears the gate). fold(2/3) still dominates,
    so this is a stepping stone, not a hover trap."""
    cube_z = env.scene[object_cfg.name].data.root_pos_w[:, 2]
    grasped = _is_grasped(env, object_cfg, min_force=0.3, require_align=False).float()
    lift_h = ((cube_z - CUBE_REST_Z) / LIFT_FULL).clamp(0.0, 1.0)   # 0 at rest(0.06) -> 1.0 at +LIFT_FULL
    return lift_h * grasped


def success_reward(env, object_cfg=SceneEntityCfg("cube")):
    """OUTCOME bonus (user 2026-06-25): +1.0 every step the SUCCESS state holds (cube_z >= SUCC_LIFT & grasped).
    The lift reward is graded toward height; this is a flat bonus AT the success threshold so reaching+holding the
    success height is directly rewarded (reinforces the 3-frame success latch). NOTE: instantaneous condition (not
    the streak) so it gives dense signal; it does not mutate the success streak counter."""
    cube_z = env.scene[object_cfg.name].data.root_pos_w[:, 2]
    grasped = _is_grasped(env, object_cfg, min_force=0.3, require_align=False).float()
    return (cube_z >= SUCC_LIFT).float() * grasped


def sq_lift(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """LIFT reward: reward raising the cube off the table WHILE gripping it -- the un-farmable success requirement
    the real-sensor rewards miss (grasp+place could be farmed with a force-touch+return WITHOUT lifting). Uses
    cube_z (PRIVILEGED, sim-only -- the deployed actor never sees it; only the reward does). Graded 0..1 over a
    5cm lift, x grasped(force) so it only pays for a lift that's actually held."""
    cube_z = env.scene[object_cfg.name].data.root_pos_w[:, 2]
    gf = env.scene["gripper_contact"].data.force_matrix_w[:, 0, 0, :].norm(dim=-1)
    jf = env.scene["jaw_contact"].data.force_matrix_w[:, 0, 0, :].norm(dim=-1)
    dist = torch.norm(env.scene[object_cfg.name].data.root_pos_w - _tcp(env, ee_frame_cfg), dim=1)
    held = _held(env, object_cfg)  # SQUINT-MATCH 2026-06-23: lift * held (grasp maintained >= HOLD_STEPS)  # was instantaneous
    height = ((cube_z - CUBE_REST_Z) / 0.08).clamp(0.0, 1.0)   # SQUINT-MATCH 2026-06-23: full lift at 8cm (was 5cm)            # the jaws (force + at grasp center + grip width)
    return torch.nan_to_num(height * held, nan=0.0)                     # -> no knock/fly/push-from-below farming


def sq_open_approach(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """OPEN-during-approach: reward opening the jaws (qpos -> 0.045) so the gripper can CLEAR the 4cm cube (REST
    0.020 = mouth ~2.3cm is too closed to fit over it), then FADE as alignment completes so close_after_align
    takes over (open x (1-align)). _REAL_SENSOR: approaching = cube visible in the wrist (eye-in-hand) view;
    else: cube within 0.15 of the TCP. x not_grasped x _DENSE_SCALE. (qpos UP=OPEN, DOWN=CLOSED.)"""
    open_amt = ((_gq(env) - 0.035) / (0.045 - 0.035)).clamp(0.0, 1.0)   # 1=open(q>=0.045), 0 at the cube width
    gf = env.scene["gripper_contact"].data.force_matrix_w[:, 0, 0, :].norm(dim=-1)
    jf = env.scene["jaw_contact"].data.force_matrix_w[:, 0, 0, :].norm(dim=-1)
    not_grasped = (torch.minimum(gf, jf) < 0.3).float()
    aligned = align_wrist_centroid(env).clamp(0.0, 1.0)               # fade open as alignment completes
    if _REAL_SENSOR[0]:
        try:
            _, _, vis, _f = _red_centroid(env.scene["wrist"].data.output["rgb"])
            approaching = vis                                         # cube in the eye-in-hand view = near it
        except Exception:
            approaching = torch.zeros_like(open_amt)
    else:
        dist = torch.norm(env.scene[object_cfg.name].data.root_pos_w - _tcp(env, ee_frame_cfg), dim=1)
        approaching = (dist < 0.15).float()
    return torch.nan_to_num(open_amt * approaching * (1.0 - aligned) * not_grasped * _DENSE_SCALE[0], nan=0.0)


# per-env alignment-hold state (module globals). `count` = consecutive OPEN-aligned steps (resets when
# alignment breaks). `latch` = an aligned-hold of >= ALIGN_UNLOCK steps was achieved this episode and not
# since lost (held until the cube drifts > ALIGN_RESET_DIST, which also covers env reset). _is_grasped
# REQUIRES the latch -> a grasp only counts AFTER an align-hold (feedback v2: grasp must include the hold).
_align_count = {}
_align_latch = {}
# NB: the ee_frame "TCP" overshoots the fingertips by ~12cm, so the cube sits ~11-13cm from the TCP when
# actually grasped (MEASURED, diag_grasp_dist: median 12cm, p25-p75 11-13). Hence the align/grasp distance
# thresholds are ~0.13 (the grasp-ready band), NOT a few cm — the old 0.035 was geometrically impossible
# (cube never that close to the TCP) which made the align reward never fire and grasp never count (0 sr).
ALIGN_DIST, ALIGN_OPEN_Q, ALIGN_HOLD_TARGET, ALIGN_UNLOCK, ALIGN_RESET_DIST = 0.13, 0.035, 30, 10, 0.20
ALIGN_GRIP_Q = 0.03      # gripper qpos where the jaws grip the 4cm cube (mouth ~3.5cm); close target

def sq_align_hold(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """Pre-grasp ALIGNMENT-HOLD shaping (feedback v2: reach/align is poor). While the OPEN jaws are
    aligned ON the cube (cube within ALIGN_DIST of the TCP AND jaws OPEN, qpos >= ALIGN_OPEN_Q so the
    mouth clears the 4cm cube), reward GROWS with how long the alignment is HELD: a per-env counter
    increments each aligned step and resets the instant alignment breaks. reward = min(hold/HOLD_TARGET,1)
    so a long hold (~3s @10Hz = 30 steps) earns far more total than a brief touch. ALSO maintains the
    `latch`: once the hold reaches ALIGN_UNLOCK steps the env is align-confirmed (held until the cube
    drifts away / env reset), and _is_grasped REQUIRES it -> the model must align-and-HOLD before a grasp
    counts (grasp after align-hold). jaws-open self-gates this off once the gripper closes (qpos drops
    below ALIGN_OPEN_Q), so there is no circular dependency on _is_grasped."""
    obj = env.scene[object_cfg.name]
    tcp = _tcp(env, ee_frame_cfg)
    dist = torch.norm(obj.data.root_pos_w - tcp, dim=1)
    aligned = (dist < ALIGN_DIST) & (_gq(env) >= ALIGN_OPEN_Q)
    n = dist.shape[0]
    c = _align_count.get(0); l = _align_latch.get(0)
    if c is None or c.shape[0] != n or c.device != dist.device:
        c = torch.zeros(n, device=dist.device)
        l = torch.zeros(n, dtype=torch.bool, device=dist.device)
    c = torch.where(aligned, c + 1.0, torch.zeros_like(c))
    pre_latched = l                                            # latch BEFORE this step's update
    l = (l | (c >= ALIGN_UNLOCK)) & (dist < ALIGN_RESET_DIST)  # set on hold; drop when cube drifts / reset
    _align_count[0] = c; _align_latch[0] = l
    # NON-FARMABLE: pay the build-up ramp (0->1 over ALIGN_UNLOCK steps; longer hold = more) ONLY until
    # the hold latches; once latched -> 0. So the policy CANNOT hover forever to farm reward (that was a
    # 0-success local optimum); after the hold it must GRASP (now unlocked) to earn the bigger grasp reward.
    return (c / float(ALIGN_UNLOCK)).clamp(0.0, 1.0) * aligned.float() * (~pre_latched).float()


def _align_held(env, dist):
    """latch from sq_align_hold: an aligned-hold (>= ALIGN_UNLOCK steps) was achieved this episode and not
    since lost. Used by sq_close_after_align to reward closing only AFTER the hold. False until
    sq_align_hold has run."""
    l = _align_latch.get(0)
    n = dist.shape[0]
    if l is None or l.shape[0] != n or l.device != dist.device:
        return torch.zeros(n, dtype=torch.bool, device=dist.device)
    return l


def sq_close_after_align(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """CLOSE-after-align bridge (align -> grasp): reward CLOSING the gripper (qpos toward the grip width) IN
    PROPORTION to the align signal, only while NOT yet grasped. close = align x close_amt x not_grasped, so it is
    ALWAYS <= align_wrist (it only kicks in once aligned, and stops once the grasp FORCE forms -> grasp/place take
    over). Fixes the missing 'close' link that left the policy hovering open (align good, grasp never)."""
    gf = env.scene["gripper_contact"].data.force_matrix_w[:, 0, 0, :].norm(dim=-1)
    jf = env.scene["jaw_contact"].data.force_matrix_w[:, 0, 0, :].norm(dim=-1)
    not_grasped = (torch.minimum(gf, jf) < 0.3).float()
    close_amt = ((ALIGN_OPEN_Q - _gq(env)) / (ALIGN_OPEN_Q - ALIGN_GRIP_Q)).clamp(0.0, 1.0)  # 0=open .. 1=grip width
    aligned = align_wrist_centroid(env)            # the align signal (physics or vision per _REAL_SENSOR; incl DENSE)
    return torch.nan_to_num(aligned * close_amt * not_grasped, nan=0.0)   # <= align_wrist by construction


def sq_not_lifted(env, object_cfg=SceneEntityCfg("cube")):
    """not_lifted penalty. _REAL_SENSOR -> real proxy: 1 while NOT securely holding (current+position; the lift
    itself is rewarded via sq_place's return-to-rest). else HYBRID (the ORIGINAL): 1 while cube_z < CUBE_REST_Z
    + 1mm (privileged)."""
    cube_z = env.scene[object_cfg.name].data.root_pos_w[:, 2]
    return (cube_z < (CUBE_REST_Z + 0.001)).float()   # CONSTANT not_lifted (-1/3) = forward push to lift (un-gated)              # SQUINT: 1 while the cube is on the table


_GRIP_BIDX = [None]; _LOCAL_DOWN = [None]
def down_orientation_penalty(env, robot_cfg=SceneEntityCfg("robot")):
    """P1 wrist-DOWN constraint 2026-06-23: penalize the gripper not pointing straight DOWN so the eye-in-hand
    wrist cam always sees the floor/cube (never the background). 0 when straight down, ->2 when it tilts up."""
    from isaaclab.utils.math import quat_apply, quat_inv
    r = env.scene[robot_cfg.name]
    if _GRIP_BIDX[0] is None:
        bn = list(r.data.body_names)
        _GRIP_BIDX[0] = bn.index("gripper_link") if "gripper_link" in bn else len(bn) - 1
        dq = torch.tensor([[-0.4309, -0.5333, -0.5689, 0.4542]], device=r.device)   # SIM down orientation
        _LOCAL_DOWN[0] = quat_apply(quat_inv(dq), torch.tensor([[0.0, 0.0, -1.0]], device=r.device))  # (1,3)
    eq = r.data.body_quat_w[:, _GRIP_BIDX[0]]
    wa = quat_apply(eq, _LOCAL_DOWN[0].expand(eq.shape[0], 3))   # gripper approach axis in world
    return (1.0 - (-wa[:, 2])).clamp(min=0.0)                    # 0=straight down, grows as it tilts up


def grasp_close_reward(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """GRASP SHAPING 1 (2026-06-23): dense pre-grasp -- reward CLOSING the gripper WHEN the TCP is at the cube.
    Bridges the sparse reaching->is_grasped gap (policy was stuck reaching, never closing). Cube pose is
    PRIVILEGED -> training reward ONLY, never a deploy input."""
    obj = env.scene[object_cfg.name]
    tcp = _tcp(env, ee_frame_cfg)
    dist = torch.norm(obj.data.root_pos_w - tcp, dim=1)
    near = 1.0 - torch.tanh(35.0 * dist)                       # TIGHT (2026-06-23): ~1 only when the cube is truly in the jaw MOUTH (<~3cm of grasp center) -> closing on empty air earns ~0 (no contact-less close farming)
    _r = env.scene["robot"]; _gi = _r.data.joint_names.index("gripper")
    gq = _r.data.joint_pos[:, _gi].clamp(0.0, 0.045)
    closed = 1.0 - (gq / 0.045)                                # 0 open .. 1 fully closed
    return near * closed                                        # reward closing ONLY when at the cube


def grasp_contact_reward(env, object_cfg=SceneEntityCfg("cube")):
    """GRASP SHAPING 2 (2026-06-23): CONTINUOUS contact -- tanh(min jaw force) so partial/forming contact gives a
    gradient (binary is_grasped gave none until a full grasp). Sensors pre-filtered to the cube body."""
    grip = env.scene["gripper_contact"].data.force_matrix_w[:, 0, 0, :].norm(dim=-1)
    jaw = env.scene["jaw_contact"].data.force_matrix_w[:, 0, 0, :].norm(dim=-1)
    obj = env.scene[object_cfg.name]; _tcpw = _tcp(env, SceneEntityCfg("ee_frame"))
    in_jaw = (torch.norm(obj.data.root_pos_w - _tcpw, dim=1) < GRASP_ZONE).float()   # PHASE3 (user): reward contact ONLY when the cube is in the inner jaw zone
    return torch.tanh(torch.minimum(grip, jaw)) * in_jaw


_succ_streak = [None]    # per-env consecutive in-fold-band frames (success fires at FOLD_FRAMES)
def is_fold_success(env, object_cfg=SceneEntityCfg("cube")):
    """SUCCESS = fold held FOLD_FRAMES consecutive steps (was SQUINT instantaneous latch).
      item_lifted  = cube_z >= CUBE_REST_Z + 0.001   (~1mm off the table, = Squint half_size+1e-3)
      grasped      = both fingers in contact-force on the cube (_is_grasped, angle-gated 110deg)
      reached_rest = ||commanded arm target - REST_QPOS|| < 0.2
    Returns a (N,) bool tensor. (min_force kept at our calibrated 0.3 -- Isaac's contact-sensor force
    scale differs from Squint's SAPIEN pairwise force, so the 0.5 N literal does not transfer.)"""
    # SUCCESS (user 2026-06-25) = the FOLD held FOLD_FRAMES(3) consecutive steps -> SAME condition as
    # fold_reward's in-band streak, so reward and success are fully consistent. Per-step cond = lifted(>=LIFT_GATE)
    # & grasped & reached_rest(||cmd arm - REST||<0.2 over all 6 joints). Latches once the streak reaches 3.
    n, device = env.num_envs, env.device
    if _succ_streak[0] is None or _succ_streak[0].shape[0] != n:
        _succ_streak[0] = torch.zeros(n, dtype=torch.long, device=device)
    just_reset = env.episode_length_buf <= 1
    _succ_streak[0][just_reset] = 0
    cube_z = env.scene[object_cfg.name].data.root_pos_w[:, 2]
    # SUCCESS (user 2026-06-25) = LIFT the grasped cube to >= SUCC_LIFT(0.15) and hold FOLD_FRAMES frames.
    # fold/return-to-REST DROPPED entirely — the task is now "grasp + lift high".
    lifted15 = cube_z >= SUCC_LIFT
    grasped = _is_grasped(env, object_cfg, min_force=0.3, require_align=False)
    cond = lifted15 & grasped & (~just_reset)
    _succ_streak[0] = torch.where(cond, _succ_streak[0] + 1, torch.zeros_like(_succ_streak[0]))
    succ = _succ_streak[0] >= FOLD_FRAMES              # success = (cube>=15cm & grasped) held 3 frames
    if os.environ.get("DIAG_FOLD"):
        _fold_diag[0] += 1
        if _fold_diag[0] % 200 == 0:
            v = ~just_reset
            print(f"[FOLD-DIAG] lifted15(>=0.15)={float((lifted15&v).float().mean()):.3f} "
                  f"grasped={float((grasped&v).float().mean()):.3f} "
                  f"cond={float(cond.float().mean()):.3f} SUCC(streak>=3)={float(succ.float().mean()):.3f} "
                  f"max_streak={int(_succ_streak[0].max())} cube_z_mean={float(cube_z.mean()):.3f} "
                  f"cube_z_max={float(cube_z.max()):.3f}", flush=True)
    return succ
_fold_diag = [0]


# ── Fleet-wide success counter (counts SUCCESSFUL EPISODES across ALL envs) ────────
# The SB3 callback only sees env_0 AND only after auto-reset, so it can neither count
# fleet-wide successes nor read the terminal state. Robust two-part scheme instead:
#   success_latch  — a termination term (ALWAYS returns False) running every step inside
#     termination_manager.compute(), i.e. BEFORE _reset_idx, where the terminal state is
#     still valid. It latches "did this episode reach success at any step?" per env.
#   count_successes — a mode="reset" event running for exactly the env_ids being reset;
#     it tallies their latch and clears it. Order-independent vs other reset events
#     (it reads the latch, not env state).
_succ_total = [0]        # cumulative successful episodes across all envs
_ep_total = [0]          # cumulative finished episodes across all envs
_succ_recent = []        # 1/0 per finished episode (last 500) -> recent success rate
_succ_flag = [None]      # per-env latch (bool tensor): ongoing episode reached success?
_last_ep_succ = [None]   # per-env success of the most recently FINISHED episode (env_0 replay)


def success_latch(env, object_cfg=SceneEntityCfg("cube")):
    """Termination term that NEVER terminates. Side-effect: latch per-env success while
    the terminal state is still valid (pre-reset). Returns an all-False (N,) bool tensor."""
    n = env.num_envs
    succ = is_fold_success(env, object_cfg)
    if _succ_flag[0] is None or _succ_flag[0].shape[0] != n:
        _succ_flag[0] = torch.zeros(n, dtype=torch.bool, device=succ.device)
    _succ_flag[0] |= succ
    return torch.zeros(n, dtype=torch.bool, device=succ.device)


def count_successes(env, env_ids):
    """Reset event: tally latched successes for the just-finished env_ids, then clear them."""
    flag = _succ_flag[0]
    if flag is None:
        return
    fin = flag[env_ids]
    _succ_total[0] += int(fin.sum().item())
    _ep_total[0] += int(fin.numel())
    _succ_recent.extend(int(v) for v in fin.tolist())
    if len(_succ_recent) > 500:
        del _succ_recent[:-500]
    if _last_ep_succ[0] is None or _last_ep_succ[0].shape[0] != flag.shape[0]:
        _last_ep_succ[0] = torch.zeros_like(flag)
    _last_ep_succ[0][env_ids] = fin
    flag[env_ids] = False


def _touching_table(env):
    """Detect gripper/jaw touching anything other than the cube (i.e. the table)."""
    gn = env.scene["gripper_contact"].data.net_forces_w[:, 0, :]
    gc = env.scene["gripper_contact"].data.force_matrix_w[:, 0, 0, :]
    jn = env.scene["jaw_contact"].data.net_forces_w[:, 0, :]
    jc = env.scene["jaw_contact"].data.force_matrix_w[:, 0, 0, :]
    return ((gn - gc).norm(dim=-1) >= 0.01) | ((jn - jc).norm(dim=-1) >= 0.01)


def table_collision_penalty(
    env,
    table_height=0.04,
    ee_frame_cfg=SceneEntityCfg("ee_frame"),
):
    """Penalty for touching the table surface."""
    return _touching_table(env).float()


# ── Smoothness penalties ──
_prev_action = [None]
_prev_prev_action = [None]

def action_rate_penalty(env):
    """||a_t - a_{t-1}||² — penalize sudden action changes."""
    cur = env.action_manager.action
    if _prev_action[0] is None or _prev_action[0].shape != cur.shape:
        _prev_action[0] = cur.clone()
    penalty = ((cur - _prev_action[0]) ** 2).sum(dim=-1)
    _prev_action[0] = cur.clone()
    return penalty


def action_jerk_penalty(env):
    """||a_t - 2*a_{t-1} + a_{t-2}||² — penalize acceleration changes (tremor)."""
    cur = env.action_manager.action
    if _prev_action[0] is None or _prev_action[0].shape != cur.shape:
        _prev_action[0] = cur.clone()
    if _prev_prev_action[0] is None or _prev_prev_action[0].shape != cur.shape:
        _prev_prev_action[0] = cur.clone()
    jerk = cur - 2 * _prev_action[0] + _prev_prev_action[0]
    penalty = (jerk ** 2).sum(dim=-1)
    _prev_prev_action[0] = _prev_action[0].clone()
    _prev_action[0] = cur.clone()
    return penalty


def joint_vel_penalty(env, robot_cfg=SceneEntityCfg("robot")):
    """||qvel||^2 over ALL joints -- velocity regularization (calm motion, no twisting/whipping).
    Pairs with action_rate_penalty to smooth the contorted approach that transfers badly to real."""
    qvel = env.scene[robot_cfg.name].data.joint_vel
    return (qvel ** 2).sum(dim=-1)
