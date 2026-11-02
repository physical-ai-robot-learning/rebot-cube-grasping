"""Squint-exact reward for SO101 LiftCube."""

import torch
import numpy as np
from isaaclab.managers import SceneEntityCfg

# Squint SO101 start keyframe
# reBot arm 6 joints [pan, lift, elbow, wrist_flex, wrist_yaw, wrist_roll]
REST_QPOS = torch.tensor([0.0, -1.15, -1.00, 0.55, 0.0, 0.0])
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


def _is_grasped(env, object_cfg=SceneEntityCfg("cube"), min_force=0.3, max_angle_deg=110, align_tol=0.14):
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
    obj = env.scene[object_cfg.name]
    tcp = _tcp(env, SceneEntityCfg("ee_frame"))
    dist = torch.norm(obj.data.root_pos_w - tcp, dim=1)
    aligned = dist < align_tol                       # cube in the grasp zone (no off-center edge-catch)
    held = _align_held(env, dist)                    # AND an align-HOLD was achieved first -> GRASP AFTER
    return grip_ok & jaw_ok & aligned & held         # ALIGN (the model must align+hold before a grasp counts).
    # (Earlier this gate gave 0 success ONLY because the align threshold was an impossible 0.035 vs the real
    # ~12cm grasp distance -> the latch could never set. With ALIGN_DIST=0.13 + sq_close_after_align bridging
    # hold->close, the latch sets and the grasp-after-align sequence is learnable.)


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
    return raw_reward * mult


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
    tgt = _get_target_qpos(env)[:, :6]
    return torch.norm(tgt - REST_QPOS.to(tgt.device), dim=1)


def sq_reaching(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """Squint reaching: 1 - tanh(5 * dist(tcp, cube)). Ungated, stationary."""
    obj = env.scene[object_cfg.name]
    tcp = _tcp(env, ee_frame_cfg)
    dist = torch.norm(obj.data.root_pos_w - tcp, dim=1)
    return 1.0 - torch.tanh(5.0 * dist)


def sq_grasped(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame"),
               robot_cfg=SceneEntityCfg("robot")):
    """Squint: binary +1 when grasped. Ungated (no _near_mouth), no dopamine."""
    return _is_grasped(env, object_cfg, min_force=0.3).float()


def sq_place(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame"),
             robot_cfg=SceneEntityCfg("robot")):
    """Place / return-to-rest while holding: exp(-0.5*dist_to_rest) * is_grasped.
    GENTLER decay than Squint's exp(-2*) -- the reBot grasp pose sits ~4 rad from REST (wrists rotated),
    where exp(-2*4)~3e-4 gives NO gradient to return; exp(-0.5*4)=0.135 keeps a real pull toward REST,
    so the policy actually learns the return (-> reached_rest -> success) instead of holding the cube out."""
    grasped = _is_grasped(env, object_cfg, min_force=0.3).float()
    return torch.exp(-0.5 * _dist_to_rest(env)) * grasped


def sq_open_approach(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """Pre-grasp shaping: reward opening the jaw WIDE while approaching the cube and NOT yet grasped,
    so the gripper arrives open and can descend over the cube. Gated by ~grasped so it stops the instant
    a grasp forms (never fights the grasp).
    MEASURED gripper convention (diag_gripper_stroke): qpos UP = OPEN (mouth wider), DOWN = CLOSED;
    mouth=4cm (cube) at q~0.035, clears the cube at q>0.04. (The old code rewarded q->0 as "open" which
    was BACKWARDS — q->0 is fully CLOSED.) open_amt=1 at the open cap (q>=0.05), 0 at the grip width."""
    obj = env.scene[object_cfg.name]
    tcp = _tcp(env, ee_frame_cfg)
    dist = torch.norm(obj.data.root_pos_w - tcp, dim=1)
    open_amt = ((_gq(env) - 0.035) / (0.045 - 0.035)).clamp(0.0, 1.0)  # 1=open(q>=0.045 cap), 0=at grip(q<=0.035)
    approaching = (dist < 0.15).float()
    not_grasped = (~_is_grasped(env, object_cfg, min_force=0.3)).float()
    return open_amt * approaching * not_grasped


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
    """CLOSE-after-align shaping (user idea): once the align-hold LATCH is set (the OPEN jaws were held
    aligned on the cube) and the cube is still in the grasp zone and NOT yet grasped, reward CLOSING the
    gripper (qpos descending from the open width ALIGN_OPEN_Q toward the grip width ALIGN_GRIP_Q). This
    bridges the hold -> close -> grasp transition (the missing link that left the policy hovering open).
    Gated by latch + near + ~grasped, so it fires only AFTER an align-hold and stops once the grasp forms
    (the bigger grasp/place reward then takes over)."""
    obj = env.scene[object_cfg.name]
    tcp = _tcp(env, ee_frame_cfg)
    dist = torch.norm(obj.data.root_pos_w - tcp, dim=1)
    latch = _align_held(env, dist)
    near = dist < ALIGN_RESET_DIST
    not_grasped = ~_is_grasped(env, object_cfg, min_force=0.3)
    close_amt = ((ALIGN_OPEN_Q - _gq(env)) / (ALIGN_OPEN_Q - ALIGN_GRIP_Q)).clamp(0.0, 1.0)  # 1=at grip width
    return close_amt * (latch & near & not_grasped).float()


def sq_not_lifted(env, object_cfg=SceneEntityCfg("cube")):
    """Squint not-lifted penalty (returns 1 while on table). Threshold ~5mm above rest (was 15cm)."""
    obj = env.scene[object_cfg.name]
    cube_z = obj.data.root_pos_w[:, 2]
    item_lifted = cube_z >= (CUBE_REST_Z + 0.005)
    return (~item_lifted).float()


def is_fold_success(env, object_cfg=SceneEntityCfg("cube")):
    """SUCCESS = SQUINT-EXACT (lift.py:354-382 evaluate()): item_lifted & is_grasped & reached_rest_qpos.
      item_lifted  = cube_z >= CUBE_REST_Z + 0.001   (~1mm off the table, = Squint half_size+1e-3)
      grasped      = both fingers in contact-force on the cube (_is_grasped, angle-gated 110deg)
      reached_rest = ||commanded arm target - REST_QPOS|| < 0.2
    Returns a (N,) bool tensor. (min_force kept at our calibrated 0.3 -- Isaac's contact-sensor force
    scale differs from Squint's SAPIEN pairwise force, so the 0.5 N literal does not transfer.)"""
    obj = env.scene[object_cfg.name]
    cube_z = obj.data.root_pos_w[:, 2]
    item_lifted = cube_z >= (CUBE_REST_Z + 0.001)     # Squint: item.z >= half_size + 1e-3
    grasped = _is_grasped(env, object_cfg, min_force=0.3)
    reached_rest = _dist_to_rest(env) < 0.2           # Squint reached_rest_qpos < 0.2
    return item_lifted & grasped & reached_rest


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
