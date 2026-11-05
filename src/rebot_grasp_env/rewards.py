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


def _getTargetQpos(env):
    """Retrieve the target joint positions from the action manager terms."""
    tAm = env.action_manager
    tTargets = []
    for tTerm in tAm._terms.values():
        if hasattr(tTerm, "_target_qpos"):
            tTargets.append(tTerm._target_qpos)
    if tTargets:
        return torch.cat(tTargets, dim=-1)
    return env.scene["robot"].data.joint_pos


def _computeAngleBetween(a, b):
    """Compute angle between two vector batches."""
    tAN = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    tBN = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return torch.acos((tAN * tBN).sum(dim=-1).clamp(-1, 1))


def _quatToYAxis(q):
    """Extract y-axis direction from quaternion (w,x,y,z format).

    Given a unit quaternion q = (w, x, y, z), the y-column of the
    corresponding rotation matrix is:
        [2(xy + wz),  1 - 2(x^2 + z^2),  2(yz - wx)]
    """
    tW, tX, tY, tZ = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack(
        [2 * (tX * tY + tW * tZ), 1 - 2 * (tX * tX + tZ * tZ), 2 * (tY * tZ - tW * tX)],
        dim=-1,
    )


def _getLinkBodyIdx(robot, link_name_substr):
    """Find body index whose name contains *link_name_substr* (case-insensitive)."""
    for tI, tName in enumerate(robot.data.body_names):
        if link_name_substr in tName.lower():
            return tI
    return None


def _cubeBetweenJaws(env, max_tcp_dist=0.05):
    """Geometric check: cube center is BETWEEN the two jaw tips.

    Uses dot product of (fixed_tip→cube) and (moving_tip→cube).
    If cube is between jaws, these vectors point in opposite directions → dot < 0.
    If robot is pressing from one side (tip pushing), both vectors point same way → dot > 0.
    """
    tEe = env.scene["ee_frame"]
    tFixedTip = tEe.data.target_pos_w[:, 0, :]   # (N, 3)
    tMovingTip = tEe.data.target_pos_w[:, 1, :]   # (N, 3)
    tCubePos = env.scene["cube"].data.root_pos_w  # (N, 3)

    tVecToFixed = tFixedTip - tCubePos    # cube → fixed jaw
    tVecToMoving = tMovingTip - tCubePos  # cube → moving jaw

    # Dot product < 0 means cube is between the two tips
    tDot = (tVecToFixed * tVecToMoving).sum(dim=-1)
    tBetween = tDot < 0

    # Also check cube is near TCP (not far away)
    tTcp = (tFixedTip + tMovingTip) * 0.5
    tTcpDist = (tCubePos - tTcp).norm(dim=-1)
    tNear = tTcpDist < max_tcp_dist

    return tBetween & tNear


def _isGrasped(env, object_cfg=SceneEntityCfg("cube"), min_force=0.3, max_angle_deg=110, align_tol=0.14,
                require_align=True):
    """Contact-based grasp, NOW ALSO REQUIRING ALIGNMENT (feedback v2 2026-06-11): BOTH fingers press the
    cube (force >= min_force) AND the cube is CENTERED at the grasp center (||cube - TCP|| < align_tol).
    So 'grasped' = a properly ALIGNED grasp, not an off-center edge-catch. This propagates to the grasp
    reward, the place/open gates, AND is_fold_success -> the policy must ALIGN to earn grasp credit and
    success, so it learns alignment (was the weak point). The gripper_contact/jaw_contact sensors are
    pre-filtered to the target cube's rigid body."""
    tGripForce = env.scene["gripper_contact"].data.force_matrix_w[:, 0, 0, :]
    tJawForce = env.scene["jaw_contact"].data.force_matrix_w[:, 0, 0, :]
    tGripOk = tGripForce.norm(dim=-1) >= min_force
    tJawOk = tJawForce.norm(dim=-1) >= min_force
    return tGripOk & tJawOk   # SQUINT-exact: BOTH jaws contact the cube => cube is BETWEEN them = inner grasp (no separate inner-pad/distance gate needed)
    if not require_align:
        return tGripOk & tJawOk
    tObj = env.scene[object_cfg.name]
    tTcp = _tcp(env, SceneEntityCfg("ee_frame"))
    tDist = torch.norm(tObj.data.root_pos_w - tTcp, dim=1)
    tAligned = tDist < align_tol                       # cube in the grasp zone (no off-center edge-catch)
    return tGripOk & tJawOk & tAligned                # STATELESS alignment gate (align-HOLD latch dropped --
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
    tHNow = (raw_reward > 0.001).float().mean().item()
    _hit_rates[key] = 0.99 * _hit_rates[key] + 0.01 * tHNow  # EMA
    tH = _hit_rates[key]
    tMult = _DOPAMINE_LAMBDA + (1 - _DOPAMINE_LAMBDA) * (1 - tH) ** _DOPAMINE_ALPHA
    return raw_reward   # DOPAMINE DISABLED on ALL stages 2026-06-23 (Squint has none; decay shrank grasp reward as it was learned -> grasp vanished)


# ======================================================================
# Reward terms
# ======================================================================


# Retry tracking: detect close→open→close cycle (grasp attempt)
_was_close_to_cube = {}
_was_jaw_closed = {}
_retry_count = {}

def graspRetryReward(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """One-time reward each time agent re-opens jaw after failed grasp attempt near cube.
    Encourages: approach → close (fail) → open → approach again → close."""
    tN = env.num_envs
    tDevice = env.device
    tKey = "retry"
    if tKey not in _retry_count or _retry_count[tKey].shape[0] != tN:
        _was_close_to_cube[tKey] = torch.zeros(tN, dtype=torch.bool, device=tDevice)
        _was_jaw_closed[tKey] = torch.zeros(tN, dtype=torch.bool, device=tDevice)
        _retry_count[tKey] = torch.zeros(tN, dtype=torch.long, device=tDevice)

    # Reset on new episodes
    tJustReset = env.episode_length_buf <= 1
    _was_close_to_cube[tKey][tJustReset] = False
    _was_jaw_closed[tKey][tJustReset] = False
    _retry_count[tKey][tJustReset] = 0

    # Current state
    tObj = env.scene[object_cfg.name]
    tEe = env.scene[ee_frame_cfg.name]
    tTcp = (tEe.data.target_pos_w[:, 0, :] + tEe.data.target_pos_w[:, 1, :]) * 0.5
    tDist = torch.norm(tObj.data.root_pos_w - tTcp, dim=1)
    tClose = tDist < 0.10
    tR = env.scene["robot"]; tGi = tR.data.joint_names.index("gripper")
    tGripperQpos = tR.data.joint_pos[:, tGi]   # prismatic, meters (stroke 0~0.0715)
    # reBot pads CONVERGE at HIGH qpos (joint value goes UP to close the fingers), so the
    # open/closed sense is INVERTED vs a naive "small=closed": gripping = qpos>0.05, released = qpos<0.035.
    tJawClosed = tGripperQpos > 0.05    # pads gripping
    tJawOpen = tGripperQpos < 0.035     # pads apart (released)

    # Detect retry: was close+closed (attempt), now open again (retry)
    tWasAttempting = _was_close_to_cube[tKey] & _was_jaw_closed[tKey]
    tRetrying = tWasAttempting & tJawOpen & tClose
    _retry_count[tKey][tRetrying] += 1

    # Update state
    _was_close_to_cube[tKey] = tClose
    _was_jaw_closed[tKey] = tJawClosed

    return tRetrying.float()


def reachingReward(
    env,
    object_cfg=SceneEntityCfg("cube"),
    ee_frame_cfg=SceneEntityCfg("ee_frame"),
):
    """Dense reaching reward: 1 - tanh(5 * dist(tcp, cube))."""
    tObj = env.scene[object_cfg.name]
    tEe = env.scene[ee_frame_cfg.name]
    tTcp = (tEe.data.target_pos_w[:, 0, :] + tEe.data.target_pos_w[:, 1, :]) * 0.5
    tDist = torch.norm(tObj.data.root_pos_w - tTcp, dim=1)
    tBase = 1 - torch.tanh(5 * tDist)
    # Only reward reaching when the gripper is NOT flipped -- its mouth must point DOWN, not up.
    # gripper_link local +z points toward the grasp; its world z-component < 0 means pointing down.
    tRobot = env.scene["robot"]
    tQ = tRobot.data.body_quat_w[:, tRobot.data.body_names.index("gripper_link"), :]
    tZc = 1.0 - 2.0 * (tQ[:, 1] ** 2 + tQ[:, 2] ** 2)    # gripper_link local +z, world z-component
    tUpright = torch.clamp(-tZc, min=0.0, max=1.0)       # 1=straight down, 0=horizontal/flipped up
    tGate = _panFaceOk(env, object_cfg).float()       # GATE: only reward reaching when pan-aligned
    return _dopamine("reaching", tBase * tUpright * tGate)


# ── Stage-gate helpers: a stage's reward is given ONLY when the PREVIOUS stage's condition holds ──
def _tcp(env, ee_frame_cfg):
    tEe = env.scene[ee_frame_cfg.name]
    return (tEe.data.target_pos_w[:, 0, :] + tEe.data.target_pos_w[:, 1, :]) * 0.5

def _gq(env):
    tR2 = env.scene["robot"]
    return tR2.data.joint_pos[:, tR2.data.joint_names.index("gripper")]

def _gripperDown(env):
    tR2 = env.scene["robot"]
    tQ = tR2.data.body_quat_w[:, tR2.data.body_names.index("gripper_link"), :]
    return (1.0 - 2.0 * (tQ[:, 1] ** 2 + tQ[:, 2] ** 2)) < 0.0       # gripper mouth points DOWN (not flipped)

def _panError(env, object_cfg=SceneEntityCfg("cube")):
    """|shoulder_pan - required_pan| (wrapped to [0,π]). required_pan points the arm (pan=0 => +Y
    world) at the cube. JOINT-based, so it is NOT fooled by a folded/flipped arm whose gripper happens
    to land at the cube's azimuth (which the old base->gripper measure mistook for 'aligned')."""
    tRobot = env.scene["robot"]; tObj = env.scene[object_cfg.name]
    tBase = tRobot.data.body_pos_w[:, tRobot.data.body_names.index("base_link"), :2]
    tDx = tObj.data.root_pos_w[:, 0] - tBase[:, 0]
    tDy = tObj.data.root_pos_w[:, 1] - tBase[:, 1]
    tReq = torch.atan2(-tDx, tDy)
    tPan = tRobot.data.joint_pos[:, tRobot.data.joint_names.index("shoulder_pan")]
    tE = tPan - tReq
    return torch.atan2(torch.sin(tE), torch.cos(tE)).abs()

def _panFaceOk(env, object_cfg=SceneEntityCfg("cube")):
    """STAGE 0 condition: pan faces the cube (joint-based) AND finger axis matches a cube face."""
    tRobot = env.scene["robot"]; tObj = env.scene[object_cfg.name]; tBn = tRobot.data.body_names
    tFacing = _panError(env, object_cfg) < 0.26
    tGl = tRobot.data.body_pos_w[:, tBn.index("gripper_left"), :2]
    tGr = tRobot.data.body_pos_w[:, tBn.index("gripper_right"), :2]
    tFaxis = tGr - tGl; tFaxis = tFaxis / (tFaxis.norm(dim=-1, keepdim=True) + 1e-6)
    tQ = tObj.data.root_quat_w
    tCx = torch.stack([1 - 2 * (tQ[:, 2] ** 2 + tQ[:, 3] ** 2), 2 * (tQ[:, 1] * tQ[:, 2] + tQ[:, 0] * tQ[:, 3])], dim=-1)
    tCx = tCx / (tCx.norm(dim=-1, keepdim=True) + 1e-6)
    tDot = (tFaxis * tCx).sum(-1).abs().clamp(max=1.0)
    tS2 = 2.0 * tDot * torch.sqrt((1.0 - tDot * tDot).clamp(min=0.0))
    return tFacing & (tS2 < 0.40)

def _reachOk(env, object_cfg, ee_frame_cfg):       # STAGE 1: tcp near cube + pointing down
    tD = torch.norm(env.scene[object_cfg.name].data.root_pos_w - _tcp(env, ee_frame_cfg), dim=1)
    return (tD < 0.12) & _gripperDown(env)

def _approachOk(env, object_cfg, ee_frame_cfg):    # STAGE 2: closer + jaw open
    tD = torch.norm(env.scene[object_cfg.name].data.root_pos_w - _tcp(env, ee_frame_cfg), dim=1)
    return (tD < 0.08) & (_gq(env) < 0.05)

def _nearMouth(env, object_cfg, ee_frame_cfg):     # STAGE 3: cube at the grasp center (gripper-agnostic)
    tD = torch.norm(env.scene[object_cfg.name].data.root_pos_w - _tcp(env, ee_frame_cfg), dim=1)
    return tD < 0.05


_pan_held = [None]

def panAlignedReward(env, object_cfg=SceneEntityCfg("cube")):
    """STAGE 0 (before reaching): shoulder_pan turns the arm to FACE the target cube -- the
    base->gripper azimuth is aligned with the base->cube azimuth (within ~15deg). Reward GROWS the
    longer it's held facing (up to ~10 s = the full episode) and RESETS when alignment is lost."""
    tN = env.num_envs
    tDevice = env.device
    if _pan_held[0] is None or _pan_held[0].shape[0] != tN:
        _pan_held[0] = torch.zeros(tN, dtype=torch.long, device=tDevice)
    tJustReset = env.episode_length_buf <= 1
    _pan_held[0][tJustReset] = 0

    tAligned = _panFaceOk(env, object_cfg)        # JOINT-based pan facing + finger-axis matches cube face

    _pan_held[0][tAligned] += 1
    _pan_held[0][~tAligned] = 0
    tReward = tAligned.float() * (_pan_held[0].float() / 50.0).clamp(max=2.0)  # 5s->1.0, 10s->2.0
    return _dopamine("pan_aligned", tReward)


def panFlipPenalty(env, object_cfg=SceneEntityCfg("cube")):
    """Penalty when the base (pan) is rotated AWAY from the cube -- discourages over-rotating /
    flipping the base around. 0 while roughly facing the cube, ramps to 1 when facing backward."""
    tAngle = _panError(env, object_cfg)            # JOINT-based pan error (robust to fold/flip)
    return torch.clamp((tAngle - 1.0) / 1.5, min=0.0, max=1.0)   # 0 below ~57deg, ->1 at ~143deg away


_prev_phi = [None]

def towardCubeReward(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """Dense PROGRESS reward (potential-based): reward pan rotating toward facing the cube AND the
    gripper getting closer. phi = pan_error + 3*tcp_dist (lower=better); reward = prev_phi - cur_phi
    (toward cube = +, away = -; telescopes so it can't be farmed). UNGATED -> guides pan/lift from
    the very first step (fixes 'pan/lift go the opposite way' during early exploration)."""
    tN = env.num_envs
    tPe = _panError(env, object_cfg)
    tD = torch.norm(env.scene[object_cfg.name].data.root_pos_w - _tcp(env, ee_frame_cfg), dim=1)
    tPhi = tPe + 3.0 * tD
    if _prev_phi[0] is None or _prev_phi[0].shape[0] != tN:
        _prev_phi[0] = tPhi.clone()
    tJustReset = env.episode_length_buf <= 1
    _prev_phi[0][tJustReset] = tPhi[tJustReset]
    tReward = (_prev_phi[0] - tPhi).clamp(-1.0, 1.0)
    _prev_phi[0] = tPhi.clone()
    return tReward


_aligned_held = [None]
_grasp_held_r = [None]

def graspHoldReward(env, object_cfg=SceneEntityCfg("cube")):
    """HOLD-PROPORTIONAL grasp bonus (2026-06-23): reward GROWS with consecutive grasped steps (0 -> 1.0 over 20
    steps) and RESETS the moment the grasp breaks -> incentivizes MAINTAINING the grasp through lift + return-to-rest,
    not just a brief touch. (not_lifted penalty still forces it to LIFT, so this can't be farmed on the table.)"""
    import torch as _t
    tN = env.num_envs; tDevice = env.device
    if _grasp_held_r[0] is None or _grasp_held_r[0].shape[0] != tN:
        _grasp_held_r[0] = _t.zeros(tN, dtype=_t.long, device=tDevice)
    tJustReset = env.episode_length_buf <= 1
    _grasp_held_r[0][tJustReset] = 0
    tG = _isGrasped(env, object_cfg, min_force=0.3, require_align=False)
    _grasp_held_r[0][tG] += 1
    _grasp_held_r[0][~tG] = 0
    return (_grasp_held_r[0].float() / 20.0).clamp(max=1.0)


def alignedReward(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """Pre-grasp ALIGNED stage: the target cube sits INSIDE the open gripper mouth (at the TCP /
    grasp center). Reward GROWS the longer it is held aligned -- 3 s (=30 steps) -> 1.0, up to 2.0 --
    and RESETS to 0 the moment the cube leaves the mouth or the gripper closes. Bridges reaching->grasp."""
    tN = env.num_envs
    tDevice = env.device
    if _aligned_held[0] is None or _aligned_held[0].shape[0] != tN:
        _aligned_held[0] = torch.zeros(tN, dtype=torch.long, device=tDevice)
    tJustReset = env.episode_length_buf <= 1
    _aligned_held[0][tJustReset] = 0

    tObj = env.scene[object_cfg.name]
    tEe = env.scene[ee_frame_cfg.name]
    tTcp = (tEe.data.target_pos_w[:, 0, :] + tEe.data.target_pos_w[:, 1, :]) * 0.5
    tDist = torch.norm(tObj.data.root_pos_w - tTcp, dim=1)
    tR = env.scene["robot"]; tGi = tR.data.joint_names.index("gripper")
    tGripperOpen = tR.data.joint_pos[:, tGi] < 0.05            # pads apart -> cube fits in the mouth
    tAligned = (tDist < 0.04) & tGripperOpen & _approachOk(env, object_cfg, ee_frame_cfg)  # gated by approach

    _aligned_held[0][tAligned] += 1
    _aligned_held[0][~tAligned] = 0                              # reset when it leaves the mouth
    tReward = tAligned.float() * (_aligned_held[0].float() / 30.0).clamp(max=2.0)  # 1.0 @3s, 2.0 @6s+
    return _dopamine("aligned", tReward)


_approach_open_fired = {}

def approachOpenReward(
    env,
    object_cfg=SceneEntityCfg("cube"),
    ee_frame_cfg=SceneEntityCfg("ee_frame"),
):
    """One-time reward for first reaching cube with jaw open."""
    tN = env.num_envs
    tDevice = env.device
    tKey = "approach_open"
    if tKey not in _approach_open_fired or _approach_open_fired[tKey].shape[0] != tN:
        _approach_open_fired[tKey] = torch.zeros(tN, dtype=torch.bool, device=tDevice)

    tJustReset = env.episode_length_buf <= 1
    _approach_open_fired[tKey][tJustReset] = False

    tObj = env.scene[object_cfg.name]
    tEe = env.scene[ee_frame_cfg.name]
    tTcp = (tEe.data.target_pos_w[:, 0, :] + tEe.data.target_pos_w[:, 1, :]) * 0.5
    tDist = torch.norm(tObj.data.root_pos_w - tTcp, dim=1)

    tClose = tDist < 0.10
    tR = env.scene["robot"]; tGi = tR.data.joint_names.index("gripper")
    tGripperQpos = tR.data.joint_pos[:, tGi]   # prismatic, meters
    tJawOpen = tGripperQpos < 0.035  # pads APART (reBot pads converge at HIGH qpos -> "open" is LOW qpos)

    tCondition = tClose & tJawOpen & _reachOk(env, object_cfg, ee_frame_cfg) & ~_approach_open_fired[tKey]
    _approach_open_fired[tKey] |= tCondition

    return tCondition.float()


def graspedReward(
    env,
    object_cfg=SceneEntityCfg("cube"),
    ee_frame_cfg=SceneEntityCfg("ee_frame"),
    robot_cfg=SceneEntityCfg("robot"),
):
    """Binary: 1 when grasping, 0 otherwise."""
    # GATE: grasp only counts when the cube is at the grasp center (came through the aligned stage)
    tG = _isGrasped(env, object_cfg, min_force=0.3) & _nearMouth(env, object_cfg, ee_frame_cfg)
    return _dopamine("grasped", tG.float())


def placeReward(
    env,
    object_cfg=SceneEntityCfg("cube"),
    ee_frame_cfg=SceneEntityCfg("ee_frame"),
    robot_cfg=SceneEntityCfg("robot"),
):
    """DISABLED — replaced by fold + fold_hold."""
    return torch.zeros(env.num_envs, device=env.device)


def notLiftedPenalty(env, object_cfg=SceneEntityCfg("cube")):
    """Penalty while the cube stays on the table (i.e. near its resting height ~0.06 m).
    Old threshold (0.016) assumed table at z=0, so it was always 'lifted' -> penalty never fired."""
    tObj = env.scene[object_cfg.name]
    tCubeZ = tObj.data.root_pos_w[:, 2]
    tItemLifted = tCubeZ >= (CUBE_REST_Z + 0.02)   # clearly off the table (>2cm above rest)
    return (~tItemLifted).float()


def liftHoldReward(env, object_cfg=SceneEntityCfg("cube")):
    """Reward for lifting cube 15cm + grasped (no fold requirement)."""
    tObj = env.scene[object_cfg.name]
    tCubeZ = tObj.data.root_pos_w[:, 2]
    tLifted = (tCubeZ >= 0.15).float()
    tGrasped = _isGrasped(env, object_cfg, min_force=0.3).float()
    return _dopamine("lift_hold", tLifted * tGrasped)


def _fold3joint(env):
    """3-joint fold errors: pan(0), shoulder_lift(1), elbow(2)."""
    tTgt = _getTargetQpos(env)[:, :-1]  # (N, 5)
    tFolded = FOLDED_QPOS[:-1].to(tTgt.device)
    tErrs = (tTgt - tFolded).abs()
    return tErrs[:, 0], tErrs[:, 1], tErrs[:, 2]  # pan, lift, elbow


def foldReward(env, object_cfg=SceneEntityCfg("cube")):
    """3-joint continuous fold: pan + shoulder_lift + elbow. All always-on."""
    tObj = env.scene[object_cfg.name]
    tCubeZ = tObj.data.root_pos_w[:, 2]
    tLifted = (tCubeZ >= 0.15).float()
    tGrasped = _isGrasped(env, object_cfg, min_force=0.3).float()
    tPanErr, tLiftErr, tElbowErr = _fold3joint(env)

    tSPan = torch.exp(-2.0 * tPanErr)        # pan (0° target)
    tSLift = torch.exp(-0.5 * tLiftErr)       # shoulder_lift (-90° target, big range)
    tSElbow = torch.exp(-0.5 * tElbowErr)     # elbow (150° target, big range)

    return _dopamine("fold", tLifted * tGrasped * (tSPan + tSLift + tSElbow) / 3.0)


_fh_first = [None]   # step when fold first achieved per env
_fh_held = [None]    # cumulative steps held since first fold

def foldHoldReward(env, object_cfg=SceneEntityCfg("cube")):
    """Fold hold: pan<5° AND lift<5° AND elbow<5°, held 70%→save, 95%→big reward."""
    tN = env.num_envs
    tDevice = env.device
    if _fh_first[0] is None or _fh_first[0].shape[0] != tN:
        _fh_first[0] = torch.full((tN,), -1, dtype=torch.long, device=tDevice)
        _fh_held[0] = torch.zeros(tN, dtype=torch.long, device=tDevice)

    tJustReset = env.episode_length_buf <= 1
    _fh_first[0][tJustReset] = -1
    _fh_held[0][tJustReset] = 0

    tObj = env.scene[object_cfg.name]
    tCubeZ = tObj.data.root_pos_w[:, 2]
    tLifted = (tCubeZ >= 0.15)
    tGrasped = _isGrasped(env, object_cfg, min_force=0.3)
    tPanErr = _fold3joint(env)[0]
    tHolding = tLifted & tGrasped & (tPanErr < 0.262)  # pan<15°

    tFirstTime = tHolding & (_fh_first[0] < 0)
    _fh_first[0][tFirstTime] = env.episode_length_buf[tFirstTime]
    _fh_held[0][tHolding] += 1

    tStarted = _fh_first[0] >= 0
    tElapsed = (env.episode_length_buf - _fh_first[0]).clamp(min=1)
    tRatio = (_fh_held[0].float() / tElapsed.float()).clamp(max=1.0)

    tAbove50 = tStarted & (tRatio >= 0.50) & (_fh_held[0] >= 5)
    tReward = torch.zeros(tN, device=tDevice)
    tHeldFrac = (_fh_held[0].float() / env.max_episode_length).clamp(max=1.0)
    tReward[tAbove50] = tHeldFrac[tAbove50] ** 2
    tReward[tStarted & ~tAbove50] = tRatio[tStarted & ~tAbove50] * 1.0

    return _dopamine("fold_hold", tReward)


# ════════════════════════════════════════════════════════════════════════════════
#  SQUINT-MATCHED reward + success (envs/lift.py compute_dense_reward + evaluate).
#  Squint solves SO101 LiftCube to ~100% in ~0.2-0.3M samples with just these:
#    reward = (1-tanh(5d)) + is_grasped + exp(-2*dist_to_rest)*is_grasped
#             - 3*touching_table - 1*not_lifted,   all /3 (normalized_dense).
#  success = lifted(~off table) & grasped & arm-target back near REST (<0.2).
#  No dopamine, no chained gates, no 15cm bar, no FOLDED pose. Stationary + ungated.
# ════════════════════════════════════════════════════════════════════════════════
def _distToRest(env):
    """||commanded arm target qpos - REST_QPOS|| over the 6 arm joints (Squint reached_rest_qpos).
    Uses the COMMANDED target (free at spawn), exactly like Squint's controller._target_qpos."""
    # SQUINT-EXACT (per user): reached_rest uses the COMMANDED arm target (controller._target_qpos), NOT actual qpos
    # -> the setpoint near REST is achievable (the reBot actual-qpos change made success too strict = a likely cause of success=0).
    tQ = env.action_manager.get_term("arm_action")._target_qpos
    return torch.norm(tQ - REST_QPOS.to(tQ.device), dim=1)


# DENSE-reward DECAY (neuro: dopamine fades as a skill is mastered). The trainer sets _DENSE_SCALE[0] from the
# success rate (high sr -> ~0 dense -> only outcome reward; sr drops -> dense returns = homeostatic, self-correcting).
# Default 1.0 -> NO effect for other trainers/teachers. Multiplies the FARMABLE dense terms (reaching + align only).
_DENSE_SCALE = [1.0]

# REAL-SENSOR reward toggle (per-process; the trainer sets MR._REAL_SENSOR[0] = args.real_sensor_reward).
# False (default) = ORIGINAL HYBRID reward (privileged physics + vision/current/position).
# True = real-robot-measurable only (vision / current / joint qpos); physics removed (the asym critic carries it).
_REAL_SENSOR = [False]

def sqReaching(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """Squint reaching: 1 - tanh(5 * dist(tcp, cube)). Ungated, stationary. Scaled by _DENSE_SCALE (decay)."""
    tObj = env.scene[object_cfg.name]
    tTcp = _tcp(env, ee_frame_cfg)
    tDist = torch.norm(tObj.data.root_pos_w - tTcp, dim=1)
    return (1.0 - torch.tanh(5.0 * tDist)) * _DENSE_SCALE[0]


# ── OBSERVABLE align reward (sim-real-consistent) ─────────────────────────────
# The cube renders RED; reward the red blob being CENTERED in the eye-in-hand WRIST view (= lateral grasp
# alignment), computed from the SAME raw rendered image the policy consumes -> reward and perception are
# coupled (visual servoing). Measurable IDENTICALLY on the real robot (HSV-red centroid in the real wrist
# frame), so it does not exploit privileged sim ground-truth. (DR is applied downstream in the trainer's
# read_imgs, NOT to env.scene[..].output, so the cube is its true red here -- matches real.)
WRIST_ALIGN_PX = (0.5, 0.80)   # target (x,y): grasp point = bottom-center of upright wrist view (between the fingers)

def _redCentroid(rgb):
    """rgb (N,H,W,3) uint8 -> (cx,cy in [0,1], visible[N]). Lighting-robust red mask: R high & R>>G,B."""
    tR2 = rgb[..., 0].float(); tG = rgb[..., 1].float(); tB = rgb[..., 2].float()
    tRed = (tR2 > 100) & (tR2 > 1.5 * tG) & (tR2 > 1.5 * tB)
    tN2, tH2, tW2 = tRed.shape
    tCnt = tRed.flatten(1).sum(1)
    tYs = torch.arange(tH2, device=rgb.device).view(1, tH2, 1).float()
    tXs = torch.arange(tW2, device=rgb.device).view(1, 1, tW2).float()
    tDenom = tCnt.clamp(min=1.0)
    tCx = (tRed * tXs).flatten(1).sum(1) / tDenom / (tW2 - 1)
    tCy = (tRed * tYs).flatten(1).sum(1) / tDenom / (tH2 - 1)
    return tCx, tCy, (tCnt >= 5).float(), tCnt / float(tH2 * tW2)   # + red-area fraction (proximity proxy)

def alignWristCentroid(env, cam_name="wrist", object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """align. _REAL_SENSOR -> VISION ONLY (cube centered on the grasp pixel x size in the WRIST view). else HYBRID
    (privileged physics 1-tanh(12d) x vision size = the ORIGINAL). x _DENSE_SCALE. (Name kept.)"""
    if _REAL_SENSOR[0]:
        try:
            tCx, tCy, tVis, tFrac = _redCentroid(env.scene[cam_name].data.output["rgb"])
        except Exception:
            return torch.zeros(env.num_envs, device=env.device)
        tTx, tTy = WRIST_ALIGN_PX
        tD = torch.sqrt((tCx - tTx) ** 2 + (tCy - tTy) ** 2)
        tCentered = (1.0 - 2.0 * tD).clamp(min=0.0)            # 1 at the grasp pixel, 0 half-frame away
        tSize = torch.tanh(tFrac * 15.0)                       # bigger red blob = closer = more grasp-ready
        return torch.nan_to_num(tVis * tCentered * tSize * _DENSE_SCALE[0], nan=0.0, posinf=1.0, neginf=0.0)
    tObj = env.scene[object_cfg.name]
    tTcp = _tcp(env, ee_frame_cfg)
    tDist = torch.norm(tObj.data.root_pos_w - tTcp, dim=1)
    tPhys = 1.0 - torch.tanh(12.0 * tDist)                     # PHYSICS: grasp-readiness (privileged)
    try:
        _, _, tVis, tFrac = _redCentroid(env.scene[cam_name].data.output["rgb"])
        tSize = tVis * (0.3 + 0.7 * torch.tanh(tFrac * 15.0))   # VISION: red-area size
    except Exception:
        tSize = torch.ones_like(tDist)
    return torch.nan_to_num(tPhys * tSize * _DENSE_SCALE[0], nan=0.0, posinf=1.0, neginf=0.0)


def reachSharp(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """reach. _REAL_SENSOR -> VISION ONLY (wrist red-area, real-measurable). else HYBRID (privileged physics
    1-tanh(8d) + near-bonus + observable wrist red-area = the ORIGINAL setup). x _DENSE_SCALE (decay)."""
    if _REAL_SENSOR[0]:
        try:
            _, _, _, tFrac = _redCentroid(env.scene["wrist"].data.output["rgb"])
            tWristProx = torch.tanh(tFrac * 15.0)
        except Exception:
            tWristProx = torch.zeros(env.num_envs, device=env.device)
        return torch.nan_to_num(tWristProx * _DENSE_SCALE[0], nan=0.0, posinf=1.0, neginf=0.0)
    tObj = env.scene[object_cfg.name]; tTcp = _tcp(env, ee_frame_cfg)
    tDist = torch.norm(tObj.data.root_pos_w - tTcp, dim=1)
    tCoarse = 1.0 - torch.tanh(8.0 * tDist)
    tNearBonus = (tDist < 0.06).float() * 0.5
    try:
        _, _, _, tFrac = _redCentroid(env.scene["wrist"].data.output["rgb"])
        tWristProx = torch.tanh(tFrac * 15.0)
    except Exception:
        tWristProx = torch.zeros_like(tDist)
    tOut = (tCoarse + tNearBonus + 0.5 * tWristProx) * _DENSE_SCALE[0]
    return torch.nan_to_num(tOut, nan=0.0, posinf=1.0, neginf=0.0)


GATE_W = 1.0   # multiplicative align-gating strength: reach * exp(GATE_W * align)
def reachAlignGated(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """Frontal/OFC GATING (alternative to additive reach+align): the SHARP reach BOOSTED multiplicatively by the
    wrist-align (visual servoing). reach_sharp * exp(GATE_W*align). align=0 (cube not in wrist view) -> exp(0)=1 ->
    reach SURVIVES (no dead zone); align high -> big boost -> alignment-first, can't farm reach alone."""
    tBase = reachSharp(env, object_cfg, ee_frame_cfg)
    tAl = alignWristCentroid(env)
    return tBase * torch.exp(GATE_W * tAl)


def sqGrasped(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame"),
               robot_cfg=SceneEntityCfg("robot")):
    """grasp. _REAL_SENSOR -> CURRENT x POSITION (real). else HYBRID = CURRENT x PHYSICS x POSITION (the ORIGINAL;
    physics 1-tanh(8d) = cube actually at the grasp center). CURRENT=both-finger force (real=motor current),
    POSITION=gripper qpos not fully closed (>=0.02, real=encoder)."""
    return _isGrasped(env, object_cfg, min_force=0.3, require_align=False).float()   # +dopamine PURE contact grasp (no align gate -> matches Squint is_grasping + our success metric)


HOLD_STEPS = 5      # SQUINT-MATCH 2026-06-23: place/lift credit only a grasp MAINTAINED this many consecutive steps
_grasp_hold = [None]
_held_cache = [None, -1]
def _held(env, object_cfg=SceneEntityCfg("cube")):
    """Squint HOLD_STEPS gate: consecutive _is_grasped steps (resets the moment the grasp breaks). Memoized per
    env step (common_step_counter) so place & lift share ONE counter update regardless of reward-term order/count."""
    tSid = int(getattr(env, "common_step_counter", -1))
    if _held_cache[1] == tSid and _held_cache[0] is not None and _held_cache[0].shape[0] == env.num_envs:
        return _held_cache[0]
    tG = _isGrasped(env, object_cfg, min_force=0.3, require_align=False)
    if _grasp_hold[0] is None or _grasp_hold[0].shape[0] != env.num_envs:
        _grasp_hold[0] = torch.zeros(env.num_envs, device=tG.device)
    _grasp_hold[0] = torch.where(tG, _grasp_hold[0] + 1.0, torch.zeros_like(_grasp_hold[0]))
    tHeld = (_grasp_hold[0] >= HOLD_STEPS).float()
    _held_cache[0], _held_cache[1] = tHeld, tSid
    return tHeld


def sqPlace(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame"),
             robot_cfg=SceneEntityCfg("robot")):
    """place = JOINT x grasp. _REAL_SENSOR -> grasp = CURRENT+POSITION (real). else HYBRID grasp = _is_grasped
    (force + privileged physics align) = the ORIGINAL. exp(-0.5*dist_to_rest) * grasp. (dist_to_rest = commanded
    arm target qpos = real-measurable; gentler decay so the ~4rad return is learnable.)"""
    # SQUINT-EXACT (per user): exp(-0.5*dist_to_rest) * is_grasped (instantaneous). Dropped reBot held+item_lifted gates.
    tGrasped = _isGrasped(env, object_cfg, min_force=0.3, require_align=False).float()
    return torch.exp(-0.5 * _distToRest(env)) * tGrasped


# ── FOLD CURRICULUM — ported from the author's earlier SO-101 project (the WORKING reference) 2026-06-25.
#    The reference taught the fold via lift_hold + a 3-joint graded fold + a sustained fold_hold; this delivery
#    had collapsed all that into a single weak `place` term -> teacher never learned the fold -> success=0.
#    Adapted to THIS delivery's REST_QPOS as the fold target (success here = reached_rest toward REST).
LIFT_GATE = 0.065      # cube z(m) counted as 'lifted' for the fold reward (2cm above the 0.06 rest; achievable).
FOLD_FRAMES = 3       # consecutive in-band frames before the fold bonus starts paying (user: "fold at 3 frames").
FOLD_TOL = 0.2        # rad: each of the 3 fold joints (pan/shoulder_lift/elbow) within this of REST = "folded".
#  USER 2026-06-25: success/fold use ONLY these 3 joints (wrists FREE) — the full-6-joint reached_rest was too
#  hard (wrists rarely return to REST). "when pan, shoulder_lift and elbow are all folded" = folded.

def _fold3Err(env):
    """(N,3) abs error of the 3 fold joints [pan(0), shoulder_lift(1), elbow(2)] vs REST (commanded target)."""
    tQ = env.action_manager.get_term("arm_action")._target_qpos
    return (tQ[:, :3] - REST_QPOS[:3].to(tQ.device)).abs()

_fold_streak = [None]
def foldReward(env, object_cfg=SceneEntityCfg("cube")):
    """FOLD = return the LIFTED+GRASPED arm to REST over ALL 6 joints (== the success reached_rest metric).
    FIX 2026-06-25: the old fold shaped only 3 joints (pan/shoulder/elbow) while success checks all 6 -> the
    wrists never folded -> reached_rest stuck 0 -> success 0. Now uses the full 6-joint dist (== reached_rest).
    Dense graded pull toward REST + a STREAK bonus that needs >=FOLD_FRAMES consecutive in-band frames and GROWS
    the longer it's sustained (replaces fold_hold; user design). lift_hold removed (was farmable hover)."""
    tN, tDevice = env.num_envs, env.device
    if _fold_streak[0] is None or _fold_streak[0].shape[0] != tN:
        _fold_streak[0] = torch.zeros(tN, dtype=torch.long, device=tDevice)
    _fold_streak[0][env.episode_length_buf <= 1] = 0
    tCubeZ = env.scene[object_cfg.name].data.root_pos_w[:, 2]
    tHeld = (tCubeZ >= LIFT_GATE) & _isGrasped(env, object_cfg, min_force=0.3, require_align=False)
    tE3 = _fold3Err(env)                                      # (N,3) pan/shoulder_lift/elbow err vs REST
    # GENTLE decay (ref-style) so there's gradient even when FAR from REST (at the grasp pose) -> pulls the arm
    # all the way to the fold. exp(-3) was too steep (~0 far away -> no fold signal -> policy stuck at lift+grasp).
    tCloseness = (torch.exp(-2.0 * tE3[:, 0]) + torch.exp(-0.5 * tE3[:, 1]) + torch.exp(-0.5 * tE3[:, 2])) / 3.0
    tInBand = tHeld & (tE3 < FOLD_TOL).all(dim=1)              # all 3 fold joints folded (wrists FREE) = success band
    _fold_streak[0] = torch.where(tInBand, _fold_streak[0] + 1, torch.zeros_like(_fold_streak[0]))
    tGrow = (_fold_streak[0].float() / 20.0).clamp(max=1.0)    # 0->1 over ~20 held frames -> "the longer it is held, the more it grows"
    tBonus = torch.where(_fold_streak[0] >= FOLD_FRAMES, tGrow, torch.zeros(tN, device=tDevice))
    return tHeld.float() * tCloseness + tBonus                  # dense approach pull (all 6 joints) + growing fold bonus


LIFT_FULL = 0.15      # lift reward saturates(=1.0) at +15cm above rest = cube_z 0.21 (user: "15 is full"; ~max reach).
SUCC_LIFT = 0.13      # SUCCESS = cube_z >= 0.13 (+7cm above rest) & grasped, held FOLD_FRAMES (user: "success is 7").
def liftReward(env, object_cfg=SceneEntityCfg("cube")):
    """GRADED lift (user 2026-06-25, re-added): reward GROWS with cube HEIGHT above rest (higher = more, capped
    at LIFT_FULL), gated by grasped -> 'the higher it lifts, the higher the score'. Unlike the old flat lift_hold, this pays more the
    higher the held cube goes, so the policy actually lifts (not just clears the gate). fold(2/3) still dominates,
    so this is a stepping stone, not a hover trap."""
    tCubeZ = env.scene[object_cfg.name].data.root_pos_w[:, 2]
    tGrasped = _isGrasped(env, object_cfg, min_force=0.3, require_align=False).float()
    tLiftH = ((tCubeZ - CUBE_REST_Z) / LIFT_FULL).clamp(0.0, 1.0)   # 0 at rest(0.06) -> 1.0 at +LIFT_FULL
    return tLiftH * tGrasped


def successReward(env, object_cfg=SceneEntityCfg("cube")):
    """OUTCOME bonus (user 2026-06-25): +1.0 every step the SUCCESS state holds (cube_z >= SUCC_LIFT & grasped).
    The lift reward is graded toward height; this is a flat bonus AT the success threshold so reaching+holding the
    success height is directly rewarded (reinforces the 3-frame success latch). NOTE: instantaneous condition (not
    the streak) so it gives dense signal; it does not mutate the success streak counter."""
    tCubeZ = env.scene[object_cfg.name].data.root_pos_w[:, 2]
    tGrasped = _isGrasped(env, object_cfg, min_force=0.3, require_align=False).float()
    return (tCubeZ >= SUCC_LIFT).float() * tGrasped


def sqLift(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """LIFT reward: reward raising the cube off the table WHILE gripping it -- the un-farmable success requirement
    the real-sensor rewards miss (grasp+place could be farmed with a force-touch+return WITHOUT lifting). Uses
    cube_z (PRIVILEGED, sim-only -- the deployed actor never sees it; only the reward does). Graded 0..1 over a
    5cm lift, x grasped(force) so it only pays for a lift that's actually held."""
    tCubeZ = env.scene[object_cfg.name].data.root_pos_w[:, 2]
    tGf = env.scene["gripper_contact"].data.force_matrix_w[:, 0, 0, :].norm(dim=-1)
    tJf = env.scene["jaw_contact"].data.force_matrix_w[:, 0, 0, :].norm(dim=-1)
    tDist = torch.norm(env.scene[object_cfg.name].data.root_pos_w - _tcp(env, ee_frame_cfg), dim=1)
    tHeld = _held(env, object_cfg)  # SQUINT-MATCH 2026-06-23: lift * held (grasp maintained >= HOLD_STEPS)  # was instantaneous
    tHeight = ((tCubeZ - CUBE_REST_Z) / 0.08).clamp(0.0, 1.0)   # SQUINT-MATCH 2026-06-23: full lift at 8cm (was 5cm)            # the jaws (force + at grasp center + grip width)
    return torch.nan_to_num(tHeight * tHeld, nan=0.0)                     # -> no knock/fly/push-from-below farming


def sqOpenApproach(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """OPEN-during-approach: reward opening the jaws (qpos -> 0.045) so the gripper can CLEAR the 4cm cube (REST
    0.020 = mouth ~2.3cm is too closed to fit over it), then FADE as alignment completes so close_after_align
    takes over (open x (1-align)). _REAL_SENSOR: approaching = cube visible in the wrist (eye-in-hand) view;
    else: cube within 0.15 of the TCP. x not_grasped x _DENSE_SCALE. (qpos UP=OPEN, DOWN=CLOSED.)"""
    tOpenAmt = ((_gq(env) - 0.035) / (0.045 - 0.035)).clamp(0.0, 1.0)   # 1=open(q>=0.045), 0 at the cube width
    tGf = env.scene["gripper_contact"].data.force_matrix_w[:, 0, 0, :].norm(dim=-1)
    tJf = env.scene["jaw_contact"].data.force_matrix_w[:, 0, 0, :].norm(dim=-1)
    tNotGrasped = (torch.minimum(tGf, tJf) < 0.3).float()
    tAligned = alignWristCentroid(env).clamp(0.0, 1.0)               # fade open as alignment completes
    if _REAL_SENSOR[0]:
        try:
            _, _, tVis, tF = _redCentroid(env.scene["wrist"].data.output["rgb"])
            tApproaching = tVis                                         # cube in the eye-in-hand view = near it
        except Exception:
            tApproaching = torch.zeros_like(tOpenAmt)
    else:
        tDist = torch.norm(env.scene[object_cfg.name].data.root_pos_w - _tcp(env, ee_frame_cfg), dim=1)
        tApproaching = (tDist < 0.15).float()
    return torch.nan_to_num(tOpenAmt * tApproaching * (1.0 - tAligned) * tNotGrasped * _DENSE_SCALE[0], nan=0.0)


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

def sqAlignHold(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """Pre-grasp ALIGNMENT-HOLD shaping (feedback v2: reach/align is poor). While the OPEN jaws are
    aligned ON the cube (cube within ALIGN_DIST of the TCP AND jaws OPEN, qpos >= ALIGN_OPEN_Q so the
    mouth clears the 4cm cube), reward GROWS with how long the alignment is HELD: a per-env counter
    increments each aligned step and resets the instant alignment breaks. reward = min(hold/HOLD_TARGET,1)
    so a long hold (~3s @10Hz = 30 steps) earns far more total than a brief touch. ALSO maintains the
    `latch`: once the hold reaches ALIGN_UNLOCK steps the env is align-confirmed (held until the cube
    drifts away / env reset), and _is_grasped REQUIRES it -> the model must align-and-HOLD before a grasp
    counts (grasp after align-hold). jaws-open self-gates this off once the gripper closes (qpos drops
    below ALIGN_OPEN_Q), so there is no circular dependency on _is_grasped."""
    tObj = env.scene[object_cfg.name]
    tTcp = _tcp(env, ee_frame_cfg)
    tDist = torch.norm(tObj.data.root_pos_w - tTcp, dim=1)
    tAligned = (tDist < ALIGN_DIST) & (_gq(env) >= ALIGN_OPEN_Q)
    tN = tDist.shape[0]
    tC = _align_count.get(0); tL = _align_latch.get(0)
    if tC is None or tC.shape[0] != tN or tC.device != tDist.device:
        tC = torch.zeros(tN, device=tDist.device)
        tL = torch.zeros(tN, dtype=torch.bool, device=tDist.device)
    tC = torch.where(tAligned, tC + 1.0, torch.zeros_like(tC))
    tPreLatched = tL                                            # latch BEFORE this step's update
    tL = (tL | (tC >= ALIGN_UNLOCK)) & (tDist < ALIGN_RESET_DIST)  # set on hold; drop when cube drifts / reset
    _align_count[0] = tC; _align_latch[0] = tL
    # NON-FARMABLE: pay the build-up ramp (0->1 over ALIGN_UNLOCK steps; longer hold = more) ONLY until
    # the hold latches; once latched -> 0. So the policy CANNOT hover forever to farm reward (that was a
    # 0-success local optimum); after the hold it must GRASP (now unlocked) to earn the bigger grasp reward.
    return (tC / float(ALIGN_UNLOCK)).clamp(0.0, 1.0) * tAligned.float() * (~tPreLatched).float()


def _alignHeld(env, dist):
    """latch from sq_align_hold: an aligned-hold (>= ALIGN_UNLOCK steps) was achieved this episode and not
    since lost. Used by sq_close_after_align to reward closing only AFTER the hold. False until
    sq_align_hold has run."""
    tL = _align_latch.get(0)
    tN = dist.shape[0]
    if tL is None or tL.shape[0] != tN or tL.device != dist.device:
        return torch.zeros(tN, dtype=torch.bool, device=dist.device)
    return tL


def sqCloseAfterAlign(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """CLOSE-after-align bridge (align -> grasp): reward CLOSING the gripper (qpos toward the grip width) IN
    PROPORTION to the align signal, only while NOT yet grasped. close = align x close_amt x not_grasped, so it is
    ALWAYS <= align_wrist (it only kicks in once aligned, and stops once the grasp FORCE forms -> grasp/place take
    over). Fixes the missing 'close' link that left the policy hovering open (align good, grasp never)."""
    tGf = env.scene["gripper_contact"].data.force_matrix_w[:, 0, 0, :].norm(dim=-1)
    tJf = env.scene["jaw_contact"].data.force_matrix_w[:, 0, 0, :].norm(dim=-1)
    tNotGrasped = (torch.minimum(tGf, tJf) < 0.3).float()
    tCloseAmt = ((ALIGN_OPEN_Q - _gq(env)) / (ALIGN_OPEN_Q - ALIGN_GRIP_Q)).clamp(0.0, 1.0)  # 0=open .. 1=grip width
    tAligned = alignWristCentroid(env)            # the align signal (physics or vision per _REAL_SENSOR; incl DENSE)
    return torch.nan_to_num(tAligned * tCloseAmt * tNotGrasped, nan=0.0)   # <= align_wrist by construction


def sqNotLifted(env, object_cfg=SceneEntityCfg("cube")):
    """not_lifted penalty. _REAL_SENSOR -> real proxy: 1 while NOT securely holding (current+position; the lift
    itself is rewarded via sq_place's return-to-rest). else HYBRID (the ORIGINAL): 1 while cube_z < CUBE_REST_Z
    + 1mm (privileged)."""
    tCubeZ = env.scene[object_cfg.name].data.root_pos_w[:, 2]
    return (tCubeZ < (CUBE_REST_Z + 0.001)).float()   # CONSTANT not_lifted (-1/3) = forward push to lift (un-gated)              # SQUINT: 1 while the cube is on the table


_GRIP_BIDX = [None]; _LOCAL_DOWN = [None]
def downOrientationPenalty(env, robot_cfg=SceneEntityCfg("robot")):
    """P1 wrist-DOWN constraint 2026-06-23: penalize the gripper not pointing straight DOWN so the eye-in-hand
    wrist cam always sees the floor/cube (never the background). 0 when straight down, ->2 when it tilts up."""
    from isaaclab.utils.math import quat_apply, quat_inv
    tR2 = env.scene[robot_cfg.name]
    if _GRIP_BIDX[0] is None:
        tBn = list(tR2.data.body_names)
        _GRIP_BIDX[0] = tBn.index("gripper_link") if "gripper_link" in tBn else len(tBn) - 1
        tDq = torch.tensor([[-0.4309, -0.5333, -0.5689, 0.4542]], device=tR2.device)   # SIM down orientation
        _LOCAL_DOWN[0] = quat_apply(quat_inv(tDq), torch.tensor([[0.0, 0.0, -1.0]], device=tR2.device))  # (1,3)
    tEq = tR2.data.body_quat_w[:, _GRIP_BIDX[0]]
    tWa = quat_apply(tEq, _LOCAL_DOWN[0].expand(tEq.shape[0], 3))   # gripper approach axis in world
    return (1.0 - (-tWa[:, 2])).clamp(min=0.0)                    # 0=straight down, grows as it tilts up


def graspCloseReward(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """GRASP SHAPING 1 (2026-06-23): dense pre-grasp -- reward CLOSING the gripper WHEN the TCP is at the cube.
    Bridges the sparse reaching->is_grasped gap (policy was stuck reaching, never closing). Cube pose is
    PRIVILEGED -> training reward ONLY, never a deploy input."""
    tObj = env.scene[object_cfg.name]
    tTcp = _tcp(env, ee_frame_cfg)
    tDist = torch.norm(tObj.data.root_pos_w - tTcp, dim=1)
    tNear = 1.0 - torch.tanh(35.0 * tDist)                       # TIGHT (2026-06-23): ~1 only when the cube is truly in the jaw MOUTH (<~3cm of grasp center) -> closing on empty air earns ~0 (no contact-less close farming)
    tR = env.scene["robot"]; tGi = tR.data.joint_names.index("gripper")
    tGq = tR.data.joint_pos[:, tGi].clamp(0.0, 0.045)
    tClosed = 1.0 - (tGq / 0.045)                                # 0 open .. 1 fully closed
    return tNear * tClosed                                        # reward closing ONLY when at the cube


def graspContactReward(env, object_cfg=SceneEntityCfg("cube")):
    """GRASP SHAPING 2 (2026-06-23): CONTINUOUS contact -- tanh(min jaw force) so partial/forming contact gives a
    gradient (binary is_grasped gave none until a full grasp). Sensors pre-filtered to the cube body."""
    tGrip = env.scene["gripper_contact"].data.force_matrix_w[:, 0, 0, :].norm(dim=-1)
    tJaw = env.scene["jaw_contact"].data.force_matrix_w[:, 0, 0, :].norm(dim=-1)
    tObj = env.scene[object_cfg.name]; tTcpw = _tcp(env, SceneEntityCfg("ee_frame"))
    tInJaw = (torch.norm(tObj.data.root_pos_w - tTcpw, dim=1) < GRASP_ZONE).float()   # PHASE3 (user): reward contact ONLY when the cube is in the inner jaw zone
    return torch.tanh(torch.minimum(tGrip, tJaw)) * tInJaw


_succ_streak = [None]    # per-env consecutive in-fold-band frames (success fires at FOLD_FRAMES)
def isFoldSuccess(env, object_cfg=SceneEntityCfg("cube")):
    """SUCCESS = fold held FOLD_FRAMES consecutive steps (was SQUINT instantaneous latch).
      item_lifted  = cube_z >= CUBE_REST_Z + 0.001   (~1mm off the table, = Squint half_size+1e-3)
      grasped      = both fingers in contact-force on the cube (_is_grasped, angle-gated 110deg)
      reached_rest = ||commanded arm target - REST_QPOS|| < 0.2
    Returns a (N,) bool tensor. (min_force kept at our calibrated 0.3 -- Isaac's contact-sensor force
    scale differs from Squint's SAPIEN pairwise force, so the 0.5 N literal does not transfer.)"""
    # SUCCESS (user 2026-06-25) = the FOLD held FOLD_FRAMES(3) consecutive steps -> SAME condition as
    # fold_reward's in-band streak, so reward and success are fully consistent. Per-step cond = lifted(>=LIFT_GATE)
    # & grasped & reached_rest(||cmd arm - REST||<0.2 over all 6 joints). Latches once the streak reaches 3.
    tN, tDevice = env.num_envs, env.device
    if _succ_streak[0] is None or _succ_streak[0].shape[0] != tN:
        _succ_streak[0] = torch.zeros(tN, dtype=torch.long, device=tDevice)
    tJustReset = env.episode_length_buf <= 1
    _succ_streak[0][tJustReset] = 0
    tCubeZ = env.scene[object_cfg.name].data.root_pos_w[:, 2]
    # SUCCESS (user 2026-06-25) = LIFT the grasped cube to >= SUCC_LIFT(0.15) and hold FOLD_FRAMES frames.
    # fold/return-to-REST DROPPED entirely — the task is now "grasp + lift high".
    tLifted15 = tCubeZ >= SUCC_LIFT
    tGrasped = _isGrasped(env, object_cfg, min_force=0.3, require_align=False)
    tCond = tLifted15 & tGrasped & (~tJustReset)
    _succ_streak[0] = torch.where(tCond, _succ_streak[0] + 1, torch.zeros_like(_succ_streak[0]))
    tSucc = _succ_streak[0] >= FOLD_FRAMES              # success = (cube>=15cm & grasped) held 3 frames
    if os.environ.get("DIAG_FOLD"):
        _fold_diag[0] += 1
        if _fold_diag[0] % 200 == 0:
            tV = ~tJustReset
            print(f"[FOLD-DIAG] lifted15(>=0.15)={float((tLifted15&tV).float().mean()):.3f} "
                  f"grasped={float((tGrasped&tV).float().mean()):.3f} "
                  f"cond={float(tCond.float().mean()):.3f} SUCC(streak>=3)={float(tSucc.float().mean()):.3f} "
                  f"max_streak={int(_succ_streak[0].max())} cube_z_mean={float(tCubeZ.mean()):.3f} "
                  f"cube_z_max={float(tCubeZ.max()):.3f}", flush=True)
    return tSucc
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


def successLatch(env, object_cfg=SceneEntityCfg("cube")):
    """Termination term that NEVER terminates. Side-effect: latch per-env success while
    the terminal state is still valid (pre-reset). Returns an all-False (N,) bool tensor."""
    tN = env.num_envs
    tSucc = isFoldSuccess(env, object_cfg)
    if _succ_flag[0] is None or _succ_flag[0].shape[0] != tN:
        _succ_flag[0] = torch.zeros(tN, dtype=torch.bool, device=tSucc.device)
    _succ_flag[0] |= tSucc
    return torch.zeros(tN, dtype=torch.bool, device=tSucc.device)


def countSuccesses(env, env_ids):
    """Reset event: tally latched successes for the just-finished env_ids, then clear them."""
    tFlag = _succ_flag[0]
    if tFlag is None:
        return
    tFin = tFlag[env_ids]
    _succ_total[0] += int(tFin.sum().item())
    _ep_total[0] += int(tFin.numel())
    _succ_recent.extend(int(tV) for tV in tFin.tolist())
    if len(_succ_recent) > 500:
        del _succ_recent[:-500]
    if _last_ep_succ[0] is None or _last_ep_succ[0].shape[0] != tFlag.shape[0]:
        _last_ep_succ[0] = torch.zeros_like(tFlag)
    _last_ep_succ[0][env_ids] = tFin
    tFlag[env_ids] = False


def _touchingTable(env):
    """Detect gripper/jaw touching anything other than the cube (i.e. the table)."""
    tGn = env.scene["gripper_contact"].data.net_forces_w[:, 0, :]
    tGc = env.scene["gripper_contact"].data.force_matrix_w[:, 0, 0, :]
    tJn = env.scene["jaw_contact"].data.net_forces_w[:, 0, :]
    tJc = env.scene["jaw_contact"].data.force_matrix_w[:, 0, 0, :]
    return ((tGn - tGc).norm(dim=-1) >= 0.01) | ((tJn - tJc).norm(dim=-1) >= 0.01)


def tableCollisionPenalty(
    env,
    table_height=0.04,
    ee_frame_cfg=SceneEntityCfg("ee_frame"),
):
    """Penalty for touching the table surface."""
    return _touchingTable(env).float()


# ── Smoothness penalties ──
_prev_action = [None]
_prev_prev_action = [None]

def actionRatePenalty(env):
    """||a_t - a_{t-1}||² — penalize sudden action changes."""
    tCur = env.action_manager.action
    if _prev_action[0] is None or _prev_action[0].shape != tCur.shape:
        _prev_action[0] = tCur.clone()
    tPenalty = ((tCur - _prev_action[0]) ** 2).sum(dim=-1)
    _prev_action[0] = tCur.clone()
    return tPenalty


def actionJerkPenalty(env):
    """||a_t - 2*a_{t-1} + a_{t-2}||² — penalize acceleration changes (tremor)."""
    tCur = env.action_manager.action
    if _prev_action[0] is None or _prev_action[0].shape != tCur.shape:
        _prev_action[0] = tCur.clone()
    if _prev_prev_action[0] is None or _prev_prev_action[0].shape != tCur.shape:
        _prev_prev_action[0] = tCur.clone()
    tJerk = tCur - 2 * _prev_action[0] + _prev_prev_action[0]
    tPenalty = (tJerk ** 2).sum(dim=-1)
    _prev_prev_action[0] = _prev_action[0].clone()
    _prev_action[0] = tCur.clone()
    return tPenalty


def jointVelPenalty(env, robot_cfg=SceneEntityCfg("robot")):
    """||qvel||^2 over ALL joints -- velocity regularization (calm motion, no twisting/whipping).
    Pairs with action_rate_penalty to smooth the contorted approach that transfers badly to real."""
    tQvel = env.scene[robot_cfg.name].data.joint_vel
    return (tQvel ** 2).sum(dim=-1)
