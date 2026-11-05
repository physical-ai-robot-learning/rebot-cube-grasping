"""Squint-exact reward for SO101 LiftCube.

Stage: June environment.
Was src/rebot_sim2real/env/maniskill_rewards.py (moved and renamed to the OA style on 2026-11-05; docs/refactor_2026-11.md).
Needs Isaac Sim 5.1 / Isaac Lab 2.3 + the LeIsaac reBot fork; not executable in CI (compile-checked only).
"""
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


def _isGrasped(env, object_cfg=SceneEntityCfg("cube"), min_force=0.3, max_angle_deg=110, align_tol=0.14):
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
    tObj = env.scene[object_cfg.name]
    tTcp = _tcp(env, SceneEntityCfg("ee_frame"))
    tDist = torch.norm(tObj.data.root_pos_w - tTcp, dim=1)
    tAligned = tDist < align_tol                       # cube in the grasp zone (no off-center edge-catch)
    tHeld = _alignHeld(env, tDist)                    # AND an align-HOLD was achieved first -> GRASP AFTER
    return tGripOk & tJawOk & tAligned & tHeld         # ALIGN (the model must align+hold before a grasp counts).
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
    tHNow = (raw_reward > 0.001).float().mean().item()
    _hit_rates[key] = 0.99 * _hit_rates[key] + 0.01 * tHNow  # EMA
    tH = _hit_rates[key]
    tMult = _DOPAMINE_LAMBDA + (1 - _DOPAMINE_LAMBDA) * (1 - tH) ** _DOPAMINE_ALPHA
    return raw_reward * tMult


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
    tTgt = _getTargetQpos(env)[:, :6]
    return torch.norm(tTgt - REST_QPOS.to(tTgt.device), dim=1)


def sqReaching(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """Squint reaching: 1 - tanh(5 * dist(tcp, cube)). Ungated, stationary."""
    tObj = env.scene[object_cfg.name]
    tTcp = _tcp(env, ee_frame_cfg)
    tDist = torch.norm(tObj.data.root_pos_w - tTcp, dim=1)
    return 1.0 - torch.tanh(5.0 * tDist)


def sqGrasped(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame"),
               robot_cfg=SceneEntityCfg("robot")):
    """Squint: binary +1 when grasped. Ungated (no _near_mouth), no dopamine."""
    return _isGrasped(env, object_cfg, min_force=0.3).float()


def sqPlace(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame"),
             robot_cfg=SceneEntityCfg("robot")):
    """Place / return-to-rest while holding: exp(-0.5*dist_to_rest) * is_grasped.
    GENTLER decay than Squint's exp(-2*) -- the reBot grasp pose sits ~4 rad from REST (wrists rotated),
    where exp(-2*4)~3e-4 gives NO gradient to return; exp(-0.5*4)=0.135 keeps a real pull toward REST,
    so the policy actually learns the return (-> reached_rest -> success) instead of holding the cube out."""
    tGrasped = _isGrasped(env, object_cfg, min_force=0.3).float()
    return torch.exp(-0.5 * _distToRest(env)) * tGrasped


def sqOpenApproach(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """Pre-grasp shaping: reward opening the jaw WIDE while approaching the cube and NOT yet grasped,
    so the gripper arrives open and can descend over the cube. Gated by ~grasped so it stops the instant
    a grasp forms (never fights the grasp).
    MEASURED gripper convention (diag_gripper_stroke): qpos UP = OPEN (mouth wider), DOWN = CLOSED;
    mouth=4cm (cube) at q~0.035, clears the cube at q>0.04. (The old code rewarded q->0 as "open" which
    was BACKWARDS — q->0 is fully CLOSED.) open_amt=1 at the open cap (q>=0.05), 0 at the grip width."""
    tObj = env.scene[object_cfg.name]
    tTcp = _tcp(env, ee_frame_cfg)
    tDist = torch.norm(tObj.data.root_pos_w - tTcp, dim=1)
    tOpenAmt = ((_gq(env) - 0.035) / (0.045 - 0.035)).clamp(0.0, 1.0)  # 1=open(q>=0.045 cap), 0=at grip(q<=0.035)
    tApproaching = (tDist < 0.15).float()
    tNotGrasped = (~_isGrasped(env, object_cfg, min_force=0.3)).float()
    return tOpenAmt * tApproaching * tNotGrasped


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
    """CLOSE-after-align shaping (user idea): once the align-hold LATCH is set (the OPEN jaws were held
    aligned on the cube) and the cube is still in the grasp zone and NOT yet grasped, reward CLOSING the
    gripper (qpos descending from the open width ALIGN_OPEN_Q toward the grip width ALIGN_GRIP_Q). This
    bridges the hold -> close -> grasp transition (the missing link that left the policy hovering open).
    Gated by latch + near + ~grasped, so it fires only AFTER an align-hold and stops once the grasp forms
    (the bigger grasp/place reward then takes over)."""
    tObj = env.scene[object_cfg.name]
    tTcp = _tcp(env, ee_frame_cfg)
    tDist = torch.norm(tObj.data.root_pos_w - tTcp, dim=1)
    tLatch = _alignHeld(env, tDist)
    tNear = tDist < ALIGN_RESET_DIST
    tNotGrasped = ~_isGrasped(env, object_cfg, min_force=0.3)
    tCloseAmt = ((ALIGN_OPEN_Q - _gq(env)) / (ALIGN_OPEN_Q - ALIGN_GRIP_Q)).clamp(0.0, 1.0)  # 1=at grip width
    return tCloseAmt * (tLatch & tNear & tNotGrasped).float()


def sqNotLifted(env, object_cfg=SceneEntityCfg("cube")):
    """Squint not-lifted penalty (returns 1 while on table). Threshold ~5mm above rest (was 15cm)."""
    tObj = env.scene[object_cfg.name]
    tCubeZ = tObj.data.root_pos_w[:, 2]
    tItemLifted = tCubeZ >= (CUBE_REST_Z + 0.005)
    return (~tItemLifted).float()


def isFoldSuccess(env, object_cfg=SceneEntityCfg("cube")):
    """SUCCESS = SQUINT-EXACT (lift.py:354-382 evaluate()): item_lifted & is_grasped & reached_rest_qpos.
      item_lifted  = cube_z >= CUBE_REST_Z + 0.001   (~1mm off the table, = Squint half_size+1e-3)
      grasped      = both fingers in contact-force on the cube (_is_grasped, angle-gated 110deg)
      reached_rest = ||commanded arm target - REST_QPOS|| < 0.2
    Returns a (N,) bool tensor. (min_force kept at our calibrated 0.3 -- Isaac's contact-sensor force
    scale differs from Squint's SAPIEN pairwise force, so the 0.5 N literal does not transfer.)"""
    tObj = env.scene[object_cfg.name]
    tCubeZ = tObj.data.root_pos_w[:, 2]
    tItemLifted = tCubeZ >= (CUBE_REST_Z + 0.001)     # Squint: item.z >= half_size + 1e-3
    tGrasped = _isGrasped(env, object_cfg, min_force=0.3)
    tReachedRest = _distToRest(env) < 0.2           # Squint reached_rest_qpos < 0.2
    return tItemLifted & tGrasped & tReachedRest


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
