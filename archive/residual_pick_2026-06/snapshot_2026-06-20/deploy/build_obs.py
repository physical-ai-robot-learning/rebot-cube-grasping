"""Assemble the policy observation from REAL-ROBOT components (deployment).

Layout MUST match src/rebot_sim2real/envs/pick_env.py::_compute_state. Two modes:
  (a) default (43):  [ qpos8 | qvel8 | target7 | dist_rest6 | cube_pos3 | cube_quat4 | tcp3 | (cube-tcp)3 | grasped1 ]
  (b) keypoint (39): [ qpos8 | qvel8 | target7 | dist_rest6 | tcp3 | grasped1 | top_uv3 | wrist_uv3 ]
      (project (b): privileged cube_pos/quat/relative DROPPED; the cube enters ONLY as the top-cam (u,v)
       token from the sim-trained CNN (deploy/cube_detector_io.py) + the wrist (u,v) — pass `top_feat`.)
  (+ optional TORQUE TAIL: arm_torque6 | grip_torque1  when the policy was trained with obs_torque)
On a real robot every component is proprioception/FK EXCEPT the (u,v) tokens (perception) and, in (a),
cube_pos (perception) + cube_quat (≈identity, yaw_rand=0). Positions are in the ENV-LOCAL frame the
policy trained in (= the robot base/world frame on the real rig).
"""
from __future__ import annotations
import torch

OBS_DIM = 43                 # (a) base (no torque); (b) keypoint = 39; +7 torque; +3 per (u,v) token
# (a) component slices (for inspection / overwriting)
SL_QPOS = slice(0, 8); SL_QVEL = slice(8, 16); SL_TARGET = slice(16, 23)
SL_DIST_REST = slice(23, 29); SL_CUBE_POS = slice(29, 32); SL_CUBE_QUAT = slice(32, 36)
SL_TCP_POS = slice(36, 39); SL_CUBE_OFF = slice(39, 42); SL_GRASPED = slice(42, 43)
SL_TORQUE = slice(43, 50)    # arm_torque6 + grip_torque1 (only present when obs_torque)

IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)   # wxyz; cube_quat at deploy (axis-aligned, yaw_rand=0)


def build_obs(arm_qpos, grip_qpos, arm_qvel, grip_qvel, arm_target, grip_target,
              rest_arm, cube_pos, tcp_pos, is_grasped, cube_quat=None,
              arm_torque=None, grip_torque=None, torque_obs_scale=0.2, wrist_feat=None, top_feat=None):
    """All inputs torch tensors [N,k] on one device:
      arm_qpos[N,6] grip_qpos[N,2] arm_qvel[N,6] grip_qvel[N,2] arm_target[N,6] grip_target[N,1]
      rest_arm[N,6] cube_pos[N,3] tcp_pos[N,3] is_grasped[N,1]  cube_quat[N,4] (default identity)
      arm_torque[N,6] grip_torque[N,1] -> append a normalized TORQUE TAIL (obs_torque policy).
      top_feat[N,3]   -> project (b): use the KEYPOINT layout (drop cube_pos/quat/relative); top (u,v,vis).
      wrist_feat[N,3] -> append the WRIST detection tail (u_norm, v_norm, visible) (obs_wrist policy).
    Returns obs[N, 43|39 (+7 torque) (+3 per token)]. Order/scales MUST match the training cfg/pick_env."""
    qpos8 = torch.cat([arm_qpos, grip_qpos], dim=-1)
    qvel8 = torch.cat([arm_qvel, grip_qvel], dim=-1)
    target7 = torch.cat([arm_target, grip_target], dim=-1)
    dist_to_rest = arm_qpos - rest_arm
    if top_feat is not None:
        # project (b): keypoint bottleneck -> NO privileged cube_pos3/quat4/(cube-tcp)3
        obs = torch.cat([qpos8, qvel8, target7, dist_to_rest, tcp_pos, is_grasped], dim=-1)
    else:
        if cube_quat is None:
            cube_quat = torch.tensor(IDENTITY_QUAT, device=arm_qpos.device).expand(arm_qpos.shape[0], 4)
        obs = torch.cat([qpos8, qvel8, target7, dist_to_rest,
                         cube_pos, cube_quat, tcp_pos, (cube_pos - tcp_pos), is_grasped], dim=-1)
    if arm_torque is not None:
        tau = torch.cat([arm_torque, grip_torque], dim=-1) * torque_obs_scale   # match sim normalization
        obs = torch.cat([obs, tau], dim=-1)
    if top_feat is not None:                         # project (b): top-cam token BEFORE wrist (layout order)
        obs = torch.cat([obs, top_feat], dim=-1)
    if wrist_feat is not None:
        obs = torch.cat([obs, wrist_feat], dim=-1)   # (u_norm, v_norm, visible); appended LAST
    return obs
