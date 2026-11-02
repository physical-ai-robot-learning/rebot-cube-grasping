"""Staged pick reward (reach -> align -> grasp -> lift) on batched state tensors.

Single SAC policy with internal staging (the reference's proven recipe). Kept
intentionally small & bounded — docs warn dense rewards invite sim-only hacking,
and unbounded penalties blow up value supports.

All inputs are torch tensors on the same device, batched over envs [N, ...]. Positions
are in the env-LOCAL frame (env grid offset already subtracted) so ``table_z`` is a valid
absolute height reference for every env.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch


def compute_pick_reward(
    state: Dict[str, torch.Tensor],
    rcfg,
    env_cfg,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Returns (reward[N], success[N] bool, components dict)."""
    tcp = state["tcp_pos"]            # [N,3] local
    cube = state["cube_pos"]          # [N,3] local
    cube_z = cube[:, 2]               # [N]
    is_grasped = state["is_grasped"].bool()  # near + gripper commanded closed
    held = state["held"].bool()              # near + cube physically lifted off table
    fallen = state["fallen"].bool()
    action = state["action"]          # [N,7]
    prev_action = state["prev_action"]  # [N,7]

    rest_cube_z = env_cfg.table_z + env_cfg.cube_size / 2.0
    success_z = env_cfg.table_z + env_cfg.success_lift_height

    dist3 = torch.linalg.norm(cube - tcp, dim=-1)             # [N]
    dist_xy = torch.linalg.norm(cube[:, :2] - tcp[:, :2], dim=-1)

    # --- staged shaping ---
    r_reach = rcfg.w_reach * (1.0 - torch.tanh(3.0 * dist3))      # in (0,1], closer = higher
    r_align = -rcfg.w_xy_align * dist_xy
    r_grasp = rcfg.w_grasp * is_grasped.float()
    lift_gain = torch.clamp(cube_z - rest_cube_z, min=0.0)        # how high above rest
    grasp_lift = state["grasp_lift"].bool()                      # CLOSED gripper holding the cube (genuine)
    r_lift = rcfg.w_lift * lift_gain * grasp_lift.float()        # NOT a flung cube near an open gripper

    # VERTICALITY shaping: reward a square gripper while APPROACHING the cube (prox), but ONLY before the
    # grasp -- once holding, the wrist MUST tilt to ~35 deg to lift (no vertical lift pose exists), so a
    # verticality term there would fight the lift. (1 - tilt/90) in [0,1]; this keeps the grasp FORMATION
    # square (off the ~41-deg lean) while leaving the lift free.
    tilt_deg = state.get("tilt_deg", torch.zeros_like(dist3))
    prox = torch.clamp(1.0 - dist3 / 0.15, min=0.0)              # 1 at the cube, 0 beyond 15cm
    r_vertical = getattr(rcfg, "w_vertical", 0.0) * (1.0 - tilt_deg / 90.0) * prox * (~is_grasped).float()

    # WRIST-CENTERING shaping: reward AIMING the wrist cam at the cube while approaching (pre-grasp) -> the
    # policy learns the wrist-driven servo (J4 keeps the cube near the wrist-image center). CONTINUOUS in the
    # clean analytic (u,v): center=1 at image-center, 0 at the frame edge -> a real gradient (NOT a flat
    # visibility bonus, which in analytic mode would be ~constant=hover-harvest). Gated by visible+prox+(pre-grasp).
    wrist_vis = state.get("wrist_visible", torch.zeros_like(dist3))
    wrist_un = state.get("wrist_un", torch.zeros_like(dist3))
    wrist_vn = state.get("wrist_vn", torch.zeros_like(dist3))
    wrist_center = (1.0 - torch.sqrt(wrist_un ** 2 + wrist_vn ** 2)).clamp(min=0.0, max=1.0)
    r_wrist = getattr(rcfg, "w_wrist_center", 0.0) * wrist_vis * wrist_center * prox * (~is_grasped).float()

    # ANTI-BOB / commit-and-hold (reflecting the real deploy: the gripper bobbed away before the grip
    # torque could build). Reward LOW arm speed while NEAR + CLOSING on the cube -> the policy SETTLES
    # and holds the grasp instead of oscillating. Bounded in [0, w_hold_still].
    near = state.get("near", torch.zeros_like(dist3, dtype=torch.bool))
    closing = state.get("closing", torch.zeros_like(dist3, dtype=torch.bool))
    arm_speed = state.get("arm_speed", torch.zeros_like(dist3))
    r_hold_still = getattr(rcfg, "w_hold_still", 0.0) * (near & closing).float() * (1.0 - torch.tanh(arm_speed))

    # instantaneous success; the env requires it SUSTAINED (success_hold_steps) and pays the bonus,
    # so a one-frame fling cannot collect it.
    success = grasp_lift & (cube_z >= success_z)

    # --- penalties (bounded) ---
    p_action = rcfg.w_action * torch.sum((action - prev_action) ** 2, dim=-1)
    p_time = rcfg.w_time * torch.ones_like(dist3)
    p_fall = rcfg.fall_penalty * fallen.float()
    # SAFETY: penalize driving the gripper below the table top (continuous in penetration depth,
    # so the gradient pushes the tcp back up). Clear penetration also ends the episode (pick_env).
    p_table = rcfg.w_table * state.get("table_pen", torch.zeros_like(dist3))
    # SAFETY: penalize swinging the gripper backward (-y) past the +y working region (continuous
    # -> gradient pushes it forward). Gross backward also ends the episode (pick_env y_back_hit).
    p_yback = rcfg.w_yback * state.get("y_back", torch.zeros_like(dist3))

    # NOTE: success_bonus is added by the env once the grasp+lift is SUSTAINED (see pick_env.step).
    reward = (r_reach + r_align + r_grasp + r_lift + r_vertical + r_wrist + r_hold_still
              - p_action - p_time - p_fall - p_table - p_yback)

    comp = {
        "r_reach": r_reach, "r_align": r_align, "r_grasp": r_grasp,
        "r_lift": r_lift, "r_vertical": r_vertical, "r_wrist": r_wrist, "r_hold_still": r_hold_still,
        "p_action": p_action, "p_time": p_time, "p_fall": p_fall, "p_table": p_table, "p_yback": p_yback,
        "dist3": dist3, "lift_gain": lift_gain, "tilt_deg": tilt_deg,
    }
    return reward, success, comp
