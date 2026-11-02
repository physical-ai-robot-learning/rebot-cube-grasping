"""rbot_calculator — reBot FK/IK engine + config (for the FSM-based deploy stack only).

The old RL training stack (state-based SAC · n-step/RLPD replay buffer · ResFiT residual agent · rewards · factory ·
train) was removed — this project uses the current FK/IK + FSM (RuleBaseController) approach, not RL.
Remaining modules:
  * base_controller — RuleBaseController: FK (matched to Pinocchio/URDF) + FD-Jacobian DLS IK engine (used by deploy for reach/IK).
  * config          — EnvConfig / load_train_config: base_pos, joint limits, cube_size, table_z, reach parameters (loaded by deploy).
(Renamed from the old directory name 'rl' — the actual content is the reBot FK/IK engine.)
"""
