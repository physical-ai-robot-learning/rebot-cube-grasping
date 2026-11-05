"""Project 2 simulation environment: the July (S4) version of the reBot LiftCube configuration on top of LeIsaac.

Modules: env_config (observation / action / reward / success terms, motor limits, dynamics randomisation),
rewards (reward terms and success latches), target_delta_action (delta-on-target joint action), visual_obs
(camera mounts and visual randomisation). Imported by rebot_grasp_rl.train. Needs Isaac Sim 5.1 / Isaac Lab 2.3 + the LeIsaac reBot fork; not executable in CI (compile-checked only).
"""
from .env_config import configureEnv, applyMotorLimits
