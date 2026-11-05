"""June (simulation S1) reBot LiftCube environment configuration on top of LeIsaac.

Needs Isaac Sim 5.1 / Isaac Lab 2.3 + the LeIsaac reBot fork; not executable in CI (compile-checked only).
"""
from .env_config import configureEnv, applyMotorLimits
