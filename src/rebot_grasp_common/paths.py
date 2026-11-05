"""Machine-specific locations used by the Isaac stages, in one place, each overridable by an environment variable.

    REBOT_ISAAC_WS   Isaac workspace holding the LeIsaac fork (reBot configuration) and its assets
                     (default ~/workspace/rebot_isaac_rl; scripts/isaac_env.sh exports it)
    LEISAAC_ASSETS   <REBOT_ISAAC_WS>/leisaac/assets (the trainers set LEISAAC_ASSETS_ROOT to it when unset)
    YOLO_EXPORT_DIR  default output of the trainer's detector-data export mode (EXPORT_DIR overrides it at run time)

The values are plain strings because the original code builds paths by string concatenation.
Usage:  from rebot_grasp_common.paths import ISAAC_WS, LEISAAC_ASSETS
"""
import os

ISAAC_WS = os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl"))
LEISAAC_ASSETS = ISAAC_WS + "/leisaac/assets"
YOLO_EXPORT_DIR = os.path.expanduser("~/yolo_cube")
