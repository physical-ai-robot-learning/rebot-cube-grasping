# Source from the repository root before any Isaac stage:  source scripts/isaac_env.sh
# REBOT_ISAAC_WS = the Isaac workspace that holds LeIsaac (reBot fork) and its assets (default ~/workspace/rebot_isaac_rl).
export REBOT_ISAAC_WS="${REBOT_ISAAC_WS:-$HOME/workspace/rebot_isaac_rl}"
export LEISAAC_ASSETS_ROOT="${LEISAAC_ASSETS_ROOT:-$REBOT_ISAAC_WS/leisaac/assets}"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PY="${PY:-$REBOT_ISAAC_WS/.venv/bin/python}"   # Isaac Sim 5.1 + Isaac Lab 2.3 venv (Python 3.11)
