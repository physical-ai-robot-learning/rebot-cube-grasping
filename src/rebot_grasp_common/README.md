# rebot_grasp_common — shared helpers

**Status:** new in the 2026-11 refactor. **Inputs → outputs:** environment variables → path strings.
**Entry points:** none (library). **Layout:** `paths.py` — `ISAAC_WS` (`REBOT_ISAAC_WS`, default
`~/workspace/rebot_isaac_rl`), `LEISAAC_ASSETS`, `YOLO_EXPORT_DIR`. **Provenance:** collects the path defaults that
were repeated in the trainers.
