# rebot_grasp_rl — Project 2 trainer (staged C51-SAC on cube detections)

**Status:** main code path of the repository (July–August 2026). Produced the 120 benches and the 2026-11-02
re-evaluation. Training and evaluation need Isaac Sim; the networks, target projection, dashboard helpers, robot
profiles and evaluation hooks import and run with torch only (covered by `tests/test_characterization.py`).

**Inputs → outputs.** Isaac Lab LiftCube scene (`rebot_grasp_env`), YOLO-OBB cube detector weights (`--detector`),
environment knobs (`configs/project2_*.env`) → checkpoints (`search_last.pt`, `approach_last.pt`), live web
dashboard, bench heatmaps and, with `EVAL_OUT_DIR`, per-episode `episodes.csv` + `trainer_summary.json`.

**Entry points**
* `python -m rebot_grasp_rl.train ...` — training, benches (`TP_EVAL=1`) and re-evaluation; see the README Pipeline and
  `scripts/eval/project2_eval.sh`.

**Layout**

| Module | What |
|---|---|
| `train.py` | the trainer script: Isaac app launch, scene set-up for the OURS / OFFICIAL robot, detection observations, search / approach / align / grasp / lift phases, C51-SAC updates, curriculum, eval-set benchmark, web dashboard |
| `nets.py` | cross-attention actor and distributional (C51) critic over front / wrist / orientation / cube / aim tokens |
| `target_projection.py` | C51 target projection (pure function) |
| `config.py` | command-line options (parsed at import, before the Isaac app starts) |
| `robot_profile.py` | OURS vs OFFICIAL robot model constants (frozen dataclasses) |
| `runtime.py` | shared runtime namespace `RT` filled by the trainer after the simulator boots |
| `eval_hooks.py` | opt-in seeding, per-episode CSV, summary and rollout video (no-op unless `EVAL_OUT_DIR` / `EVAL_VIDEO_DIR`) |
| `dashboard_render.py` | PNG / SVG / HTML helpers and image overlays of the dashboard |

**Not restyled beyond the mechanical rename** (documented in `docs/refactor_2026-11.md`): `train.py` stays a
script-style module with module-level state, because the Isaac app has to be launched before Isaac Lab is imported
and every phase reads the same globals; its module-level names, the `RT` / `RobotProfile` attribute names,
nn.Module attributes (checkpoint keys) and argparse destinations keep their original spelling.

**Provenance:** was `src/det_xattn/` (trainer copy of 2026-08-02 15:39, see `docs/provenance.md` §9); the move and
rename are checked by `analysis/refactor_equivalence.py`.
