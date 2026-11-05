# archive/june_sim_tools/ — June diagnostics and renders (verbatim)

One-off Isaac scripts of the June stage, moved unchanged from `src/rebot_sim2real/tools/` on 2026-11-05 (their
normalised-AST fingerprints in `analysis/code_fingerprints.json` still match). They import the old package name
`rebot_sim2real` and are not runnable as they are; the current names are in `src/rebot_teacher_student/`.

* `diag_*.py` — camera, gripper stroke / geometry / mounts, contact API, domain-randomisation and wrist-camera probes;
  `diag_gripper_stroke.py` measured the gripper opening direction used in `docs/deploy/DEPLOY_SPEC.md`.
* `render_successes.py`, `render_teacher_hq.py`, `render_teacher_2cam.py`, `camera_spec_render.py` — produced
  `media/success_videos_v1/`, the June teacher GIFs and `media/cam_spec/` (see `media/README.md`).
