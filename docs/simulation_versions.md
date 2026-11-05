# Simulation versions, June → August 2026

Which simulated robot, scene and environment code each stage used, and which version is the latest one actually used.

| # | Dates | Robot model | Scene | Environment code / cameras | Used by | Evidence |
|---|---|---|---|---|---|---|
| S0 | 2026-06-08 | OURS: the author's URDF→USD conversion of the vendor reBot URDF (`rebot_follower.usd`, MD5 `39c27cdc…`, 06-08 19:34) | LeIsaac `table_with_cube` with 4 cubes, grid floor | first import, no task code | set-up only (`media/readme/sim_scene_2026-06-08.jpg`) | training workstation `rebot_isaac_rl/renders/` |
| S1 | 06-09 → 06-12 | OURS (same USD) | LeIsaac LiftCube, 3 cubes (red target) | June env v1→v5: 5 s → 10 s episodes, gripper range changes, elbow gate removed (v3) and restored (v5); cameras front 64 px + wrist 16 px, then 128 px front + wrist on the gripper bracket (06-10) | June C51 teacher, DAgger students (`src/rebot_teacher_student/`) | ghrepos A/B commits |
| S2 | 06-13 → 06-29 | OURS (same USD) | same | delivery-tree env: reward rework (`rewards.py` 983 lines), office-background visual DR (`visual_obs.py`), wrist camera calibrated against a real image on 06-29 (`WRIST_OFF -0.09,0,-0.048`, 22° pitch) | DINO / mask students, perception teacher (not in this repository) | `~/rebot_eval/packages/` snapshot dates; B's uncommitted 06-29 diff |
| S3 | 07-20 → 07-23 | **OFFICIAL**: Seeed `reBot-Isaacsim` USD (`RS-rebot-dev-arm`), re-imported through PhysX (Newton tests the same day) | LiftCube | `robot_profile.py` OFFICIAL profile, camera calibration UI (`media/project2/sim_official_model_views_2026-07-23.png`) | exploration and calibration only | `curric_sweep_code/{boot,calib,calib_ui}_official.py`, `phase*_newton*`, `physx_*` logs (07-23); no bench or training entrypoint sets `ROBOT_OFFICIAL=1` |
| **S4 (latest used)** | 07-23 → 08-12 | **OURS** (same USD: MD5 `39c27cdc…` on both the training and the bench machine) | LiftCube, **single cube** (`--single-cube`), spawn box x 0.00–0.80 m, forward 0.10–0.75 m | July env snapshot (`src/rebot_grasp_env/`, `env_config.py` mtime 08-07), wrist camera of S2 (`WRIST_OFF`, `WRIST_PITCH_DEG=22`), front focal 26.4, YOLO-OBB detector on both cameras, start pose `START_ELBOW=-2.2227`, `START_WRIST=2.3` | Project 2 training (runs 8951 / 8953, `det_TP304_*`) and the 08-02/03 benches | `configs/project2_*.env` (no `ROBOT_OFFICIAL`), `results/project2/CHANGES_8953_20260731.md` ("Robot profile = OURS … NOT OFFICIAL"), asset MD5 check (2026-11-02) |
| S4' | 08-05 → 08-12 | OURS | same | same code; three- and four-phase checkpoints (align, grasp, lift) | `~/rebot_eval/ckpt_{deploy,stepeval,stepeval_app}`, run `det_TP304_ab1` (08-12) | no entrypoint file found for these evaluations, so their exact knobs are not recorded |

**Latest simulation version used = S4.** This is the OURS robot USD (unchanged since 06-08) in the LeIsaac LiftCube
scene with a single cube. It runs the July environment code with the 06-29 calibrated wrist camera and the
detector-based observations. The OFFICIAL vendor model (S3) was set up and calibrated but was not used for any
logged training or bench run. The robot asset `config.yaml` on the bench machine was edited on 07-14, but the USD
files keep their 06-08 hash, so the robot geometry was not re-converted.

## Media per version

| Version | Media in this repository | Rollout video |
|---|---|---|
| S0 | `media/readme/sim_scene_2026-06-08.jpg` | — |
| S1 | `media/readme/hero_teacher_hd_rollout.gif`, `teacher_hd_rollout_env2.gif` (June teacher, 06-10), `media/success_videos_v1/`, `media/cam_spec/` | yes (June teacher) |
| S3 | `media/project2/sim_official_model_views_2026-07-23.png` | — |
| S4 | `media/project2/hero_s4_rollouts_e3712k_2026-11-02.gif`, `media/project2/rollouts_s4/*.mp4`, `project2_log_curves.png`, `bench_heatmap_e3712k_2026-08-03.png` | yes: rendered 2026-11-02 from the included checkpoints (none existed from July–August; the trainer kept its live clip in memory only) |
