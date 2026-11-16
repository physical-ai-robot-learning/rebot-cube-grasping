# media/

Everything here was produced by the author's runs of this project in June 2026. Simulation renders depict the vendor's
robot model (Seeed Studio) in Isaac Sim; real frames come from the author's real-robot diagnosis. None of it is
evidence of a success rate; captions state only what the file shows.

## Project 2 (`project2/`, July–August 2026)

| File | What it shows | Source |
|---|---|---|
| `project2/project2_log_curves.png` | left: 120 bench results (search / privileged reach / deployable gate) over approach checkpoints, 2026-08-02/03; right: run 8953 sweeps (segment 3) | made by `analysis/plot_project2.py` from `results/project2/` |
| `project2/bench_heatmap_e3712k_2026-08-03.png` | per-cell heatmaps of the bench of the included checkpoints | `results/project2/heatmaps/bench_e3712k.png` |
| `project2/sim_official_model_views_2026-07-23.png` | simulation, OFFICIAL robot model: front, side and wrist calibration views (yellow sphere = camera aim marker) | training workstation `~/ui_*.png`, `wnow.png`, `front.png`, 2026-07-23 |

| `project2/hero_s4_rollouts_e3712k_2026-11-02.gif` (3.6 MB, 896 px) | simulation S4: the 4 rendered episodes below, one after another (front \| wrist detector-input views with detection overlay) | made from the MP4s with ffmpeg (8 fps, 48-colour palette) |
| `project2/rollouts_s4/ep0{0..3}_case{0..3}_reach_privileged.mp4` (42–53 kB, 512×256, 10 fps, 20–23 frames) + `videos.csv` | simulation S4, checkpoint pair of bench e3712k, seed 0, cases 0–3 of `results/project2/eval_grids/render_cases_4_from_86cells.npy`. Each episode: search → approach → privileged reach (true jaw-to-cube < 0.07 m), which ends the episode; deployable gate not met; no grasp or lift | rendered 2026-11-02 on the second workstation (RTX 5080) by `scripts/eval/project2_eval.sh R` (Ray job `rebot_s4_render_20261102_2058`); MD5s in docs/provenance.md §13 |

The June teacher videos further down show the earlier simulation version S1.

## Earlier stage (June, simulation version S1), media (`readme/`)

| File | What it shows | Made from (original host : path, mtime, MD5) | Processing |
|---|---|---|---|
| `readme/hero_teacher_hd_rollout.gif` (4.7 MB) | simulation, state-only C51 teacher, one rollout rendered in HD: left = fixed front view, right = close view behind the gripper; the red cube is grasped and lifted | second workstation : `~/Downloads/teacher_hq_00_env0.mp4`, 2026-06-10 15:06, `b5a1bd4ca7107f6ef812d82a02bdaa32` (1448×720, 60 fps, 8.3 s); written by `archive/june_sim_tools/render_teacher_hq.py` | ffmpeg: 12 fps, 900 px wide, 96-colour palette |
| `readme/teacher_hd_rollout_env2.gif` (1.6 MB) | simulation, another rollout of the same render run; cube lifted | second workstation : `~/Downloads/teacher_hq_02_env2.mp4`, 2026-06-10 15:06, `ea6247f0ad3bd7760cdd0a06266fbb17` | ffmpeg: 12 fps, 440 px wide |
| `readme/student_inputs_sim_vs_real.png` | 2×4 panel: front (top) and wrist (bottom) camera; simulated render, simulated 128×128 policy input, real camera frame, real policy input | simulated: `cam_spec/*.png` below (2026-06-10 20:19). Real, second workstation : `~/Downloads/real_diag4/front_live.jpg` (2026-06-10 18:57, `e6fbc72ded18ca36849ccceba4e5a7ff`), `real_diag4/front_ds128.png` (18:01, `0e931caefa30dcf59ec88d6cc977b039`), `real_diag/wrist_full.png` (12:46, `8003db9af1a123d1074eb4eb3b0e7d33`), `real_diag/wrist_ds16.png` (12:46, `ce5437c467e95cea185ab20d2445936b`, the 16×16 input of the first student) | matplotlib panel, no image edits |
| `readme/real_front_camera_2026-06-10.jpg` | real robot, front camera, during diagnosis of the camera-based student; the arm is next to the red target cube. Outcome of the attempt not recorded | second workstation : `~/Downloads/real_diag4/front_live.jpg`, 2026-06-10 18:57, `e6fbc72ded18ca36849ccceba4e5a7ff` | copied unchanged |
| `readme/sim_scene_2026-06-08.jpg` | the first reBot scene after importing the robot into Isaac Sim (four cubes, before the final three-cube layout) | training workstation : `~/workspace/rebot_isaac_rl/renders/scene_iso.png`, 2026-06-08 14:34, `1070d9fead8246a48f51704a110fc4f5` | resized to 900 px, JPEG |

The real frames were checked for people: none shows a person's face (the office background of the front camera
shows furniture and equipment only).

## Committed in the original repositories

| Path | What | Made by |
|---|---|---|
| `success_videos_v1/*.mp4` (6 clips, 18–26 kB each) | low-resolution combined-view renders of successful rollouts of the first state-only teacher (2026-06-09); file names give env index and training step of the render run | `archive/june_sim_tools/render_successes.py` |
| `cam_spec/front_full.png`, `front_ds128.png`, `wrist_full.png`, `wrist_ds128.png` | the simulated front and wrist cameras at the REST pose, full resolution and the 128×128 policy input (2026-06-10) | `archive/june_sim_tools/camera_spec_render.py` |

## Found but not included

| What | Where | Why |
|---|---|---|
| Two more HD teacher renders (`teacher_hq_01`, `_03`) | second workstation `~/Downloads/` | same run as the two used; not needed |
| Wrist-camera orientation tests (`real_diag4/wrist_live.jpg`, `wrist_rot90/180/270.jpg`) and the 64×64 front input (`real_diag/front_ds64.png`) | second workstation `~/Downloads/real_diag*` | kept out for size of the figure; same diagnosis as the frames used |
| Phone photos `IMG_2551`–`2556.jpeg` (2026-06-15/16) | second workstation `~/Downloads/`, also untracked in the residual-pick repository | taken on the first day of the later residual-pick project; one shows another company's logo and robots in the background |
| Screen recordings of 2026-06-17 and `top/wrist/quarter.jpeg` | second workstation | the later residual-pick project's scene, not this project |
| Rollout GIFs / MP4s `viz_out/ep*_SUCCESS|fail.*`, success grids, monitor GIFs in `_archive/runs/`, real camera views `deploy/view_*.png` | training workstation `~/workspace/rebot_sim2real/` (untracked) | the later residual-pick project (mixed authorship, see docs/provenance.md §6) |
| Real-robot frames `bg_images/real_rebot/*.jpg` (1,413 files) | training workstation `~/workspace/rebot_isaac_rl/` | frames of earlier teleoperation recordings of other tasks, used only as background images for domain randomisation; show people's feet and unrelated scenes |
| COCO / office background photos | same | third-party photos |
| MLflow run artefacts (experiments 4, 7–18) | cluster MLflow | runs of the later residual-pick project |

No video of a real-robot rollout of the teacher or the students was found on any machine searched (training
workstation, second workstation, laptop, external drive).
