# Third-party code, assets and licences

Status: 2026-11-02; paths updated to the code layout of 2026-11-05 ([refactor log](refactor_2026-11.md)). Licence of this repository: **PolyForm Noncommercial 1.0.0** ([`LICENSE`](../LICENSE)); credits in
[`NOTICE`](../NOTICE).

How "derived" was decided: every Python file was compared line by line with the upstream repositories named below
(non-comment lines longer than 25 characters, whitespace-stripped, exact match), LeIsaac at commit `3dbab4a`
(2025-12-18) and Squint at commit `15e16f2` (2026-10-17). Upstream licences were read from the upstream `LICENSE`
files on 2026-11-01. The exact LeIsaac revision the files were originally copied from is not recorded; the commits
given are the latest upstream commits touching those files.

## 1. Original code (written for this project)

| Path | Notes |
|---|---|
| `src/rebot_teacher_student/env/` | reBot LiftCube configuration on top of the LeIsaac `LeIsaac-SO101-LiftCube-v0` task (imported, not copied): observation / action / reward / success terms, delta-on-target action with hard joint limits, cube reset, camera mounts and visual domain randomisation. Reward and success follow Squint's ManiSkill lift task; 12 of 429 non-trivial lines of `rewards.py` match LeIsaac (all generic: `SceneEntityCfg(...)` defaults and imports) |
| `src/rebot_teacher_student/teacher/` | distributional-C51 SAC teacher (single, staged, jointly trained). Follows Squint's algorithm and hyper-parameters (its docstring calls it a port of `train_squint.py` without the CNN); 7 of 287 non-trivial lines of `train_c51.py` match Squint, all generic PyTorch lines |
| `src/rebot_teacher_student/student/` | DAgger and visual RL students. `visual_nets.py` uses Squint's encoder / projection layout; 10 of 155 lines match (class names and generic calls) |
| `src/rebot_teacher_student/dashboard.py`, `src/rebot_teacher_student/deploy/`, `src/rebot_teacher_student/evaluate.py`, `archive/june_sim_tools/` | dashboard, Isaac-free deploy loops (`visual_deploy.py`: 7 generic lines match Squint), seeded evaluation, renders and diagnostics |
| `scripts/`, `analysis/`, `tests/`, `configs/` | `configs/reward_config.yaml` is shared with the author's earlier SO-101 project ([provenance §3](provenance.md#3-shared-with-the-authors-earlier-so-101-project)) |
| `archive/residual_pick_2026-06/` | the author's files of the later residual-RL pick project ([provenance §6](provenance.md#6-source-c--rebot_sim2real-residual-rl-pick-and-hybrid-deploy-selected-by-git-blame)); `pick_env.py` and `train_cube_detector.py` share 2–4 generic lines with Squint |

Because the teacher and student code is written after Squint's design, the Squint MIT notice is included in the
repository ([`LICENSES/MIT-Squint.txt`](../LICENSES/MIT-Squint.txt)) and credited in [`NOTICE`](../NOTICE).

### July–August 2026 additions

| Path | Notes |
|---|---|
| `src/rebot_grasp_rl/` | Project 2 trainer and helpers, written by the author. Its C51-SAC machinery follows the June teacher (itself after Squint). It uses Isaac Lab's `DifferentialIKController` and the LeIsaac task as libraries. `src/rebot_grasp_env/` is the author's July version of the June environment |
| `archive/real_robot_july/` | the author's real-robot deploy files; uses OpenCV, PyTorch and Hugging Face `transformers` (OWLv2 `google/owlv2-base-patch16-ensemble`, Apache-2.0 weights) as libraries; the Damiao motor driver is not included |
| `results/project2/`, `archive/project2_log_tools/` | the author's logs and tools |
| `src/rebot_grasp_common/`, `analysis/refactor_equivalence.py`, `report/` | added in November 2026 by the author (paths module, refactor check, technical report) |

## 2. Derived code (modified copies of upstream files)

None in this version. Files derived from LeIsaac and Squint (carried over from the author's earlier SO-101 project)
were removed from the public version.

## 3. Installed, not in the tree

| Dependency | How it is used | Licence |
|---|---|---|
| Isaac Sim 5.1 | simulator (pip / Omniverse install in the Isaac venv) | NVIDIA Isaac Sim licence / EULA |
| Isaac Lab 2.3 | environment framework (`ManagerBasedRLEnv`, managers, sensors) | BSD-3-Clause (core) |
| LeIsaac (reBot fork in the Isaac workspace) | task registration `LeIsaac-SO101-LiftCube-v0`, the reBot robot config, `MirroredTargetDeltaGripperAction`, scene assets | Apache-2.0 (code) |
| PyTorch, NumPy, PyYAML | | BSD-style |
| Ultralytics YOLO (`--detector` in `src/rebot_grasp_rl/`) | YOLO-OBB cube detector inference during training / evaluation | **AGPL-3.0** (installed library; not copied; detector weights not included) |
| Hugging Face `transformers` + OWLv2 weights | open-vocabulary detection in the July deploy record | Apache-2.0 |
| Seeed `reBot-Isaacsim` USD (OFFICIAL robot model) | referenced by path (`OFFICIAL_USD`), not included | MIT (as of 2026-09) |
| Stable-Baselines3 (`isaaclab_rl.sb3` wrapper), rsl_rl, LeRobot | used by `train_c51.py` (vector-env wrapper) | MIT / BSD-3-Clause / Apache-2.0 |

## 4. Models, assets and data

| Item | In the tree? | Source and licence |
|---|---|---|
| `results/checkpoints/teacher/*.pt`, `results/checkpoints/student/*.pt` | yes (15 MB) | trained by the author with this code |
| reBot B601-DM URDF, meshes, USD | no | vendor material (Seeed Studio). Seeed's `reBot-Isaacsim` repository is MIT-licensed as of 2026-09; the archive used in this project has no licence file and its origin was not recorded, so it is not redistributed |
| LeIsaac scene / cube / table USDs and textures | no | obtained with LeIsaac; terms not stated separately from its Apache-2.0 code licence |
| Background photos for visual DR (`--bg-dir`, COCO val2017 and office photos) | no | COCO images are under their individual Flickr licences; office photos are the author's |
| `media/cam_spec/*.png` | yes | renders of the simulated cameras made with this code (they depict the vendor robot model) |
| `media/success_videos_v1/*.mp4` | yes | renders of teacher rollouts made with this code |
| `media/readme/*.gif`, `sim_scene_2026-06-08.jpg`, simulated panels of `student_inputs_sim_vs_real.png` | yes | renders made with this code / the author's set-up scripts (Isaac Sim, vendor robot model) |
| `media/readme/real_front_camera_2026-06-10.jpg`, real panels of `student_inputs_sim_vs_real.png` | yes | frames from the author's real-robot cameras |

## Obligations and how they are met

| Upstream | Applies to | Obligation | Met by |
|---|---|---|---|
| LeIsaac, Apache-2.0 | installed library only | none for redistribution (nothing copied) | — |
| Squint, MIT | design followed in `src/` (no copied files) | credit | `NOTICE`, `LICENSES/MIT-Squint.txt` (courtesy) |
| NVIDIA Isaac Sim | renders in `media/` | not redistributing Isaac Sim | — |
| copyleft / non-commercial upstream code | — | none in the tree | — |
