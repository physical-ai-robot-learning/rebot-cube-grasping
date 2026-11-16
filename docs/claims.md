# Claims audit

Status: 2026-11-02. These are claims one might make about this project, checked against the files in this
repository. The three parts are kept separate on purpose: simulation RL, RL on hardware, and rule/IK on hardware.

Verdicts:
- **SUPPORTED** = backed by code, checkpoints or a logged file here.
- **PARTLY** = backed in part, or only by notes and commit messages written at the time.
- **NOT SUPPORTED** = no evidence here or in anything searched ([provenance](provenance.md)).

Every rate names its stage, success criterion, date and source. `analysis/parse_logs.py` rebuilds each one from the
raw logs.

## NOT SUPPORTED

| Claim | Evidence | Verdict |
|---|---|---|
| A learned (RL) policy picks up the cube on the real robot | no record of any RL policy working on hardware. The June camera student and the July residual SAC both failed on the real arm (Part 2). No Project 2 policy was run on the robot | **NOT SUPPORTED** |
| Successful sim-to-real transfer | as above. The hardware successes come from a perception + IK + state-machine controller that does not use the RL policies | **NOT SUPPORTED** |
| Project 2 succeeds under the deployable criterion in simulation | the deployable gate is **0.0 % in all 120 benches** (2026-08-02/03, `bench_daemon.log`) and **0 of 646 episodes** in the seeded re-evaluation (2026-11-02, runs A/B/C) | **NOT SUPPORTED** |
| End-to-end (grasp + lift) success rate of Project 2 in simulation | the benches end at the reach (`GRASP_ON=0`). Re-evaluation run C (2026-11-02, 86 episodes, seed 0) measured lift 0.0 % (Wilson 0.0–4.3 %), but the grasp rule is only triggered through the deployable gate, which no episode met, so the grasp/lift rule was never exercised | **NOT SUPPORTED** |
| June teacher success rates 93–99.8 % | commit messages only; no log or evaluation file | **NOT SUPPORTED as results** ([earlier-stage audit](#earlier-stage-june)) |
| 512 / 2,048 parallel environments | Project 2 trained with 256 environments (entrypoint of 2026-08-02; the run notes say 512 runs out of memory on 24 GB). 2,048 appears in no file of this version | **NOT SUPPORTED** |

## PARTLY

| Claim | Evidence | Verdict |
|---|---|---|
| The perception + IK + state-machine controller grasps and lifts the cube on the real robot | author's notes: full pick confirmed 2026-07-07 (`docs/14`); grasp + lift after the FK height fix 2026-07-13/14 (`docs/22`); AUTO-GRASP cycle 2026-07-23 (`docs/33`). No trial counts, no video, repeatability sweep listed as open. The code is in `archive/real_robot_july/` without the excluded IK engine | **PARTLY**: "recorded as working in the author's notes, not counted" |
| The privileged reach criterion is high in simulation | 84.0–95.2 % over 120 benches (true jaw-to-cube distance < 0.07 m, held 1 frame; 2026-08-02/03); re-evaluation 2026-11-02: 88.1 % (82.2–92.3 %) on the 86-cell grid, 73.0 % (68.5–77.1 %) on 400 cases. This criterion needs the simulator's true cube position | **SUPPORTED as a simulation measurement**; not a deployable result |
| Run 8953 reaches up to 81 % in its sweeps | `ntfy_poller.log`: 71–81 % over 36 sweeps of segment 3, maximum 81 % at 1,101k steps. The criterion is the privileged distance gate; its threshold for that segment is not recorded (0.07–0.14 m across the run files of those days). The log has no dates | **PARTLY**: name the criterion and the missing threshold |
| Real-robot attempts exposed observation mismatch (June) | deploy guide in this repository + commit messages. Start pose ≠ REST and the wrist-camera finding come from notes outside this repository | **PARTLY** |

## SUPPORTED

| Claim | Evidence | Verdict |
|---|---|---|
| Staged C51-SAC on detection-level observations (no raw pixels, no privileged state) in Isaac Lab, with an IK action space for the approach | `src/rebot_grasp_rl/`, `configs/project2_*.env` (`NO_PRIV=1`, `APP_NO_YAW=1`) | **SUPPORTED** |
| The search stage works in simulation | search criterion 92.1–99.4 % over 120 benches (2026-08-02/03); 95–100 % over 61 sweeps of run 8951; re-evaluation 2026-11-02 (seed 0): 99.4 % (Wilson 96.5–99.9 %, 160 episodes, 86-cell grid) and 98.5 % (96.8–99.3 %, 400 cases) | **SUPPORTED** (simulation; logs + per-episode re-evaluation) |
| Checkpoints of a logged bench are available | `results/checkpoints/project2/` = the pair of bench e3712k (search 97.8 %, privileged reach 89.9 %, deployable gate 0.0 %, 86 cells, 2026-08-03); MD5s in provenance §9 | **SUPPORTED** |
| Two robot models (converted URDF vs vendor USD) in simulation | `robot_profile.py`, run notes | **SUPPORTED** (code) |
| Rendered rollouts of the included checkpoints in the latest simulation version | 4 seeded episodes (2026-11-02): search → approach → privileged reach in 4/4, deployable gate 0/4, no grasp or lift; videos in `media/project2/rollouts_s4/` | **SUPPORTED** (4 fixed cases only, not a rate) |
| Curriculum over a 10×10 spawn-cell grid with deterministic self-evaluation sweeps | trainer code, configs, `ntfy_poller.log` | **SUPPORTED** |

## Re-evaluation

Run 2026-11-02 on the second workstation (one Ray job, seed 0, the included checkpoint pair, the bench
configuration). Per-episode CSVs and summaries are in `results/project2/reeval/`.

| Run | Episodes | search | privileged reach (< 0.07 m) | deployable gate | lift |
|---|---|---|---|---|---|
| A: 86-cell bench replica | 160 (86 cases, revisited) | 99.4 % (96.5–99.9) | 88.1 % (82.2–92.3) | 0.0 % (0.0–2.3) | not run |
| B: 400-case grid | 400 | 98.5 % (96.8–99.3) | 73.0 % (68.5–77.1) | 0.0 % (0.0–0.9) | not run |
| C: end-to-end, experimental | 86 | 97.7 % (91.9–99.4) | not measured | 0.0 % (0.0–4.3) | 0.0 % (0.0–4.3) |

C could not exercise the grasp rule, because the rule starts only after the deployable gate, which is never met.

## CV-safe sentences

* "Trained a staged distributional C51-SAC in Isaac Lab for a low-cost 6-DoF arm (reBot B601-DM). The policy observes
  only camera detections, cube yaw and proprioception, and approaches with differential IK. In simulation the search
  stage succeeded in 92–99 % of eval cells (logged benches, Aug 2026). The approach reached the cube by a privileged
  distance criterion in 84–95 % of cells, but never passed the deployable success gate."
* "On the real robot, transferring learned policies failed (out-of-distribution observations). A perception + IK +
  state-machine controller grasped and lifted the cube in the author's recorded hardware sessions (July 2026,
  uncounted)."
* Avoid: "sim-to-real transfer", "RL policy on the real robot", any success rate without its stage and criterion,
  "validated".

## Earlier stage (June)

The June teacher/student audit is unchanged; see [stage_2026-06_teacher_student.md](stage_2026-06_teacher_student.md).
In short:
- Commit-recorded teacher rates 52 %, 99 %, 99.8 %, 94.8 % and 96.4 %, and student rates 34–90 %, have no log behind
  them.
- The C51-SAC teacher, the DAgger students and the domain-randomisation code are supported by code and checkpoints.
- The June real-robot attempt failed (camera input out of distribution).
