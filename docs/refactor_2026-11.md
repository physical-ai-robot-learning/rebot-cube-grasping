# Refactor 2026-11: OA layout and code style

Work of 2026-11-05 on branch `refactor/oa-layout`, starting from commit `47398fd` (the public version of 2026-11-02).
Goal: the org-wide repository layout (`src/` packages, `scripts/` launchers, `results/`, `archive/`) and the "OA" code
style (`tLocal` locals, `mField` members, camelCase functions and parameters, `main(argv) -> int`, no `sys.path`
edits), **without changing behaviour, numbers or claims**. README numbers are unchanged (`analysis/verify_claims.py`:
28 percentages and 39 other numbers, same as before).

## Old → new paths

| Old | New |
|---|---|
| `src/det_xattn/` (trainer + helpers) | `src/rebot_grasp_rl/` (`train_sac_det_xattn` → `train`, `det_xattn_nets` → `nets`, `tp_config` → `config`, `tp_net` → `target_projection`, `ui_render` → `dashboard_render`; `runtime`, `robot_profile`, `eval_hooks` keep their names) |
| `src/det_xattn/packages/sim/env_setup/` | `src/rebot_grasp_env/` (`maniskill_rewards` → `rewards`) |
| `src/rebot_sim2real/{env,teacher,student,deploy}/` | `src/rebot_teacher_student/{env,teacher,student,deploy}/` (`maniskill_rewards` → `rewards`, `rebot_deploy` → `state_deploy`, `rebot_deploy_visual` → `visual_deploy`) |
| `src/rebot_sim2real/tools/rebot_dashboard.py` | `src/rebot_teacher_student/dashboard.py` |
| `src/rebot_sim2real/tools/{diag_*,render_*,camera_spec_render}.py` | `archive/june_sim_tools/` (verbatim) |
| `scripts/eval/eval_policies.py` | `src/rebot_teacher_student/evaluate.py` (`python -m rebot_teacher_student.evaluate`) |
| `scripts/project2_tools/` | `archive/project2_log_tools/` (verbatim) |
| `checkpoints/` | `results/checkpoints/` (same files, no new binaries) |
| — | `src/rebot_grasp_common/paths.py` (machine-specific paths), `analysis/refactor_equivalence.py`, package / folder READMEs, `tests/test_characterization.py` + `tests/golden/` |

Run commands: the trainer is now `python -m rebot_grasp_rl.train` with `PYTHONPATH=src` (was
`python src/det_xattn/train_sac_det_xattn.py` with `PYTHONPATH=src/det_xattn` and a `sys.path` insert); all other
modules are `python -m rebot_teacher_student.<...>` as before with the new package name. Every file-level entry is
listed in `analysis/refactor_2026-11_map.json` (`files`: restyled modules with their rename maps, `moved_verbatim`,
`new_files`).

### Per-module map (restyled modules)

| Old | New | Renames | Check |
|---|---|---|---|
| `src/det_xattn/packages/sim/env_setup/__init__.py` | `src/rebot_grasp_env/__init__.py` | 2 | move + rename, AST-checked |
| `src/det_xattn/packages/sim/env_setup/env_config.py` | `src/rebot_grasp_env/env_config.py` | 77 | move + rename, AST-checked |
| `src/det_xattn/packages/sim/env_setup/maniskill_rewards.py` | `src/rebot_grasp_env/rewards.py` | 210 | move + rename, AST-checked |
| `src/det_xattn/packages/sim/env_setup/target_delta_action.py` | `src/rebot_grasp_env/target_delta_action.py` | 7 | move + rename, AST-checked |
| `src/det_xattn/packages/sim/env_setup/visual_obs.py` | `src/rebot_grasp_env/visual_obs.py` | 93 | move + rename, AST-checked |
| `src/det_xattn/tp_config.py` | `src/rebot_grasp_rl/config.py` | 0 | move + rename, AST-checked |
| `src/det_xattn/ui_render.py` | `src/rebot_grasp_rl/dashboard_render.py` | 0 | by hand + goldens |
| `src/det_xattn/eval_hooks.py` | `src/rebot_grasp_rl/eval_hooks.py` | 33 | move + rename, AST-checked |
| `src/det_xattn/det_xattn_nets.py` | `src/rebot_grasp_rl/nets.py` | 31 | move + rename, AST-checked |
| `src/det_xattn/robot_profile.py` | `src/rebot_grasp_rl/robot_profile.py` | 1 | move + rename, AST-checked |
| `src/det_xattn/runtime.py` | `src/rebot_grasp_rl/runtime.py` | 0 | move + rename, AST-checked |
| `src/det_xattn/tp_net.py` | `src/rebot_grasp_rl/target_projection.py` | 16 | move + rename, AST-checked |
| `src/det_xattn/train_sac_det_xattn.py` | `src/rebot_grasp_rl/train.py` | 464 | move + rename, AST-checked |
| `src/det_xattn/packages/__init__.py` | `src/rebot_teacher_student/__init__.py` | 0 | move + rename, AST-checked |
| `src/rebot_sim2real/tools/rebot_dashboard.py` | `src/rebot_teacher_student/dashboard.py` | 26 | move + rename, AST-checked |
| `src/det_xattn/packages/sim/__init__.py` | `src/rebot_teacher_student/deploy/__init__.py` | 0 | move + rename, AST-checked |
| `src/rebot_sim2real/deploy/dump_sim_spec.py` | `src/rebot_teacher_student/deploy/dump_sim_spec.py` | 2 | move + rename, AST-checked |
| `src/rebot_sim2real/deploy/rebot_deploy.py` | `src/rebot_teacher_student/deploy/state_deploy.py` | 57 | by hand + goldens |
| `src/rebot_sim2real/deploy/rebot_deploy_visual.py` | `src/rebot_teacher_student/deploy/visual_deploy.py` | 42 | by hand + goldens |
| `src/rebot_sim2real/env/__init__.py` | `src/rebot_teacher_student/env/__init__.py` | 2 | move + rename, AST-checked |
| `src/rebot_sim2real/env/env_config.py` | `src/rebot_teacher_student/env/env_config.py` | 68 | move + rename, AST-checked |
| `src/rebot_sim2real/env/maniskill_rewards.py` | `src/rebot_teacher_student/env/rewards.py` | 156 | move + rename, AST-checked |
| `src/rebot_sim2real/env/target_delta_action.py` | `src/rebot_teacher_student/env/target_delta_action.py` | 7 | move + rename, AST-checked |
| `src/rebot_sim2real/env/visual_obs.py` | `src/rebot_teacher_student/env/visual_obs.py` | 84 | move + rename, AST-checked |
| `scripts/eval/eval_policies.py` | `src/rebot_teacher_student/evaluate.py` | 16 | move + rename, AST-checked |
| `src/rebot_sim2real/__init__.py` | `src/rebot_teacher_student/student/__init__.py` | 0 | move + rename, AST-checked |
| `src/rebot_sim2real/student/train_c51_visual.py` | `src/rebot_teacher_student/student/train_c51_visual.py` | 66 | move + rename, AST-checked |
| `src/rebot_sim2real/student/train_dagger_staged.py` | `src/rebot_teacher_student/student/train_dagger_staged.py` | 49 | move + rename, AST-checked |
| `src/rebot_sim2real/student/train_dagger_visual.py` | `src/rebot_teacher_student/student/train_dagger_visual.py` | 44 | move + rename, AST-checked |
| `src/rebot_sim2real/student/train_visual_rl.py` | `src/rebot_teacher_student/student/train_visual_rl.py` | 74 | move + rename, AST-checked |
| `src/rebot_sim2real/student/train_visual_rl_staged.py` | `src/rebot_teacher_student/student/train_visual_rl_staged.py` | 84 | move + rename, AST-checked |
| `src/rebot_sim2real/student/visual_nets.py` | `src/rebot_teacher_student/student/visual_nets.py` | 45 | move + rename, AST-checked |
| `src/rebot_sim2real/deploy/__init__.py` | `src/rebot_teacher_student/teacher/__init__.py` | 0 | move + rename, AST-checked |
| `src/rebot_sim2real/teacher/infer_staged.py` | `src/rebot_teacher_student/teacher/infer_staged.py` | 6 | move + rename, AST-checked |
| `src/rebot_sim2real/teacher/train_c51.py` | `src/rebot_teacher_student/teacher/train_c51.py` | 57 | move + rename, AST-checked |
| `src/rebot_sim2real/teacher/train_c51_joint.py` | `src/rebot_teacher_student/teacher/train_c51_joint.py` | 58 | move + rename, AST-checked |
| `src/rebot_sim2real/teacher/train_c51_staged.py` | `src/rebot_teacher_student/teacher/train_c51_staged.py` | 56 | move + rename, AST-checked |

## Technical report (added after the refactor)

`report/report_en.{tex,pdf}` (5 pages) was written on the same branch; `analysis/verify_report.py` checks every number
in it against its source file and runs in CI.

## What was archived (verbatim, nothing deleted)

* `archive/june_sim_tools/` — 20 June one-off Isaac scripts (diagnostics of cameras, gripper, contacts, DR; rollout
  and camera renders that made `media/`). Their provenance fingerprints still match.
* `archive/project2_log_tools/` — `bench_daemon.py`, `ntfy_sweep_poller.py` (cluster tools that wrote the two logs).
* `archive/real_robot_july/`, `archive/residual_pick_2026-06/` — unchanged.

## Style rules applied

* **Mechanical, all of `src/`** (tool-generated, position-exact, comments kept): function-scope locals, comprehension
  and `except` names → `tPascal`; functions, nested functions and parameters → camelCase; multi-word methods of our
  own classes (`get_action`, `act_deterministic`, `q_values`, `mean_action`, the robot-interface methods, …) →
  camelCase at their definitions and call sites; keyword arguments at call sites follow the parameters. 1,933 renames.
* **By hand** (small Isaac-free modules): `deploy/state_deploy.py`, `deploy/visual_deploy.py` (`mField` members,
  `GRIP_REST` constant, `main(argv) -> int`, `DeployError` message with exit code 1 instead of `SystemExit`),
  `rebot_grasp_rl/dashboard_render.py` (one statement per line, named constants, public names `pngBytes`, `sparkSvg`,
  `trimmedMean`, `statCard`, `drawDetection`, `quatRpy`, `drawArrow`), `target_projection.py` (one statement per line).
* **Module docstrings** everywhere: stage, original path, run command, whether Isaac is needed.
* **Machine paths**: the repeated `REBOT_ISAAC_WS` / `~/workspace/rebot_isaac_rl` and `~/yolo_cube` defaults come from
  `rebot_grasp_common.paths` (same environment variables).

### Kept as they are (imposed from outside, or not checkable)

* nn.Module attribute names and `__init__` arguments (they are the state-dict keys of the shipped checkpoints);
  checkpoint dict keys; argparse destinations and flags; environment variables; CSV / JSON keys; Isaac Lab
  config attributes (e.g. the `is_grasped` observation term); framework hooks (`forward`, `reset`, `step`,
  `apply_actions`, `do_GET`, `log_message`).
* **All function signatures in `rebot_grasp_env/` and `rebot_teacher_student/env/`**, and parameters of every function
  passed as `func=` to an Isaac Lab term: the managers pass `params={...}` by name.
* **Module-level names of the script-style trainers** (`rebot_grasp_rl/train.py` and the June `train_*.py`,
  `evaluate.py`, `infer_staged.py`): they launch the Isaac app at import (Isaac Lab must be imported after
  `AppLauncher`), parse arguments at import, and share hundreds of module globals between phases. Turning them into
  `main(argv)` functions would be a restructuring that cannot be run here, so they stay scripts run as
  `python -m ...`; `rebot_grasp_rl/config.py` still parses at import for the same reason.
* Instance attributes of classes that only run inside Isaac (`Buffer` classes, `_TPBuf`, the `RT` runtime namespace,
  `RobotProfile` fields): renaming attribute accesses through untyped receivers cannot be checked without running.
* One debug-only path in a diagnostic branch of `train.py` (`~/rebot_eval/wristprobe_*.png`, written only with
  `WRISTCAM_PROBE=1`) is still inline: replacing that f-string would not be a pure rename.

### Name clashes

The rename map is one map per file, so two different old names may not get the same new name anywhere in a file.
Where they would (`r` and `_r` both → `tR`), the second one got a numeric suffix: 73 cases, 50 of them in
`rebot_grasp_rl/train.py` (e.g. `_r` → `tR3`, `_qw` → `tQw2`), the others single letters in the env and June
trainer modules (`tB2`, `tQ2`, `tR2`, `tN2`, …). They are listed in the per-file rename maps.

## Verification

| Check | Result |
|---|---|
| Characterization goldens, recorded on `47398fd` before any change (`tests/test_characterization.py`, `tests/golden/`) | unchanged expected values pass: dashboard helpers, robot profiles, runtime defaults, Project 2 networks on both shipped checkpoints + C51 target projection, June visual nets, both mock deploy loops (12 steps, every step printed) on the three June checkpoints, deploy helpers |
| `python analysis/refactor_equivalence.py` | 34 of 37 restyled modules are their `47398fd` code up to the recorded renames (syntax tree with docstrings dropped, imports compared by binding; per-scope symbol tables compared to catch a renamed name that would bind differently). The 3 hand-restyled modules are covered by the goldens. Fault injection (one wrongly renamed local, one changed constant) is detected |
| Provenance fingerprints (`analysis/code_fingerprints*.json`) | archived files: unchanged at their new paths; restyled files: checked on their `47398fd` version (needs git history) |
| Keyword-argument lint (calls whose keywords match no parameter of a same-named function) | same 101 hits before and after, all calls of external APIs (`dict.update`, `torch.load`) |
| `python -m pytest -q` | see the CI table below |
| `python analysis/parse_logs.py --check`, `python analysis/verify_claims.py` | pass; README numbers unchanged (28 percentages, 39 other numbers) |
| `python -m compileall -q src scripts analysis tests archive` | pass |
| `DRY_RUN=1 bash scripts/eval/project2_eval.sh A B C`, `bash scripts/deploy/mock_deploy.sh` | pass |
| Fresh clone of the branch, `pip install -e ".[test]"` + CPU torch in a new Python 3.11 venv, all CI steps | pass (see the final commit message for the counts) |
| Markdown relative links and anchors | no broken link |

CI now checks out the full history (`fetch-depth: 0`) and runs the equivalence check as its own step.

Follow-up (2026-11-05, after the first push): CI runs on Python 3.11, 3.12 and 3.13. Two fixes:
* the equivalence check merges symbol-table entries that map to the same original name, because Python >= 3.12
  inlines comprehensions (PEP 709) and lists their iteration variables in the enclosing scope as well; a self-test
  (`tests/test_code.py::test_refactor_check_detects_a_wrong_binding`) checks that a name re-bound to a global is
  still rejected on every version;
* the characterization goldens compare floats with a tolerance of 2e-5 + 1e-4 relative (Linux/x86_64 CPU kernels
  round differently from macOS/arm64); `test_tolerance_detects_perturbed_weight` checks that a 1 % change of one
  actor weight still fails (it changes 72 compared values; 0.1 % changes 39).

## Not executable here

Everything that needs Isaac Sim 5.1 / Isaac Lab 2.3 and the LeIsaac reBot fork: `rebot_grasp_rl/train.py` (training,
benches, re-evaluation, renders), all of `rebot_grasp_env/`, `rebot_teacher_student/{env,teacher,student/train_*,
evaluate.py,deploy/dump_sim_spec.py,dashboard.py}` (the dashboard is started by the trainers). For these the rewrite
is mechanical and checked by `analysis/refactor_equivalence.py` and `compileall`, not by running them. Not checked by
anything: string-based lookups of renamed names (none found by a scan of string literals), and behaviour that depends
on Isaac Lab reading Python names (term parameters were left unchanged for that reason).

## Suspected issues found (recorded, not fixed)

* `deploy/visual_deploy.py` caps the gripper at 0.05 m while `deploy/state_deploy.py` caps it at 0.045 m; the deploy
  spec documents both values, so this is a difference between the two June policies rather than a typo, but the two
  loops are otherwise described as identical.
* `rebot_grasp_rl/runtime.py` documents `RT` as the shared state "for get_obs (obs.py) / actions.py", modules that do
  not exist; `train.py` also still keeps its own list-wrapped globals (`_PH_CUR`, `_ACQ`, …) next to `RT`.
* `rebot_grasp_rl/config.py` defaults `--save-dir` to `/tmp/det_xattn` (kept: it is the recorded default).
* The June trainers resolve `--config configs/reward_config.yaml` relative to the working directory; they must be run
  from the repository root (as the scripts do).
