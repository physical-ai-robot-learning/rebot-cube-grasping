# Provenance

Where every file in this repository came from, what was changed on the way in, what was left out and why.
Status: 2026-11-01.

## 1. Sources

This repository was assembled with a **fresh git history**. The original repositories' histories were not imported;
their commits are listed here instead. All commits of sources A and B were made by the author (company account).

**A. `rebot-sim2real-rl`** (state-only teacher, environment, state-only deploy). Clean working tree at HEAD `c1407a3`.

| Commit | Date (+09:00) | Subject |
|---|---|---|
| `c5fc4df` | 2026-06-09 18:35 | reBot B601-DM sim RL: state-only distributional-C51 SAC teacher (LiftCube) |
| `74d8995` | 2026-06-10 19:20 | reBot sim2real: fully-random cube workspace + 99% teacher (mixed distribution) |
| `6f80818` | 2026-06-10 20:19 | Real-robot v3 fixes: free elbow + gripper range/convention + 99.8% teacher |
| `ca032b5` | 2026-06-11 17:48 | v5: elbow gate restored + align→grasp curriculum + joint 3-stage teacher |
| `c1407a3` | 2026-06-12 17:47 | v6: 4-stage joint teacher (reaching→align→grasp→lift) + 4-stage infer_staged; deploy gripper range/convention fix |

**B. `rebot-dagger-visual`** (DAgger visual students, visual domain randomisation, visual deploy). HEAD `67a11b7`.

| Commit | Date (+09:00) | Subject |
|---|---|---|
| `67875ec` | 2026-06-09 18:37 | reBot visual policy: DAgger teacher->student distillation (LiftCube) |
| `e509394` | 2026-06-09 18:40 | Add trained visual student checkpoint (dagger_final.pt) |
| `0de16a8` | 2026-06-10 12:08 | Add visual domain randomization (DR) + visual deploy package |
| `5d396ee` | 2026-06-10 12:12 | deploy_package: document the visual (camera-only) deploy in README |
| `3fcdeb2` | 2026-06-10 19:20 | Visual dual-128 student: fully-random workspace, 99% teacher, recalibrated cameras |
| `b5e597d` | 2026-06-10 20:20 | Visual v3: real-robot fixes (free elbow, gripper convention, roll/clutter DR) + 99.8% teacher |
| `9d24980` | 2026-06-11 17:49 | v5: shared env update (elbow gate restored + align→grasp curriculum) + gripper-corrected Mac deploy |
| `67a11b7` | 2026-06-12 17:47 | v6: staged visual student (DAgger) + visual RL fine-tune (single+staged) + real-office-bg DR; deploy aspect/gripper fixes |

The working tree of B had one uncommitted modification and one untracked file. **The committed HEAD content was
used.** The modification (dated 2026-06-29) changes `packages/sim/env_setup/visual_obs.py`: the wrist-camera offset
`WRIST_OFFSET_POS` from `(-0.02, 0.0, -0.05)` to `(-0.09, 0.0, -0.048)` and an extra 22° downward pitch
(`WRIST_CALIB_PITCH_DEG`), labelled as a calibration against a real wrist image. The untracked file is a backup of
the same module from before that change. Neither is in this repository; no committed checkpoint was trained with the
modified camera.

The original runs were started from a working copy of these trees inside the Isaac workspace
(`~/workspace/rebot_isaac_rl/sim2real_so101/delivery/`); the absolute paths in argument defaults pointed there.

**C. `rebot_sim2real`** (residual-RL pick, deep-reach and hybrid real-robot deploy, 2026-06-15 → 06-25). Mixed
authorship; only the author's own content is included, selected by `git blame` (§6).

## 2. What was changed on the way in

Files of A and B were merged (the 26 files present in both are byte-identical copies) and moved to the layout below.

| Change | Where | Effect on behaviour |
|---|---|---|
| Package layout `packages.sim.*` → `rebot_sim2real.{env,teacher,student,tools,deploy}` (imports rewritten) | `src/` | none (same modules) |
| Absolute path defaults → `$REBOT_ISAAC_WS` (default `~/workspace/rebot_isaac_rl`) and repository-relative `rl_output/...` | `src/` | same paths on the original machine; output defaults now relative to the repository |
| Usage lines in docstrings updated to `python -m rebot_sim2real...` | `src/` | none |
| Korean comments, docs and three dashboard labels translated to English; emoji removed | `docs/deploy/`, `src/rebot_sim2real/tools/rebot_dashboard.py`, `archive/` | none (text only) |
| Session-workflow notes (reply language, polling habits, references to personal note files) removed from the archived handoff notes `docs/08`–`12`; cluster addresses replaced by placeholders | `archive/` | none (text only) |
| Code not reachable from any reBot entry point (mostly shared with the author's earlier SO-101 project) | — | removed from the public version |

**Behaviour check.** [`analysis/ast_normalize.py`](../analysis/ast_normalize.py) compares each moved Python file with
its original after normalisation (docstrings dropped, comments absent from the AST, old→new import names, the two
path-default forms mapped to one placeholder). **38 of the 39 Python files kept in this version are equal; the exception is
`src/rebot_sim2real/tools/rebot_dashboard.py`, whose only differences are three translated HTML labels.** The
fingerprints are stored in [`analysis/code_fingerprints.json`](../analysis/code_fingerprints.json) and re-checked by
`tests/test_code.py`, so later edits show up. The original-side check needs the original files and the original
workspace prefix (`ORIG_WS_PREFIX`), which are not in this repository.

### File map (sources A and B)

A = in `rebot-sim2real-rl@c1407a3`, B = in `rebot-dagger-visual@67a11b7`. MD5 of the original file. Binary
checkpoints and the vendor URDF archive are listed in §4 and §7.

| Path here | Original path | A | B | Original MD5 | Byte-identical here | Normalised AST equal |
|---|---|---|---|---|---|---|
| `configs/reward_config.yaml` | `configs/reward_config.yaml` | x | x | `c84b71ff413905fe43f7c445924ae8c0` | yes | – |
| `docs/deploy/DEPLOY_GUIDE.md` | `deploy_package/DEPLOY_GUIDE.md` |  | x | `e06423fb08355275db0a0fbf25ebdd51` | no | – |
| `docs/deploy/DEPLOY_SPEC.md` | `deploy_package/DEPLOY_SPEC.md` | x | x | `8b562815676607a5cbfa64ea5e4a45dd` | no | – |
| `docs/deploy/README.md` | `deploy_package/README.md` | x | x | `c4bb375380baeb75794631368f59ed01` | no | – |
| `media/cam_spec/front_ds128.png` | `rl_output/cam_spec/front_ds128.png` |  | x | `08caff20d2cb42f607edf87bb05596cd` | yes | – |
| `media/cam_spec/front_full.png` | `rl_output/cam_spec/front_full.png` |  | x | `ba7c32bc8df6f965392c2f8329a3f4f2` | yes | – |
| `media/cam_spec/wrist_ds128.png` | `rl_output/cam_spec/wrist_ds128.png` |  | x | `7d080cb79162415c4909a259dc496a63` | yes | – |
| `media/cam_spec/wrist_full.png` | `rl_output/cam_spec/wrist_full.png` |  | x | `5d06c54ef64d38fe8112a419402dc3a4` | yes | – |
| `media/success_videos_v1/success_00_env9_step99.mp4` | `rl_output/success_videos/success_00_env9_step99.mp4` | x |  | `fbefae34b45366d5b97ed2383e76e862` | yes | – |
| `media/success_videos_v1/success_01_env13_step99.mp4` | `rl_output/success_videos/success_01_env13_step99.mp4` | x |  | `7fffe46a482ee228b45bb06b8cbd0191` | yes | – |
| `media/success_videos_v1/success_02_env12_step149.mp4` | `rl_output/success_videos/success_02_env12_step149.mp4` | x |  | `f93c1481317b9fc1c2b07fc5fd83249d` | yes | – |
| `media/success_videos_v1/success_03_env11_step199.mp4` | `rl_output/success_videos/success_03_env11_step199.mp4` | x |  | `4898acbd521dfa5f35f671576f14b478` | yes | – |
| `media/success_videos_v1/success_04_env9_step249.mp4` | `rl_output/success_videos/success_04_env9_step249.mp4` | x |  | `d9641f1d7780561538512c05db5199af` | yes | – |
| `media/success_videos_v1/success_05_env14_step249.mp4` | `rl_output/success_videos/success_05_env14_step249.mp4` | x |  | `0198b76d207ba36b235aea9e43397ff7` | yes | – |
| `src/rebot_sim2real/deploy/dump_sim_spec.py` | `deploy_package/dump_sim_spec.py` | x | x | `b7741e4ac49394922aeda52cb6e22a01` | no | yes |
| `src/rebot_sim2real/deploy/rebot_deploy.py` | `deploy_package/rebot_deploy.py` | x | x | `738f38bcb849f4466511bf1fad390309` | no | yes |
| `src/rebot_sim2real/deploy/rebot_deploy_visual.py` | `deploy_package/rebot_deploy_visual.py` |  | x | `d7731c3a8011d86573c435d7a153e770` | no | yes |
| `src/rebot_sim2real/deploy/requirements.txt` | `deploy_package/requirements.txt` | x | x | `09b9e04fde862178d19a71f16fac9e98` | yes | – |
| `src/rebot_sim2real/env/__init__.py` | `packages/sim/env_setup/__init__.py` | x | x | `6d8786d1c40e61486c22abbfb09ef445` | yes | yes |
| `src/rebot_sim2real/env/env_config.py` | `packages/sim/env_setup/env_config.py` | x | x | `1e7cc5b9e4f2bd51b6e80edb7f6918ac` | no | yes |
| `src/rebot_sim2real/env/maniskill_rewards.py` | `packages/sim/env_setup/maniskill_rewards.py` | x | x | `e4835bda39a65d3fedb7ec9238aa5d62` | yes | yes |
| `src/rebot_sim2real/env/target_delta_action.py` | `packages/sim/env_setup/target_delta_action.py` | x | x | `6081c2a619b3c908c059d66846a77685` | yes | yes |
| `src/rebot_sim2real/env/visual_obs.py` | `packages/sim/env_setup/visual_obs.py` |  | x | `e4869609fb79222587ce5c6b940359d3` | yes | yes |
| `src/rebot_sim2real/student/train_c51_visual.py` | `packages/sim/train_c51_visual.py` |  | x | `5108c8182eeaeaa21f5c63d9d952cdf0` | no | yes |
| `src/rebot_sim2real/student/train_dagger_staged.py` | `packages/sim/train_dagger_staged.py` |  | x | `9d1eeacfed7794614bd7eb58f730fe5e` | no | yes |
| `src/rebot_sim2real/student/train_dagger_visual.py` | `packages/sim/train_dagger_visual.py` |  | x | `e3b7ec93a78c322b804dd39211b4601c` | no | yes |
| `src/rebot_sim2real/student/train_visual_rl.py` | `packages/sim/train_visual_rl.py` |  | x | `37248187837ec58be09b6119f87f071b` | no | yes |
| `src/rebot_sim2real/student/train_visual_rl_staged.py` | `packages/sim/train_visual_rl_staged.py` |  | x | `03d8ff0493c521f299aa12a6385f8f75` | no | yes |
| `src/rebot_sim2real/student/visual_nets.py` | `packages/sim/visual_nets.py` |  | x | `a6c2559df700a2a8d6ef7f89f07ab317` | yes | yes |
| `src/rebot_sim2real/teacher/infer_staged.py` | `packages/sim/infer_staged.py` | x |  | `8f5dfcb056b67296f5b7ef198e4fe33c` | no | yes |
| `src/rebot_sim2real/teacher/train_c51.py` | `packages/sim/train_c51.py` | x |  | `ae2bb89607c38b767ba488ee437551a2` | no | yes |
| `src/rebot_sim2real/teacher/train_c51_joint.py` | `packages/sim/train_c51_joint.py` | x |  | `83d2c9bc232387f4101aa07c4d47a24a` | no | yes |
| `src/rebot_sim2real/teacher/train_c51_staged.py` | `packages/sim/train_c51_staged.py` | x |  | `d86db882eec6deeb2f592c066a41d887` | no | yes |
| `src/rebot_sim2real/tools/camera_spec_render.py` | `packages/sim/camera_spec_render.py` |  | x | `58e20f1f2868293b3b49a54033f63532` | no | yes |
| `src/rebot_sim2real/tools/diag_cam_sweep2.py` | `packages/sim/diag_cam_sweep2.py` |  | x | `1699315d16caf6474af9d219a6265753` | no | yes |
| `src/rebot_sim2real/tools/diag_contactapi.py` | `diag_contactapi.py` | x |  | `ac73120fbc537568f424a3b348fbc97f` | no | yes |
| `src/rebot_sim2real/tools/diag_dr_probe.py` | `packages/sim/diag_dr_probe.py` |  | x | `7cd714e4038da768d2455c4d27d619c0` | no | yes |
| `src/rebot_sim2real/tools/diag_env_gripper.py` | `diag_env_gripper.py` | x |  | `15db4e25334c0ffdfb60f02347b2d535` | no | yes |
| `src/rebot_sim2real/tools/diag_finger_geom.py` | `packages/sim/diag_finger_geom.py` |  | x | `1cd8f1eec3a49ab9bff39ab13d84a96c` | no | yes |
| `src/rebot_sim2real/tools/diag_front_sweep.py` | `packages/sim/diag_front_sweep.py` |  | x | `416680bfd08240dd00e217c39db4a6c6` | no | yes |
| `src/rebot_sim2real/tools/diag_grasp.py` | `diag_grasp.py` | x |  | `795c3f6862c0d4690a008803ede9a778` | no | yes |
| `src/rebot_sim2real/tools/diag_grasp_wrist.py` | `packages/sim/diag_grasp_wrist.py` |  | x | `f80725cee74c9592746157227bce1e9b` | no | yes |
| `src/rebot_sim2real/tools/diag_gripper_mounts.py` | `packages/sim/diag_gripper_mounts.py` |  | x | `6190d0561ad995c210ec10a229cbfeb0` | no | yes |
| `src/rebot_sim2real/tools/diag_gripper_stroke.py` | `packages/sim/diag_gripper_stroke.py` | x | x | `322bd2aee46716323188f64e496b1621` | no | yes |
| `src/rebot_sim2real/tools/diag_gripper_view.py` | `packages/sim/diag_gripper_view.py` |  | x | `30f6b6d6dd3e85119113318946e9ae20` | no | yes |
| `src/rebot_sim2real/tools/diag_mount_marker.py` | `packages/sim/diag_mount_marker.py` |  | x | `c960aac9b2ac3a8611bf3222d0dac8cc` | no | yes |
| `src/rebot_sim2real/tools/diag_recalib.py` | `packages/sim/diag_recalib.py` |  | x | `f50642701e0026d7b69c1f4eae7e21b8` | no | yes |
| `src/rebot_sim2real/tools/diag_visual_obs.py` | `packages/sim/diag_visual_obs.py` |  | x | `8d0f523787222da1224f06bfe539baf6` | no | yes |
| `src/rebot_sim2real/tools/diag_wrist_calc.py` | `packages/sim/diag_wrist_calc.py` |  | x | `f1af36d7ed0718b6c3aa13e7ce9b6292` | no | yes |
| `src/rebot_sim2real/tools/diag_wrist_sweep.py` | `packages/sim/diag_wrist_sweep.py` |  | x | `fcc44bb559a84778b883fe8e64de4025` | no | yes |
| `src/rebot_sim2real/tools/rebot_dashboard.py` | `packages/sim/rebot_dashboard.py` | x | x | `b7ffc430f6966b82fd3299504a4d0c7f` | no | no (3 UI strings translated) |
| `src/rebot_sim2real/tools/render_successes.py` | `packages/sim/render_successes.py` | x |  | `d0dc7c58b2739243863f8dccee35d769` | no | yes |
| `src/rebot_sim2real/tools/render_teacher_2cam.py` | `packages/sim/render_teacher_2cam.py` | x | x | `d62e05f69ae49e99bd74cdd317f13d4c` | no | yes |
| `src/rebot_sim2real/tools/render_teacher_hq.py` | `packages/sim/render_teacher_hq.py` | x | x | `bd85c1ad795767cc94df25798f8cfbbe` | no | yes |

## 3. Shared with the author's earlier SO-101 project

Files shared with the author's earlier SO-101 project were removed from the public version; only
`configs/reward_config.yaml` is kept (the reBot scripts read two keys from it), together with the one-line package
`__init__.py` files `src/rebot_sim2real/env/__init__.py` and `src/det_xattn/packages/sim/env_setup/__init__.py` and
empty package markers.

## 4. Checkpoints

Included (15 MB in total; all trained by the author in source A or B):

| File here | Original | MD5 | Train step in file |
|---|---|---|---|
| `checkpoints/teacher/c51_v5g_final.pt` | A `rl_output/rl_c51_v5g/c51_final.pt` (`ca032b5`) | `e8862ad6bc4a31823d528492bd51e379` | 2,867,200 |
| `checkpoints/teacher/c51_smooth_v1_final.pt` | A `rl_output/rl_c51_smooth/c51_final.pt` (`c5fc4df`) = A/B `deploy_package/checkpoints/c51_policy.pt` | `3015e8e823024c492dc611559b6cb1b7` | 7,577,600 |
| `checkpoints/student/dagger_dual128_v3_final.pt` | B `rl_output/rl_dagger_dual128_v3/dagger_final.pt` (`b5e597d`) = B `deploy_package/checkpoints/dagger_dual128_final.pt` | `415b78ded29186854918fe5e1b12a549` | 441,312 |

Not included (size; they stay in the original repositories A and B on the training workstation):

| Original | MD5 | Step | Note |
|---|---|---|---|
| A `rl_output/rl_c51_mixed99/c51_final.pt` | `84ccf15d48870318a414c43d9e10731a` | 1,024,000 | |
| A `rl_output/rl_c51_v3/c51_final.pt` | `aeeeca207b6347a009e13733d9b74e2f` | 1,075,200 | |
| A `rl_output/rl_c51_joint/s0_final.pt`, `s1`, `s2` | `47bdba6b…`, `bd71fbc7…`, `bd34585d…` | 307,200 each | 3-stage teacher of `ca032b5`; the 4-stage checkpoint of `c1407a3` (`s3`) was never committed |
| A `rl_output/rl_c51/c51_final.pt` + 43 periodic `c51_ckpt_*.pt` | | 8,755,200 (final) | |
| A `rl_output/rl_c51_smooth/c51_ckpt_*.pt` (23 files) | | | |
| B `rl_output/rl_dagger_dual128_mixed/dagger_{final,last}.pt` | `0bf54efd…`, `b2a54a0b…` | 494,112 (final) | |
| B `rl_output/rl_dagger_dual128_v3/dagger_last.pt` | `b9cbea59…` | | |
| B `rl_output/rl_dagger_visual/dagger_final.pt` | `7aa0dd5f…` | 857,216 | 16×16 wrist / 64×64 front student; does not load with the current 128×128 code |
| B `rl_output/rl_dagger_visual_dr/dagger_final.pt` | `6675c60b…` | 927,616 | same, with DR |
| A `rl_output/smoke_cube/sac_final.zip` (9.5 MB) | | | SB3 smoke run |

## 5. Third-party notices

See [third_party.md](third_party.md) and [`NOTICE`](../NOTICE).

## 6. Source C — `rebot_sim2real` (residual-RL pick and hybrid deploy), selected by `git blame`

The author confirmed that the commits made under their workstation git identity are their own. The
repository also contains commits by a colleague (`colleague`, 16 commits on 2026-06-23; `colleague (set-up account)`, 2 commits on
2026-06-15: the initial commit and the Isaac Sim environment set-up) and a GitHub-generated initial commit. **Rule
applied: a file is included only if every line (`git blame -w`) at the chosen commit is by the author;
otherwise it is excluded.** Two commits were used:

* `b4c128e` (2026-06-20 00:12), the last commit before the colleague's refactors of 06-23. Files that are
  author-only at this commit → `archive/residual_pick_2026-06/snapshot_2026-06-20/`.
* HEAD `65778d3` (2026-06-25 19:15). Files that are author-only at HEAD and did not exist at `b4c128e` →
  `archive/residual_pick_2026-06/additions_2026-06-21_to_25/`.

The colleague's 06-23 commits touched almost every module (refactors described as behaviour-preserving, plus a change
of the simulation coordinate frame and the table/cube placement, which change behaviour). Their lines cannot be removed
from the HEAD files without reverting those changes, so the HEAD versions of mixed files are excluded. The
environment base (`constants.py`, `scene/builder.py`, `scene/cameras.py`, the URDF, `pyproject.toml`,
`setup_env.sh`) contains the colleague's set-up from 06-15 at every commit, so **the archived code does not run on its
own**; it is a record of the author's work, compiled but not executed by the tests. The working tree of C also had
uncommitted changes and untracked data/images (21 GB); none of that was used. Four USD files committed by the author
(converted from the vendor URDF; 39.6 MB for the base file) are not included.

Internal cluster addresses, host and user names in the archived files were replaced by placeholders
(`<ray-head>`, `<nfs-host>`, `<workstation>`, `<lan-ip>`, `<cluster-node>`, `<user>`).

Commits of C by the author, oldest first: `de6086a`, `ff291fc`, `3cedeba`, `ad9e5e6`,
`9d9ef4b` (06-16), `6d82b21`, `5361299` (06-17), `9c04a05`, `fa0675e`, `f32d67d`, `3a71c3a`, `b0138dd`, `dc95fb4`,
`9607237`, `a7b8409` (06-18), `22f2ba9`, `a075ab9`, `fd1e44f` (06-19), `c67a802`, `b4c128e` (06-20), `cd35b61`,
`c476e6e`, `7268c0b`, `e52eb4b`, `4ba25a6`, `3e463ae`, `a4cfb0f`, `937cbb3` (06-24), `3a8bee1`, `8a088cb`,
`ffacc17`, `65778d3` (06-25). Not by the author: `835055a` (GitHub initial commit), `135ee85`, `ef0848e` (colleague (set-up account),
06-15), `4b913de` … `ff58a03` (colleague, 16 commits, 06-23).

### Included from C

| File | Included from | Lines at that commit (all by the author) |
|---|---|---|
| `configs/_smoke.yaml` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 18 |
| `configs/train.yaml` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 122 |
| `configs/train_v5.yaml` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 105 |
| `deploy/build_obs.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 55 |
| `deploy/calib_convention.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 88 |
| `deploy/calib_extrinsic.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 96 |
| `deploy/calib_extrinsic_known.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 97 |
| `deploy/calib_gripper.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 56 |
| `deploy/cam_front_intrinsics.npz` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | binary |
| `deploy/cube_detector_io.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 62 |
| `deploy/diag_front.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 106 |
| `deploy/diag_grasp.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 47 |
| `deploy/dry_run.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 97 |
| `deploy/grip_check.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 62 |
| `deploy/joint_map.npz` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | binary |
| `deploy/robot_io.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 212 |
| `deploy/run_pick.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 214 |
| `deploy/sysid_arm.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 97 |
| `deploy/wrist_view.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 57 |
| `docs/cluster_training.md` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 101 |
| `scripts/calib_intrinsics.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 69 |
| `scripts/collect_demos.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 144 |
| `scripts/eval_rollout.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 59 |
| `scripts/eval_verify.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 76 |
| `scripts/gen_perception_data.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 167 |
| `scripts/monitor.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 236 |
| `scripts/render_demo.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 132 |
| `scripts/render_joints.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 108 |
| `scripts/render_policy.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 127 |
| `scripts/render_views.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 163 |
| `scripts/render_wristcam.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 85 |
| `scripts/submit_train.sh` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 60 |
| `scripts/train.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 94 |
| `scripts/train_cube_detector.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 162 |
| `scripts/verify_closed_loop_sim.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 114 |
| `scripts/verify_perception_noise.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 76 |
| `scripts/verify_perception_sim.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 109 |
| `scripts/wrist_env_check.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 53 |
| `scripts/wrist_ibvs_sim.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 149 |
| `scripts/wrist_render_perf.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 94 |
| `scripts/wrist_servo_sim.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 178 |
| `src/rebot_sim2real/envs/__init__.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 1 |
| `src/rebot_sim2real/envs/pick_env.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 878 |
| `src/rebot_sim2real/grasp_traj.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 66 |
| `src/rebot_sim2real/perception.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 104 |
| `src/rebot_sim2real/perception_cnn.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 93 |
| `src/rebot_sim2real/rl/__init__.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 1 |
| `src/rebot_sim2real/rl/config.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 306 |
| `src/rebot_sim2real/rl/demo.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 34 |
| `src/rebot_sim2real/rl/rewards.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 98 |
| `src/rebot_sim2real/rl/sac.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 202 |
| `src/rebot_sim2real/rl/train.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 359 |
| `src/rebot_sim2real/viz/__init__.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 5 |
| `src/rebot_sim2real/viz/dashboard.py` | `b4c128e` (2026-06-20) → `snapshot_2026-06-20/` | 292 |
| `configs/train_residual.yaml` | `65778d3` (2026-06-25) → `additions_2026-06-21_to_25/` | 98 |
| `configs/train_residual_asym.yaml` | `65778d3` (2026-06-25) → `additions_2026-06-21_to_25/` | 111 |
| `configs/train_residual_dr.yaml` | `65778d3` (2026-06-25) → `additions_2026-06-21_to_25/` | 98 |
| `configs/train_residual_vision.yaml` | `65778d3` (2026-06-25) → `additions_2026-06-21_to_25/` | 100 |
| `configs/train_residual_wristassist.yaml` | `65778d3` (2026-06-25) → `additions_2026-06-21_to_25/` | 116 |
| `configs/train_residual_wristforce.yaml` | `65778d3` (2026-06-25) → `additions_2026-06-21_to_25/` | 115 |
| `configs/train_v6a.yaml` | `65778d3` (2026-06-25) → `additions_2026-06-21_to_25/` | 161 |
| `configs/train_v6a_yaw.yaml` | `65778d3` (2026-06-25) → `additions_2026-06-21_to_25/` | 163 |
| `configs/train_v6b_deepreach.yaml` | `65778d3` (2026-06-25) → `additions_2026-06-21_to_25/` | 170 |
| `deploy/calib_grip_close.py` | `65778d3` (2026-06-25) → `additions_2026-06-21_to_25/` | 81 |
| `deploy/calib_grip_interactive.py` | `65778d3` (2026-06-25) → `additions_2026-06-21_to_25/` | 75 |
| `deploy/debug_deepreach.py` | `65778d3` (2026-06-25) → `additions_2026-06-21_to_25/` | 664 |
| `deploy/front_gripper_z.py` | `65778d3` (2026-06-25) → `additions_2026-06-21_to_25/` | 102 |
| `docs/08_residual_rl_v6_status.md` | `65778d3` (2026-06-25) → `additions_2026-06-21_to_25/` | 144 |
| `docs/09_handoff_2026-06-24.md` | `65778d3` (2026-06-25) → `additions_2026-06-21_to_25/` | 59 |
| `docs/10_handoff_deepreach.md` | `65778d3` (2026-06-25) → `additions_2026-06-21_to_25/` | 80 |
| `docs/11_handoff_robust_hybrid.md` | `65778d3` (2026-06-25) → `additions_2026-06-21_to_25/` | 72 |
| `docs/12_handoff_deploy_grasp.md` | `65778d3` (2026-06-25) → `additions_2026-06-21_to_25/` | 69 |
| `scripts/collect_vla_dataset.py` | `65778d3` (2026-06-25) → `additions_2026-06-21_to_25/` | 131 |
| `scripts/eval_grid100.py` | `65778d3` (2026-06-25) → `additions_2026-06-21_to_25/` | 138 |
| `scripts/eval_wrist_ablation.py` | `65778d3` (2026-06-25) → `additions_2026-06-21_to_25/` | 154 |
| `scripts/gen_perception_data_wrist.py` | `65778d3` (2026-06-25) → `additions_2026-06-21_to_25/` | 257 |
| `scripts/plot_success_surface.py` | `65778d3` (2026-06-25) → `additions_2026-06-21_to_25/` | 101 |
| `scripts/sweep_success_surface.py` | `65778d3` (2026-06-25) → `additions_2026-06-21_to_25/` | 161 |
| `scripts/viz_rollout.py` | `65778d3` (2026-06-25) → `additions_2026-06-21_to_25/` | 190 |
| `src/rebot_sim2real/rl/factory.py` | `65778d3` (2026-06-25) → `additions_2026-06-21_to_25/` | 33 |
| `tests/test_nstep_buffer.py` | `65778d3` (2026-06-25) → `additions_2026-06-21_to_25/` | 160 |

### Excluded from C

| File (HEAD `65778d3`) | Line authors at HEAD | Why excluded |
|---|---|---|
| `.gitignore` | colleague 3, GitHub template 212, colleague (set-up account) 33, author 26 | contains lines by colleague, GitHub template, colleague (set-up account) (colleague / not the author); removing them would change or break the file |
| `.python-version` | colleague (set-up account) 1 | contains lines by colleague (set-up account) (colleague / not the author); removing them would change or break the file |
| `README.md` | colleague 114, GitHub template 1, colleague (set-up account) 15, author 43 | contains lines by colleague, GitHub template, colleague (set-up account) (colleague / not the author); removing them would change or break the file |
| `assets/configuration/rebot_b601_dm_base.usd` | author binary | binary USD converted from the vendor URDF; 39.6 MB (base); vendor-asset licence unclear |
| `assets/configuration/rebot_b601_dm_physics.usd` | author binary | binary USD converted from the vendor URDF; 39.6 MB (base); vendor-asset licence unclear |
| `assets/configuration/rebot_b601_dm_robot.usd` | author binary | binary USD converted from the vendor URDF; 39.6 MB (base); vendor-asset licence unclear |
| `assets/configuration/rebot_b601_dm_sensor.usd` | author binary | binary USD converted from the vendor URDF; 39.6 MB (base); vendor-asset licence unclear |
| `assets/robot_description_gripper/meshes/base_link.STL` | colleague (set-up account) binary | contains lines by colleague (set-up account) (colleague / not the author); removing them would change or break the file |
| `assets/robot_description_gripper/meshes/gripper_left.STL` | colleague (set-up account) binary | contains lines by colleague (set-up account) (colleague / not the author); removing them would change or break the file |
| `assets/robot_description_gripper/meshes/gripper_link.STL` | colleague (set-up account) binary | contains lines by colleague (set-up account) (colleague / not the author); removing them would change or break the file |
| `assets/robot_description_gripper/meshes/gripper_right.STL` | colleague (set-up account) binary | contains lines by colleague (set-up account) (colleague / not the author); removing them would change or break the file |
| `assets/robot_description_gripper/meshes/link1.STL` | colleague (set-up account) binary | contains lines by colleague (set-up account) (colleague / not the author); removing them would change or break the file |
| `assets/robot_description_gripper/meshes/link2.STL` | colleague (set-up account) binary | contains lines by colleague (set-up account) (colleague / not the author); removing them would change or break the file |
| `assets/robot_description_gripper/meshes/link3.STL` | colleague (set-up account) binary | contains lines by colleague (set-up account) (colleague / not the author); removing them would change or break the file |
| `assets/robot_description_gripper/meshes/link4.STL` | colleague (set-up account) binary | contains lines by colleague (set-up account) (colleague / not the author); removing them would change or break the file |
| `assets/robot_description_gripper/meshes/link5.STL` | colleague (set-up account) binary | contains lines by colleague (set-up account) (colleague / not the author); removing them would change or break the file |
| `assets/robot_description_gripper/meshes/link6.STL` | colleague (set-up account) binary | contains lines by colleague (set-up account) (colleague / not the author); removing them would change or break the file |
| `assets/robot_description_gripper/reBot_B601_DM_with_gripper.urdf` | colleague (set-up account) 564 | contains lines by colleague (set-up account) (colleague / not the author); removing them would change or break the file |
| `assets/robot_description_gripper/rebot_prepared.urdf` | colleague (set-up account) 285 | contains lines by colleague (set-up account) (colleague / not the author); removing them would change or break the file |
| `configs/scene.yaml` | colleague 3, colleague (set-up account) 19, author 15 | contains lines by colleague, colleague (set-up account) (colleague / not the author); removing them would change or break the file |
| `configs/train_v6.yaml` | colleague 105, author 46 | contains lines by colleague (colleague / not the author); removing them would change or break the file |
| `deploy/debug_dashboard.py` | colleague 129, author 58 | contains lines by colleague (colleague / not the author); removing them would change or break the file |
| `deploy/debug_grasp.py` | colleague 125, author 15 | contains lines by colleague (colleague / not the author); removing them would change or break the file |
| `deploy/debug_reach.py` | colleague 119, author 5 | contains lines by colleague (colleague / not the author); removing them would change or break the file |
| `docs/01_sac_sim2real_camera_rl_research.md` | colleague 11, colleague (set-up account) 232 | contains lines by colleague, colleague (set-up account) (colleague / not the author); removing them would change or break the file |
| `docs/02_isaac_fsm_architecture.md` | colleague 8, colleague (set-up account) 291 | contains lines by colleague, colleague (set-up account) (colleague / not the author); removing them would change or break the file |
| `docs/03_reference_alignment.md` | colleague 11, colleague (set-up account) 107 | contains lines by colleague, colleague (set-up account) (colleague / not the author); removing them would change or break the file |
| `docs/04_sim2real_rl_sota.md` | colleague 182 | contains lines by colleague (colleague / not the author); removing them would change or break the file |
| `docs/05_residual_rl_pick_place.md` | colleague 181 | contains lines by colleague (colleague / not the author); removing them would change or break the file |
| `docs/06_resfit_residual_rl_plan.md` | colleague 287 | contains lines by colleague (colleague / not the author); removing them would change or break the file |
| `docs/07_deploy_debugging.md` | colleague 87 | contains lines by colleague (colleague / not the author); removing them would change or break the file |
| `docs/figures/reach_map.png` | colleague binary | contains lines by colleague (colleague / not the author); removing them would change or break the file |
| `docs/figures/scene_layout.png` | colleague (set-up account) binary | contains lines by colleague (set-up account) (colleague / not the author); removing them would change or break the file |
| `docs/figures/topcam_compare.png` | colleague binary | contains lines by colleague (colleague / not the author); removing them would change or break the file |
| `pyproject.toml` | colleague 16, colleague (set-up account) 33, author 3 | contains lines by colleague, colleague (set-up account) (colleague / not the author); removing them would change or break the file |
| `setup_env.sh` | colleague 8, colleague (set-up account) 24 | contains lines by colleague, colleague (set-up account) (colleague / not the author); removing them would change or break the file |
| `src/rebot_sim2real/__init__.py` | colleague (set-up account) 2 | contains lines by colleague (set-up account) (colleague / not the author); removing them would change or break the file |
| `src/rebot_sim2real/constants.py` | colleague 25, colleague (set-up account) 88, author 102 | contains lines by colleague, colleague (set-up account) (colleague / not the author); removing them would change or break the file |
| `src/rebot_sim2real/rl/base_controller.py` | colleague 171, author 368 | contains lines by colleague (colleague / not the author); removing them would change or break the file |
| `src/rebot_sim2real/rl/nstep_buffer.py` | colleague 50, author 82 | contains lines by colleague (colleague / not the author); removing them would change or break the file |
| `src/rebot_sim2real/rl/resfit.py` | colleague 152, author 18 | contains lines by colleague (colleague / not the author); removing them would change or break the file |
| `src/rebot_sim2real/scene/__init__.py` | colleague (set-up account) 1 | contains lines by colleague (set-up account) (colleague / not the author); removing them would change or break the file |
| `src/rebot_sim2real/scene/builder.py` | colleague 5, colleague (set-up account) 59, author 45 | contains lines by colleague, colleague (set-up account) (colleague / not the author); removing them would change or break the file |
| `src/rebot_sim2real/scene/cameras.py` | colleague 9, colleague (set-up account) 82, author 80 | contains lines by colleague, colleague (set-up account) (colleague / not the author); removing them would change or break the file |
| `uv.lock` | colleague (set-up account) 2081, author 98 | contains lines by colleague (set-up account) (colleague / not the author); removing them would change or break the file |

Also excluded: the colleague-authored figures (`docs/figures/*.png`), the colleague's design documents
`docs/01`–`07`, and the vendor URDF / STL meshes (committed by `colleague (set-up account)`).

## 7. Other material not included

| What | Where | Why |
|---|---|---|
| Real-robot deploy runner for the C51 teacher and the visual student (config / FK / perception / adapter / runner, notes `PARITY.md`, `SQUINT_NOTES.md`, `ALIGNMENT.md`, `TRAINING_FEEDBACK.md`) | second workstation, `~/workspace/sim2real/` (not a git repository) | authorship cannot be established from the files (no version control; shared machine account); left out until it can be attributed. The "real-robot feedback" items quoted in the commit messages of A and B (e.g. "wanders after align, never grasps") come from this deploy work |
| Exported deploy packages `deploy_package(4)`, `(5)` | same workstation, Downloads | copies of B's `deploy_package/` |
| July–August evaluation workspace `rebot_eval/` (183 output folders, detectors, `train_c51*`, `train_dagger_*` variants) | same workstation | not a git repository, authorship unknown, later than this project |
| Vendor reBot URDF + meshes `robot_description_gripper.zip` (9.5 MB, A and B `deploy_package/`) | | the archive has no licence file and its origin is not recorded; Seeed Studio's `reBot-Isaacsim` repository is MIT-licensed as of 2026-09, but it was not confirmed that this archive is the same material. Obtain the URDF from the vendor |
| Isaac workspace scratch (URDF→USD conversion, FK and diagnosis scripts, LeIsaac clone, Squint clone, DINOv3 weights) | `~/workspace/rebot_isaac_rl/` | untracked set-up scratch, third-party clones and weights |
| MLflow experiments 4, 7–18 (2026-06-18..21, `pick_*` runs) | cluster MLflow | runs of source C under a workstation account; no metric file exported here |

### Searched for, not found (2026-11-01)

Training logs or evaluation files of A and B: the `rl_c51_dyndr*.log` files and MLflow experiment 19 cited in an
earlier internal audit, TensorBoard event files, and any per-episode evaluation output. A read-only search covered the
training workstation, the cluster NFS, the second workstation, the author's laptop and an external archive drive.
Consequently every success rate in the original commit messages is unbacked by a log (see [claims.md](claims.md));
the only success rates reported in this repository are the re-evaluation files in `results/` (if present).

## 8. Media added on 2026-11-01

The README media in `media/readme/` were made from files found in a read-only search of the training workstation,
the second workstation (the account the author confirmed as their own, via CPU-only cluster jobs), the laptop and the
external drive. Originals, timestamps, checksums, processing and the material found but not used are listed in
[media/README.md](../media/README.md). One folder on the second workstation (`~/Downloads/real_diag/`) also held an
unrelated personal document of another person; it was not opened or copied (only file names were listed).

## 9. July – August 2026 sources

Added 2026-11-02. These files come from the author's machines: the training workstation (an RTX 4090 node; its
login account is shared, and the files below are tied to the author's runs by their paths, ports 8951/8953 and
the bench node) and the second workstation (the account the author confirmed as their own). They were copied
read-only through the cluster. None of the source folders is under version control, except `rebot_sim2real` (§10).

### Trainer: which copy was taken

| Copy | MD5 | Lines | Difference to the copy taken (+/− lines) |
|---|---|---|---|
| **training workstation `~/_stage_bench/train_sac_det_xattn.py`, 2026-08-02 15:39 (taken)** | `b592d8b7b0ac362135e2471f6cf6a2be` | 2,610 | — (the bench daemon shipped this folder as the bench working directory) |
| training workstation `~/_stage_9t_dacq/`, 2026-08-02 07:12 (training of the benched approach run) | `c1e18fd82e8abd13b939ef8493636710` | 2,631 | +24 / −45 |
| training workstation `~/_stage_curric_edit/`, 2026-08-01 20:04 | `37c4571d9272db13bec4e6a5e111c0f3` | 2,555 | +96 / −41 |
| training workstation `~/curric_sweep_code/`, 2026-07-23 18:29 | `bf0bcd873ef56e0fda49ed94cec75f1f` | 2,595 | +261 / −246 |
| second workstation `~/curric_sweep_code/`, 2026-07-23 | `05d05a0366f4c92d3751a7c6cd70d4e9` | 2,595 | +260 / −245 |

The copy taken is the newest, and it is the code that produced `bench_daemon.log`. The benched approach checkpoints
were trained with the `_stage_9t_dacq` copy (differences: training-side reward/heat-label edits, see the diff stats).

Helpers: `det_xattn_nets.py` (`93706705…`), `robot_profile.py` (`da0dc231…`), `runtime.py` (`cc1b1796…`),
`tp_config.py` (`33fa3893…`), `ui_render.py` (`41c7fb11…`) from the same `_stage_bench` folder. `tp_net.py`
(`6dd950a4…`) is not in that folder; the bench found it on its `PYTHONPATH` (second workstation `~/rebot_eval/`,
2026-07-19), which is where it was taken from.

### Environment code imported by the trainer

`src/det_xattn/packages/sim/env_setup/` is the snapshot in the second workstation's `~/rebot_eval/packages/` (the
`PYTHONPATH` of the benches). Compared with the June version in `src/rebot_sim2real/env/`:

| File | June (ghrepos HEAD) | July snapshot | Diff |
|---|---|---|---|
| `env_config.py` | 531 lines, `1e7cc5b9…` | 581 lines, `857ea9f1…`, mtime 2026-08-07 15:55 | +109 / −58 |
| `maniskill_rewards.py` | 710 lines, `e4835bda…` | 983 lines, `c9e6e686…`, 2026-06-25 | +330 / −57 |
| `visual_obs.py` | 396 lines | 466 lines, `f7e61e20…`, 2026-06-25 | +93 / −23 |
| `target_delta_action.py`, `__init__.py` | | identical | 0 |

`env_config.py` was modified on 2026-08-07, after the benches (08-02/03). Its content is identical to a backup taken
at 08-07 15:44 before an edit that was then undone. Whether it changed between 08-03 and 08-07 is not recorded.

### Changes made here (all behaviour-preserving for training and the original benchmark)

* Korean comments, log strings and dashboard labels translated to English. An AST comparison with every string
  constant blanked and docstrings dropped shows identical code for all files. The anchors that the two log tools
  search for in the dashboard HTML were translated to match.
* Three machine-specific path defaults replaced (`LEISAAC_ASSETS_ROOT`, a debug PNG path, `EXPORT_DIR`).
* `eval_hooks.py` added, with three opt-in calls in the trainer (seed, per-episode CSV, summary + exit). All are
  no-ops unless `EVAL_OUT_DIR` is set.

### Checkpoints, logs, grids

| File here | Source | MD5 of the source |
|---|---|---|
| `checkpoints/project2/search_last.pt` (step 8,038,400) | training workstation `~/_stage_bench/ckpt_in/`, 2026-08-03 12:43 | `19f0629819b68f4b7899ae5451822e25` |
| `checkpoints/project2/approach_last.pt` (step 3,712,000) | same | `412453abbcce602694bd862c6e836b86` |
| `results/project2/bench_daemon.log` | training workstation `~/bench_daemon.log` (2026-08-02 19:27 → 08-03 12:46) | `af67cbb044daf24c47538e21f2811874` |
| `results/project2/ntfy_poller.log` | training workstation `~/ntfy_poller.log` (2026-07-31 → 08-05) | `f546440b57f27d3e34eb5c144036f5a1` |
| `results/project2/CHANGES_8953_20260731.md` | training workstation `~/`, 2026-07-31 07:07 | `bbec035b9fd4d1f6f3820cea5c31757b` |
| `results/project2/heatmaps/*.png` (124) | training workstation `~/bench_heatmaps/` | unchanged |
| `results/project2/eval_grids/evalgrid_{86,90,100}cells_2026-08-03.npy` | second workstation `~/rebot_eval/evalgrid100.{bak_86,bak_90,npy}` | `92fc8698…`, `6737118f…`, `0d0623ab…` |
| `results/project2/eval_grids/eval_set_400cells_2026-07-21.npy` | second workstation `~/rebot_eval/eval_sets/eval_set_20x20.npy` | |
| `scripts/project2_tools/bench_daemon.py`, `ntfy_sweep_poller.py` | training workstation `~/` | `678d2586…`, `aef508c2…` |
| `configs/project2_bench_2026-08-03.env` | knobs of `~/_bench_graphic_ep.txt` (2026-08-02 18:53) | |
| `configs/project2_train_9t_dacq_2026-08-02.env` | knobs of `~/_ep9t_dacq.txt` (2026-08-02 18:53) | |

The logs, tools and change note were redacted before commit: cluster addresses, home paths, the push-notification
topic, Ray package ids and another user's name were replaced by placeholders. The MD5 sums above are those of the
unredacted originals.

Which grid the benches used: the bench entrypoint names `evalgrid100.npy`. The log shows 90 cells until about 08-03
10:41 and 86 cells afterwards, and the file was replaced at 08-03 13:01 by a 100-cell grid after the last bench. The
86- and 90-cell backups made at those times are the grids of the benches; the re-evaluation uses the 86-cell one.

Not included: the 08-12 checkpoints `~/rebot_eval/ckpt_stepeval_app/` (search, approach, grasp, lift; 33 MB, over
the 30 MB limit; the approach file is from a 6k-step smoke run `det_TP304_ab1`). Also not included: other run
folders (183 output folders, mostly empty or checkpoints only), the YOLO-OBB detector weights (47–470 MB; the one the benches and training used, `det_cube_obb_yolo26m_ft3.pt`, is 188 MB) and
`_running_train.py`. The July training job logs are no longer on the cluster (Ray history starts 2026-08-14), and
MLflow has only three detector-training runs for this period.

## 10. Real-robot code (July): blame at `7f36e69`

Source: second workstation `~/workspace/rebot_sim2real`, branch `main`, HEAD `7f36e69` (2026-07-24). All 82 commits
since 2026-07-01 are by the author. The base of the repository still contains lines by a
colleague (`colleague`, 2026-06-23) and by `colleague (set-up account)` (the 2026-06-15 set-up). Rule: a file is included only if every
line (`git blame -w`) at `7f36e69` is by the author. Files in the repository's own `_archive/` folder are left out
even when author-only. The four uncommitted files in the working tree were not used.

Consequence: the IK engine (`rbot_calculator/base_controller.py`, `config.py`), `constants.py`, `perception.py` and
`grasp_traj.py` contain colleague lines and are excluded, so `archive/real_robot_july/` does not run. Text was
translated from Korean (code unchanged by AST check). Session-workflow notes and addresses were removed from the
handoff notes.

### Included (author-only at `7f36e69`)

| File | Lines |
|---|---|
| `configs/deploy_env.yaml` | 163 lines, all by the author |
| `deploy/cam_front_intrinsics.npz` | binary, committed by the author |
| `deploy/joint_map.npz` | binary, committed by the author |
| `deploy/main.py` | 16 lines, all by the author |
| `docs/08_residual_rl_v6_status.md` | 144 lines, all by the author |
| `docs/09_handoff_2026-06-24.md` | 59 lines, all by the author |
| `docs/10_handoff_deepreach.md` | 80 lines, all by the author |
| `docs/11_handoff_robust_hybrid.md` | 72 lines, all by the author |
| `docs/12_handoff_deploy_grasp.md` | 69 lines, all by the author |
| `docs/13_handoff_deploy_calib_smoothing.md` | 98 lines, all by the author |
| `docs/14_handoff_vision_token_retrain.md` | 97 lines, all by the author |
| `docs/15_handoff_v6c_train_diagnosis.md` | 54 lines, all by the author |
| `docs/16_handoff_perception_topwrist_coop.md` | 58 lines, all by the author |
| `docs/17_deep_reach_world.md` | 179 lines, all by the author |
| `docs/18_handoff_deploy_grasp_align.md` | 48 lines, all by the author |
| `docs/19_handoff_deploy_wristcentric_incremental.md` | 91 lines, all by the author |
| `docs/20_handoff_hover07e_sac_deploy_bringup.md` | 70 lines, all by the author |
| `docs/21_handoff_j4_reassembly_recalib_fkzcorr.md` | 122 lines, all by the author |
| `docs/22_handoff_grasp_lift_verified_and_lookat_rnd.md` | 103 lines, all by the author |
| `docs/23_handoff_multiclass_detector_and_motor_kp_apr.md` | 102 lines, all by the author |
| `docs/24_handoff_kpapr_configA_and_perception_stale_bug.md` | 59 lines, all by the author |
| `docs/25_handoff_gimbal_lookat_perception_gripper_kpapr.md` | 57 lines, all by the author |
| `docs/26_handoff_openvocab_approach_refactor.md` | 83 lines, all by the author |
| `docs/27_handoff_grasp_point_fkik_and_perception_30hz.md` | 77 lines, all by the author |
| `docs/28_deploy_3thread_refactor_design.md` | 266 lines, all by the author |
| `docs/29_handoff_3thread_refactor_and_follow_and_iothread.md` | 69 lines, all by the author |
| `docs/30_handoff_follow_autocenter_mjpg30hz_autograsp.md` | 67 lines, all by the author |
| `docs/31_handoff_reach_z_fix_and_hwlimit_and_fktouch.md` | 73 lines, all by the author |
| `docs/32_handoff_deploy_refactor_mainpy_reachZ_refit.md` | 71 lines, all by the author |
| `docs/33_handoff_grasp_working_jointmap_reachZ_yaw.md` | 61 lines, all by the author |
| `docs/34_handoff_readability_cleanup_rl_removal_main_consolidation.md` | 63 lines, all by the author |
| `docs/cluster_training.md` | 101 lines, all by the author |
| `src/rebot_sim2real/deploy/__init__.py` | 11 lines, all by the author |
| `src/rebot_sim2real/deploy/blackboard.py` | 121 lines, all by the author |
| `src/rebot_sim2real/deploy/cli.py` | 500 lines, all by the author |
| `src/rebot_sim2real/deploy/control/__init__.py` | 9 lines, all by the author |
| `src/rebot_sim2real/deploy/control/commands.py` | 431 lines, all by the author |
| `src/rebot_sim2real/deploy/control/constants.py` | 47 lines, all by the author |
| `src/rebot_sim2real/deploy/control/context.py` | 161 lines, all by the author |
| `src/rebot_sim2real/deploy/control/control_loop.py` | 287 lines, all by the author |
| `src/rebot_sim2real/deploy/control/modes.py` | 746 lines, all by the author |
| `src/rebot_sim2real/deploy/control/robot_io_facade.py` | 172 lines, all by the author |
| `src/rebot_sim2real/deploy/control/telemetry.py` | 87 lines, all by the author |
| `src/rebot_sim2real/deploy/control/tick.py` | 59 lines, all by the author |
| `src/rebot_sim2real/deploy/io/__init__.py` | 4 lines, all by the author |
| `src/rebot_sim2real/deploy/io/io_thread.py` | 168 lines, all by the author |
| `src/rebot_sim2real/deploy/io/robot_io.py` | 393 lines, all by the author |
| `src/rebot_sim2real/deploy/kinematics.py` | 192 lines, all by the author |
| `src/rebot_sim2real/deploy/main.py` | 242 lines, all by the author |
| `src/rebot_sim2real/deploy/main_dashboard.py` | 258 lines, all by the author |
| `src/rebot_sim2real/deploy/vision/__init__.py` | 4 lines, all by the author |
| `src/rebot_sim2real/deploy/vision/front_gripper_z.py` | 102 lines, all by the author |
| `src/rebot_sim2real/deploy/vision/openvocab.py` | 171 lines, all by the author |
| `src/rebot_sim2real/deploy/vision/perception_thread.py` | 602 lines, all by the author |
| `src/rebot_sim2real/rbot_calculator/__init__.py` | 9 lines, all by the author |
| `src/tools/calib_extrinsic_checkerboard.py` | 233 lines, all by the author |
| `src/tools/calib_grip.py` | 247 lines, all by the author |
| `src/tools/calib_intrinsics_checkerboard.py` | 152 lines, all by the author |
| `src/tools/damiao_params.py` | 78 lines, all by the author |
| `src/tools/fit_wrist_calib.py` | 205 lines, all by the author |
| `src/tools/fit_wrist_calib_constrained.py` | 131 lines, all by the author |
| `src/tools/fk_verify_z.py` | 137 lines, all by the author |
| `src/tools/make_checkerboard_pdf.py` | 76 lines, all by the author |
| `src/tools/measure_latency.py` | 146 lines, all by the author |

### Excluded

| File | Line authors at `7f36e69` | Why |
|---|---|---|
| `.gitignore` | GitHub template 212, author 36, colleague (set-up account) 33, colleague 3 | contains lines by a colleague or a template; not removable without changing the file |
| `.python-version` | colleague (set-up account) 1 | contains lines by a colleague or a template; not removable without changing the file |
| `README.md` | GitHub template 1, author 45, colleague (set-up account) 15, colleague 112 | contains lines by a colleague or a template; not removable without changing the file |
| `_archive/builder.py` | author 45, colleague (set-up account) 59, colleague 5 | contains lines by a colleague or a template; not removable without changing the file |
| `_archive/calib_extrinsic.py` | author 85, colleague 27 | contains lines by a colleague or a template; not removable without changing the file |
| `_archive/collect_vla_dataset.py` | author 131 | author-only, but in the repository's own `_archive/` graveyard (superseded code) |
| `_archive/deploy/blackboard.py` | author 9 | author-only, but in the repository's own `_archive/` graveyard (superseded code) |
| `_archive/deploy/debug_dashboard.py` | author 9 | author-only, but in the repository's own `_archive/` graveyard (superseded code) |
| `_archive/deploy/debug_deepreach.py` | author 2086 | author-only, but in the repository's own `_archive/` graveyard (superseded code) |
| `_archive/deploy/deep_reach_world.py` | author 625 | author-only, but in the repository's own `_archive/` graveyard (superseded code) |
| `_archive/deploy/deepreach_sac_world.py` | author 396 | author-only, but in the repository's own `_archive/` graveyard (superseded code) |
| `_archive/deploy/dry_run.py` | author 90, colleague 15 | contains lines by a colleague or a template; not removable without changing the file |
| `_archive/deploy/front_gripper_z.py` | author 9 | author-only, but in the repository's own `_archive/` graveyard (superseded code) |
| `_archive/deploy/front_openvocab.py` | author 9 | author-only, but in the repository's own `_archive/` graveyard (superseded code) |
| `_archive/deploy/io_thread.py` | author 9 | author-only, but in the repository's own `_archive/` graveyard (superseded code) |
| `_archive/deploy/kinematics.py` | author 9 | author-only, but in the repository's own `_archive/` graveyard (superseded code) |
| `_archive/deploy/robot_io.py` | author 9 | author-only, but in the repository's own `_archive/` graveyard (superseded code) |
| `_archive/deploy/run_pick.py` | author 209, colleague 38 | contains lines by a colleague or a template; not removable without changing the file |
| `_archive/deploy/vision_thread.py` | author 9 | author-only, but in the repository's own `_archive/` graveyard (superseded code) |
| `_archive/wrist_ibvs_sim.py` | author 127, colleague 52 | contains lines by a colleague or a template; not removable without changing the file |
| `_archive/wrist_servo_sim.py` | author 164, colleague 39 | contains lines by a colleague or a template; not removable without changing the file |
| `assets/configuration/rebot_b601_dm_base.usd` | binary, last commit by author | USD converted from the vendor URDF (39.6 MB) / vendor meshes |
| `assets/configuration/rebot_b601_dm_physics.usd` | binary, last commit by author | USD converted from the vendor URDF (39.6 MB) / vendor meshes |
| `assets/configuration/rebot_b601_dm_robot.usd` | binary, last commit by author | USD converted from the vendor URDF (39.6 MB) / vendor meshes |
| `assets/configuration/rebot_b601_dm_sensor.usd` | binary, last commit by author | USD converted from the vendor URDF (39.6 MB) / vendor meshes |
| `assets/robot_description_gripper/meshes/base_link.STL` | binary, last commit by colleague (set-up account) | USD converted from the vendor URDF (39.6 MB) / vendor meshes |
| `assets/robot_description_gripper/meshes/gripper_left.STL` | binary, last commit by colleague (set-up account) | USD converted from the vendor URDF (39.6 MB) / vendor meshes |
| `assets/robot_description_gripper/meshes/gripper_link.STL` | binary, last commit by colleague (set-up account) | USD converted from the vendor URDF (39.6 MB) / vendor meshes |
| `assets/robot_description_gripper/meshes/gripper_right.STL` | binary, last commit by colleague (set-up account) | USD converted from the vendor URDF (39.6 MB) / vendor meshes |
| `assets/robot_description_gripper/meshes/link1.STL` | binary, last commit by colleague (set-up account) | USD converted from the vendor URDF (39.6 MB) / vendor meshes |
| `assets/robot_description_gripper/meshes/link2.STL` | binary, last commit by colleague (set-up account) | USD converted from the vendor URDF (39.6 MB) / vendor meshes |
| `assets/robot_description_gripper/meshes/link3.STL` | binary, last commit by colleague (set-up account) | USD converted from the vendor URDF (39.6 MB) / vendor meshes |
| `assets/robot_description_gripper/meshes/link4.STL` | binary, last commit by colleague (set-up account) | USD converted from the vendor URDF (39.6 MB) / vendor meshes |
| `assets/robot_description_gripper/meshes/link5.STL` | binary, last commit by colleague (set-up account) | USD converted from the vendor URDF (39.6 MB) / vendor meshes |
| `assets/robot_description_gripper/meshes/link6.STL` | binary, last commit by colleague (set-up account) | USD converted from the vendor URDF (39.6 MB) / vendor meshes |
| `assets/robot_description_gripper/reBot_B601_DM_with_gripper.urdf` | colleague (set-up account) 564 | contains lines by a colleague or a template; not removable without changing the file |
| `assets/robot_description_gripper/rebot_prepared.urdf` | colleague (set-up account) 285 | contains lines by a colleague or a template; not removable without changing the file |
| `docs/01_sac_sim2real_camera_rl_research.md` | colleague (set-up account) 232, colleague 11 | contains lines by a colleague or a template; not removable without changing the file |
| `docs/02_isaac_fsm_architecture.md` | colleague (set-up account) 291, colleague 8 | contains lines by a colleague or a template; not removable without changing the file |
| `docs/03_reference_alignment.md` | colleague (set-up account) 107, colleague 11 | contains lines by a colleague or a template; not removable without changing the file |
| `docs/04_sim2real_rl_sota.md` | colleague 182 | contains lines by a colleague or a template; not removable without changing the file |
| `docs/05_residual_rl_pick_place.md` | colleague 181 | contains lines by a colleague or a template; not removable without changing the file |
| `docs/06_resfit_residual_rl_plan.md` | colleague 287 | contains lines by a colleague or a template; not removable without changing the file |
| `docs/07_deploy_debugging.md` | colleague 87 | contains lines by a colleague or a template; not removable without changing the file |
| `docs/figures/reach_map.png` | binary, last commit by colleague | binary |
| `docs/figures/scene_layout.png` | binary, last commit by colleague (set-up account) | binary |
| `docs/figures/topcam_compare.png` | binary, last commit by colleague | binary |
| `pyproject.toml` | author 3, colleague (set-up account) 33, colleague 16 | contains lines by a colleague or a template; not removable without changing the file |
| `scripts/calib_intrinsics.py` | author 65, colleague 13 | contains lines by a colleague or a template; not removable without changing the file |
| `setup_env.sh` | author 4, colleague (set-up account) 20, colleague 8 | contains lines by a colleague or a template; not removable without changing the file |
| `src/rebot_sim2real/__init__.py` | colleague (set-up account) 2 | contains lines by a colleague or a template; not removable without changing the file |
| `src/rebot_sim2real/constants.py` | author 102, colleague (set-up account) 88, colleague 25 | contains lines by a colleague or a template; not removable without changing the file |
| `src/rebot_sim2real/grasp_traj.py` | author 63, colleague 20 | contains lines by a colleague or a template; not removable without changing the file |
| `src/rebot_sim2real/perception.py` | author 130, colleague 7 | contains lines by a colleague or a template; not removable without changing the file |
| `src/rebot_sim2real/rbot_calculator/base_controller.py` | author 473, colleague 171 | contains lines by a colleague or a template; not removable without changing the file |
| `src/rebot_sim2real/rbot_calculator/config.py` | author 418, colleague 36 | contains lines by a colleague or a template; not removable without changing the file |
| `src/tools/calib_convention.py` | author 77, colleague 25 | contains lines by a colleague or a template; not removable without changing the file |
| `src/tools/calib_extrinsic_known.py` | author 111, colleague 14 | contains lines by a colleague or a template; not removable without changing the file |
| `src/tools/debug_grasp.py` | author 21, colleague 118 | contains lines by a colleague or a template; not removable without changing the file |
| `src/tools/debug_reach.py` | author 11, colleague 112 | contains lines by a colleague or a template; not removable without changing the file |
| `src/tools/diag_front.py` | author 96, colleague 21 | contains lines by a colleague or a template; not removable without changing the file |
| `src/tools/diag_grasp.py` | author 42, colleague 10 | contains lines by a colleague or a template; not removable without changing the file |
| `src/tools/sysid_arm.py` | author 86, colleague 23 | contains lines by a colleague or a template; not removable without changing the file |
| `src/tools/wrist_view.py` | author 52, colleague 10 | contains lines by a colleague or a template; not removable without changing the file |
| `uv.lock` | author 166, colleague (set-up account) 2032 | contains lines by a colleague or a template; not removable without changing the file |


## 11. July media

| File | Source | Note |
|---|---|---|
| `media/project2/sim_official_model_views_2026-07-23.png` | training workstation `~/ui_front.png`, `ui_side.png`, `wnow.png`, `front.png` (2026-07-23) | simulation calibration views of the OFFICIAL robot model, put side by side |
| `media/project2/bench_heatmap_e3712k_2026-08-03.png` | `results/project2/heatmaps/bench_e3712k.png` | copy |
| `media/project2/project2_log_curves.png` | made by `analysis/plot_project2.py` from the parsed logs | |

Checked and not used: the 2026-07-07 checkerboard calibration frames (a bystander is in the frame), a 2026-07-28
screenshot (another project), and the OFFICIAL-model wrist renders of 2026-07-31.

## 12. Simulation versions and rollout media (2026-11-02)

Simulation versions and the evidence for the latest one used: [simulation_versions.md](simulation_versions.md).
Robot USD check (read-only, 2026-11-02): `rebot_follower.usd` has MD5 `39c27cdccf5296e561f5e21d86dca18d` on both
the training machine (`~/workspace/rebot_isaac_rl/leisaac/assets/robots/`) and the bench machine
(`~/rebot_eval/assets/robots/`), file date 2026-06-08 19:34. Search for July–August rollout videos (read-only):
training workstation home, second workstation home, the laptop's Downloads / Desktop / Movies / Pictures, MLflow.
None was found; the trainer's live clip existed only in memory.

Rendering was not possible on 2026-11-02:
- The training workstation's GPU 0 was held by another user's job (22.4 GB), and its GPU 1 runs another session's
  training.
- The 5090 node was fully used (30.5 of 32.6 GB, 100 %). A quick check found no Isaac Sim Python package there; a `~/rebot_eval` folder exists and was not inspected further.
- The second workstation has the stack, but was not used without the owner's go-ahead.

Prepared instead:
- `eval_hooks.video_frame` / `video_end`, opt-in through `EVAL_VIDEO_DIR`;
- run `R` in `scripts/eval/project2_eval.sh` (1 env, seed 0, the 4 cases in
  `results/project2/eval_grids/render_cases_4_from_86cells.npy`, indices 36, 34, 39, 75 of the 86-cell grid).

## 13. S4 rollout render (2026-11-02)

Approved by the author for this one job. Second workstation (RTX 5080, 16 GB; another process held 3.0 GB at the
start, read-only check), Ray job `rebot_s4_render_20261102_2058`, pinned to that node with 1 GPU, `CUDA_VISIBLE_DEVICES=0`:

```
OUT=$HOME/rebot_sim2real_render_20261102 PY=$HOME/rebot_eval/.venv/bin/python REBOT_ISAAC_WS=$HOME/rebot_eval \
LEISAAC_ASSETS_ROOT=$HOME/rebot_eval/assets DETECTOR=$HOME/rebot_eval/det_cube_obb_yolo26m_ft3.pt SEED=0 \
timeout 2400 bash scripts/eval/project2_eval.sh R
```

The trainer log of the run confirms the robot profile (OURS), single cube, the YOLO-OBB detector, and both checkpoints
loaded (search step 8,038,400; approach step 3,712,000). The run finished on its own (exit 0), and no process was left
on the GPU afterwards. Result of the 4 episodes (`results/project2/render_2026-11-02/`): search 4/4, privileged reach
4/4, deployable gate 0/4, lift not run.

| File | MD5 on the node |
|---|---|
| `ep00_case0_reach_privileged.mp4` | `c89ae5fea95b033cbf93f27cd4e3a972` |
| `ep01_case1_reach_privileged.mp4` | `2f594f3b7d375f54cceee149bef6d42b` |
| `ep02_case2_reach_privileged.mp4` | `3b0815ca2c9cc0128242bfe8289455b1` |
| `ep03_case3_reach_privileged.mp4` | `41a028fed00d9887b684a3a7a1d1736c` |
| `episodes.csv` / `summary.json` / `trainer_summary.json` | `ad6bf5c3…` / `6d9451da…` / `e73931f7…` |

The outputs stay on the node in `~/rebot_sim2real_render_20261102/`; copies are in this repository.

## 14. Seeded re-evaluation (2026-11-02)

Approved by the author as a one-off exception for the second workstation (RTX 5080, 16 GB; another process held
3.0 GB at the start, read-only check). One Ray job, `rebot_p2_reeval_20261102_2134`, was pinned to that node with
1 GPU and `CUDA_VISIBLE_DEVICES=0`:

```
OUT=$HOME/rebot_sim2real_reeval_20261102 PY=$HOME/rebot_eval/.venv/bin/python REBOT_ISAAC_WS=$HOME/rebot_eval \
LEISAAC_ASSETS_ROOT=$HOME/rebot_eval/assets DETECTOR=$HOME/rebot_eval/det_cube_obb_yolo26m_ft3.pt SEED=0 \
timeout 21600 bash scripts/eval/project2_eval.sh A B C
```

- All three runs finished, and the evaluation script exited 0 (21:34 → 21:57).
- Ray reports the job as FAILED. The cause is a listing command appended after the script, which had a quoting error
  (`cat …//summary.json`). The evaluation itself is not affected.
- The trainer logs confirm the OURS profile and both checkpoints loaded (search step 8,038,400; approach step 3,712,000).
- No trainer process was left on the node afterwards.
- Outputs stay on the node in `~/rebot_sim2real_reeval_20261102/`. Copies are in `results/project2/reeval/`, with
  home and Ray working-directory paths redacted in the `log.txt` and summary files.

| File (on the node, before redaction) | MD5 |
|---|---|
| `bench86/episodes.csv`, `summary.json`, `trainer_summary.json`, `log.txt` | `47fb2c6e…`, `fdd7b627…`, `2729322d…`, `706493e4…` |
| `grid400/…` | `77a7dc6a…`, `6e746c5d…`, `4fb8cb90…`, `e7283f49…` |
| `staged_e2e/…` | `bd3fcd9e…`, `5e624675…`, `d59c1199…`, `4acd843c…` |
