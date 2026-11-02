# archive/residual_pick_2026-06/ — the author's files from the residual-RL pick project (record only)

A later project on the same robot (2026-06-15 → 06-25): a state-based SAC pick policy in a stand-alone Isaac Sim scene,
keypoint and wrist-camera observations, a residual RL policy on top of a scripted IK reach ("v6"), cube detectors,
camera calibration and a "deep-reach" / "robust-hybrid" real-robot deploy with a debugging dashboard.

That repository has mixed authorship. **Only files whose every line is by the author (`git blame`) are here**:

* `snapshot_2026-06-20/` — author-only files at commit `b4c128e` (2026-06-20), the last commit before a colleague's
  refactors;
* `additions_2026-06-21_to_25/` — files that are author-only at the final commit `65778d3` (2026-06-25) and did not
  exist on 06-20 (residual-RL configs, the deep-reach debugger, grip calibration, evaluation sweeps, handoff notes
  `docs/08`–`12`).

**This code does not run on its own.** The scene builder, the constants module, the robot description, the package
metadata and, for the 06-21→25 files, the RL core (`rl/base_controller.py`, `rl/resfit.py`, `rl/nstep_buffer.py`, …)
contain lines by the colleague and are excluded; the two folders also come from different commits. It is kept as a
record of what the author built; the tests only check that the Python files parse. Inclusion rule, file lists and the
excluded files: [docs/provenance.md §6](../../docs/provenance.md#6-source-c--rebot_sim2real-residual-rl-pick-and-hybrid-deploy-selected-by-git-blame).
Results stated in these notes (e.g. a simulated pick rate, a real-robot reach → grasp) are discussed in
[docs/claims.md](../../docs/claims.md); none is backed by a log in this repository.

Text was translated from Korean; cluster addresses and host / user names were replaced by placeholders.
