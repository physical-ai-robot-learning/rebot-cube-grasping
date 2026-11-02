# archive/real_robot_july/ — perception + IK + state-machine deploy on the real reBot (record only)

The author's files of the July 2026 real-robot work (second workstation `~/workspace/rebot_sim2real`, HEAD `7f36e69`,
2026-07-24): a deploy package with a perception thread, robot I/O thread, kinematics, a state machine (approach /
follow / align / AUTO-GRASP / lift), a web dashboard, calibration tools (front-camera intrinsics and checkerboard
extrinsics, wrist-camera fit, gripper calibration, FK table-touch check), and the handoff notes `docs/08`–`34`.

**This code does not run on its own.** The IK engine (`rbot_calculator/base_controller.py`, `config.py`),
`constants.py`, `perception.py` and `grasp_traj.py` contain lines by a colleague and are excluded by the `git blame`
rule ([docs/provenance.md §10](../../docs/provenance.md#10-real-robot-code-july-blame-at-7f36e69)).

What the notes record (no trial counts, no video):

| Date | Note | Recorded outcome |
|---|---|---|
| 2026-07-07 | `docs/14` | real-robot full pick (reach → grasp → lift) confirmed to work |
| 2026-07-13 | `docs/20` | residual SAC deploy: perception gate passed, run gate failed (arm rose; observations out of distribution) |
| 2026-07-13/14 | `docs/22` | grasp and lift succeeded after re-verifying the FK height correction |
| 2026-07-23 | `docs/33` | AUTO-GRASP cycle succeeded on a cube at an arbitrary angle after the joint-map loading fix; repeatability sweep still open |
| 2026-07-23 | `docs/34` | RL and simulation code removed from that repository; perception + IK + state machine only |

Text was translated from Korean. The code is unchanged (AST check). Addresses and session-workflow notes were
removed from the notes.
