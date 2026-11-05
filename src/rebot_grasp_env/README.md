# rebot_grasp_env — Project 2 simulation environment (S4)

**Status:** used by every Project 2 run. Needs Isaac Sim 5.1 / Isaac Lab 2.3 and the LeIsaac reBot fork; not
executable in CI (compile-checked; equivalence to the original files checked by `analysis/refactor_equivalence.py`).

**Inputs → outputs.** A parsed LeIsaac `LeIsaac-SO101-LiftCube-v0` environment config → the same config modified for
the reBot: observation terms, delta-on-target joint action with limits and elbow gate, reward and success terms,
motor limits, dynamics randomisation, camera mounts and visual randomisation.

**Entry points:** none (library). `rebot_grasp_rl.train` calls `configureEnv`, `applyMotorLimits`, `addDynamicsDr`
and the `visual_obs` helpers.

**Layout:** `env_config.py` (configuration), `rewards.py` (reward / success terms; was `maniskill_rewards.py`),
`target_delta_action.py` (Isaac Lab action term), `visual_obs.py` (cameras and visual domain randomisation).

**Kept as they are:** all function signatures (Isaac Lab term functions receive their parameters by name from
`params={...}`), action-term attributes and hooks, config attribute names.

**Provenance:** was `src/det_xattn/packages/sim/env_setup/`, the July snapshot of the June environment
(`docs/provenance.md` §9). Reward and success follow Squint's ManiSkill lift task (`docs/third_party.md`).
