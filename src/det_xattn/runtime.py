"""Shared mutable runtime state for the modularized DET-xattn SAC trainer.

`RT` is a single namespace object populated by the entry file (train_sac_det_xattn.py)
AFTER the sim boots: live handles (env / robot / arm_term / ik), derived joint/body
indices, the selected RobotProfile and its profile-derived tensors, plus the
one-element list-wrapped mutables that carry per-step state ACROSS the module-level
`get_obs` (obs.py) and the in-loop closures (actions.py / reward assembly / anneal).

Helpers do `from runtime import RT` and read `RT.env`, `RT.r`, `RT.joint_ids`,
`RT.PH_CUR`, etc. This GENERALIZES the already-working list-global pattern (formerly the
_PH_CUR / _ACQ / _PRIV_W / ... module globals in the monolith, L316-327) rather than
threading a state object through every hot-loop signature — the lowest-risk relocation
for code that must reproduce the baseline ~29% RSR byte-for-byte.

Convention: LIVE tensors / handles live on RT (namespaced, no bare duplicates); scalar
config lives in env_knobs (distinct identifiers). Never keep two globals with the same
name across modules — a closure name that used to bind a live loop tensor must not
silently resolve to a stale module global.
"""


class _RT:
    # ---- live handles (set by entry after ManagerBasedRLEnv build) ----
    env = None          # ManagerBasedRLEnv
    r = None            # env.scene["robot"] articulation
    arm_term = None     # env.action_manager.get_term("arm_action") (delta accumulates on _target_qpos)
    ik = None           # DifferentialIKController (pose)
    N = None            # num_envs
    DEVICE = None       # cuda:<ISAAC_GPU>
    args = None         # tp_config.args (target_cube, dr, save_dir, ...)

    # ---- robot topology (derived from the live articulation + profile) ----
    profile = None          # selected RobotProfile (robot_profile.OURS / OFFICIAL)
    joint_ids = None        # [r.joint_names.index(j) for j in profile.arm_joint_names]
    ee_name = None          # first profile.ee_link_candidates present in r.body_names
    ee_body_idx = None
    ee_jacobi_idx = None    # ee_body_idx + profile.jacobi_offset

    # ---- profile-derived tensors (set by entry once DEVICE is known) ----
    jaw_mouth = None    # torch (1,3) NORMALIZED gripper head/nose dir (was _HEAD; profile.jaw_mouth_offset)
    rest_qpos = None    # torch (6,) d2r reference for proprio_state (was REST_QPOS, profile.rest_qpos_d2r)
    zaxis = None        # torch (1,3) [0,0,1] gripper local +z / world-up (was _ZAXIS)

    # ---- per-step mutables (were the L316-327 one-element list globals) ----
    ENV_CELL = [None]            # per-env current TRAINING spawn cell (curric grid) for region-selective privilege
    CELL_KEEPPRIV = [None]       # per-cell bool: keep privilege in this (failing) cell
    PRIV_W = [1.0]               # current privilege weight (1=full priv, 0=full real)
    PRIV_ASTART = [-1]           # step the anneal started (-1 = RSR target not yet reached)
    SWEEPING = [False]           # mirror of _sweeping for get_obs (sweeps always use the REAL detector)
    GSTEP = [0]                  # current global env-step (gates PRIV_UNTIL)
    PH_CUR = [None]              # per-env phase tensor (0=SEARCH,1=APPROACH,2=ALIGN); set right before get_obs
    ACQ = {"pcx": None, "pcy": None, "pv": None}   # front-guided acquisition stash: front cube projected into wrist img


RT = _RT()
