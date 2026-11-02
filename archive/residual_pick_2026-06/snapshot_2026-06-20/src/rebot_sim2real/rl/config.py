"""Config dataclasses for the state-based SAC pick training (v1).

Single source of truth for env / reward / SAC / train hyperparameters. Defaults match
the frame & motion constants in ``constants.py`` and the design in docs/01-03. A YAML
file (configs/train.yaml) may override any field via :func:`load_train_config`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

from ..constants import (
    BASE_POS, BASE_QUAT_WXYZ, REST_ARM,
)


@dataclass
class EnvConfig:
    # ---- scene / parallelism ----
    num_envs: int = 64
    env_spacing: float = 2.0                 # grid spacing between cloned envs (m)
    device: str = "cuda:0"
    robot_usd: str = "assets/rebot_b601_dm.usd"
    robot_prim_name: str = "Robot"
    cube_prim_name: str = "Cube"

    # ---- frame (from constants / reference alignment) ----
    base_pos: tuple = BASE_POS               # (0.35, -0.54, 0.01) -- moved +0.10 in y toward the work area
    base_quat_wxyz: tuple = BASE_QUAT_WXYZ   # +90 deg yaw
    rest_arm: tuple = tuple(REST_ARM)
    table_z: float = 0.04                    # table top
    # real rig table = 80 (x, WIDTH) x 120 (y, robot<->camera DEPTH) cm. Depth (120) > width (80):
    # the long axis runs from the base out to the top-cam (measured base<->cam = 120 cm along +Y).
    table_size: tuple = (0.80, 1.20, 0.02)
    # the table SLAB is placed by table_prim_center (DECOUPLED from the cube-spawn table_center). The
    # slab spans y in [-0.60, +0.60]; the base moved to -0.54 so it now sits ~6cm onto the near edge
    # (robot mounted on the table, as in reality). Footprint doesn't affect the task (table rule =
    # table_z height; cube fall = workspace bounds), only the scene geometry/viz.
    table_prim_center: tuple = (0.35, 0.0)   # slab center: y in [-0.60, +0.60]
    # CUBE SPAWN center. Moved +0.10 (was -0.287) WITH the base (-0.64->-0.54) so the relative base->
    # spawn geometry (and thus rest_arm/reach/tuning) is preserved, while the absolute spawn (-0.187)
    # now matches where the real cube actually sits (~-0.13..-0.20, was outside the old training band).
    table_center: tuple = (0.35, -0.187)     # ~28-deg gripper's JAW landing point, base-relative kept
    cube_size: float = 0.038
    cube_mass: float = 0.05                  # 50 g (DR curriculum 50->300 g later)
    workspace_x: tuple = (0.15, 0.55)
    workspace_y: tuple = (-0.30, -0.10)      # moved +0.10 with the base (was [-0.40,-0.20]); same base-relative reach

    # ---- robot dofs ----
    # arm joints addressed by name (robust to dof ordering); gripper joints matched by substring.
    arm_joint_names: tuple = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
    gripper_joint_substr: str = "gripper"    # gripper_joint1 / gripper_joint2 (mirrored prismatic)
    tcp_tip_drop: float = 0.02               # finger-link midpoint -> fingertip/grip-contact level (down)
    wrist_link_name: str = "gripper_link"

    # ---- control (delta-on-target, hard-limited) ----
    physics_dt: float = 1.0 / 120.0
    decimation: int = 8                      # 120/8 = 15 Hz policy
    arm_step: float = 0.05                   # rad / policy step (reference hard limit)
    grip_step: float = 0.0143                # m / policy step (reference hard limit)
    # ANTI-BOB: low-pass the applied ARM action (a_t <- ema*a_{t-1} + (1-ema)*a_t). 0 = off. Smooths the
    # commanded joint deltas so the policy can't oscillate at the loop rate; the policy trains under it.
    action_ema: float = 0.0
    # ---- arm SOFT working range (optional; tighter than the URDF hard limits) ----
    # Cap exploration so random/early actions can't drive joints to extreme poses. Explicit
    # per-joint lo/hi (rad), intersected with the URDF hard limits; rest_arm is always kept
    # inside. None = URDF only.
    arm_limit_lo: Optional[tuple] = None     # explicit per-joint lower (rad)
    arm_limit_hi: Optional[tuple] = None     # explicit per-joint upper (rad)
    # grip qpos convention (scripts/probe_gap.py): qpos UP = finger gap WIDER (gap ~= 2*qpos;
    # qpos 0.0=closed, 0.0715=142mm open). So OPEN = high qpos, CLOSE = low qpos. The step control
    # is inverted accordingly (a_grip=+1 CLOSES = decreases qpos toward grip_close_qpos).
    grip_open_qpos: float = 0.03             # gap ~60mm: straddles the 38mm cube with margin
    grip_close_qpos: float = 0.01            # pinch floor (~20mm): cube blocks fingers at ~19mm -> grip force
    # gripper GATE: the gripper may only CLOSE when the TCP is within this 3-D distance of the cube;
    # farther away the close command is overridden to OPEN. Enforces "approach OPEN, close only when
    # close enough" (else the policy closes mid-approach and arrives already shut -> can't grasp).
    grip_gate_dist: float = 0.06
    drive_stiffness: float = 1.0e5           # gripper + default DOF drive (firm grasp)
    drive_damping: float = 1.0e3
    # A 1e5-stiff arm blasted the fingers through the table collider. A SOFTER arm drive yields at
    # contact (table physically stops it). (An effort cap was worse: it also limits the torque that
    # holds the arm UP against gravity, so the down-reaching grasp pose sagged further DOWN.) More
    # PhysX solver iterations sharpen the contact. Gripper keeps the stiff 1e5 drive for a firm grasp.
    arm_drive_stiffness: float = 1.0e5       # strong enough to DRIVE the gripper down to the cube (5e4 sagged short)
    arm_drive_damping: float = 2000.0
    # wrist (J4-J6) firm so it HOLDS orientation -> stable tcp. The USD default wrist max_effort is
    # only 7 N*m, so the wrist sagged under gravity and swung tcp_z by +/-15cm (the grasp pose was
    # effectively unreachable). Firm stiffness + a RAISED max_effort fixes the tcp instability.
    wrist_drive_stiffness: float = 1.0e5
    wrist_drive_damping: float = 2000.0
    wrist_max_effort: float = 40.0           # RAISE from USD 7 N*m (a raise, NOT a cap) so J4-6 hold
    solver_pos_iters: int = 16               # PhysX position iterations (default ~4) -> less tunneling
    # table rule on the grasp-center (tcp). A valid grasp of a cube ON the table sits at ~cube
    # center (0.059 > table 0.04); the grasp-center going BELOW the table means the fingers are
    # through it. So: continuous penalty once tcp dips below the table top, and HARD-TERMINATE the
    # instant tcp < table_z (grasp-center below table) -> a sub-table scene never persists.
    finger_tip_drop: float = 0.0             # tip_z = tcp_z - this (0 -> table rule on the grasp-center)
    table_term_depth: float = 0.03           # terminate when tcp_z < table_z - this (=0.01); loose so a
    #                                          table-level grasp (gripper must reach the cube on the table) survives
    # GENUINE grasp gate (not mere 3-D proximity, which allowed tcp BELOW the cube + flinging):
    # require the cube horizontally aligned AND the grasp-center at the cube's height (not below),
    # the gripper CLOSED, and the lift SUSTAINED for success_hold_steps (a fling can't sustain).
    grasp_xy: float = 0.035                  # max horizontal tcp<->cube distance for a grasp
    grasp_dz: float = 0.03                   # max |tcp_z - cube_z| (keeps tcp at the cube, never under it)
    # GEOMETRIC grasp: fingers closed ONTO the cube. When gripping the 38mm cube the LINK-origin
    # finger gap (2*grip_qpos) stalls at ~46mm (blade gap != link gap; ~+10mm offset), NOT at
    # cube_size. Full-close is ~20mm, open ~60mm -> a band around grip_grasp_gap isolates "cube held".
    grip_grasp_gap: float = 0.046            # link-origin finger gap when gripping the cube (probe_descent.py)
    grip_gap_tol: float = 0.016              # |finger_gap - grip_grasp_gap| <= this counts as gripping the cube
    # VERTICALITY gate: keep the gripper square-ish (is_grasped REQUIRES tilt < this), so the old
    # ~41-deg lean that caught the cube's top EDGE no longer counts. We use a ~28-deg compromise
    # trajectory (probe_vertical/probe_graspxy): GRASP ~25, APPROACH ~27, LIFT ~30 -> gate 35 admits
    # the whole motion while still rejecting the 41-deg edge grasp. (Pure-vertical ~16 deg plunges
    # straight onto the cube and knocks it; ~28 deg slides down the side and grasps cleanly = 100%.)
    tilt_gate_deg: float = 42.0    # 42 not 35: GRASP forms ~28 deg but the wrist tilts to ~35 to LIFT (no vertical lift pose); 35 clipped genuine 0.38m lifts. corner grasp still rejected by finger-gap; grasp kept square by the (pre-grasp) verticality reward.
    success_hold_steps: int = 10             # consecutive steps the grasped cube must stay >= success height
    # SAFETY (no backward tip-over): the gripper must stay on the +y (table) side, never swing -y
    # behind the base (base_y=-0.54; arm straightening/folding sends tcp_y to -0.4..-1.0). A joint
    # box can't enforce this (kinematic coupling), so it is a TASK-SPACE limit on tcp_y: below
    # y_forward_limit -> penalty (push forward); below y_back_term -> end episode. These two MOVE WITH
    # the base (base moved +0.10 to -0.54 -> both shifted +0.10): y_back_term=base_y, y_forward_limit
    # just in front of it. Reaching back toward the base is fine; only going past it (tip-over) is bad.
    y_forward_limit: float = -0.50           # small penalty gradient just in front of the base (was -0.60)
    y_back_term: float = -0.54               # = base_y: terminate if the TCP goes behind the base (was -0.64)
    grip_friction: float = 1.5               # static/dynamic friction on cube + gripper fingers (firm grasp)

    # ---- episode / task ----
    episode_len: int = 150                   # 10 s @ 15 Hz
    success_lift_height: float = 0.12        # cube z above table_z to count as success
    grasp_radius: float = 0.05               # tcp<->cube dist to consider "holding"
    # APPROACH ROBUSTNESS: jitter the arm's RESET pose by +/- this (rad, per arm joint) around rest_arm.
    # 0 = always the exact pre-grasp hover (policy only learns grasp+lift). >0 = the gripper starts OFF
    # the cube, so the policy must learn to APPROACH-correct (not assume a perfect pre-position).
    rest_rand: float = 0.0
    pos_rand_xy: float = 0.05                # cube spawn half-range (m) around table_center; 0 = fixed center (curriculum)
    yaw_rand: float = 3.14159                # cube yaw randomization (rad)
    # SPAWN CURRICULUM: start at pos_rand_xy, and whenever the rolling success >= spawn_succ_thresh
    # (with a cooldown), widen the spawn half-range by pos_rand_step up to pos_rand_max. Off when
    # pos_rand_max <= pos_rand_xy. Lets one warm-started run generalize from center to full workspace.
    pos_rand_max: float = 0.0                # 0 (<=pos_rand_xy) = curriculum OFF
    pos_rand_step: float = 0.02              # half-range increment per advance
    spawn_succ_thresh: float = 0.85          # rolling success needed to advance
    spawn_cooldown: int = 600                # min policy steps between advances (let the window refill)

    # ---- domain randomization (3) ----
    dr_enable: bool = True
    dr_mass_min: float = 0.05                # cube mass range (kg) — curriculum 50->300 g
    dr_mass_max: float = 0.30
    dr_obs_pos_noise: float = 0.005          # Gaussian std (m) on cube/tcp positions in OBS only
    # ---- sim2real DR (reflecting the real deploy: bobbing under latency, noisy proprio, motor-gain &
    #      grasp-point calibration gaps). All OFF by default (0); enabled via train.yaml. Reward/gates
    #      always use the CLEAN state -- noise/offsets live in the OBS only (asymmetric actor/critic). ----
    dr_obs_qpos_noise: float = 0.0           # Gaussian std (rad) on arm qpos in OBS
    dr_obs_qvel_noise: float = 0.0           # Gaussian std (rad/s) on arm qvel in OBS (real vel = finite diff -> noisy)
    dr_action_latency_max: int = 0           # max policy-steps of action delay; per-env fixed ~U{0..max} (real CAN+control lag)
    dr_action_noise: float = 0.0             # Gaussian std on the normalized action before apply (motor tracking)
    dr_gain_scale: float = 0.0               # per-env arm drive kp scaled by ~U(1-this,1+this) at startup (stiffness gap)
    # tcp/grasp-point calibration robustness: a per-env CONSTANT offset added to the tcp the policy SEES
    # (OBS only), so it can't rely on a perfectly-calibrated grasp point (the deploy needed JAW_FWD/
    # FK_Z_CORR fudges -- training across a band of offsets makes the policy tolerate the real residual).
    dr_tcp_offset_xy: float = 0.0            # half-range (m) of the per-env random tcp xy offset in OBS
    dr_tcp_offset_z: float = 0.0             # half-range (m) of the per-env random tcp z offset in OBS
    # ---- joint TORQUE in OBS (sim2real torque sync) ----
    # Append measured arm joint efforts (n_arm) + gripper effort (1) to the obs (obs_dim += n_arm+1).
    # The real lerobot reports per-joint .torque; the sim uses PhysX measured joint efforts. Their
    # ABSOLUTE scales differ -> normalize by torque_obs_scale AND randomize (dr_torque_scale) so the
    # policy uses torque as a SOFT/relative signal robust to the sim<->real magnitude gap. OFF by
    # default (obs stays 43-D, old ckpts/demos compatible). Enabling -> recollect demos (obs dim changes).
    obs_torque: bool = False
    torque_obs_scale: float = 0.2            # multiply raw effort (N*m) so a ~5 N*m torque -> ~1.0 (same in deploy)
    dr_torque_noise: float = 0.0             # Gaussian std on the (scaled) torque in OBS
    dr_torque_scale: float = 0.0             # per-env multiplicative torque scale jitter ~U(1-this,1+this)
    # ---- WRIST eye-in-hand cam in OBS (D: learned IBVS via a RENDERED detection feature) ----
    # Renders a low-res per-env wrist cam each step, detects the red cube (HSV), feeds the normalized
    # (u, v, visible) to the obs (obs += 3). Sim render + real BOTH run the SAME detector -> sim-real
    # consistent (no analytical-projection parallax). Requires SimulationApp(enable_cameras=True) +
    # render every step -> ~10x slower; use a MODERATE num_envs (~32) + res. OFF by default. Enabling
    # -> recollect demos (obs dim changes). High res is pointless for the low-dim (u,v) feature.
    obs_wrist: bool = False
    wrist_cam_res: tuple = (320, 180)        # low-res wrist render (W,H); enough to locate the cube
    wrist_light_intensity: float = 200.0     # dome-light intensity for the wrist render (too high washes out the red)
    wrist_mount_offset: tuple = (-0.0711, 0.0, -0.0609)   # lens in the gripper_link frame (= SceneConfig)
    wrist_aim_local: tuple = (0.0297, 0.0, -0.02)         # aim TUNED (z -0.0022->-0.02) to the real wrist photo
    #                                                       -> the v4 wrist obs (u,v) now matches the real cam (transfers)
    # wrist vision DR (rendered-detection sim2real robustness)
    dr_wrist_uv_noise: float = 0.0           # normalized (full image = 2.0) Gaussian noise on (u,v)
    dr_wrist_dropout: float = 0.0            # prob the detection is dropped (visible forced 0)
    # WRIST CLUTTER DR: the real wrist view has BACKGROUND RED (curtains/objects) that sim never had
    # ("cube=only red" assumption) -> the SAME detector locks onto it -> wrong (u,v) -> the policy bobbed.
    # Paint random red distractor blobs into the rendered wrist frame BEFORE detection, so the detector
    # sometimes returns a wrong/competing feature -> the policy learns to NOT over-trust wrist (+ fall back
    # to state via dr_wrist_dropout). Preserves the "sim+real run the SAME detector" invariant.
    dr_wrist_distractor: float = 0.0         # prob (per env, per step) of injecting distractor red blobs
    dr_wrist_distractor_n: int = 2           # max number of distractor blobs when injected
    dr_render_light: float = 0.0             # +/- fraction randomization of the dome-light intensity (per reset)
    dr_cube_shade: float = 0.0               # +/- fraction jitter on the cube's red shade (HSV robustness)
    # ---- TOP overhead cam (u,v) in OBS (project (b): keypoint bottleneck, drops privileged cube_pos) ----
    # When True the obs DROPS the privileged cube_pos3 + cube_quat4 + (cube-tcp)3 and instead carries the
    # cube's normalized TOP-cam pixel (u, v, visible). Like the wrist, this is RENDERED + HSV-detected per
    # env (NOT analytic): the render's red-centroid is the SAME quantity the deploy CubeDetectorIO (sim-
    # trained CNN) reads off the real cam, AND it is occlusion-aware (arm over the cube -> visible 0) --
    # an analytic projection matched neither (systematic visible-centroid bias + no occlusion). The cam
    # pose MUST equal the real-aligned SceneConfig top_cam_*. DR (noise + dropout) covers detector jitter
    # and missed/occluded frames. Enabling -> obs dim changes -> recollect demos. (Combine w/ obs_wrist for v5.)
    obs_top_uv: bool = False
    obs_analytic_uv: bool = False            # SPEED: top/wrist (u,v) from ANALYTIC projection (no camera render) instead of HSV-on-render -> ~10x fps (renders are the v5 bottleneck). DR (noise+dropout+occlusion proxy) matches render; deploy still uses the CNN (analytic≈render≈CNN per the consistency gate). Skips the per-env cameras entirely.
    top_occlusion_radius: float = 0.0        # analytic-obs CORRELATED occlusion: top (u,v) visible->0 when the gripper is over+above the cube (xy within this radius), matching the rendered/deploy arm-occludes-overhead. 0=off. Closes the analytic-vs-render transfer gap (clean-overfit) the cross-eval exposed.
    top_cam_res: tuple = (320, 180)          # overhead render res (W,H); enough to locate the cube
    top_cam_height: float = 0.57             # lens height above the table top (MEASURED; = SceneConfig)
    top_cam_y: float = 0.56                  # cam ABSOLUTE y (MEASURED 0.56 = SceneConfig top_cam_pos.y = -0.30+0.86); decoupled from the cube-spawn table_center(-0.187) like top_cam_aim_y, else the training cam drifts +0.113 off the deploy cam
    top_cam_aim_y: float = -0.10             # aim y (env), TUNED so the sim top view matches real (= SceneConfig)
    dr_top_uv_noise: float = 0.0             # normalized (full image = 2.0) Gaussian noise on (u,v) — detector jitter
    dr_top_uv_dropout: float = 0.0           # prob the top detection is dropped (visible 0) — arm occlusion / miss


@dataclass
class RewardConfig:
    # staged shaping (kept small/bounded — docs warn against dense sim-hacking & unbounded penalties)
    w_reach: float = 1.0          # -tanh(dist tcp->cube)
    w_xy_align: float = 0.5       # -horizontal tcp<->cube xy distance
    w_grasp: float = 2.0          # bonus when holding
    w_lift: float = 8.0           # cube height gain while held
    success_bonus: float = 20.0   # terminal on success
    w_action: float = 0.01        # action smoothness penalty (||a-a_prev||^2)
    w_time: float = 0.001         # per-step time penalty
    fall_penalty: float = 5.0     # cube dropped off table
    w_table: float = 20.0         # SAFETY: penalty per meter the gripper goes below the table top
    w_yback: float = 20.0         # SAFETY: penalty per meter the gripper goes backward past y_forward_limit
    w_vertical: float = 3.0       # SHAPING: keep the gripper near-vertical (1 - tilt/90), gated by proximity to the cube (kills the ~41-deg edge grasp)
    w_wrist_center: float = 0.0   # SHAPING: AIM the wrist cam at the cube while approaching (continuous center, gated by visible+prox+pre-grasp) -> wrist-driven servo. Default 0 = OFF (preserves v4/other configs); v5 sets it (1.0). Keep modest: r_reach+r_vertical+r_wrist must NOT dwarf the grasp/lift reward (hover-harvest).
    # ANTI-BOB / commit-and-hold (reflecting the real deploy: the policy bobbed away before the grip
    # torque could build). Reward LOW arm speed while near+closing on the cube -> the policy learns to
    # SETTLE and hold the grasp (matches the real commit-and-hold latch) instead of oscillating.
    w_hold_still: float = 0.0     # 0 = off; >0 rewards (near & closing) * (1 - tanh(arm_speed))


@dataclass
class SACConfig:
    hidden: tuple = (256, 256)
    gamma: float = 0.95
    tau: float = 0.01
    lr: float = 3.0e-4
    batch_size: int = 1024
    replay_size: int = 1_000_000
    warmup_steps: int = 2000      # env-steps of random actions before learning
    updates_per_step: int = 1     # gradient updates per env-step (UTD); raise for sample-eff.
    target_entropy_scale: float = 1.0   # target_entropy = -act_dim * scale
    layernorm: bool = True
    init_alpha: float = 0.2
    min_alpha: float = 0.02         # floor on entropy temperature -> sustained exploration (helps grasp)
    # RLPD-style demo bootstrapping (optional): mix scripted-expert demos into every minibatch.
    demo_path: Optional[str] = None  # .pt from scripts/collect_demos.py; None = pure online SAC
    demo_frac: float = 0.5           # fraction of each batch drawn from demos (RLPD = 0.5)


@dataclass
class TrainConfig:
    exp_name: str = "rebot_sac_pick"
    total_env_steps: int = 2_000_000   # counted per-env (so N*this transitions)
    seed: int = 0
    log_dir: str = "runs"
    ckpt_dir: str = "ckpt"
    log_every: int = 1                 # policy-steps between scalar logs
    eval_every: int = 20000            # policy-steps between deterministic evals
    eval_episodes: int = 16
    eval_seed: int = 12345             # fixed RNG so eval episodes are reproducible across checkpoints
    eval_holdout: Optional[dict] = None  # env-cfg overrides for an OOD validation eval (unseen mass/poses)
    ckpt_every: int = 50000            # policy-steps between NUMBERED snapshots (<exp>_<envsteps>.pt)
    latest_every: int = 2000           # policy-steps between overwriting <exp>_latest.pt (monitor polls this)
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    sac: SACConfig = field(default_factory=SACConfig)


def _overlay(dc, data: dict):
    """Recursively overlay a dict onto a dataclass instance (in place)."""
    for k, v in (data or {}).items():
        if not hasattr(dc, k):
            print(f"[config] WARN unknown key '{k}' ignored")
            continue
        cur = getattr(dc, k)
        if hasattr(cur, "__dataclass_fields__") and isinstance(v, dict):
            _overlay(cur, v)
        elif isinstance(cur, tuple) and isinstance(v, list):
            setattr(dc, k, tuple(v))
        else:
            setattr(dc, k, v)
    return dc


def load_train_config(yaml_path: Optional[str] = None) -> TrainConfig:
    cfg = TrainConfig()
    if yaml_path:
        import os
        if os.path.exists(yaml_path):
            import yaml
            with open(yaml_path) as f:
                _overlay(cfg, yaml.safe_load(f) or {})
        else:
            print(f"[config] {yaml_path} not found — using defaults")
    return cfg


def config_to_dict(cfg: TrainConfig) -> dict:
    return asdict(cfg)
