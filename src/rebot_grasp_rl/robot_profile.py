"""Robot abstraction so the SAME trainer runs on our reBot/SO101 asset (OURS) or the
official Seeed reBot-Isaacsim USD (OFFICIAL).

Selection: ROBOT_OFFICIAL=1 -> OFFICIAL, else OURS (read once in the entry file).
Target: 8951 stays OURS; 8953 trains on OFFICIAL.

Every hardcoded joint/link/index/geometry constant that used to be inline in the
monolith is captured here as a profile field. The entry file reads the selected profile
and stuffs the derived tensors/indices onto runtime.RT; helpers read RT.

IMPORTANT — the official USD is NOT a drop-in (verified from
reBot-Isaacsim/usd/RS-rebot-dev-arm/payloads/Physics/physics.usda):
  * EE link is `gripper_end` (not `gripper_link`).
  * gripper joints joint_left/joint_right are PRISMATIC (metres), not revolute.
  * joint2/joint3 live entirely in the negative range (rest -90deg / -1deg) — a
    different sign/offset convention than SO101; a direct joint-value copy is wrong.
  * self-collision is OFF in the asset (ON diverges to NaN in the home pose).
  * NO camera prims exist — wrist/front cams must be added (leisaac side).
Because 8953 trains FRESH on OFFICIAL, the convention differences are fine (the policy
learns whatever the sim gives) — but the geometry constants marked `# TODO(official)`
below MUST be measured on the loaded USD at the Stage-5 boot check, not guessed.
"""
from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass(frozen=True)
class RobotProfile:
    name: str
    # --- joints ---
    arm_joint_names: tuple          # ORDERED == the old ARM_JOINTS list (index order is load-bearing)
    idx: dict                       # positional index into arm_joint_names: PAN/WFLEX/ELBOW/WROLL
    joint_limits_sim: dict          # per-joint (lo,hi) radians in the SIM joint frame (old REAL_JLIM)
    grip_joint_limit: tuple         # (lo,hi) gripper stroke default (GRIP_LO/GRIP_HI defaults)
    # --- links / prims ---
    ee_link_candidates: tuple       # first present in r.body_names wins (old EE_NAME search)
    finger_link_left: str           # contact-sensor prim (old gripper_left)
    finger_link_right: str          # contact-sensor prim + wrist-cam sibling (old gripper_right)
    jacobi_offset: int              # ee_jacobi_idx = ee_body_idx + this (fixed base == -1)
    robot_prim_regex: str           # scene prim root for cams/contacts
    articulation_cfg_name: str      # leisaac ArticulationCfg to bind (single_arm_env_cfg selects it)
    # --- rest / start poses (ordered like arm_joint_names) ---
    rest_qpos_d2r: tuple            # d2r reference used by proprio_state (old REST_QPOS, L245)
    high_start: dict                # per-joint HIGH-START target (default before START_* env overrides)
    wrist_boot_cap: float           # USD build-time wrist limit; init must stay within it (old WRIST_BOOT_CAP)
    # --- gripper "head/nose" + wrist-cam geometry (EE-link frame) ---
    jaw_mouth_offset: tuple         # EE-link -> jaw mouth, un-normalized (old _HEAD, L257)
    # --- USD PRIM SUBPATHS (Phase 2b-iii): full prim subpath UNDER {ENV_REGEX_NS}/Robot for the EE link and the
    #     base link. OURS/SO101 is FLAT ("gripper_link"/"base_link"); the SimReady OFFICIAL USD nests links deeply.
    #     Used to build wrist-cam / ee_frame / contact-sensor prim_paths so cameras+FrameTransformer+ContactSensor
    #     resolve. Defaults reproduce the OURS FLAT paths EXACTLY (OURS-safe even if a profile omits them). ─────
    ee_prim_subpath: str = "gripper_link"     # e.g. OFFICIAL "Geometry/base_link/link1/.../link6/gripper_end"
    base_prim_subpath: str = "base_link"      # ee_frame source; OFFICIAL "Geometry/base_link"
    wrist_cam_offset_pos: Optional[tuple] = None   # default wrist-cam mount pos; None = use leisaac env-cfg default
    wrist_cam_aim: Optional[tuple] = None


# ------------------------------------------------------------------ OURS ----
# Exact current literals from the monolith. Changing ANY value here changes 8951.
OURS = RobotProfile(
    name="ours",
    arm_joint_names=("shoulder_pan", "shoulder_lift", "elbow_flex",
                     "wrist_flex", "wrist_yaw", "wrist_roll"),
    idx={"PAN": 0, "WFLEX": 3, "ELBOW": 2, "WROLL": 5},
    joint_limits_sim={"shoulder_pan": (-4.187, 1.043), "shoulder_lift": (-3.513, 0.037),
                      "elbow_flex": (-3.878, 0.052), "wrist_flex": (-0.764, 3.126),
                      "wrist_yaw": (-1.777, 1.553), "wrist_roll": (-5.619, 2.001)},
    grip_joint_limit=(0.0, 0.045),
    ee_link_candidates=("gripper_link", "wrist_roll_link", "wrist_link"),
    finger_link_left="gripper_left",
    finger_link_right="gripper_right",
    jacobi_offset=-1,
    robot_prim_regex="{ENV_REGEX_NS}/Robot",
    articulation_cfg_name="SO101_FOLLOWER_CFG",
    rest_qpos_d2r=(-1.5708, -1.30, -1.10, 0.75, 0.0, -0.10),   # L245 REST_QPOS (d2r ref; intentionally the OLD low pose)
    high_start={"shoulder_pan": -1.5708, "shoulder_lift": -1.6684, "elbow_flex": -1.2626,
                "wrist_flex": 1.55, "wrist_yaw": 0.0012, "wrist_roll": -0.0326},   # START_* env-knob defaults
    wrist_boot_cap=1.55,
    jaw_mouth_offset=(-0.0419, 0.0, 0.0),                     # user 2026-07-31: OURS jaw-mouth = OFFICIAL measured value (gripper_end->finger midpoint). The old (-0.086,0,0.103) has ||0.134|| so _dcube bottoms out -> rsr_old barely rises. New value = dcube->0 on a real grasp. nose axis +Z->-X flip (headDown↑) — approach rsr_old 95%. search is also retrained fresh to match.
    wrist_cam_offset_pos=None,                                 # OURS: leisaac cfg default (-0.04,0,0.04); WRIST_OFF env overrides
    wrist_cam_aim=None,
)

# -------------------------------------------------------------- OFFICIAL ----
# Seeed reBot-Isaacsim USD topology. Limits are the USD deg limits -> radians.
# Geometry constants marked TODO(official) MUST be measured on the loaded USD (Stage-5),
# and high_start re-derived so the cube sits in the wrist FOV at rest.
OFFICIAL = RobotProfile(
    name="official",
    arm_joint_names=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
    idx={"PAN": 0, "WFLEX": 3, "ELBOW": 2, "WROLL": 5},        # joint1=base pan..joint6=roll (verify at boot)
    joint_limits_sim={"joint1": (-2.7999, 2.7999), "joint2": (-3.1400, 0.0),
                      "joint3": (-3.1400, 0.0), "joint4": (-1.7901, 1.6900),
                      "joint5": (-1.5700, 1.5700), "joint6": (-3.1400, 3.1400)},   # deg limits -> rad
    grip_joint_limit=(0.0, 0.05),                             # prismatic joint_left stroke (m); joint_right 0..0.0715
    ee_link_candidates=("gripper_end",),
    finger_link_left="gripper_left",                          # same finger link names as OURS
    finger_link_right="gripper_right",
    jacobi_offset=-1,                                         # fixed base; VERIFY at boot (root_joint is a fixed joint)
    robot_prim_regex="{ENV_REGEX_NS}/Robot",
    articulation_cfg_name="REBOT_OFFICIAL_CFG",
    rest_qpos_d2r=(0.0, -1.5708, -0.01745, 0.0, 0.0, 0.0),    # USD targetPositions (j2=-90deg, j3=-1deg); TODO(official) confirm d2r ref
    high_start={"joint1": 1.6, "joint2": -0.084, "joint3": -2.912,
                "joint4": 1.244, "joint5": 0.0, "joint6": 0.0},  # Phase 2b-i (2026-07-31): RE-SOLVED on SimReady USD — settled nose 4.15° from straight-down (top-down ✓), gripper reach 0.477m height 0.438m. OLD CALIB pose {-1.571,-1.302,-1.693,1.063,0,0} gave nose 46° ABOVE horizontal (2a red flag) on this USD. base at 180°-Z (cfg rot). NOTE: straight-down wrist can't go below ~0.44m height w/ safe joint margins at this base height.
    wrist_boot_cap=1.69,                                      # Phase 2a (2026-07-31): = joint4 hi build limit measured on loaded USD (was guess 1.55). joint4 range [-1.79,1.69].
    jaw_mouth_offset=(-0.0419, 0.0, 0.0),                    # CONFIRMED Phase 2a (2026-07-31): measured gripper_end->finger-midpoint local=(-0.04194, +0.00002, 0) OPEN (norm 0.0419); nose/grasp axis = local -X (dot 1.000). closing drifts +5.3mm local +Y (asymmetric jaw joint_right 0.0715>joint_left 0.05) — approach-axis X unaffected. ALL 8 joint limits also matched profile.
    # Phase 2b-iii (2026-08-01): SimReady USD nested prim subpaths, VERIFIED from RS-rebot-dev-arm.usda robotLinks rel
    # (defaultPrim tn__00armrs_asmv3_hJ6D -> Geometry/base_link/link1..6/gripper_end/{gripper_left,gripper_right}).
    # When leisaac spawns the USD's defaultPrim at {ENV_REGEX_NS}/Robot, its Geometry scope composes directly under
    # /Robot, so the full EE subpath is Geometry/base_link/link1/.../link6/gripper_end (vs OURS flat "gripper_link").
    ee_prim_subpath="Geometry/base_link/link1/link2/link3/link4/link5/link6/gripper_end",
    base_prim_subpath="Geometry/base_link",
    wrist_cam_offset_pos=(0.11, 0.0, 0.0),                    # Phase 2b-i (2026-07-31) RENDER-VALIDATED: 11cm back up +X_local (world-up at home pose), clears fingertips; workspace/cube-spawn centered in view. (2a proposal (0.025,0,0.035) put cam inside the jaws.)
    wrist_cam_aim=(-0.955, 0.0, -0.30),                       # Phase 2b-i: down the -X nose, slight tilt toward base; ~85° HFOV, 320x240.
)


def selectProfile():
    """ROBOT_OFFICIAL=1 -> OFFICIAL else OURS. Read once in the entry file."""
    return OFFICIAL if int(os.environ.get("ROBOT_OFFICIAL", "0")) else OURS
