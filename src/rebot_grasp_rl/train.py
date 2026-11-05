"""Project 2: DETECTION cross-attention SAC (C51) + IK approach + parameterized rule grasp/lift.

Design (user spec):
  - RL (SAC) learns:  APPROACH (IK Cartesian delta toward cube) + grasp/lift PARAMETERS.
  - Rule executes:    grasp + lift, using the RL-output params (descend_depth, grip_target, lift_height).
  - Obs (all real-measurable, normalized): front/wrist HSV-red detection (cx,cy,area_frac,visible)
    + cube orientation (sin yaw, cos yaw) + proprio(21). NO raw pixels in the policy.
  - Fusion: cross-attention over {front, wrist, orientation} tokens (det_xattn_nets).
  - Action 7D: [dx, dy, dz, dyaw, descend_depth, grip_target, lift_height]  (all tanh [-1,1]).

C51-SAC machinery mirrors train_c51_visual; IK mirrors the validated ikval DifferentialIKController.
"""
import os, io, sys, time, math, threading
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT", os.environ.get("REBOT_ISAAC_WS", os.path.expanduser("~/workspace/rebot_isaac_rl")) + "/leisaac/assets")

from rebot_grasp_rl.config import p, args   # argparse extracted to tp_config.py (parsed at import; must run before AppLauncher)
_IGPU = int(os.environ.get("ISAAC_GPU", "0")); DEVICE = f"cuda:{_IGPU}"   # user 2026-07-14: pin Isaac(physics+render)+tensors to a physical GPU so train+eval CO-LOCATE. NOT CUDA_VISIBLE_DEVICES (Vulkan renderer ignores it -> OOM on GPU0).
if int(os.environ.get("TF32", "1")):   # user 2026-07-28: TF32 matmul/conv on Ada(4090) -> ~1.3-1.5x faster SAC update (dominant at high UTD), negligible RL accuracy impact. TF32=0 to disable.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("[TF32] enabled (matmul+cudnn allow_tf32, cudnn.benchmark)", flush=True)
_COMPILE = int(os.environ.get("COMPILE", "0"))   # user 2026-07-28: 1 = torch.compile the critic (see _mk_agent). Default off.
BUF_DEV = args.buffer_device
TRAIN_VIZ = int(os.environ.get("TRAIN_VIZ", "1"))   # user 2026-07-14: 0 = skip LIVE CLIP/PNG/GIF render in TRAINING (keeps TEXT metrics/frag UI; drops env0 frame-copy + GIF encode). Always on in TP_EVAL.
os.makedirs(args.save_dir, exist_ok=True)

from isaaclab.app import AppLauncher
app = AppLauncher({"headless": args.headless, "enable_cameras": True, "device": DEVICE}).app   # user 2026-07-14: device pins Isaac physics+render GPU (multi-GPU co-locate; replaces CUDA_VISIBLE_DEVICES which Vulkan ignored)
if int(os.environ.get("PATHTRACE", "0")):   # user 2026-07-30: force PathTracing to dodge the Blackwell(RTX 50-series) RTX-RealTime blurry/noisy render bug (IsaacLab #2188/#2869) that tanks detection. 1spp ~ RT-RealTime perf w/o blur. ONLY set on Blackwell eval nodes (.4/.5); leave OFF on 4090(.2) which renders fine.
    import carb as _carb
    _cs = _carb.settings.get_settings()
    _cs.set("/rtx/rendermode", "PathTracing")
    _cs.set("/rtx/pathtracing/spp", 1)
    _cs.set("/rtx/pathtracing/totalSpp", 1)
    _cs.set("/rtx/pathtracing/optixDenoiser/enabled", True)
    print("[PATHTRACE] rendermode=PathTracing spp=1 + OptiX denoiser (Blackwell blurry workaround)", flush=True)
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.managers import SceneEntityCfg, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, quat_apply, quat_mul
import leisaac.tasks.lift_cube  # noqa
from leisaac.tasks.lift_cube import mdp
from rebot_grasp_env.env_config import configure_env, apply_motor_limits, add_dynamics_dr
from rebot_grasp_env import visual_obs
from rebot_grasp_env.visual_obs import (composite_bg, recolor_seg_region, image_dr,
                                               viewpoint_warp, tag_gripper, IMG_RENDER)

from rebot_grasp_rl.nets import DetActor, DetCriticC51, D_DET, D_ORI
from rebot_grasp_rl.target_projection import project_target   # C51 target projection extracted to tp_net.py (pure fn; used by the TWO_PHASE _upd)
from rebot_grasp_rl import robot_profile                 # OURS (reBot/SO101) vs OFFICIAL (Seeed reBot-Isaacsim) robot abstraction
from rebot_grasp_rl.runtime import RT               # shared mutable runtime state (populated after the sim boots)
profile = robot_profile.select_profile()   # ROBOT_OFFICIAL=1 -> OFFICIAL else OURS; read once here
RT.profile = profile
print(f"[ROBOT] profile={profile.name} arm_joints={profile.arm_joint_names} ee_candidates={profile.ee_link_candidates}", flush=True)

# ---------------- env ----------------
N = args.num_envs
ec = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=N)
from rebot_grasp_rl import eval_hooks as _evh; _evh.seed(ec)   # 2026-11-01: opt-in seeded eval output (no-op unless EVAL_OUT_DIR is set)
if profile.name != "ours":   # OFFICIAL: swap in the re-imported PhysX reBot USD (joint1-6, EE gripper_end) BEFORE configure_env/high-start
    from leisaac.assets.robots.lerobot import REBOT_OFFICIAL_CFG
    _off_cfg = REBOT_OFFICIAL_CFG
    _ousd = os.environ.get("OFFICIAL_USD", "")   # user 2026-07-23: override the hardcoded OFFICIAL usd_path (leisaac hardcodes a .2 abs path) so the OFFICIAL model can be eval'd on a node (.4) where that path doesn't exist. default "" = leisaac's path (no-op for the .2 training agent).
    if _ousd:
        _off_cfg = REBOT_OFFICIAL_CFG.replace(spawn=REBOT_OFFICIAL_CFG.spawn.replace(usd_path=_ousd))
        print(f"[ROBOT-OFFICIAL] usd_path OVERRIDE -> {_ousd}", flush=True)
    ec.scene.robot = _off_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    print(f"[ROBOT-OFFICIAL] scene.robot -> REBOT_OFFICIAL_CFG ({ec.scene.robot.spawn.usd_path})", flush=True)
ec = configure_env(ec, {}, mdp, RewTerm, DoneTerm, SceneEntityCfg, target_cube=args.target_cube)
# FOV-SPAWN (user 2026-06-25): restrict the cube spawn to the wrist eye-in-hand FOV at the (high) e110 REST pose so the
# cube is VISIBLE in the wrist view from episode START -> pure approach RL (descend+align), NO acquisition bootstrap and
# NO privileged guide. Box = the e110-calibrated in-frame region (4 corners verified in the wrist view at REST); was the
# wide near-base (0.15,0.48)/(-0.44,-0.16) which sat mostly OUTSIDE the wrist FOV -> wvis~0.08 (the whole acquisition issue).
# user 2026-06-28: FOV restriction REMOVED -> spawn across the (reachable) workspace, not just the wrist FOV. env-tunable.
_SPX_LO = float(os.environ.get("SPAWN_X_LO", "0.15")); _SPX_HI = float(os.environ.get("SPAWN_X_HI", "0.50"))
_SPY_LO = float(os.environ.get("SPAWN_Y_LO", "-0.50")); _SPY_HI = float(os.environ.get("SPAWN_Y_HI", "-0.05"))
ec.events.reset_cubes_mixed.params["x_range"] = (_SPX_LO, _SPX_HI)
ec.events.reset_cubes_mixed.params["y_range"] = (_SPY_LO, _SPY_HI)
# user 2026-07-07: PEAKED spawn-distance distribution — most cubes in the dense band, tails spread over the full range.
#   distance(cm) = (y + 1.15)*100 (base at y=-1.15). So full y[_SPY_LO,_SPY_HI] and dense y[SPAWN_DENSE_LO,HI].
#   On reset, y is resampled = dense band w.p. SPAWN_DENSE_P else full range (env-origin aware). SPAWN_PEAKED=0 -> uniform.
SPAWN_PEAKED   = int(os.environ.get("SPAWN_PEAKED", "0"))
SPAWN_DENSE_LO = float(os.environ.get("SPAWN_DENSE_LO", "-0.85"))   # dense band near = 30cm
SPAWN_DENSE_HI = float(os.environ.get("SPAWN_DENSE_HI", "-0.45"))   # dense band far  = 70cm
SPAWN_DENSE_P  = float(os.environ.get("SPAWN_DENSE_P", "0.65"))     # fraction of resets drawn from the dense band
print(f"[SPAWN] cube spawn -> x{ec.events.reset_cubes_mixed.params['x_range']} "
      f"y{ec.events.reset_cubes_mixed.params['y_range']} (FOV restriction removed)", flush=True)
# HIGH START (user 2026-06-25): raise the REST pose so the cube is in the wrist eye-in-hand FOV from episode START.
# Joints VALIDATED by DUMP_REF (gripper ee_z~0.235-0.29, cube visible in wrist) vs the old low [-1.5708,-1.30,-1.10,0.75,0,-0.10]
# (ee_z~0.096, cube totally out of frame = the acquisition failure). Order = [pan,lift,elbow,wrist_flex,wrist_yaw,wrist_roll].
# user 2026-06-26: start was too head-DOWN. shoulder_pan 90deg (forward), elbow ~30deg UP. pan/elbow env-tunable -> flip START_ELBOW if the arm dives.
_START_PAN   = float(os.environ.get("START_PAN", "-1.5708"))        # 90deg forward (base yaw 180 -> -90deg points the arm at the workspace)
_START_ELBOW = float(os.environ.get("START_ELBOW", "-1.2626"))      # "up" = MORE NEGATIVE (user-confirmed); env-tunable
_START_WRIST = float(os.environ.get("START_WRIST", "1.5500"))       # wrist_flex TRUE start/rest target (sim frame; MORE down = higher). Can EXCEED the USD build limit — applied post-REAL_JLIM to default_joint_pos.
_WRIST_BOOT_CAP = float(os.environ.get("WRIST_BOOT_CAP", "1.55"))   # user 2026-07-01: USD build-time wrist_flex limit is [-1.87,1.57]; init_state MUST be within it or ManagerBasedRLEnv _validate_cfg crashes. Boot at this cap, then default_joint_pos is overridden to _START_WRIST after REAL_JLIM widens the limit to 3.126.
if profile.name == "ours":
    _HIGH_QPOS = [_START_PAN, -1.6684, _START_ELBOW, _START_WRIST, 0.0012, -0.0326]   # [pan,lift,elbow,wrist_flex,wrist_yaw,wrist_roll] — TRUE target (used for _REST_QPOS obs)
    _HIGH = {"shoulder_pan": _HIGH_QPOS[0], "shoulder_lift": _HIGH_QPOS[1], "elbow_flex": _HIGH_QPOS[2],
             "wrist_flex": _HIGH_QPOS[3], "wrist_yaw": _HIGH_QPOS[4], "wrist_roll": _HIGH_QPOS[5]}
else:   # OFFICIAL (or any non-ours profile): HIGH-START keyed by the profile's joint names (TODO(official): re-derive so cube sits in wrist FOV)
    _HIGH = dict(profile.high_start)
    _HIGH_QPOS = [_HIGH[_j] for _j in profile.arm_joint_names]
try:
    _jp = ec.scene.robot.init_state.joint_pos
    for _k in list(_jp.keys()):
        for _nm, _v in _HIGH.items():
            if _nm in _k:
                _jp[_k] = min(_v, _WRIST_BOOT_CAP) if "wrist_flex" in _k else _v   # wrist_flex boot-capped to USD limit; true target applied post-widen
    print(f"[HIGH-START] init_state.joint_pos -> {_jp} (wrist_flex boot-cap {_WRIST_BOOT_CAP}, true target {_START_WRIST} applied post-REAL_JLIM)", flush=True)
except Exception as _e:
    print(f"[HIGH-START] init_state override FAIL: {_e}", flush=True)
import rebot_grasp_env.env_config as _envc
_envc._REST_QPOS = torch.tensor(_HIGH_QPOS)                          # keep dist_to_rest obs consistent with the new start
print(f"[HIGH-START] _REST_QPOS -> {_HIGH_QPOS}", flush=True)
if args.dyn_dr:        # P1-identical dynamics DR (gripper stiffness/damping/friction/mass) BEFORE visual+single_cube
    ec = add_dynamics_dr(ec, SceneEntityCfg, target_cube=args.target_cube)
ec = visual_obs.configure_visual(ec, target_cube=args.target_cube, dr=args.dr, wrist_only=False)
if profile.name != "ours":   # OFFICIAL: EE link is gripper_end, so remap every scene prim_path that hardcodes gripper_link (ee_frame targets, wrist/fp cameras)
    # Phase 2b-iii (2026-08-01) WALL 1: the SimReady USD nests links DEEPLY, so the wrist TiledCamera parent
    # ({ENV_REGEX_NS}/Robot/gripper_link/fp_camera) and the ee_frame FrameTransformer target must resolve to the
    # FULL nested subpath, not a flat leaf. Substitute the leaf "gripper_link" -> profile.ee_prim_subpath (the
    # verified Geometry/base_link/.../gripper_end path) so the camera's parent prim + the FrameTransformer target exist.
    _oldl = "gripper_link"; _newl = profile.ee_prim_subpath   # OFFICIAL: Geometry/base_link/.../gripper_end (was ee_link_candidates[0]=flat gripper_end -> parent prim not found)
    def _fix_prim(_o):
        _pp = getattr(_o, "prim_path", None)
        if isinstance(_pp, str) and _oldl in _pp:
            _o.prim_path = _pp.replace(_oldl, _newl); print(f"[ROBOT-OFFICIAL] prim {_pp} -> {_o.prim_path}", flush=True)
    for _nm in dir(ec.scene):
        if _nm.startswith("__"): continue
        _o = getattr(ec.scene, _nm, None)
        if _o is None: continue
        _fix_prim(_o)
        for _tf in (getattr(_o, "target_frames", None) or []):
            _fix_prim(_tf)
    # WALL 1b: the ee_frame FrameTransformer SOURCE is the flat {ENV_REGEX_NS}/Robot/base_link (leisaac single_arm cfg);
    # on the nested OFFICIAL USD base_link lives under Geometry/. Remap the source to profile.base_prim_subpath so the
    # FrameTransformer initializes (its target frames were already remapped above via gripper_link -> ee_prim_subpath).
    _eef = getattr(ec.scene, "ee_frame", None)
    if _eef is not None and isinstance(getattr(_eef, "prim_path", None), str) and "Robot/base_link" in _eef.prim_path:
        _eef.prim_path = _eef.prim_path.replace("Robot/base_link", "Robot/" + profile.base_prim_subpath)
        print(f"[ROBOT-OFFICIAL] ee_frame source -> {_eef.prim_path}", flush=True)
    # action terms: arm_action(joint1-6) + gripper_action(joint_left/right) instead of the OURS joint names
    if getattr(ec.actions, "arm_action", None) is not None and hasattr(ec.actions.arm_action, "joint_names"):
        ec.actions.arm_action.joint_names = list(profile.arm_joint_names)
        print(f"[ROBOT-OFFICIAL] arm_action.joint_names -> {ec.actions.arm_action.joint_names}", flush=True)
    if getattr(ec.actions, "gripper_action", None) is not None and hasattr(ec.actions.gripper_action, "joint_names"):
        ec.actions.gripper_action.joint_names = ["joint_left"]
        if hasattr(ec.actions.gripper_action, "mirror_joint_names"):
            ec.actions.gripper_action.mirror_joint_names = ["joint_right"]
        print(f"[ROBOT-OFFICIAL] gripper_action -> joint_left/joint_right", flush=True)
    # disable OURS-specific gripper-friction event (references gripper/gripper_joint2 -> unmatched on official)
    for _ev in ("gripper_high_friction",):
        if getattr(ec.events, _ev, None) is not None:
            setattr(ec.events, _ev, None); print(f"[ROBOT-OFFICIAL] disabled event {_ev}", flush=True)
    # env ObservationManager terms hardcode OURS joint/link names (env_config._dist_to_rest_qpos etc.); the
    # trainer uses its OWN get_obs and ignores env obs, so null the robot-coupled terms so the manager builds.
    if getattr(ec.observations, "policy", None) is not None:
        for _ot in ("dist_to_rest", "tcp_pos", "tcp_to_obj", "is_grasped", "contact_force", "target_qpos"):
            if getattr(ec.observations.policy, _ot, None) is not None:
                setattr(ec.observations.policy, _ot, None); print(f"[ROBOT-OFFICIAL] nulled obs term {_ot}", flush=True)
# user 2026-06-28: ZOOM both cams in (focal_length up = narrower FoV = cube bigger -> easier detection of the far/small cube).
#   randomize_camera reads visual_obs.FRONT_FOCAL/WRIST_FOCAL (per-reset ±10% jitter) so set those; also the initial spawn.
_FF = float(os.environ.get("FRONT_FOCAL", "26.0")); _WF = float(os.environ.get("WRIST_FOCAL", "16.0"))
visual_obs.FRONT_FOCAL = _FF; visual_obs.WRIST_FOCAL = _WF
WRIST_ROT90 = int(os.environ.get("WRIST_ROT90", "0"))   # user 2026-07-20: k*90° CCW IMAGE rotation of the wrist render (render_cams) to un-rotate the OFFICIAL wrist cam. Convention-free (the quat-roll didn't affect the Fabric render). OURS: 0.
try:
    ec.scene.front.spawn.focal_length = _FF; ec.scene.wrist.spawn.focal_length = _WF
except Exception as _e: print(f"[ZOOM] spawn focal set failed: {_e}", flush=True)
_WNEAR = float(os.environ.get("WRIST_NEAR", "0"))   # user 2026-07-21: >0 = shrink the wrist cam NEAR clip (m) so it renders CLOSE cubes (<10cm) that the default clip culls (why the biggest cube maxed ~frac0.08 + close cubes vanished). Real C922 sees a few cm. OURS: 0 (unchanged).
if _WNEAR > 0:
    try:
        _crold = getattr(ec.scene.wrist.spawn, "clipping_range", None)
        ec.scene.wrist.spawn.clipping_range = (_WNEAR, 20.0)
        print(f"[WRIST-NEAR] wrist cam near clip {_crold} -> ({_WNEAR}, 20.0) (renders close/big cubes)", flush=True)
    except Exception as _e: print(f"[WRIST-NEAR] set failed: {_e}", flush=True)
print(f"[ZOOM] front focal 18->{_FF}, wrist focal 12->{_WF} (narrower FoV / cube bigger)", flush=True)
# user 2026-06-29: the real C922 is mounted ABOVE THE WRIST (higher + back than the jaw), not at the gripper mouth.
#   Override the wrist-cam offset POS (gripper_link frame: +x=fwd/cubes, +z=DOWN so -z = world UP) and re-aim.
#   env: WRIST_OFF="x,y,z" (pos), WRIST_AIM="x,y,z" (look target). Default keeps the calibrated pose unchanged.
_wox = os.environ.get("WRIST_OFF", "")
if _wox.strip():
    try:
        from isaaclab.sensors import TiledCameraCfg as _TCCw
        from rebot_grasp_env import visual_obs as _vob
        from isaaclab.utils.math import quat_mul as _qmw, quat_from_euler_xyz as _qeu
        _wp = tuple(float(v) for v in _wox.split(","))
        if profile.name == "official":
            # OFFICIAL (user 2026-07-20): PURE-EULER cam orientation (roll,pitch,yaw deg) in the gripper_end LOCAL
            #   frame, opengl convention — EXACT parity with calib_ui_official.py (_eq_quat=quat_from_euler_xyz).
            #   NO lookat, NO WRIST_ROLL180. Values were tuned by the user in that calib UI, so bake them verbatim.
            _wr = float(os.environ.get("WRIST_ROLL_DEG", "0")); _wpd = float(os.environ.get("WRIST_PITCH_DEG", "0")); _wyd = float(os.environ.get("WRIST_YAW_DEG", "0"))
            # user 2026-07-20: AIM(pitch,yaw) then ROLL about the OPTICAL AXIS (local Z) -> roll = pure image rotation, decoupled from aim. Matches calib_ui_official._cam_rot so calib bakes transfer exactly.
            _aimq = _qeu(torch.tensor([0.0]), torch.tensor([math.radians(_wpd)]), torch.tensor([math.radians(_wyd)]))
            _rc, _rs = math.cos(math.radians(_wr)/2.0), math.sin(math.radians(_wr)/2.0)
            _rot = tuple(_qmw(_aimq, torch.tensor([[_rc, 0.0, 0.0, _rs]]))[0].tolist())
            ec.scene.wrist.offset = _TCCw.OffsetCfg(pos=_wp, rot=_rot, convention="opengl")
            print(f"[WRIST-OFF] OFFICIAL pure-euler cam pos={_wp} roll/pitch/yaw=({_wr},{_wpd},{_wyd}) (gripper_end local, opengl; calib_ui parity)", flush=True)
        else:
            _aim = os.environ.get("WRIST_AIM", "")
            _wat = tuple(float(v) for v in _aim.split(",")) if _aim.strip() else _vob.WRIST_AIM_TARGET
            _rot = _vob._lookat_quat_opengl(_wp, _wat)
            if _vob.WRIST_ROLL180:   # re-apply the 180deg optical roll (upright image)
                _rot = tuple(_qmw(torch.tensor([list(_rot)]), torch.tensor([[0.0, 0.0, 0.0, 1.0]]))[0].tolist())
            _wpd2 = float(os.environ.get("WRIST_PITCH_DEG", "0")); _wyd2 = float(os.environ.get("WRIST_YAW_DEG", "0"))   # bake pitch(+down)/yaw ON TOP -> applies in TRAINING too (CALIB-block bake is skipped when WRIST_OFF is set, see below)
            if abs(_wpd2) > 1e-6 or abs(_wyd2) > 1e-6:
                _pp = math.radians(_wpd2); _yy = math.radians(_wyd2)
                _rt = _qmw(torch.tensor([list(_rot)]), torch.tensor([[math.cos(_pp/2), math.sin(_pp/2), 0.0, 0.0]]))
                _rt = _qmw(_rt, torch.tensor([[math.cos(_yy/2), 0.0, math.sin(_yy/2), 0.0]]))
                _rot = tuple(_rt[0].tolist())
            ec.scene.wrist.offset = _TCCw.OffsetCfg(pos=_wp, rot=_rot, convention="opengl")
            print(f"[WRIST-OFF] wrist cam moved to pos={_wp} aim={_wat} (gripper_link frame; -z=up)", flush=True)
    except Exception as _e: print(f"[WRIST-OFF] FAIL: {_e}", flush=True)
# (DELETED: CALIB_DUMP/CALIB_UI-gated SIDE-cam + wrist-pitch-bake block — dev calibration only, never set in training jobs.)

if args.single_cube:   # P1-identical: remove blue/purple distractors -> only the target cube
    for _dc in ("cube_blue", "cube_purple"):
        if hasattr(ec.scene, _dc):
            setattr(ec.scene, _dc, None)
    for _en in list(vars(ec.events)):
        if _en.startswith("_"):
            continue
        _ev = getattr(ec.events, _en, None)
        _p = (getattr(_ev, "params", {}) or {})
        _ac = _p.get("asset_cfg", None)
        if _ac is not None and getattr(_ac, "name", "") in ("cube_blue", "cube_purple"):
            setattr(ec.events, _en, None)
        if "cube_names" in _p:
            _p["cube_names"] = (args.target_cube,)
    print("[DET] single-cube ON (P1-identical: distractors removed)", flush=True)
_CONTACT = int(os.environ.get("CONTACT_SENSORS", "1"))   # user 2026-07-28: 0 = disable finger/cube contact sensors + PhysX contact reporting. Safe now: grasp not trained (GRASP_ON=0), gripper_contact/jaw_contact are never read, contact_force obs is nulled (L140), table penalty is geometric z (_table_penalty) not contact. Cuts PhysX contact cost from env.step.
if _CONTACT:
    _cp = getattr(ec.scene, args.target_cube).prim_path
    # Phase 2b-iii (2026-08-01) WALL 1c: OURS fingers are flat under /Robot; the SimReady OFFICIAL USD nests them
    # under gripper_end (= ee_prim_subpath). Prepend the EE subpath so the ContactSensor prim_path resolves on the
    # nested articulation. OURS ("gripper_link" default subpath is NOT prepended) keeps the exact flat path.
    _fparent = "" if profile.name == "ours" else f"{profile.ee_prim_subpath}/"   # OFFICIAL: Geometry/base_link/.../gripper_end/
    ec.scene.gripper_contact = ContactSensorCfg(prim_path=f"{profile.robot_prim_regex}/{_fparent}{profile.finger_link_left}",
                                                update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
    ec.scene.jaw_contact = ContactSensorCfg(prim_path=f"{profile.robot_prim_regex}/{_fparent}{profile.finger_link_right}",
                                            update_period=0.0, history_length=4, filter_prim_paths_expr=[_cp])
    getattr(ec.scene, args.target_cube).spawn.activate_contact_sensors = True
    ec.scene.robot.spawn.activate_contact_sensors = True
else:
    getattr(ec.scene, args.target_cube).spawn.activate_contact_sensors = False
    ec.scene.robot.spawn.activate_contact_sensors = False
    # env's ObservationManager has contact_force + is_grasped obs terms (env_config.py contact_force_obs / is_grasped_obs) that read scene["gripper_contact"]/["jaw_contact"]; null them so the manager builds without the sensors. Trainer uses its OWN get_obs so harmless (same terms the OFFICIAL branch nulls at L140).
    if getattr(ec.observations, "policy", None) is not None:
        for _ct in ("contact_force", "is_grasped"):
            if getattr(ec.observations.policy, _ct, None) is not None:
                setattr(ec.observations.policy, _ct, None); print(f"[CONTACT] nulled env obs term {_ct}", flush=True)
    print("[CONTACT] contact sensors DISABLED (CONTACT_SENSORS=0)", flush=True)
_ri_old = int(ec.sim.render_interval)   # user 2026-07-28
ec.sim.render_interval = ec.decimation  # user 2026-07-28: render ONCE per policy step. Was render_interval(2) < decimation(10) -> ~5 redundant camera renders/step, folded into env.step time (~48% of iter). Isaac warns + recommends render_interval == decimation. obs/control use only the last frame so no behavior change.
print(f"[RENDER] render_interval {_ri_old} -> {ec.sim.render_interval} (= decimation {ec.decimation})", flush=True)
ec.episode_length_s = float(os.environ.get("EPISODE_S", "20.0"))   # user 2026-06-26: 200 steps @10Hz (was 10.0=100). more time for slow approach.
print(f"[EP-LEN] episode_length_s -> {ec.episode_length_s} (= {int(ec.episode_length_s/(ec.sim.dt*ec.decimation))} steps @10Hz)", flush=True)
# user 2026-07-21: PHASE BUDGETS — SEARCH capped at SEARCH_MAX_S (no handoff by then -> FAIL, new episode); after SEARCH->APPROACH handoff, guarantee APP_GUAR_S of approach then END. 0 = off. Episode must be >= SEARCH_MAX_S + APP_GUAR_S.
_CTRL_HZ = 1.0 / (ec.sim.dt * ec.decimation)   # =10 Hz
SEARCH_MAX_S = float(os.environ.get("SEARCH_MAX_S", "0"))
APP_GUAR_S   = float(os.environ.get("APP_GUAR_S", "0"))
SEARCH_MAX_STEPS = int(round(SEARCH_MAX_S * _CTRL_HZ)); APP_GUAR_STEPS = int(round(APP_GUAR_S * _CTRL_HZ))
print(f"[PHASE-BUDGET] SEARCH_MAX_S={SEARCH_MAX_S}({SEARCH_MAX_STEPS} steps) | APP_GUAR_S={APP_GUAR_S}({APP_GUAR_STEPS} steps) @ {_CTRL_HZ:.0f}Hz", flush=True)
# user 2026-06-28 (B): move arm base to the table INNER (-y) end + front cam to the OPPOSITE (+y) end @0.58m.
#   spawn is shifted with the base (env vars) so the joint-space task is unchanged. NOTE: shifts the robot WORLD base
#   -> deploy fk.py BASE_POS must be updated + FK re-verified before real-robot deploy.
_BASE_Y = float(os.environ.get("BASE_Y", "-1.15"))
try:
    _ow = tuple(ec.scene.robot.init_state.pos)
    if profile.name == "ours":
        ec.scene.robot.init_state.pos = (_ow[0], _BASE_Y, _ow[2])
    else:   # OFFICIAL: base_link IS the root (URDF import, no SO101 internal -1.85 offset) -> place base_link where OURS' base_link lands (world ~0.35,-1.15,0.01)
        _bx = float(os.environ.get("BASE_X", "0.35")); _bz = float(os.environ.get("BASE_Z", "0.01"))
        ec.scene.robot.init_state.pos = (_bx, _BASE_Y, _bz)
    print(f"[BASE-MOVE] robot base {_ow} -> {ec.scene.robot.init_state.pos}", flush=True)
except Exception as _e: print(f"[BASE-MOVE] FAILED: {_e}", flush=True)
try:
    from rebot_grasp_env.visual_obs import _lookat_quat_opengl as _laqF
    _feye = (0.35, float(os.environ.get("FRONT_CAM_Y", "0.0")), 0.58)   # opposite (+y) table end, 58cm
    _ftgt = (0.35, _BASE_Y + 0.30, 0.08)                                # look back toward the arm/cube region
    ec.scene.front.offset.pos = _feye
    ec.scene.front.offset.rot = tuple(_laqF(_feye, _ftgt))
    print(f"[FRONT-CAM] eye={_feye} target={_ftgt}", flush=True)
except Exception as _e: print(f"[FRONT-CAM] FAILED: {_e}", flush=True)
# user 2026-06-28: scale the table to REAL size — x(L-R) 0.70->0.80, y(F-B) 0.65->1.20 (z thickness kept). Small table
#   was letting the cube sit near the edge -> background behind it. env-tunable.
try:
    # user 2026-06-28: x width 0.70->1.60 (extend +0.80 to the LEFT in front-cam view = +x world), y 0.65->1.20
    _tsx = float(os.environ.get("TABLE_SX", str(1.60 / 0.700))); _tsy = float(os.environ.get("TABLE_SY", str(1.20 / 0.650)))
    _cs = getattr(ec.scene.scene.spawn, "scale", None) or (1.0, 1.0, 1.0)
    ec.scene.scene.spawn.scale = (_cs[0] * _tsx, _cs[1] * _tsy, _cs[2])
    print(f"[TABLE-SCALE] scene.spawn.scale -> {ec.scene.scene.spawn.scale} (x1.60 y1.20 m)", flush=True)
    # scale extends about the prim origin (= RIGHT edge x~0) -> already LEFT-only (x[0,0.80]->[0,1.60]). NO shift (was +0.40 = over-shoot -> arm/cube off the right edge).
    _shx = float(os.environ.get("TABLE_SHIFT_X", "0.0"))
    _ip = tuple(ec.scene.scene.init_state.pos); ec.scene.scene.init_state.pos = (_ip[0] + _shx, _ip[1], _ip[2])
    print(f"[TABLE-SHIFT] init pos +x {_shx} -> {ec.scene.scene.init_state.pos}", flush=True)
except Exception as _e:
    print(f"[TABLE-SCALE] FAILED: {_e}", flush=True)
env = ManagerBasedRLEnv(cfg=ec); apply_motor_limits(env, {}); env.reset()
# user 2026-06-28: KILL the table USD's embedded static red cube. leisaac excludes it from SUBASSETS (RigidObject)
#   but the table USD still spawns it VISUALLY + with collision UNDER /Scene -> both policies were diving for that
#   sunken "2nd cube". Make it invisible (detector blind) + disable collision (can't be grasped). Handles instances.
try:
    import omni.usd as _ou
    from pxr import UsdGeom as _UG, UsdPhysics as _UP, Usd as _Usd, Sdf as _Sdf
    _stage = _ou.get_context().get_stage()
    # DIAGNOSTIC: dump env_0/Scene prim tree so we can SEE every prim (find ANY stray red/cube prim, not just name 'cube')
    _s0 = _stage.GetPrimAtPath("/World/envs/env_0/Scene")
    if _s0 and _s0.IsValid():
        _names = [f"{_c.GetName()}({_c.GetTypeName()})" for _c in _Usd.PrimRange(_s0)]
        print(f"[SCENE-DUMP] env_0/Scene ({len(_names)} prims): {_names[:80]}", flush=True)
    # collect first (don't mutate during traversal), then DELETE the embedded cube prims outright
    _paths = [_p.GetPath().pathString for _p in _stage.TraverseAll()
              if "/Scene" in _p.GetPath().pathString and "cube" in _p.GetName().lower()]
    _del = 0
    for _pp in _paths:
        _pr = _stage.GetPrimAtPath(_pp)
        if _pr and _pr.IsValid():
            try: _UG.Imageable(_pr).MakeInvisible()            # belt-and-suspenders: hide first
            except Exception: pass
        if _stage.RemovePrim(_Sdf.Path(_pp)): _del += 1
    print(f"[KILL-CUBE] embedded table cube prims DELETED: {_del}/{len(_paths)} -> {_paths[:4]}", flush=True)
    env.sim.forward()
except Exception as _e:
    import traceback as _tb; print(f"[KILL-CUBE] FAILED: {_e}", flush=True); _tb.print_exc()
# user 2026-06-28: measure current table world size (one-time) to compute the scale toward real 0.80 (x, L-R) x 1.20 (y, F-B) m
try:
    import omni.usd as _ou2
    from pxr import UsdGeom as _UG2, Usd as _U2
    _stage2 = _ou2.get_context().get_stage()
    _tp = _stage2.GetPrimAtPath("/World/envs/env_0/Scene")
    if _tp and _tp.IsValid():
        _rng = _UG2.BBoxCache(_U2.TimeCode.Default(), ["default", "render", "proxy", "guide"]).ComputeWorldBound(_tp).ComputeAlignedRange()
        _sz = _rng.GetSize(); _mn = _rng.GetMin(); _mx = _rng.GetMax()
        print(f"[TABLE-BBOX] size(x,y,z)=({_sz[0]:.3f},{_sz[1]:.3f},{_sz[2]:.3f}) | x[{_mn[0]:.3f},{_mx[0]:.3f}] y[{_mn[1]:.3f},{_mx[1]:.3f}] top_z={_mx[2]:.3f}", flush=True)
except Exception as _e:
    print(f"[TABLE-BBOX] failed: {_e}", flush=True)
r = env.scene["robot"]
arm_term = env.action_manager.get_term("arm_action")    # delta accumulates on _target_qpos (PD tracks it)
if args.dr:
    tag_gripper(env)                                     # separate 'gripper' semantic -> painted BLACK (real rig)
def _rand_bg(n):                                         # per-env far background for greenscreen DR composite
    base = torch.rand(n, 1, 1, 3, device=DEVICE) * 0.6 + 0.2
    grad = torch.linspace(0.85, 1.15, IMG_RENDER, device=DEVICE).view(1, IMG_RENDER, 1, 1)
    return ((base * grad).clamp(0, 1).expand(n, IMG_RENDER, IMG_RENDER, 3) * 255).to(torch.uint8)

ARM_JOINTS = list(profile.arm_joint_names)
joint_ids = [r.joint_names.index(j) for j in ARM_JOINTS]
EE_NAME = next((b for b in profile.ee_link_candidates if b in r.body_names), r.body_names[-1])
ee_body_idx = r.body_names.index(EE_NAME)
ee_jacobi_idx = ee_body_idx + profile.jacobi_offset      # fixed base: jacobian rows offset by profile.jacobi_offset (-1)
print(f"[DET] profile={profile.name} EE={EE_NAME} body={ee_body_idx} jacobi={ee_jacobi_idx} joints={joint_ids}", flush=True)
RT.env = env; RT.r = r; RT.arm_term = arm_term; RT.N = N; RT.DEVICE = DEVICE; RT.args = args   # populate shared runtime for modularized helpers
RT.joint_ids = joint_ids; RT.ee_name = EE_NAME; RT.ee_body_idx = ee_body_idx; RT.ee_jacobi_idx = ee_jacobi_idx
import os as _os   # used by the TWO_PHASE force-exit (_os._exit(0))

ik = DifferentialIKController(
    DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
    num_envs=N, device=DEVICE)                            # POSE: position + orientation (used by TWO_PHASE APPROACH/ALIGN/LIFT IK servo)
RT.ik = ik
if int(os.environ.get("WRISTCAM_PROBE", "0")):   # numeric wrist-cam aim diagnostic (OFFICIAL calib): where does the wrist cam look vs the cube, in gripper_end LOCAL frame?
    _zra = torch.zeros((N, env.action_manager.total_action_dim), device=DEVICE)
    for _ in range(30):
        env.step(_zra)
    _cw = env.scene["wrist"]; _camp = _cw.data.pos_w; _camq = _cw.data.quat_w_ros
    _cube = env.scene[args.target_cube].data.root_pos_w
    _eep = r.data.body_pos_w[:, ee_body_idx]; _eeq = r.data.body_quat_w[:, ee_body_idx]
    _optax = quat_apply(_camq, torch.tensor([[0., 0., 1.]], device=DEVICE).expand(N, 3))   # ROS optical axis = +z
    _c2c = _cube - _camp; _c2cn = _c2c / (_c2c.norm(dim=-1, keepdim=True) + 1e-9)
    from isaaclab.utils.math import quat_inv as _qinv
    _aim_local = quat_apply(_qinv(_eeq), (_cube - _eep))   # cube in gripper_end LOCAL frame == the WRIST_AIM look target
    _dot = (_optax * _c2cn).sum(-1)
    _basep = r.data.body_pos_w[:, r.body_names.index("base_link")] if "base_link" in r.body_names else r.data.body_pos_w[:, 0]
    print(f"[WCAM-PROBE] base_pos(mean)={[round(x,3) for x in _basep.mean(0).tolist()]} cam_pos(mean)={[round(x,3) for x in _camp.mean(0).tolist()]} cube_pos(mean)={[round(x,3) for x in _cube.mean(0).tolist()]} EE_pos(mean)={[round(x,3) for x in _eep.mean(0).tolist()]}", flush=True)
    print(f"[WCAM-PROBE] cube-base(mean, workspace in base frame)={[round(x,3) for x in (_cube.mean(0)-_basep.mean(0)).tolist()]}", flush=True)
    print(f"[WCAM-PROBE] optax.(cam->cube) mean={_dot.mean().item():+.3f} (1=aimed AT cube, <0=looking AWAY)", flush=True)
    print(f"[WCAM-PROBE] optax_world(mean)={[round(x,3) for x in _optax.mean(0).tolist()]}  optaxZ={_optax.mean(0)[2].item():+.3f} (<0=looking DOWN, >0=looking UP)", flush=True)
    print(f"[WCAM-PROBE] WRIST_AIM = cube in gripper_end LOCAL (mean)={[round(x,4) for x in _aim_local.mean(0).tolist()]}", flush=True)
    print(f"[WCAM-PROBE] aim_local per-axis range x[{_aim_local[:,0].min():.2f},{_aim_local[:,0].max():.2f}] y[{_aim_local[:,1].min():.2f},{_aim_local[:,1].max():.2f}] z[{_aim_local[:,2].min():.2f},{_aim_local[:,2].max():.2f}]", flush=True)
    import os as _oe; _oe._exit(0)

ACT_DIM = 7                                              # dx,dy,dz,dyaw, descend, grip, lift
_D_CUBE = int(os.environ.get("D_CUBE", "2"))             # user 2026-07-09: 2=front cube_rel only; 4=front+wrist. MUST match det_xattn_nets D_CUBE.
_D_ACQ = int(os.environ.get("D_ACQ", "0"))               # user 2026-08-01: FRONT-guided aim token dims (0=off; 3=[pcx,pcy,pv] front-cube→wrist-image, SEARCH-only). MUST match det_xattn_nets D_ACQ + deploy. Appended AFTER cube_rel at the state tail.
STATE_DIM = 21 + _D_CUBE + _D_ACQ                         # qpos8+target6+grip1+d2r6=21 proprio + cube_rel(front 2,+wrist 2 if D_CUBE=4) + acq(D_ACQ)
REST_QPOS = torch.tensor(profile.rest_qpos_d2r, device=DEVICE)   # d2r reference for proprio_state (OURS = old low pose)
RT.rest_qpos = REST_QPOS
_STEP = float(os.environ.get("STEP_M", "0.025"))   # user 2026-07-16: action->meters scale for APPROACH servo (descend _adz + lateral x/y target). was hardcoded 0.025; user 2026-07-24 made it a knob (STEP_M) to slow the per-step approach motion (finer descent). policy learns speed via action magnitude in [-1,1] within this cap.
                                                         # (user 2026-06-23: compute is cheap -> short high-freq moves, less overshoot/shaking; deploy must match this scale)
ARM_SCALE = float(os.environ.get("ARM_SCALE", "0.08"))   # joint-delta rad/step (user 2026-06-28: widened 0.05->0.08). MUST match the env arm_action scale:
arm_term._scale = ARM_SCALE                              # override the env TargetDeltaJointPositionAction step (base set it from cfg.scale=0.05) so act_env reproduces the IK soln 1:1
ROT_SCALE = float(os.environ.get("ROT_SCALE", "0.10"))   # EE orientation-delta rad/step for the FREE-orientation IK pose target (user 2026-06-28: DOWN_QUAT no longer forced)
APP_ORI = os.environ.get("APP_ORI", "free")   # user 2026-07-03: APPROACH IK orientation ANCHOR. free=hold current eq (NO anchor -> drifts/topples forward). down=slew grasp-axis(local+z) toward world-DOWN <=ROT_SCALE/step (top-grasp, vertical). freeze=hold orientation captured at APPROACH entry (stops drift, keeps handoff tilt).
ANCHOR_K = float(os.environ.get("ANCHOR_K", "0.0"))   # user 2026-07-10: APP_ORI=policy restoring SPRING toward the handoff pose (_app_ori, captured at SEARCH->APPROACH entry). >0 pulls orientation back to the search-handoff aim each step (rule-based, bounded ROT_SCALE), so the POLICY only CORRECTS around the anchor instead of free-drifting away (which lost the cube: nvis~9%). Max deviation ~ ROT_SCALE/ANCHOR_K per axis. 0 = off (pure policy drift, old). Hybrid: rule anchor + learned correction.
ANCHOR_ANNEAL = float(os.environ.get("ANCHOR_ANNEAL", "0.0"))   # user 2026-07-10: >0 = TRAINING-TIME curriculum — anneal the anchor strength LINEARLY from ANCHOR_K down to 0 over this many env-steps (global g), then 0 = FULLY POLICY. _ak = ANCHOR_K*max(0,1-g/ANCHOR_ANNEAL). Removes the rule safety-net as the policy learns to hold orientation itself. 0 = off (constant ANCHOR_K, no anneal).
CUBE_GONE_R = float(os.environ.get("CUBE_GONE_R", "2.0"))   # user 2026-07-10: cube is "flew away" (physics blowup / knocked far by a flailing fresh policy) if its ENV-LOCAL horizontal dist from env origin exceeds this (m) — spawn box is ~<1.2m so 2.0 only flags real flyaways. Such envs: clear false success + force-reset (fail) + exclude from cube_dist/RSR monitors. Was corrupting metrics (median 0.6 / trimmed-mean 45m bimodal; false 100% success).
W_GONE = float(os.environ.get("W_GONE_CUBE", "0.0"))   # user 2026-07-10: per-step penalty (subtracted from r_sf/r_af/r_lf) when the cube flew away (_cube_gone) — teaches the policy NOT to knock the cube (force-reset alone gives no gradient). e.g. 10. 0=off.
_ZAXIS = torch.tensor([[0.0, 0.0, 1.0]], device=DEVICE)   # gripper local +z = approach/grasp axis (also = world +Z numerically, reused as world-up in pitch-clamp cross products)
_HEAD = torch.tensor([profile.jaw_mouth_offset], device=DEVICE); _HEAD = _HEAD / _HEAD.norm()   # user 2026-07-05: gripper "head/nose" direction (EE link origin -> jaw mouth), normalized. This is the axis the WRIST-PITCH clamp keeps within ±WSERVO_CLAMP° of horizontal (NOT local +z — they differ ~40°). Profile-sourced (TODO(official): measure on Seeed USD).
_JAW_OFF3 = torch.tensor([profile.jaw_mouth_offset], device=DEVICE)   # raw EE-link -> jaw mouth offset (m), un-normalized (profile-sourced)
RT.jaw_mouth = _HEAD; RT.jaw_off_raw = _JAW_OFF3; RT.zaxis = _ZAXIS
IK_DAMP = float(os.environ.get("IK_DAMP", "0.05"))   # user 2026-07-03: DLS damping for position-only APPROACH IK (jaw-mouth offset is +z dominant)
IK_WRIST_W = float(os.environ.get("IK_WRIST_W", "1.0"))   # user 2026-07-10 (B): WEIGHTED DLS penalty on wrist_flex motion in the APPROACH IK. >1 = wrist moves LESS -> ELBOW/LIFT do the reach instead of over-cranking the wrist. 1.0 = off (equal weights). e.g. 3.0. Only affects the 6D-DLS path (APP_ORI in cube/wservo/policy); no effect on DIRECT_JOINTS.
DESC_CEN = float(os.environ.get("DESC_CEN", "0.15"))   # user 2026-07-03: APPROACH descends ONLY when cube centered(ncen<this)+visible; else center first
DESC_SOFT = float(os.environ.get("DESC_SOFT", "0.0"))  # user 2026-07-05: SOFT descent gate — descend rate = exp(-DESC_SOFT*ncen), descends WHILE centering (faster centered). 0=off (hard DESC_CEN gate).
FREEZE_SEARCH = int(os.environ.get("FREEZE_SEARCH", "0"))  # user 2026-07-27: 1 = FREEZE the search net (skip its gradient updates) — search stays at the resumed 100% ckpt, only APPROACH trains. Use with TP_FRESH_SEARCH=0 (resume search) so approach learns against a fixed, perfect search (stable handoffs). Search still ACTS (its fixed policy) but never learns.
FREEZE_APPROACH = int(os.environ.get("FREEZE_APPROACH", "0"))  # user 2026-07-30: 1 = FREEZE the approach net (skip its gradient updates) — SEARCH-ONLY training. Approach still ACTS (fixed) but never learns. Pair with FREEZE_SEARCH=0 + APP_GUAR_S small (end ~immediately on handoff) to train search alone.
DESC_GATE_PITCH = int(os.environ.get("DESC_GATE_PITCH", "1"))  # user 2026-07-27: 1 = also gate the PITCH descent path (APP_ORI=policy a_a[3]). The position gate _desc_ok only clamps EE-z (a_a[2], line 1739); the ungated wrist-pitch lowered the offset jaw-mouth off-center ("gate not taking effect"). When off-center (_desc_ok=0), project OUT the component of the policy pitch that lowers the jaw-mouth z. MUST be mirrored in the deploy action-map. 0 = legacy (pitch ungated).
APP_FAIL_Z = float(os.environ.get("APP_FAIL_Z", "0.0"))  # user 2026-07-05: >0 = APPROACH that descends BELOW this z WITHOUT a grasp -> FAIL-terminate the episode NOW. Fixes success-rate bias under NO_RESRCH (success ends the episode instantly but failures otherwise run the full 120s -> only successes complete early -> RSR looks 100%). Fast fail = balanced rolling window; also stops table-diving.
RESUME_ALPHA = float(os.environ.get("RESUME_ALPHA", "0.0"))  # user 2026-07-05: on RESUME, set search+resumed-approach(+align) SAC alpha to this (extra exploration to adapt to a dynamics change). 0=off (old: search 0.12, approach init 1.0).
SWEEP_ALPHA = float(os.environ.get("SWEEP_ALPHA", "0.0"))  # user 2026-07-28: after each curriculum sweep, BUMP approach alpha UP to this (small, ~0.25) so the policy re-explores the newly re-weighted failing cells; anneals back to floor over the next CURRIC_EVERY. Cyclic exploration tied to curriculum. Small so it re-learns WITHOUT the full 0.5 RESUME_ALPHA reset that collapsed 8951. 0=off.
DIRECT_JOINTS = int(os.environ.get("DIRECT_JOINTS", "0"))  # user 2026-06-28: 1 = plain policy learning (no IK), policy outputs 6 raw joint-deltas. A/B test (e.g. 8897 direct vs 8896 IK)
ALIGN_DIRECT  = int(os.environ.get("ALIGN_DIRECT", "0"))   # user 2026-07-01: 1 = ALIGN uses DIRECT 6 raw joint-deltas while APPROACH stays IK (mixed: IK-approach + DIRECT-align). n_act(align)=6. (DIRECT_JOINTS=1 makes BOTH direct.)

# ---------------- geometry helpers ----------------
def cube_z_w():
    return env.scene[args.target_cube].data.root_pos_w[:, 2]

def cube_xyz_from_front(front_cxcy, plane_z):
    """Coordinate transform (user 2026-06-29): back-project FRONT detection (cx,cy normalized) through intrinsics +
    front-cam pose, intersect the table plane (z=plane_z, = known cube-center height) -> cube WORLD (x,y,z).
    REAL-measurable: front extrinsic + intrinsic + known cube height. Standard ROS cam frame (+Z optical, x right, y down)."""
    Cf = env.scene["front"]; cfp = Cf.data.pos_w; cq = Cf.data.quat_w_ros; Kf = Cf.data.intrinsic_matrices
    Hf = Cf.data.output["rgb"].shape[1]; Wf = Cf.data.output["rgb"].shape[2]
    u = front_cxcy[:, 0] * Wf; v = front_cxcy[:, 1] * Hf
    x_c = (u - Kf[:, 0, 2]) / Kf[:, 0, 0]                            # ray dir in cam frame (z=1 forward)
    y_c = (v - Kf[:, 1, 2]) / Kf[:, 1, 1]
    z_c = torch.ones_like(x_c)
    qw, qx, qy, qz = cq[:, 0], cq[:, 1], cq[:, 2], cq[:, 3]          # cam->world rotation R, ray_world = R @ ray_cam
    rx = (1-2*(qy*qy+qz*qz))*x_c + (2*(qx*qy-qz*qw))*y_c + (2*(qx*qz+qy*qw))*z_c
    ry = (2*(qx*qy+qz*qw))*x_c + (1-2*(qx*qx+qz*qz))*y_c + (2*(qy*qz-qx*qw))*z_c
    rz = (2*(qx*qz-qy*qw))*x_c + (2*(qy*qz+qx*qw))*y_c + (1-2*(qx*qx+qy*qy))*z_c
    pz = plane_z if torch.is_tensor(plane_z) else torch.full_like(cfp[:, 2], float(plane_z))
    t = (pz - cfp[:, 2]) / torch.where(rz.abs() < 1e-6, torch.full_like(rz, 1e-6), rz)
    X = cfp[:, 0] + t * rx; Y = cfp[:, 1] + t * ry
    rz_n = (rz / (torch.sqrt(rx * rx + ry * ry + rz * rz) + 1e-9)).abs()   # |sin(elevation)|: small = grazing = unreliable plane-hit
    return torch.stack([X, Y, pz], dim=-1), rz_n

def cube_xyz_from_wrist(wrist_cxcy, plane_z):
    """user 2026-07-09: same ray∩table-plane unprojection as cube_xyz_from_front, but via the WRIST (eye-in-hand) cam.
    During APPROACH the wrist looks near-vertically down at the cube -> steep ray -> reliable; cam pose is from FK (robust).
    REAL-measurable (wrist extrinsic from FK + intrinsic + known cube height). Returns cube WORLD (x,y,z) + ray reliability."""
    Cw = env.scene["wrist"]; cwp = Cw.data.pos_w; cq = Cw.data.quat_w_ros; Kw = Cw.data.intrinsic_matrices
    Hw = Cw.data.output["rgb"].shape[1]; Ww = Cw.data.output["rgb"].shape[2]
    u = wrist_cxcy[:, 0] * Ww; v = wrist_cxcy[:, 1] * Hw
    x_c = (u - Kw[:, 0, 2]) / Kw[:, 0, 0]; y_c = (v - Kw[:, 1, 2]) / Kw[:, 1, 1]; z_c = torch.ones_like(x_c)
    qw, qx, qy, qz = cq[:, 0], cq[:, 1], cq[:, 2], cq[:, 3]
    rx = (1-2*(qy*qy+qz*qz))*x_c + (2*(qx*qy-qz*qw))*y_c + (2*(qx*qz+qy*qw))*z_c
    ry = (2*(qx*qy+qz*qw))*x_c + (1-2*(qx*qx+qz*qz))*y_c + (2*(qy*qz-qx*qw))*z_c
    rz = (2*(qx*qz-qy*qw))*x_c + (2*(qy*qz+qx*qw))*y_c + (1-2*(qx*qx+qy*qy))*z_c
    pz = plane_z if torch.is_tensor(plane_z) else torch.full_like(cwp[:, 2], float(plane_z))
    t = (pz - cwp[:, 2]) / torch.where(rz.abs() < 1e-6, torch.full_like(rz, 1e-6), rz)
    X = cwp[:, 0] + t * rx; Y = cwp[:, 1] + t * ry
    rz_n = (rz / (torch.sqrt(rx * rx + ry * ry + rz * rz) + 1e-9)).abs()
    return torch.stack([X, Y, pz], dim=-1), rz_n

PRIV_UNTIL = int(os.environ.get("PRIV_UNTIL", "0"))   # user 2026-07-19: SAFETY CAP (steps) — if the RSR target is never reached, force the anneal to start here anyway. 0 = no cap (pure RSR-triggered). Requires PRIV_RSR_TARGET>0 to enable privilege at all.
PRIV_RSR_TARGET = float(os.environ.get("PRIV_RSR_TARGET", "0.0"))   # user 2026-07-19: >0 enables PRIVILEGED bootstrap. Stay 100% privileged until the rolling RSR (n>=500) reaches this (≈mastered w/ perfect perception), THEN linearly anneal privilege->real over PRIV_ANNEAL steps. Watch the live RSR climb to this = "privileged RSR" (auto-measured, same dcube metric).
PRIV_ANNEAL = int(os.environ.get("PRIV_ANNEAL", "2000000"))   # user 2026-07-19: steps to linearly anneal privilege weight 1->0 after the target is hit (then fully deployable).
PRIV_SEARCH = int(os.environ.get("PRIV_SEARCH", "1"))   # user 2026-07-19: 1 = privilege BOTH phases (whole-pipeline bootstrap — search+approach perfect perception until RSR target, then anneal both -> real). Needs the front-guided acquisition reward (W_ACQ_F>0) so privileged FRESH search can actually bootstrap (perfect perception alone can't aim an off-frame cube). 0 = APPROACH-only privilege (search always real).
PRIV_POS = int(os.environ.get("PRIV_POS", "1"))   # user 2026-07-19: 1 = under privilege, feed the TRUE gripper->cube offset into cube_rel (state 21,22), bypassing the front-visibility/RZ_MIN/FRONT_DROP gates. So the policy ALWAYS knows the cube position during bootstrap (not just when front happens to see it) -> fastest search bootstrap. Also skips FRONT_DROP for privileged envs. Anneals to the real front-derived cube_rel as privilege->0. 0 = position only via the (gated) front unprojection, even under privilege.
PRIV_KEEP_FAIL = int(os.environ.get("PRIV_KEEP_FAIL", "0"))   # user 2026-07-19: 1 = AFTER the global anneal, KEEP privilege ON only for envs spawned in FAILING curriculum cells (latest sweep RSR < PRIV_KEEP_RSR); passing cells go fully deployable. Focuses perfect-perception help where the policy still fails, instead of over-concentrating spawns there. Needs CURRIC=1 (the sweep supplies the per-cell fail map).
PRIV_KEEP_RSR = float(os.environ.get("PRIV_KEEP_RSR", "0.5"))   # a curric cell with sweep RSR below this = "failing" -> keeps privilege
_ENV_CELL = [None]        # user 2026-07-19: per-env current TRAINING spawn cell (curric grid), exposed to get_obs for region-selective privilege
_CELL_KEEPPRIV = [None]   # user 2026-07-19: per-cell bool — keep privilege in this cell (failing cell per the latest sweep)
_PRIV_W = [1.0]   # current privilege weight (1=full priv, 0=full real); set in the train loop, read by get_obs
_priv_astart = [-1]   # step the anneal started (-1 = target not yet reached)
_SWEEPING = [False]   # user 2026-07-19: mirror of _sweeping for get_obs — sweeps ALWAYS use the real detector (deployable eval), never privileged
CUBE_EDGE  = float(os.environ.get("CUBE_EDGE", "0.04"))   # cube edge length (m) — privileged frac model: frac ≈ (fx·CUBE_EDGE/depth)²/(W·H). tune if priv frac scale mismatches the real detector.
FRAC_PRIV_K = float(os.environ.get("FRAC_PRIV_K", "0.0"))  # >0 overrides the analytic frac·depth² constant (else computed from fx,CUBE_EDGE,W,H). lets you anchor priv frac to the sim's calibrated frac without a code change.
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "0"))   # user 2026-07-19: >0 = ALSO keep a permanent step-named checkpoint (search_<TAG>_<step>.pt/approach_..., NEVER overwritten) every this many env-steps. 0 = only rolling last/best. Fixes "only 1 ckpt kept" — full history to bench any point.
CKPT_TAG = os.environ.get("CKPT_TAG", "") or time.strftime("r%m%d_%H%M")   # user 2026-07-19: UNIQUE-per-run tag prefixed to step-named ckpts. FIX for the "not everything is saved" bug: g RESETS to 0 on every (re)launch, so bare search_<step>.pt collided across runs (a new run's search_200k overwrote the prior run's). Tagging by run (default = submit id, else launch time) => every run's full history is preserved, never overwritten.
_GSTEP = [0]   # current global env-step (updated in the train loop) — read by get_obs to gate PRIV_UNTIL
_PH_CUR = [None]   # user 2026-07-19: current per-env phase tensor (0=SEARCH,1=APPROACH,2=ALIGN); set by the train loop right before get_obs -> APPROACH-ONLY privilege (search stays on the real detector + front-guided acquisition reward)
_ACQ = {"pcx": None, "pcy": None, "pv": None}   # user 2026-07-19: FRONT-GUIDED search acquisition — the FRONT-known cube (recon) projected into the WRIST image (pcx,pcy)+validity; stashed by get_obs, consumed by r_s

def cube_to_cam(cam_name):
    """user 2026-07-19: PRIVILEGED detection — project the TRUE cube world pose into cam_name -> (cx,cy,frac,vis,yaw,conf),
    same layout as _yolo_det. Inverse of cube_xyz_from_* (ROS optical frame: R^T@(cube-cam_pos), perspective divide).
    frac from a 1/depth² size model (monotonic in depth like the real detector). NOT deployable — bootstrap only."""
    C = env.scene[cam_name]; p = C.data.pos_w; cq = C.data.quat_w_ros; K = C.data.intrinsic_matrices
    H = C.data.output["rgb"].shape[1]; W = C.data.output["rgb"].shape[2]
    cube = env.scene[args.target_cube].data.root_pos_w                              # (N,3) TRUE world
    dx = cube[:, 0] - p[:, 0]; dy = cube[:, 1] - p[:, 1]; dz = cube[:, 2] - p[:, 2]
    qw, qx, qy, qz = cq[:, 0], cq[:, 1], cq[:, 2], cq[:, 3]                          # R = cam->world; cube_cam = Rᵀ·d
    xc = (1-2*(qy*qy+qz*qz))*dx + (2*(qx*qy+qz*qw))*dy + (2*(qx*qz-qy*qw))*dz
    yc = (2*(qx*qy-qz*qw))*dx + (1-2*(qx*qx+qz*qz))*dy + (2*(qy*qz+qx*qw))*dz
    zc = (2*(qx*qz+qy*qw))*dx + (2*(qy*qz-qx*qw))*dy + (1-2*(qx*qx+qy*qy))*dz        # forward depth (optical +Z)
    _zs = torch.where(zc.abs() < 1e-6, torch.full_like(zc, 1e-6), zc)
    u = (xc / _zs) * K[:, 0, 0] + K[:, 0, 2]; v = (yc / _zs) * K[:, 1, 1] + K[:, 1, 2]
    cx = u / W; cy = v / H
    vis = ((zc > 0.02) & (cx >= 0.0) & (cx <= 1.0) & (cy >= 0.0) & (cy <= 1.0)).float()
    _Kf = FRAC_PRIV_K if FRAC_PRIV_K > 0.0 else (K[:, 0, 0]**2 * (CUBE_EDGE**2) / (W * H))
    frac = torch.clamp(_Kf / (zc.clamp(min=1e-3)**2), 0.0, 1.0) * vis
    _cq2 = env.scene[args.target_cube].data.root_quat_w                             # TRUE cube yaw about world-Z
    cyaw = torch.atan2(2*(_cq2[:, 0]*_cq2[:, 3] + _cq2[:, 1]*_cq2[:, 2]), 1 - 2*(_cq2[:, 2]**2 + _cq2[:, 3]**2))
    _half = torch.full_like(cx, 0.5)
    cx = torch.where(vis > 0.5, cx.clamp(0.0, 1.0), _half); cy = torch.where(vis > 0.5, cy.clamp(0.0, 1.0), _half)
    return torch.stack([cx, cy, frac, vis, cyaw, vis], dim=-1)

def _cam_ray(cam, cxcy):
    """cam sensor + normalized (cx,cy) -> (origin_w (N,3), dir_w unit (N,3)). Standard ROS optical frame."""
    p = cam.data.pos_w; cq = cam.data.quat_w_ros; K = cam.data.intrinsic_matrices
    H = cam.data.output["rgb"].shape[1]; W = cam.data.output["rgb"].shape[2]
    u = cxcy[:, 0] * W; v = cxcy[:, 1] * H
    xc = (u - K[:, 0, 2]) / K[:, 0, 0]; yc = (v - K[:, 1, 2]) / K[:, 1, 1]; zc = torch.ones_like(xc)
    qw, qx, qy, qz = cq[:, 0], cq[:, 1], cq[:, 2], cq[:, 3]
    rx = (1-2*(qy*qy+qz*qz))*xc + (2*(qx*qy-qz*qw))*yc + (2*(qx*qz+qy*qw))*zc
    ry = (2*(qx*qy+qz*qw))*xc + (1-2*(qx*qx+qz*qz))*yc + (2*(qy*qz-qx*qw))*zc
    rz = (2*(qx*qz-qy*qw))*xc + (2*(qy*qz+qx*qw))*yc + (1-2*(qx*qx+qy*qy))*zc
    d = torch.stack([rx, ry, rz], dim=-1)
    return p, d / (d.norm(dim=-1, keepdim=True) + 1e-9)

def cube_xyz_triangulate(front_cxcy, wrist_cxcy):
    """user 2026-07-10: STEREO triangulate cube 3D from FRONT ray (fixed calib) + WRIST ray (FK pose from joints).
    NO plane assumption — closest point between the two world-space rays. Returns (mid_xyz (N,3), gap (N,)).
    gap = distance between the two rays' closest points (0=rays intersect=reliable; large=inconsistent detection/near-parallel)."""
    P1, d1 = _cam_ray(env.scene["front"], front_cxcy)
    P2, d2 = _cam_ray(env.scene["wrist"], wrist_cxcy)
    r = P1 - P2
    b = (d1 * d2).sum(-1)                                            # d1·d2 (both unit) ; |b|->1 = parallel = ill-conditioned
    d = (d1 * r).sum(-1); e = (d2 * r).sum(-1)
    denom = (1.0 - b * b).clamp(min=1e-6)
    t1 = (b * e - d) / denom; t2 = (e - b * d) / denom
    c1 = P1 + t1.unsqueeze(-1) * d1; c2 = P2 + t2.unsqueeze(-1) * d2
    mid = 0.5 * (c1 + c2); gap = (c1 - c2).norm(dim=-1)
    return mid, gap

# ---------------- learned detector (YOLO-OBB red cube) -> (N,5): cx,cy,frac,vis,yaw ----------------
_YOLO = None
def _load_yolo():
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO
        _YOLO = YOLO(args.detector)
        _YOLO.to(DEVICE)
        print(f"[DET] YOLO-OBB detector loaded: {args.detector} (conf={args.det_conf}) -> replaces HSV", flush=True)
    return _YOLO

@torch.no_grad()
def _yolo_det(rgb_uint8):
    """rgb (N,H,W,3) uint8 -> (N,5): cx,cy in [0,1], area_frac, visible, yaw(rad). Learned red-cube OBB.
    Real-measurable ONLY (image pixels) — NO privileged cube world pose. Keeps the HSV interface so the
    net + cross-attention are unchanged; just a better, sim2real-robust front-end."""
    model = _load_yolo()
    Nn, H, W = rgb_uint8.shape[0], rgb_uint8.shape[1], rgb_uint8.shape[2]
    x = rgb_uint8.permute(0, 3, 1, 2).float() / 255.0                 # (N,3,H,W) RGB, GPU (matches training)
    res = model.predict(x, imgsz=H, conf=args.det_conf, max_det=1, verbose=False, device=DEVICE)
    # user 2026-07-18 (perf B): assemble the per-image Results into GPU tensors (NO per-item .item()/float() sync),
    #   then do ALL the arithmetic vectorized ONCE. Loop only pulls tensor slices (GPU->GPU). was: python arithmetic + float() per env = 256-512 CPU syncs/step.
    _xywhr = torch.zeros(Nn, 5, device=DEVICE); _cf = torch.zeros(Nn, device=DEVICE); _val = torch.zeros(Nn, dtype=torch.bool, device=DEVICE)
    for i, rr in enumerate(res):
        ob = rr.obb
        if ob is not None and ob.xywhr is not None and len(ob.xywhr) > 0:
            _xywhr[i] = ob.xywhr[0]; _val[i] = True
            if ob.conf is not None and len(ob.conf) > 0: _cf[i] = ob.conf[0]
    _half = torch.full((Nn,), 0.5, device=DEVICE); _zero = torch.zeros(Nn, device=DEVICE)
    cx  = torch.where(_val, (_xywhr[:, 0] / W).clamp(0, 1), _half)
    cy  = torch.where(_val, (_xywhr[:, 1] / H).clamp(0, 1), _half)
    frac = torch.where(_val, ((_xywhr[:, 2] * _xywhr[:, 3]) / (W * H)).clamp(0, 1), _zero)
    vis = _val.float()
    yaw = torch.where(_val, _xywhr[:, 4], _zero)                       # OBB angle (rad); sin/cos used downstream
    return torch.stack([cx, cy, frac, vis, yaw, _cf], dim=-1)          # conf = YOLO box conf (col 5) for det-rate-vs-distance diagnostic

def detect(rgb_uint8):
    return _yolo_det(rgb_uint8)   # jobs ALWAYS pass --detector -> learned YOLO-OBB (HSV _red_det fallback removed)

_LAST_IMG = {"front": None, "wrist": None}
_SEG_DBG = False   # one-time [SEG-LABELS] dump (verify recolor class names match the table/gripper semantic labels)
# user 2026-06-28: PER-EPISODE viewpoint warp (was per-frame in visual_obs.viewpoint_warp -> jittered the detected
#   cube cx,cy every step = noisy policy obs). Sample the affine ONCE per episode (resampled in reset_fsm), apply
#   the SAME warp every frame within the episode -> stable obs, still DR across episodes.
_WARP_SCALE = float(os.environ.get("WARP_SCALE", "0.06")); _WARP_ROT = float(os.environ.get("WARP_ROT", "0.05"))
_WARP_SHIFT = float(os.environ.get("WARP_SHIFT", "0.03")); _WARP_SHEAR = float(os.environ.get("WARP_SHEAR", "0.03"))   # user 2026-07-08: WARP_SHIFT env-tunable — big per-episode cx+cy image shift (≈ wrist-cam extrinsic DR) breaks the wflex↔cy shortcut, forces visual servo
MASK_PROPRIO = int(os.environ.get("MASK_PROPRIO", "0"))   # user 2026-07-08: 1 = zero the ENTIRE proprio block (state idx 0:21 = qpos,arm_target,grip,d2r), keep cube_rel(21,22) + detection tokens -> policy is PURELY perception-driven (no proprio shortcut anywhere) -> forced visual servo. search+approach.
PROPRIO_KEEP = os.environ.get("PROPRIO_KEEP", "")         # user 2026-07-08: WHITELIST — keep ONLY these state[0:21] indices, zero the rest (overrides MASK_PROPRIO). e.g. "15,18" = d2r pan+wflex ONLY (minimal camera-orientation feedback for IK search velocity loop, without full proprio shortcut). Needs PROPRIO_OFF=0 (state_proj MLP alive to consume the kept dims).
_PKEEP_IDX = [int(i) for i in PROPRIO_KEEP.split(",") if i.strip() != ""]
_SWDR_LO = float(os.environ.get("START_WRIST_LO", "0")); _SWDR_HI = float(os.environ.get("START_WRIST_HI", "0"))   # user 2026-07-08: >0 both -> per-episode START_WRIST DR in [LO,HI] (start-pose variation; policy can't assume fixed start wflex)
_warp_p = torch.zeros(N, 5, device=DEVICE)   # per-env [s, th, sh, tx, ty], held for the whole episode
def resample_warp(ids=None):
    _sl = slice(None) if ids is None else ids
    _n = N if ids is None else (len(ids) if not torch.is_tensor(ids) else ids.numel())
    _r = torch.rand(_n, 5, device=DEVICE) - 0.5
    _warp_p[_sl, 0] = 1.0 + _r[:, 0] * 2 * _WARP_SCALE
    _warp_p[_sl, 1] = _r[:, 1] * 2 * _WARP_ROT
    _warp_p[_sl, 2] = _r[:, 2] * 2 * _WARP_SHEAR
    _warp_p[_sl, 3] = _r[:, 3] * 2 * _WARP_SHIFT
    _warp_p[_sl, 4] = _r[:, 4] * 2 * _WARP_SHIFT
    _ep_bg[_sl] = _rand_bg(_n)        # user 2026-06-28: greenscreen bg ALSO per-episode (was per-frame -> flickered every step)
_ep_bg = _rand_bg(N)                  # per-episode background, resampled on reset
resample_warp()
def _apply_warp(img):
    import torch.nn.functional as _Fw
    _n2, _h, _w, _ = img.shape
    x = img.permute(0, 3, 1, 2).float()
    s, th, sh, tx, ty = _warp_p[:, 0], _warp_p[:, 1], _warp_p[:, 2], _warp_p[:, 3], _warp_p[:, 4]
    cos, sin = torch.cos(th), torch.sin(th)
    theta = torch.zeros(_n2, 2, 3, device=img.device)
    theta[:, 0, 0] = (cos + sh * sin) * s; theta[:, 0, 1] = (-sin + sh * cos) * s; theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin * s; theta[:, 1, 1] = cos * s; theta[:, 1, 2] = ty
    grid = _Fw.affine_grid(theta, x.shape, align_corners=False)
    out = _Fw.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8)
def render_cams():
    """P1-identical _cam_ds pipeline (--dr): greenscreen bg + table LIGHT-GRAY + gripper BLACK + image_dr +
    viewpoint_warp -> matches the real rig, and HSV detection runs on the SAME sim2real-matched image."""
    out = {}
    for nm in ("front", "wrist"):
        rgb = env.scene[nm].data.output["rgb"][..., :3]               # (N,H,W,3) uint8
        if args.dr:
            seg = env.scene[nm].data.output["semantic_segmentation"]
            info = env.scene[nm].data.info["semantic_segmentation"]
            global _SEG_DBG
            if not _SEG_DBG:
                try: print(f"[SEG-LABELS] {nm} idToLabels = {info.get('idToLabels', info)}", flush=True)
                except Exception as _e: print(f"[SEG-LABELS] {nm} info={info} ({_e})", flush=True)
                if nm == "wrist": _SEG_DBG = True
            rgb = composite_bg(rgb, seg, info, _ep_bg, keep_lower=0.10)   # per-EPISODE bg (resampled on reset) — was _rand_bg(N) per frame
            rgb = recolor_seg_region(rgb, seg, info, ("table",), 0.88, 1.05, 0.0, grain=False, base=0.50)   # gray table
            rgb = recolor_seg_region(rgb, seg, info, ("gripper",), 0.02, 0.12, 0.0, grain=False, base=0.0)  # black gripper
            rgb = image_dr(rgb, scale=False)   # user 2026-06-28: scale=False kills image_dr's PER-FRAME affine (zoom/shift) — the per-episode _apply_warp handles geometric DR now (was scale=(nm=="front") -> per-frame angle jitter)
            rgb = _apply_warp(rgb)   # PER-EPISODE warp (params in _warp_p, resampled on reset) — was per-frame viewpoint_warp
        if nm == "wrist" and WRIST_ROT90:   # user 2026-07-20: un-rotate the OFFICIAL wrist cam (mounted 90° off) by an IMAGE rotation (convention-free, unlike the quat roll). k*90° CCW. Detector + net + reward all see the SAME upright image. OURS: WRIST_ROT90=0 (unchanged).
            rgb = torch.rot90(rgb, int(WRIST_ROT90), dims=(1, 2)).contiguous()
        out[nm] = rgb
    _LAST_IMG["front"] = out["front"]; _LAST_IMG["wrist"] = out["wrist"]
    return out

def proprio_state():
    qpos = r.data.joint_pos[:, :8]                                    # 8
    tq = arm_term._target_qpos                                        # 6 arm PD targets (commanded)
    grip_t = r.data.joint_pos[:, 6:7]                                 # 1 gripper qpos (proxy for grip target)
    d2r = (r.data.joint_pos[:, joint_ids] - REST_QPOS)               # 6
    return torch.cat([qpos, tq, grip_t, d2r], dim=-1)                # 8+6+1+6 = 21

_cube_plane_z = None   # cached known cube-CENTER world height (constant; deployable = table_top + cube_half)
CUBE_REL_CLAMP = float(os.environ.get("CUBE_REL_CLAMP", "1.0"))   # clamp front-derived cube-rel offset (m); kills grazing-angle garbage
FRONT_SEARCH_ONLY = int(os.environ.get("FRONT_SEARCH_ONLY","1"))   # 2026-07-22: 1=front cube_rel SEARCH-only(approach uses wrist), 0=all phases (for fair evaluation of old ckpts)
FRONT_APP_OFF = int(os.environ.get("FRONT_APP_OFF","0"))   # 2026-07-22: 1=also zero the FRONT detection TOKEN (fd) in APPROACH/ALIGN (ph>=1) -> approach is WRIST-ONLY (front token+position both gone). SEARCH keeps front. 0=keep front token in approach (xattn still sees it).
FRONT_OFF_WHEN_WRIST = int(os.environ.get("FRONT_OFF_WHEN_WRIST","0"))   # user 2026-07-30: in SEARCH(ph==0), if the WRIST sees the cube (wd vis>0.5), turn OFF both the FRONT token and cube_rel -> state-conditional enforcement of "acquire=front, center=wrist" (prevents front over-reliance / divergence on the real robot). front is on only when the wrist is blind (for acquisition). Parity with deploy FRONT_OFF_WHEN_WRIST. 0=off.
RZ_MIN = float(os.environ.get("RZ_MIN", "0.30"))   # ray reliability gate: |sin(elevation)| must exceed this (~17deg); grazing rays -> unreliable plane-hit -> gate cube_rel to 0
RZ_MIN_W = float(os.environ.get("RZ_MIN_W", str(RZ_MIN)))   # user 2026-07-09: WRIST-specific steepness gate (higher, e.g. 0.6). wrist unproj accurate only when looking DOWN steeply (approach); shallow (search) -> garbage -> gate cube_rel_wrist to 0. front keeps RZ_MIN.
FRONT_DROP_P = float(os.environ.get("FRONT_DROP_P", "0.0"))   # user 2026-07-01: per-episode prob of HIDING front detection (reduce front dependence -> wrist-led search). 0=off
FRONT_WEIGHT = float(os.environ.get("FRONT_WEIGHT", "1.0"))   # user 2026-07-31: weight for blending front det(cx,cy) toward neutral (0.5) in SEARCH training (1=full, 0.5=halved, ->0=front meaningless). Weakened to "front gives direction only" -> robust to real-robot front calibration error. eval/sweep clean.
_front_drop = None                                            # per-env episode dropout mask for FRONT_DROP_P (lazy-init in get_obs, resampled on reset)

_PROF_ON = int(os.environ.get("PROF", "0"))   # user 2026-07-18: 1 = time env.step/render/detect/update phases (adds cuda.synchronize -> profiling only). 0 = zero overhead.
_PROF = {"step": 0.0, "render": 0.0, "detect": 0.0, "upd": 0.0, "n": 0}
def _psync():
    if _PROF_ON: torch.cuda.synchronize(DEVICE)

def get_obs():
    _tpr = time.perf_counter() if _PROF_ON else 0.0
    cams = render_cams()
    if _PROF_ON: _psync(); _PROF["render"] += time.perf_counter() - _tpr; _tpd = time.perf_counter()
    _pw = 0.0 if _SWEEPING[0] else (_PRIV_W[0] if PRIV_RSR_TARGET > 0.0 else 0.0)   # user 2026-07-19: privilege weight (1=full priv, 0=full real). APPROACH-ONLY now: search learns w/ the real detector + front-guided acquisition reward; privilege only removes YOLO noise on the already-framed cube during APPROACH (where the weak wrist-frac signal is the real bottleneck). sweeps ALWAYS real.
    _phv = _PH_CUR[0]
    _priv_env = None                                                 # user 2026-07-19: per-env bool of which envs are PRIVILEGED this step (reused for FRONT_DROP skip + PRIV_POS true cube_rel)
    _keepon = (PRIV_KEEP_FAIL and PRIV_RSR_TARGET > 0.0 and not _SWEEPING[0] and _phv is not None and _CELL_KEEPPRIV[0] is not None and _ENV_CELL[0] is not None)   # region-selective privilege active (post-anneal, keep priv in failing cells)
    if (_pw <= 1e-9 and not _keepon) or _phv is None:               # FULL real detector (deployable) — all envs (no anneal weight AND no failing-cell keep)
        fdf = detect(cams["front"]); wdf = detect(cams["wrist"])       # user 2026-07-18: kept as 2 calls — batching front+wrist into one 512-img yolo11x predict OOM'd (peak +3GB). vectorized postproc (in _yolo_det) is the retained perf-B win.
    else:                                                            # per-phase: SEARCH always real, APPROACH privileged (weight _pw, per-env). YOLO runs for everyone (search needs it) — no more full-priv YOLO-skip.
        fdf = detect(cams["front"]); wdf = detect(cams["wrist"])
        _appr = (_phv >= 0) if PRIV_SEARCH else (_phv == 1)            # PRIV_SEARCH=1 -> privilege BOTH phases (whole-pipeline); 0 -> APPROACH envs only
        _priv_env = _appr & (torch.rand(_appr.shape[0], device=DEVICE) < _pw)   # _pw=1 -> all eligible envs priv; anneal(0<_pw<1) -> per-env prob; SEARCH envs (PRIV_SEARCH=0) always real
        if _keepon:                                                   # user 2026-07-19: ALSO keep privilege for envs spawned in FAILING curric cells (region-selective; survives the global anneal)
            _kp = _appr & (_ENV_CELL[0] >= 0) & _CELL_KEEPPRIV[0][_ENV_CELL[0].clamp(min=0)]
            _priv_env = _priv_env | _kp
        if bool(_priv_env.any()):
            _fp = cube_to_cam("front"); _wp = cube_to_cam("wrist")
            _m = _priv_env.float().unsqueeze(-1)
            fdf = _m * _fp + (1.0 - _m) * fdf; wdf = _m * _wp + (1.0 - _m) * wdf
    if _PROF_ON: _psync(); _PROF["detect"] += time.perf_counter() - _tpd
    fd = fdf[:, :4]; wd = wdf[:, :4]                                 # 4D detection vectors for the net (cx,cy,frac,vis)
    yaw = wdf[:, 4]                                                  # detection cube yaw (REAL: wrist blob principal axis)
    ori = torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1)
    st = proprio_state()
    # COORDINATE TRANSFORM (user 2026-06-29): front-cam detection -> cube WORLD 3D (ray∩table plane) -> rel-to-gripper.
    #   gives the policy a NON-blind approach signal (front sees the cube even when the wrist can't). all real-measurable.
    global _cube_plane_z, _unproj_n, _front_drop
    if _cube_plane_z is None:
        _cube_plane_z = float(env.scene[args.target_cube].data.root_pos_w[:, 2].mean().item())   # const cube-center height (deploy = table+half)
        try:   # user 2026-07-10: ONE-TIME front-cam FOV footprint on the table plane (env-local) -> spawn should live INSIDE this (front must SEE the cube to start approach)
            _Cf = env.scene["front"]; _cfp0 = _Cf.data.pos_w[0]; _cq0 = _Cf.data.quat_w_ros[0]; _Kf0 = _Cf.data.intrinsic_matrices[0]
            _Hf = _Cf.data.output["rgb"].shape[1]; _Wf = _Cf.data.output["rgb"].shape[2]
            _gx, _gy = torch.meshgrid(torch.linspace(0.03, 0.97, 16, device=DEVICE), torch.linspace(0.03, 0.97, 16, device=DEVICE), indexing="ij")
            _uu = _gx.reshape(-1) * _Wf; _vv = _gy.reshape(-1) * _Hf
            _xc = (_uu - _Kf0[0, 2]) / _Kf0[0, 0]; _yc = (_vv - _Kf0[1, 2]) / _Kf0[1, 1]; _zc = torch.ones_like(_xc)
            _qw, _qx, _qy, _qz = _cq0[0], _cq0[1], _cq0[2], _cq0[3]
            _rx = (1-2*(_qy*_qy+_qz*_qz))*_xc + (2*(_qx*_qy-_qz*_qw))*_yc + (2*(_qx*_qz+_qy*_qw))*_zc
            _ry = (2*(_qx*_qy+_qz*_qw))*_xc + (1-2*(_qx*_qx+_qz*_qz))*_yc + (2*(_qy*_qz-_qx*_qw))*_zc
            _rz = (2*(_qx*_qz-_qy*_qw))*_xc + (2*(_qy*_qz+_qx*_qw))*_yc + (1-2*(_qx*_qx+_qy*_qy))*_zc
            _t = (_cube_plane_z - _cfp0[2]) / torch.where(_rz.abs() < 1e-6, torch.full_like(_rz, 1e-6), _rz)
            _org = env.scene.env_origins[0]
            _Xl = (_cfp0[0] + _t*_rx) - _org[0]; _Yl = (_cfp0[1] + _t*_ry) - _org[1]
            _ok = (_t > 0) & (_Xl.abs() < 3.0) & (_Yl.abs() < 3.0)   # in front of cam + drop near-horizon blowups
            if bool(_ok.any()):
                print(f"[FRONT-FOV] env-local  X[{float(_Xl[_ok].min()):.3f}, {float(_Xl[_ok].max()):.3f}]  Y[{float(_Yl[_ok].min()):.3f}, {float(_Yl[_ok].max()):.3f}]  (fwd-dist Y+1.15 -> [{float(_Yl[_ok].min())+1.15:.3f}, {float(_Yl[_ok].max())+1.15:.3f}])  cam-local({float(_cfp0[0]-_org[0]):.2f},{float(_cfp0[1]-_org[1]):.2f},{float(_cfp0[2]-_org[2]):.2f}) plane_z={_cube_plane_z:.3f}  hits {int(_ok.sum())}/{_ok.numel()}", flush=True)
            else:
                print("[FRONT-FOV] no valid plane hits (cam not looking at table?)", flush=True)
        except Exception as _fe:
            print(f"[FRONT-FOV] footprint calc failed: {_fe}", flush=True)
    if FRONT_DROP_P > 0.0:                                          # FRONT DROPOUT (user 2026-07-01): hide front det for masked envs -> "front not visible"; cube_rel auto-zeros via the vis gate below
        if _front_drop is None or _front_drop.shape[0] != fd.shape[0]:
            _front_drop = (torch.rand(fd.shape[0], device=DEVICE) < FRONT_DROP_P)
        _fdrop = _front_drop if (not PRIV_POS or _priv_env is None) else (_front_drop & ~_priv_env)   # user 2026-07-19: don't hide front for PRIVILEGED envs (front is the position source during bootstrap; hiding it defeats privilege)
        _dm = _fdrop.float().unsqueeze(-1)
        fd = fd * (1.0 - _dm) + _dm * torch.tensor([0.5, 0.5, 0.0, 0.0], device=fd.device, dtype=fd.dtype)
    if FRONT_WEIGHT != 1.0 and (_phv is not None) and (not TP_EVAL) and not (_curric_on and _sweeping[0]):   # user 2026-07-31: in SEARCH training, blend front det(cx,cy) toward neutral (0.5) -> keeps direction, weakens magnitude ("front gives direction only"). Prevents over-reliance on precise front (robust to real-robot calibration error). eval/sweep stay clean. weight=0.5=halved.
        _fwm = (_phv == 0).float().unsqueeze(-1); _fw = 1.0 - _fwm * (1.0 - FRONT_WEIGHT)
        fd = fd.clone(); fd[:, :2] = 0.5 + _fw * (fd[:, :2] - 0.5)
    recon, rz_n = cube_xyz_from_front(fd[:, :2], _cube_plane_z)      # (N,3) world (deployable) + ray reliability
    ee = r.data.body_pos_w[:, ee_body_idx]                          # gripper world (FK, real)
    gate = fd[:, 3:4] * (rz_n > RZ_MIN).float().unsqueeze(-1)        # 0 unless front sees cube AND ray steep enough (reliable)
    _crel = torch.clamp(recon[:, :2] - ee[:, :2], -CUBE_REL_CLAMP, CUBE_REL_CLAMP)               # clamp FIRST -> finite even for a grazing/degenerate ray
    cube_rel = torch.where(gate > 0, _crel, torch.zeros_like(_crel))                             # ROOT FIX: gate via where, NOT *gate (NaN*0=NaN would leak a diverged-env FK)
    if PRIV_POS and _priv_env is not None and bool(_priv_env.any()):   # user 2026-07-19: PRIVILEGED envs get the TRUE gripper->cube offset (ungated) as cube_rel -> policy always knows the position during bootstrap; anneals to the front-derived value as privilege->0
        _true_rel = torch.clamp(env.scene[args.target_cube].data.root_pos_w[:, :2] - ee[:, :2], -CUBE_REL_CLAMP, CUBE_REL_CLAMP)
        _pm = _priv_env.float().unsqueeze(-1)
        cube_rel = _pm * _true_rel + (1.0 - _pm) * cube_rel
    if FRONT_SEARCH_ONLY and _phv is not None:   # user 2026-07-22: FRONT cube_rel is SEARCH-only. In APPROACH/ALIGN (ph>=1) the policy must TRACK with the WRIST (cube_rel_w via RZ_MIN_W + wrist detection cx/cy), NOT lean on the external front cam. Zero the front slot off-search.
        cube_rel = torch.where((_phv == 0).unsqueeze(-1), cube_rel, torch.zeros_like(cube_rel))
    st = torch.cat([st, cube_rel], dim=-1)                          # 21 + 2 = 23
    if _D_CUBE >= 4:                                                # user 2026-07-09: append cube_rel_WRIST (eye-in-hand unproj) as extra cube-token dims -> 25
        recon_w, rz_w = cube_xyz_from_wrist(wd[:, :2], _cube_plane_z)
        gate_w = wd[:, 3:4] * (rz_w > RZ_MIN_W).float().unsqueeze(-1)  # 0 unless wrist sees cube AND ray STEEP (RZ_MIN_W; shallow=garbage)
        _crelw = torch.clamp(recon_w[:, :2] - ee[:, :2], -CUBE_REL_CLAMP, CUBE_REL_CLAMP)
        cube_rel_w = torch.where(gate_w > 0, _crelw, torch.zeros_like(_crelw))                   # ROOT FIX: gate via where, NOT *gate (NaN*0=NaN leak)
        st = torch.cat([st, cube_rel_w], dim=-1)                    # 23 + 2 = 25
    # user 2026-07-19: FRONT-GUIDED SEARCH acquisition — project the FRONT-known cube (recon world xyz) into the WRIST image.
    #   gives r_s a dense "swing the wrist toward where FRONT sees the cube" gradient BEFORE the wrist itself acquires it.
    #   fully deployable (front-homography recon + wrist FK projection = inverse of cube_xyz_from_wrist). valid = front reliable AND point in front of the wrist optical plane. pcx/pcy are UNCLAMPED so an off-frame cube gives a large err -> strong pull toward frame.
    try:
        _Cw2 = env.scene["wrist"]; _pw2 = _Cw2.data.pos_w; _cqw = _Cw2.data.quat_w_ros; _Kw2 = _Cw2.data.intrinsic_matrices
        _Hw2 = _Cw2.data.output["rgb"].shape[1]; _Ww2 = _Cw2.data.output["rgb"].shape[2]
        _dxw = recon[:, 0] - _pw2[:, 0]; _dyw = recon[:, 1] - _pw2[:, 1]; _dzw = recon[:, 2] - _pw2[:, 2]
        _qww, _qxw, _qyw, _qzw = _cqw[:, 0], _cqw[:, 1], _cqw[:, 2], _cqw[:, 3]      # R = cam->world; cube_cam = Rᵀ·d (ROS optical)
        _xcw = (1-2*(_qyw*_qyw+_qzw*_qzw))*_dxw + (2*(_qxw*_qyw+_qzw*_qww))*_dyw + (2*(_qxw*_qzw-_qyw*_qww))*_dzw
        _ycw = (2*(_qxw*_qyw-_qzw*_qww))*_dxw + (1-2*(_qxw*_qxw+_qzw*_qzw))*_dyw + (2*(_qyw*_qzw+_qxw*_qww))*_dzw
        _zcw = (2*(_qxw*_qzw+_qyw*_qww))*_dxw + (2*(_qyw*_qzw-_qxw*_qww))*_dyw + (1-2*(_qxw*_qxw+_qyw*_qyw))*_dzw
        _zsw = torch.where(_zcw.abs() < 1e-6, torch.full_like(_zcw, 1e-6), _zcw)
        _ACQ["pcx"] = ((_xcw / _zsw) * _Kw2[:, 0, 0] + _Kw2[:, 0, 2]) / _Ww2
        _ACQ["pcy"] = ((_ycw / _zsw) * _Kw2[:, 1, 1] + _Kw2[:, 1, 2]) / _Hw2
        _ACQ["pv"]  = (fd[:, 3] > 0.5) & (rz_n > RZ_MIN) & (_zcw > 0.02)             # front reliable + cube in front of the wrist
    except Exception:
        _ACQ["pcx"] = None; _ACQ["pcy"] = None; _ACQ["pv"] = None
    # [UNPROJ] diagnostic vs TRUE cube (sim GT only): error distribution AMONG the reliability-gated (kept) set
    _unproj_n = (globals().get("_unproj_n", -1) + 1)
    if (_unproj_n % 500 == 0 or _unproj_n < 6) and PRIV_RSR_TARGET > 0.0:   # user 2026-07-19: PRIV wiring check (also first 6 calls for fast boot confirmation) — col5 mean ~1.0 when privileged (cube_to_cam conf=vis=1), ~0.9 if real YOLO. phv_none=True means _PH_CUR not set -> privilege skipped.
        _fcol5 = float(fdf[fdf[:, 3] > 0.5, 5].mean()) if bool((fdf[:, 3] > 0.5).any()) else -1.0
        print(f"[PRIV-DBG] _pw={_pw:.2f} phv_none={_phv is None} PRIV_SEARCH={PRIV_SEARCH} sweeping={_SWEEPING[0]} fdf_conf_mean={_fcol5:.3f} (priv->~1.0 / real->~0.9)", flush=True)
    if _unproj_n % 500 == 0:
        ctrue = env.scene[args.target_cube].data.root_pos_w
        rr, rz_d = cube_xyz_from_front(fd[:, :2], ctrue[:, 2])
        seen = fd[:, 3] > 0.5; keep = seen & (rz_d > RZ_MIN)
        if bool(seen.any()):
            ek = (rr[keep, :2] - ctrue[keep, :2]).norm(dim=-1) if bool(keep.any()) else torch.zeros(1, device=DEVICE)
            kp = float(keep.float().sum()) / max(1.0, float(seen.float().sum())) * 100
            print(f"[UNPROJ] GATED err_xy med={float(ek.median()):.3f} p90={float(ek.quantile(0.9)):.3f} max={float(ek.max()):.3f} "
                  f"<5cm={float((ek<0.05).float().mean())*100:.0f}% | kept {kp:.0f}% of seen (rz>{RZ_MIN}) n_keep={int(keep.sum())}", flush=True)
        _cl = env.scene[args.target_cube].data.root_pos_w - env.scene.env_origins     # user 2026-07-10: front/wrist DETECTION RATE + front conf by cube FORWARD distance (does front actually catch it?)
        _cd = _cl[:, 1] + 1.15                                                          # forward distance from base (= env-local y + 1.15)
        _dm = []
        for _lo, _hi in [(0.0,0.15),(0.15,0.30),(0.30,0.45),(0.45,0.60),(0.60,0.90)]:
            _mb = (_cd >= _lo) & (_cd < _hi); _nb = int(_mb.sum())
            if _nb > 0:
                _fsel = _mb & (fdf[:, 3] > 0.5)
                _fcf = float(fdf[_fsel, 5].mean()) if bool(_fsel.any()) else 0.0
                _dm.append(f"{_lo:.2f}-{_hi:.2f}(n{_nb}):F{float((fdf[_mb,3]>0.5).float().mean()):.2f}/W{float((wdf[_mb,3]>0.5).float().mean()):.2f}/cf{_fcf:.2f}")
        print("[DET-DIST fwd] " + " | ".join(_dm) + "  (F=front detection rate W=wrist detection rate cf=front mean conf)", flush=True)
        _wcamp = env.scene["wrist"].data.pos_w                                       # user 2026-07-09: FRAC-vs-distance — does wrist frac grow as cube gets closer?
        _dtc = (ctrue - _wcamp).norm(dim=-1)                                         # true wrist-cam <-> cube dist
        _sw = wd[:, 3] > 0.5
        for _lo, _hi in [(0.0,0.15),(0.15,0.25),(0.25,0.4),(0.4,0.7),(0.7,9.0)]:
            _b = _sw & (_dtc >= _lo) & (_dtc < _hi)
            if bool(_b.any()): print(f"[FRACDBG] dist[{_lo:.2f},{_hi:.2f}) frac_med={float(wd[_b,2].median()):.4f} n={int(_b.sum())}", flush=True)
        rrw, rzw_d = cube_xyz_from_wrist(wd[:, :2], ctrue[:, 2])                     # user 2026-07-09: WRIST unproj vs TRUE cube — validate the wrist calc
        seenw = wd[:, 3] > 0.5; keepw = seenw & (rzw_d > RZ_MIN)
        if bool(seenw.any()):
            ekw = (rrw[keepw, :2] - ctrue[keepw, :2]).norm(dim=-1) if bool(keepw.any()) else torch.zeros(1, device=DEVICE)
            print(f"[UNPROJ-W] WRIST err_xy med={float(ekw.median()):.3f} p90={float(ekw.quantile(0.9)):.3f} max={float(ekw.max()):.3f} "
                  f"<5cm={float((ekw<0.05).float().mean())*100:.0f}% n_seen={int(seenw.sum())} n_keep={int(keepw.sum())}", flush=True)
            _steep = seenw & (rzw_d > 0.4)                                          # user 2026-07-09: wrist looking DOWN (steep = approach-like) -> unproj should be accurate here
            rrf, _ = cube_xyz_from_wrist(1.0 - wd[:, :2], ctrue[:, 2])               # hypothesis: 180deg image roll -> flip cx,cy
            rry, _ = cube_xyz_from_wrist(torch.stack([wd[:,0], 1.0-wd[:,1]], -1), ctrue[:, 2])  # flip cy only
            rrx, _ = cube_xyz_from_wrist(torch.stack([1.0-wd[:,0], wd[:,1]], -1), ctrue[:, 2])  # flip cx only
            if bool(_steep.any()):
                _m = _steep
                e0 = (rrw[_m, :2] - ctrue[_m, :2]).norm(dim=-1).median()
                ef = (rrf[_m, :2] - ctrue[_m, :2]).norm(dim=-1).median()
                ey = (rry[_m, :2] - ctrue[_m, :2]).norm(dim=-1).median()
                ex = (rrx[_m, :2] - ctrue[_m, :2]).norm(dim=-1).median()
                print(f"[UNPROJ-W-STEEP] n={int(_m.sum())} err(raw)={float(e0):.3f} err(flipXY)={float(ef):.3f} err(flipY)={float(ey):.3f} err(flipX)={float(ex):.3f}  <- smallest = fix", flush=True)
            else:
                print(f"[UNPROJ-W-STEEP] no steep(rz>0.4) wrist rays yet", flush=True)
            Cwd = env.scene["wrist"]; Cfd = env.scene["front"]
            _Kw = Cwd.data.intrinsic_matrices; _Kf = Cfd.data.intrinsic_matrices
            print(f"[CAMDBG] WRIST pos={Cwd.data.pos_w[0].tolist()} quat_ros={Cwd.data.quat_w_ros[0].tolist()} "
                  f"K[fx,fy,cx,cy]=[{float(_Kw[0,0,0]):.1f},{float(_Kw[0,1,1]):.1f},{float(_Kw[0,0,2]):.1f},{float(_Kw[0,1,2]):.1f}] HW={Cwd.data.output['rgb'].shape[1]}x{Cwd.data.output['rgb'].shape[2]}", flush=True)
            print(f"[CAMDBG] FRONT pos={Cfd.data.pos_w[0].tolist()} quat_ros={Cfd.data.quat_w_ros[0].tolist()} "
                  f"K[fx,fy,cx,cy]=[{float(_Kf[0,0,0]):.1f},{float(_Kf[0,1,1]):.1f},{float(_Kf[0,0,2]):.1f},{float(_Kf[0,1,2]):.1f}] HW={Cfd.data.output['rgb'].shape[1]}x{Cfd.data.output['rgb'].shape[2]}", flush=True)
    if _PKEEP_IDX:                                                   # WHITELIST: keep ONLY chosen proprio idx (e.g. d2r pan+wflex 15,18), zero the rest -> minimal orientation feedback
        st = st.clone(); _m = torch.zeros(21, device=st.device); _m[_PKEEP_IDX] = 1.0; st[:, :21] = st[:, :21] * _m
    elif MASK_PROPRIO:                                              # zero ENTIRE proprio block (0:21), keep cube_rel(21:) -> purely perception-driven policy, no proprio shortcut
        st = st.clone(); st[:, :21] = 0.0
    # user 2026-07-21: NaN GUARD — a degenerate unproj ray (rz->0 in cube_xyz_from_front/wrist) can emit inf/NaN in
    #   cube_rel -> propagates through the net -> actor loc NaN -> crash (official9 died @38.4k). Sanitize obs so no
    #   NaN/inf reaches the net. nan->0, inf-> large finite (no-op when clean). All obs already in bounded ranges.
    fd = torch.nan_to_num(fd, nan=0.0, posinf=0.0, neginf=0.0); wd = torch.nan_to_num(wd, nan=0.0, posinf=0.0, neginf=0.0)
    ori = torch.nan_to_num(ori, nan=0.0, posinf=0.0, neginf=0.0); st = torch.nan_to_num(st, nan=0.0, posinf=0.0, neginf=0.0)
    if FRONT_APP_OFF and _phv is not None:   # user 2026-07-22: WRIST-ONLY approach — zero the FRONT detection TOKEN for APPROACH/ALIGN (ph>=1). neutral=[0.5,0.5,0,0] (FRONT_DROP no-detect). SEARCH (ph==0) keeps front.
        _apm = (_phv >= 1).unsqueeze(-1).float()
        fd = fd * (1.0 - _apm) + _apm * torch.tensor([0.5, 0.5, 0.0, 0.0], device=fd.device, dtype=fd.dtype)
    if FRONT_OFF_WHEN_WRIST and _phv is not None:   # user 2026-07-30: in SEARCH, if the WRIST sees the cube, FRONT OFF (token fd + front cube_rel st[21:23]) -> force wrist-led centering. If the wrist is blind, keep front (acquisition). Parity with deploy.
        _wsm = ((_phv == 0) & (wd[:, 3] > 0.5)).unsqueeze(-1).float()
        fd = fd * (1.0 - _wsm) + _wsm * torch.tensor([0.5, 0.5, 0.0, 0.0], device=fd.device, dtype=fd.dtype)
        st = st.clone(); st[:, 21:23] = st[:, 21:23] * (1.0 - _wsm)   # front cube_rel = state dims 21,22 (line ~707 cat order)
    if _D_ACQ > 0:   # user 2026-08-01: FRONT-guided AIM vector [pcx,pcy,pv] (front cube projected into WRIST image) as the 5th net token. SEARCH-only (approach uses wrist). deploy MUST replicate: same clamp(-3,3)+pv gate + off-search/wrist-sees zeroing.
        if _ACQ.get("pv") is not None:
            _apv = _ACQ["pv"].float()
            _apcx = torch.nan_to_num(_ACQ["pcx"], nan=0.0, posinf=0.0, neginf=0.0).clamp(-3.0, 3.0) * _apv   # zero pcx/pcy when front unreliable (pv=0)
            _apcy = torch.nan_to_num(_ACQ["pcy"], nan=0.0, posinf=0.0, neginf=0.0).clamp(-3.0, 3.0) * _apv
            _acq3 = torch.stack([_apcx, _apcy, _apv], dim=-1)
        else:
            _acq3 = torch.zeros(st.shape[0], 3, device=st.device, dtype=st.dtype)
        if _phv is not None:   # SEARCH-only: zero the aim token off-search (ph>=1 uses wrist)
            _acq3 = torch.where((_phv == 0).unsqueeze(-1), _acq3, torch.zeros_like(_acq3))
        if FRONT_OFF_WHEN_WRIST and _phv is not None:   # if the wrist sees the cube, the aim token is also OFF (front-guide unnecessary; parity with the (1-nvis) gate in _rt_acq)
            _acq3 = _acq3 * (1.0 - ((_phv == 0) & (wd[:, 3] > 0.5)).unsqueeze(-1).float())
        st = torch.nan_to_num(st, nan=0.0, posinf=0.0, neginf=0.0)
        st = torch.cat([st, _acq3], dim=-1)
    return fd, wd, ori, st

# ---- reward/phase thresholds still referenced by the LIVE TWO_PHASE block (rest of the old
#      single-agent rule-executor + reward constants were DELETED as dead) ----
Z_APPROACH = float(os.environ.get("Z_APPROACH", "0.18"))   # APP->ALIGN: gripper FK z must be BELOW this (descended enough) — user 2026-07-01 env-tunable
APP_GZ     = float(os.environ.get("APP_GZ", str(Z_APPROACH)))   # user 2026-07-20: SEPARATE height gate for appr_ready (was hardcoded Z_APPROACH). Decoupled so the descend-reward ramp (Z_APPROACH) and the SUCCESS height gate can differ — needed when the DESC_FLOOR clamp stops the jaw ABOVE Z_APPROACH (gz<Z_APPROACH then unreachable). Default = Z_APPROACH (OURS unchanged). Set higher for OFFICIAL (clamp floor).
APP_MZ     = float(os.environ.get("APP_MZ", "0.0"))   # user 2026-07-20: >0 = gate appr_ready HEIGHT on the jaw-MOUTH world z (FK, = what DESC_FLOOR clamps) instead of EE-origin gz. More consistent with the clamp (gz vs mouth-z flips with wrist pitch). 0 = use gz<APP_GZ (OURS unchanged). OFFICIAL: set to DESC_FLOOR+~0.01.
SUCCESS_IN_SEARCH = int(os.environ.get("SUCCESS_IN_SEARCH", "0"))   # user 2026-07-24: 1 = RSR/success (_reached: center+mouth-height+frac) fires in SEARCH too, not only APPROACH. Phase-agnostic success (criteria met = success regardless of FSM phase label).
# user 2026-07-22: METRIC horizontal MOUTH-over-cube alignment (deployable: wrist-detection ray ∩ known table-plane − mouth FK; NOT privileged). Fixes the lateral drift that pixel-centering (ncen) can't (parallax: the "mouth-over-cube" pixel shifts with descent height). All 0/off = unchanged.
MOUTH_XY = float(os.environ.get("MOUTH_XY", "0.0"))   # >0 = _reached ALSO requires metric mouth<->cube horizontal < this (m), wrist-reliable. e.g. 0.04
APP_D3   = float(os.environ.get("APP_D3", "0.0"))     # user 2026-07-27: >0 = RSR success uses the DEPLOYABLE 3D distance (wrist-unproj cube(xy,plane_z) <-> jaw-MOUTH FK) < this (m) INSTEAD of the absolute mouth-z height gate ("actual distance" — relative, works regardless of cube height). 0 = legacy mouth-z height.
RSR_MXY  = float(os.environ.get("RSR_MXY", "0.0"))    # user 2026-07-27: >0 = rsr_old (privileged) ALSO requires TRUE horizontal xy dist < this (m) — "over the cube from above" ON TOP of the 3D RSR_DCUBE. 0 = 3D-only.
DESC_MXY = float(os.environ.get("DESC_MXY", "0.0"))   # >0 = IK descent gated on metric mouth<->cube horizontal < this (m) when wrist-reliable (else pixel gate). e.g. 0.06 (looser than MOUTH_XY)
# user 2026-07-22: ASYMMETRIC descent shaping — the descent-LEVEL reward (_d) rewards + down to the shelf (Z_APPROACH); BELOW the shelf this adds a growing − so the policy STOPS at the 5cm pre-grasp shelf instead of pushing into the hard floor (frac/dwell keep pulling down otherwise). Pair: keep DESC_FLOOR as a LOW safety backstop (below the shelf), NOT at the shelf (else this never fires).
APP_CEN    = float(os.environ.get("APP_CEN", "0.15"))       # APP->ALIGN: wrist centering error |cx,cy-0.5| must be BELOW this (cube under the gripper) — user 2026-07-01
SWEEP_APP_CEN = float(os.environ.get("SWEEP_APP_CEN", str(APP_CEN)))   # user 2026-07-26: APP_CEN used ONLY during the curriculum sweep (loose eval gate), training rollout keeps tight APP_CEN. sweep-only like deterministic.
CR_APP     = float(os.environ.get("CR_APP", "0.0"))         # user 2026-07-09: >0 = appr_ready ALSO requires (wrist+front cube_rel horizontal dist avg) < CR_APP (both cams valid) — "mouth actually OVER the cube" (deployable). Fixes ncen+gz being satisfied while landing BESIDE the cube. 0 = off.
Z_GRASP = 0.055        # descend target z (press to grasp height; table-relative)             [TUNE]
Z_TABLE  = float(os.environ.get("Z_TABLE", "0.035"))         # gz below this = gripper ramming into the table (= GRASP_Z_LO floor); graded penalty below it
W_TABLE  = float(os.environ.get("W_TABLE", "0.5"))           # table-penalty weight (Squint used -3*touching/4~-0.75); graded -W_TABLE at gz=0 -> 0 at gz>=Z_TABLE
W_TABLE_HARD = float(os.environ.get("W_TABLE_HARD", "5.0"))  # user 2026-07-01: STEEP quadratic penetration penalty ADDED below the table (peaks -W_TABLE_HARD at TABLE_PEN_REF deep); strongly discourages ramming without touching normal descent
TABLE_PEN_REF = float(os.environ.get("TABLE_PEN_REF", "0.03"))  # penetration depth (m) at which the quadratic penalty saturates to -W_TABLE_HARD
PEN_REF_CUBE = int(os.environ.get("PEN_REF_CUBE", "0"))   # user 2026-07-05: 1 = descent penalty reference = CUBE CENTER height (= table + cube/2), measured at the jaw MOUTH -> "descend only to half the cube" (penalty grows once the mouth dips below the cube's middle). 0 = old table-penetration ref (Z_TABLE, measured at gz/EE).
PEN_Z_OFF = float(os.environ.get("PEN_Z_OFF", "0.0"))     # bias (m) added to the cube-center reference when PEN_REF_CUBE=1 (+ = allow lower before penalty, − = stop higher)
PEN_REF_Z = float(os.environ.get("PEN_REF_Z", "0.0"))     # user 2026-07-05: >0 = penalty reference = this FIXED world z (m), measured at the jaw MOUTH. Overrides PEN_REF_CUBE. e.g. 0.04 = descend only to 4cm.
DESC_FLOOR = float(os.environ.get("DESC_FLOOR", "0.0"))   # user 2026-07-05: >0 = HARD descent floor — the jaw MOUTH (FK) cannot descend below this fixed world z (m). Clamps the IK descent TARGET, not a reward. REAL-DEPLOYABLE (fixed height in base frame + FK from encoders; NOT privileged). Same clamp must live in the deploy script. Replaces the table penalty.
FRAC_CAP   = float(os.environ.get("FRAC_CAP", "0.27"))    # B-guard (recalib 2026-06-26 -> 0.27, near wfmax~0.30): no + for GROWING past this (anti-ram); shrink still penalized (was a DEAD 0.50, unreachable)
FRAC_REF   = float(os.environ.get("FRAC_REF", "0.0"))     # user 2026-07-09: TWO-SIDED W_AF neutral. frac>FRAC_REF -> +expo, frac<FRAC_REF -> -expo (penalize sitting far/retreating). 0 = old one-sided (0..+expo). Set e.g. 0.15 to punish low frac ("sit back, only center, then descend" fix).
CEN_REF    = float(os.environ.get("CEN_REF", "0.0"))      # user 2026-07-09: TWO-SIDED approach-centering neutral (ncen). ncen<CEN_REF -> +expo(center), ncen>CEN_REF -> -expo(moving away). 0 = old one-sided _cen_a (tanh-expo, 0..+). Set e.g. 0.15.
# user 2026-07-09: close the "look away to escape the far cube_rel penalty" exploit. Instead of gating cube_rel to 0 when the cam
# can't reliably see the cube, PENALIZE not-seeing (per-step) + EXTRA penalize acquired-then-lost ("entered the frame then left again"). wrist + front.
# user 2026-07-10: FRONT-3D reward (deployable 3D dist front-unproj->mouth). Replaces the 2D front cube_rel (set W_CRF=0). front reliable FAR (arm not occluding) -> primary approach-distance shaping.
# user 2026-07-10: WRIST-3D reward — kept ONLY where wrist is accurate (near grasp: gz<CRW3_GZ AND steep _vw). fine grasp-alignment; front reliable far, wrist reliable near. no NOVIS (not-seeing wrist far is normal).
# user 2026-07-10: W_AC DISTANCE-SCALING — weaken wrist-image centering (ncen) NEAR the cube. eye-in-hand geom: as the gripper gets over+down the cube, cy goes off-center, so full W_AC FIGHTS the final approach (pulls back to re-center). far = full W_AC (track cube); near = weakened.
# user 2026-07-10: POTENTIAL/DELTA reward on FRONT-3D (deployable) — reward CLOSING (df3↓ → +), penalize backing off (df3↑ → −). per-step reason to keep approaching past the level-reward plateau. front valid both steps only (occlusion/reset → 0); clamped anti-spike.

# ---------------- C51-SAC networks ----------------
# The single-agent actor/critic/optimizers + legacy replay buffer (store/sample/update/zc)
# were DELETED: they were used only by the legacy single-agent main loop (removed). The live
# TWO_PHASE block builds its own per-phase agents (_mk_agent/_TPBuf/_upd) and reuses only
# DetActor/DetCriticC51 (det_xattn_nets) + project_target (tp_net). --init-from resume applied
# only to the legacy agents and is likewise unused by TWO_PHASE (which uses TP_RESUME/TP_EVAL).
cap = args.buffer_size   # kept solely for the [DET] start banner below

# ---------------- UI ----------------
# The single-policy dashboard (_H server, _UI/_HIST state, _push/_graph/update_ui, _EVAL,
# and the EVAL_STEP interactive loop) was DELETED: it drove the removed single-agent loop and
# its server was never started when TWO_PHASE=1 (all jobs set it). The TWO_PHASE block has its
# own UI (_TPUI/_TPHND/_tpg). The remaining helpers (_png/_spark/_card/_draw_det) are shared.
from http.server import BaseHTTPRequestHandler, HTTPServer
import imageio
from rebot_grasp_rl.dashboard_render import _png, _spark, _trimmean, _card, _draw_det, _quat_rpy, _draw_arrow   # REFACTOR 2026-07-27: pure UI helpers -> ui_render.py

# update_ui() and the MLflow logging block were DELETED: both were driven only by the removed
# single-agent loop (update_ui wrote the dead _UI dict; MLflow's _MLF was logged only there).

# ---------------- main loop ----------------
print(f"[DET] start num_envs={N} act={ACT_DIM} state={STATE_DIM} buffer={cap}@{BUF_DEV} UTD={args.num_updates}", flush=True)
fd, wd, ori, s = get_obs()
# ---- CALIBRATION (user 2026-06-28): place a cube AT the gripper gap, detect its (cx,cy,frac) = the centering + grasp-size target.
#   PURE DETECTION (no camera projection calib). eye-in-hand -> the gap is at a FIXED wrist-image spot, so one measurement holds
#   for all poses. Real-robot equivalent: put a cube in the gripper, run the detector, read (cx,cy,frac). ----
_GOFF = torch.tensor([float(os.environ.get("GOFF_X", "0.0")), float(os.environ.get("GOFF_Y", "0.0")), float(os.environ.get("GOFF_Z", "0.0"))], device=DEVICE)   # SNAPPED privileged grasp offset (gripper frame)
_CTGT = torch.tensor([float(os.environ.get("CTGT_X", "0.5")), float(os.environ.get("CTGT_Y", "0.5"))], device=DEVICE)   # centering target (SNAP cx,cy or fallback image center)
_FRAC_TGT = float(os.environ.get("FRAC_TGT", "0.30"))
_SNAP_SET = bool(os.environ.get("CTGT_X"))
if _SNAP_SET: print(f"[CALIB] SNAP targets: CTGT=({float(_CTGT[0]):.3f},{float(_CTGT[1]):.3f}) FRAC={_FRAC_TGT:.3f} GOFF={[round(float(x),4) for x in _GOFF.tolist()]}", flush=True)
try:
    if _SNAP_SET: raise StopIteration("snap-calibrated -> skip auto-calib")
    _cube = env.scene[args.target_cube]
    for _ in range(15):   # OPEN the gripper jaws first (grip=+1) so a cube can sit BETWEEN the open fingers (user 2026-06-28)
        _ao = torch.zeros(N, ACT_DIM, device=DEVICE); _ao[:, 6] = 1.0; env.step(_ao)
    _jof = torch.tensor([float(os.environ.get("JAW_OFF_X", str(profile.jaw_mouth_offset[0]))), float(os.environ.get("JAW_OFF_Y", str(profile.jaw_mouth_offset[1]))), float(os.environ.get("JAW_OFF_Z", str(profile.jaw_mouth_offset[2])))], device=DEVICE).expand(N, 3)
    _gapw = r.data.body_pos_w[:, ee_body_idx, :] + quat_apply(r.data.body_quat_w[:, ee_body_idx, :], _jof)   # JAW-MOUTH (between the gripper jaws), env-tunable JAW_OFF_X/Y/Z
    _opos = _cube.data.root_pos_w.clone(); _oquat = _cube.data.root_quat_w.clone()
    _cube.write_root_pose_to_sim(torch.cat([_gapw, _oquat], dim=-1))                        # move cube -> gap
    for _ in range(3): env.sim.render()
    _fc, _wc, _, _ = get_obs()
    _sc = _wc[:, 3] > 0.5
    if int(_sc.sum().item()) >= max(4, N // 8):
        _CTGT = _wc[_sc, 0:2].mean(0); _FRAC_TGT = float(_wc[_sc, 2].mean())
        print(f"[CALIB] cube@gripper-gap seen {int(_sc.sum().item())}/{N}: CENTER=({float(_CTGT[0]):.3f},{float(_CTGT[1]):.3f}) FRAC={_FRAC_TGT:.3f} -> targets", flush=True)
    else:
        print(f"[CALIB] FAIL: cube@gap seen only {int(_sc.sum().item())}/{N} -> fallback CENTER=(0.5,0.5) FRAC=0.30", flush=True)
    _cube.write_root_pose_to_sim(torch.cat([_opos, _oquat], dim=-1)); env.sim.render()      # restore cube
    fd, wd, ori, s = get_obs()
except Exception as _e:
    if not _SNAP_SET: print(f"[CALIB] EXCEPTION {_e} -> fallback CENTER=(0.5,0.5) FRAC=0.30", flush=True)
if os.environ.get("YOLO_PROBE"):   # user 2026-07-15: FORCED sweep — place cube ON the wrist optical axis at fixed distances, render, run YOLO. Isolates detector capability from policy orientation. Exits after.
    _cube = env.scene[args.target_cube]; _Cw = env.scene["wrist"]
    _cwp = _Cw.data.pos_w.clone(); _cq = _Cw.data.quat_w_ros.clone()                       # wrist cam world pos + ROS-optical quat (+Z forward)
    _fwd = quat_apply(_cq, torch.tensor([0.0, 0.0, 1.0], device=DEVICE).expand(N, 3))       # optical axis in world
    _oq = _cube.data.root_quat_w.clone()
    _zv = torch.zeros(N, 6, device=DEVICE)
    _hold = torch.zeros(N, ACT_DIM, device=DEVICE); _hold[:, 6] = 1.0                        # hold arm + open gripper
    for _ in range(10): env.step(_hold)                                                       # WARM the render pipeline (CALIB relies on prior env.step for render()-after-pin to reflect the new pose; SNAP-run skips CALIB so warm here)
    _cwp = _Cw.data.pos_w.clone(); _cq = _Cw.data.quat_w_ros.clone()                          # re-read cam pose AFTER warmup (arm may have settled)
    _fwd = quat_apply(_cq, torch.tensor([0.0, 0.0, 1.0], device=DEVICE).expand(N, 3))
    print(f"[YOLO-PROBE] cube forced ON wrist optical axis (N={N} envs, median over detected):", flush=True)
    for _d in [0.05, 0.10, 0.15, 0.20, 0.30]:
        for _k in range(4):   # re-pin + flush + re-CAPTURE sensor (render() alone leaves the camera buffer stale; sensor.update re-captures WITHOUT a physics step -> no gravity fall)
            _cube.write_root_pose_to_sim(torch.cat([_cwp + _d*_fwd, _oq], dim=-1))
            _cube.write_root_velocity_to_sim(_zv)
            env.sim.forward(); env.sim.render()
            for _s in ("wrist", "front"):
                try: env.scene[_s].update(0.0, force_recompute=True)
                except Exception:
                    try: env.scene[_s].update(0.0)
                    except Exception: pass
        _f2, _w2, _, _ = get_obs()
        _wv = _w2[:, 3] > 0.5; _fv = _f2[:, 3] > 0.5
        _wfr = float(_w2[_wv, 2].median()) if bool(_wv.any()) else 0.0
        _wcx = float(_w2[_wv, 0].median()) if bool(_wv.any()) else 0.0
        _wcy = float(_w2[_wv, 1].median()) if bool(_wv.any()) else 0.0
        print(f"[YOLO-PROBE] d={_d*100:>2.0f}cm | WRIST det={float(_wv.float().mean())*100:>3.0f}% frac={_wfr:.4f} cx,cy=({_wcx:.2f},{_wcy:.2f}) | FRONT det={float(_fv.float().mean())*100:>3.0f}% frac={float(_f2[_fv,2].median()) if bool(_fv.any()) else 0.0:.4f}", flush=True)
    # user 2026-07-20: BELOW-MOUTH sweep — pin the cube directly UNDER the jaw MOUTH at height d (= "jaw d above cube",
    #   the real centered-hover geometry) and read the REAL wrist frac. Answers "what frac at jaw 2cm above cube?".
    try:
        _eep = r.data.body_pos_w[:, ee_body_idx, :].clone(); _eeq = r.data.body_quat_w[:, ee_body_idx, :].clone()
        _mouth = _eep + quat_apply(_eeq, _JAW_OFF3.expand(N, 3))                                   # jaw mouth world (FK)
        print(f"[YOLO-PROBE] --- BELOW-MOUTH sweep (cube directly under jaw mouth at height d = jaw d above cube) ---", flush=True)
        for _d in [0.02, 0.04, 0.06, 0.08, 0.10]:
            _cpos = _mouth.clone(); _cpos[:, 2] = _mouth[:, 2] - _d                                # cube d below the mouth
            for _k in range(4):
                _cube.write_root_pose_to_sim(torch.cat([_cpos, _oq], dim=-1))
                _cube.write_root_velocity_to_sim(_zv)
                env.sim.forward(); env.sim.render()
                for _s in ("wrist", "front"):
                    try: env.scene[_s].update(0.0, force_recompute=True)
                    except Exception:
                        try: env.scene[_s].update(0.0)
                        except Exception: pass
            _f2, _w2, _, _ = get_obs()
            _wv = _w2[:, 3] > 0.5
            _wfr = float(_w2[_wv, 2].median()) if bool(_wv.any()) else 0.0
            _wcx = float(_w2[_wv, 0].median()) if bool(_wv.any()) else 0.0
            _wcy = float(_w2[_wv, 1].median()) if bool(_wv.any()) else 0.0
            print(f"[YOLO-PROBE] jaw {_d*100:>2.0f}cm above cube | WRIST det={float(_wv.float().mean())*100:>3.0f}% frac={_wfr:.4f} cx,cy=({_wcx:.2f},{_wcy:.2f})", flush=True)
            # user 2026-07-20: RAW OBB (pixels) + save the env0 wrist PNG so we can SEE what the detector boxes ("fills the whole frame but 0.0xx?")
            try:
                import imageio as _iio
                _wrgb = env.scene["wrist"].data.output["rgb"][..., :3]                      # (N,H,W,3) uint8
                _H0, _W0 = int(_wrgb.shape[1]), int(_wrgb.shape[2])
                _mdl = _load_yolo(); _rr = _mdl.predict(_wrgb.permute(0,3,1,2).float()/255.0, imgsz=_H0, conf=args.det_conf, max_det=1, verbose=False, device=DEVICE)
                for _e0 in range(min(3, N)):
                    _ob = _rr[_e0].obb
                    if _ob is not None and _ob.xywhr is not None and len(_ob.xywhr) > 0:
                        _xw = _ob.xywhr[0].tolist()
                        print(f"[YOLO-PROBE]   env{_e0} d={_d*100:.0f}cm RAW-OBB(px) cx={_xw[0]:.1f} cy={_xw[1]:.1f} w={_xw[2]:.1f} h={_xw[3]:.1f} | img {_W0}x{_H0} | w*h/(W*H)={_xw[2]*_xw[3]/(_W0*_H0):.4f}", flush=True)
                    else:
                        print(f"[YOLO-PROBE]   env{_e0} d={_d*100:.0f}cm RAW-OBB: none", flush=True)
                _iio.imwrite(os.path.expanduser(f"~/rebot_eval/wristprobe_{int(_d*100):02d}cm.png"), _wrgb[0].cpu().numpy())
            except Exception as _e2:
                print(f"[YOLO-PROBE]   raw-obb/png dump fail: {_e2}", flush=True)
    except Exception as _e:
        print(f"[YOLO-PROBE] below-mouth sweep FAIL: {_e}", flush=True)
    print("[YOLO-PROBE] done -> exit", flush=True)
    import sys as _sys; _sys.exit(0)
TWO_PHASE = int(os.environ.get("TWO_PHASE", "0"))   # user 2026-06-29: 8896 two-policy FSM. SEARCH (pan+wrist_flex -> center cube in wrist view) -> APPROACH (vertical descend, keep in view) -> lose -> back to SEARCH. one job, two agents.
if TWO_PHASE:
    print("[TP] ===== TWO-PHASE (search + approach) training =====", flush=True)
    _TPCAP   = int(os.environ.get("TP_BUF", "300000"))
    SEARCH_CENTER = float(os.environ.get("SEARCH_CENTER", "0.12"))   # |wrist centroid - 0.5| under this (+visible) = aligned
    SWEEP_CENTER = float(os.environ.get("SWEEP_CENTER", str(SEARCH_CENTER)))   # user 2026-07-26: SEARCH_CENTER used ONLY during the curriculum sweep (loose eval handoff gate); training rollout keeps tight SEARCH_CENTER. sweep-only like deterministic.
    # user 2026-07-07: centering VERTICAL retarget+weight (applies to BOTH search & approach ncen). The wrist cam is
    #   down-tilted(22deg)+offset -> cy=0.5 is NOT a reachable equilibrium, so the pitch servo can't zero-cross (diag:
    #   a1 stuck ~-0.7, ignores cy). Retarget cy to the REACHABLE cy* (CY_TGT) and weight the vertical axis (CY_W>1)
    #   so vertical centering gets a non-flat gradient. CY_TGT=0.5 & CY_W=1.0 -> identical to the old ncen.
    CY_TGT = float(os.environ.get("CY_TGT", "0.5"))
    CX_TGT = float(os.environ.get("CX_TGT", "0.5"))   # user 2026-07-20: HORIZONTAL centering target (was hardcoded 0.5). For the OFFICIAL wide-FOV wrist cam the grasp point (cube directly under the jaw) lands at cx≈0.07 (aim offset), NOT 0.5 — retarget so "centered" == "jaw over cube". Default 0.5 (OURS unchanged).
    CY_W   = float(os.environ.get("CY_W", "1.0"))
    SEARCH_STEP   = float(os.environ.get("SEARCH_STEP", "1.0"))     # user 2026-07-02: scale SEARCH pan/wflex action (<1 = smaller steps -> steadier centering, less ncen jitter)
    SEARCH_IK     = int(os.environ.get("SEARCH_IK", "0"))           # user 2026-07-03: SEARCH scans by ROTATING the camera boresight via pose-IK (EE POSITION HELD) instead of raw pan+wflex joint deltas. a_s=[d_yaw(world-Z), d_pitch(local-x)]. Needs DIRECT_JOINTS=0 (IK path); pairs with APP_ORI=cube for a seamless search->approach handoff. Action MEANING changes -> retrain SEARCH fresh (TP_FRESH_SEARCH=1).
    SEARCH_ROT    = float(os.environ.get("SEARCH_ROT", "0.10"))     # rad per unit search action for the IK boresight scan (yaw/pitch step)
    WSERVO_KX     = float(os.environ.get("WSERVO_KX", "1.0")); WSERVO_KY = float(os.environ.get("WSERVO_KY", "1.0"))   # user 2026-07-04: APP_ORI=wservo visual-servo gains (cx->yaw, cy->pitch) to keep cube centered in wrist view during descent
    WSERVO_SX     = float(os.environ.get("WSERVO_SX", "1.0")); WSERVO_SY = float(os.environ.get("WSERVO_SY", "1.0"))   # wservo signs (flip if wrist servos AWAY from center) — tune empirically in eval
    WSERVO_CLAMP  = float(os.environ.get("WSERVO_CLAMP", "0.0"))   # user 2026-07-05: >0 = CLAMP the wservo grasp-axis DOWN-pitch to this many deg below horizontal (can't crane the head lower). 8951 uses this instead of the W_WPITCH reward penalty (wrist is scripted, not a policy DOF).
    SEARCH_HOLD   = int(os.environ.get("SEARCH_HOLD", "5"))          # consecutive aligned steps -> hand off to APPROACH
    LOSE_TOL      = float(os.environ.get("LOSE_TOL", "0.30"))        # center beyond this (or invisible) = losing the cube
    LOSE_HOLD     = int(os.environ.get("LOSE_HOLD", "3"))            # consecutive lost steps -> back to SEARCH
    NO_RESRCH     = int(os.environ.get("NO_RESRCH", "0"))            # user 2026-07-05: 1 = SEARCH runs ONCE per episode (centers -> APPROACH), APPROACH NEVER returns to SEARCH (to_srch disabled). Approach OWNS re-centering; cube may vanish (that's OK). Removes search<->approach oscillation ("mixing"). ph resets to 0 only on episode reset (line ~1312).
    RESRCH_GONE   = int(os.environ.get("RESRCH_GONE", "0"))          # user 2026-07-24: under NO_RESRCH, STILL hand APPROACH->SEARCH when the cube is COMPLETELY lost (vanished from wrist view = `lost`) held GONE_HOLD frames. Off-center does NOT trigger (only total vanish) -> re-acquire without the off-center oscillation NO_RESRCH removed.
    GONE_HOLD     = int(os.environ.get("GONE_HOLD", "15"))           # consecutive vanished frames before RESRCH_GONE re-search (raise if close-range FOV-exit bails too eagerly)
    W_RESRCH      = float(os.environ.get("W_RESRCH", "0.0"))         # user 2026-07-24: ONE-TIME penalty at APPROACH->SEARCH re-handoff. Discourages DEPENDENCE on the re-search crutch (bail-to-search instead of keeping the cube during descent). Keep SMALL — larger than this and the policy stops bailing even when genuinely lost (blind dive). 0 = off.
    LOSE_CEN      = float(os.environ.get("LOSE_CEN", "9.0"))       # user 2026-07-03: APPROACH off-center threshold -> back to SEARCH. ncen>LOSE_CEN counts as lost (default 9=off). TIGHT (small) = re-search whenever not tightly centered.
    FRAC_GRASP    = float(os.environ.get("FRAC_GRASP", "0.20"))      # wrist area_frac above this (cube big = gripper descended close) + centered = approach success. REAL/detection, frame-independent (NO privileged z)
    W_SC = float(os.environ.get("W_SC", "3.0")); W_SV = float(os.environ.get("W_SV", "0.5"))   # search: centering / visibility weights (W_SC up 1->3 for stronger centering, user 2026-06-30)
    W_ACQ_F = float(os.environ.get("W_ACQ_F", "0.0"))   # user 2026-07-19: FRONT-GUIDED search acquisition weight (0=off). When the WRIST can't see the cube (nvis<0.5) but FRONT can, reward reducing the FRONT-predicted wrist-centering err -> the policy swings the boresight to bring the cube INTO the wrist view. Same tanh-inside-expo shape/peak as W_SC centering (K_CE). Fixes the fresh-search acquisition stall (search reward was wrist-only -> no gradient to aim when blind).
    ACQ_K   = float(os.environ.get("ACQ_K", "1.5"))
    W_UP_SRCH   = float(os.environ.get("W_UP_SRCH", "0.0"))   # user 2026-08-01: SEARCH "head raised up" penalty — − when nose-direction z exceeds UP_SRCH_SIN, GRADUATED (quadratic: the higher it is raised, the larger). Allowed up to a certain threshold.
    UP_SRCH_SIN = float(os.environ.get("UP_SRCH_SIN", "0.17"))  # nose z (=sin(up angle)) threshold (~10° up). Penalty starts above this.     # user 2026-07-19: rolloff sharpness of the front-guided acquisition expo (tanh(ACQ_K·pncen)). LOWER than CEN_K(2.5) on purpose -> WIDER capture zone: an off-frame cube (pncen up to ~2 frame-widths) still gets a pull toward the wrist view, not just near-center.
    CEN_K = float(os.environ.get("CEN_K", "3.0")); CEN_K_FINE = float(os.environ.get("CEN_K_FINE", "15.0"))   # centering reward sharpness: coarse pull from far + fine sharpen near center
    W_CE = float(os.environ.get("W_CE", "0.0")); K_CE = float(os.environ.get("K_CE", "8.0"))   # user 2026-07-03: combined-centering shape knobs (K_CE=peak height/sharpness, CEN_K=rolloff). SEARCH cen_r.
    K_CE_A = float(os.environ.get("K_CE_A", str(K_CE))); CEN_K_A = float(os.environ.get("CEN_K_A", str(CEN_K)))   # user 2026-07-03: APPROACH-specific centering sharpness (own exp). Default = search values; set K_CE_A high (e.g. 2.0) to make approach spike MUCH harder near dead-center (grasp precision).
    SEARCH_BONUS = float(os.environ.get("SEARCH_BONUS", "5.0"))
    # --- anti-stall descent (user 2026-07-02): ABOVE HOVER_Z the gripper MUST keep descending each step; staying put OR rising = penalty. Below HOVER_Z pressure stops (centering/align settles). Chosen over a descend-delta term. ---
    W_SMOOTH = float(os.environ.get("W_SMOOTH", "0.4"))     # user 2026-07-16: ACTION-RATE (smoothness) penalty = W_SMOOTH·mean(|a_t − a_{t−1}|) per active env. removes jitter -> smoother motion (deploy-relevant: jerky=risky/inaccurate on the real robot). gated by same-phase-both-steps (no phase-transition spike).
    # user 2026-07-24: GRASP-POSE mouth-height TWO-SIDED peak — max at (cube-center-z + MH_ABOVE), decreasing to − as the jaw-MOUTH deviates (above OR below). SAME shape/scale as _rt_cen/_rt_frac (peak = W_MH*(exp(MH_K)-1)); pair with W_DEPTH=0 (replaces the monotonic descent pull). mouth-z = FK (deployable); cube-z = sim-true (training shaping).
    W_AF = float(os.environ.get("W_AF", "10.0"))   # approach: FRAC-growth weight (user 2026-06-30: reward cube getting BIGGER+centered directly, not gz-descend)
    EXP_KZ = float(os.environ.get("EXP_KZ", "1.5"))   # descend-rate EXP gain (2026-07-01): per-cm descent reward × e^(EXP_KZ·depth_frac) -> pays exponentially more when LOW
    EXP_KF = float(os.environ.get("EXP_KF", "1.0"))   # frac closeness EXP (2026-07-01): W_AF·(exp(EXP_KF·frac/FRAC_CAP)−1) -> persistent LEVEL, exp-grows as cube fills
    W_DEPTH = float(os.environ.get("W_DEPTH", "1.5"))  # descend LEVEL weight (persistent exp: lower gz = more +)
    DESC_GATE = int(os.environ.get("DESC_GATE", "1"))  # user 2026-07-09: 1 = descent reward GATED by centering-sign (_csign, old). 0 = PLAIN descent (decoupled) — use W_AC for centering separately. Removes the "center-far-then-farm-gated-descent = sitting back" exploit.
    DGATE_K = float(os.environ.get("DGATE_K", "0.0"))  # user 2026-07-04: depth *= exp(-DGATE_K*ncen) (smooth centering gate). 0=off.
    DGATE_CEN = float(os.environ.get("DGATE_CEN", "0.0"))  # user 2026-07-04: HARD gate — depth reward = 0 when ncen >= DGATE_CEN (off-center = NO depth reward). 0=off (use smooth DGATE_K instead).
    DESC_D = float(os.environ.get("DESC_D", "0.0"))   # user 2026-07-10: >0 = descent gate = front HORIZONTAL proximity(_df) AND centering(ncen), product. descend+ only when OVER cube (_df<DESC_D, front-valid) AND centered (ncen<DGATE_CEN). fixes sit-back (aim from afar + descend). 0 = legacy _csign (ncen only).
    W_MR    = float(os.environ.get("W_MR", "10.0"))    # ALIGN mouth DELTA weight (closer +, AWAY −)
    RSR_DCUBE = float(os.environ.get("RSR_DCUBE", "0.0"))   # user 2026-07-09: >0 = RSR success METRIC = did TRUE mouth<->cube dist (dcube_min) drop below this (privileged, MONITOR-ONLY — not reward/termination). Honest "did it actually reach" vs the deployable FSM appr_ready. 0 = legacy (succ_lat FSM).
    RSR_HOLD  = int(os.environ.get("RSR_HOLD", "1"))        # user 2026-07-14: RSR success requires TRUE dcube<RSR_DCUBE HELD this many CONSECUTIVE frames (@10Hz). 1 = instantaneous (legacy); 5 = must dwell 0.5s within reach (real grasp).
    APP_DCUBE = float(os.environ.get("APP_DCUBE", "0.0"))   # user 2026-07-10: >0 = APPROACH success (appr_ready) requires TRUE 3D mouth<->cube dist < this (PRIVILEGED). 0 = off. (superseded by APP_CR3 — appr_ready affects training/termination so it should be DEPLOYABLE, not privileged.)
    APP_CR3   = float(os.environ.get("APP_CR3", "0.0"))     # user 2026-07-10: >0 = appr_ready requires DEPLOYABLE 3D dist (front+wrist unproj avg to jaw mouth) < this, both cams valid. 0 = off. This is the real appr_ready gate (training success == what the REAL robot can measure). e.g. 0.10.
    # user 2026-07-14: DENSE DWELL reward — reward for STAYING within DWELL_D of the cube, growing EXPONENTIALLY with consecutive frames (nonzero from frame 1, ref DWELL_REF, capped at DWELL_CAP). DEPLOYABLE (uses _df3/_dw3 camera 3D dist like appr_ready — NOT privileged dcube). 0=off.
    # user 2026-07-14: ADAPTIVE CURRICULUM (hard-example mining) — periodically self-eval a grid over the spawn workspace, then UP-WEIGHT the spawn frequency of FAILING cells so training concentrates where it's weak. CURRIC=0 -> fully off (identical to legacy). Self-contained in the training job (no external eval needed).
    CURRIC       = int(os.environ.get("CURRIC", "0"))
    CURRIC_EVERY = int(os.environ.get("CURRIC_EVERY", "600000"))   # env-steps between self-eval sweeps
    CURRIC_FIRST = int(os.environ.get("CURRIC_FIRST", str(CURRIC_EVERY)))   # user 2026-07-18: first sweep at this step, then every CURRIC_EVERY after
    CURRIC_SWEEP0 = int(os.environ.get("CURRIC_SWEEP0", "0"))       # user 2026-07-16: 1 = run a self-eval sweep IMMEDIATELY at start (step 0) to seed the spawn map from the (resumed) ckpt's actual per-cell RSR, instead of starting uniform. Use on RESUME so the failing cells are up-weighted from the first training step.
    CURRIC_EXT = int(os.environ.get("CURRIC_EXT", "0"))            # user 2026-07-23: OFFLOADED sweep — .2 CONSUMER: run NO internal sweep (never pauses training); instead POLL CURRIC_WRITE for _cw computed by a .4 worker. graceful: if the file never updates, keeps the last _cw.
    CURRIC_ONESHOT = int(os.environ.get("CURRIC_ONESHOT", "0"))    # user 2026-07-23: .4 PRODUCER: os._exit(0) right after the first sweep writes _cw (one-shot sweep runner; Mac daemon re-launches per cycle).
    CURRIC_WRITE = os.environ.get("CURRIC_WRITE", "")             # path: PRODUCER np.saves _cw here after each sweep; CONSUMER polls this path for a newer _cw.
    CURRIC_NX    = int(os.environ.get("CURRIC_NX", "80"))          # grid lateral cells (X)
    CURRIC_NY    = int(os.environ.get("CURRIC_NY", "80"))          # grid forward cells (fwd)  -> 80x80 = 6400
    CURRIC_BOOST = float(os.environ.get("CURRIC_BOOST", "2.5"))    # spawn-weight multiplier for a fully-failing cell (moderate 2~3x)
    CURRIC_FLOOR = float(os.environ.get("CURRIC_FLOOR", "0.0"))    # user 2026-07-28: HARD FLOOR — reserve this FRACTION of the spawn budget for UNIFORM coverage (every cell guaranteed >= CURRIC_FLOOR/_NCELL) so PASSING cells are never starved -> no catastrophic forgetting. remaining (1-FLOOR) concentrates on failing cells (BOOST-shaped). 0 = legacy pure fail-weight (soft floor).
    CURRIC_BY_SEARCH = int(os.environ.get("CURRIC_BY_SEARCH", "0"))   # user 2026-07-24: 1 = curriculum fail-weight uses SEARCH-fail (cell didn't reach APPROACH in the sweep) instead of RSR-fail -> concentrate spawn on cells where SEARCH can't hand off (search-FIRST curriculum for early training; drives search->100% -> then approach)
    CURRIC_STAGED = int(os.environ.get("CURRIC_STAGED", "0"))   # user 2026-07-24: 1 = PER-CELL staged: while a cell's SEARCH mastery < CURRIC_SW_TH -> weight by SEARCH-fail; once SEARCH>=CURRIC_SW_TH in that cell -> switch that cell's spawn weight to RSR-fail (approach). "once search hits 100, switch that cell's spawn criterion to approach rsr"
    CURRIC_SW_TH = float(os.environ.get("CURRIC_SW_TH", "0.99"))  # per-cell SEARCH-mastery threshold to flip that cell from SEARCH-fail to RSR-fail weighting
    CURRIC_STAGE_TH = float(os.environ.get("CURRIC_STAGE_TH", "0.99"))  # user 2026-07-26: OVERALL sweep SEARCH% threshold to flip the WHOLE curriculum SEARCH-stage -> RSR-stage (was all-cells>=CURRIC_SW_TH). e.g. 0.90 = start approach eval+RSR-weighting once overall search reaches 90%.
    CURRIC_BLUR  = int(os.environ.get("CURRIC_BLUR", "2"))         # box-blur radius (cells) on the fail map (smooth per-cell noise into regions)
    W_NOACQ = float(os.environ.get("W_NOACQ", "1.0"))   # user 2026-07-03: NOT-acquired PENALTY — each step the cube is NOT visible in wrist -> -W_NOACQ. Replaces the acq bonus (③) + sweep (④): "penalty if not found" pressures find-and-keep instead of rewarding the find.
    W_NOVIS_A = float(os.environ.get("W_NOVIS_A", "0.0"))   # user 2026-07-05: APPROACH not-visible PENALTY — cube lost from wrist view -> -W_NOVIS_A/step. Needed under NO_RESRCH (approach never bails to SEARCH) so the policy is pushed to KEEP/re-acquire the cube instead of coasting at r_a≈0.
    Z_HIGH_REW = float(os.environ.get("Z_HIGH_REW", "0.35"))   # top of depth-reward ramp (hover altitude); ramps 0->1 down to Z_GRASP (0.055)
    W_REG = float(os.environ.get("W_REG", "10.0"))   # REGRESSION penalty (user 2026-06-30): got close then moved away -> negative. SEARCH: ncen rose above best; APPROACH: frac dropped below best
    APP_BONUS = float(os.environ.get("APP_BONUS", "10.0"))
    BONUS_BY_OLD = int(os.environ.get("BONUS_BY_OLD", "0"))  # user 2026-07-27: 1 = APP_BONUS also fires on succ_old (PRIVILEGED true-reach rsr_old, _dcube<RSR_DCUBE). Credits genuine reaches when deployable succ fires ~0%. Privileged REWARD only (obs stays deployable, NO_PRIV=1) -> DEPLOY-SAFE (reward is training-only). 0 = deployable succ only.
    BONUS_RSR     = float(os.environ.get("BONUS_RSR", "1.0"))                                  # user 2026-07-28: WEIGHT (×APP_BONUS) for deployable-rsr success (succ|to_align).
    BONUS_RSR_OLD = float(os.environ.get("BONUS_RSR_OLD", "1.0" if BONUS_BY_OLD else "0.0"))    # user 2026-07-28: WEIGHT (×APP_BONUS) for PRIVILEGED true-reach success (succ_old). Both can be on -> weighted combo (e.g. rsr 1.5 / rsr_old 0.5). Defaults preserve BONUS_BY_OLD behavior.
    # ---- 3rd phase ALIGN (user 2026-06-30): search -> approach -> ALIGN (cube into jaw-mouth) -> in-mouth ready ----
    THREE_PHASE = int(os.environ.get("THREE_PHASE", "0"))        # enable ALIGN as the 3rd phase (det_TP19d stays 2-phase)
    ALIGN_FRAC  = float(os.environ.get("ALIGN_FRAC", "0.15"))    # APPROACH->ALIGN: cube area_frac above this (+centered, held) hands off to ALIGN
    APP_HOLD    = int(os.environ.get("APP_HOLD", "3"))           # consecutive (close+centered) steps before APPROACH->ALIGN
    SUCC_HOLD   = int(os.environ.get("SUCC_HOLD", "3"))          # user 2026-07-02: 2-phase RSR = hold (centered+descended) this many frames = stable grasp
    W_ML = float(os.environ.get("W_ML", "1.5"))                  # ALIGN: mouth-align centering (2026-07-01: 3->1.5, parallel to W_AC — was farmable by hovering at the mouth)
    W_YW = float(os.environ.get("W_YW", "1.0"))                  # ALIGN: yaw align (|sin cube_yaw| -> 0 via wrist_roll)
    MOUTH_OK = float(os.environ.get("MOUTH_OK", "0.08"))         # in-mouth: ||cube - _CTGT|| below
    YAW_OK   = float(os.environ.get("YAW_OK", "0.30"))           # in-mouth: yaw_abs below
    YAW_SYM  = float(os.environ.get("YAW_SYM", "2.0"))           # user 2026-07-08: cube rotational-symmetry fold for ALIGN yaw. yaw_err=|sin(YAW_SYM·yaw)|. 2=4-fold(SQUARE cube: 0/90/180/270 all aligned=mod90°); 1=2-fold(rect:0/180=mod180°, old behavior). Square red cube -> 2.
    ALIGN_GZ = float(os.environ.get("ALIGN_GZ", "0.12"))         # in-mouth SUCCESS (user 2026-07-02): gripper FK z must be BELOW this (descended to grasp region; Z_GRASP=0.055)
    WROLL_I  = profile.idx["WROLL"]                              # wrist_roll joint index (align yaw control)
    # --- RULE grasp+lift (user 2026-07-02): on ALIGN in-mouth success, RULE (not learned) closes the gripper, holds, then lifts. Deploy-matched (run_tp3 GRASP/LIFT). ---
    RULE_GL   = int(os.environ.get("RULE_GL", "1"))              # 1 = after ALIGN success, rule GRASP (close+hold) -> LIFT (raise z). 0 = old freeze-on-success.
    RGL_HOLD  = int(os.environ.get("RGL_HOLD", "5"))             # frames to hold the gripper CLOSED before lifting
    RGL_LIFTZ = float(os.environ.get("RGL_LIFTZ", "0.12"))       # lift the EE this many meters above the grasp point
    RGL_STEP  = float(os.environ.get("RGL_STEP", "0.01"))        # lift z increment per step (m) — gentle raise via IK
    RGL_SUCC  = float(os.environ.get("RGL_SUCC", "0.05"))        # LIFT success = cube_z rose this many meters from the grasp start (real pick)
    PAN_I, WFLEX_I, ELBOW_I = profile.idx["PAN"], profile.idx["WFLEX"], profile.idx["ELBOW"]   # joint_ids indices: pan, wrist_flex, elbow
    SEARCH_ELBOW = int(os.environ.get("SEARCH_ELBOW", "0"))    # user 2026-06-30: DROP elbow -> 2-DOF search (pan+wrist_flex). =1 re-enables 3-DOF.
    WFLEX_GAIN   = float(os.environ.get("WFLEX_GAIN", "1.0"))  # search wrist_flex authority: >1 lets wrist PITCH DOWN faster/further per step (user: wrist barely lowers at inference)
    EVAL_STEP_SCALE = float(os.environ.get("EVAL_STEP_SCALE", "1.0"))  # TP_EVAL only: shrink per-STEP motion for finer step-through (<1 = smaller steps; e.g. 0.25 = 4x finer)
    EVAL_FREEZE_SUCC = float(os.environ.get("EVAL_FREEZE_SUCC", "0"))  # user 2026-07-26: TP_EVAL only — on RSR success, HOLD the arm at the success pose for this many SECONDS (frozen via succ_lat act_env=0), THEN reset+advance. e.g. 3 = 3s pause to inspect the grasp. 0 = end immediately (bench behavior).
    EVAL_SUCC_BY_OLD = int(os.environ.get("EVAL_SUCC_BY_OLD", "0"))  # user 2026-07-27: TP_EVAL freeze/advance keys off PRIVILEGED rsr_old (true _dcube<RSR_DCUBE held RSR_HOLD) instead of deployable RSR (_reached) — inspect TRUE reaches only (deployable RSR can false-fire on a fresh policy). UI still shows both metrics.
    APP_SERVO    = int(os.environ.get("APP_SERVO", "1"))       # approach: 1=3-DOF dx,dy,descend servo; 0=1-DOF descend-only (overnight ckpt arch)
    N_ACT_S = 3 if SEARCH_ELBOW else 2
    APP_NO_YAW = int(os.environ.get("APP_NO_YAW", "0"))   # user 2026-07-18: 1 = APPROACH drops YAW action + locks wrist_yaw/roll -> 4-DOF [dx,dy,descend,pitch], 4 joints [pan,lift,elbow,wrist_flex]
    N_ACT_A = 6 if DIRECT_JOINTS else ((4 if APP_NO_YAW else 5) if (APP_SERVO and APP_ORI == "policy") else (3 if APP_SERVO else 1))   # DIRECT_JOINTS: 6 raw joints; APP_ORI=policy: 5 (dx,dy,descend,pitch,yaw) — policy learns orientation too (user 2026-07-04 b); else 3 (dx,dy,descend, IK-position + scripted orientation)
    class _TPBuf:
        def __init__(self, cap, na):
            self.cap = cap; self.ptr = 0; self.full = False
            z = lambda *sh: torch.zeros(*sh, device=BUF_DEV)
            self.fd=z(cap,D_DET); self.wd=z(cap,D_DET); self.ori=z(cap,D_ORI); self.s=z(cap,STATE_DIM)
            self.a=z(cap,na); self.r=z(cap,1); self.nfd=z(cap,D_DET); self.nwd=z(cap,D_DET); self.nori=z(cap,D_ORI); self.ns=z(cap,STATE_DIM); self.d=z(cap,1)
        def store(self, fd,wd,ori,s,a,r,nfd,nwd,nori,ns,d):
            n = fd.shape[0]
            if n == 0: return
            idx = (torch.arange(n, device=BUF_DEV) + self.ptr) % self.cap
            self.fd[idx]=fd.to(BUF_DEV); self.wd[idx]=wd.to(BUF_DEV); self.ori[idx]=ori.to(BUF_DEV); self.s[idx]=s.to(BUF_DEV)
            self.a[idx]=a.to(BUF_DEV); self.r[idx]=r.to(BUF_DEV); self.nfd[idx]=nfd.to(BUF_DEV); self.nwd[idx]=nwd.to(BUF_DEV); self.nori[idx]=nori.to(BUF_DEV); self.ns[idx]=ns.to(BUF_DEV); self.d[idx]=d.to(BUF_DEV)
            self.ptr=(self.ptr+n)%self.cap; self.full=self.full or (self.ptr<n)
        def size(self): return self.cap if self.full else self.ptr
        def sample(self, bs):
            hi = self.size(); idx = torch.randint(0, hi, (bs,), device=BUF_DEV); g = lambda B: B[idx].to(DEVICE)
            return (g(self.fd),g(self.wd),g(self.ori),g(self.s),g(self.a),g(self.r),g(self.nfd),g(self.nwd),g(self.nori),g(self.ns),g(self.d))
    def _mk_agent(na):
        ac = DetActor(STATE_DIM, na).to(DEVICE)
        cr = DetCriticC51(STATE_DIM, na, args.num_atoms, args.v_min, args.v_max, args.num_q).to(DEVICE)
        ct = DetCriticC51(STATE_DIM, na, args.num_atoms, args.v_min, args.v_max, args.num_q).to(DEVICE); ct.load_state_dict(cr.state_dict())
        if _COMPILE:   # user 2026-07-28: torch.compile the CRITIC only (cr+ct). Critic runs ONLY in _upd at fixed batch 512 (logits/q_values/project_target) -> no recompile, and C51(num_q x atoms) is the heaviest update compute. Actor stays eager (rollout calls it with VARIABLE per-phase batch -> would recompile-thrash). dynamic=False (static batch). Compiling .forward covers all methods that route through it. COMPILE=0 to disable.
            cr.forward = torch.compile(cr.forward, dynamic=False)
            ct.forward = torch.compile(ct.forward, dynamic=False)
            print(f"[COMPILE] critic cr+ct forward compiled (dynamic=False) na={na}", flush=True)
        la = torch.zeros(1, device=DEVICE, requires_grad=True)
        return {"ac":ac, "cr":cr, "ct":ct, "na":na, "te":-float(na),
                "aopt":torch.optim.Adam(ac.parameters(), lr=args.lr), "copt":torch.optim.Adam(cr.parameters(), lr=args.lr),
                "la":la, "alopt":torch.optim.Adam([la], lr=(args.alpha_lr if args.alpha_lr>0 else args.lr)),
                "alpha":la.exp().clamp(min=args.alpha_floor).detach(), "buf":_TPBuf(_TPCAP, na)}
    def _upd(ag):
        fd_,wd_,ori_,s_,a_,r_,nfd_,nwd_,nori_,ns_,d_ = ag["buf"].sample(args.batch_size)
        with torch.no_grad():
            na_, nlp_ = ag["ac"].get_action(nfd_,nwd_,nori_,ns_); boot=(1.0-d_.squeeze(-1))
            rent_ = r_.squeeze(-1) - boot*args.gamma*(ag["alpha"].squeeze()*nlp_.squeeze(-1))
            td_ = project_target(ag["ct"], nfd_,nwd_,nori_,ns_, na_, rent_, boot, args.gamma)
        lg_ = ag["cr"].logits(fd_,wd_,ori_,s_,a_); lp_ = F.log_softmax(lg_, dim=-1)
        cl_ = (-(td_*lp_).sum(-1).mean(-1)).sum()
        ag["copt"].zero_grad(set_to_none=True); cl_.backward(); torch.nn.utils.clip_grad_norm_(ag["cr"].parameters(),10.0); ag["copt"].step()
        api_, lpi_ = ag["ac"].get_action(fd_,wd_,ori_,s_); qv_ = ag["cr"].q_values(fd_,wd_,ori_,s_,api_).mean(0)
        al_ = (ag["alpha"].squeeze()*lpi_.squeeze(-1) - qv_).mean()
        ag["aopt"].zero_grad(set_to_none=True); al_.backward(); torch.nn.utils.clip_grad_norm_(ag["ac"].parameters(),10.0); ag["aopt"].step()
        with torch.no_grad(): _, lp2_ = ag["ac"].get_action(fd_,wd_,ori_,s_)
        ale_ = -(ag["la"].exp()*(lp2_+ag["te"])).mean(); ag["alopt"].zero_grad(set_to_none=True); ale_.backward(); ag["alopt"].step()
        ag["alpha"] = ag["la"].exp().clamp(min=args.alpha_floor).detach()
        for tp,pp in zip(ag["ct"].parameters(), ag["cr"].parameters()): tp.data.mul_(1-args.tau).add_(args.tau*pp.data)
    srch = _mk_agent(N_ACT_S); appr = _mk_agent(N_ACT_A)   # search: pan+wrist_flex(+elbow if SEARCH_ELBOW). approach: dx,dy,descend servo (APP_SERVO=1) or descend-only (=0, overnight arch)
    algn = _mk_agent(6 if (DIRECT_JOINTS or ALIGN_DIRECT) else 4) if THREE_PHASE else None   # ALIGN: DIRECT(_JOINTS or ALIGN_DIRECT) -> 6 raw joint-deltas (incl roll); else dx,dy,roll,descend (4-DOF) via IK
    import time as _tm_eval
    TP_EVAL = int(os.environ.get("TP_EVAL", "0"))     # step-through eval — load ckpts, set cube, STEP the FSM one step at a time (deterministic, NO training)
    EVAL_AUTOPLAY = int(os.environ.get("EVAL_AUTOPLAY", "0"))  # user 2026-07-14: TP_EVAL auto-run (play=True at start) — for the continuous EVAL_SET benchmark (no manual UI stepping)
    TP_RESUME = int(os.environ.get("TP_RESUME", "0"))  # user 2026-06-30: warm-start both agents from search/approach ckpts then KEEP TRAINING (don't lose progress on restart)
    if TP_EVAL or TP_RESUME:
        _ed = os.environ.get("TP_EVAL_DIR", args.save_dir)
        _tag = "EVAL" if TP_EVAL else "RESUME"
        _fresh_search   = int(os.environ.get("TP_FRESH_SEARCH", "0"))     # relearn SEARCH from scratch
        _fresh_approach = int(os.environ.get("TP_FRESH_APPROACH", "0"))   # user 2026-07-02: APPROACH reward CHANGED -> relearn from scratch; resume only search(+align)
        _fresh_align    = int(os.environ.get("TP_FRESH_ALIGN", "0"))      # relearn ALIGN from scratch
        _resume_set = [] if _fresh_search else [(srch, "search_last.pt")]
        if not _fresh_approach: _resume_set.append((appr, "approach_last.pt"))
        if THREE_PHASE and algn is not None and not _fresh_align: _resume_set.append((algn, "align_last.pt"))   # user 2026-07-01: resume ALIGN too (unless TP_FRESH_ALIGN)
        if _fresh_search:   print("[TP-RESUME] SEARCH = FRESH (reward changed)", flush=True)
        if _fresh_approach: print("[TP-RESUME] APPROACH = FRESH (reward changed)", flush=True)
        if _fresh_align:    print("[TP-RESUME] ALIGN = FRESH (reward changed)", flush=True)
        for _ag, _nm in _resume_set:
            _pp = os.path.join(_ed, _nm)
            if os.path.isfile(_pp):
                try:
                    _ck = torch.load(_pp, map_location=DEVICE); _ag["ac"].load_state_dict(_ck["actor"])
                    if "critic" in _ck:   # resume also restores critic + target (eval ignores them but harmless)
                        _ag["cr"].load_state_dict(_ck["critic"]); _ag["ct"].load_state_dict(_ck["critic"])
                    print(f"[TP-{_tag}] loaded {_nm} (step {_ck.get('step','?')}) from {_ed}", flush=True)
                except Exception as _le:
                    print(f"[TP-{_tag}] {_nm} SKIPPED (arch changed -> fresh): {_le}", flush=True)
            else:
                print(f"[TP-{_tag}] !!! MISSING {_pp} -> random policy", flush=True)
        if TP_EVAL:
            print("[TP-EVAL] step-through mode: set cube + STEP. deterministic, no training.", flush=True)
        elif not _fresh_search:
            _ra = max(args.alpha_floor, RESUME_ALPHA) if RESUME_ALPHA > 0 else max(args.alpha_floor, 0.12)   # user 2026-07-05: RESUME_ALPHA bumps exploration on resume (dynamics change)
            with torch.no_grad():   # resumed = TRAINED -> modest alpha (don't re-explore from 1.0 & lose skill)
                srch["la"].copy_(torch.tensor([math.log(_ra)], device=DEVICE, dtype=srch["la"].dtype))
                if RESUME_ALPHA > 0 and not _fresh_approach: appr["la"].copy_(torch.tensor([math.log(_ra)], device=DEVICE, dtype=appr["la"].dtype))
                if RESUME_ALPHA > 0 and THREE_PHASE and (algn is not None) and not _fresh_align: algn["la"].copy_(torch.tensor([math.log(_ra)], device=DEVICE, dtype=algn["la"].dtype))
            srch["alpha"] = srch["la"].exp().clamp(min=args.alpha_floor).detach()
            if RESUME_ALPHA > 0 and not _fresh_approach: appr["alpha"] = appr["la"].exp().clamp(min=args.alpha_floor).detach()
            if RESUME_ALPHA > 0 and THREE_PHASE and (algn is not None) and not _fresh_align: algn["alpha"] = algn["la"].exp().clamp(min=args.alpha_floor).detach()
            print(f"[TP-RESUME] warm-started (buffer fresh); search α->{float(srch['alpha']):.3f} approach α->{float(appr['alpha']):.3f} (RESUME_ALPHA={RESUME_ALPHA})", flush=True)
        else:
            print(f"[TP-RESUME] FRESH search -> keeps HIGH alpha {float(srch['alpha']):.2f} (explore); approach/align resumed", flush=True)
    # ---- TWO-PHASE live UI (port = args.http_port) so you can SEE search/approach happening ----
    import threading as _tph
    from http.server import HTTPServer as _TPHS, BaseHTTPRequestHandler as _TPBH
    _cube0 = env.scene[args.target_cube].data.root_pos_w[0].detach().cpu().tolist()   # initial cube world pos (slider default)
    _TPUI = {"step":0,"phs":N,"pha":0,"sbuf":0,"abuf":0,"rsr":0.0,"n":0,"rs":0.0,"ra":0.0,"vis":0.0,"gz":0.0,
             "als":1.0,"ala":1.0,"front":b"","wrist":b"","succ":0,
             "cube":_cube0, "stepreq":0, "play":bool(EVAL_AUTOPLAY), "phase0":0, "resetreq":0, "backreq":0, "forcereq":0, "essc":EVAL_STEP_SCALE}   # eval: cube + step/reset/back/force counters + live per-step scale
    _TPHIST = {"rsr":[],"rsr_old":[],"vis":[],"rs":[],"ra":[],"rl":[],"crl":[],"crs":[],"cra":[],"dmin":[],"dcube":[],"dcube_s":[],"ncen_s":[],"ncen_a":[]}   # +align reward & per-phase CUMULATIVE + approach-progress + centering split search/approach (user 2026-07-03)
    def _tppush(k, v, cap=160):
        _TPHIST[k].append(v)
        if len(_TPHIST[k]) > cap: _TPHIST[k].pop(0)
    def _tpg(title, k, c): return f"<div style='display:inline-block;margin:6px'><div style='color:#aaa;font-size:11px'>{title}</div>{_spark(_TPHIST[k], c)}</div>"
    class _TPHND(_TPBH):
        def log_message(self, *a): pass
        def do_GET(self):
            if self.path.startswith("/front"):
                self.send_response(200); self.send_header("Content-Type","image/png"); self.end_headers()
                if _TPUI["front"]: self.wfile.write(_TPUI["front"]); return
            if self.path.startswith("/wrist"):
                self.send_response(200); self.send_header("Content-Type","image/png"); self.end_headers()
                if _TPUI["wrist"]: self.wfile.write(_TPUI["wrist"]); return
            if self.path.startswith("/clip"):
                self.send_response(200); self.send_header("Content-Type","image/gif"); self.end_headers()
                if _TPUI.get("clip"): self.wfile.write(_TPUI["clip"]); return
            if self.path.startswith("/set"):     # eval: cube x/y/z sliders
                from urllib.parse import urlparse as _upe, parse_qs as _pqe
                _qe = _pqe(_upe(self.path).query)
                for _k, _i in (("cx",0),("cy",1),("cz",2)):
                    if _k in _qe: _TPUI["cube"][_i] = float(_qe[_k][0])
                self.send_response(200); self.end_headers(); self.wfile.write(b"ok"); return
            if self.path.startswith("/step"):     # eval: advance ONE FSM control step
                _TPUI["stepreq"] = _TPUI.get("stepreq",0) + 1; _TPUI["play"] = False
                self.send_response(200); self.end_headers(); self.wfile.write(b"ok"); return
            if self.path.startswith("/play"):     # eval: toggle autoplay
                _TPUI["play"] = not _TPUI.get("play", False)
                self.send_response(200); self.end_headers(); self.wfile.write(b"ok"); return
            if self.path.startswith("/reset"):    # eval: NEW ENVIRONMENT (env.reset -> random cube + arm to start)
                _TPUI["resetreq"] = _TPUI.get("resetreq",0) + 1; _TPUI["play"] = False
                self.send_response(200); self.end_headers(); self.wfile.write(b"ok"); return
            if self.path.startswith("/back"):     # eval: step BACK one control step (restore snapshot)
                _TPUI["backreq"] = _TPUI.get("backreq",0) + 1; _TPUI["play"] = False
                self.send_response(200); self.end_headers(); self.wfile.write(b"ok"); return
            if self.path.startswith("/force"):    # eval: manually TOGGLE env0 phase (SEARCH<->APPROACH) to test each policy in isolation
                _TPUI["forcereq"] = _TPUI.get("forcereq",0) + 1; _TPUI["play"] = False
                self.send_response(200); self.end_headers(); self.wfile.write(b"ok"); return
            if self.path.startswith("/setss"):    # eval: live per-STEP motion scale (granularity)
                from urllib.parse import urlparse as _up2, parse_qs as _pq2
                _q2 = _pq2(_up2(self.path).query)
                if "v" in _q2: _TPUI["essc"] = max(0.02, min(1.0, float(_q2["v"][0])))
                self.send_response(200); self.end_headers(); self.wfile.write(b"ok"); return
            u = _TPUI
            cards = (_card("STEP", f"{u['step']:,}", "#4fc3f7")
                     + _card("RSR", f"{u['rsr']*100:.1f}%", "#ffca28", f"n={u['n']} · approach success (unified center+frac+height condition)")
                     + _card("RSR DCUBE (priv)", f"{u.get('rsr_old',0.0)*100:.1f}%", "#9575cd", f"n={u.get('n_old',0)} · PRIVILEGED reference (actual jaw↔cube<{RSR_DCUBE}m, not used for reward/termination)")
                     + _card("SEARCH success rate", f"{u.get('wr_s',0.0)*100:.1f}%", "#42a5f5", f"{int(round(u.get('wr_s',0.0)*u.get('wn_s',0)))}/{u.get('wn_s',0)} episodes · found cube and handed off within {SEARCH_MAX_S:.0f}s (1 vote per episode, counted on reset)")
                     + _card("approach→search re-handoff rate", f"{u.get('wr_resrch',0.0)*100:.1f}%", "#ef5350", f"{int(round(u.get('wr_resrch',0.0)*u.get('wn_resrch',0)))}/{u.get('wn_resrch',0)} approach-reaching episodes with full cube loss→re-search (lower = approach keeps/looks at the cube better)")
                     + _card("centering reach rate", f"{u.get('cenrate',0.0)*100:.0f}%", "#42a5f5", f"fraction currently in SEARCH with ncen&lt;{SEARCH_CENTER} · 256env · leading indicator of handoff")
                     + _card("head angle (search)", f"{u.get('head_deg_s',0.0):+.0f}°", "#ffa726", "SEARCH mean nose angle · +tilt down (down=dive direction)/−tilt back (up)/0 level · wflex divergence watch (ramping to the 128° limit = dive)")
                     + _card("→APPROACH transition rate", f"{u.get('apprate',0.0)*100:.0f}%", "#66bb6a", f"fraction of all envs reaching APPROACH · handoff throughput")
                     + _card("APPROACH centering", f"{u.get('appr_cen',0.0)*100:.0f}%", "#ab47bc", f"fraction in APPROACH with ncen&lt;{APP_CEN} · actual approach centering (for NO_RESRCH)")
                     + _card("wrist frac (search)", f"{u.get('frac_s',0.0):.3f}", "#4dd0e1", "SEARCH-phase mean wrist frac (visible env) · cube size")
                     + _card("wrist frac (approach)", f"{u.get('frac_a',0.0):.3f}", "#ffa726", f"APPROACH-phase mean wrist frac (visible env) · success gate frac&gt;{FRAC_GRASP}")
                     + _card("head angle", f"{u.get('head_deg',0.0):+.0f}°", "#ffa726", f"APPROACH mean nose angle · +tilt down (down)/−tilt back (up)/0 level · |value|↑=more bent (suppressed by W_WPITCH expo)")
                     + _card("APPROACH success rate", f"{u.get('wr_a',0.0)*100:.0f}%", "#66bb6a", f"last {u.get('wn_a',0)} runs · " + ("ALIGN reached" if u.get('three') else "grasp height reached"))
                     + (_card("ALIGN handoff rate", f"{u.get('wr_h',0.0)*100:.0f}%", "#26c6da", f"last {u.get('wn_h',0)} runs · approach→ALIGN transition") if u.get('three') else "")
                     + (_card("ALIGN success rate", f"{u.get('wr_l',0.0)*100:.0f}%", "#ab47bc", f"last {u.get('wn_l',0)} runs · aligned in mouth (in-mouth)") if u.get('three') else "")
                     + _card("SEARCH envs", f"{u['phs']}", "#42a5f5", f"buf {u['sbuf']:,} · α{u['als']:.2f}")
                     + _card("APPROACH envs", f"{u['pha']}", "#66bb6a", f"buf {u['abuf']:,} · α{u['ala']:.2f}")
                     + (_card("ALIGN envs", f"{u.get('pla',0)}", "#ab47bc", f"buf {u.get('labuf',0):,} · α{u.get('all',0.0):.2f}") if u.get('three') else "")
                     + _card("wrist vis", f"{u['vis']*100:.0f}%", "#26c6da", "search leading indicator")
                     + _card("gripper z", f"{u['gz']:.3f}", "#ef5350", "FK (m)")
                     + _card("gripper orientation RPY", (lambda _r: f"{_r[0]:.0f},{_r[1]:.0f},{_r[2]:.0f}°")(u.get('rpy',(0,0,0))), "#26c6da", "roll,pitch,yaw · quat "+str(u.get('quat',[])))
                     + _card("r_search", f"{u['rs']:+.2f}", "#42a5f5", f"Σcum {u.get('crs',0.0):+.0f}")
                     + _card("r_approach", f"{u['ra']:+.2f}", "#66bb6a", f"Σcum {u.get('cra',0.0):+.0f}")
                     + _card("wrist frac (abs)", f"{u.get('frac',0.0):.3f}", "#ffa726", "ABSOLUTE wrist frac = YOLO bbox area/img area (median, APPROACH envs, larger cube↑)")                     + _card("search ncen", f"{u.get('ncen_s',0.0):.3f}", "#42a5f5", f"0=dead center · handoff&lt;{SEARCH_CENTER}")
                     + _card("approach ncen", f"{u.get('ncen_a',0.0):.3f}", "#66bb6a", f"0=dead center · env0 {u.get('ncen0',0.0):.3f} · success&lt;{APP_CEN}")
                     + _card("search sweep", f"{u.get('sweep',0.0):.2f}", "#7e57c2", "mean |a_s| — anti-stall: should rise while blind")
                     + _card("search acq(front)", f"{u.get('acqr',0.0):+.2f}", "#26a69a", f"FRONT-guided acquisition reward (W_ACQ_F) mean · fraction of blind SEARCH envs with a front guide {u.get('acqcov',0.0)*100:.0f}% · steers aiming via front when the wrist cannot see")
                     + _card("descend depth", f"{u.get('dmin',0.0):.3f}", "#ef5350", f"live avg gz (APPROACH, →0.055 grasp) · reach&lt;0.12 {u.get('reachz',0.0)*100:.0f}%")
                     + _card("cube dist (SEARCH)", f"{u.get('dcube_s',0.0):.3f}", "#26c6da", "live median gripper↔cube (SEARCH envs, m)")
                     + _card("cube dist (APPROACH)", f"{u.get('dcube_a',u.get('dcube',0.0)):.3f}", "#26a69a", "live median gripper↔cube (APPROACH envs, m →0)")
                     + _card("cube_dist_front", f"{u.get('cdf',0.0):.3f}", "#8d6e63", "front back-projected horizontal distance (m, APPROACH)")
                     + _card("cube_dist_wrist", f"{u.get('cdw',0.0):.3f}", "#8d6e63", "wrist back-projected horizontal distance (m, wrist-valid envs)")
                     + _card("cube_dist_front_3d", f"{u.get('cdf3',0.0):.3f}", "#5c6bc0", "front back-projected 3D distance (m) — deployable dcube approximation. Compare with the privileged 'cube dist' (dcube_v) next to it to verify back-projection accuracy")
                     + _card("dcube (front-valid)", f"{u.get('dcube_v',0.0):.3f}", "#26a69a", "privileged 3D distance, same front-valid envs — for apples-to-apples comparison with cube_dist_front_3d")
                     + _card("cube_dist_wrist_3d", f"{u.get('cdw3',0.0):.3f}", "#5c6bc0", "wrist back-projected 3D distance (m) — deployable. Accurate in the close (steep) range")
                     + _card("cube_dist_tri_3d", f"{u.get('cdtri',0.0):.3f}", "#7e57c2", f"STEREO triangulated 3D distance (m, front+wrist ray intersection, no plane assumption) · ray gap {u.get('tgap',0.0):.3f}(smaller = more reliable) — compare with dcube above")
                     + _card("F/W matched", f"{u.get('cdfm',0.0):.3f} / {u.get('cdwm',0.0):.3f}", "#4db6ac", f"front/wrist distance on the same both-valid envs (n={u.get('cdnm',0)}) · similar = computation OK")
                     + "<div style='width:100%;height:0'></div><div style='color:#888;font-size:11px;margin:6px 0 0 6px'>▼ trimmed-mean (mean after removing 4 from each end = 8 outliers)</div>"
                     + _card("cube dist (tm)", f"{u.get('dcube_tm',0.0):.3f}", "#00897b", "trimmed-mean gripper↔cube (APPROACH, 8 outliers removed, flown-away cubes excluded)")
                     + _card("cube flew away", f"{u.get('cube_gone',0)}", "#ef5350", "number of envs whose cube flew away due to physics (forced reset/fail, excluded from metrics)")
                     + _card("front_3d (tm)", f"{u.get('cdf3_tm',0.0):.3f}", "#3949ab", "trimmed-mean front-3D distance")
                     + _card("approach ncen (tm)", f"{u.get('ncen_a_tm',0.0):.3f}", "#43a047", "trimmed-mean centering err (visible APPROACH)")
                     + _card("descend z (tm)", f"{u.get('dmin_tm',0.0):.3f}", "#e53935", "trimmed-mean gz (APPROACH)")
                     + _card("wrist frac (tm)", f"{u.get('frac_tm',0.0):.3f}", "#fb8c00", "trimmed-mean wrist frac (APPROACH)")
                     + "<div style='width:100%;height:0'></div><div style='color:#888;font-size:11px;margin:6px 0 0 6px'>▼ r_approach per-term breakdown (APPROACH env mean · +reward/−penalty). A large + while ncen/dist are bad means farming</div>"
                     + _card("rb: descend", f"{u.get('rb_desc',0):+.2f}", "#26a69a", "descend (W_DEPTH·center gate)")
                     + _card("rb: frac", f"{u.get('rb_frac',0):+.2f}", "#ffa726", "frac LEVEL(W_AF, ungated)")
                     + _card("rb: novis", f"{u.get('rb_novis',0):+.2f}", "#ef5350", "not-visible penalty (−W_NOVIS_A)")
                     + _card("rb: gone", f"{u.get('rb_gone',0):+.2f}", "#ef5350", "cube-flung penalty (−W_GONE)")
                     + _card("rb: table", f"{u.get('rb_tab',0):+.2f}", "#ef5350", "table penalty (−)")
                     + (_card("r_align", f"{u.get('rl',0.0):+.2f}", "#ab47bc", f"Σcum {u.get('crl',0.0):+.0f}") if u.get('three') else ""))
            _chist = u.get('curric_hist') or []                              # user 2026-07-15: in-training EVAL history (curriculum self-eval sweeps) — sparkline in the training UI
            if _chist:
                _cmax = max(1.0, max(r for _, r in _chist))
                _bars = "".join(
                    f"<div title='step {s:,} · RSR {r:.1f}%' style='flex:1;min-width:3px;height:{4+56*r/_cmax:.0f}px;"
                    f"background:{'#66bb6a' if (i==len(_chist)-1) else '#2e7d32'};border-radius:2px 2px 0 0'></div>"
                    for i, (s, r) in enumerate(_chist))
                _clast = _chist[-1]
                _hm_cur = u.get('curric_heat',''); _hm_prev = u.get('curric_heat_prev',''); _hm_srch = u.get('curric_heat_srch','')
                _heat = ("<div style='width:100%;display:flex;gap:16px;margin:4px 6px 6px;flex-wrap:wrap'>"
                         + (f"<div style='background:#181818;border-radius:8px;padding:8px 10px'><div style='color:#66bb6a;font-size:11px;font-weight:700'>latest RSR heatmap</div>{_hm_cur}</div>" if _hm_cur else "")
                         + (f"<div style='background:#181818;border-radius:8px;padding:8px 10px'><div style='color:#42a5f5;font-size:11px;font-weight:700'>latest SEARCH heatmap</div>{_hm_srch}</div>" if _hm_srch else "")
                         + (f"<div style='background:#181818;border-radius:8px;padding:8px 10px;opacity:0.75'><div style='color:#888;font-size:11px'>previous RSR sweep</div>{_hm_prev}</div>" if _hm_prev else "")
                         + "</div>") if _hm_cur else ""
                _shist = u.get('curric_hist_srch') or []                         # user 2026-07-18: SEARCH-success(reach) history per sweep -> histogram
                _sbars = "".join(
                    f"<div title='step {s:,} · SEARCH {r:.1f}%' style='flex:1;min-width:3px;height:{4+56*r/100.0:.0f}px;"
                    f"background:{'#42a5f5' if (i==len(_shist)-1) else '#1565c0'};border-radius:2px 2px 0 0'></div>"
                    for i, (s, r) in enumerate(_shist))
                _sblock = ((f"<div style='width:100%;font-size:13px;font-weight:700;color:#42a5f5;margin:8px 6px 2px'>■ in-training SEARCH success (reach) history</div>"
                            + "<div style='width:560px;display:flex;align-items:flex-end;gap:2px;height:64px;margin:6px 6px;padding:4px 6px;background:#111;border-radius:6px'>"
                            + _sbars + "</div>"
                            + f"<div style='width:560px;font-size:11px;color:#888;margin:0 6px 6px'>left→right = training progress · bar height ∝ SEARCH success rate (0-100%) · bright blue=latest ({_shist[-1][1]:.1f}%)</div>") if _shist else "")
                _spark = (f"<div style='width:100%;font-size:15px;font-weight:700;color:#66bb6a;margin:6px 6px 2px'>■ in-training EVAL history (curriculum self-eval, every {CURRIC_EVERY:,} steps)</div>"
                          + _card("★ latest EVAL RSR", f"{_clast[1]:.1f}%", "#66bb6a", f"@ step {_clast[0]:,} · {len(_chist)} sweeps · grid {CURRIC_NX}×{CURRIC_NY}")
                          + "<div style='width:560px;display:flex;align-items:flex-end;gap:2px;height:64px;margin:6px 6px;padding:4px 6px;background:#111;border-radius:6px'>"
                          + _bars + "</div>"
                          + f"<div style='width:560px;font-size:11px;color:#888;margin:0 6px 6px'>left→right = training progress · bar height ∝ EVAL RSR (peak {_cmax:.1f}%) · bright green=latest</div>"
                          + _sblock
                          + _heat
                          + "<div style='width:100%;height:1px;background:#333;margin:8px 0'></div>")
                cards = _spark + cards
            if u.get('eval_M'):                                              # user 2026-07-14: FIXED eval-set benchmark panel at the TOP (RSR + coverage + success heatmap)
                _covpct = 100*u.get('eval_cov',0)/max(1,u.get('eval_M',1))
                cards = ("<div style='width:100%;font-size:15px;font-weight:700;color:#ffca28;margin:6px 6px 2px'>■ fixed EVAL benchmark (reproducible)</div>"
                         + _card("★ EVAL RSR", f"{u.get('eval_rsr',0.0):.1f}%", "#ffca28", f"fixed {u.get('eval_M',0):,} cases · deployable proxy success rate")
                         + _card("★ RSR DCUBE (actual reach)", f"{u.get('eval_rsr_old',0.0):.1f}%", "#26a69a", "privileged true 3D distance<RSR_DCUBE · true reach rate (gap vs RSR = overestimate)")
                         + _card("SEARCH reached", f"{u.get('eval_srch',0.0):.1f}%", "#42a5f5", "fraction where SEARCH handed off to APPROACH (search success rate)")
                         + _card("coverage", f"{u.get('eval_cov',0):,}", "#4fc3f7", f"/{u.get('eval_M',0):,} ({_covpct:.1f}%) · full sweeps {u.get('eval_sweeps',0)}")
                         + u.get('eval_heat','')
                         + u.get('eval_heat_old','')
                         + u.get('eval_heat_srch','')
                         + "<div style='width:100%;height:1px;background:#333;margin:8px 0'></div>"
                         + "<div style='width:100%;font-size:12px;color:#888;margin:2px 6px'>▼ live diagnostics (below)</div>"
                         + cards)
            bar = (f"<div style='display:flex;width:560px;border-radius:6px;overflow:hidden;margin:10px 0'>"
                   f"<div style='flex:{max(u['phs'],0.001)};background:#42a5f5;height:32px;display:flex;align-items:center;justify-content:center;color:#111;font-size:12px;font-weight:700'>SEARCH {u['phs']}</div>"
                   f"<div style='flex:{max(u['pha'],0.001)};background:#66bb6a;height:32px;display:flex;align-items:center;justify-content:center;color:#111;font-size:12px;font-weight:700'>APPROACH {u['pha']}</div>"
                   + (f"<div style='flex:{max(u.get('pla',0),0.001)};background:#ab47bc;height:32px;display:flex;align-items:center;justify-content:center;color:#111;font-size:12px;font-weight:700'>ALIGN {u.get('pla',0)}</div>" if u.get('three') else "")
                   + "</div>")
            graphs = _tpg("RSR (approach success)","rsr","#ffca28") + _tpg("RSR DCUBE (privileged ref)","rsr_old","#ffa726") + _tpg("wrist vis (search progress)","vis","#26c6da") + _tpg("search ncen (→0)","ncen_s","#42a5f5") + _tpg("approach ncen (→0)","ncen_a","#66bb6a") + _tpg("r_search","rs","#42a5f5") + _tpg("r_approach","ra","#66bb6a") + _tpg("descend realtime avg z","dmin","#ef5350") + _tpg("cube dist (SEARCH)","dcube_s","#26c6da") + _tpg("cube dist (APPROACH)","dcube","#26a69a")
            if u.get('three'):
                graphs += (_tpg("r_align","rl","#ab47bc") + _tpg("Σcum search","crs","#42a5f5") + _tpg("Σcum approach","cra","#66bb6a") + _tpg("Σcum ALIGN","crl","#ab47bc"))
            _evalctl = ""
            if TP_EVAL:
                _cb = u["cube"]
                _sl = "".join(f"<div>cube {_n} <input type=range min={_v-0.4:.3f} max={_v+0.4:.3f} step=0.01 value={_cb[_i]:.3f} oninput=\"fetch('/set?{_c}='+this.value)\"> <span>{_cb[_i]:.3f}</span></div>" for _n,_c,_i,_v in (("x(fwd)","cx",0,_cube0[0]),("y(side)","cy",1,_cube0[1]),("z","cz",2,_cube0[2])))
                _phn = "APPROACH" if u.get("phase0") else "SEARCH"
                _nc=u.get('ncen0',0.0); _ch=u.get('chold0',0); _lh=u.get('lhold0',0); _fr=u.get('frac0',0.0); _vs=u.get('vis0',0.0)
                def _chk(ok): return "<span style='color:#66bb6a'>✓</span>" if ok else "<span style='color:#ef5350'>✗</span>"
                _cen_ok=(_nc<SEARCH_CENTER and _vs>0.5)
                # phase-transition gates, exactly as the FSM computes them
                _gate=(f"<div style='font-size:12px;color:#ccc;margin-top:8px;line-height:1.7'>"
                       f"<b style='color:#42a5f5'>SEARCH→APPROACH</b>: centered {_chk(_cen_ok)} (ncen <b>{_nc:.3f}</b> &lt; {SEARCH_CENTER}, vis {_vs:.0f}) → held <b>{_ch}/{SEARCH_HOLD}</b> {_chk(_ch>=SEARCH_HOLD)}<br>"
                       f"<b style='color:#66bb6a'>APPROACH→SUCCESS</b>: frac <b>{_fr:.3f}</b> &gt; {FRAC_GRASP} {_chk(_fr>FRAC_GRASP)} &nbsp;&amp;&nbsp; centered {_chk(_cen_ok)}<br>"
                       f"<b style='color:#ef5350'>APPROACH→SEARCH (lost)</b>: lost {_chk((_vs<0.5) or (_nc>LOSE_TOL))} (vis {_vs:.0f} / ncen {_nc:.3f} &gt; {LOSE_TOL}) → held <b>{_lh}/{LOSE_HOLD}</b> {_chk(_lh>=LOSE_HOLD)}</div>")
                _plog=u.get('plog',[])
                _logh=("<div style='font-size:12px;color:#888;margin-top:8px;border-top:1px solid #333;padding-top:6px'><b style='color:#ffca28'>phase transitions:</b><br>"
                       + ("<br>".join(reversed(_plog)) if _plog else "<i>none yet — STEP to advance</i>") + "</div>")
                _evalctl = (f"<div style='background:#161616;border-radius:8px;padding:12px;margin:8px 0;border:1px solid #333'>"
                            f"<b style='color:#ffca28'>STEP-EVAL</b> &nbsp; env0 phase = <b style='color:{'#66bb6a' if u.get('phase0') else '#42a5f5'};font-size:16px'>{_phn}</b> &nbsp;&nbsp;"
                            f"<button onclick=\"fetch('/back')\" style='font-size:15px;padding:6px 14px'>⏮ BACK</button> "
                            f"<button onclick=\"fetch('/step')\" style='font-size:15px;padding:6px 18px;font-weight:700'>⏭ STEP</button> "
                            f"<button onclick=\"fetch('/play')\" style='font-size:15px;padding:6px 18px'>▶ PLAY / ⏸</button> "
                            f"<button onclick=\"fetch('/reset')\" style='font-size:15px;padding:6px 16px;background:#7e57c2;color:#fff;font-weight:700;border-radius:4px'>🔄 NEW ENV</button> "
                            f"<button onclick=\"fetch('/force')\" style='font-size:15px;padding:6px 14px;background:#ef6c00;color:#fff;font-weight:700;border-radius:4px'>⚙ FORCE S⇄A</button>"
                            f"<div style='margin-top:8px;color:#bbb'>STEP size <input type=range min=0.05 max=1.0 step=0.05 value={u.get('essc',0.25):.2f} oninput=\"fetch('/setss?v='+this.value)\"> <b style='color:#ffca28'>{u.get('essc',0.25):.2f}×</b> <span style='color:#777'>(smaller=precise, larger=fast — for checking wrist range)</span></div>"
                            f"{_gate}{_logh}"
                            f"<div style='margin-top:8px;color:#bbb'>drag to place the cube LIVE (paused) · actual world pos = <b style='color:#26c6da'>{u.get('cubew','?')}</b>:</div>{_sl}</div>")
            if TP_EVAL and u.get('eval_M'):   # user 2026-08-01: BENCH UI = key EVAL metrics + heatmaps only (graphs/diagnostic cards removed, clean)
                _covpct2 = 100*u.get('eval_cov',0)/max(1,u.get('eval_M',1))
                _frag = ("<div style='width:100%;font-size:16px;font-weight:700;color:#ffca28;margin:6px 6px 8px'>■ fixed EVAL benchmark</div>"
                         + _card("★ EVAL RSR", f"{u.get('eval_rsr',0.0):.1f}%", "#ffca28", f"fixed {u.get('eval_M',0):,} cases · deployable proxy")
                         + _card("★ RSR DCUBE (actual reach)", f"{u.get('eval_rsr_old',0.0):.1f}%", "#26a69a", f"true 3D distance<{RSR_DCUBE}m · true reach rate")
                         + _card("SEARCH reached", f"{u.get('eval_srch',0.0):.1f}%", "#42a5f5", "handoff success rate")
                         + _card("coverage", f"{u.get('eval_cov',0):,}", "#4fc3f7", f"/{u.get('eval_M',0):,} ({_covpct2:.1f}%) · sweeps {u.get('eval_sweeps',0)}")
                         + "<div style='display:flex;gap:14px;flex-wrap:wrap;margin-top:10px'>"
                         + u.get('eval_heat','') + u.get('eval_heat_old','') + u.get('eval_heat_srch','')
                         + "</div>")
            else:
                _frag = f"<div>{cards}</div><div>{bar}</div><div>{graphs}</div>"
            if "frag=1" in self.path:                                     # user 2026-07-03: metrics-only fragment for JS polling — NO full-page reload -> smooth video (was meta-refresh content=2 = 2s hard reload + flicker)
                self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8"); self.end_headers(); self.wfile.write(_frag.encode()); return
            page = ("<html><head><meta charset=utf-8><title>reBot 8896 two-phase</title></head>"
                    "<body style='font-family:monospace;background:#0e0e0e;color:#eee;padding:16px'>"
                    f"<h2 style='color:#ddd'>8896 — TWO-PHASE {'STEP-EVAL' if TP_EVAL else '(search &#9654; approach)'}</h2>"
                    f"{_evalctl}"
                    f"<div id=frag>{_frag}</div>"
                    "<div style='display:flex;gap:10px;margin:10px 0'>"
                    "<div><div style='color:#aaa;font-size:11px'>FRONT (env0, DR = what the detector sees)</div><img id=imF src=/front.png width=256 style='background:#000;border-radius:6px'></div>"
                    "<div><div style='color:#aaa;font-size:11px'>WRIST (env0, DR)</div><img id=imW src=/wrist.png width=256 style='background:#000;border-radius:6px'></div>"
                    "<div><div style='color:#aaa;font-size:11px'>LIVE CLIP (env0 front|wrist, ~last 20s)</div><img id=imC src=/clip.gif width=320 style='background:#000;border-radius:6px'></div></div>"
                    "<script>"
                    "var _t=0;"
                    "function _swap(id,url){var im=new Image();im.onload=function(){var e=document.getElementById(id);if(e)e.src=im.src;};im.src=url+'?'+(++_t);}"
                    "setInterval(function(){_swap('imF','/front.png');_swap('imW','/wrist.png');},200);"
                    "setInterval(function(){_swap('imC','/clip.gif');},1200);"
                    "setInterval(function(){fetch('/?frag=1').then(function(r){return r.text();}).then(function(h){var d=document.getElementById('frag');if(d)d.innerHTML=h;}).catch(function(){});},1000);"
                    "</script>"
                    "</body></html>")
            self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8"); self.end_headers(); self.wfile.write(page.encode())
    _TPHS.allow_reuse_address = True                              # SIMPLE bind (no retry — retry made dying zombies re-grab the port). mirrors the working single-policy _H server.
    try:
        _tpsrv = _TPHS(("0.0.0.0", args.http_port), _TPHND)
        _tph.Thread(target=_tpsrv.serve_forever, daemon=True).start()
        print(f"[TP] UI http://0.0.0.0:{args.http_port} (bound)", flush=True)
    except Exception as _e:
        print(f"[TP] UI bind FAILED: {_e}", flush=True)
    _tp_succ_total = 0; _succ_s = 0; _succ_a = 0; _succ_l = 0      # per-phase success counters: SEARCH->APPROACH / APPROACH(grasp-ready or ->ALIGN) / ALIGN in-mouth
    _ent_s = 0; _ent_a = 0; _ent_l = 0                            # per-phase ENTRY counts -> success RATE = succ/entry
    _METRIC_WIN = int(os.environ.get("METRIC_WIN", "500"))   # user 2026-07-23: UNIFIED rolling-window size for ALL per-episode success rates (search / approach / RSR / rsr_old) — same aggregation: one 0/1 per finished episode, recorded at reset. Was split (search/approach 100, rsr/rsr_old 500).
    _win_s = []; _win_a = []; _win_l = []; _win_h = []; _win_resrch = []                     # user 2026-07-24: _win_resrch = of APPROACH-reached episodes, did approach->search re-handoff (cube fully lost) fire? lower = approach keeps the cube (look-at) better.  # user 2026-07-02: ROLLING per-attempt success windows (last 100). SEARCH attempt = did it find+center+hand off? (1) vs episode ended still searching (0). APPROACH attempt = reached grasp/align? (1) vs lost->search or episode ended mid-approach (0). Rolling -> shows CURRENT skill, not lifetime-diluted ratio.
    if int(os.environ.get("REAL_JLIM", "1")):                     # apply REAL B601-DM joint limits in SIM frame (raw_calib + JOINT_OFFSET) so sim range == real
        try:
            # SIM-FRAME limits = raw_calib + JOINT_OFFSET (deploy maps sim = raw + OFFSET, sign all +1).
            # OFFSET=[-1.5673,0.0369,0.0519,0.8861,-0.0768,-0.3286]. BUG FIX: raw was applied WITHOUT offset before ->
            # wrist capped at sim2.24 = raw1.35 (78deg) instead of raw2.24 (128deg). Now wrist max = sim3.126 = raw2.24.
            _jl = dict(profile.joint_limits_sim)   # real range in SIM joint frame (OURS: raw_calib+OFFSET 2026-06-30; profile-sourced)
            _lim = torch.tensor([[_jl[_n] for _n in ARM_JOINTS]], device=DEVICE).expand(N, -1, -1).clone()  # (N, 6, 2)
            r.write_joint_position_limit_to_sim(_lim, joint_ids=joint_ids)
            print(f"[REAL-JLIM] B601-DM SIM-FRAME joint limits applied: {[(_n, _jl[_n]) for _n in ARM_JOINTS]}", flush=True)
            # user 2026-07-01: NOW that the wrist limit is widened to 3.126, set the TRUE start/rest wrist (_START_WRIST,
            #   e.g. 2.77) that the USD build-cap (1.57) blocked at init. Overwrite default_joint_pos -> every episode reset
            #   starts the wrist at the real-robot down pose. joint_ids[WFLEX_I] = wrist_flex index in the full joint array.
            if profile.name == "ours" and _START_WRIST > _WRIST_BOOT_CAP + 1e-4:   # OURS-only USD-build-cap workaround; OFFICIAL sets joint4 via high_start within its limit
                _wf_full = int(joint_ids[WFLEX_I]) if not torch.is_tensor(joint_ids) else int(joint_ids[WFLEX_I].item())
                r.data.default_joint_pos[:, _wf_full] = _START_WRIST
                _dj = r.data.default_joint_pos.clone(); r.write_joint_state_to_sim(_dj, torch.zeros_like(_dj)); env.reset()
                print(f"[REAL-JLIM] wrist_flex START/rest -> {_START_WRIST} (default_joint_pos[{_wf_full}] overridden post-widen; was boot-capped {_WRIST_BOOT_CAP})", flush=True)
        except Exception as _je: print(f"[REAL-JLIM] failed: {_je}", flush=True)
    if int(os.environ.get("GRIP_JLIM", "0")):                     # user 2026-07-02: DM shares the RS URDF -> finger stroke 0-0.045m. Match sim gripper joint limit so the RULE grasp closes faithfully. Gated (fresh only).
        try:
            _gids = [jj for jj in range(r.data.joint_pos.shape[1]) if jj not in set(int(x) for x in joint_ids)]   # gripper (non-arm) joint indices
            _glo = float(os.environ.get("GRIP_LO", "0.0")); _ghi = float(os.environ.get("GRIP_HI", "0.045"))
            for _gj in _gids:                                     # PRINT current USD limit + default pos FIRST so we can verify zero/sign before trusting the clamp
                try: _clim = tuple(round(float(x), 4) for x in r.data.joint_pos_limits[0, _gj])
                except Exception: _clim = "?"
                print(f"[GRIP-JLIM] '{r.joint_names[_gj]}' idx{_gj} USD-limit={_clim} default={float(r.data.default_joint_pos[0, _gj]):.4f}", flush=True)
            if _gids:
                _glim = torch.tensor([[(_glo, _ghi)] * len(_gids)], device=DEVICE).expand(N, -1, -1).clone()   # (N, n_grip, 2)
                r.write_joint_position_limit_to_sim(_glim, joint_ids=_gids)
                print(f"[GRIP-JLIM] applied stroke [{_glo},{_ghi}]m to {[r.joint_names[g] for g in _gids]} (DM=RS URDF); adjust via GRIP_LO/GRIP_HI if the printed zero differs", flush=True)
        except Exception as _je: print(f"[GRIP-JLIM] failed: {_je}", flush=True)
    if int(os.environ.get("EXPORT_YOLO", "0")):                   # user 2026-07-08: sim -> YOLO-OBB cube dataset (seg-mask labels). Runs then exits. NEEDS --dr(seg) + WARP_*=0 (RGB/seg aligned). single-cube.
        import cv2, sys as _sys, random as _rnd
        from rebot_grasp_env.visual_obs import build_fg_ids
        _OUT   = os.environ.get("EXPORT_DIR", os.path.expanduser("~/yolo_cube"))
        _NR    = int(os.environ.get("EXPORT_RESETS", "16"))       # frames ≈ NR*N*2cams (16*256*2 ≈ 8k)
        _MINPX = int(os.environ.get("EXPORT_MIN_PX", "5"))        # cube px < this -> negative (no label)
        _YLO   = float(os.environ.get("EXPORT_Y_LO", "-0.94")); _YHI = float(os.environ.get("EXPORT_Y_HI", "0.02"))  # env-local y: near->far = big->small cube (uniform = small cubes well-represented)
        _XLO   = float(os.environ.get("EXPORT_X_LO", "0.085")); _XHI = float(os.environ.get("EXPORT_X_HI", "0.67"))
        _VAL   = 0.1
        for _sp in ("images/train","images/val","labels/train","labels/val","debug"): os.makedirs(os.path.join(_OUT,_sp), exist_ok=True)
        _cube = env.scene[args.target_cube]; _fid = 0
        print(f"[EXPORT] -> {_OUT} resets={_NR} minpx={_MINPX} y[{_YLO},{_YHI}] (seg-mask OBB labels)", flush=True)
        for _ri in range(_NR):
            env.reset(); resample_warp()
            _org = env.scene.env_origins
            _cp = _cube.data.root_pos_w.clone()
            _cp[:,0] = _org[:,0] + (_XLO + (_XHI-_XLO)*torch.rand(N, device=DEVICE))
            _cp[:,1] = _org[:,1] + (_YLO + (_YHI-_YLO)*torch.rand(N, device=DEVICE))
            _yaw = (torch.rand(N, device=DEVICE)*2-1)*3.14159
            _q = torch.stack([torch.cos(_yaw/2), torch.zeros_like(_yaw), torch.zeros_like(_yaw), torch.sin(_yaw/2)], -1)
            _cube.write_root_pose_to_sim(torch.cat([_cp,_q],-1))
            _ao = torch.zeros(N, ACT_DIM, device=DEVICE); _ao[:,6]=1.0; env.step(_ao)   # render update (gripper open)
            _out = render_cams()
            for _nm in ("front","wrist"):
                _seg = env.scene[_nm].data.output["semantic_segmentation"]; _info = env.scene[_nm].data.info["semantic_segmentation"]
                _ids = build_fg_ids(_info, (args.target_cube,)); _idt = torch.tensor(_ids, device=_seg.device) if _ids else None
                _rgb = _out[_nm]; _H = _rgb.shape[1]; _W = _rgb.shape[2]
                for _i in range(N):
                    _spl = "val" if _rnd.random() < _VAL else "train"; _name = f"{_nm}_{_fid:06d}"
                    _img = _rgb[_i].detach().cpu().numpy()
                    cv2.imwrite(os.path.join(_OUT, f"images/{_spl}/{_name}.png"), cv2.cvtColor(_img, cv2.COLOR_RGB2BGR))
                    _lbl = ""
                    if _idt is not None:
                        _m = torch.isin(_seg[_i,:,:,0], _idt).detach().cpu().numpy().astype("uint8"); _ys,_xs = _m.nonzero()
                        if _xs.size >= _MINPX:
                            _rect = cv2.minAreaRect(np.stack([_xs,_ys],1).astype(np.float32)); _box = cv2.boxPoints(_rect)
                            _bn = _box.copy(); _bn[:,0] /= _W; _bn[:,1] /= _H; _bn = _bn.clip(0,1)
                            _lbl = "0 " + " ".join(f"{v:.6f}" for v in _bn.reshape(-1))
                            if _fid % 400 == 0:
                                _dbg = cv2.cvtColor(_img, cv2.COLOR_RGB2BGR).copy(); cv2.polylines(_dbg, [_box.astype(int)], True, (0,255,0), 1)
                                cv2.imwrite(os.path.join(_OUT, f"debug/{_name}.png"), _dbg)
                    open(os.path.join(_OUT, f"labels/{_spl}/{_name}.txt"), "w").write(_lbl)
                    _fid += 1
            # user 2026-07-21: BIG-CUBE augmentation — pin the cube in front of the WRIST cam at CLOSE range (5-25cm) so
            #   it appears LARGE (fills the frame), the distribution the current detector MISSES (fails on close/big cubes).
            #   seg-mask labels are view-independent so wherever the cube lands is correctly boxed. WRIST cam only.
            if int(os.environ.get("EXPORT_BIG", "1")):
                _nbig = int(os.environ.get("EXPORT_BIG_PASSES", "3"))   # extra close-cube passes per reset (each = N wrist frames)
                # user 2026-07-21: BIG cubes via the REAL approach geometry — descend the arm to a HOVER pose, then place the
                #   cube on the TABLE directly UNDER the jaw mouth. Cube is SUPPORTED (no fall) and GUARANTEED in the wrist view
                #   (jaw is right over it). Per-env jitter on lift/wrist_flex varies the jaw height -> a range of BIG cube sizes.
                # user 2026-07-21: hover joints = the user's MEASURED approach pose (frac~0.083 = the exact size YOLO currently
                #   fails on). Jitter lift DOWN (bias) -> a range of BIG cubes up to grasp height. Cube stays on the TABLE
                #   (SUPPORTED, no fall, guaranteed in view) directly under the jaw.
                _hoverj = [float(x) for x in os.environ.get("EXPORT_HOVER_JOINTS", "-1.571,-1.99,-1.684,1.0,0.0,0.0").split(",")]
                _hjb = torch.tensor([_hoverj], device=DEVICE).expand(N, -1)
                _cz_tab = float(_cube.data.root_pos_w[:, 2].mean().item())   # table cube-center height (from the table pass)
                for _bp in range(_nbig):
                    _jz = r.data.joint_pos.clone()
                    _jit = (torch.rand(N, 6, device=DEVICE) * 2 - 1) * torch.tensor([0.02, 0.04, 0.03, 0.06, 0.03, 0.03], device=DEVICE)   # TINY jitter -> stay ~at the approach pose (big cube); just enough size/angle variety
                    for _ji, _jid in enumerate(joint_ids): _jz[:, int(_jid)] = _hjb[:, _ji] + _jit[:, _ji]
                    r.write_joint_state_to_sim(_jz, torch.zeros_like(_jz))
                    env.step(_ao)                                            # settle at hover -> FK ee pose
                    # user 2026-07-21: place the cube EXACTLY at the user's measured grasp offset (SNAP GRASP_OFF, gripper frame) ->
                    #   where it read frac~0.083 (frame-filling, centered). NO z-force (that moved it far/tiny). GRASP_OFF z ~= table.
                    _goff = torch.tensor([[float(x) for x in os.environ.get("EXPORT_GRASP_OFF", "0.0338,-0.0036,0.0068").split(",")]], device=DEVICE)
                    _cw_b = r.data.body_pos_w[:, ee_body_idx] + quat_apply(r.data.body_quat_w[:, ee_body_idx], _goff.expand(N, 3))
                    _cpb = _cube.data.root_pos_w.clone()
                    _cpb[:, 0] = _cw_b[:, 0] + (torch.rand(N, device=DEVICE) * 2 - 1) * 0.01    # at the grasp spot (+ tiny XY jitter)
                    _cpb[:, 1] = _cw_b[:, 1] + (torch.rand(N, device=DEVICE) * 2 - 1) * 0.01
                    _cpb[:, 2] = _cw_b[:, 2]                                 # keep the grasp-offset z (~table) — NO force (force moved it far -> tiny)
                    _ybg = (torch.rand(N, device=DEVICE) * 2 - 1) * 3.14159
                    _qbg = torch.stack([torch.cos(_ybg/2), torch.zeros_like(_ybg), torch.zeros_like(_ybg), torch.sin(_ybg/2)], -1)
                    _cube.write_root_pose_to_sim(torch.cat([_cpb, _qbg], -1)); _cube.write_root_velocity_to_sim(torch.zeros(N, 6, device=DEVICE))
                    env.step(_ao)                                            # cube settles under jaw + seg updates
                    _outb = render_cams()
                    _segb = env.scene["wrist"].data.output["semantic_segmentation"]; _infob = env.scene["wrist"].data.info["semantic_segmentation"]
                    _idsb = build_fg_ids(_infob, (args.target_cube,)); _idtb = torch.tensor(_idsb, device=_segb.device) if _idsb else None
                    _rgbb = _outb["wrist"]; _Hb = _rgbb.shape[1]; _Wb = _rgbb.shape[2]
                    for _i in range(N):
                        _spl = "val" if _rnd.random() < _VAL else "train"; _name = f"wristbig_{_fid:06d}"
                        _imgb = _rgbb[_i].detach().cpu().numpy()
                        cv2.imwrite(os.path.join(_OUT, f"images/{_spl}/{_name}.png"), cv2.cvtColor(_imgb, cv2.COLOR_RGB2BGR))
                        _lblb = ""
                        if _idtb is not None:
                            _mb = torch.isin(_segb[_i,:,:,0], _idtb).detach().cpu().numpy().astype("uint8"); _ysb,_xsb = _mb.nonzero()
                            if _xsb.size >= _MINPX:
                                _rectb = cv2.minAreaRect(np.stack([_xsb,_ysb],1).astype(np.float32)); _boxb = cv2.boxPoints(_rectb)
                                _bnb = _boxb.copy(); _bnb[:,0] /= _Wb; _bnb[:,1] /= _Hb; _bnb = _bnb.clip(0,1)
                                _lblb = "0 " + " ".join(f"{v:.6f}" for v in _bnb.reshape(-1))
                                if _fid % 400 == 0:
                                    _dbgb = cv2.cvtColor(_imgb, cv2.COLOR_RGB2BGR).copy(); cv2.polylines(_dbgb, [_boxb.astype(int)], True, (0,255,0), 1)
                                    cv2.imwrite(os.path.join(_OUT, f"debug/{_name}.png"), _dbgb)
                        open(os.path.join(_OUT, f"labels/{_spl}/{_name}.txt"), "w").write(_lblb)
                        _fid += 1
            if _ri % 4 == 0: print(f"[EXPORT] reset {_ri+1}/{_NR} frames={_fid}", flush=True)
        open(os.path.join(_OUT, "data.yaml"), "w").write(f"path: {_OUT}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['cube']\n")
        print(f"[EXPORT] DONE {_fid} frames -> {_OUT}/ (data.yaml + debug/ overlays). zip & scp to Mac.", flush=True); _sys.exit(0)
    ph = torch.zeros(N, dtype=torch.long, device=DEVICE)          # 0=SEARCH, 1=APPROACH, 2=ALIGN
    _PH_CUR[0] = ph                                                # user 2026-07-19: init BEFORE the first get_obs(line ~1370) so privilege applies from step 0 (was None -> that one call fell back to real detection)
    seen = torch.zeros(N, device=DEVICE)                          # SEEN latch (per phase): 1 once the cube was visible in this phase -> "lost" = seen 1->0 edge (clean state-change signal, user 2026-06-30)
    chold = torch.zeros(N, device=DEVICE); lhold = torch.zeros(N, device=DEVICE)
    _prev_a_a = torch.zeros(N, N_ACT_A, device=DEVICE); _prev_a_s = torch.zeros(N, N_ACT_S, device=DEVICE)   # user 2026-07-16: prev raw policy action (smoothness penalty)
    _prev_is_a = torch.zeros(N, dtype=torch.bool, device=DEVICE); _prev_is_s = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    ahold = torch.zeros(N, device=DEVICE); lhold_l = torch.zeros(N, device=DEVICE); shold_l = torch.zeros(N, device=DEVICE)   # ahold: APPROACH->ALIGN hold ; lhold_l: ALIGN-lost hold ; shold_l: ALIGN in-mouth SUCCESS hold (user 2026-07-03)
    _dwell = torch.zeros(N, device=DEVICE)   # user 2026-07-14: consecutive frames within DWELL_D (deployable 3D) during APPROACH -> expo dwell reward
    # user 2026-07-14: adaptive-curriculum state (spawn-weight map + periodic self-eval sweep). Fully gated on _curric_on.
    _curric_on = bool(CURRIC) and (not TP_EVAL) and (not os.environ.get("EVAL_SET", ""))   # curriculum owns spawn; mutually exclusive with the fixed EVAL_SET
    if _curric_on:
        import torch.nn.functional as _Fnn
        _NCELL = CURRIC_NX*CURRIC_NY
        _cw = torch.ones(_NCELL, device=DEVICE)                                  # spawn weight per cell (uniform to start)
        _curric_excl = torch.zeros(_NCELL, dtype=torch.bool, device=DEVICE)       # user 2026-07-30: CURRIC_EXCLUDE flat cell indices (j*NX+i) -> 0 spawn weight + excluded from sweep RSR aggregate (real-arm-unreachable far-corners)
        for _e in os.environ.get("CURRIC_EXCLUDE","").split(","):
            if _e.strip().isdigit() and int(_e) < _NCELL: _curric_excl[int(_e)] = True
        if bool(_curric_excl.any()): print(f"[CURRIC] EXCLUDE {int(_curric_excl.sum())} cells (spawn+sweep): {os.environ.get('CURRIC_EXCLUDE','')}", flush=True)
        _kept_idx = torch.nonzero(~_curric_excl, as_tuple=False).squeeze(-1)   # user 2026-07-30: SWEEP cycles ONLY these cells (excluded skipped entirely -> not evaluated -> _cur_seen=0 -> "#222" blank in heatmap)
        _NKEEP = int(_kept_idx.numel())
        _cur_seen = torch.zeros(_NCELL, device=DEVICE); _cur_hit = torch.zeros(_NCELL, device=DEVICE)
        _cur_srch = torch.zeros(_NCELL, device=DEVICE)                           # user 2026-07-16: per-cell SEARCH-success (episode reached APPROACH) in the sweep -> search heatmap
        _srch_reached = torch.zeros(N, dtype=torch.bool, device=DEVICE)          # per-env latch: reached APPROACH this episode (reset on episode reset)
        _sweeping = [False]; _swcur = [0]; _NOM_KP = [None]; _NOM_KD = [None]   # user 2026-07-22: sweep DR-off — nominal arm gains captured lazily (list-wrapped)
        _stage_rsr = [False]   # user 2026-07-24: CURRIC_STAGED — False = still mastering SEARCH (sweep ends at hand-off, search-only 15s, weight by search-fail); flips True once a sweep finds EVERY cell at search>=CURRIC_SW_TH -> from then on sweep runs full approach->RSR and weights by RSR-fail
        _next_sw = [10**18 if PRIV_RSR_TARGET > 0.0 else (0 if CURRIC_SWEEP0 else CURRIC_FIRST)]   # user 2026-07-19: PRIV mode -> NO sweep until the anneal starts (set _next_sw=g at trigger). else SWEEP0/CURRIC_FIRST.
        _cw_ext_mtime = [0.0]   # user 2026-07-23: CURRIC_EXT consumer — mtime of the last-loaded external _cw file (poll for newer)
        _cur_cell = torch.full((N,), -1, dtype=torch.long, device=DEVICE)        # sweep cell each env is running (-1 = not scored)
        _env_cell = torch.full((N,), -1, dtype=torch.long, device=DEVICE)        # user 2026-07-19: per-env TRAINING spawn cell (for region-selective privilege PRIV_KEEP_FAIL)
        _cell_keeppriv = torch.zeros(_NCELL, dtype=torch.bool, device=DEVICE)    # per-cell keep-privilege mask (failing cells, updated each sweep)
        _ENV_CELL[0] = _env_cell; _CELL_KEEPPRIV[0] = _cell_keeppriv             # expose to get_obs
        _cell_x = _SPX_LO + (torch.arange(CURRIC_NX, device=DEVICE).float()+0.5)/CURRIC_NX*(_SPX_HI-_SPX_LO)   # (NX,) env-local x centers
        _cell_y = _SPY_LO + (torch.arange(CURRIC_NY, device=DEVICE).float()+0.5)/CURRIC_NY*(_SPY_HI-_SPY_LO)   # (NY,) env-local y centers
        print(f"[CURRIC] adaptive curriculum ON: {CURRIC_NX}x{CURRIC_NY}={_NCELL} cells over x[{_SPX_LO},{_SPX_HI}] fwd[{_SPY_LO+1.15:.2f},{_SPY_HI+1.15:.2f}] | self-eval every {CURRIC_EVERY:,} steps | boost x{CURRIC_BOOST} blur {CURRIC_BLUR}", flush=True)
    succ_lat = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    gl_hold = torch.zeros(N, device=DEVICE); gl_ztgt = torch.zeros(N, device=DEVICE)   # RULE grasp+lift: grasp-hold counter / lift z-target (world)
    gl_cz0  = torch.zeros(N, device=DEVICE); lift_lat = torch.zeros(N, dtype=torch.bool, device=DEVICE)  # cube z at grasp start / lift-success latch
    _tp_recent = []; _lift_recent = []; _tp_frames = []          # _tp_frames = rolling env0 DR front|wrist frames for the live clip; _lift_recent = lift-success rate window
    _rsrd_dist = []; _rsrd_hit = []                              # user 2026-07-13: rolling (spawn fwd-dist, RSR-hit) per finished episode -> RSR-by-distance (reachable vs unreachable analysis)
    # user 2026-07-14: FIXED eval benchmark. EVAL_SET=path/cases.npy (M,4)=[x_local,y_local,yaw_rad,wrist_start].
    #   Each reset draws the NEXT (round-robin) cases instead of random spawn -> reproducible RSR + per-case accumulation.
    _EVAL_CASES=None; _eval_cursor=[0]; _env_case=None; _eval_seen=None; _eval_hits=None; _eval_sweeps=[0]
    _EVAL_SINGLE = int(os.environ.get("EVAL_SINGLE","0"))   # user 2026-07-31: 1 = SINGLE-TRIAL bench — each eval cell scored EXACTLY once (monotonic cursor, no modulo revisit; cells >= M parked -1 -> uncounted). per-cell 0/1 (binary heatmap exact) + ~1-pass rollout (faster). 0 = old coverage-based multi-visit.
    if os.environ.get("EVAL_SET",""):
        _EVAL_CASES = torch.tensor(np.load(os.environ["EVAL_SET"]), dtype=torch.float32, device=DEVICE)
        _env_case = torch.full((N,), -1, dtype=torch.long, device=DEVICE)
        _eval_seen = torch.zeros(_EVAL_CASES.shape[0], device=DEVICE); _eval_hits = torch.zeros(_EVAL_CASES.shape[0], device=DEVICE)
        _eval_hits_old = torch.zeros(_EVAL_CASES.shape[0], device=DEVICE)   # user 2026-07-22: privileged rsr_old hits per case (parallel to _eval_hits)
        _eval_srch = torch.zeros(_EVAL_CASES.shape[0], device=DEVICE)       # user 2026-07-22: per-case SEARCH success (reached APPROACH ph>=1) — SEARCH heatmap
        _eval_srch_reached = torch.zeros(N, dtype=torch.bool, device=DEVICE)   # per-env latch: reached APPROACH this episode (eval)
        _EH_NX = int(os.environ.get("EVAL_HEAT_NX", "40")); _EH_NY = int(os.environ.get("EVAL_HEAT_NY", "40"))   # user 2026-07-16: heatmap grid (lat cols × fwd rows) env-configurable — set to match the eval-set grid (e.g. 10x10) so the UI heatmap cells align with the actual eval cells
        _ehx0,_ehx1 = _SPX_LO,_SPX_HI; _ehf0,_ehf1 = (_SPY_LO+1.15),(_SPY_HI+1.15)  # heatmap bounds follow the actual spawn range (auto-adapts to X/fwd width)
        # user 2026-08-02: EVAL-SET grid fwd extent may differ from SPAWN_Y (e.g. grid 0-65cm vs spawn 10-75). Bin & label the fwd axis by the ACTUAL grid so near rows don't collapse and cm labels match. Half-cell pad centers each grid row in its bin.
        if _EVAL_CASES is not None and _EVAL_CASES.shape[0] > 0:
            _gf=(_EVAL_CASES[:,1]+1.15); _gfmin=float(_gf.min()); _gfmax=float(_gf.max())
            _ghalf=(_gfmax-_gfmin)/max(1,_EH_NY-1)/2.0
            _ehf0,_ehf1=_gfmin-_ghalf,_gfmax+_ghalf
        _ev_ix = ((_EVAL_CASES[:,0]-_ehx0)/max(1e-6,_ehx1-_ehx0)*_EH_NX).clamp(0,_EH_NX-1).long()  # lateral bin
        _ev_iy = (((_EVAL_CASES[:,1]+1.15)-_ehf0)/max(1e-6,_ehf1-_ehf0)*_EH_NY).clamp(0,_EH_NY-1).long()  # forward-dist bin
        _ev_cell = (_ev_iy*_EH_NX + _ev_ix)                                     # flat cell per case
        print(f"[EVAL-SET] {_EVAL_CASES.shape[0]} fixed cases from {os.environ['EVAL_SET']} (round-robin, num-envs={N})", flush=True)
    fd, wd, ori, s = get_obs()
    gz_prev = r.data.body_pos_w[:, ee_body_idx, 2].clone()
    gz_min = gz_prev.clone()                                     # lowest gripper z reached this episode -> APPROACH/ALIGN up-penalty grows with (gz - gz_min)
    _mz_prev = torch.full_like(gz_prev, 9.0); _mz_min = torch.full_like(gz_prev, 9.0)   # user 2026-07-22: jaw-MOUTH z prev/min (MOUTH_DESC mode) — init high so step-0 hover/up penalties don't false-fire
    _app_ori = r.data.body_quat_w[:, ee_body_idx, :].clone()     # user 2026-07-03: EE orientation captured at APPROACH entry (APP_ORI=freeze anchor)
    _app_start_es = torch.full((N,), -1.0, device=DEVICE)        # user 2026-07-21: episode_length_buf value at SEARCH->APPROACH handoff (-1 = not yet handed off). Drives APP_GUAR_S approach-time guarantee.
    _srch_start_es = torch.zeros(N, device=DEVICE)               # user 2026-07-24: episode_length_buf value at SEARCH ENTRY (episode start OR approach->search re-entry). SEARCH_MAX_S budget is PHASE-RELATIVE (fresh per search entry), not episode-relative.
    _srch_ok = torch.zeros(N, dtype=torch.bool, device=DEVICE)   # user 2026-07-23: per-episode latch — did SEARCH hand off to APPROACH this episode?
    _did_resrch = torch.zeros(N, dtype=torch.bool, device=DEVICE)   # user 2026-07-24: per-episode latch — did APPROACH->SEARCH re-handoff (cube fully lost) fire this episode? Recorded ONCE per episode at reset (unbiased search-success rate; was pushed at handoff-time -> success entered rolling window ~10s before same-cohort failures -> rate biased HIGH).
    _resrch_mz = []   # user 2026-07-24: rolling jaw-MOUTH z (m) at each approach->search re-handoff -> [RESRCH-H] histogram (which HEIGHT the re-search bails at: low=close-range FOV-exit, high=bad handoff aim)
    ncen_min = torch.ones(N, device=DEVICE); frac_max = torch.zeros(N, device=DEVICE)   # best centering / biggest frac reached -> regression penalty (got close then moved away)
    _crw_seen = torch.zeros(N, device=DEVICE); _crf_seen = torch.zeros(N, device=DEVICE)   # user 2026-07-09: per-phase latch — wrist/front has RELIABLY seen the cube (for the cube_rel acquired-then-lost penalty)
    dcube_min = torch.full((N,), 9.0, device=DEVICE)             # min TRUE gripper<->cube dist this episode (approach-progress monitor; privileged, monitor-only)
    _rsr_hold = torch.zeros(N, device=DEVICE); _rsr_holdmax = torch.zeros(N, device=DEVICE)   # user 2026-07-14: consecutive / max-consecutive frames MEETING the unified success condition this episode (RSR hold metric)
    _rsr_old_hold = torch.zeros(N, device=DEVICE); _rsr_old_holdmax = torch.zeros(N, device=DEVICE)   # user 2026-07-20: PRIVILEGED reference (true _dcube<RSR_DCUBE) hold -> UI "rsr_old" only
    _rsr_old_recent = []                                                                       # rolling privileged-RSR-hit window (UI "rsr_old")
    df3_prev = torch.full((N,), 0.4, device=DEVICE); _vf_prev = torch.zeros(N, dtype=torch.bool, device=DEVICE)   # user 2026-07-10: prev-step FRONT-3D + front-valid, for the potential/delta reward
    _gzmin_recent = []                                           # rolling ep-min gz over recent episodes (approach-progress metric)
    _tp_best = -1.0                                              # user 2026-07-02: BEST-checkpoint score (RSR% + reach%) -> save *_best.pt so regression doesn't lose the good policy
    _last_psave = [-1]                                           # user 2026-07-19: last SAVE_EVERY bucket already written (permanent step-named ckpts)
    dm_min = torch.full((N,), 9.0, device=DEVICE); yaw_min = torch.full((N,), 9.0, device=DEVICE)   # ALIGN: best mouth-dist / yaw reached -> regression penalty (drifted away / twisted)
    _cum = {"s": 0.0, "a": 0.0, "l": 0.0}                        # cumulative mean reward per phase (progress monitor)
    _dbg_spawn = [True]                                          # one-time [SPAWN-PEAKED] frame-check print
    g = 0; _last_step = 0; _last_reset = 0; _last_back = 0; _last_force = 0; _tp_undo = []; _prev_ph0 = 0   # eval: undo stack + reset/back/force edge counters + env0 phase tracker
    def _tp_refresh(fd_, wd_):                                    # redraw env0 front/wrist + vis after a RESET/BACK (no control step taken)
        try:
            _cf0 = _LAST_IMG["front"] if _LAST_IMG["front"] is not None else env.scene["front"].data.output["rgb"][..., :3]
            _cw0 = _LAST_IMG["wrist"] if _LAST_IMG["wrist"] is not None else env.scene["wrist"].data.output["rgb"][..., :3]
            _cf = _cf0[0, ..., :3].detach().cpu().numpy().astype("uint8").copy(); _draw_det(_cf, fd_[0].tolist()); _TPUI["front"] = _png(_cf)
            _wimg = _cw0[0, ..., :3].detach().cpu().numpy().astype("uint8").copy(); _draw_det(_wimg, wd_[0].tolist()); _TPUI["wrist"] = _png(_wimg)   # renamed _cw->_wimg (curric spawn-weight collision)
            _TPUI["vis0"] = float(wd_[0, 3] > 0.5)
        except Exception: pass

    # ---- hot-loop helpers (user 2026-07-07 modularization): pure per-step math extracted from the loop body ----
    def _boresight_oerr(yaw_act, pitch_act, eq_, rot, pitch_gain=1.0):
        """world-frame boresight angular delta: yaw about world-Z + pitch about EE local-x. Shared by SEARCH
        (a_s=[yaw,pitch], rot=SEARCH_ROT, pitch_gain=WFLEX_GAIN) and APPROACH policy-orientation (a_a[4],a_a[3],
        rot=ROT_SCALE). MIRRORS the deploy _boresight_omega (rebot_deploy_p2_v2_real) for sim2real parity."""
        _z = torch.zeros(eq_.shape[0], 3, device=DEVICE); _z[:, 2] = 1.0                                    # world-Z yaw axis
        _lx = quat_apply(eq_, torch.tensor([1.0, 0.0, 0.0], device=DEVICE).expand(eq_.shape[0], -1))        # EE local-x pitch axis
        return (yaw_act.clamp(-1, 1)*rot).unsqueeze(-1)*_z + ((pitch_act*pitch_gain).clamp(-1, 1)*rot).unsqueeze(-1)*_lx

    def _table_penalty(gz_, mouth_w_):
        """PENETRATION penalty (linear + STEEP quadratic below the reference; ~0 above). Reference = fixed world z
        (PEN_REF_Z, jaw MOUTH), or cube-center (PEN_REF_CUBE), or the table (default, gripper FK z). All 3 phases."""
        if PEN_REF_Z > 0.0:
            _pd = (PEN_REF_Z - mouth_w_[:, 2]).clamp(min=0.0)
        elif PEN_REF_CUBE and (_cube_plane_z is not None):
            _pd = (float(_cube_plane_z) + PEN_Z_OFF - mouth_w_[:, 2]).clamp(min=0.0)
        else:
            _pd = (Z_TABLE - gz_).clamp(min=0.0)
        return W_TABLE*_pd + W_TABLE_HARD*(_pd / max(TABLE_PEN_REF, 1e-6)).clamp(0.0, 1.0)**2

    def _cen_rewards(ncen_):
        """COMBINED centering (tanh inside expo). SEARCH shape (K_CE/CEN_K) and APPROACH shape (K_CE_A/CEN_K_A);
        each max exp(K)−1 at dead-center, →0 far."""
        return (torch.exp(K_CE*(1.0 - torch.tanh(CEN_K*ncen_))) - 1.0,
                torch.exp(K_CE_A*(1.0 - torch.tanh(CEN_K_A*ncen_))) - 1.0)

    def _compute_act_env(a_s, a_a, a_l, is_s, is_a, is_l, s, wd, ph):
        """user 2026-07-07 modularization: per-step action mapping (DIRECT/IK/SEARCH_IK/ALIGN + MOVE_CAP + RULE grasp/lift).
        Args = phase actions/masks/obs; reads module globals (r, env, arm_term, ik, W_*, config flags, gl_ztgt/gl_cz0,
        _app_ori). RETURNS act_env (N,7); mutates arm_term._target_qpos in place. VERBATIM extraction of the former inline
        block (outputs proven not to leak past env.step) — no logic change."""
        act_env = torch.zeros(N, 7, device=DEVICE)
        _essc = float(_TPUI.get("essc", EVAL_STEP_SCALE)) if TP_EVAL else 1.0     # finer per-STEP motion in step-eval (live-tunable via UI slider)
        if not SEARCH_IK:                                                                                    # SEARCH via raw pan+wflex joint deltas (default). SEARCH_IK=1 -> handled by the IK block below (boresight rotation)
            act_env[:, PAN_I]   = torch.where(is_s, a_s[:, 0]*SEARCH_STEP*_essc, act_env[:, PAN_I])                          # SEARCH: pan
            act_env[:, WFLEX_I] = torch.where(is_s, (a_s[:, 1]*WFLEX_GAIN).clamp(-1, 1)*SEARCH_STEP*_essc, act_env[:, WFLEX_I])  # SEARCH: wrist_flex (WFLEX_GAIN -> pitch down more per step)
            if SEARCH_ELBOW:
                act_env[:, ELBOW_I] = torch.where(is_s, a_s[:, 2]*_essc, act_env[:, ELBOW_I])                    # SEARCH: elbow_flex (reach/center) — only when enabled
        Jb = r.root_physx_view.get_jacobians()[:, ee_jacobi_idx, 0:6, joint_ids]   # jacobian for the IK solve
        if DIRECT_JOINTS:                                                          # 8953: DIRECT joint-delta learning (NO IK) for APPROACH + ALIGN. sim REAL_JLIM clamps targets to real-robot ranges.
            for _j in range(6):
                act_env[:, _j] = torch.where(is_a, torch.clamp(a_a[:, _j], -1.0, 1.0)*_essc, act_env[:, _j])       # APPROACH: 6 raw joint-deltas
            if THREE_PHASE:
                for _j in range(6):
                    act_env[:, _j] = torch.where(is_l, torch.clamp(a_l[:, _j], -1.0, 1.0)*_essc, act_env[:, _j])   # ALIGN: 6 raw joint-deltas (incl wrist_roll for yaw)
        else:
            ep = r.data.body_pos_w[:, ee_body_idx, :]; eq = r.data.body_quat_w[:, ee_body_idx, :]
            tgt = ep.clone()
            if APP_SERVO:
                _adz = (a_a[:, 2].clamp(-1, 1)*0.5 + 0.5) * _STEP * _essc         # APPROACH descend (>=0 DOWN)
                tgt[:, 0] = ep[:, 0] + a_a[:, 0].clamp(-1, 1) * _STEP * _essc     # + LATERAL servo
                tgt[:, 1] = ep[:, 1] + a_a[:, 1].clamp(-1, 1) * _STEP * _essc
            else:
                _adz = (a_a[:, 0].clamp(-1, 1)*0.5 + 0.5) * _STEP * _essc         # APPROACH descend-only
            _ncen_now = torch.sqrt((wd[:, 0] - CX_TGT)**2 + (CY_W*(wd[:, 1] - CY_TGT))**2); _vis_now = (wd[:, 3] > 0.5).float()   # descent gate — same H/V retarget+weight as ncen (CX_TGT,CY_TGT)
            _pixel_ok = (torch.exp(-DESC_SOFT * _ncen_now)) if DESC_SOFT > 0 else (_ncen_now < DESC_CEN).float()   # PIXEL centering gate (fallback when wrist unproj unreliable)
            if DESC_MXY > 0.0:                                                            # user 2026-07-27: METRIC-PRIMARY descent gate — when wrist-reliable use the TRUE mouth<->cube horizontal (m; unproj+FK; tilt/parallax-immune); else fall back to the pixel gate. Fully deployable (real wrist-det + calib + FK).
                _recon_ik, _rz_ik = cube_xyz_from_wrist(wd[:, :2], _cube_plane_z)
                _mouth_ik = ep + quat_apply(eq, _JAW_OFF3.expand(N, 3))
                _mxy_ik = torch.norm(_recon_ik[:, :2] - _mouth_ik[:, :2], dim=-1)
                _wv_ik = (wd[:, 3] > 0.5) & (_rz_ik > RZ_MIN_W)
                _gate = (_wv_ik & (_mxy_ik < DESC_MXY)).float()      # user 2026-07-27: METRIC-ONLY (pixel fallback removed) — wrist unreliable -> NO descent (isolate pure metric behavior)
            else:
                _gate = _pixel_ok
            _desc_ok = _gate * _vis_now                                                  # require the cube detected
            tgt[:, 2] = ep[:, 2] - _adz * _desc_ok
            if DESC_FLOOR > 0.0:                                                      # user 2026-07-05: HARD floor — the LOWEST gripper point (EE or jaw MOUTH, FK) can't go below DESC_FLOOR. ee_floor = DESC_FLOOR - min(0, moff_z) so whichever is lower (EE when mouth is above it, or mouth when below) stops at the floor. Prevents the wrist diving into the table. Real-deployable; replaces table penalty.
                _moff_z = quat_apply(eq, _JAW_OFF3.expand(N, 3))[:, 2]
                tgt[:, 2] = torch.where(is_a, torch.maximum(tgt[:, 2], DESC_FLOOR - torch.clamp(_moff_z, max=0.0)), tgt[:, 2])
            if THREE_PHASE and not ALIGN_DIRECT:                                      # ALIGN (IK): servo cube toward jaw-mouth + fine descend
                _ldz = (a_l[:, 3].clamp(-1, 1)*0.5 + 0.5) * _STEP * _essc
                tgt[:, 0] = torch.where(is_l, ep[:, 0] + a_l[:, 0].clamp(-1, 1)*_STEP*_essc, tgt[:, 0])
                tgt[:, 1] = torch.where(is_l, ep[:, 1] + a_l[:, 1].clamp(-1, 1)*_STEP*_essc, tgt[:, 1])
                tgt[:, 2] = torch.where(is_l, ep[:, 2] - _ldz, tgt[:, 2])
            if APP_ORI == "freeze":                                                   # user 2026-07-03: hold APPROACH-entry orientation (anchor -> no drift)
                eq_tgt = _app_ori
            elif APP_ORI == "down":                                                   # slew grasp-axis (local +z) toward world-DOWN, <=ROT_SCALE/step (top-grasp)
                _ax = quat_apply(eq, _ZAXIS.expand(eq.shape[0], -1))                  # current grasp axis in world
                _dn = torch.zeros_like(_ax); _dn[:, 2] = -1.0
                _cr = torch.cross(_ax, _dn, dim=-1); _sn = _cr.norm(dim=-1, keepdim=True)
                _ang = torch.atan2(_sn.squeeze(-1), (_ax * _dn).sum(-1))              # angle from straight-down (rad)
                _hs = (torch.clamp(_ang, max=ROT_SCALE) * 0.5).unsqueeze(-1)          # half of the slew-limited step
                _axn = _cr / (_sn + 1e-8)
                _qc = torch.cat([torch.cos(_hs), _axn * torch.sin(_hs)], dim=-1)      # world-frame correction quat (w,x,y,z)
                eq_tgt = quat_mul(_qc, eq)
            elif APP_ORI == "cube":                                                   # user 2026-07-03 (B): AIM grasp-axis(+z) AT the cube -> gripper homes/sucked into cube (wrist keeps cube in FOV)
                _crxy = s[:, -2:]                                                      # front-cam cube xy offset (world, rel gripper); ~0 if unreliable
                _aim = torch.zeros_like(ep); _aim[:, 0] = _crxy[:, 0]; _aim[:, 1] = _crxy[:, 1]
                _aim[:, 2] = (_cube_plane_z - ep[:, 2])                                # downward toward table height
                _flat = (_crxy.abs().sum(-1) < 1e-4)                                   # front didn't localize -> fall back straight DOWN
                _aim[_flat] = torch.tensor([0.0, 0.0, -1.0], device=DEVICE)
                _aim = _aim / (_aim.norm(dim=-1, keepdim=True) + 1e-8)
                _ax = quat_apply(eq, _ZAXIS.expand(eq.shape[0], -1))                   # current grasp axis in world
                _cr = torch.cross(_ax, _aim, dim=-1); _sn = _cr.norm(dim=-1, keepdim=True)
                _ha = (torch.atan2(_sn.squeeze(-1), (_ax * _aim).sum(-1)) * 0.5).unsqueeze(-1)
                _axn = _cr / (_sn + 1e-8)
                eq_tgt = quat_mul(torch.cat([torch.cos(_ha), _axn * torch.sin(_ha)], dim=-1), eq)   # rotate grasp-axis -> AIM at cube
            elif APP_ORI == "wservo":                                                 # user 2026-07-04 (A): VISUAL-SERVO — rotate wrist to keep the cube CENTERED in the WRIST view (cx,cy->0.5) while descending. Not grasp-axis re-aim (cube), not frozen; wrist TRACKS the cube -> doesn't lose it on descent.
                _wex = (wd[:, 0] - 0.5); _wey = (wd[:, 1] - 0.5)                       # wrist detection centering error (0=centered). NOTE: nwd not defined yet here -> use wd
                _wzz = torch.zeros_like(ep); _wzz[:, 2] = 1.0                          # world-Z = yaw axis (cx error -> pan-like)
                _wlx = quat_apply(eq, torch.tensor([1.0, 0.0, 0.0], device=DEVICE).expand(eq.shape[0], -1))   # EE local-x = pitch axis (cy error -> wflex-like)
                _oerr_ws = ((WSERVO_SX*WSERVO_KX*_wex)*wd[:, 3]).unsqueeze(-1)*_wzz \
                         + ((WSERVO_SY*WSERVO_KY*_wey)*wd[:, 3]).unsqueeze(-1)*_wlx    # world-frame angular correction (gated by wrist vis)
                _wn = _oerr_ws.norm(dim=-1, keepdim=True); _oerr_ws = _oerr_ws * torch.clamp(ROT_SCALE/(_wn+1e-8), max=1.0)   # bound per-step to ROT_SCALE
                if WSERVO_CLAMP > 0.0:                                                 # user 2026-07-05: hard-clamp DOWN-pitch — project out the servo component that tilts the grasp-axis further below WSERVO_CLAMP°
                    _gax_ws = quat_apply(eq, _HEAD.expand(eq.shape[0], -1))             # NOSE/head direction (wrist->jaw) in world
                    _dang_ws = torch.asin(torch.clamp(-_gax_ws[:, 2], -1.0, 1.0))       # rad below horizontal (+ = nose down)
                    _pdir = torch.cross(_ZAXIS.expand(eq.shape[0], -1), _gax_ws, dim=-1)  # d(down-angle) ∝ ω·(ẑ×gax): axis whose +rotation pitches the head DOWN
                    _pdir = _pdir / (_pdir.norm(dim=-1, keepdim=True) + 1e-8)
                    _prate = (_oerr_ws * _pdir).sum(dim=-1, keepdim=True)               # servo's pitch component along _pdir (+ = pitching DOWN)
                    _lim_ws = WSERVO_CLAMP*0.0174533
                    _da = _dang_ws.unsqueeze(-1)                                         # signed pitch from horizontal (+down / −up)
                    _blk = (((_da >= _lim_ws) & (_prate > 0)) | ((_da <= -_lim_ws) & (_prate < 0))).float()   # block pushing further OUTSIDE ±lim (both crane-down and crane-up)
                    _oerr_ws = _oerr_ws - _blk * _prate * _pdir
                    _rest = torch.clamp((_da - _lim_ws).clamp(min=0.0), max=ROT_SCALE) + torch.clamp((_da + _lim_ws).clamp(max=0.0), min=-ROT_SCALE)   # restore toward nearest ±lim boundary
                    _oerr_ws = _oerr_ws - _pdir * _rest                                  # keep grasp-axis WITHIN ±WSERVO_CLAMP° of horizontal
                eq_tgt = eq                                                           # (6D DLS below uses _oerr_ws directly; eq_tgt only feeds the unused q_des path)
            elif APP_ORI == "policy":                                                 # user 2026-07-04 (b): POLICY controls orientation — a_a[3]=pitch(EE local-x), a_a[4]=yaw(world-Z). Approach LEARNS to orient (keep cube in view) while descending, not scripted.
                _yaw_in = torch.zeros(a_a.shape[0], device=DEVICE) if APP_NO_YAW else a_a[:, 4]   # user 2026-07-18: APP_NO_YAW -> pitch-only orientation (no yaw)
                _oerr_pol = _boresight_oerr(_yaw_in, a_a[:, 3], eq, ROT_SCALE)       # policy-commanded orientation delta (yaw world-Z + pitch EE-x)
                _ak = (ANCHOR_K * max(0.0, 1.0 - g/ANCHOR_ANNEAL)) if ANCHOR_ANNEAL > 0.0 else ANCHOR_K   # user 2026-07-10: anneal anchor 2.0->0 over ANCHOR_ANNEAL env-steps (g), then fully policy
                if _ak > 0.0:                                                          # user 2026-07-10: restoring spring toward handoff pose (_app_ori) — policy CORRECTS around the search aim, no free drift (fixes cube-loss)
                    _qe = eq.clone(); _qe[:, 1:] = -_qe[:, 1:]                          # conj(eq) = inverse (unit quat)
                    _qr = quat_mul(_app_ori, _qe)                                       # world-frame rotation: current eq -> handoff anchor
                    _qr = _qr * torch.sign(_qr[:, :1] + 1e-9)                           # canonical (w>=0 -> shortest arc)
                    _sn = _qr[:, 1:].norm(dim=-1, keepdim=True)                         # |xyz| = sin(theta/2)
                    _th = 2.0*torch.atan2(_sn.squeeze(-1), _qr[:, 0].clamp(-1.0, 1.0))  # angle to anchor (rad)
                    _sp = (_qr[:, 1:]/(_sn + 1e-8)) * (_ak*_th).unsqueeze(-1)           # spring = _ak * rotvec(eq -> anchor)
                    _spn = _sp.norm(dim=-1, keepdim=True)
                    _sp = _sp * torch.clamp(ROT_SCALE/(_spn + 1e-8), max=1.0)           # bound per-step to ROT_SCALE
                    _oerr_pol = _oerr_pol + _sp                                          # policy nudge + rule-based pull back to handoff aim (anneals to 0)
                if WSERVO_CLAMP > 0.0:                                                 # user 2026-07-05: same DOWN-pitch hard clamp as wservo, applied to the POLICY orientation delta (8953 uses clamp instead of the W_WPITCH reward penalty)
                    _gax_p = quat_apply(eq, _HEAD.expand(eq.shape[0], -1))              # NOSE/head direction (wrist->jaw) in world
                    _dang_p = torch.asin(torch.clamp(-_gax_p[:, 2], -1.0, 1.0))
                    _pdir_p = torch.cross(_ZAXIS.expand(eq.shape[0], -1), _gax_p, dim=-1)
                    _pdir_p = _pdir_p / (_pdir_p.norm(dim=-1, keepdim=True) + 1e-8)
                    _prate_p = (_oerr_pol * _pdir_p).sum(dim=-1, keepdim=True)
                    _lim_p = WSERVO_CLAMP*0.0174533
                    _dap = _dang_p.unsqueeze(-1)                                        # signed pitch from horizontal (+down / −up)
                    _blkp = (((_dap >= _lim_p) & (_prate_p > 0)) | ((_dap <= -_lim_p) & (_prate_p < 0))).float()
                    _oerr_pol = _oerr_pol - _blkp * _prate_p * _pdir_p                  # block pushing further OUTSIDE ±lim
                    _restp = torch.clamp((_dap - _lim_p).clamp(min=0.0), max=ROT_SCALE) + torch.clamp((_dap + _lim_p).clamp(max=0.0), min=-ROT_SCALE)
                    _oerr_pol = _oerr_pol - _pdir_p * _restp                            # keep grasp-axis WITHIN ±WSERVO_CLAMP° of horizontal
                if DESC_GATE_PITCH:                                                    # user 2026-07-27: PITCH descent gate — when off-center, project out the pitch component that LOWERS the jaw-mouth (mirror of the EE-z gate at 1739; covers the a_a[3] leak). Deployable (FK eq + fixed JAW_OFF3 + wrist-detect ncen); MUST mirror in deploy.
                    _moz_p = quat_apply(eq, _JAW_OFF3.expand(eq.shape[0], 3))            # jaw-mouth offset r (world)
                    _ncap_p = torch.stack([_moz_p[:, 1], -_moz_p[:, 0], torch.zeros_like(_moz_p[:, 0])], dim=-1)   # ∂(ω×r)_z/∂ω : mouth-z sensitivity to the angular delta
                    _mzv_p = (_oerr_pol * _ncap_p).sum(dim=-1)                           # mouth-z velocity from the pitch (ω·_ncap); <0 = mouth going DOWN
                    _coef_p = (1.0 - _desc_ok) * _mzv_p.clamp(max=0.0) / (_ncap_p.pow(2).sum(dim=-1) + 1e-8)   # off-center(_desc_ok=0) AND down -> removal coef
                    _oerr_pol = _oerr_pol - _coef_p.unsqueeze(-1) * _ncap_p              # cancel the mouth-lowering pitch component when off-center
                eq_tgt = eq
            else:
                eq_tgt = eq                                                           # free (legacy): hold current -> drifts
            ik.set_command(torch.cat([tgt, eq_tgt], dim=-1), ep, eq)                  # user 2026-07-03: orientation target = APP_ORI anchor (was always current eq -> topple)
            q_des = ik.compute(ep, eq, Jb, r.data.joint_pos[:, joint_ids])
            _ik_mask = is_a if (THREE_PHASE and ALIGN_DIRECT) else (is_a | is_l)          # SEARCH is NOT here — it moves ONLY pan+wflex (SEARCH_IK=0 direct, or =1 poshold2 block below)
            _actq = r.data.joint_pos[:, joint_ids]
            arm_term._target_qpos[_ik_mask] = _actq[_ik_mask]                            # sync PD target to ACTUAL before IK servo (drift-fix)
            if APP_ORI in ("cube", "wservo", "policy"):                                  # user 2026-07-03/04: INCREMENTAL damped 6D-Jacobian LS (position + orientation). Stays near actual (no full-IK branch-jump runaway).
                _perr = (tgt - ep)                                                        # position error (N,3)
                if APP_ORI == "cube":
                    _oang = torch.clamp(torch.atan2(_sn.squeeze(-1), (_ax * _aim).sum(-1)), max=ROT_SCALE)   # grasp-axis -> aim angle, bounded
                    _oerr = (_cr / (_sn + 1e-8)) * _oang.unsqueeze(-1)                    # cube-aim orientation error
                elif APP_ORI == "wservo":                                                 # visual-servo omega (keep cube centered in wrist)
                    _oerr = _oerr_ws
                else:                                                                     # policy: orientation error = policy-commanded delta
                    _oerr = _oerr_pol
                _err6 = torch.cat([_perr, _oerr], dim=-1)                                 # (N,6) pose error
                if IK_WRIST_W != 1.0:                                                      # user 2026-07-10 (B): WEIGHTED DLS — penalize wrist_flex motion so ELBOW/LIFT do the reach (was over-cranking wrist). dq = W⁻¹Jᵀ(JW⁻¹Jᵀ+λI)⁻¹e
                    _wiv = torch.ones(6, device=DEVICE); _wiv[WFLEX_I] = 1.0 / IK_WRIST_W
                    _JWi = Jb * _wiv.view(1, 1, 6)                                          # J·W⁻¹ (scale joint columns)
                    _JJt6 = torch.bmm(_JWi, Jb.transpose(1, 2)) + IK_DAMP * torch.eye(6, device=DEVICE).unsqueeze(0)
                    _dq = torch.bmm(Jb.transpose(1, 2), torch.linalg.solve(_JJt6, _err6.unsqueeze(-1))).squeeze(-1) * _wiv.view(1, 6)
                else:
                    _JJt6 = torch.bmm(Jb, Jb.transpose(1, 2)) + IK_DAMP * torch.eye(6, device=DEVICE).unsqueeze(0)
                    _dq = torch.bmm(Jb.transpose(1, 2), torch.linalg.solve(_JJt6, _err6.unsqueeze(-1))).squeeze(-1)
            else:
                _perr = (tgt - ep); _Jp = Jb[:, 0:3, :]                                  # position-only DLS (approach), pose IK (align)
                _JJt = torch.bmm(_Jp, _Jp.transpose(1, 2)) + IK_DAMP * torch.eye(3, device=DEVICE).unsqueeze(0)
                _dq_a = torch.bmm(_Jp.transpose(1, 2), torch.linalg.solve(_JJt, _perr.unsqueeze(-1))).squeeze(-1)
                _dq = torch.where(is_a.unsqueeze(-1), _dq_a, (q_des - _actq))
            app_dq = torch.clamp(_dq / ARM_SCALE, -1.0, 1.0)
            for _j in range(6):
                act_env[:, _j] = torch.where(_ik_mask, app_dq[:, _j], act_env[:, _j])   # IK servo (APPROACH + ALIGN)
            if APP_NO_YAW:                                                            # user 2026-07-18: lock wrist_yaw(4)+wrist_roll(5) in APPROACH -> no wrist twist, 4-joint approach
                act_env[is_a, 4] = 0.0; act_env[is_a, 5] = 0.0
            if SEARCH_IK:                                                             # user 2026-07-07 SEARCH_IK=1 = "poshold2" (8953): 2-DOF (pan+wflex ONLY) DLS against the 6D twist [ZERO linear = HOLD EE ; boresight angular]. The position rows make pan/wflex rotate the camera while MINIMIZING EE translation (partial hold, no full-arm IK). Only pan+wflex move (NOT in _ik_mask). Trade-off: pan (big EE arc) is throttled by the hold. SEARCH_IK=0 = direct pan/wflex deltas (8951), handled by the `if not SEARCH_IK` block above.
                _pax = Jb[:, 3:6, PAN_I]; _wax = Jb[:, 3:6, WFLEX_I]                                        # pan/wflex angular axes (world)
                _paxn = _pax / (_pax.norm(dim=-1, keepdim=True) + 1e-8); _waxn = _wax / (_wax.norm(dim=-1, keepdim=True) + 1e-8)
                _om = (a_s[:, 0].clamp(-1, 1)*SEARCH_ROT*_essc).unsqueeze(-1)*_paxn \
                    + ((a_s[:, 1]*WFLEX_GAIN).clamp(-1, 1)*SEARCH_ROT*_essc).unsqueeze(-1)*_waxn            # desired boresight angular vel: yaw + pitch(*WFLEX_GAIN)
                _J2 = Jb[:, :, [PAN_I, WFLEX_I]]                                                            # (N,6,2) FULL twist columns (linear 0:3 + angular 3:6)
                _tw = torch.cat([torch.zeros_like(_om), _om], dim=-1)                                       # (N,6) ZERO linear (hold EE) + boresight angular
                _JtJ = torch.bmm(_J2.transpose(1, 2), _J2) + IK_DAMP*torch.eye(2, device=DEVICE).unsqueeze(0)
                _dpw = torch.linalg.solve(_JtJ, torch.bmm(_J2.transpose(1, 2), _tw.unsqueeze(-1))).squeeze(-1)   # (N,2) [d_pan,d_wflex]
                _sdq = torch.clamp(_dpw / ARM_SCALE, -1.0, 1.0)
                act_env[:, PAN_I]   = torch.where(is_s, _sdq[:, 0], act_env[:, PAN_I])
                act_env[:, WFLEX_I] = torch.where(is_s, _sdq[:, 1], act_env[:, WFLEX_I])
            if THREE_PHASE and ALIGN_DIRECT:                                         # ALIGN (DIRECT): 6 raw joint-deltas (incl wrist_roll for yaw), while APPROACH stayed IK
                for _j in range(6):
                    act_env[:, _j] = torch.where(is_l, torch.clamp(a_l[:, _j], -1.0, 1.0)*_essc, act_env[:, _j])
            elif THREE_PHASE:                                                        # ALIGN (IK): wrist_roll aligns gripper to cube yaw
                act_env[:, WROLL_I] = torch.where(is_l, (a_l[:, 2].clamp(-1, 1)*_essc), act_env[:, WROLL_I])
        act_env[succ_lat & (ph < 3)] = 0.0                                       # reached grasp height -> hold passive (EXCEPT ph>=3 rule grasp/lift, which drive below)
        act_env[:, 6] = 1.0                                                       # gripper OPEN (approach-only, no grasp yet)
        if RULE_GL and THREE_PHASE:                                               # RULE grasp+lift override (ph3=GRASP hold, ph4=LIFT). Runs AFTER the open-default so it wins.
            _is_g = (ph == 3); _is_lift = (ph == 4); _gl = _is_g | _is_lift
            if bool(_is_lift.any()):                                              # LIFT: IK-raise EE z toward gl_ztgt (xy+orientation held), gentle RGL_STEP/step
                _ep = r.data.body_pos_w[:, ee_body_idx, :]; _eq = r.data.body_quat_w[:, ee_body_idx, :]
                _t = _ep.clone(); _t[:, 2] = torch.minimum(_ep[:, 2] + RGL_STEP, gl_ztgt)
                ik.set_command(torch.cat([_t, _eq], dim=-1), _ep, _eq)
                arm_term._target_qpos[_is_lift] = r.data.joint_pos[:, joint_ids][_is_lift]   # user 2026-07-02: same drift-fix for LIFT IK (sync PD target to actual before servo)
                _ldq = torch.clamp((ik.compute(_ep, _eq, Jb, r.data.joint_pos[:, joint_ids]) - arm_term._target_qpos) / ARM_SCALE, -1.0, 1.0)
                for _j in range(6):
                    act_env[:, _j] = torch.where(_is_lift, _ldq[:, _j], act_env[:, _j])
            for _j in range(6):
                act_env[:, _j] = torch.where(_is_g, torch.zeros_like(act_env[:, _j]), act_env[:, _j])   # GRASP: hold pose (no arm delta)
            act_env[:, 6] = torch.where(_gl, torch.full_like(act_env[:, 6], -1.0), act_env[:, 6])        # CLOSE gripper during grasp+lift
        return act_env

    if _EVAL_CASES is not None:   # user 2026-07-23: PLACE eval-set cubes at STARTUP so the FIRST round already runs assigned cases -> counts. Was: first round = default spawn, _env_case=-1 -> uncounted -> ~half the bench wasted.
        _rid = torch.arange(N, device=DEVICE); _no = N
        _ci = (_eval_cursor[0] + torch.arange(_no, device=DEVICE)) % _EVAL_CASES.shape[0]
        if _eval_cursor[0] + _no >= _EVAL_CASES.shape[0]: _eval_sweeps[0] += 1
        _eval_cursor[0] = int((_eval_cursor[0] + _no) % _EVAL_CASES.shape[0])
        _env_case[_rid] = _ci
        _cs = _EVAL_CASES[_ci]
        _cpw = env.scene[args.target_cube].data.root_pos_w[_rid].clone()
        _cpw[:, 0] = env.scene.env_origins[_rid, 0] + _cs[:, 0]; _cpw[:, 1] = env.scene.env_origins[_rid, 1] + _cs[:, 1]
        _hy = _cs[:, 2] * 0.5
        _q = torch.zeros(_no, 4, device=DEVICE); _q[:, 0] = torch.cos(_hy); _q[:, 3] = torch.sin(_hy)
        env.scene[args.target_cube].write_root_pose_to_sim(torch.cat([_cpw, _q], dim=-1), env_ids=_rid)
        _jpr = r.data.joint_pos[_rid].clone(); _jpr[:, int(joint_ids[3])] = _cs[:, 3]
        r.write_joint_state_to_sim(_jpr, torch.zeros_like(_jpr), env_ids=_rid)
        arm_term._target_qpos[_rid, 3] = _cs[:, 3]
        env.sim.forward(); env.sim.render()
        print(f"[EVAL-SET] startup: placed {_no} eval cubes (first round now counts)", flush=True)
    while g < args.total_timesteps:
        if _curric_on and CURRIC_EXT and CURRIC_WRITE and (g // N) % 20 == 0:   # user 2026-07-23: CONSUMER — poll for an external _cw computed by the .4 sweep worker (never sweeps here, so training never pauses)
            try:
                _emt = os.path.getmtime(CURRIC_WRITE)
                if _emt > _cw_ext_mtime[0]:
                    import numpy as _npL; _wv = _npL.load(CURRIC_WRITE)
                    if _wv.shape[0] == _NCELL:
                        _cw = torch.tensor(_wv, device=DEVICE, dtype=torch.float32); _cw_ext_mtime[0] = _emt
                        print(f"[CURRIC-EXT] consumer loaded external _cw ({_wv.shape[0]} cells, w {float(_cw.min()):.2f}~{float(_cw.max()):.2f}) from {CURRIC_WRITE}", flush=True)
            except Exception:
                pass
        if _curric_on and (not CURRIC_EXT) and (not _sweeping[0]) and g >= _next_sw[0]:   # user 2026-07-14: time to self-eval -> enter deterministic sweep over the grid (mean actions, no training, sequential cells). 2026-07-23: skipped in CURRIC_EXT consumer mode (sweep offloaded to .4).
            _sweeping[0] = True; _SWEEPING[0] = True; _swcur[0] = 0; _cur_seen.zero_(); _cur_hit.zero_(); _cur_srch.zero_(); _cur_cell.fill_(-1)
            if _NOM_KP[0] is None:                                                # user 2026-07-22: capture nominal (cfg) arm gains once — held during every sweep so the sweep is DR-free (== bench)
                try:
                    _NOM_KP[0] = r.data.default_joint_stiffness.clone(); _NOM_KD[0] = r.data.default_joint_damping.clone()
                    print(f"[SWEEP-NODR] captured nominal gains kp[0]={[round(float(x),1) for x in _NOM_KP[0][0].tolist()]} — actuator-gain DR held at nominal during sweep (== no-DR bench)", flush=True)
                except Exception as _e:
                    print(f"[SWEEP-NODR] nominal-gain capture FAILED ({_e}); sweep keeps training DR", flush=True)
            print(f"[CURRIC] === self-eval sweep START @ step {g:,} ({_NCELL} cells) ===", flush=True)
        if TP_EVAL:                                              # STEP-EVAL: block until STEP / RESET / BACK (or PLAY)
            _wtick = 0                                           # while PAUSED: place the cube LIVE from the sliders + refresh view, so it goes exactly where you drag (no STEP needed)
            while (_TPUI["stepreq"] <= _last_step and _TPUI.get("resetreq",0) <= _last_reset
                   and _TPUI.get("backreq",0) <= _last_back and _TPUI.get("forcereq",0) <= _last_force and not _TPUI["play"]):
                try:
                    _cp = env.scene[args.target_cube].data.root_pos_w.clone()
                    _cb = _TPUI["cube"]; _cp[:,0]=_cb[0]; _cp[:,1]=_cb[1]; _cp[:,2]=_cb[2]
                    env.scene[args.target_cube].write_root_pose_to_sim(torch.cat([_cp, env.scene[args.target_cube].data.root_quat_w], dim=-1))
                    env.sim.forward()
                    if _wtick % 6 == 0:                          # ~0.3s: re-render cameras (paused) + rerun detector + redraw so the cube/signals track the slider live
                        try: env.scene.update(0.0)
                        except Exception: pass
                        for _ in range(2): env.sim.render()
                        fd, wd, ori, s = get_obs(); _tp_refresh(fd, wd)
                        _cw0 = env.scene[args.target_cube].data.root_pos_w[0].detach().cpu().tolist()
                        _TPUI["cubew"] = [round(float(x),3) for x in _cw0]
                        _TPUI["ncen0"]=float(torch.norm(wd[0,0:2]-0.5)); _TPUI["frac0"]=float(wd[0,2]); _TPUI["vis0"]=float(wd[0,3]>0.5)
                except Exception: pass
                _wtick += 1; _tm_eval.sleep(0.05)
            if _TPUI.get("resetreq",0) > _last_reset:            # RESET -> NEW ENVIRONMENT (env.reset: random cube + arm to start). no control step.
                _last_reset = _TPUI["resetreq"]
                try:
                    env.reset()
                    arm_term._target_qpos.copy_(r.data.joint_pos[:, joint_ids])   # CRITICAL: re-sync PD target to the reset pose, else the arm lurches (all joints) chasing the stale pre-reset target
                    ph = torch.zeros(N, dtype=torch.long, device=DEVICE); chold = torch.zeros(N, device=DEVICE)
                    lhold = torch.zeros(N, device=DEVICE); succ_lat = torch.zeros(N, dtype=torch.bool, device=DEVICE)
                    _tp_undo.clear(); _tp_frames.clear(); _prev_ph0 = 0   # NEW ENV: also reset live clip (user 2026-07-02)
                    for _hk in _TPHIST: _TPHIST[_hk].clear()   # NEW ENV: clear sparklines/history + cumulatives + RSR window so UI shows only the current env (user 2026-07-02)
                    _cum["s"] = _cum["a"] = _cum["l"] = 0.0
                    _tp_recent.clear(); _gzmin_recent.clear(); _lift_recent.clear()
                    _nc = env.scene[args.target_cube].data.root_pos_w[0].detach().cpu().tolist()
                    _TPUI["cube"][0], _TPUI["cube"][1], _TPUI["cube"][2] = _nc[0], _nc[1], _nc[2]
                    fd, wd, ori, s = get_obs(); gz_prev = r.data.body_pos_w[:, ee_body_idx, 2].clone()
                    _TPUI["phase0"] = 0; _TPUI["plog"] = (_TPUI.get("plog", []) + [f"step {g:,}: 🔄 RESET → SEARCH (new env)"])[-8:]
                    _tp_refresh(fd, wd)
                    print(f"[TP-EVAL] RESET -> new env, cube={[round(x,3) for x in _nc]}", flush=True)
                except Exception as _e: print(f"[TP-EVAL] reset fail: {_e}", flush=True)
                continue
            if _TPUI.get("backreq",0) > _last_back:               # BACK -> restore the snapshot taken before the last step. no control step.
                _last_back = _TPUI["backreq"]
                if _tp_undo:
                    _snp = _tp_undo.pop()
                    try:
                        r.write_joint_state_to_sim(_snp["jp"], _snp["jv"])
                        env.scene[args.target_cube].write_root_pose_to_sim(torch.cat([_snp["cu"], _snp["cq"]], dim=-1))
                        if _snp["tq"] is not None: arm_term._target_qpos.copy_(_snp["tq"])
                        env.sim.forward()
                        ph = _snp["ph"]; chold = _snp["ch"]; lhold = _snp["lh"]; succ_lat = _snp["sl"]; gz_prev = _snp["gz"]
                        _TPUI["cube"][0], _TPUI["cube"][1], _TPUI["cube"][2] = _snp["sld"]
                        fd, wd, ori, s = get_obs(); _prev_ph0 = int(ph[0].item()); _TPUI["phase0"] = _prev_ph0
                        _TPUI["plog"] = (_TPUI.get("plog", []) + [f"step {g:,}: ⏮ BACK ({len(_tp_undo)} left)"])[-8:]
                        _tp_refresh(fd, wd)
                        print(f"[TP-EVAL] BACK -> {len(_tp_undo)} snapshots left", flush=True)
                    except Exception as _e: print(f"[TP-EVAL] back fail: {_e}", flush=True)
                else:
                    print("[TP-EVAL] BACK: no history", flush=True)
                continue
            if _TPUI.get("forcereq",0) > _last_force:              # FORCE: manually toggle env0 phase (test each policy alone). no control step.
                _last_force = _TPUI["forcereq"]
                ph[0] = 1 - int(ph[0].item()); chold[0] = 0; lhold[0] = 0; succ_lat[0] = False
                _prev_ph0 = int(ph[0].item()); _TPUI["phase0"] = _prev_ph0
                _fp = "APPROACH" if _prev_ph0 else "SEARCH"
                _TPUI["plog"] = (_TPUI.get("plog", []) + [f"step {g:,}: ⚙ FORCED → {_fp}"])[-8:]
                print(f"[TP-EVAL] FORCE -> env0 phase = {_fp}", flush=True)
                continue
            _last_step = _TPUI["stepreq"]
            try:
                if not _TPUI["play"]:                                # user 2026-07-14: manual step pins the cube to the UI slider; FREE-RUN (autoplay/EVAL_SET) leaves it where the reset-block spawn + physics placed it (else the per-step pin would clobber the eval-set position -> policy searches a phantom cube -> 0% RSR)
                    _cp = env.scene[args.target_cube].data.root_pos_w.clone()
                    _cb = _TPUI["cube"]; _cp[:,0]=_cb[0]; _cp[:,1]=_cb[1]; _cp[:,2]=_cb[2]
                    env.scene[args.target_cube].write_root_pose_to_sim(torch.cat([_cp, env.scene[args.target_cube].data.root_quat_w], dim=-1))
                    env.sim.forward()
            except Exception as _e: print(f"[TP-EVAL] setcube fail: {_e}", flush=True)
        if TP_EVAL or (_curric_on and _sweeping[0]):             # deterministic (mean) actions for eval + curriculum self-eval sweep (user 2026-07-16: reverted stochastic — noise tanks APPROACH precision -> RSR 13.6%->1%; deterministic is the deploy mode)
            with torch.no_grad():
                a_s = srch["ac"].mean_action(fd, wd, ori, s); a_a = appr["ac"].mean_action(fd, wd, ori, s)
                a_l = algn["ac"].mean_action(fd, wd, ori, s) if THREE_PHASE else None
        else:
            with torch.no_grad():
                a_s, _ = srch["ac"].get_action(fd, wd, ori, s)
                a_a, _ = appr["ac"].get_action(fd, wd, ori, s)
                a_l = (algn["ac"].get_action(fd, wd, ori, s)[0]) if THREE_PHASE else None
        if g < args.learning_starts and not TP_EVAL:
            a_s = torch.rand(N, N_ACT_S, device=DEVICE)*2-1; a_a = torch.rand(N, N_ACT_A, device=DEVICE)*2-1
            if THREE_PHASE: a_l = torch.rand(N, 6 if (DIRECT_JOINTS or ALIGN_DIRECT) else 4, device=DEVICE)*2-1
        is_s = (ph == 0); is_a = (ph == 1); is_l = (ph == 2)
        # user 2026-07-16: action-rate (smoothness) penalty — |Δa| per active env, gated same-phase-both-steps (avoids phase-transition spike). subtracted from r_s/r_a below.
        _smooth_s = W_SMOOTH * (a_s - _prev_a_s).abs().mean(dim=1) * (is_s & _prev_is_s).float()
        act_env = _compute_act_env(a_s, a_a, a_l, is_s, is_a, is_l, s, wd, ph)
        if TP_EVAL:                                                               # snapshot BEFORE stepping so BACK can rewind one control step
            _tp_undo.append({"jp": r.data.joint_pos.clone(), "jv": r.data.joint_vel.clone(),
                             "cu": env.scene[args.target_cube].data.root_pos_w.clone(), "cq": env.scene[args.target_cube].data.root_quat_w.clone(),
                             "tq": arm_term._target_qpos.clone() if hasattr(arm_term, "_target_qpos") else None,
                             "ph": ph.clone(), "ch": chold.clone(), "lh": lhold.clone(), "sl": succ_lat.clone(),
                             "gz": gz_prev.clone(), "sld": list(_TPUI["cube"])})
            if len(_tp_undo) > 100: _tp_undo.pop(0)
        _qb_dbg = r.data.joint_pos[0, joint_ids].clone() if TP_EVAL else None       # DBG: joints BEFORE step
        _tps = time.perf_counter() if _PROF_ON else 0.0
        _pjp = r.data.joint_pos.detach().clone(); _pjv = r.data.joint_vel.detach().clone()   # DIAG: last-finite (pre-step) joint state to reveal the divergence trigger
        env.step(act_env)
        if _curric_on and _sweeping[0] and _NOM_KP[0] is not None:               # user 2026-07-22: DR-off during sweep (guard _curric_on FIRST — _sweeping/_NOM_KP only exist when curriculum is on; short-circuit avoids NameError under CURRIC=0) — actuator-gain DR is the ONLY dynamics DR that affects the reach RSR (GRASP_ON=0 -> cube mass/friction + gripper friction don't). Pin gains to nominal each sweep step (overrides the reset-time DR re-roll) so sweep == no-DR bench.
            try: r.write_joint_stiffness_to_sim(_NOM_KP[0]); r.write_joint_damping_to_sim(_NOM_KD[0])
            except Exception: pass
        if _PROF_ON: _psync(); _PROF["step"] += time.perf_counter() - _tps; _PROF["n"] += 1
        if TP_EVAL and _qb_dbg is not None:                                          # DBG: which joints were COMMANDED vs which actually MOVED (env0)
            _qa = r.data.joint_pos[0, joint_ids]; _dq = (_qa - _qb_dbg)
            _nm = ["pan","lift","elbow","wflex","wyaw","wroll"]
            _cmd = ["%s=%+.3f"%(_nm[_j], float(act_env[0,_j])) for _j in range(6)]
            _mov = ["%s=%+.3f"%(_nm[_j], float(_dq[_j])) for _j in range(6)]
            print("[TPDBG ph=%d] CMD %s | MOVED %s"%(int(ph[0].item()), " ".join(_cmd), " ".join(_mov)), flush=True)
        _PH_CUR[0] = ph                                                           # user 2026-07-19: expose current per-env phase so get_obs applies privilege to APPROACH envs only (+ stashes the front-guided acquisition projection for r_s)
        nfd, nwd, nori, ns = get_obs()
        gz = r.data.body_pos_w[:, ee_body_idx, 2]
        ncen = torch.sqrt((nwd[:, 0] - CX_TGT)**2 + (CY_W*(nwd[:, 1] - CY_TGT))**2); nvis = (nwd[:, 3] > 0.5).float()   # centering err w/ H/V retarget(CX_TGT,CY_TGT)+weight(CY_W) — search & approach
        seen = torch.maximum(seen, nvis)                                          # latch: cube has been seen in this phase
        cen_r, _cen_a = _cen_rewards(ncen)                                        # SEARCH (K_CE/CEN_K) + APPROACH (K_CE_A/CEN_K_A) centering rewards
        _mouth_w = r.data.body_pos_w[:, ee_body_idx] + quat_apply(r.data.body_quat_w[:, ee_body_idx], _JAW_OFF3.expand(N, 3))   # jaw MOUTH (FK): gripper_link origin + _goff (link->mouth geometry)
        _recon_w, _rz_w = cube_xyz_from_wrist(nwd[:, :2], _cube_plane_z)                          # user 2026-07-22: DEPLOYABLE wrist-detection ray ∩ table-plane -> cube world xy
        _wv = (nwd[:, 3] > 0.5) & (_rz_w > RZ_MIN_W)                                              # wrist RELIABLE (sees cube AND steep ray)
        _mouth_xy_err = torch.norm(_recon_w[:, :2] - _mouth_w[:, :2], dim=-1)                     # metric MOUTH<->cube horizontal (m), height-independent (fixes pixel-parallax lateral drift)
        _tabpen = _table_penalty(gz, _mouth_w)                                    # PENETRATION penalty (linear + steep quadratic below the reference)
        gz_min = torch.minimum(gz_min, gz)                                        # track lowest reached
        _mz_min = torch.minimum(_mz_min, _mouth_w[:, 2])                          # user 2026-07-22: lowest jaw-MOUTH z this episode
        _hz = gz                                # REFACTOR 2026-07-27: descent height = EE gz (MOUTH_DESC removed)
        _hz_min = gz_min; _hz_prev = gz_prev
        _cube_w = env.scene[args.target_cube].data.root_pos_w
        _dcube = torch.norm(_mouth_w - _cube_w, dim=-1)   # TRUE 3D jaw-mouth<->cube dist (monitor-only, privileged; ->0 at a real grasp, was link-origin => floored at ||_goff||=0.134)
        _dcube_h = torch.norm(_mouth_w[:, :2] - _cube_w[:, :2], dim=-1)   # user 2026-07-27: HORIZONTAL (top-down "from above") privileged mouth<->cube dist -> rsr_old uses THIS (3D conflated side+vertical; "over the cube from above" = xy only)
        _diverged = ~(torch.isfinite(r.data.body_pos_w).flatten(1).all(dim=1) & torch.isfinite(r.data.joint_pos).all(dim=1) & torch.isfinite(r.data.joint_vel).all(dim=1) & torch.isfinite(_cube_w).all(dim=1))   # ROOT FIX (8953/official9 crash): PhysX integrated a body to a non-finite pose ("OrthogonalizeBasis did not converge") -> NaN FK. The nan_to_num in get_obs only MASKS it (feeds zeros forever); here we detect the diverged env from the RAW sim state and route it through the SAME corrupted-env force-reset as a flown cube.
        if bool(_diverged.any()):
            _dids = torch.nonzero(_diverged, as_tuple=False).squeeze(-1)
            print(f"[DIVERGED] PhysX non-finite -> force-reset {int(_diverged.sum())} env(s): {_dids.tolist()[:16]}", flush=True)
            for _di in _dids.tolist()[:4]:   # DIAG: last-finite (pre-step) arm state -> limit-slam vs velocity-runaway vs contact
                _jp = _pjp[_di, joint_ids]; _jv = _pjv[_di, joint_ids]
                print(f"[DIVERGED-DIAG] env{_di} prev q(rad)={[round(float(x),3) for x in _jp]} prev qd={[round(float(x),2) for x in _jv]} |qd|max={float(_jv.abs().max()):.2f}", flush=True)
        _cube_gone = _diverged | ((_cube_w - env.scene.env_origins)[:, :2].norm(dim=-1) > CUBE_GONE_R) | (_cube_w[:, 2] < -0.5) | (_cube_w[:, 2] > 1.5)   # user 2026-07-10: cube knocked far / off table / launched (physics) -> corrupted env (drop from success + metrics, force-reset). 2026-07-21: + non-finite robot FK (_diverged)
        if bool(_diverged.any()):   # ROOT FIX 2026-07-21: a diverged env's FK is NaN -> its reward/metrics are NaN. get_obs's nan_to_num sanitizes the OBS but NOT the reward -> a NaN reward stored in the buffer poisons the C51 loss -> NaN weights -> Normal(loc=nan) crash in the UPDATE (what killed official10-era 8953). Drop diverged envs from _cum / phase-transitions / the BUFFER; they are force-reset next step anyway.
            is_s = is_s & ~_diverged; is_a = is_a & ~_diverged; is_l = is_l & ~_diverged
        dcube_min = torch.minimum(dcube_min, _dcube)                              # closest the gripper got to the cube this episode
        # user 2026-07-20: UNIFIED success/RSR condition = DEPLOYABLE center + frac + height (NO privileged dist).
        #   _reached drives BOTH the RSR metric (held RSR_HOLD) AND appr_ready->succ (held SUCC_HOLD) -> one criterion.
        _dw3d = torch.norm(_recon_w - _mouth_w, dim=-1)   # user 2026-07-27: DEPLOYABLE 3D dist (wrist-unproj cube(xy,plane_z) <-> jaw-MOUTH FK) = "actual distance" for RSR
        _hgate = (_dw3d < APP_D3) if APP_D3 > 0.0 else ((_mouth_w[:, 2] < APP_MZ) if APP_MZ > 0.0 else (gz < APP_GZ))   # user 2026-07-27: APP_D3>0 -> gate on the DEPLOYABLE actual 3D distance (actual distance) instead of absolute mouth-z height
        _mxy_gate = ((_mouth_xy_err < MOUTH_XY) & _wv) if MOUTH_XY > 0.0 else torch.ones(N, dtype=torch.bool, device=DEVICE)   # user 2026-07-22: metric MOUTH-over-cube horizontal (deployable, height-independent)
        _frac_gate = (nwd[:, 2] > FRAC_GRASP) if FRAC_GRASP > 0.0 else (nvis > 0.5)   # user 2026-07-24: FRAC_GRASP<=0 -> DROP the frac size gate (unreachable ~0.02 in wrist at grasp range); require DETECTION (nvis) instead so success still needs the cube seen (not stale-default ncen)
        _eff_appcen = SWEEP_APP_CEN if (_curric_on and _sweeping[0]) else APP_CEN   # user 2026-07-26: loose centering ONLY during sweep
        _reached = ((is_a | is_s) if SUCCESS_IN_SEARCH else is_a) & (ncen < _eff_appcen) & _hgate & _frac_gate & (~_cube_gone) & _mxy_gate   # user 2026-07-24: SUCCESS_IN_SEARCH -> success also fires in SEARCH (phase-agnostic); frac dropped -> center + mouth-height + detection
        _rsr_hold = torch.where(_reached, _rsr_hold + 1.0, torch.zeros_like(_rsr_hold))   # consecutive frames MEETING the unified success condition
        _rsr_holdmax = torch.maximum(_rsr_holdmax, _rsr_hold)                     # longest run this episode -> RSR-hit needs >= RSR_HOLD
        _rsr_old_ok = (_dcube < RSR_DCUBE) & ((_dcube_h < RSR_MXY) if RSR_MXY > 0.0 else torch.ones(N, dtype=torch.bool, device=DEVICE))   # user 2026-07-27: PRIVILEGED = TRUE 3D dist < RSR_DCUBE AND TRUE horizontal xy < RSR_MXY ("over the cube from above")
        _rsr_old_hold = (torch.where(_rsr_old_ok, _rsr_old_hold + 1.0, torch.zeros_like(_rsr_old_hold)) if RSR_DCUBE > 0.0 else _rsr_old_hold)   # PRIVILEGED reference (3D + horizontal xy) -> UI "rsr_old" ONLY (not reward/termination)
        _rsr_old_holdmax = torch.maximum(_rsr_old_holdmax, _rsr_old_hold)
        ncen_min = torch.where(nvis > 0.5, torch.minimum(ncen_min, ncen), ncen_min)   # best (smallest) centering error reached
        frac_max = torch.where(nvis > 0.5, torch.maximum(frac_max, nwd[:, 2]), frac_max)  # biggest (closest) frac reached
        _reg_s = W_REG * (ncen - ncen_min).clamp(min=0.0) * nvis                  # SEARCH regression: moved away from center after getting close
        # user 2026-07-19: FRONT-GUIDED acquisition term — when the WRIST can't see the cube (nvis<0.5) but FRONT can (pv),
        #   pull the FRONT-predicted wrist position toward the vertical target (CY_TGT) so the policy AIMS the camera to acquire.
        #   tanh-inside-expo (same shape/peak as cen_r): peak K_CE at aim, ->0 when the predicted cube is far off-frame. search-only (gated ×(1-nvis)); hands the reward off to W_SC wrist-centering the instant the wrist actually sees it.
        if W_ACQ_F > 0.0 and _ACQ.get("pv") is not None:
            _pncen = torch.sqrt((_ACQ["pcx"] - CX_TGT)**2 + (CY_W*(_ACQ["pcy"] - CY_TGT))**2)
            _rt_acq = W_ACQ_F * (torch.exp(K_CE*(1.0 - torch.tanh(ACQ_K*_pncen))) - 1.0) * _ACQ["pv"].float() * (1.0 - nvis)
        else:
            _rt_acq = torch.zeros(N, device=DEVICE)
        _up_pen_s = W_UP_SRCH * (quat_apply(r.data.body_quat_w[:, ee_body_idx], _HEAD.expand(N, -1))[:, 2] - UP_SRCH_SIN).clamp(min=0.0)**2 * is_s.float()   # user 2026-08-01: head-up GRADUATED penalty
        r_s = W_SV*(nvis - 0.5) + W_SC*cen_r*nvis - W_NOACQ*(1.0 - nvis) + _rt_acq - _tabpen - _reg_s - _smooth_s - _up_pen_s   # user 2026-07-03: SEARCH = see + CENTER(combined) − NOT-acquired penalty − table − regression. (2026-07-19: + FRONT-GUIDED acquisition _rt_acq — dense aim gradient when wrist-blind)
        _d = ((Z_HIGH_REW - _hz)/(Z_HIGH_REW - Z_APPROACH)).clamp(0.0, 1.0)   # user 2026-07-16: descend reward maxes at the HANDOFF height (Z_APPROACH), NOT grasp (Z_GRASP). 2026-07-22: _hz = mouth-z under MOUTH_DESC -> reward lowering the MOUTH (Z_APPROACH becomes a mouth-frame target).
        _doff = ncen - DGATE_CEN                                                                      # distance past the center-pivot (<0 = closer than pivot, >0 = farther)
        _csign = (-torch.sign(_doff) * (torch.exp(DGATE_K * _doff.abs().clamp(max=0.35)) - 1.0)) if DGATE_CEN > 0 else torch.ones_like(ncen)   # user 2026-07-04: EXP signed centering — closer = expo +, farther = expo − (both grow exponentially; DGATE_K=sharpness, DGATE_CEN=pivot)
        _gax_w = quat_apply(r.data.body_quat_w[:, ee_body_idx, :], _HEAD.expand(N, 3))                # NOSE dir (wrist->jaw) in world — the visible "head"
        _downang = torch.asin(torch.clamp(_gax_w[:, 2], -1.0, 1.0))                                   # user 2026-07-06: SIGN FIX — + = tilt down (nose down), − = tilt back (nose up), 0 = level (matches the video; was inverted). penalty uses |·| so symmetric behavior unchanged.
        _frn = ((nwd[:, 2] - FRAC_REF)/FRAC_CAP).clamp(-1.0, 1.0)   # user 2026-07-09: signed normalized frac vs FRAC_REF -> two-sided W_AF (frac>ref +expo, <ref -expo). FRAC_REF=0 -> _frn>=0 = old one-sided.
        if CEN_REF > 0:                                             # user 2026-07-09: TWO-SIDED approach centering — ncen<CEN_REF +expo(center), >CEN_REF -expo(moving away)
            _cnn = ((CEN_REF - ncen)/CEN_REF).clamp(-1.0, 1.0)
            _cen_a2 = torch.sign(_cnn)*(torch.exp(K_CE_A*_cnn.abs()) - 1.0)
        else:
            _cen_a2 = _cen_a                                        # old one-sided (tanh-inside-expo, 0..+)
        _agate = _csign if DESC_GATE else torch.ones_like(_csign)                                     # user 2026-07-09: DESC_GATE=0 -> PLAIN descent (no centering gate); centering handled separately by W_AC
        # user 2026-07-09: cube_rel horizontal-align reward (wrist + front unproj, two-sided expo, gated on that cam's reliable detection). deployable, non-privileged. penalizes "sit back".
        _ee_xy = r.data.body_pos_w[:, ee_body_idx, :2]                                             # ALWAYS compute front+wrist cube-dist (reward + UI validation)
        _rw, _rzw = cube_xyz_from_wrist(nwd[:, :2], _cube_plane_z)
        _dw = torch.norm(_rw[:, :2] - _ee_xy, dim=-1); _vw = (nwd[:, 3] > 0.5) & (_rzw > RZ_MIN_W)   # wrist cube horizontal dist (m) + STEEP-reliable mask (RZ_MIN_W)
        _rf2, _rzf2 = cube_xyz_from_front(nfd[:, :2], _cube_plane_z)
        _df = torch.norm(_rf2[:, :2] - _ee_xy, dim=-1); _vf = (nfd[:, 3] > 0.5) & (_rzf2 > RZ_MIN)   # front cube horizontal dist (m) + reliable
        if DESC_D > 0.0:   # user 2026-07-10: COMBINED descent gate — front HORIZONTAL proximity(_df) AND centering(ncen). override _csign.
            _gprox = torch.where(_vf, ((2.0*DESC_D - _df)/DESC_D).clamp(0.0, 1.0), torch.zeros(N, device=DEVICE))   # 1 at _df<=DESC_D(over cube) -> 0 at _df>=2*DESC_D ; front-invalid -> 0 (no blind descend)
            _dcref = DGATE_CEN if DGATE_CEN > 0.0 else 0.16
            _gcen = ((2.0*_dcref - ncen)/_dcref).clamp(0.0, 1.0)   # 1 at ncen<=_dcref(centered) -> 0 at >=2*_dcref
            _agate = _gprox * _gcen   # AND: descend+ only when OVER cube AND centered
        _df3 = torch.norm(_rf2 - _mouth_w, dim=-1); _dw3 = torch.norm(_rw - _mouth_w, dim=-1)   # user 2026-07-10: DEPLOYABLE 3D dist (unproj cube (X,Y,plane_z) <-> jaw MOUTH FK) — approximates privileged _dcube w/o cube GT. plane_z=known cube-center height. MONITOR (UI vs _dcube to validate unproj accuracy).
        _tri, _tgap = cube_xyz_triangulate(nfd[:, :2], nwd[:, :2]); _dtri = torch.norm(_tri - _mouth_w, dim=-1)   # user 2026-07-10: STEREO 3D (front+wrist rays, NO plane assumption) <-> mouth. Reliable when both cams valid AND rays consistent (gap small).
        _crw_seen = torch.maximum(_crw_seen, _vw.float())        # user 2026-07-09: latch — wrist has RELIABLY seen the cube this phase
        _crf_seen = torch.maximum(_crf_seen, _vf.float())        # front latch
        # user 2026-07-14: DWELL reward — consecutive frames within DWELL_D (DEPLOYABLE 3D dist, front OR wrist valid, like _cr3_ok) -> EXPO growth
        # user 2026-07-10: named terms so the UI can show per-term reward breakdown (find what farms reward w/o improving ncen/dist)
        _rt_desc = W_DEPTH*(torch.exp(EXP_KZ*_d) - 1.0)*nvis*_agate                                    # descend LEVEL (× _csign if DESC_GATE=1)
        _rt_frac = W_AF*torch.sign(_frn)*(torch.exp(EXP_KF*_frn.abs()) - 1.0)*nvis*is_a.float()   # REFACTOR 2026-07-27: frac LEVEL two-sided, UNGATED (DESC_MXY/FRAC_GZ_GATE removed) — pure closeness driver
        _rt_novisA = -W_NOVIS_A*(1.0 - nvis)                                                          # not-visible penalty
        r_a = _rt_frac + _rt_desc - _tabpen + _rt_novisA   # REFACTOR 2026-07-27: frac closeness + gated descend + guards(table,novis)
        if bool(is_s.any()): _cum["s"] += float(r_s[is_s].mean().item())         # cumulative mean reward per phase (progress)
        if bool(is_a.any()): _cum["a"] += float(r_a[is_a].mean().item())
        _ma = (lambda _t: float(_t[is_a].mean().item())) if bool(is_a.any()) else (lambda _t: 0.0)   # user 2026-07-10: mean of a term over APPROACH envs (per-term reward breakdown)
        centered = (ncen < (SWEEP_CENTER if (_curric_on and _sweeping[0]) else SEARCH_CENTER)) & (nvis > 0.5)   # user 2026-07-26: loose handoff center ONLY during sweep
        chold = torch.where(centered, chold + 1, torch.zeros_like(chold))
        to_app = is_s & (chold >= SEARCH_HOLD)
        lost = ((seen > 0.5) & (nvis < 0.5))                 # user 2026-07-03: LOST = cube VANISHED only. Center-based loss `| (ncen > LOSE_CEN)` COMMENTED OUT — APPROACH must OWN centering (learn to re-center in place), not bail back to SEARCH whenever off-center.
        lhold = torch.where(is_a & lost, lhold + 1, torch.where(is_a, torch.zeros_like(lhold), lhold))
        _srch_hold = GONE_HOLD if (NO_RESRCH and RESRCH_GONE) else LOSE_HOLD                                          # user 2026-07-24: RESRCH_GONE uses GONE_HOLD (complete-loss re-acquire)
        to_srch = (is_a & (lhold >= _srch_hold)) if (not NO_RESRCH or RESRCH_GONE) else torch.zeros(N, dtype=torch.bool, device=DEVICE)   # NO_RESRCH: no bail; +RESRCH_GONE: bail to SEARCH only when cube COMPLETELY vanished (lost) held GONE_HOLD (off-center still never bails)
        term = env.termination_manager.dones if hasattr(env, "termination_manager") else torch.zeros(N, dtype=torch.bool, device=DEVICE)
        tmo = term.bool() if torch.is_tensor(term) else torch.zeros(N, dtype=torch.bool, device=DEVICE)
        # APPROACH "ready" = cube close + centered + visible (REAL/detection). 2-phase: episode success; 3-phase: hand off to ALIGN.
        _cr_ok = (((_dw + _df)*0.5 < CR_APP) & _vw & _vf) | (CR_APP <= 0.0)   # user 2026-07-09: CR_APP>0 -> mouth OVER cube (wrist+front unproj horizontal dist avg < CR_APP, both cams valid). deployable. 0 = off
        _cr3_ok = (((_df3 < APP_CR3) & _vf) | ((_dw3 < APP_CR3) & _vw)) | (APP_CR3 <= 0.0)   # user 2026-07-10: appr_ready gate = EITHER reliable cam reaches (front far / wrist near — near grasp front occludes, wrist steep). deployable. 0 = off
        appr_ready = _reached   # user 2026-07-20: UNIFIED with the RSR metric (center + frac + height, deployable). Old deployable/privileged gates (_cr_ok/_cr3_ok/APP_DCUBE) folded away — set those knobs to 0 to keep this pure.   # user 2026-07-10: gate = CENTERED(ncen) + DESCENDED(gz) + DEPLOYABLE 3D dist<APP_CR3 (front+wrist) + cube NOT flown away (~_cube_gone, no spoofed success on a physics-corrupted env). CR_APP/APP_DCUBE off (=APP_CR3 replaces them).
        if THREE_PHASE:
            ahold = torch.where(appr_ready, ahold + 1, torch.where(is_a, torch.zeros_like(ahold), ahold))
            to_align = is_a & (ahold >= APP_HOLD)
            succ = torch.zeros(N, dtype=torch.bool, device=DEVICE)            # 2-phase success replaced by ALIGN in-mouth
        else:
            to_align = torch.zeros(N, dtype=torch.bool, device=DEVICE)
            ahold = torch.where(appr_ready, ahold + 1, torch.where(is_a, torch.zeros_like(ahold), ahold))   # 2-phase: hold ready SUCC_HOLD frames = stable grasp (user 2026-07-02)
            succ = is_a & (ahold >= SUCC_HOLD) & (~succ_lat)
        succ_old = ((is_a & (_rsr_old_hold >= RSR_HOLD) & (~succ_lat)) if RSR_DCUBE > 0.0 else torch.zeros(N, dtype=torch.bool, device=DEVICE))   # user 2026-07-27: PRIVILEGED true-reach success — must HOLD rsr_old (_dcube<RSR_DCUBE) for RSR_HOLD consecutive frames (was hold=1 -> policy swept THROUGH the reach zone = "passing through"; dwell forces settle over cube). TERMINATION/eval-freeze/sweep-advance trigger (NO extra bonus unless BONUS_BY_OLD).
        _gone_pen = W_GONE * _cube_gone.float()   # user 2026-07-10: −W_GONE when the cube flew away (knocked by flailing) — teaches "don't knock the cube"; applied to all phases
        d_s = (to_app | tmo).float(); r_sf = torch.nan_to_num(r_s + SEARCH_BONUS*to_app.float() - _gone_pen, nan=0.0, posinf=0.0, neginf=0.0)   # backstop: never store a NaN/Inf reward (buffer-poison guard)
        if bool(is_s.any()) and not TP_EVAL and not (_curric_on and _sweeping[0]):
            srch["buf"].store(fd[is_s],wd[is_s],ori[is_s],s[is_s],a_s[is_s],r_sf[is_s].unsqueeze(-1),
                              nfd[is_s],nwd[is_s],nori[is_s],ns[is_s],d_s[is_s].unsqueeze(-1))
        d_a = (to_srch | succ | succ_old | to_align | tmo).float(); r_af = torch.nan_to_num(r_a + APP_BONUS*(BONUS_RSR*(succ | to_align).float() + BONUS_RSR_OLD*succ_old.float()) - _gone_pen - W_RESRCH*to_srch.float(), nan=0.0, posinf=0.0, neginf=0.0)   # user 2026-07-28: WEIGHTED bonus = APP_BONUS·(BONUS_RSR·deployable-succ + BONUS_RSR_OLD·true-reach)   # lost penalty removed (user 2026-07-01); −_gone_pen if cube knocked away (2026-07-10); −W_RESRCH one-time at approach->search re-handoff (anti-crutch, user 2026-07-24); nan_to_num backstop (buffer-poison guard 2026-07-21)
        if bool(is_a.any()) and not TP_EVAL and not (_curric_on and _sweeping[0]):
            appr["buf"].store(fd[is_a],wd[is_a],ori[is_a],s[is_a],a_a[is_a],r_af[is_a].unsqueeze(-1),
                              nfd[is_a],nwd[is_a],nori[is_a],ns[is_a],d_a[is_a].unsqueeze(-1))
        # ---- ALIGN phase: cube -> jaw-mouth (_CTGT) + size + yaw -> in-mouth ready ----
        succ_l = torch.zeros(N, dtype=torch.bool, device=DEVICE); to_srch_l = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        if THREE_PHASE:
            d_m = torch.norm(nwd[:, 0:2] - _CTGT, dim=-1)
            mouth_r = (1.0 - torch.tanh(CEN_K*d_m)) + (1.0 - torch.tanh(CEN_K_FINE*d_m))   # toward jaw-mouth (max 2.0)
            _yawv = torch.atan2(nori[:, 0], nori[:, 1]); yaw_abs = torch.sin(YAW_SYM * _yawv).abs()   # user 2026-07-08: |sin(YAW_SYM·yaw)| — SQUARE cube(YAW_SYM=2) folds yaw to mod-90° (0/90/180/270 all count as aligned); was |sin yaw| (mod-180° only, penalized equivalent 90° grasp)
            sz_err = (nwd[:, 2] - _FRAC_TGT).abs()
            dm_min = torch.where(is_l, torch.minimum(dm_min, d_m), torch.full_like(dm_min, 9.0))       # best mouth-dist while in ALIGN (auto-reset when not aligning)
            yaw_min = torch.where(is_l, torch.minimum(yaw_min, yaw_abs), torch.full_like(yaw_min, 9.0)) # best yaw alignment
            _reg_l = W_REG * ((d_m - dm_min).clamp(min=0.0) + (yaw_abs - yaw_min).clamp(min=0.0)) * nvis  # ALIGN regression: drifted from mouth / twisted -> penalty (user 2026-06-30)
            d_m_prev = torch.norm(wd[:, 0:2] - _CTGT, dim=-1)                              # previous mouth dist (for the direction delta)
            _mouth_r = (1.0 - torch.tanh(CEN_K*d_m)) + (1.0 - torch.tanh(CEN_K_FINE*d_m))   # mouth-closeness LEVEL (persistent, ∈[0,2], →2 at jaw-mouth)
            _yaw_r = (1.0 - torch.tanh(CEN_K*yaw_abs)) + (1.0 - torch.tanh(CEN_K_FINE*yaw_abs))   # yaw-ALIGN LEVEL (user 2026-07-02): ∈[0,2], →2 when yaw aligned (+reward, not just penalty)
            r_l = (W_DEPTH*(torch.exp(EXP_KZ*_d) - 1.0)*nvis*_csign                         # descend LEVEL × exp centering-sign (user 2026-07-04)
                 + W_ML*_mouth_r*nvis                                                        # mouth-centering LEVEL (like approach cen_r)
                 + W_YW*_yaw_r*nvis                                                          # yaw-align LEVEL (user 2026-07-02): aligned = + (was − penalty only)
                 + W_MR*(d_m_prev - d_m)*nvis                                               # mouth delta (W_MR=0 by default): closer +, AWAY −
                 - _tabpen - _reg_l)   # user 2026-07-02: descend + mouth-center(+) + yaw-align(+) − table − UP-pen − REGRESSION(_reg_l) − HOVER-pen(above 7.5cm must keep descending)
            lost_l = ((seen > 0.5) & (nvis < 0.5)) | (ncen > LOSE_TOL)           # ALIGN lost = seen 1->0 edge OR off-center
            lhold_l = torch.where(is_l & lost_l, lhold_l + 1, torch.where(is_l, torch.zeros_like(lhold_l), lhold_l))
            to_srch_l = is_l & (lhold_l >= LOSE_HOLD)
            _inmouth = is_l & (d_m < MOUTH_OK) & (yaw_abs < YAW_OK) & (gz < ALIGN_GZ)   # ALIGN in-mouth THIS frame
            shold_l = torch.where(_inmouth, shold_l + 1, torch.where(is_l, torch.zeros_like(shold_l), shold_l))   # user 2026-07-03: hold in-mouth SUCC_HOLD frames (was instant)
            succ_l = is_l & (shold_l >= SUCC_HOLD) & (~succ_lat)   # ALIGN SUCCESS = in-mouth HELD SUCC_HOLD frames (stable, matches approach)
            if bool(is_l.any()): _cum["l"] += float(r_l[is_l].mean().item())
            d_l = (to_srch_l | succ_l | tmo).float(); r_lf = torch.nan_to_num(r_l + APP_BONUS*succ_l.float() - _gone_pen, nan=0.0, posinf=0.0, neginf=0.0)   # ALIGN done; lost penalty removed (2026-07-01); −_gone_pen (2026-07-10); nan_to_num backstop (2026-07-21)
            if bool(is_l.any()) and not TP_EVAL:
                algn["buf"].store(fd[is_l],wd[is_l],ori[is_l],s[is_l],a_l[is_l],r_lf[is_l].unsqueeze(-1),
                                  nfd[is_l],nwd[is_l],nori[is_l],ns[is_l],d_l[is_l].unsqueeze(-1))
        ph = torch.where(to_app, torch.ones_like(ph), ph)        # SEARCH->APPROACH
        if bool(to_app.any()): arm_term._target_qpos[to_app] = r.data.joint_pos[:, joint_ids][to_app]   # user 2026-07-02 BUGFIX: re-sync PD target at SEARCH->APPROACH handoff so APPROACH IK starts from ACTUAL pose, not SEARCH's stale accumulated _target_qpos (was making EE lurch UP at handoff)
        if bool(to_app.any()): _app_ori[to_app] = r.data.body_quat_w[to_app, ee_body_idx, :]   # user 2026-07-03: capture APPROACH-entry orientation (APP_ORI=freeze)
        if APP_GUAR_STEPS > 0 and bool(to_app.any()): _app_start_es[to_app] = env.episode_length_buf[to_app].float()   # user 2026-07-21: stamp handoff episode-step -> approach guaranteed APP_GUAR_S from here
        ph = torch.where(to_srch, torch.zeros_like(ph), ph)      # APPROACH->SEARCH
        if bool(to_srch.any()): _srch_start_es[to_srch] = env.episode_length_buf[to_srch].float()   # user 2026-07-24: re-entered SEARCH -> stamp entry step so SEARCH_MAX_S restarts fresh (15s again)
        if bool(to_srch.any()):
            _did_resrch[to_srch] = True                                                              # user 2026-07-24: latch approach->search re-handoff (UI rate)
            _resrch_mz.extend(_mouth_w[to_srch, 2].tolist())                                          # user 2026-07-24: record jaw-MOUTH z at the re-handoff moment
            del _resrch_mz[:-400]
            if len(_resrch_mz) >= 20:                                                                 # print a height histogram once we have a sample
                import numpy as _np; _rz = _np.array(_resrch_mz)
                _b = [(_rz<0.06).sum(),((_rz>=0.06)&(_rz<0.10)).sum(),((_rz>=0.10)&(_rz<0.15)).sum(),((_rz>=0.15)&(_rz<0.25)).sum(),(_rz>=0.25).sum()]
                print(f"[RESRCH-H] n={len(_rz)} mouthz mean={_rz.mean():.3f} med={_np.median(_rz):.3f} | <0.06:{_b[0]} .06-.10:{_b[1]} .10-.15:{_b[2]} .15-.25:{_b[3]} >=.25:{_b[4]}  (cube_z~0.06; APP_MZ=0.10)", flush=True)
                _resrch_mz.clear()
        if THREE_PHASE:
            ph = torch.where(to_align, torch.full_like(ph, 2), ph)   # APPROACH->ALIGN
            ph = torch.where(to_srch_l, torch.zeros_like(ph), ph)     # user 2026-07-07: ALIGN lost -> SEARCH (was ->APPROACH). Re-acquire from scratch (search->approach->align) instead of bouncing align<->approach.
        seen = torch.where(to_app | to_srch | to_align | to_srch_l, torch.zeros_like(seen), seen)   # re-acquire SEEN in the new phase after any transition
        _reset_tr = to_app | to_srch | to_align | to_srch_l                                          # user 2026-07-09: reset cube_rel reliable-seen latches on any phase transition (fresh acquire in the new phase)
        if _curric_on: _srch_reached |= (ph >= 1)                                                     # user 2026-07-16: latch SEARCH success = reached APPROACH(≥1) this episode (for sweep search heatmap)
        if _EVAL_CASES is not None: _eval_srch_reached |= (ph >= 1)                                    # user 2026-07-22: same latch for the eval-set SEARCH heatmap
        _crw_seen = torch.where(_reset_tr, torch.zeros_like(_crw_seen), _crw_seen)
        _crf_seen = torch.where(_reset_tr, torch.zeros_like(_crf_seen), _crf_seen)
        if RULE_GL and THREE_PHASE:                                              # RULE grasp+lift FSM: ALIGN success -> GRASP(hold) -> LIFT(raise)
            _czw = cube_z_w()
            _to_grasp = succ_l                                                   # in-mouth this step -> begin rule grasp (RSR still credited via succ_l/succ_lat below)
            gl_cz0  = torch.where(_to_grasp, _czw, gl_cz0)                        # cube z at grasp start (lift-success baseline)
            gl_ztgt = torch.where(_to_grasp, r.data.body_pos_w[:, ee_body_idx, 2] + RGL_LIFTZ, gl_ztgt)  # lift target = grasp EE z + LIFTZ
            gl_hold = torch.where(_to_grasp, torch.zeros_like(gl_hold), gl_hold)
            ph = torch.where(_to_grasp, torch.full_like(ph, 3), ph)              # ALIGN->GRASP
            _in_g = (ph == 3)
            gl_hold = torch.where(_in_g, gl_hold + 1, gl_hold)
            ph = torch.where(_in_g & (gl_hold >= RGL_HOLD), torch.full_like(ph, 4), ph)   # GRASP->LIFT after hold
            lift_lat = lift_lat | ((ph == 4) & ((_czw - gl_cz0) >= RGL_SUCC))    # LIFT success = cube rose >= RGL_SUCC from grasp start
        _succ_any = succ | succ_old | succ_l   # user 2026-07-27: latch/end on deployable OR privileged true-reach
        succ_lat = succ_lat | _succ_any; _tp_succ_total += int(_succ_any.sum().item())
        if (not TP_EVAL) and not (_curric_on and _sweeping[0]) and bool(_succ_any.any()):   # user 2026-07-16: also exclude sweep -> no early termination on appr_ready (terminate only on rsr_hold, like the bench)                             # user 2026-07-03: SUCCESS (appr_ready/in-mouth held SUCC_HOLD) -> END episode (force time-out -> auto-reset next step). stops arm drifting back to SEARCH after success.
            try: env.episode_length_buf[_succ_any] = int(env.max_episode_length)
            except Exception as _te:
                if g % 5000 == 0: print(f'[SUCC-TERM] episode_length_buf set failed: {_te}', flush=True)
        if (_curric_on and _sweeping[0]) and (CURRIC_STAGED and not _stage_rsr[0]) and bool(_srch_reached.any()):   # user 2026-07-24: SWEEP in SEARCH stage (not yet all cells at search 100%) — end the episode the MOMENT SEARCH hands off (approach not evaluated yet — "until 100%, terminate as soon as search succeeds"). With SEARCH_MAX_S=15 this caps every sweep episode at <=15s and scores search-success only. Once _stage_rsr flips True, this branch is skipped -> sweep runs full approach->RSR via the block below.
            _sw_done = _srch_reached & (_cur_cell >= 0)
            if bool(_sw_done.any()):
                try: env.episode_length_buf[_sw_done] = int(env.max_episode_length)
                except Exception: pass
        elif (TP_EVAL or (_curric_on and _sweeping[0])) and RSR_DCUBE > 0.0:   # user 2026-07-16->07-24: BENCH, or SWEEP in RSR stage (all cells reached search 100% -> now include approach) — the moment RSR success (dcube<RSR_DCUBE held RSR_HOLD frames) is reached, END the episode -> count the hit + advance immediately
            _hold_sel = _rsr_old_hold if EVAL_SUCC_BY_OLD else _rsr_hold   # user 2026-07-27: EVAL_SUCC_BY_OLD -> gate freeze/advance on PRIVILEGED true-reach
            _rsr_done = _hold_sel >= RSR_HOLD
            if bool(_rsr_done.any()):
                try:
                    if EVAL_FREEZE_SUCC < 0.0 and TP_EVAL:
                        pass   # user 2026-07-27: EVAL_FREEZE_SUCC<0 = INDEFINITE freeze — succ_lat (set above) holds the arm at the success frame; never advance (manual NEW ENV to continue)
                    elif EVAL_FREEZE_SUCC > 0.0 and TP_EVAL:
                        _fs = max(1, int(round(EVAL_FREEZE_SUCC * _CTRL_HZ)))   # freeze steps = seconds × 10Hz
                        _just = _rsr_done & (_hold_sel <= float(RSR_HOLD))       # FIRST success frame only
                        if bool(_just.any()):
                            env.episode_length_buf[_just] = int(env.max_episode_length) - _fs   # start hold -> auto-increments to max after _fs steps -> reset. arm frozen (act_env=0 via succ_lat) during hold.
                    else:
                        env.episode_length_buf[_rsr_done] = int(env.max_episode_length)   # end episode -> advance to next case
                except Exception: pass
        if APP_FAIL_Z > 0.0 and (not TP_EVAL):                                   # user 2026-07-05: APPROACH dove below APP_FAIL_Z with NO grasp -> FAILED -> end episode NOW (succ_lat stays False -> recorded as fail at reset). Fast fail = success-rate unbiased under NO_RESRCH + stops table-diving.
            _app_fail = is_a & (gz < APP_FAIL_Z) & (~appr_ready) & (~succ_lat)
            if bool(_app_fail.any()):
                try: env.episode_length_buf[_app_fail] = int(env.max_episode_length)
                except Exception: pass
        if SEARCH_MAX_STEPS > 0:                                                 # user 2026-07-21: SEARCH capped — no handoff within SEARCH_MAX_S -> FAIL, end episode NOW (succ_lat False -> recorded fail), auto-reset next step -> new episode
            _srch_tmo = (ph == 0) & ((env.episode_length_buf.float() - _srch_start_es) >= float(SEARCH_MAX_STEPS))   # user 2026-07-24: PHASE-relative (fresh SEARCH_MAX_S per search entry, incl. approach->search re-entry)
            if bool(_srch_tmo.any()):
                try: env.episode_length_buf[_srch_tmo] = int(env.max_episode_length)
                except Exception: pass
        if APP_GUAR_STEPS > 0:                                                   # user 2026-07-21: guarantee APP_GUAR_S of approach after handoff, then END (success or fail by succ_lat). episode-steps since handoff (_app_start_es) >= budget -> force time-out
            _app_tmo = (ph >= 1) & (_app_start_es >= 0.0) & ((env.episode_length_buf.float() - _app_start_es) >= float(APP_GUAR_STEPS))
            if bool(_app_tmo.any()):
                try: env.episode_length_buf[_app_tmo] = int(env.max_episode_length)
                except Exception: pass
        if (not TP_EVAL) and not (_curric_on and _sweeping[0]) and bool(_cube_gone.any()):   # user 2026-07-16: also exclude sweep (same as bench)                            # user 2026-07-10: cube flew away (physics) -> clear any (false) success latch + force episode-end (recorded as FAIL) so a corrupted env can't spoof appr_ready or pollute the metrics; cube respawns on reset
            succ_lat = succ_lat & (~_cube_gone)
            try: env.episode_length_buf[_cube_gone] = int(env.max_episode_length)
            except Exception: pass
        if bool(to_srch.any()): succ_lat = succ_lat & (~to_srch)   # user 2026-07-03 BUGFIX: back to SEARCH (lost) -> CLEAR success latch, else act_env[succ_lat]=0 freezes the arm and it can't re-search
        if THREE_PHASE and bool(to_srch_l.any()): succ_lat = succ_lat & (~to_srch_l)   # same for ALIGN->APPROACH (lost)
        _succ_s += int(to_app.sum().item())                                      # SEARCH success = handed off to APPROACH
        _succ_a += int((to_align | succ).sum().item())                           # APPROACH success = ->ALIGN (3-phase) or grasp-ready (2-phase)
        _succ_l += int(succ_l.sum().item())                                      # ALIGN success = in-mouth ready
        _ent_a += int((to_app | to_srch_l).sum().item())                          # entries: into APPROACH (from search or align-lost)
        _ent_l += int(to_align.sum().item())                                     # entries: into ALIGN
        _ent_s += int(to_srch.sum().item())                                      # entries: into SEARCH (approach-lost; +episode resets added below)
        if bool(to_app.any()): _srch_ok[to_app] = True     # user 2026-07-23: SEARCH handed off this episode -> latch. Rolling _win_s now recorded once per episode at RESET (below), not here (removes early-success timing bias).
        _ano = int(to_srch.sum().item())
        # user 2026-07-06: APPROACH success rate is now counted ONCE PER EPISODE at episode END (succ_lat, in the rid block below) — NOT mid-episode. The old mid-episode +1 (_aok) counted successes immediately while failures only counted at the 1200-step timeout -> the rolling window filled with early successes -> inflated (e.g. 100% n=1). Per-episode counting removes that bias (RSR already worked this way).
        if THREE_PHASE:                                                          # user 2026-07-02: ALIGN rolling — in-mouth (1) vs lost->APPROACH (0)
            _hok = int(to_align.sum().item())                                    # user 2026-07-03: ALIGN HANDOFF (met align criterion) — separate section
            if _hok: _win_h.extend([1.0]*_hok)
            if _ano: _win_h.extend([0.0]*_ano)                                   # lost before handoff
            _lok = int(succ_l.sum().item()); _lno = int(to_srch_l.sum().item())
            if _lok: _win_l.extend([1.0]*_lok)
            if _lno: _win_l.extend([0.0]*_lno)
        if TP_EVAL:                                                               # env0 FSM signals + exact phase-transition log
            _TPUI["ncen0"]=float(ncen[0]); _TPUI["chold0"]=int(chold[0].item()); _TPUI["lhold0"]=int(lhold[0].item())
            _TPUI["frac0"]=float(nwd[0,2]); _TPUI["vis0"]=float(nvis[0]); _TPUI["phase0"]=int(ph[0].item())
            _np0=int(ph[0].item())
            if _np0!=_prev_ph0 or bool(succ[0]):
                if bool(succ[0]): _ev="✅ APPROACH→SUCCESS (frac %.2f>%.2f, centered)"%(float(nwd[0,2]),FRAC_GRASP)
                elif _np0==1: _ev="🔵→🟢 SEARCH→APPROACH (centered held %d steps)"%SEARCH_HOLD
                else: _ev="🟢→🔵 APPROACH→SEARCH (lost: vis %.0f / ncen %.2f)"%(float(nvis[0]),float(ncen[0]))
                _TPUI["plog"]=(_TPUI.get("plog",[])+["step %d: %s"%(g,_ev)])[-8:]
                print("[TP-EVAL] %s @step %d"%(_ev,g), flush=True)
            _prev_ph0=_np0
        rid = torch.nonzero(tmo, as_tuple=False).squeeze(-1)
        if rid.numel() > 0:
            _rsr_hit = _rsr_holdmax[rid] >= RSR_HOLD   # user 2026-07-20: RSR = UNIFIED success cond (center+frac+height) HELD >=RSR_HOLD consecutive frames == what appr_ready/succ uses
            _rsr_old_hit = _rsr_old_holdmax[rid] >= RSR_HOLD   # PRIVILEGED reference (true dcube<RSR_DCUBE held) -> UI "rsr_old" only
            if _curric_on and _sweeping[0]:                                       # curriculum self-eval: attribute this episode's RSR-hit to the grid cell it ran
                _cfin = _cur_cell[rid]; _cfm = _cfin >= 0
                if bool(_cfm.any()):
                    _cfi = _cfin[_cfm]; _chh = (_rsr_old_hit if EVAL_SUCC_BY_OLD else _rsr_hit).float()[_cfm]   # user 2026-07-27: EVAL_SUCC_BY_OLD -> curriculum weights by PRIVILEGED true-reach (deployable RSR miscalibrated ~0%); UI RSR metric (_tp_recent) stays deployable
                    _cur_seen.index_add_(0, _cfi, torch.ones_like(_chh)); _cur_hit.index_add_(0, _cfi, _chh)
                    _cur_srch.index_add_(0, _cfi, _srch_reached[rid].float()[_cfm])   # user 2026-07-16: per-cell SEARCH-reached-APPROACH (sweep search heatmap)
            _cd_rid = (env.scene[args.target_cube].data.root_pos_w[rid] - env.scene.env_origins[rid])[:, 1] + 1.15   # user 2026-07-13: per-episode spawn fwd-dist (cube ~static) -> RSR-by-distance
            _rsrd_dist.extend(_cd_rid.tolist()); _rsrd_hit.extend(_rsr_hit.float().tolist())
            if len(_rsrd_dist) > 3000: del _rsrd_dist[:len(_rsrd_dist)-3000]; del _rsrd_hit[:len(_rsrd_hit)-3000]
            if _EVAL_CASES is not None:                                            # attribute this episode's RSR-hit to its assigned fixed case
                _fin = _env_case[rid]; _fm = _fin >= 0
                if bool(_fm.any()):
                    _fi = _fin[_fm]; _hh = _rsr_hit.float()[_fm]
                    _eval_seen.index_add_(0, _fi, torch.ones_like(_hh)); _eval_hits.index_add_(0, _fi, _hh)
                    _eval_hits_old.index_add_(0, _fi, _rsr_old_hit.float()[_fm])   # user 2026-07-22: privileged rsr_old per case
                    _eval_srch.index_add_(0, _fi, _eval_srch_reached[rid].float()[_fm])   # user 2026-07-22: SEARCH-reached per case
                    _evh.log_episodes(_EVAL_CASES, _fi, rid[_fm], _eval_srch_reached[rid][_fm], _rsr_hit[_fm], _rsr_old_hit[_fm], lift_lat[rid][_fm], g)   # 2026-11-01: opt-in per-episode CSV
                    _e0 = (rid[_fm] == 0).nonzero()
                    if _e0.numel() > 0:                                   # 2026-11-02: env 0 finished -> close its video (opt-in)
                        _j = int(_e0[0, 0]); _evh.video_end(int(_fi[_j]), bool(_eval_srch_reached[rid][_fm][_j]), bool(_rsr_hit[_fm][_j]), bool(_rsr_old_hit[_fm][_j]), bool(lift_lat[rid][_fm][_j]))
            _tp_recent.extend(_rsr_hit.float().tolist())
            _rsr_old_recent.extend(_rsr_old_hit.float().tolist())                   # privileged-RSR-hit rolling (UI "rsr_old")
            if len(_rsr_old_recent) > _METRIC_WIN: del _rsr_old_recent[:len(_rsr_old_recent) - _METRIC_WIN]
            _lift_recent.extend(lift_lat[rid].float().tolist())                    # RULE grasp+lift outcome per finished episode -> lift-success rate (LSR)
            if len(_lift_recent) > 500: del _lift_recent[:len(_lift_recent) - 500]
            if len(_tp_recent) > _METRIC_WIN: del _tp_recent[:len(_tp_recent) - _METRIC_WIN]
            _gzmin_recent.extend(gz_min[rid].tolist())   # capture ep-min gz BEFORE reset
            if len(_gzmin_recent) > 500: del _gzmin_recent[:len(_gzmin_recent) - 500]
            _ent_s += int(rid.numel())                                            # episode reset -> re-enters SEARCH
            _phr = ph[rid]                                                         # user 2026-07-02: phase at episode end (before reset) -> rolling fail
            _sf = int((_phr == 0).sum().item()); _af = int((_phr == 1).sum().item())
            _win_s.extend(_srch_ok[rid].float().tolist())                          # user 2026-07-23: ONE sample per finished episode — 1 if SEARCH handed off within SEARCH_MAX_S this episode, else 0. Unbiased search-success rate (both outcomes recorded at reset).
            _ho_rid = _srch_ok[rid]                                                 # user 2026-07-24: episodes that reached APPROACH -> record whether approach->search re-handoff fired (look-at quality)
            if bool(_ho_rid.any()): _win_resrch.extend(_did_resrch[rid][_ho_rid].float().tolist())
            _a_ep = (_phr == 1)                                                    # user 2026-07-06: episodes that were in APPROACH at end (both grasped-and-latched and failed). Record ONE outcome per episode = succ_lat (1 grasped / 0 not) -> unbiased per-episode APPROACH success rate.
            if bool(_a_ep.any()): _win_a.extend(succ_lat[rid][_a_ep].float().tolist())
            if THREE_PHASE: _win_h.extend([0.0]*_af)   # user 2026-07-03: ended mid-approach -> align handoff failed
            if THREE_PHASE: _win_l.extend([0.0]*int((_phr == 2).sum().item()))       # episode ended mid-ALIGN -> align failed
            ph[rid] = 0; chold[rid] = 0; lhold[rid] = 0; succ_lat[rid] = False; ahold[rid] = 0; lhold_l[rid] = 0; shold_l[rid] = 0; gz_min[rid] = gz[rid]; _mz_min[rid] = _mouth_w[rid, 2]; ncen_min[rid] = 1.0; frac_max[rid] = 0.0; seen[rid] = 0.0; dcube_min[rid] = 9.0; _dwell[rid] = 0.0; _rsr_hold[rid] = 0.0; _rsr_holdmax[rid] = 0.0; _rsr_old_hold[rid] = 0.0; _rsr_old_holdmax[rid] = 0.0; _app_start_es[rid] = -1.0; _srch_ok[rid] = False; _srch_start_es[rid] = 0.0; _did_resrch[rid] = False
            gl_hold[rid] = 0; lift_lat[rid] = False                                # RULE grasp+lift state fresh per episode
            if _curric_on: _srch_reached[rid] = False                              # user 2026-07-16: reset SEARCH-success latch for the new episode (sweep search heatmap)
            if _EVAL_CASES is not None: _eval_srch_reached[rid] = False            # user 2026-07-22: reset eval SEARCH latch
            if FRONT_DROP_P > 0.0 and _front_drop is not None: _front_drop[rid] = (torch.rand(rid.numel(), device=DEVICE) < FRONT_DROP_P)   # resample front-dropout per episode
            if _EVAL_CASES is not None:                                            # FIXED eval spawn: override just-reset cube x/y/yaw + wrist_flex from the next cases (round-robin, deterministic)
                _no = rid.numel(); _M = _EVAL_CASES.shape[0]
                if _EVAL_SINGLE:                                                    # user 2026-07-31: SINGLE-TRIAL — monotonic cursor, no wrap; cells >= M parked (-1 -> uncounted via L2229 _fm) so each cell scored EXACTLY once (0/1 binary) + ~1-pass rollout
                    _ci = _eval_cursor[0] + torch.arange(_no, device=DEVICE)
                    if _eval_cursor[0] < _M and _eval_cursor[0] + _no >= _M: _eval_sweeps[0] = 1
                    _eval_cursor[0] = int(_eval_cursor[0] + _no)
                    _ci = torch.where(_ci < _M, _ci, torch.full_like(_ci, -1))
                else:
                    _ci = (_eval_cursor[0] + torch.arange(_no, device=DEVICE)) % _M
                    if _eval_cursor[0] + _no >= _M: _eval_sweeps[0] += 1
                    _eval_cursor[0] = int((_eval_cursor[0] + _no) % _M)
                _env_case[rid] = _ci
                _cs = _EVAL_CASES[_ci.clamp(min=0)]                                 # (no,4)=[x,y,yaw,wrist]; parked (-1) idx->0 spawn but uncounted
                _cpw = env.scene[args.target_cube].data.root_pos_w[rid].clone()     # env-LOCAL x,y -> world (env origin + local); z (table height) kept
                _cpw[:, 0] = env.scene.env_origins[rid, 0] + _cs[:, 0]
                _cpw[:, 1] = env.scene.env_origins[rid, 1] + _cs[:, 1]
                _hy = _cs[:, 2] * 0.5                                               # yaw about world-Z -> quat (w,x,y,z)
                _q = torch.zeros(_no, 4, device=DEVICE); _q[:, 0] = torch.cos(_hy); _q[:, 3] = torch.sin(_hy)
                env.scene[args.target_cube].write_root_pose_to_sim(torch.cat([_cpw, _q], dim=-1), env_ids=rid)
                _jpr = r.data.joint_pos[rid].clone(); _jpr[:, int(joint_ids[3])] = _cs[:, 3]   # fixed wrist_flex start
                r.write_joint_state_to_sim(_jpr, torch.zeros_like(_jpr), env_ids=rid)
                arm_term._target_qpos[rid, 3] = _cs[:, 3]
            if _curric_on:                                                          # user 2026-07-14: ADAPTIVE-CURRICULUM spawn — SWEEP: sequential grid cells (deterministic); TRAINING: sample cells ∝ fail-weight (concentrate on weak regions)
                _no = rid.numel()
                if _sweeping[0]:
                    _cc = _swcur[0] + torch.arange(_no, device=DEVICE)
                    _valid = _cc < _NKEEP; _swcur[0] = _swcur[0] + _no; _ccl = _kept_idx[_cc.clamp(max=_NKEEP-1)]   # user 2026-07-30: cycle KEPT cells only (excluded 6 skipped -> never evaluated/lit)
                    _cur_cell[rid] = torch.where(_valid, _ccl, torch.full_like(_ccl, -1)); _jit = 0.0   # user 2026-07-16: SWEEP = DETERMINISTIC (cell CENTER, no jitter) -> FAIR/reproducible eval (was _jit=1 random -> not fair)
                    _ix = _ccl % CURRIC_NX; _iy = torch.div(_ccl, CURRIC_NX, rounding_mode='floor')
                    _cyaw = torch.deg2rad(-40.0 + 80.0*(((_ix*3 + _iy*5) % 16).float()/15.0))          # deterministic per-cell yaw (SAME formula as gen_eval_set/bench) -> fair, reproducible, matches bench
                else:
                    _ccl = torch.multinomial(_cw, _no, replacement=True); _jit = 1.0                    # TRAINING: spawn ∝ fail-weight, RANDOM jitter+yaw (coverage/robustness)
                    _cyaw = (torch.rand(_no, device=DEVICE)*2-1)*3.141592653589793
                    _env_cell[rid] = _ccl                                                              # user 2026-07-19: record each env's TRAINING spawn cell for region-selective privilege
                _xl = _cell_x[_ccl % CURRIC_NX] + (_jit*(torch.rand(_no, device=DEVICE)-0.5))*((_SPX_HI-_SPX_LO)/CURRIC_NX)   # cell center (+jitter only in training)
                _yl = _cell_y[torch.div(_ccl, CURRIC_NX, rounding_mode='floor')] + (_jit*(torch.rand(_no, device=DEVICE)-0.5))*((_SPY_HI-_SPY_LO)/CURRIC_NY)
                _cpwc = env.scene[args.target_cube].data.root_pos_w[rid].clone()
                _cpwc[:, 0] = env.scene.env_origins[rid, 0] + _xl; _cpwc[:, 1] = env.scene.env_origins[rid, 1] + _yl
                _cq = torch.zeros(_no, 4, device=DEVICE); _cq[:, 0] = torch.cos(_cyaw*0.5); _cq[:, 3] = torch.sin(_cyaw*0.5)   # CLEAN UPRIGHT (Z-yaw only)
                env.scene[args.target_cube].write_root_pose_to_sim(torch.cat([_cpwc, _cq], dim=-1), env_ids=rid)
                if _sweeping[0]:                                                     # user 2026-07-16 FIX: SWEEP starts wrist at START_WRIST (like the eval-set) so the wrist-cam points at the workspace -> cube DETECTED -> search actuates the wrist. WAS unset (curric block never set wrist) -> stuck at 0 -> vis~0 -> policy frozen -> sweep RSR fake-low. Only the START pose is pinned; the SEARCH_IK servo moves it freely after (START_WRIST DR skipped in sweep, so deterministic 2.3 start).
                    _jpw = r.data.joint_pos[rid].clone(); _jpw[:, int(joint_ids[3])] = _START_WRIST
                    r.write_joint_state_to_sim(_jpw, torch.zeros_like(_jpw), env_ids=rid)
                    arm_term._target_qpos[rid, 3] = _START_WRIST
                if _sweeping[0] and _swcur[0] >= _NKEEP and not bool((_cur_cell >= 0).any()):   # all KEPT cells assigned AND all in-flight scored -> reweight + end sweep (user 2026-07-30: _NKEEP not _NCELL, excluded skipped)
                    _sn = _cur_seen.clamp(min=1.0)
                    if CURRIC_STAGED:                                                                # user 2026-07-24: GLOBAL staged — SEARCH stage (sweep search-only) weights by search-fail; once EVERY seen cell reaches SEARCH>=CURRIC_SW_TH the NEXT sweep runs full approach->RSR and weights by RSR-fail
                        _seenm = _cur_seen > 0.5
                        _all_master = bool((_cur_srch.sum()/_cur_seen.sum().clamp(min=1.0) >= CURRIC_STAGE_TH).item()) if bool(_seenm.any()) else False   # user 2026-07-26: flip on OVERALL sweep SEARCH% >= CURRIC_STAGE_TH (e.g. 0.90), not every-cell>=CURRIC_SW_TH
                        _fm = 1.0 - (_cur_hit if _stage_rsr[0] else _cur_srch)/_sn                    # weight by what THIS sweep actually measured (RSR only once we're in the RSR stage & the sweep ran approach)
                        _prev_stage = _stage_rsr[0]; _stage_rsr[0] = _all_master                       # flip stage for the NEXT sweep
                        print(f"[CURRIC-STAGE] this sweep weighted by {'RSR' if _prev_stage else 'SEARCH'}; next sweep -> {'RSR (all cells reached search>=%.0f%%)'%(CURRIC_SW_TH*100) if _all_master else 'SEARCH (still mastering search)'}", flush=True)
                    else:
                        _fm = 1.0 - (_cur_srch if CURRIC_BY_SEARCH else _cur_hit)/_sn
                    _fail = torch.where(_cur_seen > 0.5, _fm, torch.zeros_like(_sn))   # user 2026-07-24: CURRIC_BY_SEARCH -> fail = SEARCH-reached fail (search-first) instead of RSR fail; CURRIC_STAGED -> global search-first-then-RSR
                    if PRIV_KEEP_FAIL:                                                                # user 2026-07-19: mark failing cells (sweep RSR < PRIV_KEEP_RSR) to KEEP privilege there post-anneal
                        _cell_keeppriv[:] = (_cur_seen > 0.5) & ((_cur_hit/_sn) < PRIV_KEEP_RSR)
                        print(f"[PRIV-KEEP] {int(_cell_keeppriv.sum())}/{_NCELL} cells kept privileged (sweep RSR<{PRIV_KEEP_RSR:.0%})", flush=True)
                    _fg = _fail.view(1, 1, CURRIC_NY, CURRIC_NX)
                    if CURRIC_BLUR > 0: _fg = _Fnn.avg_pool2d(_fg, kernel_size=2*CURRIC_BLUR+1, stride=1, padding=CURRIC_BLUR)   # box-blur fail map -> region weights
                    _wf = (1.0 + (CURRIC_BOOST - 1.0)*_fg.reshape(-1)).clamp(min=0.05)   # boosted per-cell fail weight
                    if CURRIC_FLOOR > 0.0:                                                # HARD FLOOR: FLOOR of budget uniform (guarantees passing cells) + (1-FLOOR) fail-weighted
                        _pf = _wf / _wf.sum().clamp(min=1e-8)
                        _cw = (CURRIC_FLOOR / _NCELL) + (1.0 - CURRIC_FLOOR) * _pf        # sums to 1; every cell >= CURRIC_FLOOR/_NCELL
                    else:
                        _cw = _wf                                                        # legacy soft floor
                    _cw = _cw.masked_fill(_curric_excl, 0.0)                              # user 2026-07-30: excluded cells -> 0 spawn weight (never trained)
                    _keepc = ~_curric_excl                                                # user 2026-07-30: exclude unreachable cells from the sweep RSR/SEARCH aggregate
                    _ov = float((_cur_hit[_keepc].sum()/_cur_seen[_keepc].sum().clamp(min=1.0)).item())*100.0
                    _srch_ov = float((_cur_srch[_keepc].sum()/_cur_seen[_keepc].sum().clamp(min=1.0)).item())*100.0   # user 2026-07-24: SEARCH success overall — the meaningful metric in the SEARCH-stage (search-only) sweep, where RSR is trivially 0 (approach not run). Show both so the log isn't misread as total failure.
                    print(f"[CURRIC] === sweep DONE @ step {g:,}: SEARCH={_srch_ov:.1f}% RSR={_ov:.1f}% ({int((_cur_seen>0).sum())}/{_NCELL} cells) -> reweighted (w {float(_cw.min()):.2f}~{float(_cw.max()):.2f}) ===", flush=True)
                    if SWEEP_ALPHA > 0.0:   # user 2026-07-28: per-sweep small alpha BUMP (up only) on APPROACH -> re-explore the newly re-weighted failing cells; anneals back to floor over the next CURRIC_EVERY. Small (~0.2) so it re-learns WITHOUT the policy-destroying full 0.5 RESUME_ALPHA reset (that caused the 8951 collapse).
                        _tla = math.log(SWEEP_ALPHA)
                        if float(appr["la"]) < _tla:
                            appr["la"].data.copy_(torch.tensor([_tla], device=DEVICE, dtype=appr["la"].dtype))
                            appr["alpha"] = appr["la"].exp().clamp(min=args.alpha_floor).detach()
                            print(f"[SWEEP-ALPHA] approach α bumped -> {float(appr['alpha']):.3f} (re-explore failing cells; anneals back)", flush=True)
                    if CURRIC_WRITE:   # user 2026-07-23: PRODUCER (.4 offloaded worker) — write the fail-weight _cw so the .2 consumer can pick it up
                        try:
                            import numpy as _npW, os as _osW
                            _tmpW = CURRIC_WRITE + ".tmp.npy"; _npW.save(_tmpW, _cw.detach().cpu().numpy()); _osW.replace(_tmpW, CURRIC_WRITE)   # atomic write (np.save to a .npy tmp, then rename)
                            print(f"[CURRIC-EXT] producer wrote _cw -> {CURRIC_WRITE} (overall RSR {_ov:.1f}%)", flush=True)
                        except Exception as _eW:
                            print(f"[CURRIC-EXT] producer write FAILED: {_eW}", flush=True)
                        if CURRIC_ONESHOT:
                            print("[CURRIC-EXT] one-shot: sweep written -> exit(0)", flush=True); sys.stdout.flush(); os._exit(0)
                    _TPUI["curric_rsr"] = _ov; _TPUI["curric_step"] = int(g); _sweeping[0] = False; _SWEEPING[0] = False; _next_sw[0] = g + CURRIC_EVERY
                    _ch = _TPUI.setdefault("curric_hist", []); _ch.append((int(g), round(_ov, 1)))   # user 2026-07-15: in-training eval history — (step, RSR%) per curriculum self-eval sweep, shown as sparkline in the training UI
                    if len(_ch) > 60: del _ch[:len(_ch)-60]
                    _sn2 = _cur_seen.reshape(CURRIC_NY, CURRIC_NX).cpu().tolist()   # user 2026-07-15/16: per-cell curric-sweep heatmaps (RSR + SEARCH), keep previous sweep for before/after
                    def _mk_curric_heat(_hitT, _lbl, _rate):
                        _pp = (_hitT/_cur_seen.clamp(min=1.0)).reshape(CURRIC_NY, CURRIC_NX).cpu().tolist(); _rws = ""
                        for _ry in range(CURRIC_NY-1, -1, -1):                      # far (top) -> near (bottom)
                            _cl = ""
                            for _rx in range(CURRIC_NX):
                                _bg = "#222" if _sn2[_ry][_rx] < 1 else (lambda _v: f"rgb({int(200*(1-_v))},{int(200*_v)},40)")(_pp[_ry][_rx])
                                _cl += f"<td style='width:13px;height:13px;background:{_bg}'></td>"
                            _rws += f"<tr>{_cl}</tr>"
                        return (f"<div style='color:#888;font-size:10px'>{_lbl} @{g//1000}k · {_rate:.0f}%</div>"
                                f"<table style='border-spacing:2px'>{_rws}</table>")
                    _srsr_c = float((_cur_srch.sum()/_cur_seen.sum().clamp(min=1.0)).item())*100.0
                    _TPUI["curric_heat_prev"] = _TPUI.get("curric_heat", "")
                    _TPUI["curric_heat"] = _mk_curric_heat(_cur_hit, "RSR", _ov)
                    _TPUI["curric_heat_srch"] = _mk_curric_heat(_cur_srch, "SEARCH", _srsr_c)   # user 2026-07-16: SEARCH-success heatmap in the sweep UI
                    _chs = _TPUI.setdefault("curric_hist_srch", []); _chs.append((int(g), round(_srsr_c, 1)))   # user 2026-07-18: SEARCH-success history — (step, reach%) per sweep, shown as histogram in the training UI
                    if len(_chs) > 60: del _chs[:len(_chs)-60]
            if _EVAL_CASES is None and not (_curric_on and _sweeping[0]) and _SWDR_HI > _SWDR_LO > 0.0:   # user 2026-07-08: per-episode START_WRIST DR (kept in curric training, skipped during the deterministic sweep)
                _wfr = _SWDR_LO + (_SWDR_HI - _SWDR_LO) * torch.rand(rid.numel(), device=DEVICE)
                _jpr = r.data.joint_pos[rid].clone(); _jpr[:, int(joint_ids[3])] = _wfr   # joint_ids[3] = wrist_flex full-array index
                r.write_joint_state_to_sim(_jpr, torch.zeros_like(_jpr), env_ids=rid)
                arm_term._target_qpos[rid, 3] = _wfr                                # sync PD target so the controller HOLDS the new wflex (else it snaps back to default)
            if _EVAL_CASES is None and not _curric_on and SPAWN_PEAKED:             # user 2026-07-07: PEAKED spawn distance (OFF when curriculum owns spawn) — dense band w.p. SPAWN_DENSE_P else full range; overrides the uniform y of just-reset cubes (env-origin aware; x/z/quat kept)
                _no = rid.numel()
                _dz = (torch.rand(_no, device=DEVICE) < SPAWN_DENSE_P)
                _ylo = torch.where(_dz, torch.full((_no,), SPAWN_DENSE_LO, device=DEVICE), torch.full((_no,), _SPY_LO, device=DEVICE))
                _yhi = torch.where(_dz, torch.full((_no,), SPAWN_DENSE_HI, device=DEVICE), torch.full((_no,), _SPY_HI, device=DEVICE))
                _ylocal = _ylo + (_yhi - _ylo) * torch.rand(_no, device=DEVICE)     # env-LOCAL y (= distance-1.15)
                _cpw = env.scene[args.target_cube].data.root_pos_w[rid].clone()
                _cpw[:, 1] = env.scene.env_origins[rid, 1] + _ylocal               # world y = env origin y + local y
                env.scene[args.target_cube].write_root_pose_to_sim(torch.cat([_cpw, env.scene[args.target_cube].data.root_quat_w[rid]], dim=-1), env_ids=rid)
                if _dbg_spawn[0]:                                                   # one-time frame-check: confirm resampled distance range
                    _ly = env.scene[args.target_cube].data.root_pos_w[rid, 1] - env.scene.env_origins[rid, 1]
                    print(f"[SPAWN-PEAKED] {_no} cubes | dist {(float(_ly.min())+1.15)*100:.0f}-{(float(_ly.max())+1.15)*100:.0f}cm dense%={float(_dz.float().mean())*100:.0f}", flush=True); _dbg_spawn[0] = False
        _prev_a_a = a_a.detach().clone(); _prev_a_s = a_s.detach().clone(); _prev_is_a = is_a.clone(); _prev_is_s = is_s.clone()   # user 2026-07-16: smoothness prev-action
        gz_prev = gz.clone(); _mz_prev = _mouth_w[:, 2].clone(); df3_prev = _df3.clone(); _vf_prev = _vf.clone(); fd, wd, ori, s = nfd, nwd, nori, ns
        g += (0 if (_curric_on and _sweeping[0]) else N)   # user 2026-07-15: FREEZE g during the curriculum self-eval sweep -> CURRIC_EVERY counts only real TRAINING steps (sweep no longer consumes the training budget)
        _GSTEP[0] = g   # user 2026-07-19: expose current step to get_obs for the PRIV_UNTIL privileged-obs gate
        _tpu = time.perf_counter() if _PROF_ON else 0.0
        if g >= args.learning_starts and not TP_EVAL and not (_curric_on and _sweeping[0]):   # user 2026-07-27: SKIP updates during the curriculum self-eval sweep -> sweep is deterministic eval, doesn't need training; the 64 updates/iter were ~70% of sweep wall-clock (see [PROF]). g is already frozen during sweep so no training-budget change; only wall-clock idles for the ~2-5min sweep. (2026-07-23 kept them ON to not idle the GPU; reverted for sweep speed.)
            for _ in range(args.num_updates):
                if (not FREEZE_SEARCH) and srch["buf"].size() > args.batch_size: _upd(srch)   # user 2026-07-27: FREEZE_SEARCH -> skip search updates (search stays at resumed 100%, only approach trains)
                if (not FREEZE_APPROACH) and appr["buf"].size() > args.batch_size: _upd(appr)   # user 2026-07-30: FREEZE_APPROACH -> skip approach updates (SEARCH-ONLY training)
                if THREE_PHASE and algn["buf"].size() > args.batch_size: _upd(algn)
        if _PROF_ON:
            _psync(); _PROF["upd"] += time.perf_counter() - _tpu
            if _PROF["n"] >= 100:
                _tt = _PROF["step"] + _PROF["render"] + _PROF["detect"] + _PROF["upd"]; _nn = max(_PROF["n"], 1)
                print(f"[PROF] over {_nn} iters | step={_PROF['step']/_nn*1000:.0f}ms render={_PROF['render']/_nn*1000:.0f}ms detect={_PROF['detect']/_nn*1000:.0f}ms upd={_PROF['upd']/_nn*1000:.0f}ms | iter={_tt/_nn*1000:.0f}ms "
                      f"| step={_PROF['step']/_tt*100:.0f}% render={_PROF['render']/_tt*100:.0f}% detect={_PROF['detect']/_tt*100:.0f}% upd={_PROF['upd']/_tt*100:.0f}%", flush=True)
                for _k in _PROF: _PROF[_k] = 0.0 if _k != "n" else 0
        # ---- LIVE UI: numbers EVERY iteration (so the page shows data immediately, not after 50 iters) ----
        for _w in (_win_s, _win_a, _win_l, _win_h, _win_resrch):
            if len(_w) > _METRIC_WIN: del _w[:len(_w)-_METRIC_WIN]                # keep last _METRIC_WIN episodes (unified with rsr/rsr_old)
        _sr = (sum(_tp_recent)/len(_tp_recent)) if _tp_recent else 0.0
        _sro = (sum(_rsr_old_recent)/len(_rsr_old_recent)) if _rsr_old_recent else 0.0   # user 2026-07-22: privileged-ref RSR (rsr_old)
        _tppush("rsr", _sr, cap=600); _tppush("rsr_old", _sro, cap=600)                  # user 2026-07-22: push rsr + rsr_old EVERY iter (others stay %4) — most responsive graphs
        if PRIV_RSR_TARGET > 0.0 and not TP_EVAL and not (_curric_on and _sweeping[0]):   # user 2026-07-19: privilege trigger + anneal
            if _priv_astart[0] < 0:
                if (len(_tp_recent) >= 500 and _sr >= PRIV_RSR_TARGET) or (PRIV_UNTIL > 0 and g >= PRIV_UNTIL):
                    _priv_astart[0] = g
                    _next_sw[0] = g + PRIV_ANNEAL   # user 2026-07-19 FIX: first sweep AFTER the anneal COMPLETES (privilege->0), not at onset. Sweeping at onset measured the still-privilege-dependent policy on real detection = 0% + froze g at 256 + would mark every cell keep-priv. No sweeps during full-priv OR during the anneal ramp; sweeps + region-priv begin once deployable.
                    print(f"[PRIV] anneal START @step {g:,} -> privilege 1->0 over {PRIV_ANNEAL:,} steps; first eval sweep at step {g + PRIV_ANNEAL:,} (anneal done), then every {CURRIC_EVERY:,}", flush=True)
                _PRIV_W[0] = 1.0
            else:
                _PRIV_W[0] = max(0.0, 1.0 - (g - _priv_astart[0]) / max(PRIV_ANNEAL, 1))
        _TPUI.update(step=g, phs=int(is_s.sum().item()), pha=int(is_a.sum().item()), pla=int(is_l.sum().item()), sbuf=srch["buf"].size(), abuf=appr["buf"].size(),
                     labuf=(algn["buf"].size() if THREE_PHASE else 0), all=(float(algn["alpha"]) if THREE_PHASE else 0.0),
                     rsr=_sr, n=len(_tp_recent), rs=float(torch.nan_to_num(r_s).mean().item()), ra=float(torch.nan_to_num(r_a).mean().item()), vis=float(nvis.mean().item()), rl=(float(torch.nan_to_num(r_l).mean().item()) if THREE_PHASE else 0.0),   # nan_to_num: a diverged env's reward is NaN; keep the DISPLAY mean finite (2026-07-21)
                     rsr_old=((sum(_rsr_old_recent)/len(_rsr_old_recent)) if _rsr_old_recent else 0.0), n_old=len(_rsr_old_recent),   # user 2026-07-20: PRIVILEGED reference RSR (true jaw<->cube dist<RSR_DCUBE held) — monitor only, compare vs deployable RSR
                     frac=(float(nwd[is_a & (nvis > 0.5), 2].mean().item()) if bool((is_a & (nvis > 0.5)).any()) else (float(nwd[nvis > 0.5, 2].mean().item()) if bool((nvis > 0.5).any()) else 0.0)), sweep=float(a_s.abs().mean().item()),   # user 2026-07-15: ABS wrist frac = YOLO bbox area/img area, MEAN over APPROACH+visible envs (fallback any-visible) — visible-gated so non-detecting zeros don't drown it; mean smoother than median (display only, reward is per-env)
                     frac_s=(float(nwd[is_s & (nvis > 0.5), 2].mean().item()) if bool((is_s & (nvis > 0.5)).any()) else 0.0),   # user 2026-07-20: SEARCH-phase wrist frac (visible search envs) — separate from approach frac
                     frac_a=(float(nwd[is_a & (nvis > 0.5), 2].mean().item()) if bool((is_a & (nvis > 0.5)).any()) else 0.0),   # user 2026-07-20: APPROACH-phase wrist frac (visible approach envs) — explicit
                     fracd=(float((nwd[is_a & (nvis > 0.5) & (wd[:, 3] > 0.5), 2] - wd[is_a & (nvis > 0.5) & (wd[:, 3] > 0.5), 2]).mean().item()) if bool((is_a & (nvis > 0.5) & (wd[:, 3] > 0.5)).any()) else (float((nwd[(nvis > 0.5) & (wd[:, 3] > 0.5), 2] - wd[(nvis > 0.5) & (wd[:, 3] > 0.5), 2]).mean().item()) if bool(((nvis > 0.5) & (wd[:, 3] > 0.5)).any()) else 0.0)),   # user 2026-07-15: RAW per-step wrist Δfrac, MEAN over BOTH-frames-valid+APPROACH envs (matches the reward's new both-valid gate; NO W_FR) — ~0 during pure SEARCH is expected; rb:frac-delta = this × W_FR

                     acqr=(float(_rt_acq[is_s].mean().item()) if bool(is_s.any()) else 0.0),   # user 2026-07-19: mean FRONT-GUIDED acquisition reward over SEARCH envs (rises as blind-search envs get pulled toward acquiring; ->0 once centered/visible)
                     acqcov=(float(((_ACQ["pv"] if _ACQ.get("pv") is not None else torch.zeros(N, device=DEVICE, dtype=torch.bool)) & is_s & (nvis < 0.5)).sum().item()/max(int((is_s & (nvis < 0.5)).sum().item()), 1)) if W_ACQ_F > 0.0 else 0.0),   # % of blind SEARCH envs that have a FRONT guide (front sees cube) — how many are being actively aimed
                     ncen_s=(float(ncen[is_s & (nvis > 0.5)].median().item()) if bool((is_s & (nvis > 0.5)).any()) else 0.0),   # user 2026-07-10: SEARCH-phase MEDIAN centering err (visible search envs) — unified with UI median
                     ncen_a=(float(ncen[is_a & (nvis > 0.5)].median().item()) if bool((is_a & (nvis > 0.5)).any()) else 0.0),   # user 2026-07-10: APPROACH-phase MEDIAN centering err (visible approach envs)
                     ncen0=float(ncen[0].item()),   # env0 centering (always, not just eval)
                     cenrate=float(((ncen < SEARCH_CENTER) & (nvis > 0.5) & is_s).sum().item()/max(int(is_s.sum().item()), 1)),   # user 2026-07-03: % of SEARCHING+visible envs currently CENTERED (ncen<SEARCH_CENTER) — leading indicator of handoff (robust over 256 env, replaces noisy rolling success)
                     apprate=float(int(is_a.sum().item())/max(N, 1)),   # user 2026-07-03: SEARCH->APPROACH throughput = fraction of ALL envs currently in APPROACH (made it past search)
                     appr_cen=float(((ncen < APP_CEN) & (nvis > 0.5) & is_a).sum().item()/max(int(is_a.sum().item()), 1)),   # user 2026-07-06: % of APPROACH+visible envs CENTERED (ncen<APP_CEN) — the real approach-centering metric (the search-based cenrate reads ~0 under NO_RESRCH since is_s≈0)
                     head_deg=float((_downang[is_a].mean().item()*57.2958) if bool(is_a.any()) else 0.0),   # user 2026-07-06: mean SIGNED NOSE angle from horizontal (deg) over APPROACH envs — "head angle": + = tilt down (down), − = tilt back (up), 0 = level (level). W_WPITCH(expo, symmetric) pulls |this|->0.
                     head_deg_s=float((_downang[is_s].mean().item()*57.2958) if bool(is_s.any()) else 0.0),   # user 2026-07-07: SEARCH-phase mean nose angle — watch wflex dive during search (ramps toward +128° = nose-dive)
                     dmin=(float(gz[is_a].median().item()) if bool(is_a.any()) else float(gz.median().item())), dcube=(float(_dcube[is_a & ~_cube_gone].median().item()) if bool((is_a & ~_cube_gone).any()) else 0.0),   # user 2026-07-10: exclude flown-away cubes (else bimodal median/tm garbage)
                     dcube_s=(float(_dcube[is_s & ~_cube_gone].median().item()) if bool((is_s & ~_cube_gone).any()) else 0.0),   # user 2026-07-22: SEARCH-phase gripper<->cube dist (median, flown-away excluded) — separate from approach
                     dcube_a=(float(_dcube[is_a & ~_cube_gone].median().item()) if bool((is_a & ~_cube_gone).any()) else 0.0),   # user 2026-07-22: APPROACH-phase gripper<->cube dist (explicit; == legacy 'dcube')
                     cdf=(float(_df[_vf & is_a].median().item()) if bool((_vf & is_a).any()) else 0.0), cdw=(float(_dw[_vw & is_a].median().item()) if bool((_vw & is_a).any()) else 0.0),   # front / wrist over their OWN valid subsets (differ by subset, not comparable)
                     cdf3=(float(_df3[_vf & is_a].median().item()) if bool((_vf & is_a).any()) else 0.0), cdw3=(float(_dw3[_vw & is_a].median().item()) if bool((_vw & is_a).any()) else 0.0),   # user 2026-07-10: DEPLOYABLE 3D dist (front/wrist) — compare vs privileged 'dcube' to validate unproj accuracy
                     dcube_tm=_trimmean(_dcube[is_a & ~_cube_gone]), cdf3_tm=_trimmean(_df3[_vf & is_a]), ncen_a_tm=_trimmean(ncen[is_a & (nvis > 0.5)]), dmin_tm=_trimmean(gz[is_a]), frac_tm=_trimmean(nwd[is_a, 2]), cube_gone=int(_cube_gone.sum().item()),   # user 2026-07-10: TRIMMED-MEAN (8 outliers removed) versions — separate cards; cube_gone = #envs whose cube flew away (physics)
                     rb_desc=_ma(_rt_desc), rb_frac=_ma(_rt_frac), rb_novis=_ma(_rt_novisA), rb_tab=_ma(-_tabpen), rb_gone=_ma(-_gone_pen), mxy=(float(_mouth_xy_err[is_a & _wv].median().item()) if bool((is_a & _wv).any()) else 0.0),   # REFACTOR 2026-07-27: only surviving reward terms
                     dcube_v=(float(_dcube[_vf & is_a].median().item()) if bool((_vf & is_a).any()) else 0.0),   # privileged dcube on the SAME front-valid subset (apples-to-apples vs cdf3)
                     cdtri=(float(_dtri[_vf & _vw & is_a].median().item()) if bool((_vf & _vw & is_a).any()) else 0.0), tgap=(float(_tgap[_vf & _vw & is_a].median().item()) if bool((_vf & _vw & is_a).any()) else 0.0),   # user 2026-07-10: STEREO triangulated 3D dist (median, both-cam-valid) + ray gap (intersection error; smaller = more reliable)
                     cdfm=(float(_df[_vf & _vw & is_a].median().item()) if bool((_vf & _vw & is_a).any()) else 0.0), cdwm=(float(_dw[_vf & _vw & is_a].median().item()) if bool((_vf & _vw & is_a).any()) else 0.0), cdnm=int((_vf & _vw & is_a).sum().item()),   # user 2026-07-09: MATCHED — front vs wrist on the SAME both-valid envs -> should AGREE if unproj correct

                     reachz=((sum(1 for x in _gzmin_recent if x < 0.12)/len(_gzmin_recent)) if _gzmin_recent else 0.0),   # approach-progress: ep-min gz + ep-min cube dist + reach% below 0.12
                     crs=_cum["s"], cra=_cum["a"], crl=_cum["l"], three=THREE_PHASE,
                     gz=float(torch.nan_to_num(gz).median().item()), als=float(srch["alpha"]), ala=float(appr["alpha"]), succ=_tp_succ_total,   # nan_to_num: diverged env gz(FK)=NaN; keep DISPLAY finite (2026-07-21)
                     scs=_succ_s, sca=_succ_a, scl=_succ_l,
                     cur_s=int(to_app.sum().item()), cur_a=int((to_align | succ).sum().item()), cur_l=int(succ_l.sum().item()),
                     rate_s=(_succ_s/max(_ent_s,1)), rate_a=(_succ_a/max(_ent_a,1)), rate_l=(_succ_l/max(_ent_l,1)), phase0=int(ph[0].item()),
                     wr_s=(sum(_win_s)/len(_win_s) if _win_s else 0.0), wn_s=len(_win_s), wr_a=(sum(_win_a)/len(_win_a) if _win_a else 0.0), wn_a=len(_win_a), wr_l=(sum(_win_l)/len(_win_l) if _win_l else 0.0), wn_l=len(_win_l), wr_h=(sum(_win_h)/len(_win_h) if _win_h else 0.0), wn_h=len(_win_h), wr_resrch=(sum(_win_resrch)/len(_win_resrch) if _win_resrch else 0.0), wn_resrch=len(_win_resrch))   # user 2026-07-02: rolling per-phase success rates (last 100 attempts)
        if TP_EVAL or (TRAIN_VIZ and (g // N) % 3 == 0):                                 # accumulate env0 DR frame (front|wrist) for the LIVE CLIP (every step in eval)
            try:
                _ff = _LAST_IMG["front"]; _ww = _LAST_IMG["wrist"]
                if _ff is not None and _ww is not None:
                    _fr = torch.cat([_ff[0, ::3, ::3, :3], _ww[0, ::3, ::3, :3]], dim=1).detach().cpu().numpy().astype("uint8")
                    _tp_frames.append(_fr)
                    if len(_tp_frames) > 120: _tp_frames.pop(0)   # 20s clip @ fps6 (user 2026-07-02, was 50=8s)
            except Exception: pass
        if TP_EVAL or (TRAIN_VIZ and (g // N) % 2 == 0):                                  # user 2026-07-03: front/wrist PNG draw at HIGHER cadence (split out of the %4 GIF block) so JS-polled video is smooth, not choppy
            try:                                                          # DR-composited (gray table/black gripper = detector input), det overlay + direction arrows
                _cf0 = _LAST_IMG["front"] if _LAST_IMG["front"] is not None else env.scene["front"].data.output["rgb"][..., :3]
                _cw0 = _LAST_IMG["wrist"] if _LAST_IMG["wrist"] is not None else env.scene["wrist"].data.output["rgb"][..., :3]
                _eeq = r.data.body_quat_w[0, ee_body_idx]; (_rr,_pp,_yy), _gax = _quat_rpy(_eeq)   # user 2026-07-03: gripper orientation for UI
                _TPUI["rpy"]=(round(_rr,1),round(_pp,1),round(_yy,1)); _TPUI["quat"]=[round(float(v),3) for v in _eeq.tolist()]
                _cf = _cf0[0, ..., :3].detach().cpu().numpy().astype("uint8").copy(); _draw_det(_cf, nfd[0].tolist())
                _H=_cf.shape[0]; _sc=_H*0.42; _draw_arrow(_cf, _H//2, _H//2, _gax[0]*_sc, -_gax[1]*_sc); _TPUI["front"] = _png(_cf)
                _wimg = _cw0[0, ..., :3].detach().cpu().numpy().astype("uint8").copy(); _wd=nwd[0].tolist(); _draw_det(_wimg, _wd)   # user 2026-07-20: renamed _cw->_wimg (name collision with curric spawn-weight _cw -> multinomial crash under TRAIN_VIZ=1+CURRIC=1)
                _ww=_wimg.shape[0]                                                # user 2026-07-03: wrist shows ITS OWN dir = toward the cube DETECTED in the wrist view (center->detection). Was a dup of front's world grasp-axis so both looked identical. Short arrow / into-page ring == cube centered.
                if len(_wd)>3 and _wd[3]>0.5: _draw_arrow(_wimg, _ww//2, _ww//2, (_wd[0]-0.5)*_ww, (_wd[1]-0.5)*_ww, col=(255,140,0))
                _TPUI["wrist"] = _png(_wimg)
                _evh.video_frame(_cf, _wimg, int(ph[0].item()))   # 2026-11-02: opt-in rollout video of env 0 (EVAL_VIDEO_DIR)
            except Exception: pass
        if TP_EVAL or (TRAIN_VIZ and (g // N) % 4 == 0):                                  # sparklines + clip GIF — user 2026-07-02: 10->4 (more frequent live clip)
            _tppush("vis", _TPUI["vis"]); _tppush("rs", _TPUI["rs"]); _tppush("ra", _TPUI["ra"]); _tppush("ncen_s", _TPUI.get("ncen_s",0.0)); _tppush("ncen_a", _TPUI.get("ncen_a",0.0))   # user 2026-07-22: rsr/rsr_old moved to every-iter push above
            _tppush("rl", _TPUI.get("rl",0.0)); _tppush("crl", _TPUI.get("crl",0.0)); _tppush("crs", _TPUI.get("crs",0.0)); _tppush("cra", _TPUI.get("cra",0.0))
            _tppush("dmin", _TPUI.get("dmin",0.0)); _tppush("dcube", _TPUI.get("dcube",0.0)); _tppush("dcube_s", _TPUI.get("dcube_s",0.0))
            try:
                if len(_tp_frames) >= 4:
                    _gb = io.BytesIO(); imageio.mimsave(_gb, _tp_frames, format="GIF", fps=6, loop=0); _TPUI["clip"] = _gb.getvalue()
            except Exception: pass
        if (g // N) % 50 == 0:
            _algn_str = (f"| ALIGN envs={int(is_l.sum().item())} buf={algn['buf'].size()} a={float(algn['alpha']):.3f} " if THREE_PHASE else "")
            _dbg_da = float(_downang[is_a].mean().item()*57.2958) if bool(is_a.any()) else 0.0    # mean grasp-axis(local+z) DOWN-angle (deg) over approach envs
            _dbg_mz = float(_mouth_w[is_a, 2].mean().item()) if bool(is_a.any()) else 0.0          # mean jaw-MOUTH z over approach envs (DESC_FLOOR clamps this)
            _hdir_dbg = quat_apply(r.data.body_quat_w[:, ee_body_idx, :], _JAW_OFF3.expand(N, 3))   # wrist->jaw MOUTH direction (the visible "head/nose")
            _hda_dbg = torch.asin(torch.clamp(_hdir_dbg[:, 2] / (_hdir_dbg.norm(dim=-1) + 1e-8), -1.0, 1.0))   # nose/mouth-dir angle: + = tilt down (down), − = tilt back (up) — sign fixed 2026-07-06
            _dbg_hd = float(_hda_dbg[is_a].mean().item()*57.2958) if bool(is_a.any()) else 0.0
            _dbg_wf = float(r.data.joint_pos[:, joint_ids][is_a, WFLEX_I].mean().item()*57.2958) if bool(is_a.any()) else 0.0   # wrist_flex joint angle (deg)
            print(f"[TP] step={g:,} | SEARCH envs={_TPUI['phs']} buf={srch['buf'].size()} a={float(srch['alpha']):.3f} | "
                  f"APPROACH envs={_TPUI['pha']} buf={appr['buf'].size()} a={float(appr['alpha']):.3f} {_algn_str}| "
                  f"RSR={_sr*100:.1f}% (n={len(_tp_recent)}) rsr_old={_TPUI.get('rsr_old',0.0)*100:.1f}%(n={len(_rsr_old_recent)}) LSR={(sum(_lift_recent)/len(_lift_recent)*100 if _lift_recent else 0.0):.1f}% r_s={_TPUI['rs']:+.3f} r_a={_TPUI['ra']:+.3f} gz={_TPUI['gz']:.3f} gaxDown={_dbg_da:.0f}° headDown={_dbg_hd:.0f}° wflex={_dbg_wf:.0f}° mouthz={_dbg_mz:.3f} dcubeS={_TPUI.get('dcube_s',0.0):.3f} dcubeA={_TPUI.get('dcube_a',0.0):.3f} mxy={_TPUI.get('mxy',0.0):.3f} vis={_TPUI['vis']:.2f}{(' priv=%.2f'%_PRIV_W[0]) if PRIV_RSR_TARGET>0.0 else ''}", flush=True)
            _rdm = _reached if bool(_reached.any()) else (is_a & ~_cube_gone)   # user 2026-07-22: standoff diag — over RSR-counted (reached) envs; fallback all approach
            if bool(_rdm.any()):   # decompose TRUE mouth<->cube dist into HORIZONTAL vs VERTICAL to reveal WHY dcube>>0 while RSR fires (lateral miss vs mouth-too-high vs FK offset)
                _hxy = float(torch.norm(_mouth_w[_rdm, :2] - _cube_w[_rdm, :2], dim=-1).mean().item())
                _vz  = float((_mouth_w[_rdm, 2] - _cube_w[_rdm, 2]).mean().item())   # signed: + = mouth ABOVE cube
                print(f"[REACH-DIAG] n={int(_rdm.sum().item())} reached={bool(_reached.any())} dcube={float(_dcube[_rdm].mean().item()):.3f} | horiz(xy)={_hxy:.3f} vert(mouth-cube)={_vz:+.3f} | mouthz={float(_mouth_w[_rdm,2].mean().item()):.3f} cubez={float(_cube_w[_rdm,2].mean().item()):.3f} gz(EE)={float(gz[_rdm].mean().item()):.3f} (RSR uses gz<{APP_GZ}, NOT mouthz)", flush=True)
            _visa = is_a & (nwd[:, 3] > 0.5) & (~_cube_gone)   # user 2026-07-22: AIM-CALIB for CX_TGT/CY_TGT — where does the cube appear in the wrist image WHEN the mouth is actually over it (true horiz < 0.06m)? that pixel = the retarget so "centered" == "mouth over cube".
            if bool(_visa.any()):
                _hoff = torch.norm(_mouth_w[:, :2] - _cube_w[:, :2], dim=-1)
                _sel = _visa & (_hoff < 0.06)
                if bool(_sel.any()):
                    print(f"[AIM-CALIB] n={int(_sel.sum().item())} (horiz<0.06) wrist-cube pixel cx={float(nwd[_sel,0].mean().item()):.3f} cy={float(nwd[_sel,1].mean().item()):.3f}  <- set CX_TGT/CY_TGT to these (now CX_TGT={CX_TGT} CY_TGT={CY_TGT})", flush=True)
            if _rsrd_dist:   # user 2026-07-13: RSR by SPAWN fwd-DISTANCE — split reachable vs unreachable-far
                _rb = []
                for _lo, _hi in [(0.0,0.25),(0.25,0.35),(0.35,0.45),(0.45,0.55),(0.55,0.65),(0.65,1.0)]:
                    _ix = [_i for _i, _d in enumerate(_rsrd_dist) if _lo <= _d < _hi]
                    if _ix: _rb.append(f"{_lo:.2f}-{_hi:.2f}:R{100*sum(_rsrd_hit[_i] for _i in _ix)/len(_ix):.0f}%(n{len(_ix)})")
                print("[RSR-DIST] " + " | ".join(_rb), flush=True)
            if _EVAL_CASES is not None:                                            # FIXED eval-set sweep progress + reproducible RSR over all cases seen
                _sn = _eval_seen; _msk = _sn > 0; _cov = int(_msk.sum().item()); _M = _EVAL_CASES.shape[0]
                _srsr = float((_eval_hits[_msk].sum()/_sn[_msk].sum()).item())*100 if _cov else 0.0
                _srsr_old = float((_eval_hits_old[_msk].sum()/_sn[_msk].sum()).item())*100 if _cov else 0.0   # user 2026-07-22: privileged rsr_old over covered cases
                _ssrch = float((_eval_srch[_msk].sum()/_sn[_msk].sum()).item())*100 if _cov else 0.0   # user 2026-07-22: SEARCH-reached rate over covered cases
                print(f"[EVAL-SET] RSR={_srsr:.1f}% rsr_old={_srsr_old:.1f}% SEARCH={_ssrch:.1f}% | covered {_cov}/{_M} ({100*_cov/_M:.0f}%) | full sweeps={_eval_sweeps[0]} cursor={_eval_cursor[0]}", flush=True)
                _TPUI["eval_rsr"]=_srsr; _TPUI["eval_rsr_old"]=_srsr_old; _TPUI["eval_srch"]=_ssrch; _TPUI["eval_cov"]=_cov; _TPUI["eval_M"]=_M; _TPUI["eval_sweeps"]=_eval_sweeps[0]
                # success HEATMAP over (lateral × forward-dist) — aggregate per-case hits/seen back to grid cells. build for BOTH RSR(deployable) and rsr_old(privileged true-reach).
                def _mk_eval_heat(_hitsT, _lbl):
                    _hh = torch.zeros(_EH_NX*_EH_NY, device=DEVICE); _hs = torch.zeros(_EH_NX*_EH_NY, device=DEVICE)
                    _hh.index_add_(0, _ev_cell, _hitsT); _hs.index_add_(0, _ev_cell, _eval_seen)
                    _hh=_hh.cpu().tolist(); _hs=_hs.cpu().tolist(); _rows=""
                    for _ry in range(_EH_NY-1,-1,-1):                              # far (top) -> near (bottom)
                        _fwd=_ehf0+(_ry+0.5)/_EH_NY*(_ehf1-_ehf0); _cellshtml=""
                        for _rx in range(_EH_NX):
                            _c=_ry*_EH_NX+_rx; _s=_hs[_c]
                            if _s<1: _bg="#222"
                            else:
                                _p=_hh[_c]/_s; _g=int(200*_p); _r=int(200*(1-_p)); _bg=f"rgb({_r},{_g},40)"
                            _cellshtml+=f"<td style='width:11px;height:11px;background:{_bg}' title='fwd~{_fwd*100:.0f}cm {int(_s)}ep R{int(100*_hh[_c]/_s) if _s>0 else 0}%'></td>"
                        _rows+=f"<tr><td style='color:#777;font-size:9px;padding-right:4px'>{_fwd*100:.0f}cm</td>{_cellshtml}</tr>"
                    return (f"<div style='background:#181818;border-radius:10px;padding:10px 14px;margin:5px'>"
                        f"<div style='color:#888;font-size:11px;letter-spacing:1px'>{_lbl} (vertical=forward distance, horizontal=left-right {_ehx0*100:.0f}→{_ehx1*100:.0f}cm · green=success)</div>"
                        f"<table style='border-spacing:2px;margin-top:6px'>{_rows}</table></div>")
                _TPUI["eval_heat"]=_mk_eval_heat(_eval_hits, "RSR heatmap (deployable)")
                _TPUI["eval_heat_old"]=_mk_eval_heat(_eval_hits_old, "rsr_old heatmap (privileged actual reach)")
                _TPUI["eval_heat_srch"]=_mk_eval_heat(_eval_srch, "SEARCH heatmap (APPROACH reached)")
                if _cov >= _M and not int(os.environ.get("EVAL_NOEXIT", "0")):        # user 2026-07-15: EVAL-SET fully covered ONCE -> freeze on FINAL RSR (deterministic; re-runs don't change it). EVAL_NOEXIT=1 keeps live-looping instead.
                    print(f"[EVAL-SET] FULL COVERAGE {_M}/{_M} -> FINAL RSR={_srsr:.1f}% rsr_old={_srsr_old:.1f}% SEARCH={_ssrch:.1f}% (FROZEN — UI live)", flush=True)
                    _evh.finish({"rsr_pct": _srsr, "rsr_old_pct": _srsr_old, "search_pct": _ssrch, "covered": _cov, "M": _M})   # 2026-11-01: opt-in summary + exit
                    while True: _tm_eval.sleep(1.0)                                   # user 2026-07-16 FIX: was _sys.exit(0) -> killed HTTP server, browser stuck on last-polled PARTIAL (e.g. 22%/9). FREEZE instead: process idles, UI keeps serving the FINAL full-coverage RSR+heatmap (_TPUI set just above).
        if (g // N) % 100 == 0 and not TP_EVAL and not (_curric_on and _sweeping[0]):   # user 2026-07-10: save every 100 iters (~25.6k env-steps). user 2026-07-15: don't save during a sweep (g frozen -> would spam identical saves)
            torch.save({"actor":srch["ac"].state_dict(),"critic":srch["cr"].state_dict(),"step":g}, os.path.join(args.save_dir, "search_last.pt"))
            torch.save({"actor":appr["ac"].state_dict(),"critic":appr["cr"].state_dict(),"step":g}, os.path.join(args.save_dir, "approach_last.pt"))
            if THREE_PHASE:
                torch.save({"actor":algn["ac"].state_dict(),"critic":algn["cr"].state_dict(),"step":g}, os.path.join(args.save_dir, "align_last.pt"))
            if SAVE_EVERY > 0 and (g // SAVE_EVERY) > _last_psave[0]:   # user 2026-07-19: PERMANENT step-named ckpt every SAVE_EVERY (never overwritten) -> full history to bench any point
                _last_psave[0] = g // SAVE_EVERY; _sk = f"{CKPT_TAG}_{g//1000}k"
                torch.save({"actor":srch["ac"].state_dict(),"critic":srch["cr"].state_dict(),"step":g}, os.path.join(args.save_dir, f"search_{_sk}.pt"))
                torch.save({"actor":appr["ac"].state_dict(),"critic":appr["cr"].state_dict(),"step":g}, os.path.join(args.save_dir, f"approach_{_sk}.pt"))
                if THREE_PHASE:
                    torch.save({"actor":algn["ac"].state_dict(),"critic":algn["cr"].state_dict(),"step":g}, os.path.join(args.save_dir, f"align_{_sk}.pt"))
                print(f"[TP] PERIODIC ckpt saved (never overwritten): search_{_sk}.pt / approach_{_sk}.pt", flush=True)
            # BEST checkpoint (user 2026-07-02): score = RSR%*1 + reach%(<0.12) as tiebreak -> keep the best policy so regression can't lose it
            _reachz = (sum(1 for x in _gzmin_recent if x < 0.12)/len(_gzmin_recent)) if _gzmin_recent else 0.0
            _score = _sr*100.0 + _reachz
            if len(_tp_recent) >= 200 and _score > _tp_best:
                _tp_best = _score
                torch.save({"actor":srch["ac"].state_dict(),"critic":srch["cr"].state_dict(),"step":g}, os.path.join(args.save_dir, "search_best.pt"))
                torch.save({"actor":appr["ac"].state_dict(),"critic":appr["cr"].state_dict(),"step":g}, os.path.join(args.save_dir, "approach_best.pt"))
                if THREE_PHASE:
                    torch.save({"actor":algn["ac"].state_dict(),"critic":algn["cr"].state_dict(),"step":g}, os.path.join(args.save_dir, "align_best.pt"))
                print(f"[TP] BEST saved @step {g:,}: RSR={_sr*100:.1f}% reach={_reachz*100:.0f}% (score {_score:.3f})", flush=True)
    _os._exit(0)
