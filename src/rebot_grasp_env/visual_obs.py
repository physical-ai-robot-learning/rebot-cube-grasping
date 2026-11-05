"""Visual (eye-in-hand wrist + fixed front) observation setup for the reBot LiftCube env.

Converts the state-only env into a SQUINT-style VISUAL env:
  - policy `state` = proprioception ONLY (qpos 8 + target_qpos 7 + dist_to_rest 6 = 21).
    All PRIVILEGED terms (qvel, obj_pos, obj_quat, tcp_pos, tcp_to_obj, is_grasped) are removed --
    a real robot can't read those directly; the cameras replace them.
  - wrist camera (gripper_link, eye-in-hand) downsampled to 16x16, and a fixed FRONT camera
    downsampled to 64x64. The training loop reads them directly (NOT obs-manager terms).

`env_config.configure_env()` already adds the `wrist` camera (gripper_link/fp_camera, 128x128 rgb)
and disables front/side. configure_visual(): (1) strips privileged obs, (2) adds a FRONT camera at
env level with a baked look-at offset.

Why a baked offset (not set_world_poses_from_view): a scene Camera re-applies its OffsetCfg every
step, so set_world_poses* does NOT stick. But OffsetCfg pos+rot IS honored at spawn. The front cam
is at env level (static parent, identity rot) so its world pose == offset; we bake the look-at quat
into the offset (convention="opengl", exactly what set_world_poses_from_view would compute).

State concat order (MUST match the deploy-side state builder): qpos(8) + target_qpos(7) + dist_to_rest(6).
"""

# privileged / perception-derived obs a real robot can't read -> cameras replace them
_PRIVILEGED_TERMS = ("qvel", "obj_pos", "obj_quat", "tcp_pos", "tcp_to_obj", "is_grasped")

STATE_DIM = 21          # qpos(8) + target_qpos(7) + dist_to_rest(6)

# For DAgger (keep_privileged=True), the env keeps the FULL 43D obs the teacher trained on:
#   qpos[0:8] qvel[8:16] target_qpos[16:23] dist_to_rest[23:29] obj_pos[29:32] obj_quat[32:36]
#   tcp_pos[36:39] tcp_to_obj[39:42] is_grasped[42]  (= 43)
# The student's deployable proprio state (21D) is the qpos+target_qpos+dist_to_rest slice of that:
STUDENT_STATE_IDX = list(range(0, 8)) + list(range(16, 29))   # 8 + 13 = 21
TEACHER_STATE_DIM = 45   # +2 privileged contact-force (2026-06-23); STUDENT_STATE_IDX unchanged (tail-appended)
WRIST_SIZE = 128        # eye-in-hand downsample (raised 16->128 for the dual-cam high-res variant)
FRONT_SIZE = 128        # fixed front downsample (4cm cube ~7-10px at 128 vs ~3-5px at 64).
IMG_RENDER = 256        # PARITY 2026-06-25 (user): render = deploy YOLO imgsz (256). 128 was deploy-mismatched (real needs >=256; frac stats differ by res). was 128
FRONT_FOCAL = 18.0      # front pinhole focal (FoV); jittered per-episode under DR camera variation
WRIST_FOCAL = 12.0      # wrist pinhole focal (FoV); jittered per-episode under DR camera variation

# Front camera look-at (env frame) -- MATCHED TO THE REAL RIG (critical for sim2real):
#   real camera = 90cm FORWARD of the arm base, 55cm high, ~45deg downward, HEAD-ON to the arm.
# base(arm) at (0.35,-0.64,0.01), forward = +y. eye/target x ALIGNED to the arm centerline (0.35) so
# the view is dead-on (no side tilt); eye = base + 0.90*(+y) at z=0.55 -> ~41deg down at the workspace
# center (0.35,-0.30), which is also the centered cube-spawn box center.
FRONT_EYE = (0.35, 0.26, 0.55)
FRONT_TARGET = (0.35, -0.30, 0.06)

# Wrist (eye-in-hand) camera — mounted on the PHYSICAL camera bracket on TOP of the gripper (the small
# square on the gripper's upper face), matching the real rig. gripper_link LOCAL frame axes (verified
# via diag_mount_marker at rest): +x -> world +y (forward/cubes), +y -> world +x (side), +z -> world -z
# (DOWN). So the gripper TOP (world up) = link -z. The camera sits just above the bracket, slightly
# forward of the palm, looking forward+down at the grasp -> the grasped cube + workspace ahead, fingers
# at the frame bottom. 180deg optical roll makes it upright. ROT is computed (not hard-coded) from
# POS->AIM by _wrist_offset_rot() so it exactly matches the calibration render (diag_recalib tmRf10).
WRIST_OFFSET_POS = (-0.012338, 0.0, -0.023574)   # CALIB 2026-06-19: +pitch25,fwd1cm,up3cm     # gripper_link frame: above the bracket, just fwd of the palm
WRIST_AIM_TARGET = (0.05, 0.0, 0.10)       # look forward (+x) + down (+z) at the grasp / workspace ahead
WRIST_ROLL180 = True                       # 180deg optical-axis roll -> upright (fingers at bottom)


def _lookatQuatOpengl(eye, target):
    """Camera orientation quaternion (wxyz, opengl convention) looking from eye to target.
    Replicates Camera.set_world_poses_from_view's math so it can be baked into OffsetCfg."""
    import torch
    from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix
    tE = torch.tensor([list(eye)], dtype=torch.float32)
    tT = torch.tensor([list(target)], dtype=torch.float32)
    tR = create_rotation_matrix_from_view(tE, tT, "Z", device="cpu")
    return tuple(quat_from_matrix(tR)[0].tolist())


def _wristOffsetRot():
    """Wrist camera orientation (gripper_link frame). CALIB 2026-06-19: baked look-at + 180roll + pitch+25deg."""
    return (0.298931, -0.640812, -0.640812, 0.298931)   # calibrated; below is the original derivation (unused)
    tRot = _lookatQuatOpengl(WRIST_OFFSET_POS, WRIST_AIM_TARGET)
    if WRIST_ROLL180:
        import torch
        from isaaclab.utils.math import quat_mul
        tQ = quat_mul(torch.tensor([list(tRot)]), torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
        tRot = tuple(tQ[0].tolist())
    return tRot


def configureVisual(env_cfg, target_cube="cube", keep_privileged=False, dr=False, front_only=False, wrist_only=False, seg=False):
    """Add a fixed front camera (baked look-at). By default strip privileged obs -> 21D state.

    keep_privileged=True  -> DAgger: keep the FULL 43D obs (the teacher needs it); the student
                             state is sliced via STUDENT_STATE_IDX. Only the front camera is added.
    keep_privileged=False -> visual-RL student-only: strip privileged terms -> 21D state.
    dr=True  -> visual domain randomization: cameras also output semantic_segmentation (uncolorized)
               and robot/cube prims are semantic-tagged, so the training loop can GREENSCREEN the
               background (replace non robot/cube pixels with a random background). DR is train-only.
    seg=True -> emit semantic_segmentation + apply robot/cube/table tags WITHOUT the full RGB DR
               (the lighting/camera-FoV DR events stay dr-only). Used by the MASK front-end
               (--mask-obs), which builds a domain-invariant {cube,gripper} mask from the seg.
               Backward-compatible: callers that don't pass seg behave exactly as before (want_seg==dr).
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg

    tWantSeg = dr or seg   # cameras output semantic_segmentation + foreground tags when EITHER is set

    # 1) (student-only) strip privileged obs; remaining = qpos + target_qpos + dist_to_rest = 21
    if not keep_privileged:
        tPol = env_cfg.observations.policy
        for tTerm in _PRIVILEGED_TERMS:
            if hasattr(tPol, tTerm):
                setattr(tPol, tTerm, None)

    tDtypes = ["rgb", "semantic_segmentation"] if tWantSeg else ["rgb"]

    # 2) wrist camera. front_only -> drop it entirely (the real wrist mount/framing is OOD; the
    #    student uses front+state only). else OVERRIDE its offset+FoV to the "between the jaws" pose.
    if front_only:
        if hasattr(env_cfg.scene, "wrist"):
            env_cfg.scene.wrist = None
    elif getattr(env_cfg.scene, "wrist", None) is not None:
        env_cfg.scene.wrist.offset = TiledCameraCfg.OffsetCfg(
            pos=WRIST_OFFSET_POS, rot=_wristOffsetRot(), convention="opengl")
        env_cfg.scene.wrist.spawn = sim_utils.PinholeCameraCfg(
            focal_length=12.0, clipping_range=(0.005, 50.0), lock_camera=True)
        env_cfg.scene.wrist.data_types = list(tDtypes)
        env_cfg.scene.wrist.width = IMG_RENDER     # match front render res (else composite/downsample mismatch)
        env_cfg.scene.wrist.height = IMG_RENDER
        if tWantSeg:
            env_cfg.scene.wrist.colorize_semantic_segmentation = False

    # 3) front camera: fixed third-person, env-level prim (static parent) with baked look-at offset.
    #    wrist_only -> drop it entirely (real rig has no front cam; policy is wrist+state = Squint LiftCube).
    if wrist_only:
        if hasattr(env_cfg.scene, "front"):
            env_cfg.scene.front = None
    else:
        tSpawn = sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.01, 50.0), lock_camera=True)
        tFrontKw = dict(
            prim_path="{ENV_REGEX_NS}/front_camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=FRONT_EYE, rot=_lookatQuatOpengl(FRONT_EYE, FRONT_TARGET), convention="opengl"),
            data_types=list(tDtypes), spawn=tSpawn, width=IMG_RENDER, height=IMG_RENDER, update_period=0)
        if tWantSeg:
            tFrontKw["colorize_semantic_segmentation"] = False
        env_cfg.scene.front = TiledCameraCfg(**tFrontKw)

    # 4) semantic-tag robot + all cubes + table as foreground. Needed for BOTH greenscreen (dr) and the
    #    {cube,gripper} mask (seg) -> gated on want_seg so --mask-obs gets tags without the full RGB DR.
    if tWantSeg:
        if getattr(env_cfg.scene, "robot", None) is not None and getattr(env_cfg.scene.robot, "spawn", None) is not None:
            env_cfg.scene.robot.spawn.semantic_tags = [("class", "robot")]
        for tCn in ("cube", "cube_blue", "cube_purple"):
            tC = getattr(env_cfg.scene, tCn, None)
            if tC is not None and getattr(tC, "spawn", None) is not None:
                tC.spawn.semantic_tags = [("class", "cube")]
        # also tag the TABLE scene USD so the greenscreen KEEPS the rendered table (cube/arm sit on a real
        # surface by its actual silhouette -> no floating, no hard horizon); only the true far-bg is replaced.
        tScn = getattr(env_cfg.scene, "scene", None)
        if tScn is not None and getattr(tScn, "spawn", None) is not None:
            tScn.spawn.semantic_tags = [("class", "table")]

    # 5/6) lighting + camera-FoV DR events — RGB-DR only (a mask has no appearance/lighting), so dr-gated.
    if dr:
        # 5) (dr) lighting randomization per reset — per-env scene light (/World/envs/env_*/Light).
        #    only changes rendered rgb (student input); does NOT touch physics/state -> teacher safe.
        from isaaclab.managers import EventTermCfg as _EventTerm
        env_cfg.events.dr_lighting = _EventTerm(func=randomizeLight, mode="reset",
            params={"intensity_range": (700.0, 3500.0), "color_lo": 0.65, "color_hi": 1.0})
        # 6) (dr) camera FoV (focal) variation per reset for front + wrist -> robust to the real
        #    camera's focal/FoV not exactly matching sim. focal is a per-prim USD attr (sticks).
        if not front_only:
            env_cfg.events.dr_camera = _EventTerm(func=randomizeCamera, mode="reset",
                params={"focal_jitter": 0.10})   # Squint-aligned (was 0.30; +-30% FoV swing drowned the small cube)

    return env_cfg


def randomizeCamera(env, env_ids, focal_jitter=0.10):
    """(DR reset event) jitter front + wrist camera focal length (FoV) per env, within +/-focal_jitter.
    Only affects the rendered FoV (the student's visual input) -> robust to real camera focal mismatch.
    focalLength is a per-prim USD attr (UsdGeom.Camera), so unlike pose it persists. Fully guarded."""
    try:
        import random
        import omni.usd
        from pxr import UsdGeom
        tStage = omni.usd.get_context().get_stage()
        tIds = env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)
        for tI in tIds:
            tIi = int(tI)
            for tPath, tFocal in ((f"/World/envs/env_{tIi}/front_camera", FRONT_FOCAL),
                                (f"/World/envs/env_{tIi}/Robot/gripper_link/fp_camera", WRIST_FOCAL)):
                tP = tStage.GetPrimAtPath(tPath)
                if not tP or not tP.IsValid():
                    continue
                UsdGeom.Camera(tP).GetFocalLengthAttr().Set(tFocal * (1.0 + random.uniform(-focal_jitter, focal_jitter)))
    except Exception:
        pass


def randomizeLight(env, env_ids, intensity_range=(700.0, 3500.0), color_lo=0.65, color_hi=1.0):
    """(DR reset event) randomize each resetting env's scene light intensity + color (UsdLux).
    Only affects rendered rgb (the student's visual input), not physics/state — teacher unaffected.
    Fully guarded: any failure is a no-op so it can never crash training."""
    try:
        import random
        import omni.usd
        from pxr import UsdLux, Gf
        tStage = omni.usd.get_context().get_stage()
        tIds = env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)
        for tI in tIds:
            tP = tStage.GetPrimAtPath(f"/World/envs/env_{int(tI)}/Light")
            if not tP or not tP.IsValid():
                continue
            tApi = UsdLux.LightAPI(tP)
            tApi.GetIntensityAttr().Set(random.uniform(*intensity_range))
            tApi.GetColorAttr().Set(Gf.Vec3f(*(random.uniform(color_lo, color_hi) for _ in range(3))))
    except Exception:
        pass


def downsample(rgb, size):
    """(N,H,W,C>=3) uint8 -> (N,size,size,3) uint8 via area interpolation (SQUINT-style)."""
    import torch
    import torch.nn.functional as F
    tX = rgb[..., :3].permute(0, 3, 1, 2).float()
    tX = F.interpolate(tX, size=(size, size), mode="area")
    return tX.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8)


def segToMask(seg, info, class_groups, size):
    """Build a multi-channel DOMAIN-INVARIANT mask image from semantic segmentation (the MASK front-end).

    seg (N,H,W,1) int class-ids; info = camera .info["semantic_segmentation"] (idToLabels);
    class_groups = e.g. (("cube",), ("gripper",)) -> ONE channel per group (build_fg_ids matches labels
    by substring). Each channel = that group's silhouette, area-downsampled to `size` (soft 0..1 -> 0..255).

    Why a mask: the policy input is then IDENTICAL in sim (GT seg) and real (HSV / segmenter) -> raw-pixel
    vision OOD is removed by construction. Returns (N,size,size,len(class_groups)) uint8."""
    import torch
    import torch.nn.functional as F
    tSids = seg[..., 0]
    tN2, tH, tW = tSids.shape[0], tSids.shape[1], tSids.shape[2]
    tChans = []
    for tGrp in class_groups:
        tIds = buildFgIds(info, fg_classes=tGrp)
        if tIds:
            tIdT = torch.as_tensor(tIds, device=tSids.device, dtype=tSids.dtype)
            tM = torch.isin(tSids, tIdT).float()
        else:
            tM = torch.zeros((tN2, tH, tW), device=tSids.device, dtype=torch.float32)
        tChans.append(tM)
    tMask = torch.stack(tChans, dim=1)                          # (N,C,H,W)
    tMask = F.interpolate(tMask, size=(size, size), mode="area")  # soft (anti-aliased) -> keeps tiny-cube coverage
    return (tMask.permute(0, 2, 3, 1) * 255.0).round().clamp(0, 255).to(torch.uint8)


def studentState(full_obs):
    """Slice the deployable 21D proprio state (qpos+target_qpos+dist_to_rest) from the full 43D obs.
    Used in DAgger where the env keeps the full privileged obs for the teacher."""
    return full_obs[:, STUDENT_STATE_IDX]


# ── visual domain randomization (DR) — train-only ─────────────────────────────
def tagGripper(env, body_names=("gripper_link", "gripper_left", "gripper_right")):
    """(DR setup) apply a separate 'gripper' semantic class to the gripper body prims (post-build) so the
    seg distinguishes the gripper from the rest of the robot -> recolor_seg_region(("gripper",)) can paint
    it BLACK while the arm ('robot') is left as rendered. Fully guarded (no-op on failure)."""
    try:
        import omni.usd
        from pxr import Semantics
        tStage = omni.usd.get_context().get_stage()
        tNtag = 0
        for tI in range(env.num_envs):
            for tBn in body_names:
                tP = tStage.GetPrimAtPath(f"/World/envs/env_{tI}/Robot/{tBn}")
                if not tP or not tP.IsValid():
                    continue
                tSem = Semantics.SemanticsAPI.Apply(tP, "Semantics")
                tSem.CreateSemanticTypeAttr().Set("class")
                tSem.CreateSemanticDataAttr().Set("gripper")
                tNtag += 1
        print(f"[DAG] tag_gripper: tagged {tNtag} gripper prims as 'gripper'", flush=True)
    except Exception as tE:
        print(f"[DAG] tag_gripper failed ({tE}) -> gripper stays 'robot'", flush=True)


def buildFgIds(info, fg_classes=("robot", "cube", "table", "gripper")):
    """Foreground semantic ids (robot+cube) from a camera's segmentation info dict. Rebuilt each
    frame because replicator ids are not stable across resets. Returns a list[int] (empty if none)."""
    tId2l = (info or {}).get("idToLabels", {}) if isinstance(info, dict) else {}
    tOut = []
    for tK, tV in tId2l.items():
        cls = tV.get("class", "") if isinstance(tV, dict) else str(tV)
        if any(tC2 in cls for tC2 in fg_classes):
            try:
                tOut.append(int(tK))
            except (ValueError, TypeError):
                pass
    return tOut


def compositeBg(rgb, seg, info, bg, keep_lower=0.0):
    """Greenscreen: keep foreground (robot+cube) pixels — PLUS the lower `keep_lower` fraction of the image
    (the rendered TABLE surface, with a mild per-env size jitter) so the cube/arm sit on a REAL surface and
    don't FLOAT on the background — and replace only the rest (the far background = upper non-foreground)
    with `bg`. rgb (N,H,W,>=3) uint8, seg (N,H,W,1) int, info seg-info dict, bg (N,H,W,3). -> (N,H,W,3) uint8.
    Returns rgb unchanged if no foreground id resolves (seg not ready)."""
    import torch
    tFg = buildFgIds(info)
    tRgb3 = rgb[..., :3]
    if not tFg:
        return tRgb3
    tN2, tH, tW = tRgb3.shape[0], tRgb3.shape[1], tRgb3.shape[2]
    tSids = seg[..., 0]
    tFgT = torch.as_tensor(tFg, device=tSids.device, dtype=tSids.dtype)
    tKeep = torch.isin(tSids, tFgT)                             # (N,H,W) True=foreground
    if keep_lower > 0.0:
        tRows = torch.linspace(0.0, 1.0, tH, device=tRgb3.device).view(1, tH, 1)
        tLo = (1.0 - keep_lower) + torch.empty(tN2, 1, 1, device=tRgb3.device).uniform_(-0.04, 0.04)
        tKeep = tKeep | (tRows >= tLo)                            # + lower table region (mild per-env size)
    tOut = torch.where(tKeep.unsqueeze(-1), tRgb3.float(), bg[..., :3].float())
    return tOut.round().clamp(0, 255).to(torch.uint8)


def recolorSegRegion(rgb, seg, info, classes=("table",), b_lo=0.50, b_hi=1.05, per_ch=0.18, grain=True, base=0.0):
    """(DR) recolor (+ optional procedural RETEXTURE) ONLY the pixels of `classes` (by semantic silhouette),
    e.g. ("table",) or ("robot",). Keeps the rendered luminance/shading (lum) so shape/perspective are
    preserved (the table/robot stays grounded, no floating), then applies a per-env color tint (+ grain for
    surface texture). `base` lifts the luminance floor (base + (1-base)*lum) so a WHITE surface stays bright
    (e.g. base~0.62 + per_ch=0 = a fixed WHITE table with subtle shading). `per_ch=0` = no color, just
    brightness. NOTE: cubes are deliberately NOT recolored (cube color = the task cue). -> (N,H,W,3) uint8."""
    import torch
    import torch.nn.functional as F
    tX = rgb[..., :3].float()
    tIds = buildFgIds(info, fg_classes=classes)
    if not tIds:
        return tX.round().clamp(0, 255).to(torch.uint8)
    tN2, tH, tW = rgb.shape[0], rgb.shape[1], rgb.shape[2]
    tDev = rgb.device
    tSids = seg[..., 0]
    tIdT = torch.as_tensor(tIds, device=tDev, dtype=tSids.dtype)
    tMask = torch.isin(tSids, tIdT).unsqueeze(-1).float()                        # (N,H,W,1) target pixels
    tLum = tX.mean(-1, keepdim=True) / 255.0                                     # keep shading/perspective
    tLum = base + (1.0 - base) * tLum                                            # lift floor (white -> bright)
    tB = torch.empty(tN2, 1, 1, 1, device=tDev).uniform_(b_lo, b_hi)              # per-env brightness
    tTint = tB * (1.0 + per_ch * (torch.rand(tN2, 1, 1, 3, device=tDev) - 0.5))    # per-env color (per_ch=0 -> none)
    tNew = tLum * tTint
    if grain:
        tF = torch.randint(4, 16, (1,)).item()                                 # per-env grain frequency = TEXTURE
        tG = torch.rand(tN2, tF, tF, 1, device=tDev)
        tG = F.interpolate(tG.permute(0, 3, 1, 2), size=(tH, tW), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        tNew = tNew * (0.78 + 0.44 * tG)
    tNew = (tNew * 255.0).clamp(0, 255)
    tOut = tMask * tNew + (1.0 - tMask) * tX
    return tOut.round().clamp(0, 255).to(torch.uint8)


def viewpointWarp(img, scale=0.18, rot=0.14, shift=0.08, shear=0.10):
    """(DR, FRONT cam) per-env random AFFINE warp — scale (=camera distance), rotation, translation
    (=mount offset) and shear (=mild viewpoint tilt) — so the student is robust to the real front camera
    pose not exactly matching sim (deploy feedback: a top-down-ish real mount put the cube far OOD).
    NOTE: affine is an APPROXIMATION of a viewpoint change (no true perspective/parallax); matching the
    real mount to DEPLOY_SPEC §4 (90cm fwd / 55cm / ~41deg) is still the primary fix — this is the margin.
    img (N,H,W,3) uint8 -> (N,H,W,3) uint8."""
    import torch
    import torch.nn.functional as F
    tN2, tH, tW, _ = img.shape
    tDev = img.device
    tX = img.permute(0, 3, 1, 2).float()                              # (N,3,H,W)
    tS = 1.0 + (torch.rand(tN2, device=tDev) - 0.5) * 2 * scale          # per-env scale (distance)
    tTh = (torch.rand(tN2, device=tDev) - 0.5) * 2 * rot                 # rotation (roll)
    tSh = (torch.rand(tN2, device=tDev) - 0.5) * 2 * shear               # shear (tilt)
    tTx = (torch.rand(tN2, device=tDev) - 0.5) * 2 * shift               # translate (mount offset)
    tTy = (torch.rand(tN2, device=tDev) - 0.5) * 2 * shift
    tCos, tSin = torch.cos(tTh), torch.sin(tTh)
    tTheta = torch.zeros(tN2, 2, 3, device=tDev)
    tTheta[:, 0, 0] = (tCos + tSh * tSin) * tS
    tTheta[:, 0, 1] = (-tSin + tSh * tCos) * tS
    tTheta[:, 1, 0] = tSin * tS
    tTheta[:, 1, 1] = tCos * tS
    tTheta[:, 0, 2] = tTx
    tTheta[:, 1, 2] = tTy
    tGrid = F.affine_grid(tTheta, tX.shape, align_corners=False)
    tOut = F.grid_sample(tX, tGrid, mode="bilinear", padding_mode="border", align_corners=False)
    return tOut.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8)


def wristAug(img, roll, tx, ty):
    """(DR, wrist only) per-env roll + translate augmentation so the student is robust to the real
    wrist-cam mount being rotated/offset vs sim (real rig was 90deg off; 128px eye-in-hand is very
    roll/pos sensitive). img (N,H,W,3) uint8; roll (N,) rad; tx,ty (N,) normalized (fraction of half-
    extent). Applied per-EPISODE (angles redrawn on reset) via a single affine grid_sample."""
    import torch
    import torch.nn.functional as F
    tN2, tH, tW, _ = img.shape
    tX = img.permute(0, 3, 1, 2).float()
    tCos, tSin = torch.cos(roll), torch.sin(roll)
    tTheta = torch.zeros(tN2, 2, 3, device=img.device, dtype=torch.float32)
    tTheta[:, 0, 0] = tCos;  tTheta[:, 0, 1] = -tSin; tTheta[:, 0, 2] = tx
    tTheta[:, 1, 0] = tSin;  tTheta[:, 1, 1] = tCos;  tTheta[:, 1, 2] = ty
    tGrid = F.affine_grid(tTheta, tX.shape, align_corners=False)
    tOut = F.grid_sample(tX, tGrid, mode="bilinear", padding_mode="border", align_corners=False)
    return tOut.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8)


def imageDr(img, glare=True, scale=True):
    """(DR, TRAIN-ONLY, image-space) close the sim2real gaps the greenscreen/lighting DR MISSED, applied
    per-env at FULL render res BEFORE downsample. img (N,H,W,3) uint8 -> (N,H,W,3) uint8.
      gap4 ISP color: per-channel gain (white-balance) + gamma + contrast + brightness — the real camera
           ISP tone/exposure/WB shifts the whole image vs Isaac's linear render (the encoder uses fixed
           /255-0.5 norm, so a global tone shift is pure OOD unless trained against it).
      gap1 specular glare: a per-env elliptical near-white HOTSPOT blended toward white = the glossy white
           table's bright reflection streak (the sim has no glossy table / no specular DR). Cheap proxy
           for full material+light randomization.
      gap2/3 cube apparent-size + aspect/anisotropy: per-env INDEPENDENT x/y affine scale + translate ->
           the cube spans tiny<->large screen-occupancy AND the net is robust to aspect/crop residual
           (real 16:9 center-crop never perfectly matches the sim square FoV). scale only on `scale=True`
           (front); the wrist's geometry DR is wrist_aug (roll/translate)."""
    import torch
    import torch.nn.functional as F
    tDev = img.device
    tN2, tH, tW, _ = img.shape
    tX = img[..., :3].float() / 255.0
    # ISP color: gain (WB) + gamma + contrast + brightness
    tX = tX * torch.empty(tN2, 1, 1, 3, device=tDev).uniform_(0.97, 1.05)   # WB gain (TIGHT -> keep the red cube red; was 0.92-1.10)
    tX = tX.clamp(1e-4, 4.0).pow(torch.empty(tN2, 1, 1, 1, device=tDev).uniform_(0.85, 1.18))
    tM = tX.mean(dim=(1, 2), keepdim=True)
    tX = (tX - tM) * torch.empty(tN2, 1, 1, 1, device=tDev).uniform_(0.90, 1.12) + tM
    tX = (tX + torch.empty(tN2, 1, 1, 1, device=tDev).uniform_(-0.05, 0.05)).clamp(0.0, 1.0)
    # specular-glare proxy: per-env elliptical bright hotspot toward white
    if glare:
        tYy = torch.linspace(0, 1, tH, device=tDev).view(1, tH, 1)
        tXx = torch.linspace(0, 1, tW, device=tDev).view(1, 1, tW)
        tCx = torch.empty(tN2, 1, 1, device=tDev).uniform_(0.1, 0.9)
        tCy = torch.empty(tN2, 1, 1, device=tDev).uniform_(0.05, 0.6)
        tAng = torch.empty(tN2, 1, 1, device=tDev).uniform_(0.0, 3.14159)
        tSx = torch.empty(tN2, 1, 1, device=tDev).uniform_(0.04, 0.30)
        tSy = torch.empty(tN2, 1, 1, device=tDev).uniform_(0.02, 0.10)
        tAlpha = torch.empty(tN2, 1, 1, device=tDev).uniform_(0.0, 0.20)   # gloss TAMED 0.50->0.20 (hotspot can't blow out the red cube)
        tDx = tXx - tCx; tDy = tYy - tCy
        tCa = torch.cos(tAng); tSa = torch.sin(tAng)
        tU = (tDx * tCa + tDy * tSa) / tSx; tV = (-tDx * tSa + tDy * tCa) / tSy
        tG = (tAlpha * torch.exp(-(tU * tU + tV * tV))).unsqueeze(-1)
        tX = (tX + tG * (1.0 - tX)).clamp(0.0, 1.0)
    # cube apparent-size + aspect/anisotropy: per-env independent x/y scale + translate
    if scale:
        tXt = tX.permute(0, 3, 1, 2)
        tTheta = torch.zeros(tN2, 2, 3, device=tDev)
        tTheta[:, 0, 0] = torch.empty(tN2, device=tDev).uniform_(0.80, 1.25)  # scene/table apparent-size variation
        tTheta[:, 0, 2] = torch.empty(tN2, device=tDev).uniform_(-0.07, 0.07)
        tTheta[:, 1, 1] = torch.empty(tN2, device=tDev).uniform_(0.80, 1.25)  # independent -> slight aspect
        tTheta[:, 1, 2] = torch.empty(tN2, device=tDev).uniform_(-0.07, 0.07)
        tGrid = F.affine_grid(tTheta, tXt.shape, align_corners=False)
        tX = F.grid_sample(tXt, tGrid, mode="bilinear", padding_mode="reflection", align_corners=False).permute(0, 2, 3, 1)
    return (tX * 255.0).round().clamp(0, 255).to(torch.uint8)


def addProprioNoise(s21, arm_std=0.0122, grip_std=0.001, tgt_std=0.0122):
    """Add encoder-style gaussian noise to the 21D student state (TRAIN-ONLY, student path).
    Layout (STUDENT_STATE_IDX): qpos[0:8] (arm 0:6 + gripper 6:8) | target_qpos[8:15] | dist_to_rest[15:21].
    dist_to_rest = arm_qpos - REST, so the SAME per-joint draw added to arm qpos MUST be added to
    dist_to_rest (else a physically impossible (qpos,dist) pair). arm_std 0.0122 rad ~= 0.7deg."""
    import torch
    tS = s21.clone()
    tN = tS.shape[0]
    tArm = torch.randn(tN, 6, device=tS.device) * arm_std        # shared draw (qpos arm == dist_to_rest base)
    tS[:, 0:6] += tArm
    tS[:, 15:21] += tArm
    tS[:, 6:8] += torch.randn(tN, 2, device=tS.device) * grip_std     # gripper qpos
    tS[:, 8:15] += torch.randn(tN, 7, device=tS.device) * tgt_std     # target_qpos (controller cmd, independent)
    return tS
