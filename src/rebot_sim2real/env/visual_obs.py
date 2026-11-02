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
TEACHER_STATE_DIM = 43
WRIST_SIZE = 128        # eye-in-hand downsample (raised 16->128 for the dual-cam high-res variant)
FRONT_SIZE = 128        # fixed front downsample (4cm cube ~7-10px at 128 vs ~3-5px at 64).
IMG_RENDER = 192        # native camera render resolution, then area-downsampled (192->128 anti-aliased)
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
WRIST_OFFSET_POS = (-0.02, 0.0, -0.05)     # gripper_link frame: above the bracket, just fwd of the palm
WRIST_AIM_TARGET = (0.05, 0.0, 0.10)       # look forward (+x) + down (+z) at the grasp / workspace ahead
WRIST_ROLL180 = True                       # 180deg optical-axis roll -> upright (fingers at bottom)


def _lookat_quat_opengl(eye, target):
    """Camera orientation quaternion (wxyz, opengl convention) looking from eye to target.
    Replicates Camera.set_world_poses_from_view's math so it can be baked into OffsetCfg."""
    import torch
    from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix
    e = torch.tensor([list(eye)], dtype=torch.float32)
    t = torch.tensor([list(target)], dtype=torch.float32)
    R = create_rotation_matrix_from_view(e, t, "Z", device="cpu")
    return tuple(quat_from_matrix(R)[0].tolist())


def _wrist_offset_rot():
    """Wrist camera orientation (gripper_link frame): look from WRIST_OFFSET_POS toward
    WRIST_AIM_TARGET, then (WRIST_ROLL180) apply a 180deg optical-axis roll so the image is upright.
    Computed lazily (needs isaaclab math + a running app) so it exactly matches the calibration render."""
    rot = _lookat_quat_opengl(WRIST_OFFSET_POS, WRIST_AIM_TARGET)
    if WRIST_ROLL180:
        import torch
        from isaaclab.utils.math import quat_mul
        q = quat_mul(torch.tensor([list(rot)]), torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
        rot = tuple(q[0].tolist())
    return rot


def configure_visual(env_cfg, target_cube="cube", keep_privileged=False, dr=False, front_only=False):
    """Add a fixed front camera (baked look-at). By default strip privileged obs -> 21D state.

    keep_privileged=True  -> DAgger: keep the FULL 43D obs (the teacher needs it); the student
                             state is sliced via STUDENT_STATE_IDX. Only the front camera is added.
    keep_privileged=False -> visual-RL student-only: strip privileged terms -> 21D state.
    dr=True  -> visual domain randomization: cameras also output semantic_segmentation (uncolorized)
               and robot/cube prims are semantic-tagged, so the training loop can GREENSCREEN the
               background (replace non robot/cube pixels with a random background). DR is train-only.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg

    # 1) (student-only) strip privileged obs; remaining = qpos + target_qpos + dist_to_rest = 21
    if not keep_privileged:
        pol = env_cfg.observations.policy
        for term in _PRIVILEGED_TERMS:
            if hasattr(pol, term):
                setattr(pol, term, None)

    _dtypes = ["rgb", "semantic_segmentation"] if dr else ["rgb"]

    # 2) wrist camera. front_only -> drop it entirely (the real wrist mount/framing is OOD; the
    #    student uses front+state only). else OVERRIDE its offset+FoV to the "between the jaws" pose.
    if front_only:
        if hasattr(env_cfg.scene, "wrist"):
            env_cfg.scene.wrist = None
    elif getattr(env_cfg.scene, "wrist", None) is not None:
        env_cfg.scene.wrist.offset = TiledCameraCfg.OffsetCfg(
            pos=WRIST_OFFSET_POS, rot=_wrist_offset_rot(), convention="opengl")
        env_cfg.scene.wrist.spawn = sim_utils.PinholeCameraCfg(
            focal_length=12.0, clipping_range=(0.005, 50.0), lock_camera=True)
        env_cfg.scene.wrist.data_types = list(_dtypes)
        env_cfg.scene.wrist.width = IMG_RENDER     # match front render res (else composite/downsample mismatch)
        env_cfg.scene.wrist.height = IMG_RENDER
        if dr:
            env_cfg.scene.wrist.colorize_semantic_segmentation = False

    # 3) front camera: fixed third-person, env-level prim (static parent) with baked look-at offset.
    _spawn = sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.01, 50.0), lock_camera=True)
    _front_kw = dict(
        prim_path="{ENV_REGEX_NS}/front_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=FRONT_EYE, rot=_lookat_quat_opengl(FRONT_EYE, FRONT_TARGET), convention="opengl"),
        data_types=list(_dtypes), spawn=_spawn, width=IMG_RENDER, height=IMG_RENDER, update_period=0)
    if dr:
        _front_kw["colorize_semantic_segmentation"] = False
    env_cfg.scene.front = TiledCameraCfg(**_front_kw)

    # 4) (dr) semantic-tag robot + all cubes as foreground so the greenscreen keeps them.
    if dr:
        if getattr(env_cfg.scene, "robot", None) is not None and getattr(env_cfg.scene.robot, "spawn", None) is not None:
            env_cfg.scene.robot.spawn.semantic_tags = [("class", "robot")]
        for _cn in ("cube", "cube_blue", "cube_purple"):
            _c = getattr(env_cfg.scene, _cn, None)
            if _c is not None and getattr(_c, "spawn", None) is not None:
                _c.spawn.semantic_tags = [("class", "cube")]
        # also tag the TABLE scene USD so the greenscreen KEEPS the rendered table (cube/arm sit on a real
        # surface by its actual silhouette -> no floating, no hard horizon); only the true far-bg is replaced.
        _scn = getattr(env_cfg.scene, "scene", None)
        if _scn is not None and getattr(_scn, "spawn", None) is not None:
            _scn.spawn.semantic_tags = [("class", "table")]

        # 5) (dr) lighting randomization per reset — per-env scene light (/World/envs/env_*/Light).
        #    only changes rendered rgb (student input); does NOT touch physics/state -> teacher safe.
        from isaaclab.managers import EventTermCfg as _EventTerm
        env_cfg.events.dr_lighting = _EventTerm(func=randomize_light, mode="reset",
            params={"intensity_range": (700.0, 3500.0), "color_lo": 0.65, "color_hi": 1.0})
        # 6) (dr) camera FoV (focal) variation per reset for front + wrist -> robust to the real
        #    camera's focal/FoV not exactly matching sim. focal is a per-prim USD attr (sticks).
        if not front_only:
            env_cfg.events.dr_camera = _EventTerm(func=randomize_camera, mode="reset",
                params={"focal_jitter": 0.30})   # widened 0.10->0.30: real cube apparent-size/FoV spans more

    return env_cfg


def randomize_camera(env, env_ids, focal_jitter=0.10):
    """(DR reset event) jitter front + wrist camera focal length (FoV) per env, within +/-focal_jitter.
    Only affects the rendered FoV (the student's visual input) -> robust to real camera focal mismatch.
    focalLength is a per-prim USD attr (UsdGeom.Camera), so unlike pose it persists. Fully guarded."""
    try:
        import random
        import omni.usd
        from pxr import UsdGeom
        stage = omni.usd.get_context().get_stage()
        ids = env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)
        for i in ids:
            ii = int(i)
            for path, focal in ((f"/World/envs/env_{ii}/front_camera", FRONT_FOCAL),
                                (f"/World/envs/env_{ii}/Robot/gripper_link/fp_camera", WRIST_FOCAL)):
                p = stage.GetPrimAtPath(path)
                if not p or not p.IsValid():
                    continue
                UsdGeom.Camera(p).GetFocalLengthAttr().Set(focal * (1.0 + random.uniform(-focal_jitter, focal_jitter)))
    except Exception:
        pass


def randomize_light(env, env_ids, intensity_range=(700.0, 3500.0), color_lo=0.65, color_hi=1.0):
    """(DR reset event) randomize each resetting env's scene light intensity + color (UsdLux).
    Only affects rendered rgb (the student's visual input), not physics/state — teacher unaffected.
    Fully guarded: any failure is a no-op so it can never crash training."""
    try:
        import random
        import omni.usd
        from pxr import UsdLux, Gf
        stage = omni.usd.get_context().get_stage()
        ids = env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)
        for i in ids:
            p = stage.GetPrimAtPath(f"/World/envs/env_{int(i)}/Light")
            if not p or not p.IsValid():
                continue
            api = UsdLux.LightAPI(p)
            api.GetIntensityAttr().Set(random.uniform(*intensity_range))
            api.GetColorAttr().Set(Gf.Vec3f(*(random.uniform(color_lo, color_hi) for _ in range(3))))
    except Exception:
        pass


def downsample(rgb, size):
    """(N,H,W,C>=3) uint8 -> (N,size,size,3) uint8 via area interpolation (SQUINT-style)."""
    import torch
    import torch.nn.functional as F
    x = rgb[..., :3].permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(size, size), mode="area")
    return x.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8)


def student_state(full_obs):
    """Slice the deployable 21D proprio state (qpos+target_qpos+dist_to_rest) from the full 43D obs.
    Used in DAgger where the env keeps the full privileged obs for the teacher."""
    return full_obs[:, STUDENT_STATE_IDX]


# ── visual domain randomization (DR) — train-only ─────────────────────────────
def tag_gripper(env, body_names=("gripper_link", "gripper_left", "gripper_right")):
    """(DR setup) apply a separate 'gripper' semantic class to the gripper body prims (post-build) so the
    seg distinguishes the gripper from the rest of the robot -> recolor_seg_region(("gripper",)) can paint
    it BLACK while the arm ('robot') is left as rendered. Fully guarded (no-op on failure)."""
    try:
        import omni.usd
        from pxr import Semantics
        stage = omni.usd.get_context().get_stage()
        ntag = 0
        for i in range(env.num_envs):
            for bn in body_names:
                p = stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot/{bn}")
                if not p or not p.IsValid():
                    continue
                sem = Semantics.SemanticsAPI.Apply(p, "Semantics")
                sem.CreateSemanticTypeAttr().Set("class")
                sem.CreateSemanticDataAttr().Set("gripper")
                ntag += 1
        print(f"[DAG] tag_gripper: tagged {ntag} gripper prims as 'gripper'", flush=True)
    except Exception as e:
        print(f"[DAG] tag_gripper failed ({e}) -> gripper stays 'robot'", flush=True)


def build_fg_ids(info, fg_classes=("robot", "cube", "table", "gripper")):
    """Foreground semantic ids (robot+cube) from a camera's segmentation info dict. Rebuilt each
    frame because replicator ids are not stable across resets. Returns a list[int] (empty if none)."""
    id2l = (info or {}).get("idToLabels", {}) if isinstance(info, dict) else {}
    out = []
    for k, v in id2l.items():
        cls = v.get("class", "") if isinstance(v, dict) else str(v)
        if any(c in cls for c in fg_classes):
            try:
                out.append(int(k))
            except (ValueError, TypeError):
                pass
    return out


def composite_bg(rgb, seg, info, bg, keep_lower=0.0):
    """Greenscreen: keep foreground (robot+cube) pixels — PLUS the lower `keep_lower` fraction of the image
    (the rendered TABLE surface, with a mild per-env size jitter) so the cube/arm sit on a REAL surface and
    don't FLOAT on the background — and replace only the rest (the far background = upper non-foreground)
    with `bg`. rgb (N,H,W,>=3) uint8, seg (N,H,W,1) int, info seg-info dict, bg (N,H,W,3). -> (N,H,W,3) uint8.
    Returns rgb unchanged if no foreground id resolves (seg not ready)."""
    import torch
    fg = build_fg_ids(info)
    rgb3 = rgb[..., :3]
    if not fg:
        return rgb3
    N, H, W = rgb3.shape[0], rgb3.shape[1], rgb3.shape[2]
    sids = seg[..., 0]
    fg_t = torch.as_tensor(fg, device=sids.device, dtype=sids.dtype)
    keep = torch.isin(sids, fg_t)                             # (N,H,W) True=foreground
    if keep_lower > 0.0:
        rows = torch.linspace(0.0, 1.0, H, device=rgb3.device).view(1, H, 1)
        lo = (1.0 - keep_lower) + torch.empty(N, 1, 1, device=rgb3.device).uniform_(-0.04, 0.04)
        keep = keep | (rows >= lo)                            # + lower table region (mild per-env size)
    out = torch.where(keep.unsqueeze(-1), rgb3.float(), bg[..., :3].float())
    return out.round().clamp(0, 255).to(torch.uint8)


def recolor_seg_region(rgb, seg, info, classes=("table",), b_lo=0.50, b_hi=1.05, per_ch=0.18, grain=True, base=0.0):
    """(DR) recolor (+ optional procedural RETEXTURE) ONLY the pixels of `classes` (by semantic silhouette),
    e.g. ("table",) or ("robot",). Keeps the rendered luminance/shading (lum) so shape/perspective are
    preserved (the table/robot stays grounded, no floating), then applies a per-env color tint (+ grain for
    surface texture). `base` lifts the luminance floor (base + (1-base)*lum) so a WHITE surface stays bright
    (e.g. base~0.62 + per_ch=0 = a fixed WHITE table with subtle shading). `per_ch=0` = no color, just
    brightness. NOTE: cubes are deliberately NOT recolored (cube color = the task cue). -> (N,H,W,3) uint8."""
    import torch
    import torch.nn.functional as F
    x = rgb[..., :3].float()
    ids = build_fg_ids(info, fg_classes=classes)
    if not ids:
        return x.round().clamp(0, 255).to(torch.uint8)
    N, H, W = rgb.shape[0], rgb.shape[1], rgb.shape[2]
    dev = rgb.device
    sids = seg[..., 0]
    id_t = torch.as_tensor(ids, device=dev, dtype=sids.dtype)
    mask = torch.isin(sids, id_t).unsqueeze(-1).float()                        # (N,H,W,1) target pixels
    lum = x.mean(-1, keepdim=True) / 255.0                                     # keep shading/perspective
    lum = base + (1.0 - base) * lum                                            # lift floor (white -> bright)
    b = torch.empty(N, 1, 1, 1, device=dev).uniform_(b_lo, b_hi)              # per-env brightness
    tint = b * (1.0 + per_ch * (torch.rand(N, 1, 1, 3, device=dev) - 0.5))    # per-env color (per_ch=0 -> none)
    new = lum * tint
    if grain:
        f = torch.randint(4, 16, (1,)).item()                                 # per-env grain frequency = TEXTURE
        g = torch.rand(N, f, f, 1, device=dev)
        g = F.interpolate(g.permute(0, 3, 1, 2), size=(H, W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        new = new * (0.78 + 0.44 * g)
    new = (new * 255.0).clamp(0, 255)
    out = mask * new + (1.0 - mask) * x
    return out.round().clamp(0, 255).to(torch.uint8)


def wrist_aug(img, roll, tx, ty):
    """(DR, wrist only) per-env roll + translate augmentation so the student is robust to the real
    wrist-cam mount being rotated/offset vs sim (real rig was 90deg off; 128px eye-in-hand is very
    roll/pos sensitive). img (N,H,W,3) uint8; roll (N,) rad; tx,ty (N,) normalized (fraction of half-
    extent). Applied per-EPISODE (angles redrawn on reset) via a single affine grid_sample."""
    import torch
    import torch.nn.functional as F
    N, H, W, _ = img.shape
    x = img.permute(0, 3, 1, 2).float()
    cos, sin = torch.cos(roll), torch.sin(roll)
    theta = torch.zeros(N, 2, 3, device=img.device, dtype=torch.float32)
    theta[:, 0, 0] = cos;  theta[:, 0, 1] = -sin; theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin;  theta[:, 1, 1] = cos;  theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8)


def image_dr(img, glare=True, scale=True):
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
    dev = img.device
    N, H, W, _ = img.shape
    x = img[..., :3].float() / 255.0
    # ISP color: gain (WB) + gamma + contrast + brightness
    x = x * torch.empty(N, 1, 1, 3, device=dev).uniform_(0.92, 1.10)   # WB gain (mild -> keep cube colors readable)
    x = x.clamp(1e-4, 4.0).pow(torch.empty(N, 1, 1, 1, device=dev).uniform_(0.85, 1.18))
    m = x.mean(dim=(1, 2), keepdim=True)
    x = (x - m) * torch.empty(N, 1, 1, 1, device=dev).uniform_(0.90, 1.12) + m
    x = (x + torch.empty(N, 1, 1, 1, device=dev).uniform_(-0.05, 0.05)).clamp(0.0, 1.0)
    # specular-glare proxy: per-env elliptical bright hotspot toward white
    if glare:
        yy = torch.linspace(0, 1, H, device=dev).view(1, H, 1)
        xx = torch.linspace(0, 1, W, device=dev).view(1, 1, W)
        cx = torch.empty(N, 1, 1, device=dev).uniform_(0.1, 0.9)
        cy = torch.empty(N, 1, 1, device=dev).uniform_(0.05, 0.6)
        ang = torch.empty(N, 1, 1, device=dev).uniform_(0.0, 3.14159)
        sx = torch.empty(N, 1, 1, device=dev).uniform_(0.04, 0.30)
        sy = torch.empty(N, 1, 1, device=dev).uniform_(0.02, 0.10)
        alpha = torch.empty(N, 1, 1, device=dev).uniform_(0.0, 0.50)   # milder gloss (don't blow out cubes)
        dx = xx - cx; dy = yy - cy
        ca = torch.cos(ang); sa = torch.sin(ang)
        u = (dx * ca + dy * sa) / sx; v = (-dx * sa + dy * ca) / sy
        g = (alpha * torch.exp(-(u * u + v * v))).unsqueeze(-1)
        x = (x + g * (1.0 - x)).clamp(0.0, 1.0)
    # cube apparent-size + aspect/anisotropy: per-env independent x/y scale + translate
    if scale:
        xt = x.permute(0, 3, 1, 2)
        theta = torch.zeros(N, 2, 3, device=dev)
        theta[:, 0, 0] = torch.empty(N, device=dev).uniform_(0.80, 1.25)  # scene/table apparent-size variation
        theta[:, 0, 2] = torch.empty(N, device=dev).uniform_(-0.07, 0.07)
        theta[:, 1, 1] = torch.empty(N, device=dev).uniform_(0.80, 1.25)  # independent -> slight aspect
        theta[:, 1, 2] = torch.empty(N, device=dev).uniform_(-0.07, 0.07)
        grid = F.affine_grid(theta, xt.shape, align_corners=False)
        x = F.grid_sample(xt, grid, mode="bilinear", padding_mode="reflection", align_corners=False).permute(0, 2, 3, 1)
    return (x * 255.0).round().clamp(0, 255).to(torch.uint8)


def add_proprio_noise(s21, arm_std=0.0122, grip_std=0.001, tgt_std=0.0122):
    """Add encoder-style gaussian noise to the 21D student state (TRAIN-ONLY, student path).
    Layout (STUDENT_STATE_IDX): qpos[0:8] (arm 0:6 + gripper 6:8) | target_qpos[8:15] | dist_to_rest[15:21].
    dist_to_rest = arm_qpos - REST, so the SAME per-joint draw added to arm qpos MUST be added to
    dist_to_rest (else a physically impossible (qpos,dist) pair). arm_std 0.0122 rad ~= 0.7deg."""
    import torch
    s = s21.clone()
    n = s.shape[0]
    arm = torch.randn(n, 6, device=s.device) * arm_std        # shared draw (qpos arm == dist_to_rest base)
    s[:, 0:6] += arm
    s[:, 15:21] += arm
    s[:, 6:8] += torch.randn(n, 2, device=s.device) * grip_std     # gripper qpos
    s[:, 8:15] += torch.randn(n, 7, device=s.device) * tgt_std     # target_qpos (controller cmd, independent)
    return s
