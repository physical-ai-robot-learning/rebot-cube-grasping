"""Vectorized, state-based reBot red-cube pick environment (Isaac Sim 5.1, no rendering).

v1 teacher env per docs/01-03: N parallel envs via GridCloner, batched Articulation +
RigidPrim views, joint delta-on-target action with hard limits, 43-D (50-D with obs_torque)
privileged state obs, staged pick reward (compute_pick_reward + sustained-success bonus), async per-env
reset. Physics steps with render=False (perception-to-state ⇒ no camera render during
training ⇒ thousands of cheap steps).

The caller MUST create ``SimulationApp`` (headless) BEFORE importing/instantiating this.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

from ..rl.rewards import compute_pick_reward
from ..scene.cameras import look_at_quat   # pure-numpy helper (isaac imported lazily inside cameras.py)
from .. import perception as P             # pure-numpy HSV red detection (sim render + real, same code)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vec v[N,3] by quaternion q[N,4] (w,x,y,z)."""
    w = q[:, 0:1]
    xyz = q[:, 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    """[N] yaw -> [N,4] wxyz."""
    h = 0.5 * yaw
    z = torch.zeros_like(yaw)
    return torch.stack([torch.cos(h), z, z, torch.sin(h)], dim=-1)


def project_cube_to_wrist(cube, glink_pos, glink_quat, mount, aim, fx, fy, cx, cy):
    """Project the cube center -> the eye-in-hand WRIST cam pixel (u,v) analytically (NO rendering),
    so the obs_wrist learned-IBVS feature is cheap in the state-based env. The wrist cam is at `mount`
    in the gripper_link frame, aimed at `aim` (same geometry as scene/cameras.make_wrist_camera): USD
    cam optical axis = local -Z. Returns (u, v, depth) [N]; depth>0 = in front of the cam.
    All inputs in ONE frame (env-local). Sign convention VALIDATED vs the rendered centroid."""
    N = cube.shape[0]
    cam = glink_pos + quat_rotate(glink_quat, mount.expand(N, 3))
    aim_w = glink_pos + quat_rotate(glink_quat, aim.expand(N, 3))
    fwd = aim_w - cam
    fwd = fwd / fwd.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    up = torch.tensor([0.0, 0.0, 1.0], device=cube.device).expand(N, 3)
    z = -fwd                                                   # USD cam looks down its local -Z
    x = torch.cross(up, z, dim=-1); x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    y = torch.cross(z, x, dim=-1)
    rel = cube - cam
    xc = (rel * x).sum(-1); yc = (rel * y).sum(-1); zc = (rel * z).sum(-1)
    depth = (-zc).clamp_min(1e-6)                              # in front => zc<0 => depth>0
    u = cx + fx * (xc / depth)
    v = cy + fy * (yc / depth)                                 # sign VALIDATED vs rendered centroid
    return u, v, -zc


class RebotPickEnv:
    ACT_DIM = 7                      # 6 arm-joint deltas + 1 gripper delta; obs_dim is computed in __init__

    def __init__(self, cfg, rcfg):
        self.cfg = cfg
        self.rcfg = rcfg
        self.device = torch.device(cfg.device)
        self.num_envs = cfg.num_envs
        self.act_dim = self.ACT_DIM

        from isaacsim.core.api import World
        from isaacsim.core.cloner import GridCloner
        from isaacsim.core.prims import Articulation, RigidPrim, XFormPrim
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
        from isaacsim.core.utils.prims import define_prim
        from pxr import UsdGeom, Gf, UsdShade, UsdPhysics

        self.world = World(physics_dt=cfg.physics_dt, rendering_dt=cfg.physics_dt,
                           stage_units_in_meters=1.0, backend="torch", device=cfg.device)
        # GPU broadphase buffers must scale with env count, else PhysX drops collision
        # pairs ("simulation will miss interactions") -> unreliable cube/gripper contacts.
        try:
            pc = self.world.get_physics_context()
            cap = max(2 ** 20, 8192 * cfg.num_envs)
            for setter, val in (
                ("set_gpu_found_lost_aggregate_pairs_capacity", cap),
                ("set_gpu_total_aggregate_pairs_capacity", cap),
                ("set_gpu_found_lost_pairs_capacity", cap),
                ("set_gpu_max_rigid_contact_count", 2 ** 23),
                ("set_gpu_max_rigid_patch_count", 2 ** 23),
            ):
                fn = getattr(pc, setter, None)
                if fn is not None:
                    try:
                        fn(int(val))
                    except Exception as e:  # noqa: BLE001
                        print(f"[env] {setter} skipped: {e}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[env] physics GPU capacity config skipped: {e}", flush=True)
        stage = get_current_stage()
        ground_z = cfg.table_z - cfg.table_size[2]
        self.world.scene.add_default_ground_plane(z_position=ground_z)

        # ---- base env template at /World/envs/env_0 ----
        cloner = GridCloner(spacing=cfg.env_spacing)
        cloner.define_base_env("/World/envs")
        define_prim("/World/envs/env_0", "Xform")

        robot_usd = os.path.abspath(cfg.robot_usd)
        if not os.path.exists(robot_usd):
            raise FileNotFoundError(f"robot USD not found: {robot_usd} (run convert_urdf_to_usd.py)")
        robot_path = "/World/envs/env_0/Robot"
        add_reference_to_stage(usd_path=robot_usd, prim_path=robot_path)
        # base pose (double precision to match importer-authored xformOps)
        rp = stage.GetPrimAtPath(robot_path)
        xf = UsdGeom.Xformable(rp)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*cfg.base_pos))
        w, x, y, z = cfg.base_quat_wxyz
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(w), float(x), float(y), float(z)))

        tpc = getattr(cfg, "table_prim_center", None) or cfg.table_center   # slab center (decoupled from cube spawn)
        FixedCuboid(
            prim_path="/World/envs/env_0/Table", name="table_0",
            position=np.array([tpc[0], tpc[1], cfg.table_z - cfg.table_size[2] / 2]),
            scale=np.array(cfg.table_size, dtype=float),
            color=np.array([0.95, 0.95, 0.95]),
        )
        rest_cube_z = cfg.table_z + cfg.cube_size / 2.0
        DynamicCuboid(
            prim_path="/World/envs/env_0/Cube", name="cube_0",
            position=np.array([0.25, -0.31, rest_cube_z]),   # red cube nominal spawn
            scale=np.array([cfg.cube_size] * 3, dtype=float),
            color=np.array([0.86, 0.12, 0.12]),
            mass=cfg.cube_mass,
        )

        # high friction on the cube + gripper fingers so a closed grip actually HOLDS the cube
        # through a lift (default material friction is too low -> cube slips, never lifts). Bound
        # on the env_0 template before cloning so all envs inherit it.
        try:
            mat_path = "/World/envs/env_0/grip_physmat"
            UsdShade.Material.Define(stage, mat_path)
            pm = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(mat_path))
            pm.CreateStaticFrictionAttr().Set(float(cfg.grip_friction))
            pm.CreateDynamicFrictionAttr().Set(float(cfg.grip_friction))
            pm.CreateRestitutionAttr().Set(0.0)
            grip_mat = UsdShade.Material(stage.GetPrimAtPath(mat_path))
            for p in ("/World/envs/env_0/Cube",
                      f"/World/envs/env_0/Robot/{cfg.wrist_link_name}",
                      "/World/envs/env_0/Robot/gripper_left",
                      "/World/envs/env_0/Robot/gripper_right"):
                prim = stage.GetPrimAtPath(p)
                if prim and prim.IsValid():
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                        grip_mat, bindingStrength=UsdShade.Tokens.weakerThanDescendants,
                        materialPurpose="physics")
        except Exception as e:  # noqa: BLE001
            print(f"[env] grip friction material skipped: {e}", flush=True)

        # ---- clone ----
        self.env_paths = cloner.generate_paths("/World/envs/env", cfg.num_envs)
        cloner.clone(source_prim_path="/World/envs/env_0", prim_paths=self.env_paths,
                     replicate_physics=True, base_env_path="/World/envs", copy_from_source=False)

        # env origins (grid offsets) read from the cloned prims
        xcache = UsdGeom.XformCache()
        origins = []
        for p in self.env_paths:
            t = xcache.GetLocalToWorldTransform(stage.GetPrimAtPath(p)).ExtractTranslation()
            origins.append([t[0], t[1], t[2]])
        self.env_origins = torch.tensor(origins, dtype=torch.float32, device=self.device)  # [N,3]

        # ---- batched views (created then initialized after play) ----
        self.robots = Articulation("/World/envs/env_*/Robot", name="robots")
        self.cubes = RigidPrim("/World/envs/env_*/Cube", name="cubes")
        tcp_expr = f"/World/envs/env_*/Robot/{cfg.wrist_link_name}"
        self.tcp_link = RigidPrim(tcp_expr, name="tcp")
        self.world.reset()
        for v in (self.robots, self.cubes):
            if not v.initialized:
                v.initialize()
        # gripper_link may not bind as a rigid body on every import -> fall back to XFormPrim
        try:
            if not self.tcp_link.initialized:
                self.tcp_link.initialize()
        except Exception as e:  # noqa: BLE001
            print(f"[env] tcp RigidPrim init failed ({e}); using XFormPrim for the wrist pose", flush=True)
            self.tcp_link = XFormPrim(tcp_expr, name="tcp_xf")
            self.tcp_link.initialize()
        # TCP (grasp center) = MIDPOINT of the two fingers, not a fixed wrist offset (that offset was
        # ~13cm wrong -> the reward aimed at empty space 13cm above the jaw). The finger midpoint is
        # the actual jaw center where the cube is held.
        self.finger_l = RigidPrim("/World/envs/env_*/Robot/gripper_left", name="finger_l")
        self.finger_r = RigidPrim("/World/envs/env_*/Robot/gripper_right", name="finger_r")
        for _fv in (self.finger_l, self.finger_r):
            try:
                if not _fv.initialized:
                    _fv.initialize()
            except Exception as e:  # noqa: BLE001
                print(f"[env] finger view init failed: {e}", flush=True)

        # ---- dof mapping (by name) ----
        names = list(self.robots.dof_names)
        self.arm_idx = torch.tensor([names.index(n) for n in cfg.arm_joint_names],
                                    dtype=torch.long, device=self.device)
        self.grip_idx = torch.tensor([i for i, n in enumerate(names) if cfg.gripper_joint_substr in n],
                                     dtype=torch.long, device=self.device)
        self.num_dof = self.robots.num_dof
        # obs layout depends on actual dof counts (gripper may import as 1 or 2 dofs):
        #   qpos + qvel = 2*(n_arm+n_grip); target = n_arm+1 (grip commanded as 1 scalar);
        #   dist_to_rest = n_arm; cube_pos3 + cube_quat4 + tcp3 + tcp_to_obj3 + is_grasped1 = 14
        n_arm, n_grip = len(self.arm_idx), len(self.grip_idx)
        # base = qpos+qvel + target + dist_to_rest. Then the cube/tcp block depends on the obs MODE:
        #   default (a): cube_pos3 + cube_quat4 + tcp3 + (cube-tcp)3 + is_grasped1 = 14
        #   keypoint (b, obs_top_uv): DROP privileged cube_pos/quat/relative -> tcp3 + is_grasped1 = 4
        #                              (+ the top (u,v,visible)=3 is appended below)
        self.obs_dim = 2 * (n_arm + n_grip) + (n_arm + 1) + n_arm + (4 if cfg.obs_top_uv else 14)
        if cfg.obs_top_uv:
            self.obs_dim += 3                   # top-cam (u_norm, v_norm, visible)
        # joint TORQUE in OBS (sim2real torque sync): + arm efforts (n_arm) + gripper effort (1)
        self._eff_ok = False
        if cfg.obs_torque:
            try:
                _e = self.robots.get_measured_joint_efforts()
                self._eff_ok = _e is not None
            except Exception as ex:  # noqa: BLE001
                print(f"[env] obs_torque: get_measured_joint_efforts unavailable ({ex}) -> torque obs = 0", flush=True)
            self.obs_dim += n_arm + 1
        if cfg.obs_wrist:                       # rendered wrist detection feature: (u_norm, v_norm, visible)
            self.obs_dim += 3
        print(f"[env] dof_names={names} arm_idx={self.arm_idx.tolist()} grip_idx={self.grip_idx.tolist()} "
              f"obs_dim={self.obs_dim} obs_torque={cfg.obs_torque}(eff_ok={self._eff_ok}) "
              f"obs_wrist={cfg.obs_wrist} obs_top_uv={cfg.obs_top_uv}", flush=True)

        # joint limits for arm clamping
        limits = self.robots.get_dof_limits().to(self.device)   # [N,dof,2] or [dof,2]
        if limits.dim() == 3:
            limits = limits[0]
        self.arm_lo = limits[self.arm_idx, 0]
        self.arm_hi = limits[self.arm_idx, 1]
        # optional tighter SOFT working range, intersected with the URDF hard limits, so random/
        # early-exploration actions stay in a sane region (rest_arm is always kept reachable).
        rest_t = torch.tensor(cfg.rest_arm, dtype=torch.float32, device=self.device)
        soft_lo = soft_hi = None
        if cfg.arm_limit_lo is not None and cfg.arm_limit_hi is not None:
            soft_lo = torch.tensor(cfg.arm_limit_lo, dtype=torch.float32, device=self.device)
            soft_hi = torch.tensor(cfg.arm_limit_hi, dtype=torch.float32, device=self.device)
        if soft_lo is not None:
            self.arm_lo = torch.maximum(self.arm_lo, soft_lo)
            self.arm_hi = torch.minimum(self.arm_hi, soft_hi)
            # keep rest pose inside the band, else reset would command an out-of-range target
            self.arm_lo = torch.minimum(self.arm_lo, rest_t)
            self.arm_hi = torch.maximum(self.arm_hi, rest_t)
            print(f"[env] arm soft limits (deg): lo={[round(x,1) for x in (self.arm_lo*57.29578).tolist()]} "
                  f"hi={[round(x,1) for x in (self.arm_hi*57.29578).tolist()]}", flush=True)

        try:
            kps = torch.full((self.num_envs, self.num_dof), cfg.drive_stiffness, device=self.device)
            kds = torch.full((self.num_envs, self.num_dof), cfg.drive_damping, device=self.device)
            # softer ARM drive -> wrist yields on table contact instead of tunneling through it;
            # gripper DOFs keep the stiff drive (firm grasp).
            kps[:, self.arm_idx] = cfg.arm_drive_stiffness
            kds[:, self.arm_idx] = cfg.arm_drive_damping
            kps[:, self.arm_idx[3:]] = cfg.wrist_drive_stiffness   # J4-J6 firm -> stable tcp (no wrist sag)
            kds[:, self.arm_idx[3:]] = cfg.wrist_drive_damping
            # DR: per-env arm stiffness scale (motor-gain / backlash gap). Fixed per env for the run.
            if cfg.dr_enable and cfg.dr_gain_scale > 0:
                gs = 1.0 + (torch.rand(self.num_envs, 1, device=self.device) * 2 - 1) * cfg.dr_gain_scale
                kps[:, self.arm_idx] = kps[:, self.arm_idx] * gs
                print(f"[env] DR arm gain scale +/-{cfg.dr_gain_scale}", flush=True)
            self.robots.set_gains(kps=kps, kds=kds)
            # RAISE the wrist max_effort (USD default 7 N*m is too weak -> wrist sags -> tcp swings)
            me = torch.as_tensor(self.robots.get_max_efforts(), device=self.device).reshape(-1, self.num_dof)
            if me.shape[0] == 1:
                me = me.repeat(self.num_envs, 1)
            me[:, self.arm_idx[3:]] = cfg.wrist_max_effort
            self.robots.set_max_efforts(me)
            print(f"[env] gains: arm kp={cfg.arm_drive_stiffness}, wrist kp={cfg.wrist_drive_stiffness} "
                  f"eff={cfg.wrist_max_effort}, grip kp={cfg.drive_stiffness}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[env] set_gains skipped: {e}", flush=True)
        try:    # more PhysX position iterations -> crisper table contact, less tunneling
            self.robots.set_solver_position_iteration_counts(
                torch.full((self.num_envs,), int(cfg.solver_pos_iters), dtype=torch.long, device=self.device))
            print(f"[env] solver_pos_iters={cfg.solver_pos_iters}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[env] solver-iter setup skipped: {e}", flush=True)

        # ---- buffers ----
        self.rest_arm = torch.tensor(cfg.rest_arm, dtype=torch.float32, device=self.device)
        self.arm_target = self.rest_arm.unsqueeze(0).repeat(self.num_envs, 1)  # [N,6]
        self.grip_target = torch.full((self.num_envs, 1), cfg.grip_open_qpos, device=self.device)
        self.prev_action = torch.zeros(self.num_envs, self.act_dim, device=self.device)
        self._ema_action = torch.zeros(self.num_envs, self.act_dim, device=self.device)  # anti-bob action low-pass state
        self.ep_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.grasp_hold = torch.zeros(self.num_envs, device=self.device)   # consecutive steps of genuine grasp+lift
        self.was_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.gate_near = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)      # gripper-close gate (near the cube?)
        self.grip_engaged = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)   # closed near the cube once -> stop force-opening (hold through lift)
        # ---- sim2real DR buffers (reflecting the real deploy) ----
        self.act_latency = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # per-env action delay (policy-steps)
        self._act_hist = None                                                                # [L+1,N,act_dim] ring buffer (lazy)
        self.tcp_off = torch.zeros(self.num_envs, 3, device=self.device)                     # per-env tcp OBS offset (calib robustness)
        self.torque_scale = torch.ones(self.num_envs, device=self.device)                    # per-env torque OBS scale (DR)
        self.pos_rand = float(cfg.pos_rand_xy)   # RUNTIME cube-spawn half-range; the spawn curriculum widens it
        self.ep_return = torch.zeros(self.num_envs, device=self.device)
        # episode bookkeeping returned via info
        self._last_ep_return = torch.zeros(self.num_envs, device=self.device)
        self._last_ep_len = torch.zeros(self.num_envs, device=self.device)
        self._last_ep_success = torch.zeros(self.num_envs, device=self.device)
        # extra validation bookkeeping (sticky within episode -> reported on done)
        self.ep_grasped = torch.zeros(self.num_envs, device=self.device)   # did it ever hold the cube
        self.ep_fell = torch.zeros(self.num_envs, device=self.device)      # did the cube ever fall/leave
        self._last_ep_grasped = torch.zeros(self.num_envs, device=self.device)
        self._last_ep_fell = torch.zeros(self.num_envs, device=self.device)
        self.ep_success = torch.zeros(self.num_envs, device=self.device)  # sticky within episode
        self._render = False                  # set True for GUI/inference (renders each physics step)
        self.wrist_cams = []
        if cfg.obs_wrist and not cfg.obs_analytic_uv:   # rendered wrist cams; SKIP when analytic (u,v) -> no render -> fast
            self._setup_wrist()
            self._render = True
        self.top_cams = []
        if cfg.obs_top_uv and not cfg.obs_analytic_uv:  # rendered overhead cams; SKIP when analytic (u,v)
            self._setup_top()
            self._render = True
        if cfg.obs_analytic_uv:
            print(f"[env] obs_analytic_uv: ANALYTIC (u,v) projection (no cameras, no render) -> fast", flush=True)

        self.reset()

    def set_render(self, on: bool):
        self._render = bool(on)

    # ------------------------------------------------------------------ helpers
    def _full_targets(self):
        tgt = self.robots.get_joint_positions().clone()
        tgt[:, self.arm_idx] = self.arm_target
        tgt[:, self.grip_idx] = self.grip_target  # broadcast [N,1] -> mirrored on both grip dofs
        return tgt

    def _measured_efforts(self):
        """PhysX measured joint efforts [N, num_dof] (the sim analogue of the real motor .torque).
        Zeros if the API is unavailable. Used only for the obs_torque sim2real torque sync."""
        if not getattr(self, "_eff_ok", False):
            return torch.zeros(self.num_envs, self.num_dof, device=self.device)
        e = self.robots.get_measured_joint_efforts()
        return torch.as_tensor(e, device=self.device).reshape(self.num_envs, self.num_dof)

    def _setup_wrist(self):
        """obs_wrist: one low-res wrist cam per env (parented to that env's gripper_link) + a dome light
        (the scene is otherwise unlit -> black renders). Mirrors scene/cameras.make_wrist_camera geometry."""
        from isaacsim.sensors.camera import Camera
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import UsdGeom, Gf, UsdLux
        from ..constants import C922
        from ..scene.cameras import _apply_c922
        c922 = C922()
        stage = get_current_stage()
        self._dome = UsdLux.DomeLight.Define(stage, "/World/wrist_dome")
        self._dome_base = float(self.cfg.wrist_light_intensity)
        self._dome.CreateIntensityAttr(self._dome_base)
        mount = Gf.Vec3d(*self.cfg.wrist_mount_offset); aim = Gf.Vec3d(*self.cfg.wrist_aim_local)
        res = (int(self.cfg.wrist_cam_res[0]), int(self.cfg.wrist_cam_res[1]))
        for p in self.env_paths:
            link = f"{p}/Robot/{self.cfg.wrist_link_name}"
            if not stage.GetPrimAtPath(link).IsValid():
                self.wrist_cams.append(None); continue
            path = link + "/wrist_cam"
            UsdGeom.Camera.Define(stage, path)
            xf = UsdGeom.Xformable(stage.GetPrimAtPath(path)); xf.ClearXformOpOrder()
            xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(mount)
            pw = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(link))
            cw = pw.Transform(mount); aw = pw.Transform(aim)
            w, x, y, z = look_at_quat((cw[0], cw[1], cw[2]), (aw[0], aw[1], aw[2]))
            ql = pw.ExtractRotation().GetQuat().GetInverse() * Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z)))
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(ql)
            cam = Camera(prim_path=path, name=f"wrist_{len(self.wrist_cams)}", resolution=res)
            cam.initialize()
            _apply_c922(cam, c922)              # focal/aperture/clipping (else default intrinsics -> wrong FOV)
            self.wrist_cams.append(cam)
        print(f"[env] obs_wrist: built {sum(c is not None for c in self.wrist_cams)} wrist cams {res}", flush=True)

    def _setup_top(self):
        """obs_top_uv: one overhead cam per env at the real-aligned top_cam pose (env_origin + offset),
        looking down at the workspace (FIXED world pose, NOT link-parented). Rendered + HSV-detected like
        the wrist -> the policy's top (u,v) token shares the deploy CNN's domain (red centroid in the top
        image) AND is occlusion-aware (the arm over the cube zeroes the detection, unlike a projection)."""
        from isaacsim.sensors.camera import Camera
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import UsdLux
        from ..constants import C922
        from ..scene.cameras import _apply_c922
        c922 = C922()
        stage = get_current_stage()
        if getattr(self, "_dome", None) is None:        # ensure the scene is lit (else black renders)
            self._dome = UsdLux.DomeLight.Define(stage, "/World/top_dome")
            self._dome_base = float(self.cfg.wrist_light_intensity)
            self._dome.CreateIntensityAttr(self._dome_base)
        res = (int(self.cfg.top_cam_res[0]), int(self.cfg.top_cam_res[1]))
        tcc = self.cfg.table_center
        pos_l = np.array([tcc[0], self.cfg.top_cam_y, self.cfg.table_z + self.cfg.top_cam_height])  # cam y = MEASURED abs (= deploy/SceneConfig), NOT tcc[1]+back_dist (that anchored to the old table_center)
        aim_l = np.array([tcc[0], self.cfg.top_cam_aim_y, self.cfg.table_z + 0.03])
        for i, p in enumerate(self.env_paths):
            org = self.env_origins[i].cpu().numpy()
            cam = Camera(prim_path=f"{p}/top_cam", name=f"top_{i}", resolution=res)
            cam.initialize(); _apply_c922(cam, c922)
            cpos = org + pos_l; caim = org + aim_l
            cam.set_world_pose(position=cpos, orientation=look_at_quat(cpos, caim), camera_axes="usd")
            self.top_cams.append(cam)
        print(f"[env] obs_top_uv: built {len(self.top_cams)} top cams {res}", flush=True)

    def _top_obs(self):
        """project (b): render each env's overhead cam, batched HSV-detect the red cube -> [N,3]
        (u_norm, v_norm, visible). SAME detector domain as the deploy CNN (red centroid in the top image)
        and OCCLUSION-AWARE (arm over the cube -> no red -> visible 0). DR: (u,v) noise + dropout; not-
        visible -> sentinel 0. Mirrors _wrist_obs (sim render + real BOTH read the top image the same way)."""
        if self.cfg.obs_analytic_uv:                                    # ANALYTIC (u,v): no render; DR (dropout) approximates occlusion
            t = self._a_top.clone()
            if self.cfg.dr_enable:
                if self.cfg.dr_top_uv_noise > 0:
                    t[:, :2] += torch.randn(self.num_envs, 2, device=self.device) * self.cfg.dr_top_uv_noise
                if self.cfg.dr_top_uv_dropout > 0:
                    t[(torch.rand(self.num_envs, device=self.device) < self.cfg.dr_top_uv_dropout)] = 0.0
            t[t[:, 2] < 0.5] = 0.0
            return t
        W, Hh = int(self.cfg.top_cam_res[0]), int(self.cfg.top_cam_res[1])
        frames = np.zeros((self.num_envs, Hh, W, 3), dtype=np.float32)
        ok = np.zeros(self.num_envs, dtype=bool)
        for i, cam in enumerate(self.top_cams):
            if cam is None:
                continue
            rgba = cam.get_rgba()
            if rgba is None or getattr(rgba, "size", 0) == 0:
                continue
            a = np.asarray(rgba)
            if a.shape[0] == Hh and a.shape[1] == W:
                frames[i] = a[..., :3].astype(np.float32); ok[i] = True
        mask = P.red_mask(frames)                                       # [N,Hh,W] bool
        # DOMINANT-blob centroid per env (= the deploy CNN/HSV estimator P._largest_blob_centroid), NOT the
        # area-weighted mean of ALL red: the mean gets dragged by stray red (real bg clutter, OOD), so train
        # and deploy must use the SAME robust estimator or the (u,v) token diverges off-center.
        un = np.zeros(self.num_envs, dtype=np.float32)
        vn = np.zeros(self.num_envs, dtype=np.float32)
        det = np.zeros(self.num_envs, dtype=bool)
        for i in range(self.num_envs):
            if not ok[i]:
                continue
            c = P._largest_blob_centroid(mask[i])
            if c is None:                                               # <15 red px -> not visible
                continue
            cu, cv, _ = c
            un[i] = (cu - W / 2.0) / (W / 2.0); vn[i] = (cv - Hh / 2.0) / (Hh / 2.0)
            det[i] = True
        feat = np.stack([un, vn, det.astype(np.float32)], axis=-1)
        t = torch.tensor(feat, dtype=torch.float32, device=self.device)
        if self.cfg.dr_enable:
            if self.cfg.dr_top_uv_noise > 0:
                t[:, :2] += torch.randn(self.num_envs, 2, device=self.device) * self.cfg.dr_top_uv_noise
            if self.cfg.dr_top_uv_dropout > 0:
                t[(torch.rand(self.num_envs, device=self.device) < self.cfg.dr_top_uv_dropout)] = 0.0
        t[t[:, 2] < 0.5] = 0.0                                          # not-visible -> sentinel 0
        return t

    def _wrist_obs(self):
        """Grab each env's wrist frame, batched HSV-detect the red cube -> [N,3] (u_norm, v_norm, visible).
        Area-weighted centroid (the cube is the only red in sim). DR: (u,v) noise + random dropout; the
        not-visible feature is zeroed (sentinel). Sim render + real deploy run the SAME detector."""
        if self.cfg.obs_analytic_uv:                                    # ANALYTIC (u,v): no render; DR matches render/deploy
            t = self._a_wrist.clone()
            if self.cfg.dr_enable:
                if self.cfg.dr_wrist_distractor > 0:                    # FALSE detection (analytic bg-clutter proxy): lock onto a wrong (u,v)
                    fd = torch.rand(self.num_envs, device=self.device) < self.cfg.dr_wrist_distractor
                    nfd = int(fd.sum())
                    if nfd > 0:
                        t[fd, 0] = torch.rand(nfd, device=self.device) * 2 - 1
                        t[fd, 1] = torch.rand(nfd, device=self.device) * 2 - 1
                        t[fd, 2] = 1.0
                if self.cfg.dr_wrist_uv_noise > 0:
                    t[:, :2] += torch.randn(self.num_envs, 2, device=self.device) * self.cfg.dr_wrist_uv_noise
                if self.cfg.dr_wrist_dropout > 0:
                    t[(torch.rand(self.num_envs, device=self.device) < self.cfg.dr_wrist_dropout)] = 0.0
            t[t[:, 2] < 0.5] = 0.0
            return t
        W, Hh = int(self.cfg.wrist_cam_res[0]), int(self.cfg.wrist_cam_res[1])
        frames = np.zeros((self.num_envs, Hh, W, 3), dtype=np.float32)
        ok = np.zeros(self.num_envs, dtype=bool)
        for i, cam in enumerate(self.wrist_cams):
            if cam is None:
                continue
            rgba = cam.get_rgba()
            if rgba is None or getattr(rgba, "size", 0) == 0:
                continue
            a = np.asarray(rgba)
            if a.shape[0] == Hh and a.shape[1] == W:
                frames[i] = a[..., :3].astype(np.float32); ok[i] = True
        # WRIST CLUTTER DR: paint random red distractor blobs into the frame BEFORE detection, so the SAME
        # detector sometimes locks onto background red (the real sim2real failure) -> the policy learns to
        # NOT over-trust the wrist feature (and fall back to state via dropout).
        if self.cfg.dr_enable and self.cfg.dr_wrist_distractor > 0.0:
            for i in range(self.num_envs):
                if not ok[i] or np.random.rand() >= self.cfg.dr_wrist_distractor:
                    continue
                for _ in range(np.random.randint(1, int(self.cfg.dr_wrist_distractor_n) + 1)):
                    bw = np.random.randint(max(2, W // 20), max(3, W // 5))
                    bh = np.random.randint(max(2, Hh // 12), max(3, Hh // 4))
                    x0 = np.random.randint(0, max(1, W - bw)); y0 = np.random.randint(0, max(1, Hh - bh))
                    frames[i, y0:y0 + bh, x0:x0 + bw] = (200.0, 30.0, 30.0)   # saturated red distractor
        mask = P.red_mask(frames)                                       # [N,Hh,W] bool (batched, 0..255 ok)
        xs = np.arange(W, dtype=np.float32); ys = np.arange(Hh, dtype=np.float32)
        area = mask.sum(axis=(1, 2)).astype(np.float32)                 # [N]
        det = ok & (area > 8)
        a1 = np.maximum(area, 1.0)
        u = (mask * xs[None, None, :]).sum(axis=(1, 2)) / a1
        vv = (mask * ys[None, :, None]).sum(axis=(1, 2)) / a1
        un = (u - W / 2.0) / (W / 2.0); vn = (vv - Hh / 2.0) / (Hh / 2.0)
        feat = np.stack([un, vn, det.astype(np.float32)], axis=-1)      # [N,3] normalized
        t = torch.tensor(feat, dtype=torch.float32, device=self.device)
        if self.cfg.dr_enable:
            if self.cfg.dr_wrist_uv_noise > 0:
                t[:, :2] += torch.randn(self.num_envs, 2, device=self.device) * self.cfg.dr_wrist_uv_noise
            if self.cfg.dr_wrist_dropout > 0:
                t[(torch.rand(self.num_envs, device=self.device) < self.cfg.dr_wrist_dropout)] = 0.0
        t[t[:, 2] < 0.5] = 0.0                                          # not-visible -> sentinel 0
        return t

    def _compute_state(self, action):
        qpos = self.robots.get_joint_positions()
        qvel = self.robots.get_joint_velocities()
        arm_qpos = qpos[:, self.arm_idx]
        arm_qvel = qvel[:, self.arm_idx]
        grip_qpos = qpos[:, self.grip_idx]
        grip_qvel = qvel[:, self.grip_idx]

        cube_pos_w, cube_quat = self.cubes.get_world_poses()
        cube_pos = cube_pos_w - self.env_origins
        # TCP = grasp center = midpoint of the two fingers (the actual jaw), minus a small drop to the
        # fingertip/grip-contact level. (The old fixed wrist offset put it ~13cm above the jaw.)
        lpw, _ = self.finger_l.get_world_poses()
        rpw, _ = self.finger_r.get_world_poses()
        tcp_pos = 0.5 * (lpw + rpw) - self.env_origins
        tcp_pos[:, 2] = tcp_pos[:, 2] - self.cfg.tcp_tip_drop

        # GRIPPER FRAME TILT from straight-down. Frame-independent: of the wrist frame's 3 local axes,
        # take the one whose world direction is closest to vertical (= the tool/approach axis in this
        # near-upright regime) and measure its deviation from vertical. Validated vs the monitor images
        # (clean top-down grasp ~16 deg; the ~41-deg edge grasp the policy used catches the cube's top
        # corner). Drives the verticality reward + gates is_grasped (scripts/probe_align.py).
        wp, wq = self.tcp_link.get_world_poses()                 # world pos + quat (wxyz)
        zx = 2.0 * (wq[:, 1] * wq[:, 3] - wq[:, 0] * wq[:, 2])    # local-x . worldZ
        zy = 2.0 * (wq[:, 2] * wq[:, 3] + wq[:, 0] * wq[:, 1])    # local-y . worldZ
        zz = 1.0 - 2.0 * (wq[:, 1] * wq[:, 1] + wq[:, 2] * wq[:, 2])  # local-z . worldZ
        vert = torch.maximum(torch.maximum(zx.abs(), zy.abs()), zz.abs()).clamp(max=1.0)
        tilt_deg = torch.rad2deg(torch.acos(vert))               # 0 = perfectly vertical

        # analytic CLEAN wrist (u,v) of the cube (no render) -> the wrist-centering reward (policy learns to
        # aim the wrist cam at the cube during the approach/descend). Same geometry as the wrist cam.
        if not hasattr(self, "_wrist_K"):
            from ..constants import C922
            _c = C922(); _Ww, _Hw = float(self.cfg.wrist_cam_res[0]), float(self.cfg.wrist_cam_res[1])
            self._wrist_K = (_c.focal_length_mm * _Ww / _c.horizontal_aperture_mm,
                             _c.focal_length_mm * _Hw / _c.vertical_aperture_mm, _Ww / 2.0, _Hw / 2.0)
            self._wrist_mount = torch.tensor(self.cfg.wrist_mount_offset, device=self.device)
            self._wrist_aim = torch.tensor(self.cfg.wrist_aim_local, device=self.device)
        _fx, _fy, _cx, _cy = self._wrist_K
        _wu, _wv, _wdepth = project_cube_to_wrist(cube_pos, wp - self.env_origins, wq,
                                                  self._wrist_mount, self._wrist_aim, _fx, _fy, _cx, _cy)
        wrist_un = (_wu - _cx) / _cx
        wrist_vn = (_wv - _cy) / _cy
        wrist_vis = ((_wdepth > 0.01) & (wrist_un.abs() < 1.0) & (wrist_vn.abs() < 1.0)).float()
        self._a_wrist = torch.stack([wrist_un, wrist_vn, wrist_vis], dim=-1)   # CLEAN analytic wrist (u,v,vis) for obs/reward
        # analytic TOP (u,v): world-fixed overhead cam, projected with NO render (obs_analytic_uv -> ~10x fps)
        if self.cfg.obs_top_uv:
            if not hasattr(self, "_top_cam"):
                from ..constants import C922
                _tc = C922(); _Wt, _Ht = float(self.cfg.top_cam_res[0]), float(self.cfg.top_cam_res[1])
                _co = torch.tensor([self.cfg.table_center[0], self.cfg.top_cam_y,
                                    self.cfg.table_z + self.cfg.top_cam_height], device=self.device)
                _ao = torch.tensor([self.cfg.table_center[0], self.cfg.top_cam_aim_y,
                                    self.cfg.table_z + 0.03], device=self.device)
                _fwd = _ao - _co; _fwd = _fwd / _fwd.norm()
                _up = torch.tensor([0.0, 0.0, 1.0], device=self.device)
                _zc = -_fwd; _xc = torch.cross(_up, _zc, dim=-1); _xc = _xc / _xc.norm(); _yc = torch.cross(_zc, _xc, dim=-1)
                self._top_cam = (_co, _xc, _yc, _zc, _Wt, _Ht,
                                 _tc.focal_length_mm * _Wt / _tc.horizontal_aperture_mm,
                                 _tc.focal_length_mm * _Ht / _tc.vertical_aperture_mm)
            _co, _xc, _yc, _zc, _Wt, _Ht, _tfx, _tfy = self._top_cam
            _rel = cube_pos - _co
            _xcc = (_rel * _xc).sum(-1); _ycc = (_rel * _yc).sum(-1); _zcc = (_rel * _zc).sum(-1)
            _dep = (-_zcc).clamp_min(1e-6)
            _tun = (_tfx * (_xcc / _dep)) / (_Wt / 2.0)
            _tvn = (_tfy * (_ycc / _dep)) / (_Ht / 2.0)
            _tvis = ((-_zcc > 0.01) & (_tun.abs() < 1.0) & (_tvn.abs() < 1.0)).float()
            self._a_top = torch.stack([_tun, _tvn, _tvis], dim=-1)
            # OCCLUSION proxy: the overhead cam loses the cube when the gripper is over+above it (arm
            # silhouette covers it) -> top visible 0. Random dropout is UNcorrelated; this is the CORRELATED
            # occlusion (reliably during descend) the policy must learn to ride out, matching render/deploy.
            if self.cfg.top_occlusion_radius > 0:
                _occ = (torch.linalg.norm(tcp_pos[:, :2] - cube_pos[:, :2], dim=-1) < self.cfg.top_occlusion_radius) \
                       & (tcp_pos[:, 2] > cube_pos[:, 2])
                self._a_top[:, 2] = self._a_top[:, 2] * (~_occ).float()

        dist3 = torch.linalg.norm(cube_pos - tcp_pos, dim=-1)
        dist_xy = torch.linalg.norm(cube_pos[:, :2] - tcp_pos[:, :2], dim=-1)
        dz = cube_pos[:, 2] - tcp_pos[:, 2]                       # >0 if the grasp-center is BELOW the cube
        rest_cube_z = self.cfg.table_z + self.cfg.cube_size / 2.0
        # GENUINE alignment: horizontally ON the cube AND grasp-center at the cube's height (NOT under
        # it). Plain 3-D proximity let tcp sit below the cube / fling it up -> fake grasps.
        near = (dist_xy < self.cfg.grasp_xy) & (dz.abs() < self.cfg.grasp_dz)
        # "closing" = commanded toward the pinch (LOW qpos), per the inverted grip convention
        closing = self.grip_target.squeeze(-1) <= 0.5 * (self.cfg.grip_open_qpos + self.cfg.grip_close_qpos)
        # GEOMETRIC grasp (user's criterion): TCP at the cube + the fingers stopped AT the cube width
        # (finger gap ~= cube edge => the cube is between the jaws). gap ~= 2*grip_qpos (probe_gap.py).
        finger_gap = 2.0 * grip_qpos.mean(dim=-1)
        on_cube = (finger_gap - self.cfg.grip_grasp_gap).abs() < self.cfg.grip_gap_tol
        lifted = cube_pos[:, 2] > rest_cube_z + 0.01
        self.gate_near = dist3 < self.cfg.grip_gate_dist     # gripper may close only within this of the cube
        # is_grasped EVERY step (no latch): the cube must be genuinely between the fingers -- a real
        # grip holds the gap at the cube width (on_cube stays True), while an air-close drops to ~20mm
        # (on_cube False). (A latch that only required "closing" let a 1-frame on_cube fake a hold.)
        # vertical_ok: the gripper must be near-vertical -> a ~41-deg edge/corner grasp is NOT a grasp.
        vertical_ok = tilt_deg < self.cfg.tilt_gate_deg
        is_grasped = near & closing & on_cube & vertical_ok
        held = near & lifted
        grasp_lift = is_grasped & lifted                         # gripper holding the cube off the table

        # failure: cube fell off table or left workspace (local frame)
        wx, wy = self.cfg.workspace_x, self.cfg.workspace_y
        out_xy = (cube_pos[:, 0] < wx[0] - 0.15) | (cube_pos[:, 0] > wx[1] + 0.15) | \
                 (cube_pos[:, 1] < wy[0] - 0.15) | (cube_pos[:, 1] > wy[1] + 0.15)
        fallen = (cube_pos[:, 2] < self.cfg.table_z - 0.05) | out_xy
        # SAFETY (table): use the LOWER of the two fingers, NOT the TCP midpoint. A tilted gripper can
        # dig ONE fingertip into the table while the midpoint stays above the table -> the old
        # midpoint-based rule missed single-finger digging (user-reported). table_pen = how far the
        # lower finger is below the table top (continuous -> gradient pushes it up); table_hit = clear
        # penetration -> terminate. A valid grasp keeps both fingers ~at the cube (>table), so unpunished.
        lower_finger_z = torch.minimum(lpw[:, 2], rpw[:, 2]) - self.env_origins[:, 2]
        tip_z = lower_finger_z - self.cfg.finger_tip_drop
        table_pen = torch.clamp(self.cfg.table_z - tip_z, min=0.0)
        table_hit = tip_z < (self.cfg.table_z - self.cfg.table_term_depth)
        # SAFETY (no backward tip-over): keep the gripper on the +y (table) side. y_back = how far
        # tcp is behind y_forward_limit (continuous -> gradient pushes it forward); y_back_hit =
        # gross backward swing -> terminate. base is at y~-0.64, table/workspace at y~-0.2..-0.4.
        y_back = torch.clamp(self.cfg.y_forward_limit - tcp_pos[:, 1], min=0.0)
        y_back_hit = tcp_pos[:, 1] < self.cfg.y_back_term

        # DR: the OBS carries proprio + perception noise AND a per-env tcp calibration offset; the
        # reward/gates always use the CLEAN state (asymmetric). Mirrors the real build_obs -- dist_to_rest
        # is from the (noisy) MEASURED qpos, and tcp_off mimics the JAW_FWD/FK_Z_CORR calibration residual.
        arm_qpos_o, arm_qvel_o = arm_qpos, arm_qvel
        cube_obs, tcp_obs = cube_pos, tcp_pos
        if self.cfg.dr_enable:
            if self.cfg.dr_obs_qpos_noise > 0:
                arm_qpos_o = arm_qpos + torch.randn_like(arm_qpos) * self.cfg.dr_obs_qpos_noise
            if self.cfg.dr_obs_qvel_noise > 0:
                arm_qvel_o = arm_qvel + torch.randn_like(arm_qvel) * self.cfg.dr_obs_qvel_noise
            ns = self.cfg.dr_obs_pos_noise
            if ns > 0:
                cube_obs = cube_pos + torch.randn_like(cube_pos) * ns
                tcp_obs = tcp_pos + torch.randn_like(tcp_pos) * ns
            tcp_obs = tcp_obs + self.tcp_off
        qpos8 = torch.cat([arm_qpos_o, grip_qpos], dim=-1)
        qvel8 = torch.cat([arm_qvel_o, grip_qvel], dim=-1)
        target7 = torch.cat([self.arm_target, self.grip_target], dim=-1)
        dist_to_rest = arm_qpos_o - self.rest_arm.unsqueeze(0)
        if self.cfg.obs_top_uv:
            # project (b): keypoint bottleneck -> NO privileged cube_pos3/quat4/(cube-tcp)3 (the top (u,v)
            # token, appended below, carries the cube location; the policy learns visuomotor servoing).
            obs = torch.cat([qpos8, qvel8, target7, dist_to_rest, tcp_obs,
                             is_grasped.float().unsqueeze(-1)], dim=-1)
        else:
            obs = torch.cat([
                qpos8, qvel8, target7, dist_to_rest,
                cube_obs, cube_quat, tcp_obs, (cube_obs - tcp_obs),
                is_grasped.float().unsqueeze(-1),
            ], dim=-1)
        # joint TORQUE in OBS (sim2real torque sync): measured arm efforts (n_arm) + gripper effort (1),
        # normalized + per-env scale/noise DR (bridges the sim<->real torque magnitude gap). Appended LAST.
        if self.cfg.obs_torque:
            eff = self._measured_efforts()
            tau = torch.cat([eff[:, self.arm_idx], eff[:, self.grip_idx].mean(dim=-1, keepdim=True)], dim=-1)
            tau = tau * self.cfg.torque_obs_scale
            if self.cfg.dr_enable:
                tau = tau * self.torque_scale.unsqueeze(-1)
                if self.cfg.dr_torque_noise > 0:
                    tau = tau + torch.randn_like(tau) * self.cfg.dr_torque_noise
            obs = torch.cat([obs, tau], dim=-1)
        if self.cfg.obs_top_uv:                 # project (b): rendered top-cam (u_norm, v_norm, visible)
            obs = torch.cat([obs, self._top_obs()], dim=-1)
        if self.cfg.obs_wrist:                  # rendered wrist detection -> (u_norm, v_norm, visible)
            obs = torch.cat([obs, self._wrist_obs()], dim=-1)

        state = {
            "tcp_pos": tcp_pos, "cube_pos": cube_pos, "is_grasped": is_grasped,
            "held": held, "fallen": fallen, "action": action, "prev_action": self.prev_action,
            "grip": self.grip_target.squeeze(-1),
            "table_pen": table_pen, "table_hit": table_hit,
            "y_back": y_back, "y_back_hit": y_back_hit,
            "grasp_lift": grasp_lift,
            "near": near, "closing": closing, "on_cube": on_cube, "dist_xy": dist_xy, "dz": dz,
            "tilt_deg": tilt_deg, "vertical_ok": vertical_ok,
            "wrist_un": wrist_un, "wrist_vn": wrist_vn, "wrist_visible": wrist_vis,
            "arm_speed": torch.linalg.norm(arm_qvel, dim=-1),   # CLEAN, for the anti-bob hold-still reward
        }
        return obs, state, fallen

    # ------------------------------------------------------------------ API
    def reset(self, env_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        n = env_ids.numel()
        if n == 0:
            return
        # robot -> rest pose. With rest_rand>0, JITTER the arm start per-joint so the gripper begins OFF
        # the cube -> the policy must learn to APPROACH-correct, not assume a perfect pre-position.
        if self.cfg.rest_rand > 0.0:
            jit = (torch.rand(n, len(self.arm_idx), device=self.device) * 2.0 - 1.0) * self.cfg.rest_rand
            arm_start = torch.clamp(self.rest_arm.unsqueeze(0) + jit, self.arm_lo, self.arm_hi)
        else:
            arm_start = self.rest_arm.unsqueeze(0).expand(n, -1)
        qpos = self.robots.get_joint_positions().clone()
        qpos[env_ids[:, None], self.arm_idx] = arm_start
        qpos[env_ids[:, None], self.grip_idx] = self.cfg.grip_open_qpos
        self.robots.set_joint_positions(qpos[env_ids], indices=env_ids)
        zero = torch.zeros(n, self.num_dof, device=self.device)
        self.robots.set_joint_velocities(zero, indices=env_ids)
        self.arm_target[env_ids] = arm_start   # drive holds the jittered start (else it snaps back to rest_arm)
        self.grip_target[env_ids] = self.cfg.grip_open_qpos
        self.robots.set_joint_position_targets(self._full_targets()[env_ids], indices=env_ids)

        # cube spawn: around the table center, +/- pos_rand_xy (curriculum knob: 0 = fixed center),
        # clamped to the workspace. Expand pos_rand_xy over training to cover the full workspace.
        wx, wy = self.cfg.workspace_x, self.cfg.workspace_y
        cxc, cyc = self.cfg.table_center
        r = self.pos_rand                        # runtime spawn half-range (widened by the curriculum)
        cx = (cxc + torch.empty(n, device=self.device).uniform_(-r, r)).clamp(wx[0], wx[1])
        cy = (cyc + torch.empty(n, device=self.device).uniform_(-r, r)).clamp(wy[0], wy[1])
        cz = torch.full((n,), self.cfg.table_z + self.cfg.cube_size / 2.0 + 0.002, device=self.device)
        pos_local = torch.stack([cx, cy, cz], dim=-1)
        pos_world = pos_local + self.env_origins[env_ids]
        yaw = torch.empty(n, device=self.device).uniform_(-self.cfg.yaw_rand, self.cfg.yaw_rand)
        self.cubes.set_world_poses(positions=pos_world, orientations=yaw_quat(yaw), indices=env_ids)
        self.cubes.set_velocities(torch.zeros(n, 6, device=self.device), indices=env_ids)

        # DR: randomize cube mass (dynamics gap)
        if self.cfg.dr_enable:
            masses = torch.empty(n, device=self.device).uniform_(self.cfg.dr_mass_min, self.cfg.dr_mass_max)
            try:
                self.cubes.set_masses(masses, indices=env_ids)
            except Exception as e:  # noqa: BLE001
                print(f"[env] set_masses skipped: {e}", flush=True)

        self.prev_action[env_ids] = 0.0
        self._ema_action[env_ids] = 0.0
        self.ep_step[env_ids] = 0
        self.grasp_hold[env_ids] = 0.0
        self.was_grasped[env_ids] = False
        self.grip_engaged[env_ids] = False
        self.ep_return[env_ids] = 0.0
        self.ep_success[env_ids] = 0.0
        self.ep_grasped[env_ids] = 0.0
        self.ep_fell[env_ids] = 0.0

        # DR: redraw per-env action latency + tcp OBS offset for the new episode
        if self.cfg.dr_enable:
            if self.cfg.dr_action_latency_max > 0:
                self.act_latency[env_ids] = torch.randint(
                    0, self.cfg.dr_action_latency_max + 1, (n,), device=self.device)
            ox, oz = self.cfg.dr_tcp_offset_xy, self.cfg.dr_tcp_offset_z
            if ox > 0 or oz > 0:
                u = torch.rand(n, 3, device=self.device) * 2.0 - 1.0
                self.tcp_off[env_ids] = u * torch.tensor([ox, ox, oz], device=self.device)
            if self.cfg.obs_torque and self.cfg.dr_torque_scale > 0:
                self.torque_scale[env_ids] = 1.0 + (torch.rand(n, device=self.device) * 2 - 1) * self.cfg.dr_torque_scale

        obs, _, _ = self._compute_state(self.prev_action)
        return obs

    def get_obs(self):
        obs, _, _ = self._compute_state(self.prev_action)
        return obs

    def _delayed_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """DR: per-env action delay (real CAN round-trip + control lag). Apply the action issued
        act_latency steps ago. A small delay makes high-frequency bobbing unrewarding -> the policy
        learns a smoother, latency-tolerant approach (the real arm bobbed under this lag)."""
        L = self.cfg.dr_action_latency_max
        if self._act_hist is None or self._act_hist.shape[0] != L + 1:
            self._act_hist = actions.unsqueeze(0).repeat(L + 1, 1, 1)         # prime with current
        self._act_hist = torch.cat([actions.unsqueeze(0), self._act_hist[:-1]], dim=0)  # push newest @0
        idx = self.act_latency.clamp(0, L)
        return self._act_hist[idx, torch.arange(self.num_envs, device=self.device)]

    def _apply_action(self, actions: torch.Tensor):
        """Joint delta-on-target + decimated physics. Returns the (noised/delayed) APPLIED action."""
        actions = actions.clamp(-1.0, 1.0)
        if self.cfg.dr_enable and self.cfg.dr_action_noise > 0:
            actions = (actions + torch.randn_like(actions) * self.cfg.dr_action_noise).clamp(-1.0, 1.0)
        if self.cfg.dr_enable and self.cfg.dr_action_latency_max > 0:
            actions = self._delayed_actions(actions)
        # ANTI-BOB: low-pass the ARM action (dims 0:5) so the policy can't oscillate at the loop rate.
        # Grip (dim 6) is left RAW so the one-way close-gate isn't delayed. The smoothed action is what's
        # applied AND returned (-> prev_action / reward), so training is consistent with the smoothing.
        if self.cfg.action_ema > 0.0:
            e = self.cfg.action_ema
            sm = e * self._ema_action[:, :6] + (1.0 - e) * actions[:, :6]
            self._ema_action[:, :6] = sm.detach()
            actions = torch.cat([sm, actions[:, 6:7]], dim=-1)
        self.arm_target = torch.clamp(
            self.arm_target + actions[:, :6] * self.cfg.arm_step, self.arm_lo, self.arm_hi)
        # grip: a_grip=+1 CLOSES (narrows gap = DECREASES qpos; qpos up = wider, per scripts/probe_gap.py).
        # ONE-WAY GATE: force OPEN only while still APPROACHING (far from the cube AND not yet engaged a
        # close). Once the gripper closes near the cube (grip_engaged), stop gating so it HOLDS through
        # the lift even as the lifted cube's distance grows (else a tiny slip -> force-open -> drop).
        gate_open = (~self.gate_near) & (~self.grip_engaged)
        a_grip = torch.where(gate_open.unsqueeze(-1), torch.full_like(actions[:, 6:7], -1.0), actions[:, 6:7])
        self.grip_engaged |= self.gate_near & (a_grip.squeeze(-1) > 0.0)
        self.grip_target = torch.clamp(
            self.grip_target - a_grip * self.cfg.grip_step,
            self.cfg.grip_close_qpos, self.cfg.grip_open_qpos)
        self.robots.set_joint_position_targets(self._full_targets())
        # obs_wrist: jitter the dome light (rendered-detection sim2real robustness) before the render
        if self.cfg.obs_wrist and self.cfg.dr_enable and self.cfg.dr_render_light > 0 and getattr(self, "_dome", None) is not None:
            f = 1.0 + float(torch.rand(1).item() * 2 - 1) * self.cfg.dr_render_light
            self._dome.GetIntensityAttr().Set(self._dome_base * f)
        # render ONLY the last physics substep (cameras need one fresh frame/step; 8x cheaper than every substep)
        last = self.cfg.decimation - 1
        for i in range(self.cfg.decimation):
            self.world.step(render=(self._render and i == last))
        return actions

    def step(self, actions: torch.Tensor):
        actions = self._apply_action(actions)
        obs, state, fallen = self._compute_state(actions)
        reward, success_inst, comp = compute_pick_reward(state, self.rcfg, self.cfg)
        # SUSTAINED success: a genuine grasp+lift must hold success_hold_steps; only then pay the
        # bonus + count it. A one-frame fling (cube batted up near the gripper) can't sustain.
        self.grasp_hold = torch.where(success_inst.bool(), self.grasp_hold + 1.0,
                                      torch.zeros_like(self.grasp_hold))
        success = self.grasp_hold >= self.cfg.success_hold_steps
        r_succ = self.rcfg.success_bonus * success.float()
        reward = reward + r_succ
        comp["r_success"] = r_succ

        self.ep_step += 1
        self.ep_return += reward
        self.ep_success = torch.maximum(self.ep_success, success.float())   # sticky within episode
        self.ep_grasped = torch.maximum(self.ep_grasped, state["is_grasped"].float())
        self.ep_fell = torch.maximum(self.ep_fell, fallen.float())
        timeout = self.ep_step >= self.cfg.episode_len
        info = {"success": success, "components": comp, "timeout": timeout,
                "table_hit": state["table_hit"], "y_back_hit": state["y_back_hit"], "fallen": fallen}

        # async per-env reset on success | fallen | timeout | table_hit | y_back_hit
        done = success.bool() | fallen.bool() | timeout.bool() | state["table_hit"].bool() | state["y_back_hit"].bool()
        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            self._last_ep_return[done_ids] = self.ep_return[done_ids]
            self._last_ep_len[done_ids] = self.ep_step[done_ids].float()
            self._last_ep_success[done_ids] = success[done_ids].float()
            self._last_ep_grasped[done_ids] = self.ep_grasped[done_ids]
            self._last_ep_fell[done_ids] = self.ep_fell[done_ids]
            self.reset(done_ids)
            obs = self.get_obs()

        info["ep_return"] = self._last_ep_return.clone()
        info["ep_len"] = self._last_ep_len.clone()
        info["ep_success"] = self._last_ep_success.clone()
        info["ep_grasped"] = self._last_ep_grasped.clone()
        info["ep_fell"] = self._last_ep_fell.clone()
        info["num_done"] = int(done_ids.numel())
        self.prev_action = actions.clone()
        return obs, reward, done, info

    def close(self):
        pass
