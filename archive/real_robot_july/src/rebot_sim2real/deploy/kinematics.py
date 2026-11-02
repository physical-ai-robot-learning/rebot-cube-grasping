"""Deploy kinematics helpers (3-thread refactor Step 1, docs/28).

Pure-ish IK/FK helpers extracted verbatim from deploy/debug_deepreach.py so the control thread can
own them in one testable module (PyCharm-debuggable in isolation). Each takes the RuleBaseController
`base` as an argument and uses only base's FK/kinematics + numpy/torch -- no cameras, no motor I/O.

WRIST_* are the wrist (un,vn) affine calibration constants (module state). The deploy --wrist-calib
override writes them via set_wrist_calib() (the old `global WRIST_*` write in the deploy script no
longer works once these live in another module).
"""

import numpy as np
import torch

# ⚠️ The old affine (-1.028*vn-0.142) came from a 1-pose J4-only fit that is NON-IDENTIFIABLE (see fit_wrist_calib.py
# / scratchpad/wrist_calib_*.py, 2026-07-14): its slope/offset absorbed geometry error and did NOT generalize. The
# CORRECT fix is a PHYSICAL fit (redesigned calib_wrist -> fit_wrist_calib.py) applied via `--wrist-calib`, which
# corrects the GEOMETRY (mount/aim[/focal]) and reduces this affine to a PURE SIGN (A=±1, B=0). These module
# defaults are only the fallback for the (failed) feedforward look-at when NO --wrist-calib is given; --wrist-calib
# overwrites them with (sign_u, 0) / (sign_v, 0) from the fit.
WRIST_VN_A, WRIST_VN_B = -1.028, -0.142
WRIST_UN_A, WRIST_UN_B = 1.0, 0.0


def set_wrist_calib(un_a, un_b, vn_a, vn_b):
    """Override the wrist (un,vn) affine calib (deploy --wrist-calib path). Module state so _project_wrist_unvn
    below reads the new values. Replaces the deploy script's old cross-module `global WRIST_*` write."""
    global WRIST_UN_A, WRIST_UN_B, WRIST_VN_A, WRIST_VN_B
    WRIST_UN_A, WRIST_UN_B = float(un_a), float(un_b)
    WRIST_VN_A, WRIST_VN_B = float(vn_a), float(vn_b)


def _wrist_lookat_j4(base, arm_q, cube_world, plane_z, target_vn, iters=6, fd_eps=1.5e-3, step_cap=0.5):
    """FEEDFORWARD wrist look-at (open-loop, geometry-only — CANNOT oscillate: no wrist-pixel feedback).
    Return (j4_target_rad, vn_at_current_j4): the J4 (wrist pitch) that projects the cube CENTER to normalized
    image v == target_vn, solved by Newton on the 1-DOF vn(J4). The forward projection casts cube_world through
    the LIVE FK wrist-cam using the SAME world-up look-at frame as base._wrist_uv_to_world (so forward<->inverse
    are self-consistent): un=(xc/depth)(fx/cx), vn=(yc/depth)(fy/cy), depth=-zc, axes z=-optical, x=up×z, y=z×x.
    J4 also TRANSLATES the lens (cam_w moves as the wrist pitches) so the model captures that coupling via the
    finite-diff derivative. CAVEAT: the wrist model is C922-spec (uncalibrated: cx/cy=center, ~9px v residual +
    FK cam-pose gap), so the ABSOLUTE vn carries a CONSTANT bias -> the framing is STABLE-but-biased; trim
    target_vn (--lookat-vn) to where the jaws actually close. Feedback-free = the fix for the seg-servo overshoot."""
    fx, fy, cx, cy = base.wrist_K
    p = torch.tensor([float(cube_world[0]), float(cube_world[1]), float(plane_z)],
                     dtype=torch.float32, device=base.device)
    q = torch.tensor(np.asarray(arm_q, dtype=np.float32), device=base.device).view(1, 6)
    lo, hi = float(base.lo[3]), float(base.hi[3])
    up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=base.device)

    aq0 = np.asarray(arm_q, dtype=np.float32)
    p3 = [float(cube_world[0]), float(cube_world[1]), float(plane_z)]
    def vn_of(j4):                                          # CALIBRATED vn (real-detector frame) via _project_wrist_unvn
        aq = aq0.copy(); aq[3] = float(min(hi, max(lo, j4)))
        _un, vn, depth, _vis = _project_wrist_unvn(base, aq, p3)
        return vn if depth > 1e-3 else None

    j4 = float(q[0, 3]); vn0 = vn_of(j4); cur = j4
    for _ in range(iters):
        vn = vn_of(cur)
        if vn is None:
            break
        vn_e = vn_of(cur + fd_eps)
        if vn_e is None:
            break
        dvn = (vn_e - vn) / fd_eps
        if abs(dvn) < 1e-6:                                 # flat -> can't step
            break
        step = max(-step_cap, min(step_cap, -(vn - target_vn) / dvn))
        cur = max(lo, min(hi, cur + step))
        if abs(step) < 1e-4:
            break
    return cur, (vn0 if vn0 is not None else 0.0)


def _project_wrist_unvn(base, arm_q, p_world, calib=True):
    """FORWARD wrist projection (SAME world-up look-at frame as base._wrist_uv_to_world): world point p_world[3]
    at joints arm_q[6] -> (un, vn, depth, vis). un=(xc/depth)(fx/cx), vn=(yc/depth)(fy/cy), depth=-zc; axes
    z=-optical, x=up×z, y=z×x. With calib=True (default) applies the AFFINE real-detector calibration
    (WRIST_*_A/B) so (un,vn) match what the detector reports (the feedforward look-at target frame). Pass
    calib=False to get the RAW model projection (calib_wrist records this to (re)fit the affine constants)."""
    fx, fy, cx, cy = base.wrist_K
    q = torch.tensor(np.asarray(arm_q, dtype=np.float32), device=base.device).view(1, 6)
    p = torch.tensor([float(p_world[0]), float(p_world[1]), float(p_world[2])], dtype=torch.float32, device=base.device)
    _, Rg, cam_w = base._fk_full(q)
    fwd = base._cam_axis(Rg)[0]; fwd = fwd / fwd.norm().clamp_min(1e-6)
    z = -fwd
    up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=base.device)
    x = torch.cross(up, z, dim=-1); x = x / x.norm().clamp_min(1e-6)
    y = torch.cross(z, x, dim=-1)
    rel = p - cam_w[0]
    zc = float(torch.dot(rel, z)); depth = -zc
    if depth <= 1e-3:
        return 0.0, 0.0, depth, False
    un = float(torch.dot(rel, x)) / depth * (fx / cx)
    vn = float(torch.dot(rel, y)) / depth * (fy / cy)
    if calib:
        un = WRIST_UN_A * un + WRIST_UN_B                    # affine cam-model calib -> real-detector frame
        vn = WRIST_VN_A * vn + WRIST_VN_B                    #   (fixes the legacy-vs-render v-sign flip + offset)
    return un, vn, depth, (depth > 0.01 and abs(un) < 1.5 and abs(vn) < 1.5)


def _gimbal_lookat_ik(base, seed_q, target_world, cube_world, iters=80, lam=0.08, pstep=0.03, astep=0.02,
                      aim_gain=1.0, freeze=(), eps=1e-4, meas_uv=None):
    """GIMBAL look-at IK ('approach'): drive TCP -> target_world AND aim the wrist cam AT the cube with a
    task-PRIORITY NULL-SPACE step — primary = position (damped pinv); secondary = cam-aim as a projected gradient
    in the position null space, so it HOLDS the TCP while aiming (unlike ik_oriented's combined DLS, which trades
    position for orientation and diverges). Offline (test_gimbal_ik2): TCP 0mm + cam 0deg for dz>=~5cm; only the
    grasp-close (dz<~2cm, cam beside the cube -> extreme angle) can't fully aim -> the caller TAPERS to straight-down
    there. aim_gain scales the aim step (>1 = the cam LOCKS ON faster per tick); astep caps it.

    AIM TARGET (two modes):
      • WORLD (meas_uv=None): aim the optical axis at `cube_world` — uses the nominal wrist extrinsic + the (possibly
        biased) front-est cube xy, so a hand-eye/front error leaves the cube OFF-CENTER in the image.
      • IMAGE (meas_uv=[N,2] = measured normalized wrist (un,vn)): aim the axis at the MEASURED cube BEARING, i.e.
        rotate the axis by exactly the (un,vn) image offset (a RELATIVE rotation off the current axis) so it drives
        measured (un,vn) -> (0,0). Intrinsic-only + relative => the absolute extrinsic/front-est error CANCELS
        (true image-based look-at; the fix for 'the cube sits off-center even though detection is correct').
    seed_q [N,6], target_world/cube_world [N,3] tensors. Returns q [N,6] (clamped)."""
    dev = base.device
    q = seed_q.clone(); N = q.shape[0]
    I6 = torch.eye(6, device=dev); I3 = torch.eye(3, device=dev)
    des_fixed = None
    if meas_uv is not None:
        # IMAGE-based desired axis, computed ONCE at the seed pose (per control tick; the outer loop re-measures):
        # ray through measured (un,vn) in the SAME cam frame as _wrist_uv_to_world, world-transformed. This is the
        # current axis rotated by the measured offset -> nulling (un,vn) needs only this relative rotation.
        fx, fy, cx, cy = base.wrist_K
        _, Rg0, _ = base._fk_full(seed_q)
        f0 = base._cam_axis(Rg0); f0 = f0 / f0.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        z0 = -f0
        up0 = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(N, 3)
        x0 = torch.cross(up0, z0, dim=-1); x0 = x0 / x0.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        y0 = torch.cross(z0, x0, dim=-1)
        ray_x = (meas_uv[:, 0:1] * cx / fx); ray_y = (meas_uv[:, 1:2] * cy / fy)
        des_fixed = ray_x * x0 + ray_y * y0 - z0
        des_fixed = des_fixed / des_fixed.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    for _ in range(int(iters)):
        tcp, Rg, campos = base._fk_full(q)
        cam = base._cam_axis(Rg)
        pos_err = target_world - tcp
        if des_fixed is not None:
            des = des_fixed
        else:
            des = cube_world - campos
            des = des / des.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        ax_err = des - cam
        J_pos = torch.zeros(N, 3, 6, device=dev); J_aim = torch.zeros(N, 3, 6, device=dev)
        for j in range(6):
            qp = q.clone(); qp[:, j] += eps
            tcpj, Rgj, _ = base._fk_full(qp)
            J_pos[:, :, j] = (tcpj - tcp) / eps
            J_aim[:, :, j] = (base._cam_axis(Rgj) - cam) / eps
        # ---- Symbol map (task-priority null-space step) ----
        #   @lam      = damping factor (suppresses divergence near singularities)   @aim_gain = cam-aim step scale   @ax_err = des − cam (aim-axis error)
        #   J_pos_pinv = J_posᵀ(J_pos J_posᵀ + lam²·I)⁻¹        : damped pseudo-inverse (position = primary task)
        #   null_space = I − J_pos_pinv·J_pos                   : position-task null-space projector (preserves TCP)
        #   dq_aim     = null_space·(J_aimᵀ·aim_gain·ax_err)    : aiming only inside the null-space -> never fights position
        J_pos_T = J_pos.transpose(1, 2)
        J_pos_pinv = J_pos_T @ torch.linalg.inv(J_pos @ J_pos_T + (lam ** 2) * I3)   # [N,6,3] damped pinv
        dq_position = (J_pos_pinv @ pos_err.unsqueeze(-1)).squeeze(-1).clamp(-pstep, pstep)
        null_space = I6 - J_pos_pinv @ J_pos                              # TCP-preserving null space (I − J⁺J)
        dq_aim = (null_space @ (J_aim.transpose(1, 2) @ (aim_gain * ax_err).unsqueeze(-1))).squeeze(-1).clamp(-astep, astep)
        dq = dq_position + dq_aim
        if freeze:
            dq[:, list(freeze)] = 0.0        # remove these joints from the solution IN-LOOP -> the IK aims WITHOUT them
        q = (q + dq).clamp(base.lo, base.hi)
    return q


def _danger(base, qf, target_xy, surf_z, resid_max=0.05, floor=0.015):
    """Judge a DANGEROUS commanded pose PURELY from kinematics (FK/IK) — no extra sensors. Returns a reason
    string (-> caller auto-returns to REST) or None.

    ⚠️ Two load-bearing assumptions — get either wrong and it FALSE-TRIPS every tick (constant aborts):
      1. CLAMP: qf is the IK output ALREADY clamped to [base.lo, base.hi]. So a per-joint 'near its HW limit'
         test is USELESS — a saturated joint sits EXACTLY at the limit even for a perfectly fine pose (e.g. the
         gimbal aim legitimately parks J5 ~90°, at its limit). We therefore do NOT test joint-limit proximity;
         genuine over-extension instead shows up as the IK failing to REACH the xy target -> the `resid` test.
      2. FRAME: every z here is in the BASE-CONTROLLER FK frame (base.fk_world). `surf_z` (= base.cube_top_z,
         the auto-z descend surface) is in that SAME frame, so the floor test is frame-consistent. Comparing a
         base-FK z against a raw cfg.table_z (a DIFFERENT frame, offset by the live auto-z z_bias) would
         false-trip at the legit +5cm hover whenever the z_bias is large — hence surf_z, NOT table_z.

    Checks: (1) commanded pose misses the xy target by > resid_max (cube out of reach / arm over-stretch),
    (2) TCP would drop `floor` below the descend surface (a plunge). qf = commanded q [6]; target_xy = tracked cube xy."""
    tcp = base.fk_world(torch.tensor(np.asarray(qf, float), dtype=torch.float32, device=base.device).view(1, 6))[0].cpu().numpy()
    resid = float(np.linalg.norm(tcp[:2] - np.asarray(target_xy, float)))
    if resid > resid_max:
        return f"IK can't reach xy ({resid * 1000:.0f}mm residual = cube out of reach / over-stretch)"
    if float(tcp[2]) < surf_z - floor:
        return f"TCP z {tcp[2]:.3f} below the descend surface {surf_z:.3f} (plunge)"
    return None
