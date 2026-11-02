#!/usr/bin/env python3
"""Fit the WRIST camera PHYSICAL params (lens position + optical axis [+ focal]) so the analytic wrist projection
matches the real detector's (u,v) — the foundation for a geometric look-at (option-3 R&D, [[deep-reach-deploy-fkzgap-latch]]).

WHY (session 13-14): the wrist model is spec-derived + UNCALIBRATED — its lens pose is off and the J4->image
sensitivity is wrong/pose-dependent, which broke every look-at attempt. A single-pose OUTPUT affine (old WRIST_VN_A/B)
did NOT generalize. Fitting the PHYSICAL params instead generalizes (it fixes the root geometry), and reuses the
EXACT deploy projection (same world-up look-at convention as base._wrist_uv_to_world / `_project_wrist_unvn`), so the
result plugs straight back into deploy via debug_deepreach.py `--wrist-calib deploy/wrist_calib_fit.npz` (a
DEPLOY-ONLY override of base.wrist_mount/wrist_aim [+ wrist_K focal] + the WRIST_*_A sign; the shared config/sim is
NOT touched).

⭐ IDENTIFIABILITY (validated offline 2026-07-14, scratchpad/wrist_calib_*.py): the OLD protocol (J4-only sweep,
cam straight DOWN) is NON-IDENTIFIABLE — even starting at the true params the fit drifts 2-46deg in axis at ~0 RMS
(the x=up×z image frame is GIMBAL-SINGULAR at cam-vertical, and a small pitch sweep with the cube near image-center
carries almost no signal). Multiple cube positions at fixed clearance do NOT fix it. The REDESIGNED calib_wrist
(offset look-at hover => ~15deg off-vertical; J1 + wider J4 sweep; >=6 DISTINCT cube positions with rotating offset
direction) IS identifiable: extrinsic recovers axis to ~0.1deg across random truths, focal to a few %. FOCAL is the
weakest DOF; extrinsic-only (--no-fit-focal) still recovers the axis (focal error is absorbed by the --lookat-vn
trim), so applying focal is OPTIONAL.

DATA: deploy/wrist_calib_data.npz from the redesigned `calib_wrist`. Cols: q0..q5, cx,cy,cz (cube_world), mun,mvn,
mdepth (RAW model proj), sun,svn,svis (seg centroid), wun,wvn,wvis (sd4 CNN center) [, base_z]. base_z is recorded
so the fit's FK frame MATCHES the record-time frame (else a ~40mm z error is silently absorbed into the params).

USAGE:  .venv-deploy/bin/python src/tools/fit_wrist_calib.py [--target cnn|seg] [--no-fit-focal]
Prints the fitted params + before/after reprojection RMS + the exact numbers to apply. Does NOT auto-apply
(review, then run deploy with `--wrist-calib deploy/wrist_calib_fit.npz` and HW-validate the (u,v) match).
"""
import sys, os, argparse, math
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rebot_sim2real.rbot_calculator.base_controller import RuleBaseController
from rebot_sim2real.rbot_calculator.config import load_train_config

# KNOWN detector-vs-model sign (world-up look-at model uses v=+yc/depth; sd4 CNN reports v with the RENDER sign,
# opposite -> sv=-1; un is ~identity -> su=+1). The J4/J1 sweep auto-detect is only a CROSS-CHECK (a near-centered
# cube makes the u auto-detect noisy), so we DEFAULT to the known convention and WARN on a confident disagreement.
KNOWN_SIGN = {"cnn": (1.0, -1.0), "seg": (1.0, -1.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="deploy/wrist_calib_data.npz")
    ap.add_argument("--out", default="deploy/wrist_calib_fit.npz")
    ap.add_argument("--config", default="configs/deploy_env.yaml")
    ap.add_argument("--target", choices=("cnn", "seg"), default="cnn",
                    help="which detector (u,v) to calibrate to. cnn = sd4 wun/wvn (what look-at servos on); "
                         "seg = red-mask sun/svn. Default cnn.")
    ap.add_argument("--base-z", type=float, default=None,
                    help="world-frame base_pos.z used AT RECORD TIME (debug_deepreach --base-z, default 0.010). "
                         "Normally read from the npz (calib_wrist records it); this overrides for legacy data.")
    ap.add_argument("--no-fit-focal", action="store_true",
                    help="fit ONLY the extrinsic (mount+aim), keep focal at spec. Focal is the weakest DOF and the "
                         "axis recovers without it; the residual focal error is absorbed by the --lookat-vn trim.")
    ap.add_argument("--fov", type=float, default=0.95,
                    help="drop samples with |un| or |vn| > this (out-of-frame; the detector would report vis=0).")
    ap.add_argument("--sign-u", type=float, default=None, choices=(-1.0, 1.0), dest="sign_u",
                    help="EXPLICIT detector-vs-model u-sign override (default = KNOWN_SIGN[target]). Only set if the "
                         "cross-check WARNS and you have confirmed the detector convention really differs.")
    ap.add_argument("--sign-v", type=float, default=None, choices=(-1.0, 1.0), dest="sign_v",
                    help="EXPLICIT detector-vs-model v-sign override (default = KNOWN_SIGN[target]).")
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=0.02)
    args = ap.parse_args()

    cfg = load_train_config(args.config).env
    base = RuleBaseController(cfg, "cpu")

    d = np.load(args.data, allow_pickle=True)
    rows = d["rows"]; cols = list(d["cols"]); ci = {str(c): i for i, c in enumerate(cols)}
    # FRAME MATCH: the FK must use the SAME base_pos.z the data was recorded at (else a ~40mm z error is absorbed).
    if args.base_z is not None:
        base_z = float(args.base_z)
    elif "base_z" in ci:
        base_z = float(np.median(rows[:, ci["base_z"]]))
    else:
        base_z = 0.010
        print("[fit] ⚠️ npz has no base_z col (legacy data) -> assuming the deploy default 0.010. If the data was "
              "recorded with a different --base-z, pass --base-z to match (a mismatch biases the fitted z).")
    base.base_pos[2] = base_z

    mount0 = base.wrist_mount.detach().clone().float()          # [3] lens pos in gripper frame
    aim0 = base.wrist_aim.detach().clone().float()              # [3] aim pt in gripper frame
    fx0, fy0, cx, cy = [float(v) for v in base.wrist_K]
    uc, vc, visc = (("wun", "wvn", "wvis") if args.target == "cnn" else ("sun", "svn", "svis"))

    # ---- precompute per-sample FK constants (NO grad; params are fit separately) ----
    Rg_l, glink_l, cube_l, obs_l, cubexy = [], [], [], [], set()
    n_raw = n_novis = n_oof = 0
    for r in rows:
        n_raw += 1
        if float(r[ci[visc]]) < 0.5:
            n_novis += 1; continue
        un, vn = float(r[ci[uc]]), float(r[ci[vc]])
        if abs(un) > args.fov or abs(vn) > args.fov:            # out-of-frame (detector wouldn't localize it)
            n_oof += 1; continue
        aq = torch.tensor(r[ci["q0"]:ci["q0"] + 6], dtype=torch.float32).view(1, 6)
        with torch.no_grad():
            _, Rg, cam_w = base._fk_full(aq)                    # cam_w = lens world (uses base_pos.z = base_z now)
            glink = cam_w[0] - Rg[0] @ mount0                   # gripper-link world pos (independent of fitted mount)
        Rg_l.append(Rg[0]); glink_l.append(glink)
        cube_l.append(torch.tensor([r[ci["cx"]], r[ci["cy"]], r[ci["cz"]]], dtype=torch.float32))
        obs_l.append(torch.tensor([un, vn], dtype=torch.float32))
        cubexy.add((round(float(r[ci["cx"]]), 2), round(float(r[ci["cy"]]), 2)))
    if len(obs_l) < 8:
        print(f"[fit] only {len(obs_l)} usable samples (raw {n_raw}, no-vis {n_novis}, out-of-frame {n_oof}) — "
              "need more (move the cube to more positions, re-run calib_wrist). Abort.")
        return
    Rg = torch.stack(Rg_l); glink = torch.stack(glink_l); cube = torch.stack(cube_l); obs = torch.stack(obs_l)
    M = obs.shape[0]; npos = len(cubexy)
    print(f"[fit] base_z={base_z:+.3f} · target={args.target} · {M} usable / {n_raw} raw "
          f"(dropped {n_novis} no-vis, {n_oof} out-of-frame) · {npos} distinct cube positions")

    up = torch.tensor([0.0, 0.0, 1.0])

    def project(mount, aim, s):
        """Differentiable forward wrist projection [M,2] in the SAME world-up look-at convention as
        _project_wrist_unvn / base._wrist_uv_to_world. s = focal scale."""
        lens = glink + (Rg @ mount.view(1, 3, 1)).squeeze(-1)             # [M,3]
        axl = aim - mount; axl = axl / axl.norm().clamp_min(1e-9)
        fwd = (Rg @ axl.view(1, 3, 1)).squeeze(-1)                       # [M,3] world optical axis
        fwd = fwd / fwd.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        z = -fwd
        x = torch.cross(up.expand_as(z), z, dim=-1); x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        y = torch.cross(z, x, dim=-1)
        rel = cube - lens
        xc = (rel * x).sum(-1); yc = (rel * y).sum(-1); zc = (rel * z).sum(-1)
        depth = (-zc).clamp_min(1e-3)
        un = xc / depth * (fx0 * s / cx)
        vn = yc / depth * (fy0 * s / cy)
        return torch.stack([un, vn], dim=-1)

    # ---- detector-vs-model SIGN: USE THE KNOWN CONVENTION (the auto value would come from the UNCALIBRATED spec
    # geometry — exactly what we are fixing — and a near-centered / J4-only-ish sweep makes it noisy). Cross-check
    # with a proper BOUNDED Pearson correlation and WARN on a confident disagreement, but do NOT auto-flip: a
    # spurious flip freezes obs_m before the Adam loop -> unrecoverable, wrong-way look-at. Override with --sign-*.
    su = args.sign_u if args.sign_u is not None else KNOWN_SIGN[args.target][0]
    sv = args.sign_v if args.sign_v is not None else KNOWN_SIGN[args.target][1]

    def _pearson(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))

    with torch.no_grad():
        m0 = project(mount0, aim0, torch.tensor(1.0))
        for nm, chosen, mv, ov in (("u", su, m0[:, 0], obs[:, 0]), ("v", sv, m0[:, 1], obs[:, 1])):
            r = _pearson(mv, ov)                                          # in [-1,1]; sign(r) = data-suggested sign
            if abs(r) > 0.6 and ((r < 0) != (chosen < 0)):
                print(f"[fit] ⚠️ {nm}-sign: using {chosen:+.0f} (KNOWN/--sign-{nm}) but the data suggests "
                      f"{(-1 if r < 0 else 1):+.0f} (pearson {r:+.2f}). If the detector convention REALLY differs, "
                      f"re-run with --sign-{nm} {(-1 if r < 0 else 1):+.0f}.")
        obs_m = obs * torch.tensor([su, sv])                             # flip obs into the model (legacy) frame
        rms0 = ((m0 - obs_m) ** 2).mean().sqrt().item()
    print(f"[fit] sign vs detector: u {su:+.0f}, v {sv:+.0f}  (KNOWN default; model v=+, sd4 v=-)")
    print(f"[fit] reprojection RMS (current spec model) = {rms0:.4f}")

    # ---- fit lens-offset delta (3) + aim delta (3) [+ log focal scale (1)] ----
    dt = torch.zeros(3, requires_grad=True)
    da = torch.zeros(3, requires_grad=True)
    logs = torch.zeros(1, requires_grad=not args.no_fit_focal)
    params = [dt, da] + ([logs] if not args.no_fit_focal else [])
    opt = torch.optim.Adam(params, lr=args.lr)
    for _ in range(args.iters):
        opt.zero_grad()
        pred = project(mount0 + dt, aim0 + da, torch.exp(logs))
        loss = ((pred - obs_m) ** 2).mean() + 1e-4 * (dt.pow(2).sum() + da.pow(2).sum())   # tiny reg -> gauge
        loss.backward(); opt.step()
    with torch.no_grad():
        pred = project(mount0 + dt, aim0 + da, torch.exp(logs))
        rms1 = ((pred - obs_m) ** 2).mean().sqrt().item()
        mount_fit = (mount0 + dt).tolist(); aim_fit = (aim0 + da).tolist(); scale = float(torch.exp(logs))
    ax0 = (aim0 - mount0) / (aim0 - mount0).norm()
    axf = (torch.tensor(aim_fit) - torch.tensor(mount_fit)); axf = axf / axf.norm()
    ax_move = math.degrees(math.acos(float(torch.dot(ax0, axf).clamp(-1, 1))))
    print(f"[fit] reprojection RMS (fitted model)       = {rms1:.4f}   (was {rms0:.4f})")
    print(f"[fit] wrist_mount_offset: {[round(v,4) for v in mount0.tolist()]} -> {[round(v,4) for v in mount_fit]}")
    print(f"[fit] wrist_aim_local  : {[round(v,4) for v in aim0.tolist()]} -> {[round(v,4) for v in aim_fit]}")
    print(f"[fit] optical-axis moved {ax_move:.1f} deg from spec"
          + ("" if args.no_fit_focal else f" · focal scale {scale:.4f} (fx {fx0:.1f} -> {fx0*scale:.1f})"))

    # ---- identifiability guard for the REDESIGNED protocol (needs off-vertical + a J1 sweep + >=6 positions) ----
    j1span = float(rows[:, ci["q0"]].max() - rows[:, ci["q0"]].min())
    if npos < 6 or j1span < 0.15:
        print(f"[fit] ⚠️ UNDER-IDENTIFIED risk: {npos} distinct positions (want >=6) / J1 span {j1span:.2f} rad "
              "(want a J1 sweep, >=0.15). The J4-only near-vertical protocol is NON-IDENTIFIABLE — recollect with "
              "the redesigned calib_wrist (offset look-at + J1 sweep + move the cube to >=6 positions).")

    np.savez(args.out, wrist_mount_offset=np.array(mount_fit, np.float32),
             wrist_aim_local=np.array(aim_fit, np.float32),
             focal_scale=np.float32(scale if not args.no_fit_focal else 1.0),
             fit_focal=np.bool_(not args.no_fit_focal),
             sign_u=np.float32(su), sign_v=np.float32(sv), base_z=np.float32(base_z),
             rms_before=np.float32(rms0), rms_after=np.float32(rms1), n=np.int32(M), n_pos=np.int32(npos))
    print(f"[fit] saved -> {args.out}")
    print(f"[fit] TO APPLY (deploy-only, review first): rerun debug_deepreach.py with  --wrist-calib {args.out}")
    print("[fit]   (it overrides base.wrist_mount/wrist_aim"
          + ("" if args.no_fit_focal else " + wrist_K focal") + f" and sets WRIST_UN_A={su:+.0f}, WRIST_VN_A={sv:+.0f}, "
          "WRIST_*_B=0). Then HW-validate: model (u,v) should track the detector across a J4 sweep.")


if __name__ == "__main__":
    main()
