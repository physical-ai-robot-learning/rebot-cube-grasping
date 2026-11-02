"""Read (and optionally raise) the Damiao arm motors' POS_VEL profile registers — the likely fix for the slow
POS_VEL step response (measure_latency step: t@63%=711ms; sending vlim=32 vs 8 gave the SAME ~2.6 deg/s -> the
motor is NOT honoring our send_pos_vel vlim; it caps to its OWN registers).

In POS_VEL mode the DM motor moves to the target under its INTERNAL limits:
  RID 6 MAX_SPD (rad/s)  -> the speed ceiling (clamps the vlim we send)
  RID 4 ACC / 5 DEC      -> the accel/decel ramp (governs the step-response time-constant)
  RID 27 KP_APR / 28 KI_APR -> position-loop gains (low -> sluggish tracking)
If MAX_SPD/ACC are low, raising them makes POS_VEL fast (no need to switch to MIT). READ-ONLY by default.

  # SAFE read (no motion):
  .venv-deploy/bin/python src/tools/damiao_params.py
  # raise (⚠️ makes the arm move FASTER -> clear space; --store persists to motor flash):
  .venv-deploy/bin/python src/tools/damiao_params.py --set-max-spd 12 --set-acc 60 --set-dec -60 --store
then re-run:  measure_latency.py --mode step --joint 0 --deg 5 --go   (t@63% should drop a lot)
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rebot_sim2real.rbot_calculator.config import load_train_config
from rebot_sim2real.deploy.io.robot_io import DamiaoRobotIO, LEROBOT_ARM

URDF = "assets/robot_description_gripper/reBot_B601_DM_with_gripper.urdf"
F_REGS = {4: "ACC", 5: "DEC", 6: "MAX_SPD", 22: "VMAX", 23: "TMAX", 27: "KP_APR", 28: "KI_APR"}
U_REGS = {10: "CTRL_MODE"}   # 1=MIT 2=POS_VEL 3=VEL 4=FORCE_POS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/deploy_env.yaml")
    ap.add_argument("--set-max-spd", type=float, default=None, help="RID6 MAX_SPD rad/s (raise to unclamp vlim)")
    ap.add_argument("--set-acc", type=float, default=None, help="RID4 ACC (>0; profile ramp — ⚠️ may need --store+reboot to apply)")
    ap.add_argument("--set-dec", type=float, default=None, help="RID5 DEC (<0)")
    ap.add_argument("--set-kp-apr", type=float, default=None, help="RID27 KP_APR position-loop Kp (raise to speed the 1st-order response)")
    ap.add_argument("--set-ki-apr", type=float, default=None, help="RID28 KI_APR position-loop Ki")
    ap.add_argument("--set-kp-asr", type=float, default=None, help="RID25 KP_ASR speed-loop Kp")
    ap.add_argument("--store", action="store_true", help="persist writes to motor flash (profile regs may ONLY apply after --store + power-cycle)")
    ap.add_argument("--joint", type=int, default=None, help="0..5 = write ONLY this joint (per-joint tuning: wrist "
                    "is lighter -> oscillates at a LOWER KP_APR than the shoulder). Omit = all arm joints.")
    args = ap.parse_args()
    _wr = {6: args.set_max_spd, 4: args.set_acc, 5: args.set_dec, 27: args.set_kp_apr, 28: args.set_ki_apr, 25: args.set_kp_asr}
    _wr = {rid: v for rid, v in _wr.items() if v is not None}
    writing = bool(_wr)

    io = DamiaoRobotIO(load_train_config(args.config).env, URDF)
    io.connect()
    print(f"[damiao_params] connected. arm motors = {LEROBOT_ARM}\n"
          f"{'joint':<16}" + "".join(f"{n:>10}" for n in list(U_REGS.values()) + list(F_REGS.values())), flush=True)
    try:
        for lj in LEROBOT_ARM:
            m = io.robot.motors[lj]
            row = {}
            for rid, name in U_REGS.items():
                try: row[name] = m.get_register_u32(rid)
                except Exception as e: row[name] = f"e"
            for rid, name in F_REGS.items():
                try: row[name] = round(float(m.get_register_f32(rid)), 3)
                except Exception as e: row[name] = f"e"
            print(f"{lj:<16}" + "".join(f"{str(row[n]):>10}" for n in list(U_REGS.values()) + list(F_REGS.values())), flush=True)
            if writing and args.joint is not None and lj != LEROBOT_ARM[args.joint]:
                continue                                   # per-joint tuning: only write the targeted joint
            if writing:
                for rid, v in _wr.items():
                    m.write_register_f32(rid, float(v))
                if args.store: m.store_parameters()
                print(f"  -> {lj}: wrote {{{', '.join(f'{F_REGS[r]}={v}' for r, v in _wr.items())}}} store={int(args.store)}", flush=True)
        if writing:
            print("\n[damiao_params] ⚠️ registers written. Re-run measure_latency step -> t@63% should drop. "
                  "If NOT (MAX_SPD/ACC already high but still 711ms) -> the POS_VEL profile is the floor -> use MIT.")
        else:
            print("\n[damiao_params] READ-ONLY. If MAX_SPD/ACC look LOW vs a fast arm -> raise them with --set-* "
                  "(clear space first). CTRL_MODE should be 2 (POS_VEL) for the arm.")
    finally:
        io.estop()
        print("[damiao_params] done (torque off).")


if __name__ == "__main__":
    main()
