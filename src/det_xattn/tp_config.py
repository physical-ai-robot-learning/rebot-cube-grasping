"""tp_config: argparse for the DETECTION cross-attention SAC (C51) two-phase trainer.

Extracted verbatim from train_sac_det_xattn.py (the entry file). Only the DEAD
Squint privileged-reaching curriculum args (--reach-w/--reach-gate-sr/--reach-anneal)
were pruned; they were never referenced by the live TWO_PHASE path.

Import this BEFORE isaaclab.app.AppLauncher (parsing must happen before the sim
launches). Exposes `p` (the parser) and `args` (parsed, known-args-only).
"""
import argparse

p = argparse.ArgumentParser()
p.add_argument("--num-envs", type=int, default=256)
p.add_argument("--total-timesteps", type=int, default=1_000_000_000)
p.add_argument("--learning-starts", type=int, default=5000)
p.add_argument("--buffer-size", type=int, default=1_000_000)
p.add_argument("--buffer-device", type=str, default="cuda:0")
p.add_argument("--batch-size", type=int, default=512)
p.add_argument("--num-updates", type=int, default=64)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--gamma", type=float, default=0.9)
p.add_argument("--tau", type=float, default=0.01)
p.add_argument("--num-atoms", type=int, default=101)
p.add_argument("--v-min", type=float, default=-20.0)
p.add_argument("--v-max", type=float, default=20.0)
p.add_argument("--num-q", type=int, default=2)
p.add_argument("--alpha-floor", type=float, default=1e-3)
p.add_argument("--alpha-lr", type=float, default=0.0, help="separate alpha(entropy-temp) optimizer lr (0 = use --lr). user 2026-06-28: small (e.g. 3e-5) slows alpha collapse under high UTD")
p.add_argument("--target-entropy", type=float, default=0.0, help="SAC target entropy (0 = default -ACT_DIM). user 2026-06-28: RELAX (less negative, e.g. -3.5) -> autotune holds alpha higher -> more exploration")
p.add_argument("--http-port", type=int, default=8892)
p.add_argument("--save-dir", type=str, default="/tmp/det_xattn")
p.add_argument("--target-cube", type=str, default="cube")
p.add_argument("--single-cube", action="store_true")
p.add_argument("--dr", action="store_true")
p.add_argument("--dyn-dr", action="store_true")
p.add_argument("--headless", action="store_true")
p.add_argument("--render-size", type=int, default=64, help="HSV detection render res (detection only, not stored)")
p.add_argument("--detector", type=str, default="", help="path to YOLO-OBB red-cube weights (best.pt). If set, REPLACES HSV _red_det with the learned detector for BOTH cams.")
p.add_argument("--det-conf", type=float, default=0.15, help="YOLO detection confidence threshold (deploy uses the same)")
p.add_argument("--init-from", type=str, default="", help="warm-start actor/critic/log_alpha from a checkpoint .pt (resume). buffer NOT restored.")
args, _ = p.parse_known_args()
