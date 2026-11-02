"""Entry point: launch Isaac Sim (headless) then run state-based SAC pick training.

Usage (uv, from repo root):
    OMNI_KIT_ACCEPT_EULA=YES uv run python scripts/train.py
    OMNI_KIT_ACCEPT_EULA=YES uv run python scripts/train.py --num-envs 128 --steps 1000000 --exp run1
    # watch:  tensorboard --logdir runs   ->  http://localhost:6006
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description="reBot state-based SAC pick training.")
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None, help="per-env steps (total_env_steps)")
    ap.add_argument("--exp", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--init-ckpt", default=None,
                    help="warm-start: load this checkpoint's weights before training (e.g. a converged center policy)")
    ap.add_argument("--smoke-test", action="store_true",
                    help="tiny end-to-end run (4 envs, ~300 steps, fast eval/ckpt) to sanity-check the "
                         "pipeline on a GPU box BEFORE submitting the full run to the Ray cluster")
    args = ap.parse_args()

    # 1) load config FIRST (pure Python) so we know whether cameras are needed (obs_wrist renders).
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from rebot_sim2real.rl.config import load_train_config
    cfg = load_train_config(args.config)
    if args.num_envs is not None:
        cfg.env.num_envs = args.num_envs
    if args.steps is not None:
        cfg.total_env_steps = args.steps
    if args.exp is not None:
        cfg.exp_name = args.exp
    if args.device is not None:
        cfg.env.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed
    # CLUSTER: assets (USD/meshes ~74M) + demos are STAGED to a worker-local path (NFS -> /data), NOT
    # shipped in the ray working-dir (code-only). Resolve relative robot_usd/demo_path against this root.
    data_root = os.environ.get("REBOT_DATA_ROOT")
    if data_root:
        if not os.path.isabs(cfg.env.robot_usd):
            cfg.env.robot_usd = os.path.join(data_root, cfg.env.robot_usd)
        if cfg.sac.demo_path and not os.path.isabs(cfg.sac.demo_path):
            cfg.sac.demo_path = os.path.join(data_root, cfg.sac.demo_path)
        print(f"[train] REBOT_DATA_ROOT={data_root} -> robot_usd={cfg.env.robot_usd} demo_path={cfg.sac.demo_path}", flush=True)
    # Ray runs in an EPHEMERAL working-dir -> a relative ckpt_dir is wiped after the job. Point it at a
    # persistent path ($REBOT_CKPT_DIR, e.g. the real repo or NFS) so ckpts survive (MLflow artifacts too).
    ckpt_dir_env = os.environ.get("REBOT_CKPT_DIR")
    if ckpt_dir_env:
        cfg.ckpt_dir = ckpt_dir_env
        print(f"[train] REBOT_CKPT_DIR -> ckpt_dir={cfg.ckpt_dir}", flush=True)
    if args.smoke_test:                       # quick pre-submit sanity: tiny + fast, exercises the full pipeline
        cfg.env.num_envs = min(cfg.env.num_envs, 4)
        cfg.total_env_steps = 300
        cfg.sac.warmup_steps = 50
        cfg.eval_every = 200
        cfg.eval_episodes = 2
        cfg.latest_every = 100
        cfg.ckpt_every = 0
        print("[train] SMOKE TEST: 4 envs / 300 steps / eval@200 (pipeline check only)", flush=True)

    # 2) Isaac Sim MUST start before importing isaacsim.core / our env. obs_wrist/obs_top_uv need the
    #    render pipeline (per-env wrist/top cams) -> enable_cameras.
    from isaacsim import SimulationApp
    _need_cams = bool((cfg.env.obs_wrist or cfg.env.obs_top_uv) and not cfg.env.obs_analytic_uv)  # analytic (u,v) builds no cameras / never renders
    app = SimulationApp({"headless": True, "enable_cameras": _need_cams})

    # 3) now our code
    from rebot_sim2real.rl.train import train

    try:
        train(cfg, init_ckpt=args.init_ckpt)
    except Exception:
        # Isaac's fastShutdown hard-exits (code 0) and eats buffered stderr, so persist
        # the traceback to a file we can read after the run.
        import traceback
        with open("/tmp/train_err.txt", "w") as f:
            traceback.print_exc(file=f)
            f.flush()
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
