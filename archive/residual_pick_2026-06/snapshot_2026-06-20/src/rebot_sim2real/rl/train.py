"""State-based SAC training loop for the reBot pick (v1), with TensorBoard logging.

Run via ``scripts/train.py`` (which creates the Isaac SimulationApp first). Logs scalar
curves to ``<log_dir>/<exp_name>`` — view with ``tensorboard --logdir runs``.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque

import torch


def _dump_live_json(cfg, payload: dict) -> None:
    """Atomically dump latest metrics to ``<log_dir>/<exp_name>_live.json`` (tmp + os.replace).

    The headless trainer can't render; scripts/monitor.py reads this file and relays it to the
    live dashboard. Best-effort — visualization must never break training.
    """
    try:
        path = os.path.join(cfg.log_dir, f"{cfg.exp_name}_live.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001
        pass

from .config import TrainConfig, config_to_dict
from .sac import SAC, ReplayBuffer


def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten a nested config dict to dotted keys (for MLflow params)."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


class _MLflowLogger:
    """Best-effort MLflow logging to the central tracking server (BH cluster head:5000).

    NEVER breaks training: if mlflow is absent, MLFLOW_DISABLE is set, or the server is
    unreachable (WiFi), it disables itself silently. tensorboard logging is unaffected.
    Tracking URI from $MLFLOW_TRACKING_URI (default the cluster head); experiment = exp_name.
    Logged at LOW cadence only (200-step summaries + eval points + ckpt artifacts) to spare WiFi.
    """

    def __init__(self, cfg):
        self.enabled = False
        self._mlflow = None
        if os.environ.get("MLFLOW_DISABLE"):
            print("[mlflow] disabled (MLFLOW_DISABLE set)", flush=True)
            return
        uri = os.environ.get("MLFLOW_TRACKING_URI", "http://<ray-head>:5000")
        try:
            import mlflow
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(cfg.exp_name)
            run = mlflow.start_run(run_name=cfg.exp_name)
            self._mlflow = mlflow
            self.enabled = True
            print(f"[mlflow] logging to {uri} exp={cfg.exp_name} run={run.info.run_id[:8]}", flush=True)
        except Exception as e:  # noqa: BLE001  -- server down / mlflow absent -> just skip
            print(f"[mlflow] disabled ({type(e).__name__}: {e}) -> tensorboard only", flush=True)

    def log_params(self, params: dict) -> None:
        if not self.enabled:
            return
        try:
            self._mlflow.log_params({k: str(v) for k, v in params.items()})
        except Exception:  # noqa: BLE001
            pass

    def log_metrics(self, metrics: dict, step: int) -> None:
        if not self.enabled:
            return
        try:
            self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=int(step))
        except Exception:  # noqa: BLE001
            pass

    def log_artifact(self, path: str) -> None:
        if not self.enabled:
            return
        try:
            self._mlflow.log_artifact(path)
        except Exception:  # noqa: BLE001
            pass

    def end(self) -> None:
        if not self.enabled:
            return
        try:
            self._mlflow.end_run()
        except Exception:  # noqa: BLE001
            pass


def train(cfg: TrainConfig, init_ckpt: str = None):
    """Single-policy ("full") training. init_ckpt warm-starts the weights."""
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.env.device)

    # env import is deferred until SimulationApp exists (done by the entry script)
    from ..envs.pick_env import RebotPickEnv
    env = RebotPickEnv(cfg.env, cfg.reward)

    from torch.utils.tensorboard import SummaryWriter
    cfg_dict = config_to_dict(cfg)
    writer = SummaryWriter(os.path.join(cfg.log_dir, cfg.exp_name))
    writer.add_text("config", "```\n" + _yaml(cfg_dict) + "\n```")
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    mlflow_log = _MLflowLogger(cfg)                   # central tracking (best-effort, alongside tensorboard)
    mlflow_log.log_params(_flatten(cfg_dict))

    agent = SAC(env.obs_dim, env.act_dim, cfg.sac, device)
    if init_ckpt and os.path.exists(init_ckpt):       # warm-start from a converged policy
        agent.load_state_dict(torch.load(init_ckpt, map_location=device))
        print(f"[train] warm-started from {init_ckpt}", flush=True)
    buffer = ReplayBuffer(cfg.sac.replay_size, env.obs_dim, env.act_dim, device)
    # RLPD-style demo bootstrapping (optional): load scripted-expert demos to mix into each batch
    demo_buffer = None
    if cfg.sac.demo_path and os.path.exists(cfg.sac.demo_path):
        from .demo import load_demo_buffer
        demo_buffer = load_demo_buffer(cfg.sac.demo_path, device, expected_obs_dim=env.obs_dim)
    try:
        run_sac(env, agent, buffer, cfg, writer, cfg.total_env_steps,
                demo_buffer=demo_buffer, mlflow_log=mlflow_log)
    finally:
        writer.close()
        mlflow_log.end()
    print("[train] done.", flush=True)


def run_sac(env, agent, buffer, cfg, writer, total_env_steps, tag="", demo_buffer=None, mlflow_log=None):
    """Reusable SAC rollout/update loop. Logs under an optional `tag/` prefix and writes
    checkpoints. Returns the trained agent. `mlflow_log` (best-effort) mirrors LOW-cadence
    metrics (200-step summaries + eval + ckpt artifacts) to the central MLflow server."""
    device = torch.device(cfg.env.device)
    N = env.num_envs
    pfx = f"{tag}/" if tag else ""
    ckpt_name = f"{cfg.exp_name}_{tag}" if tag else cfg.exp_name
    label = tag or "full"                 # console + log label
    print(f"[train] phase={label} | {N} envs | obs={env.obs_dim} act={env.act_dim} | steps={total_env_steps}", flush=True)

    obs = env.get_obs()
    ret_window = deque(maxlen=200)
    succ_window = deque(maxlen=200)
    len_window = deque(maxlen=200)
    last_log_t = time.time()
    last_log_step = 0
    best_eval_sr = -1.0
    last_adv_step = 0                          # spawn curriculum: last step pos_rand was widened

    for step in range(1, total_env_steps + 1):
        if step * N < cfg.sac.warmup_steps:
            action = torch.empty(N, env.act_dim, device=device).uniform_(-1, 1)
        else:
            action = agent.act(obs, deterministic=False)

        next_obs, reward, done, info = env.step(action)
        buffer.add_batch(obs, action, reward, next_obs, done)
        obs = next_obs

        if info["num_done"] > 0:
            for i in done.nonzero(as_tuple=False).squeeze(-1).tolist():
                ret_window.append(float(info["ep_return"][i]))
                succ_window.append(float(info["ep_success"][i]))
                len_window.append(float(info["ep_len"][i]))

        metrics = {}
        if len(buffer) >= max(cfg.sac.batch_size, cfg.sac.warmup_steps):
            for _ in range(cfg.sac.updates_per_step):
                metrics = agent.update(buffer, cfg.sac.batch_size,
                                       demo_buffer=demo_buffer, demo_frac=cfg.sac.demo_frac)

        if step % cfg.log_every == 0 and metrics:
            gstep = step * N
            for k, v in metrics.items():
                writer.add_scalar(f"{pfx}losses/{k}", v, gstep)
            if ret_window:
                writer.add_scalar(f"{pfx}train/ep_return", sum(ret_window) / len(ret_window), gstep)
                writer.add_scalar(f"{pfx}train/success_rate", sum(succ_window) / len(succ_window), gstep)
                writer.add_scalar(f"{pfx}train/ep_len", sum(len_window) / len(len_window), gstep)
            for k, v in info["components"].items():
                try:
                    writer.add_scalar(f"{pfx}reward/{k}", float(v.mean()), gstep)
                except Exception:  # noqa: BLE001
                    pass
            writer.add_scalar(f"{pfx}reward/mean_step", float(reward.mean()), gstep)
            writer.add_scalar(f"{pfx}perf/buffer", len(buffer), gstep)

        if step % 200 == 0:
            now = time.time()
            fps = (step - last_log_step) * N / max(now - last_log_t, 1e-6)
            writer.add_scalar(f"{pfx}perf/fps", fps, step * N)
            writer.flush()
            sr = (sum(succ_window) / len(succ_window)) if succ_window else 0.0
            rr = (sum(ret_window) / len(ret_window)) if ret_window else 0.0
            print(f"[train] {label} step={step} env_steps={step*N} buf={len(buffer)} "
                  f"succ={sr:.3f} ret={rr:.2f} fps={fps:.0f} alpha={metrics.get('alpha', 0):.3f} "
                  f"pos_rand={env.pos_rand:.3f}", flush=True)
            if mlflow_log is not None:                 # LOW-cadence mirror to central MLflow (WiFi-friendly)
                try:                                   # ALL reward components -> MLflow (r_reach, r_wrist, r_grasp, ...)
                    rew_mlf = {f"{pfx}reward/{k}": float(v.float().mean()) for k, v in info["components"].items()}
                except Exception:  # noqa: BLE001
                    rew_mlf = {}
                mlflow_log.log_metrics({
                    f"{pfx}train/success_rate": sr, f"{pfx}train/ep_return": rr,
                    f"{pfx}train/ep_len": (sum(len_window) / len(len_window)) if len_window else 0.0,
                    f"{pfx}perf/fps": fps, f"{pfx}perf/buffer": len(buffer),
                    f"{pfx}train/alpha": float(metrics.get("alpha", 0.0)),
                    f"{pfx}train/pos_rand": env.pos_rand,
                    f"{pfx}train/best_success_rate": max(best_eval_sr, 0.0),
                    f"{pfx}reward/mean_step": float(reward.mean()),
                    **rew_mlf,
                }, step * N)
            last_log_t, last_log_step = now, step
            # SPAWN CURRICULUM: once the policy masters the current spawn level (rolling success high),
            # widen the cube-spawn half-range, up to pos_rand_max. Clear the window so it re-measures.
            if cfg.env.pos_rand_max > env.pos_rand and sr >= cfg.env.spawn_succ_thresh \
                    and (step - last_adv_step) >= cfg.env.spawn_cooldown and len(succ_window) >= 50:
                env.pos_rand = min(cfg.env.pos_rand_max, env.pos_rand + cfg.env.pos_rand_step)
                last_adv_step = step
                succ_window.clear(); ret_window.clear()
                print(f"[curriculum] sr>={cfg.env.spawn_succ_thresh} -> widen cube spawn pos_rand={env.pos_rand:.3f} "
                      f"(max {cfg.env.pos_rand_max})", flush=True)
                writer.add_scalar(f"{pfx}curriculum/pos_rand", env.pos_rand, step * N)
            # live dashboard feed (read+relayed by scripts/monitor.py); cheap, throttled to 200 steps
            try:
                terms = {k: float(v.float().mean()) for k, v in info["components"].items()}
            except Exception:  # noqa: BLE001
                terms = {}
            _dump_live_json(cfg, {
                "step": step * N, "phase": label,
                "ep_return": rr, "success_rate": sr, "pos_rand": env.pos_rand,
                "ep_len": (sum(len_window) / len(len_window)) if len_window else 0.0,
                "critic_loss": float(metrics.get("critic_loss", 0.0)),
                "actor_loss": float(metrics.get("actor_loss", 0.0)),
                "alpha": float(metrics.get("alpha", 0.0)),
                "entropy": float(metrics.get("entropy", 0.0)),
                "q_mean": float(metrics.get("q_mean", 0.0)),
                "buffer": len(buffer), "fps": fps,
                "best_success_rate": max(best_eval_sr, 0.0),
                "mean_step_reward": float(reward.mean()),
                "reward_terms": terms,
            })

        if cfg.eval_every and step % cfg.eval_every == 0:
            gstep = step * N
            m_in = evaluate(env, agent, cfg, seed=cfg.eval_seed)        # in-distribution (fixed set)
            for k, v in m_in.items():
                writer.add_scalar(f"{pfx}eval/{k}", v, gstep)
            # OOD validation: re-eval under held-out DR (unseen mass/poses) to expose overfitting
            if cfg.eval_holdout:
                m_ho = evaluate(env, agent, cfg, dr_overrides=cfg.eval_holdout, seed=cfg.eval_seed + 1)
                for k in ("success_rate", "ep_return", "grasp_success", "fall_rate"):
                    writer.add_scalar(f"{pfx}eval_holdout/{k}", m_ho[k], gstep)
                writer.add_scalar(f"{pfx}eval/gen_gap_success",
                                  m_in["success_rate"] - m_ho["success_rate"], gstep)
            # best-checkpoint selection by in-distribution eval success
            best_path = os.path.join(cfg.ckpt_dir, f"{ckpt_name}_best.pt")
            improved = m_in["success_rate"] > best_eval_sr
            if improved:
                best_eval_sr = m_in["success_rate"]
                torch.save(agent.state_dict(), best_path)
            writer.add_scalar(f"{pfx}eval/best_success_rate", best_eval_sr, gstep)
            print(f"[eval] {label} step={step} success={m_in['success_rate']:.3f} "
                  f"return={m_in['ep_return']:.2f} grasp={m_in['grasp_success']:.3f} "
                  f"fall={m_in['fall_rate']:.3f} best={best_eval_sr:.3f}", flush=True)
            if mlflow_log is not None:                 # eval metrics + best ckpt -> central MLflow
                em = {f"{pfx}eval/{k}": v for k, v in m_in.items()}
                if cfg.eval_holdout:
                    for k in ("success_rate", "ep_return", "grasp_success", "fall_rate"):
                        em[f"{pfx}eval_holdout/{k}"] = m_ho[k]
                    em[f"{pfx}eval/gen_gap_success"] = m_in["success_rate"] - m_ho["success_rate"]
                em[f"{pfx}eval/best_success_rate"] = best_eval_sr
                mlflow_log.log_metrics(em, gstep)
                if improved:                           # durable: push the new best to central store
                    mlflow_log.log_artifact(best_path)
            obs = env.reset()

        # refresh _latest.pt often (monitor polls it) without spamming numbered snapshots
        if cfg.latest_every and step % cfg.latest_every == 0:
            torch.save(agent.state_dict(), os.path.join(cfg.ckpt_dir, f"{ckpt_name}_latest.pt"))
        if cfg.ckpt_every and step % cfg.ckpt_every == 0:
            torch.save(agent.state_dict(), os.path.join(cfg.ckpt_dir, f"{ckpt_name}_{step*N}.pt"))
            torch.save(agent.state_dict(), os.path.join(cfg.ckpt_dir, f"{ckpt_name}_latest.pt"))

    final_path = os.path.join(cfg.ckpt_dir, f"{ckpt_name}_final.pt")
    torch.save(agent.state_dict(), final_path)
    writer.flush()
    if mlflow_log is not None:
        mlflow_log.log_artifact(final_path)
    return agent


@torch.no_grad()
def evaluate(env, agent, cfg, dr_overrides=None, seed=None, max_steps=None):
    """Deterministic eval on the training env. Returns a metrics dict:
    success_rate, ep_return, ep_len, grasp_success, fall_rate, n.

    `seed` fixes the env RNG so the eval episode set (cube poses/masses) is reproducible
    across checkpoints. `dr_overrides` temporarily overrides EnvConfig fields (e.g. cube
    mass band / workspace) for an out-of-distribution validation pass; restored on exit.
    """
    max_steps = max_steps or cfg.env.episode_len
    saved = {}
    if dr_overrides:
        for k, v in dr_overrides.items():
            if hasattr(env.cfg, k):
                saved[k] = getattr(env.cfg, k)
                setattr(env.cfg, k, tuple(v) if isinstance(v, list) else v)
            else:
                print(f"[eval] WARN holdout key '{k}' not on EnvConfig; ignored", flush=True)
    if seed is not None:
        torch.manual_seed(seed)
    try:
        obs = env.reset()
        succ, rets, lens, grasped, fell = [], [], [], [], []
        for _ in range(max_steps + 5):
            a = agent.act(obs, deterministic=True)
            obs, _, done, info = env.step(a)
            if info["num_done"] > 0:
                for i in done.nonzero(as_tuple=False).squeeze(-1).tolist():
                    succ.append(float(info["ep_success"][i]))
                    rets.append(float(info["ep_return"][i]))
                    lens.append(float(info["ep_len"][i]))
                    grasped.append(float(info["ep_grasped"][i]))
                    fell.append(float(info["ep_fell"][i]))
            if len(succ) >= cfg.eval_episodes:
                break
    finally:
        for k, v in saved.items():            # restore so rollout DR is never leaked
            setattr(env.cfg, k, v)

    def m(x):
        return (sum(x) / len(x)) if x else 0.0
    return {"success_rate": m(succ), "ep_return": m(rets), "ep_len": m(lens),
            "grasp_success": m(grasped), "fall_rate": m(fell), "n": len(succ)}


def _yaml(d, indent=0):
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out.append("  " * indent + f"{k}:")
            out.append(_yaml(v, indent + 1))
        else:
            out.append("  " * indent + f"{k}: {v}")
    return "\n".join(out)
