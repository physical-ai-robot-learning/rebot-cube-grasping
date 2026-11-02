"""Opt-in per-episode output for the fixed eval-set benchmark of train_sac_det_xattn.py (added 2026-11-01).

Everything here is a no-op unless EVAL_OUT_DIR is set, so training and the original benchmark behave exactly as
before. With EVAL_OUT_DIR set (and TP_EVAL=1 + EVAL_SET=...):
  * EVAL_SEED (default 0) seeds python / numpy / torch and the Isaac Lab env config before the env is built;
  * every finished eval-set episode is appended to <EVAL_OUT_DIR>/episodes.csv (one row per episode);
  * when the eval set is fully covered, <EVAL_OUT_DIR>/trainer_summary.json is written (the trainer's own
    aggregate) and, with EVAL_EXIT_WHEN_DONE=1, the process exits instead of idling to keep its web UI alive.
Success definitions are the trainer's own (no change): search = the episode reached the APPROACH phase;
rsr = the deployable success gate held RSR_HOLD frames; rsr_old = the privileged gate (true jaw-mouth to cube
distance < RSR_DCUBE) held RSR_HOLD frames; lift = the rule grasp+lift latch (only meaningful with GRASP_ON=1).
"""
import csv
import json
import os
import random

OUT = os.environ.get("EVAL_OUT_DIR", "")
FIELDS = ["episode", "case", "case_x", "case_y", "case_yaw", "case_wrist_start", "env", "global_step",
          "search", "rsr", "rsr_old", "lift", "seed"]
_n = [0]


def enabled():
    return bool(OUT)


def seed(env_cfg):
    if not OUT:
        return
    import numpy as np
    import torch
    s = int(os.environ.get("EVAL_SEED", "0"))
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    try:
        env_cfg.seed = s
    except Exception:
        pass
    os.makedirs(OUT, exist_ok=True)
    print(f"[EVAL-HOOK] seed={s} out={OUT}", flush=True)


def log_episodes(cases, case_idx, env_idx, search, rsr, rsr_old, lift, global_step):
    """cases: (M,4) tensor; case_idx/env_idx: (k,) long; search/rsr/rsr_old/lift: (k,) bool/float tensors."""
    if not OUT:
        return
    p = os.path.join(OUT, "episodes.csv")
    new = not os.path.exists(p)
    s = int(os.environ.get("EVAL_SEED", "0"))
    rows = []
    ci = case_idx.tolist()
    ei = env_idx.tolist()
    sr, r, ro, lf = (x.float().tolist() for x in (search, rsr, rsr_old, lift))
    cs = cases[case_idx].tolist()
    for j in range(len(ci)):
        c = cs[j] + [0.0] * (4 - len(cs[j]))
        rows.append({"episode": _n[0], "case": ci[j], "case_x": round(c[0], 5), "case_y": round(c[1], 5),
                     "case_yaw": round(c[2], 5), "case_wrist_start": round(c[3], 5), "env": ei[j],
                     "global_step": int(global_step), "search": int(sr[j] > 0.5), "rsr": int(r[j] > 0.5),
                     "rsr_old": int(ro[j] > 0.5), "lift": int(lf[j] > 0.5), "seed": s})
        _n[0] += 1
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def finish(summary):
    if not OUT:
        return
    keys = ("EVAL_SET", "EVAL_SEED", "TP_EVAL_DIR", "RSR_DCUBE", "RSR_HOLD", "EVAL_SUCC_BY_OLD", "GRASP_ON",
            "APP_CEN", "APP_MZ", "FRAC_GRASP", "ROBOT_OFFICIAL")
    summary = dict(summary, env={k: os.environ.get(k) for k in keys})
    with open(os.path.join(OUT, "trainer_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[EVAL-HOOK] wrote {OUT}/trainer_summary.json", flush=True)
    if int(os.environ.get("EVAL_EXIT_WHEN_DONE", "0")):
        os._exit(0)


# ---- opt-in rollout video of env 0 (EVAL_VIDEO_DIR; added 2026-11-02) ------------------------------------------------
# Frames are the trainer's own annotated dashboard images of env 0 (front | wrist, the detector-input view with the
# detection box and direction arrow drawn), one per control step in TP_EVAL. One MP4 per finished env-0 episode is
# written as ep<NN>_case<C>_<outcome>.mp4 (outcome: highest stage reached). No-op unless EVAL_VIDEO_DIR is set.
VIDEO = os.environ.get("EVAL_VIDEO_DIR", "")
VIDEO_MAX = int(os.environ.get("EVAL_VIDEO_MAX", "6"))
_vframes = []
_vcount = [0]
PHASES = {0: "search", 1: "approach", 2: "align", 3: "grasp", 4: "lift"}


def video_frame(front_rgb, wrist_rgb, phase):
    if not VIDEO or _vcount[0] >= VIDEO_MAX:
        return
    import numpy as np
    h = max(front_rgb.shape[0], wrist_rgb.shape[0])

    def pad(a):
        if a.shape[0] == h:
            return a
        out = np.zeros((h, a.shape[1], 3), dtype=a.dtype)
        out[: a.shape[0]] = a
        return out
    _vframes.append((np.concatenate([pad(front_rgb[..., :3]), pad(wrist_rgb[..., :3])], axis=1), int(phase)))


def video_end(case, search, rsr, rsr_old, lift):
    if not VIDEO or _vcount[0] >= VIDEO_MAX:
        _vframes.clear()
        return
    if len(_vframes) < 2:
        _vframes.clear()
        return
    import imageio
    os.makedirs(VIDEO, exist_ok=True)
    outcome = "lift" if lift else ("reach_deployable" if rsr else ("reach_privileged" if rsr_old else ("approach" if search else "search_only")))
    name = f"ep{_vcount[0]:02d}_case{int(case)}_{outcome}.mp4"
    fps = int(os.environ.get("EVAL_VIDEO_FPS", "10"))
    imageio.mimsave(os.path.join(VIDEO, name), [f for f, _ in _vframes], fps=fps, macro_block_size=1)
    with open(os.path.join(VIDEO, "videos.csv"), "a") as f:
        f.write(f"{name},{int(case)},{int(search)},{int(rsr)},{int(rsr_old)},{int(lift)},{len(_vframes)},"
                + "|".join(PHASES.get(p, str(p)) for p in sorted({p for _, p in _vframes})) + "\n")
    print(f"[EVAL-HOOK] wrote video {name} ({len(_vframes)} frames)", flush=True)
    _vcount[0] += 1
    _vframes.clear()
