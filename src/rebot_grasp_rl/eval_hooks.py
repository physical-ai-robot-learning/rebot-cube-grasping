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


def seed(envCfg):
    if not OUT:
        return
    import numpy as np
    import torch
    tS = int(os.environ.get("EVAL_SEED", "0"))
    random.seed(tS)
    np.random.seed(tS)
    torch.manual_seed(tS)
    try:
        envCfg.seed = tS
    except Exception:
        pass
    os.makedirs(OUT, exist_ok=True)
    print(f"[EVAL-HOOK] seed={tS} out={OUT}", flush=True)


def logEpisodes(cases, caseIdx, envIdx, search, rsr, rsrOld, lift, globalStep):
    """cases: (M,4) tensor; case_idx/env_idx: (k,) long; search/rsr/rsr_old/lift: (k,) bool/float tensors."""
    if not OUT:
        return
    tP = os.path.join(OUT, "episodes.csv")
    tNew = not os.path.exists(tP)
    tS = int(os.environ.get("EVAL_SEED", "0"))
    tRows = []
    tCi = caseIdx.tolist()
    tEi = envIdx.tolist()
    tSr, tR, tRo, tLf = (tX.float().tolist() for tX in (search, rsr, rsrOld, lift))
    tCs = cases[caseIdx].tolist()
    for tJ in range(len(tCi)):
        tC = tCs[tJ] + [0.0] * (4 - len(tCs[tJ]))
        tRows.append({"episode": _n[0], "case": tCi[tJ], "case_x": round(tC[0], 5), "case_y": round(tC[1], 5),
                     "case_yaw": round(tC[2], 5), "case_wrist_start": round(tC[3], 5), "env": tEi[tJ],
                     "global_step": int(globalStep), "search": int(tSr[tJ] > 0.5), "rsr": int(tR[tJ] > 0.5),
                     "rsr_old": int(tRo[tJ] > 0.5), "lift": int(tLf[tJ] > 0.5), "seed": tS})
        _n[0] += 1
    with open(tP, "a", newline="") as tF:
        tW = csv.DictWriter(tF, fieldnames=FIELDS)
        if tNew:
            tW.writeheader()
        tW.writerows(tRows)


def finish(summary):
    if not OUT:
        return
    tKeys = ("EVAL_SET", "EVAL_SEED", "TP_EVAL_DIR", "RSR_DCUBE", "RSR_HOLD", "EVAL_SUCC_BY_OLD", "GRASP_ON",
            "APP_CEN", "APP_MZ", "FRAC_GRASP", "ROBOT_OFFICIAL")
    summary = dict(summary, env={tK: os.environ.get(tK) for tK in tKeys})
    with open(os.path.join(OUT, "trainer_summary.json"), "w") as tF:
        json.dump(summary, tF, indent=1)
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


def videoFrame(frontRgb, wristRgb, phase):
    if not VIDEO or _vcount[0] >= VIDEO_MAX:
        return
    import numpy as np
    tH = max(frontRgb.shape[0], wristRgb.shape[0])

    def pad(a):
        if a.shape[0] == tH:
            return a
        tOut = np.zeros((tH, a.shape[1], 3), dtype=a.dtype)
        tOut[: a.shape[0]] = a
        return tOut
    _vframes.append((np.concatenate([pad(frontRgb[..., :3]), pad(wristRgb[..., :3])], axis=1), int(phase)))


def videoEnd(case, search, rsr, rsrOld, lift):
    if not VIDEO or _vcount[0] >= VIDEO_MAX:
        _vframes.clear()
        return
    if len(_vframes) < 2:
        _vframes.clear()
        return
    import imageio
    os.makedirs(VIDEO, exist_ok=True)
    tOutcome = "lift" if lift else ("reach_deployable" if rsr else ("reach_privileged" if rsrOld else ("approach" if search else "search_only")))
    tName = f"ep{_vcount[0]:02d}_case{int(case)}_{tOutcome}.mp4"
    tFps = int(os.environ.get("EVAL_VIDEO_FPS", "10"))
    imageio.mimsave(os.path.join(VIDEO, tName), [tF for tF, _ in _vframes], fps=tFps, macro_block_size=1)
    with open(os.path.join(VIDEO, "videos.csv"), "a") as tF:
        tF.write(f"{tName},{int(case)},{int(search)},{int(rsr)},{int(rsrOld)},{int(lift)},{len(_vframes)},"
                + "|".join(PHASES.get(tP, str(tP)) for tP in sorted({tP for _, tP in _vframes})) + "\n")
    print(f"[EVAL-HOOK] wrote video {tName} ({len(_vframes)} frames)", flush=True)
    _vcount[0] += 1
    _vframes.clear()
