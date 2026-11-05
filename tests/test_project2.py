"""Project 2 checks that run without Isaac Sim: log parser, README claims, trainer argparse, eval writer + summary,
re-evaluation script (dry run) and the recorded configurations."""
import csv
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

R = Path(__file__).resolve().parents[1]


def run(*args, env=None):
    return subprocess.run([sys.executable, *args], cwd=R, capture_output=True, text=True, env=env, timeout=300)


def test_parse_logs_matches_committed_summary():
    r = run("analysis/parse_logs.py", "--check")
    assert r.returncode == 0, r.stdout + r.stderr


def test_reported_bench_facts():
    s = json.loads((R / "results/project2/log_summary.json").read_text())["bench"]
    assert s["n_benches"] == 120 and s["n_rsr_zero"] == 120          # deployable gate 0.0 % in every bench
    assert s["last"]["approach_step_k"] == 3712 and s["last"]["cells"] == 86
    assert s["search_pct"] == [92.1, 99.4] and s["rsr_dcube_pct"] == [84.0, 95.2]


def test_verify_claims_passes():
    r = run("analysis/verify_claims.py")
    assert r.returncode == 0, r.stdout[-3000:]


def test_trainer_argparse_without_isaac():
    env = dict(os.environ, PYTHONPATH=str(R / "src"))
    r = subprocess.run([sys.executable, "-c", "import sys; sys.argv=['x','--num-envs','32','--det-conf','0.15'];"
                        "from rebot_grasp_rl import config as t; print(t.args.num_envs, t.args.num_atoms, t.args.det_conf)"],
                       cwd=R, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["32", "101", "0.15"]


def test_configs_have_bench_criteria():
    kv = dict(l.split("=", 1) for l in (R / "configs/project2_bench_2026-08-03.env").read_text().splitlines()
              if l and not l.startswith("#"))
    assert kv["TP_EVAL"] == "1" and kv["GRASP_ON"] == "0" and kv["NO_PRIV"] == "1"
    assert kv["RSR_DCUBE"] == "0.07" and kv["RSR_HOLD"] == "1" and kv["EVAL_SUCC_BY_OLD"] == "1"
    for f in ("project2_bench_2026-08-03.env", "project2_train_9t_dacq_2026-08-02.env"):
        txt = (R / "configs" / f).read_text()
        assert "/home/" not in txt and "EVAL_SET=" not in txt


def test_eval_hooks_writer_and_summary(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("EVAL_OUT_DIR", str(tmp_path))
    monkeypatch.setenv("EVAL_SEED", "7")
    sys.path.insert(0, str(R / "src"))
    sys.modules.pop("rebot_grasp_rl.eval_hooks", None)
    eval_hooks = importlib.import_module("rebot_grasp_rl.eval_hooks")   # re-read the EVAL_* variables
    assert eval_hooks.enabled()

    class Cfg:
        seed = None
    cfg = Cfg()
    eval_hooks.seed(cfg)
    assert cfg.seed == 7
    cases = torch.tensor([[0.1, -0.5, 0.2, 0.0], [0.3, -0.6, -0.1, 0.0], [0.5, -0.7, 0.0, 0.0]])
    eval_hooks.log_episodes(cases, torch.tensor([0, 2]), torch.tensor([4, 5]), torch.tensor([True, True]),
                            torch.tensor([False, False]), torch.tensor([True, False]), torch.tensor([False, False]), 640)
    eval_hooks.log_episodes(cases, torch.tensor([1]), torch.tensor([0]), torch.tensor([False]),
                            torch.tensor([False]), torch.tensor([False]), torch.tensor([False]), 960)
    monkeypatch.setenv("EVAL_EXIT_WHEN_DONE", "0")
    eval_hooks.finish({"rsr_pct": 0.0, "rsr_old_pct": 33.3, "search_pct": 66.7, "covered": 3, "M": 3})
    rows = list(csv.DictReader(open(tmp_path / "episodes.csv")))
    assert [r["case"] for r in rows] == ["0", "2", "1"] and rows[0]["seed"] == "7"
    assert [r["search"] for r in rows] == ["1", "1", "0"] and [r["rsr_old"] for r in rows] == ["1", "0", "0"]
    sys.path.insert(0, str(R / "analysis"))
    import project2_eval_summary as S
    out = S.summarise(tmp_path)
    assert out["episodes"] == 3 and out["search"]["successes"] == 2 and out["rsr"]["successes"] == 0
    lo, hi = out["search"]["wilson95"]
    assert 0.0 < lo < 2 / 3 < hi < 1.0
    assert out["trainer_summary"]["rsr_old_pct"] == 33.3


def test_wilson_reference_values():
    sys.path.insert(0, str(R / "analysis"))
    import project2_eval_summary as S
    assert S.wilson(0, 10) == [0.0, 0.2775]
    assert S.wilson(10, 10) == [0.7225, 1.0]


def test_eval_script_dry_run(tmp_path):
    env = dict(os.environ, DRY_RUN="1", OUT=str(tmp_path), SEED="3")
    r = subprocess.run(["bash", "scripts/eval/project2_eval.sh", "A", "B", "C"], cwd=R, capture_output=True,
                       text=True, env=env)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    assert len(lines) == 3
    for l in lines:
        assert "TP_EVAL=1" in l and "EVAL_OUT_DIR=" in l and "EVAL_SEED=3" in l and "EVAL_EXIT_WHEN_DONE=1" in l
        assert "results/checkpoints/project2" in l and "-m rebot_grasp_rl.train" in l
    assert "evalgrid_86cells" in lines[0] and "eval_set_400cells" in lines[1] and "GRASP_ON=1" in lines[2]


def test_video_hook_writes_episode_clip(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    pytest.importorskip("imageio_ffmpeg")
    monkeypatch.setenv("EVAL_VIDEO_DIR", str(tmp_path))
    sys.path.insert(0, str(R / "src"))
    sys.modules.pop("rebot_grasp_rl.eval_hooks", None)
    eval_hooks = importlib.import_module("rebot_grasp_rl.eval_hooks")   # re-read the EVAL_* variables
    for i in range(6):
        eval_hooks.video_frame(np.full((64, 64, 3), 10 * i, np.uint8), np.zeros((64, 64, 3), np.uint8), 0 if i < 3 else 1)
    eval_hooks.video_end(36, True, False, True, False)
    assert (tmp_path / "ep00_case36_reach_privileged.mp4").stat().st_size > 0
    assert (tmp_path / "videos.csv").read_text().startswith("ep00_case36_reach_privileged.mp4,36,1,0,1,0,6,search|approach")


def test_render_run_dry(tmp_path):
    env = dict(os.environ, DRY_RUN="1", OUT=str(tmp_path))
    r = subprocess.run(["bash", "scripts/eval/project2_eval.sh", "R"], cwd=R, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "--num-envs 1 " in r.stdout and "EVAL_VIDEO_DIR=" in r.stdout and "render_cases_4_from_86cells.npy" in r.stdout


@pytest.mark.parametrize("run", ["bench86", "grid400", "staged_e2e"])
def test_reeval_summary_recomputes_from_episodes(run, tmp_path):
    import shutil
    sys.path.insert(0, str(R / "analysis"))
    import project2_eval_summary as S
    src = R / "results/project2/reeval" / run
    shutil.copy(src / "episodes.csv", tmp_path / "episodes.csv")
    shutil.copy(src / "trainer_summary.json", tmp_path / "trainer_summary.json")
    new = S.summarise(tmp_path)
    old = json.loads((src / "summary.json").read_text())
    for c in ("search", "rsr_old", "rsr", "lift"):
        assert new[c] == old[c], (run, c)
    assert new["episodes"] == old["episodes"] and new["seeds"] == [0]
    assert old[c] is not None and old["rsr"]["successes"] == 0      # deployable gate never met
