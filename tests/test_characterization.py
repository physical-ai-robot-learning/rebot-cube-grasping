"""Characterization (golden) tests recorded on the pre-refactor code (commit 47398fd, 2026-11-05) and kept unchanged
through the 2026-11 OA restyle: pure-Python helpers, the Project 2 networks on the shipped checkpoints, the June
networks and both Isaac-free deploy loops on seeded inputs.

Expected values live in tests/golden/characterization_2026-11.json. Regenerate ONLY on a behaviour-changing commit:
    GOLDEN_RECORD=1 python -m pytest tests/test_characterization.py
"""
import hashlib
import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

R = Path(__file__).resolve().parents[1]
GOLDEN_PATH = R / "tests" / "golden" / "characterization_2026-11.json"
RECORD = os.environ.get("GOLDEN_RECORD") == "1"
ROUND_DIGITS = 5
_recorded = {}


def _golden():
    if GOLDEN_PATH.is_file():
        return json.loads(GOLDEN_PATH.read_text())
    return {}


def _check(key, value):
    if RECORD:
        _recorded[key] = value
        tData = _golden()
        tData[key] = value
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(tData, indent=1, sort_keys=True))
        return
    assert key in _golden(), f"no golden value for {key}"
    assert value == _golden()[key], key


def _r(x):
    """Round a tensor / array / scalar to a JSON-comparable nested list."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.round(np.asarray(x, dtype=np.float64), ROUND_DIGITS).tolist()


def _sha(b):
    return hashlib.sha256(b).hexdigest()


@pytest.fixture(autouse=True)
def _paths(monkeypatch):
    monkeypatch.syspath_prepend(str(R / "src"))


# ---------------------------------------------------------------- Project 2 helpers
def _p2():
    # network shape knobs of the shipped checkpoints (configs/project2_bench_2026-08-03.env)
    os.environ["D_CUBE"] = "4"
    os.environ["D_ACQ"] = "3"
    sys.modules.pop("rebot_grasp_rl.nets", None)
    from rebot_grasp_rl import dashboard_render as ui_render
    from rebot_grasp_rl import nets
    from rebot_grasp_rl import robot_profile
    from rebot_grasp_rl import runtime
    from rebot_grasp_rl import target_projection as tp_net
    return nets, robot_profile, runtime, tp_net, ui_render


def _arrow(ui, img, *args):
    tIm = img.copy()
    ui.drawArrow(tIm, *args)
    return tIm.tobytes()


def test_ui_render_helpers():
    _, _, _, _, ui = _p2()
    tRng = np.random.default_rng(0)
    tImg = (tRng.random((48, 64, 3)) * 255).astype(np.uint8)
    tImg[10:20, 30:44] = (230, 20, 20)
    tOut = {
        "spark": ui.sparkSvg([0.1, 0.5, 0.3, 0.9], "#ff0000"),
        "spark_short": ui.sparkSvg([1.0], "#00ff00", w=100, h=20),
        "card": ui.statCard("RSR", "12.5", sub="x"),
        "trimmean": [ui.trimmedMean(torch.arange(20, dtype=torch.float32)), ui.trimmedMean(torch.tensor([3.0, 1.0])),
                     ui.trimmedMean(torch.zeros(0))],
        "quat_rpy": _r(np.concatenate([np.array(v) for v in ui.quatRpy([0.9, 0.1, -0.3, 0.2])])),
        "draw_det": _sha(ui.drawDetection(tImg.copy(), [0.5, 0.4, 0.1, 1.0]).tobytes()),
        "draw_arrow_long": _sha(_arrow(ui, tImg, 30, 24, 20, -10)),
        "draw_arrow_ring": _sha(_arrow(ui, tImg, 30, 24, 1, 1)),
        "png_len": len(ui.pngBytes(tImg)) > 0,
    }
    _check("ui_render", tOut)


def test_robot_profile_values(monkeypatch):
    import dataclasses
    _, rp, _, _, _ = _p2()
    tOut = {"ours": [repr(v) for v in dataclasses.astuple(rp.OURS)],
            "official": [repr(v) for v in dataclasses.astuple(rp.OFFICIAL)]}
    monkeypatch.setenv("ROBOT_OFFICIAL", "1")
    tOut["select_official"] = rp.selectProfile() is rp.OFFICIAL
    monkeypatch.setenv("ROBOT_OFFICIAL", "0")
    tOut["select_ours"] = rp.selectProfile() is rp.OURS
    _check("robot_profile", tOut)


def test_runtime_defaults():
    _, _, rt, _, _ = _p2()
    tNames = sorted(n for n in dir(type(rt.RT)) if not n.startswith("__"))
    _check("runtime", {"n_fields": len(tNames), "values": [repr(getattr(rt.RT, n)) for n in tNames]})


def test_project_target_and_nets_on_checkpoints():
    nets, _, _, tp, _ = _p2()
    torch.manual_seed(0)
    tN, tState, tAct = 5, 21 + nets.D_CUBE + nets.D_ACQ, 4
    tFd, tWd = torch.rand(tN, nets.D_DET), torch.rand(tN, nets.D_DET)
    tOri, tS = torch.rand(tN, nets.D_ORI), torch.rand(tN, tState)
    tOut = {}
    for tName in ("search_last", "approach_last"):
        tCk = torch.load(R / "results" / "checkpoints" / "project2" / f"{tName}.pt", map_location="cpu", weights_only=False)
        tActDim = tCk["actor"]["mean.weight"].shape[0]
        tStateDim = tCk["actor"]["proj.state_proj.0.weight"].shape[1] + nets.D_CUBE + nets.D_ACQ
        tActor = nets.DetActor(tStateDim, tActDim)
        tActor.load_state_dict(tCk["actor"])
        tCritic = nets.DetCriticC51(tStateDim, tActDim, v_min=-20.0, v_max=400.0)
        tCritic.load_state_dict(tCk["critic"])
        tActor.eval()
        tCritic.eval()
        torch.manual_seed(1)
        tS2 = torch.rand(tN, tStateDim)
        tA = tActor.actDeterministic(tFd, tWd, tOri, tS2)
        tQ = tCritic.qValues(tFd, tWd, tOri, tS2, tA)
        tM = tp.projectTarget(tCritic, tFd, tWd, tOri, tS2, tA, torch.linspace(-1, 3, tN),
                               torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0]), 0.97)
        tOut[tName] = {"act": _r(tA), "q": _r(tQ), "proj_sum": _r(tM.sum(-1)), "proj_first": _r(tM[0, :, :8])}
    del tState, tAct, tS
    _check("project2_nets", tOut)


# ---------------------------------------------------------------- June networks + deploy loops
def test_june_visual_nets_shapes():
    from rebot_teacher_student.student import visual_nets as vn
    torch.manual_seed(0)
    tEnc = vn.CNNEncoder(64)
    tActor = vn.Actor(21, 7)
    tImg = (torch.rand(2, 64, 64, 3) * 255).to(torch.uint8)
    tF = tEnc(tImg)
    tOut = {"enc_shape": list(tF.shape), "enc_sum": round(float(tF.detach().sum()), 3),
            "actor_keys": list(tActor.state_dict().keys())}
    _check("june_visual_nets", tOut)


def _run_deploy(module, ckpt, extra, monkeypatch):
    import importlib
    tMod = importlib.import_module(module)
    monkeypatch.setattr(tMod.time, "sleep", lambda s: None)
    np.random.seed(0)
    torch.manual_seed(0)
    tBuf = io.StringIO()
    tArgv = ["--checkpoint", str(R / ckpt), "--mock", "--max-steps", "12", "--hz", "1", *extra]
    tOld = sys.argv
    sys.argv = ["deploy", *tArgv]
    try:
        with redirect_stdout(tBuf):
            tMod.main()
    finally:
        sys.argv = tOld
    return [l for l in tBuf.getvalue().splitlines() if not l.startswith("[deploy")]


@pytest.mark.parametrize("module,ckpt,extra", [
    ("rebot_teacher_student.deploy.state_deploy", "results/checkpoints/teacher/c51_v5g_final.pt", []),
    ("rebot_teacher_student.deploy.state_deploy", "results/checkpoints/teacher/c51_smooth_v1_final.pt", []),
    ("rebot_teacher_student.deploy.visual_deploy", "results/checkpoints/student/dagger_dual128_v3_final.pt", ["--device", "cpu"]),
])
def test_deploy_loop_golden(module, ckpt, extra, monkeypatch):
    _check(f"deploy::{Path(ckpt).stem}", _run_deploy(module, ckpt, extra, monkeypatch))


def test_deploy_helpers():
    from rebot_teacher_student.deploy import state_deploy as d
    from rebot_teacher_student.deploy import visual_deploy as dv
    tQ = [0.9, 0.1, -0.3, 0.2]
    tTcp = d.tcpFromGripperLink([0.3, -0.2, 0.1], tQ)
    tObs = d.buildObs(np.arange(8) * 0.1, np.arange(8) * -0.01, np.arange(7) * 0.02, [0.4, -0.5, 0.02],
                       [1, 0, 0, 0], tTcp, True)
    tCtrl = d.ActionController()
    tCtrl.reset(d.REST_QPOS, 0.02)
    tSteps = [tCtrl.step(np.array([1, -1, 0.5, 0.2, 0, -0.3, 1.0]), tPan) for tPan in (0.0, 1.6, 1.6)]
    tState = dv.buildState(np.arange(8) * 0.1, np.arange(7) * 0.05)
    tDs = dv.VisualPolicy._ds((np.arange(90 * 160 * 3) % 251).reshape(90, 160, 3).astype(np.uint8), 16)
    tOut = {"tcp": _r(tTcp), "obs": _r(tObs), "steps": [[_r(a), round(g, 6)] for a, g in tSteps],
            "target7": _r(tCtrl.targetQpos7), "state": _r(tState), "ds_sha": _sha(tDs.numpy().tobytes()),
            "quat_rotate": _r(d.quatRotate(tQ, [1.0, 2.0, 3.0])), "pick_device": dv.pickDevice("cpu")}
    _check("deploy_helpers", tOut)
