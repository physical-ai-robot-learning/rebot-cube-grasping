"""Isaac-free deploy loops run end to end against the mock robots with the shipped checkpoints (needs torch)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
R = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, PYTHONPATH=str(R / "src"))


@pytest.mark.parametrize("mod,ckpt,extra", [
    ("rebot_teacher_student.deploy.state_deploy", "results/checkpoints/teacher/c51_v5g_final.pt", []),
    ("rebot_teacher_student.deploy.state_deploy", "results/checkpoints/teacher/c51_smooth_v1_final.pt", []),
    ("rebot_teacher_student.deploy.visual_deploy", "results/checkpoints/student/dagger_dual128_v3_final.pt", ["--device", "cpu"]),
])
def test_mock_deploy(mod, ckpt, extra):
    r = subprocess.run([sys.executable, "-m", mod, "--checkpoint", ckpt, "--mock", "--max-steps", "12", "--hz", "1000",
                        *extra], cwd=R, env=ENV, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "MOCK mode" in r.stdout and "[t=0]" in r.stdout


def test_deploy_constants_match_env_source():
    """The deploy controller repeats the env's action limits; check the literal values agree with the sim code."""
    dep = (R / "src/rebot_teacher_student/deploy/state_deploy.py").read_text()
    env = (R / "src/rebot_teacher_student/env/env_config.py").read_text()
    for lit in ["0.05", "0.0143", "0.045", "1.54", "(-0.086, 0.0, 0.103)"]:
        assert lit in env, lit
    for lit in ["ARM_SCALE = 0.05", "GRIP_SCALE = 0.0143", "GRIP_LO, GRIP_HI = 0.0, 0.045", "PAN_FORWARD = 1.54"]:
        assert lit in dep, lit
