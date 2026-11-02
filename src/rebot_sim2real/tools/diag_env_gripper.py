"""End-to-end: in the ACTUAL lift_cube env, is gripper_action a MirroredGripperAction with
_mirror_ids set, and do BOTH fingers move on a close command?"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); app = AppLauncher(args).app

import torch  # noqa
from isaaclab.envs import ManagerBasedRLEnv  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa
from isaaclab.managers import SceneEntityCfg  # noqa
from isaaclab.managers import RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm  # noqa
import leisaac.tasks.lift_cube  # noqa
from leisaac.tasks.lift_cube import mdp  # noqa
from rebot_sim2real.env.env_config import configure_env  # noqa

out = []
def w(s): out.append(str(s)); print(s, flush=True)

from isaaclab.sensors import ContactSensorCfg  # noqa
env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device="cpu", num_envs=2)
env_cfg = configure_env(env_cfg, {}, mdp, RewTerm, DoneTerm, SceneEntityCfg, target_cube="cube")
# contact sensors (obs references them) -- replicate train_sac
_t = env_cfg.scene.cube
env_cfg.scene.gripper_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_left",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_t.prim_path])
env_cfg.scene.jaw_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_right",
    update_period=0.0, history_length=4, filter_prim_paths_expr=[_t.prim_path])
_t.spawn.activate_contact_sensors = True
env_cfg.scene.robot.spawn.activate_contact_sensors = True
# strip cameras not needed for this check
for cam in ("front", "iso", "wrist"):
    if hasattr(env_cfg.scene, cam):
        setattr(env_cfg.scene, cam, None)

env = ManagerBasedRLEnv(cfg=env_cfg)
env.reset()
robot = env.scene["robot"]
gi = robot.joint_names.index("gripper")
g2 = robot.joint_names.index("gripper_joint2")

term = env.action_manager.get_term("gripper_action")
w("gripper_action term TYPE: %s" % type(term).__name__)
w("  primary _joint_ids=%s   _mirror_ids=%s" % (getattr(term, "_joint_ids", "MISSING"), getattr(term, "_mirror_ids", "MISSING")))
w("  mirror names=%s" % (getattr(term, "_mirror_names", "MISSING"),))

adim = env.action_manager.total_action_dim
w("total action dim=%d (expect 7: 6 arm + 1 gripper); gripper is last" % adim)
bl = robot.body_names.index("gripper_left")
br = robot.body_names.index("gripper_right")

def settle(gv, n=40):
    act = torch.zeros((env.num_envs, adim)); act[:, -1] = gv
    for _ in range(n):
        env.step(act)

def report(tag):
    lp = robot.data.joint_pos[0, gi].item(); rp = robot.data.joint_pos[0, g2].item()
    Lpos = robot.data.body_pos_w[0, bl]; Rpos = robot.data.body_pos_w[0, br]
    gap = (Lpos - Rpos).norm().item()
    w("  [%s] joints g=%.4f g2=%.4f | gripper_left_w=%s gripper_right_w=%s | finger gap=%.4f" %
      (tag, lp, rp, [round(x, 4) for x in Lpos.tolist()], [round(x, 4) for x in Rpos.tolist()], gap))
    return Lpos.clone(), Rpos.clone(), gap

w("--- OPEN (-1) then CLOSE (+1): do BOTH finger BODIES move & converge in world? ---")
settle(-1.0); Lo, Ro, gap_open = report("OPEN")
settle(+1.0); Lc, Rc, gap_close = report("CLOSE")
w("  gripper_left  world travel = %.4f m" % (Lc - Lo).norm().item())
w("  gripper_right world travel = %.4f m" % (Rc - Ro).norm().item())
w("  finger gap: open=%.4f -> close=%.4f  (should DECREASE if converging to center)" % (gap_open, gap_close))
w("VERDICT: both travels > 0 and gap shrinks -> symmetric converging grip (correct).")

with open("/tmp/diag_env_gripper.txt", "w") as f:
    f.write("\n".join(out) + "\n")
import os as _os; _os._exit(0)
