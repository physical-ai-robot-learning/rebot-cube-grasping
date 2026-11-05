"""Custom action: delta accumulated on previous target (ManiSkill use_target=True)."""
import torch
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass


class TargetDeltaJointPositionAction(JointAction):
    """Delta added to PREVIOUS TARGET, not current qpos.
    
    ManiSkill pd_joint_target_delta_pos equivalent:
        target = previous_target + clip(action, -1, 1) * scale
    """
    
    cfg: "TargetDeltaJointPositionActionCfg"
    
    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if cfg.use_zero_offset:
            self._offset = 0.0
        self._target_qpos = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        # Optional per-joint HARD limit on the accumulated target (order = joint_names order).
        # Used to keep shoulder_pan in the front 180deg and stop the wrist from over-rotating
        # (real-servo / cable safety). None = no extra clamp (physics joint limits still apply).
        self._tlo = self._thi = None
        if getattr(cfg, "target_lo", None) is not None and getattr(cfg, "target_hi", None) is not None:
            tDev = self._target_qpos.device
            self._tlo = torch.tensor(cfg.target_lo, device=tDev, dtype=self._target_qpos.dtype)
            self._thi = torch.tensor(cfg.target_hi, device=tDev, dtype=self._target_qpos.dtype)
        # Coupled limit: gate elbow EXTENSION on the shoulder_pan facing `couple_object`. Until the pan
        # points at the cube, the elbow is held bent (<= couple_elbow_cap) so the arm cannot straighten
        # out and extend backward; once the pan faces the cube, the elbow is free to reach forward.
        self._couple = getattr(cfg, "couple_object", None)
        if self._couple is not None:
            self._elbow_local = cfg.couple_elbow_local
            self._pan_jid = self._joint_ids[cfg.couple_pan_local]
            self._pan_forward = cfg.couple_pan_forward   # elbow free ONLY when pan >= this (rad, ~88deg)
            # when pan < forward, elbow held in this COMPACT tuck range (NOT straight, NOT folded-back)
            self._elbow_lo = cfg.couple_elbow_lo         # most-bent allowed (avoid over-fold to -3.14)
            self._elbow_hi = cfg.couple_elbow_hi         # least-bent allowed (avoid straightening out)

    def reset(self, env_ids):
        self._target_qpos[env_ids] = self._asset.data.joint_pos[env_ids][:, self._joint_ids]

    def apply_actions(self):
        # ManiSkill exact: clamp RAW action to [-1,1] THEN scale
        tRawClamped = self.raw_actions.clamp(-1.0, 1.0)
        tDelta = tRawClamped * self._scale
        # Accumulate on previous target
        self._target_qpos = self._target_qpos + tDelta
        # HARD per-joint restriction (pan front-180, wrist anti-overspin) on the COMMANDED target.
        if self._tlo is not None:
            self._target_qpos = self._target_qpos.clamp(self._tlo, self._thi)
        # COUPLED: elbow may extend ONLY once shoulder_pan has swung forward past couple_pan_forward
        # (~90deg). Below that the elbow is folded (<= couple_elbow_cap), so the arm cannot straighten
        # out and extend backward; once pan is at the front edge, the elbow is free to reach.
        if self._couple is not None:
            tPan = self._asset.data.joint_pos[:, self._pan_jid]           # actual shoulder_pan
            tForward = tPan >= self._pan_forward
            # pan forward enough -> elbow free over its full limit; else HOLD elbow in a compact tuck
            # range [elbow_lo, elbow_hi] (BOTH sides) so it neither straightens out nor folds back to
            # -3.14. Clamping both sides makes the elbow fold naturally into the tuck when pan swings back.
            tLo = torch.where(tForward, torch.full_like(tPan, -3.1416), torch.full_like(tPan, self._elbow_lo))
            tHi = torch.where(tForward, torch.zeros_like(tPan),         torch.full_like(tPan, self._elbow_hi))
            self._target_qpos[:, self._elbow_local] = self._target_qpos[:, self._elbow_local].clamp(tLo, tHi)
        # Set as position target (PD controller tracks this)
        self._asset.set_joint_position_target(self._target_qpos, joint_ids=self._joint_ids)


@configclass
class TargetDeltaJointPositionActionCfg(JointActionCfg):
    """Config for TargetDeltaJointPositionAction."""
    class_type = TargetDeltaJointPositionAction
    use_zero_offset: bool = True
    target_lo: list = None   # per-joint min for the accumulated target (joint_names order); None=off
    target_hi: list = None   # per-joint max
    # Coupled elbow gate: elbow may extend only once shoulder_pan faces `couple_object`. None=off.
    couple_object: str = None
    couple_pan_local: int = 0       # local idx of shoulder_pan within joint_names
    couple_elbow_local: int = 2     # local idx of elbow_flex within joint_names
    couple_pan_forward: float = 1.54  # elbow free ONLY when pan >= this (rad; ~88deg, just under the +90 cap)
    couple_elbow_lo: float = -1.4     # when pan < forward, most-bent elbow_flex allowed (no fold-back to -3.14)
    couple_elbow_hi: float = -0.7     # when pan < forward, least-bent allowed (no straightening out)
