"""Agent + replay-buffer factories shared by train / monitor / eval / deploy.

Centralizes the ``cfg.algo`` ("sac"|"resfit") and n-step buffer selection so every consumer
builds the SAME agent class for a given config/checkpoint (docs/06 §5). Importing this pulls in
torch — import it only AFTER SimulationApp is created (in the Isaac venv) or in the deploy venv.
"""
from __future__ import annotations


def make_agent(algo, obs_dim, act_dim, sac_cfg, device, critic_obs_dim=None):
    """Build the policy agent for ``algo`` ('sac' | 'resfit'). ``critic_obs_dim`` (obs + privileged
    block) enables the asymmetric critic; None/==obs_dim is symmetric (old behavior)."""
    algo = (algo or "sac").lower()
    if algo == "resfit":
        from .resfit import ResfitResidualAgent
        return ResfitResidualAgent(obs_dim, act_dim, sac_cfg, device, critic_obs_dim=critic_obs_dim)
    if algo == "sac":
        from .sac import SAC
        return SAC(obs_dim, act_dim, sac_cfg, device, critic_obs_dim=critic_obs_dim)
    raise ValueError(f"unknown algo '{algo}' (expected 'sac' or 'resfit')")


def make_buffer(sac_cfg, obs_dim, act_dim, device, priv_dim: int = 0):
    """Online replay buffer: NStepReplayBuffer when n_step>1, else the flat ReplayBuffer.
    Both emit the canonical 8-tuple, so the agent consumes either identically."""
    n_step = int(getattr(sac_cfg, "n_step", 1))
    if n_step > 1:
        from .nstep_buffer import NStepReplayBuffer
        return NStepReplayBuffer(sac_cfg.replay_size, obs_dim, act_dim, n_step,
                                 sac_cfg.gamma, device, priv_dim=priv_dim)
    from .sac import ReplayBuffer
    return ReplayBuffer(sac_cfg.replay_size, obs_dim, act_dim, device,
                        priv_dim=priv_dim, gamma=sac_cfg.gamma)
