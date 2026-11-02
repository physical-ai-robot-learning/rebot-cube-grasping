"""Demonstration storage for RLPD-style demo bootstrapping.

A scripted expert (scripts/collect_demos.py) records successful pick transitions; training
loads them into a ReplayBuffer and SAC mixes them into every minibatch (symmetric sampling).
"""
from __future__ import annotations

import torch

from .sac import ReplayBuffer


def save_demos(path: str, obs, act, rew, next_obs, done) -> None:
    """Save demo transitions (each a [K, ...] tensor/array) to a .pt file."""
    torch.save({
        "obs": torch.as_tensor(obs, dtype=torch.float32),
        "act": torch.as_tensor(act, dtype=torch.float32),
        "rew": torch.as_tensor(rew, dtype=torch.float32),
        "next_obs": torch.as_tensor(next_obs, dtype=torch.float32),
        "done": torch.as_tensor(done, dtype=torch.float32),
    }, path)


def load_demo_buffer(path: str, device, expected_obs_dim=None) -> ReplayBuffer:
    """Load saved demos into a (capacity = #demos) ReplayBuffer for symmetric sampling."""
    d = torch.load(path, map_location=device)
    n, obs_dim, act_dim = d["obs"].shape[0], d["obs"].shape[1], d["act"].shape[1]
    assert expected_obs_dim is None or obs_dim == expected_obs_dim, (
        f"demo obs_dim {obs_dim} != env obs_dim {expected_obs_dim} ({path}) -> re-collect demos for this obs config")
    buf = ReplayBuffer(n, obs_dim, act_dim, device)
    buf.add_batch(d["obs"].to(device), d["act"].to(device), d["rew"].to(device),
                  d["next_obs"].to(device), d["done"].to(device))
    print(f"[demo] loaded {n} demo transitions from {path} (obs={obs_dim}, act={act_dim})", flush=True)
    return buf
