"""Minimal, self-contained SAC for continuous control (no Isaac dependency).

tanh-squashed Gaussian actor, twin Q critics + targets, automatic entropy temperature.
Designed for a *vectorized* env: transitions from all N envs are pushed each step into a
GPU tensor replay buffer. LayerNorm is on by default (docs/03 notes it stabilized this
robot's training).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def mlp(sizes, layernorm: bool):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            if layernorm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, layernorm):
        super().__init__()
        self.net = mlp([obs_dim, *hidden], layernorm)
        self.mu = nn.Linear(hidden[-1], act_dim)
        self.log_std = nn.Linear(hidden[-1], act_dim)

    def forward(self, obs):
        h = F.relu(self.net(obs))
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs, deterministic=False, with_logprob=True):
        mu, log_std = self.forward(obs)
        if deterministic:
            return torch.tanh(mu), None
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        x = dist.rsample()
        a = torch.tanh(x)
        logp = None
        if with_logprob:
            # tanh change-of-variables correction
            logp = dist.log_prob(x).sum(-1) - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, logp


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, layernorm):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim, *hidden, 1], layernorm)
        self.q2 = mlp([obs_dim + act_dim, *hidden, 1], layernorm)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class ReplayBuffer:
    """Flat GPU tensor buffer; ingests a whole [N,...] batch of transitions per step."""

    def __init__(self, capacity, obs_dim, act_dim, device):
        self.cap, self.device = capacity, device
        self.obs = torch.zeros(capacity, obs_dim, device=device)
        self.next_obs = torch.zeros(capacity, obs_dim, device=device)
        self.act = torch.zeros(capacity, act_dim, device=device)
        self.rew = torch.zeros(capacity, device=device)
        self.done = torch.zeros(capacity, device=device)
        self.ptr = 0
        self.full = False

    def add_batch(self, obs, act, rew, next_obs, done):
        n = obs.shape[0]
        idx = (torch.arange(n, device=self.device) + self.ptr) % self.cap
        self.obs[idx] = obs
        self.act[idx] = act
        self.rew[idx] = rew
        self.next_obs[idx] = next_obs
        self.done[idx] = done.float()
        self.ptr = (self.ptr + n) % self.cap
        if self.ptr < n:
            self.full = True

    def __len__(self):
        return self.cap if self.full else self.ptr

    def sample(self, batch_size):
        hi = len(self)
        idx = torch.randint(0, hi, (batch_size,), device=self.device)
        return self.obs[idx], self.act[idx], self.rew[idx], self.next_obs[idx], self.done[idx]


class SAC:
    def __init__(self, obs_dim, act_dim, cfg, device):
        self.device = device
        self.gamma, self.tau = cfg.gamma, cfg.tau
        self.act_dim = act_dim

        self.actor = Actor(obs_dim, act_dim, cfg.hidden, cfg.layernorm).to(device)
        self.critic = Critic(obs_dim, act_dim, cfg.hidden, cfg.layernorm).to(device)
        self.critic_targ = Critic(obs_dim, act_dim, cfg.hidden, cfg.layernorm).to(device)
        self.critic_targ.load_state_dict(self.critic.state_dict())
        for p in self.critic_targ.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

        self.target_entropy = -act_dim * cfg.target_entropy_scale
        self.min_alpha = getattr(cfg, "min_alpha", 0.0)
        self.log_alpha = torch.tensor(
            float(torch.log(torch.tensor(cfg.init_alpha))), requires_grad=True, device=device
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr)

    @property
    def alpha(self):
        # floor keeps a minimum exploration temperature (helps discover grasp/lift)
        return self.log_alpha.exp().clamp(min=self.min_alpha)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        a, _ = self.actor.sample(obs, deterministic=deterministic, with_logprob=False)
        return a

    def update(self, buffer: ReplayBuffer, batch_size, demo_buffer=None, demo_frac=0.5) -> dict:
        # RLPD-style symmetric sampling: half the batch from demos, half from online replay.
        if demo_buffer is not None and len(demo_buffer) > 0:
            n_d = int(batch_size * demo_frac)
            n_o = batch_size - n_d
            od, ad, rd, nod, dd = demo_buffer.sample(n_d)
            oo, ao, ro, noo, do = buffer.sample(n_o)
            obs = torch.cat([od, oo]); act = torch.cat([ad, ao]); rew = torch.cat([rd, ro])
            next_obs = torch.cat([nod, noo]); done = torch.cat([dd, do])
        else:
            obs, act, rew, next_obs, done = buffer.sample(batch_size)

        # --- critic ---
        with torch.no_grad():
            na, nlogp = self.actor.sample(next_obs)
            tq1, tq2 = self.critic_targ(next_obs, na)
            tq = torch.min(tq1, tq2) - self.alpha * nlogp
            target = rew + self.gamma * (1.0 - done) * tq
        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # --- actor ---
        for p in self.critic.parameters():
            p.requires_grad_(False)
        a, logp = self.actor.sample(obs)
        qa1, qa2 = self.critic(obs, a)
        qa = torch.min(qa1, qa2)
        actor_loss = (self.alpha.detach() * logp - qa).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        # --- temperature ---
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        # --- target soft update ---
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_targ.parameters()):
                tp.mul_(1 - self.tau).add_(self.tau * p)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
            "q_mean": q1.mean().item(),
            "entropy": -logp.mean().item(),
        }

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "log_alpha": self.log_alpha.detach(),
        }

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        with torch.no_grad():
            self.log_alpha.copy_(sd["log_alpha"])
        self.critic_targ.load_state_dict(self.critic.state_dict())
