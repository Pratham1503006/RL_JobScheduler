"""callbacks.py -- Q-value logging + episode-reward logging (plan 7.5).

QOverestimationCallback:
  before training, roll a random policy to collect a FIXED set of 500
  observations (fixed seed). Every Q_LOG_INTERVAL timesteps, compute
  mean(max_a Q(s, a)) over that set and append to a CSV. This gives the
  DQN-vs-DoubleDQN overestimation plot for Phase 6 with zero training noise
  in the measurement itself (same observations every time).

EpisodeRewardCallback:
  append (timesteps, episode_reward) rows so Phase 6 can plot training
  curves without tensorboard.
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (Q_LOG_INTERVAL, Q_OBS_POOL, RESULTS_DIR)  # noqa: E402

from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

TRAIN_LOGS = os.path.join(RESULTS_DIR, "train_logs")


class QOverestimationCallback(BaseCallback):
    def __init__(self, log_name: str):
        super().__init__()
        self.log_name = log_name
        self.obs_pool = None
        self.rows = []

    @property
    def has_q_net(self) -> bool:
        # Only value-based algos (DQN / DoubleDQN) expose q_net; on-policy
        # policies are logged by EpisodeRewardCallback instead.
        return hasattr(self.model, "q_net")

    def _on_training_start(self) -> None:
        # Fixed, fully-seeded observation set for ZERO measurement noise:
        # instead of collecting with an unseeded random policy (SB3's
        # action_space.sample() is not seedable per call), sample from the
        # observation Box bounds, which stays within the physical [0,1] range.
        rng = np.random.default_rng(0)
        lo = self.model.observation_space.low
        hi = self.model.observation_space.high
        self.obs_pool = rng.uniform(low=lo, high=hi,
                                    size=(Q_OBS_POOL, len(lo))).astype(np.float32)
        self.rows = []

    def _on_step(self) -> bool:
        if self.has_q_net and self.num_timesteps % Q_LOG_INTERVAL == 0:
            with th_no_grad():
                q = self.model.q_net(
                    _th().as_tensor(self.obs_pool)).max(dim=1).values
                mean_max_q = float(q.mean().item())
            self.rows.append((self.num_timesteps, mean_max_q))
        return True

    def _on_training_end(self) -> None:
        os.makedirs(TRAIN_LOGS, exist_ok=True)
        path = os.path.join(TRAIN_LOGS, self.log_name)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timesteps", "mean_max_q"])
            w.writerows(self.rows)


class th_no_grad:
    """torch.no_grad with a lazy torch import (keeps heuristic-only runs
    torch-free)."""

    def __enter__(self):
        import torch
        self.ctx = torch.no_grad()
        self.ctx.__enter__()

    def __exit__(self, *exc):
        self.ctx.__exit__(*exc)


def _th():
    """Lazy torch import helper."""
    import torch
    return torch


class EpisodeRewardCallback(BaseCallback):
    def __init__(self, log_name: str):
        super().__init__()
        self.log_name = log_name
        self.rows = []
        self._ep_reward = 0.0

    def _on_step(self) -> bool:
        self._ep_reward += float(self.locals.get("rewards", [0])[0])
        dones = self.locals.get("dones", [False])
        if dones[0]:
            self.rows.append((self.num_timesteps, self._ep_reward))
            self._ep_reward = 0.0
        return True

    def _on_training_end(self) -> None:
        os.makedirs(TRAIN_LOGS, exist_ok=True)
        path = os.path.join(TRAIN_LOGS, self.log_name)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timesteps", "episode_reward"])
            w.writerows(self.rows)
