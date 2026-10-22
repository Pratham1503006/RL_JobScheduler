"""Test that the DoubleDQN subclass trains without error (plan 7.2)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.monitor import Monitor  # noqa: E402

from env.scheduling_env import CloudSchedulingEnv  # noqa: E402


def test_double_dqn_trains_1000_steps():
    from agents.double_dqn import DoubleDQN
    model = DoubleDQN(
        "MlpPolicy", Monitor(CloudSchedulingEnv(mode="train")),
        device="cpu", seed=0, learning_rate=1e-4, buffer_size=10_000,
        learning_starts=100, batch_size=32, train_freq=4, gradient_steps=1,
        target_update_interval=100, exploration_fraction=0.2,
        exploration_final_eps=0.05, policy_kwargs=dict(net_arch=[32, 32]),
        verbose=0)
    model.learn(total_timesteps=1000)
    assert model._n_updates > 0
