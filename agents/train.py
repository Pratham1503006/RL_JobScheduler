"""train.py -- one CLI for all four RL algorithms (plan 7).

Usage:
  python -m agents.train --algo dqn --seed 0 [--timesteps 10000]

All hyperparameters come from config.py. WHY [128,128] net for all four:
the plan's fairness requirement — same network size for a fair comparison.
"""

import argparse
import json
import os
import sys

import numpy as np
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (A2C_TIMESTEPS, DQN_TIMESTEPS, MODELS_DIR, PPO_TIMESTEPS,
                    RESULTS_DIR)  # noqa: E402

from agents.callbacks import (EpisodeRewardCallback,                      # noqa: E402
                              QOverestimationCallback)
from agents.double_dqn import DoubleDQN  # noqa: E402
from env.scheduling_env import CloudSchedulingEnv  # noqa: E402

TRAIN_LOGS = os.path.join(RESULTS_DIR, "train_logs")


def make_env() -> Monitor:
    return Monitor(CloudSchedulingEnv(mode="train"))


def build_model(algo: str, seed: int, total_timesteps: int):
    """Hyperparameters exactly as in plan sections 7.1-7.4."""
    policy_kwargs = dict(net_arch=[128, 128])
    common = dict(policy="MlpPolicy", env=make_env(), device="cpu", seed=seed,
                  policy_kwargs=policy_kwargs, verbose=0)
    if algo == "dqn" or algo == "ddqn":
        # DQN Plan-B tuning (2026-10-12): with the plan's vanilla 1e-4 LR and
        # batch 64 Huber loss, DQN seed0 collapsed after retraining with the
        # raised W_INVALID=20 (reward diverged, mean-max-Q → -1565; the agent
        # chose WAIT-only / invalid spirals). Tighter gradient clip + lower LR
        # (Plan-B tuning per plan 7.6) stabilizes it.
        dqn_kwargs = dict(
            learning_rate=3e-5, buffer_size=50_000, learning_starts=1_000,
            batch_size=64, gamma=0.99, train_freq=4, gradient_steps=1,
            target_update_interval=500, exploration_fraction=0.3,
            exploration_final_eps=0.05, **common)
        model = (DoubleDQN if algo == "ddqn" else DQN)(**dqn_kwargs)
        model.max_grad_norm = 1.0
        return model
    if algo == "ppo":
        return PPO(learning_rate=3e-4, n_steps=1024, batch_size=64,
                   n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                   ent_coef=0.01, **common)
    if algo == "a2c":
        return A2C(learning_rate=7e-4, n_steps=16, gamma=0.99,
                   gae_lambda=0.95, ent_coef=0.01, vf_coef=0.5, **common)
    raise ValueError(algo)


def dump_hyperparams(algo: str, seed: int, total_timesteps: int) -> None:
    path = os.path.join(RESULTS_DIR, "hyperparameters.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data = json.load(open(path)) if os.path.exists(path) else {}
    data[f"{algo}_seed{seed}"] = {"timesteps": total_timesteps}
    json.dump(data, open(path, "w"), indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True,
                    choices=["dqn", "ddqn", "ppo", "a2c"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=None,
                    help="override the default (sanity runs: 10000)")
    args = ap.parse_args()

    default_ts = {"dqn": DQN_TIMESTEPS, "ddqn": DQN_TIMESTEPS,
                  "ppo": PPO_TIMESTEPS, "a2c": A2C_TIMESTEPS}[args.algo]
    total_timesteps = args.timesteps or default_ts

    model = build_model(args.algo, args.seed, total_timesteps)
    model.learn(total_timesteps=total_timesteps,
                callback=[QOverestimationCallback(
                              f"qvalues_{args.algo}_seed{args.seed}.csv"),
                          EpisodeRewardCallback(
                              f"rewards_{args.algo}_seed{args.seed}.csv")],
                progress_bar=False)

    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{args.algo}_seed{args.seed}")
    model.save(path)
    dump_hyperparams(args.algo, args.seed, total_timesteps)
    print(f"saved -> {path}.zip after {total_timesteps} timesteps")


if __name__ == "__main__":
    main()
