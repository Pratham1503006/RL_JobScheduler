"""policy_loader.py -- resolve RL policy names to loaded SB3 models.

Supported names: dqn, ddqn, ppo, a2c (a model saved as models/{name}_seed{N}.zip).
WHY best-seed selection instead of hardcoding: Phase 4 trains 3 seeds and the
evaluation wants the BEST seed per algorithm (documented choice), while full
comparison uses every seed.
"""

import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR  # noqa: E402

from agents.evaluate import RLPolicy  # noqa: E402


def _find_model(name: str, seed: int) -> str | None:
    """Path of the saved model for `name` and seed, or None."""
    path = os.path.join(MODELS_DIR, f"{name}_seed{seed}.zip")
    return path if os.path.exists(path) else None


def list_seeds(name: str) -> list[int]:
    """Seeds that have a saved model for `name` (sorted)."""
    pat = os.path.join(MODELS_DIR, f"{name}_seed*.zip")
    files = sorted(glob.glob(pat),
                   key=lambda f: int(re.search(r"seed(\d+)", f).group(1)))
    return [int(re.search(r"seed(\d+)", f).group(1)) for f in files]


def load_rl_model(name: str, seed: int):
    """One loaded SB3 model (lazily imports SB3), or None if not saved."""
    from stable_baselines3 import DQN, PPO, A2C  # noqa: F401

    from agents.double_dqn import DoubleDQN

    classes = {"dqn": DQN, "ddqn": DoubleDQN, "ppo": PPO, "a2c": A2C}
    path = _find_model(name, seed)
    if path is None:
        return None
    return classes[name].load(path, device="cpu")


def load_rl_policy(name: str, seed: int) -> RLPolicy | None:
    """One named model wrapped as RLPolicy, or None if not saved."""
    model = load_rl_model(name, seed)
    return RLPolicy(model) if model is not None else None


def load_rl_policies(names=("dqn", "ddqn", "ppo", "a2c")) -> dict:
    """Returns {name: RLPolicy} using seed 0 of each algorithm. Kept for
    quick smoke checks -- the full comparison (evaluate.py CLI) uses every
    saved seed instead. Imports SB3 lazily (heuristic-only runs stay
    torch-free)."""
    out = {}
    for name in names:
        pol = load_rl_policy(name, 0)
        if pol is None:
            print(f"[policy_loader] no model for {name!r}, skipping")
            continue
        out[name] = pol
    return out
