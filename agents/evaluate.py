"""evaluate.py -- run ANY policy on ANY scenario and log metrics (plan 6).

A "policy" is a callable taking (obs, env) -> action. Heuristic policies come
from baselines/heuristics.py; RL policies are wrapped as
`model.predict(obs, deterministic=True)[0]`.

WHY one harness for everything: the fairness argument in the viva ("same
environment, same scenarios, same metrics") is only true if ALL policies run
through this exact loop.

CLI:
  python -m agents.evaluate --policies heuristic [--scenarios light heavy bursty]
"""

import argparse
import json
import os
import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_SEED, RESULTS_DIR  # noqa: E402

from env.scheduling_env import CloudSchedulingEnv  # noqa: E402

COMPARISON_CSV = os.path.join(RESULTS_DIR, "full_comparison.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary_table.csv")
TRACE_DIR = os.path.join(RESULTS_DIR, "traces")

METRIC_COLS = ["avg_wait", "p95_wait", "avg_cpu_util",
               "completion_rate", "invalid_actions", "makespan"]
FULL_COLS = ["policy", "scenario", "seed"] + METRIC_COLS


class RLPolicy:
    """Wraps an SB3 model as policy(obs, env) -> action.

    WHY deterministic=True: evaluation must measure the learned policy, not
    its exploration noise.
    """

    def __init__(self, model):
        self.model = model

    def __call__(self, obs, env):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def run_episode(policy_fn_factory, env, seed: int, trace_name=None) -> dict:
    """One seeded episode; optionally save the replay trace as JSON.

    WHY a factory: RoundRobin holds a pointer (per-episode state), so a fresh
    policy instance per episode keeps heuristic runs independent.
    """
    policy = policy_fn_factory()
    obs, _ = env.reset(seed=seed)
    done, info = False, {}
    while not done:
        obs, _, term, trunc, info = env.step(int(policy(obs, env)))
        done = term or trunc
    if trace_name:
        os.makedirs(TRACE_DIR, exist_ok=True)
        with open(os.path.join(TRACE_DIR, f"{trace_name}.json"), "w") as f:
            json.dump(env._trace, f)
    return info


def evaluate(policy_fn_factory, scenario: str, n_episodes: int = 1,
             seed: int = BASE_SEED, trace_name=None) -> list[dict]:
    """Run n_episodes and return metric dicts (one per episode)."""
    rows = []
    for ep in range(n_episodes):
        env = CloudSchedulingEnv(mode="eval", scenario=scenario,
                                 record_trace=trace_name is not None)
        info = run_episode(policy_fn_factory, env, seed=seed + ep,
                           trace_name=f"{trace_name}_{ep}" if trace_name else None)
        info.update({"scenario": scenario, "seed": seed + ep})
        rows.append(info)
    return rows


def append_rows(rows: list[dict], path: str, cols: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = pd.DataFrame(rows)[cols]
    if os.path.exists(path):
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def summarize() -> None:
    """One row per (policy, scenario): mean/std of each metric across seeds."""
    df = pd.read_csv(COMPARISON_CSV)
    g = df.groupby(["policy", "scenario"])[METRIC_COLS]
    mean = g.mean()
    std = g.std().fillna(0.0).add_suffix("_std")
    out = pd.concat([mean, std], axis=1).reset_index()
    out.to_csv(SUMMARY_CSV, index=False)
    print(f"summary -> {SUMMARY_CSV} ({len(out)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", required=True,
                    choices=["heuristic", "rl"])
    ap.add_argument("--scenarios", nargs="+",
                    default=["light", "heavy", "bursty"])
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="evaluate only these RL seeds (default: every saved seed)")
    ap.add_argument("--no-traces", action="store_true",
                    help="skip saving replay traces")
    ap.add_argument("--no-csv", action="store_true",
                    help="trace-only run: do not append rows or re-summarize")
    args = ap.parse_args()

    from baselines.heuristics import HEURISTIC_FACTORIES
    from agents.policy_loader import list_seeds, load_rl_policy

    rows = []
    for scen in args.scenarios:
        if "heuristic" in args.policies:
            for name, factory in HEURISTIC_FACTORIES.items():
                n_ep = args.episodes or 1
                trace = None if args.no_traces else f"{name}_{scen}"
                for info in tqdm(evaluate(factory, scen, n_ep, trace_name=trace),
                                 desc=f"{name} {scen}"):
                    info["policy"] = "heuristic." + name
                    rows.append(info)
        if "rl" in args.policies:
            # WHY one episode per saved seed: scenario job sets are fixed and
            # RL acts deterministic=True, so re-runs would be identical rows;
            # the seed dimension is carried by WHICH trained model is used.
            n_ep = args.episodes or 1
            for name in ["dqn", "ddqn", "ppo", "a2c"]:
                seeds = args.seeds if args.seeds is not None else list_seeds(name)
                for s in seeds:
                    pol = load_rl_policy(name, s)
                    if pol is None:
                        continue
                    trace = None if args.no_traces else f"{name}_seed{s}_{scen}"
                    for info in tqdm(evaluate(lambda p=pol: p, scen, n_ep,
                                              trace_name=trace),
                                     desc=f"{name} s{s} {scen}"):
                        info["policy"] = f"rl.{name}"
                        info["seed"] = s
                        rows.append(info)

    if args.no_csv:
        print(f"trace-only run: {len(rows)} episodes, CSV untouched")
    else:
        append_rows(rows, COMPARISON_CSV, FULL_COLS)
        summarize()
    last = pd.DataFrame(rows)[FULL_COLS]
    print(last.groupby("policy")[["avg_wait", "avg_cpu_util",
                                  "invalid_actions"]].mean().round(2))


if __name__ == "__main__":
    main()
