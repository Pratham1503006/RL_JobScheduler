"""evaluate_soft.py -- evaluate FIS and GA-FIS on the held-out scenarios.

Extends Phase 5 without touching the RL harness: it calls the SAME
``agents.evaluate.evaluate`` loop, saves traces with the deterministic naming
convention (``{name}_{scenario}_0.json``) and appends ``soft.fis`` /
``soft.ga_fis`` rows to ``results/full_comparison.csv``.  Re-running is
idempotent: existing soft rows are dropped before the new ones are appended.

The GA-FIS row uses the chromosome with the best TRAINING fitness across the
three GA seeds (``select_best_ga``), never a test-scenario score.  A
supplementary robustness table evaluates all three seed chromosomes on the
scenarios.

    python -m softcomputing.evaluate_soft
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_SEED, GA_SEEDS, RESULTS_DIR, SOFT_POLICY_NAMES  # noqa: E402

from agents.evaluate import (COMPARISON_CSV, FULL_COLS, append_rows,  # noqa: E402
                             evaluate, summarize)
from softcomputing.fis_policy import FuzzyScheduler  # noqa: E402
from softcomputing.ga import decode, load_best, select_best_ga  # noqa: E402

SCENARIOS = ["light", "heavy", "bursty"]
SOFT_POLICIES = list(SOFT_POLICY_NAMES.values())
ROBUST_CSV = os.path.join(RESULTS_DIR, "ga_logs", "ga_seed_robustness.csv")


def run_policy(factory, scenario: str, trace_name: str, policy_name: str) -> list:
    rows = evaluate(factory, scenario, n_episodes=1, seed=BASE_SEED,
                    trace_name=trace_name)
    for row in rows:
        row["policy"] = policy_name
    return rows


def _robustness_rows(scenarios) -> pd.DataFrame:
    """Every GA seed's chromosome on the held-out scenarios (supplementary)."""
    selected = select_best_ga()
    rows = []
    for seed in GA_SEEDS:
        rec = load_best(seed)
        if rec is None:
            print(f"WARNING: missing GA seed {seed} artifacts")
            continue
        params = decode(rec["chromosome"])
        for scen in scenarios:
            info = evaluate(lambda p=params: FuzzyScheduler(p), scen,
                            n_episodes=1, seed=BASE_SEED)[0]
            rows.append({
                "ga_seed": seed,
                "train_fitness": rec["fitness"],
                "selected": int(selected is not None
                                and rec["seed"] == selected["seed"]),
                "scenario": scen,
                **{k: info[k] for k in FULL_COLS[3:]},
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    ap.add_argument("--no-csv", action="store_true",
                    help="write traces/robustness but leave the CSVs untouched")
    ap.add_argument("--no-robustness", action="store_true")
    args = ap.parse_args()

    rows = []
    # SC1: the hand-designed FIS (deterministic, one run per scenario).
    for scen in args.scenarios:
        rows += run_policy(lambda: FuzzyScheduler(), scen, f"fis_{scen}",
                           SOFT_POLICY_NAMES["fis"])

    # SC2: the selected GA-tuned FIS.
    selected = select_best_ga()
    if selected is None:
        raise SystemExit("no GA chromosome found: run "
                         "`python -m softcomputing.train_ga --seed N` first")
    params = decode(selected["chromosome"])
    print(f"GA-FIS uses seed {selected['seed']} "
          f"(training fitness {selected['fitness']:.3f})")
    for scen in args.scenarios:
        rows += run_policy(lambda p=params: FuzzyScheduler(p), scen,
                           f"ga_fis_{scen}", SOFT_POLICY_NAMES["ga_fis"])

    if not args.no_csv:
        # WHY drop first: re-running must not duplicate the soft rows.
        if os.path.exists(COMPARISON_CSV):
            old = pd.read_csv(COMPARISON_CSV)
            old = old[~old["policy"].isin(SOFT_POLICIES)]
            old.to_csv(COMPARISON_CSV, index=False)
        append_rows(rows, COMPARISON_CSV, FULL_COLS)
        summarize()

    if not args.no_robustness:
        robust = _robustness_rows(args.scenarios)
        if not robust.empty:
            os.makedirs(os.path.dirname(ROBUST_CSV), exist_ok=True)
            robust.to_csv(ROBUST_CSV, index=False)
            print(f"robustness -> {ROBUST_CSV} ({len(robust)} rows)")

    out = pd.DataFrame(rows)[FULL_COLS]
    print(out.groupby("policy")[["avg_wait", "avg_cpu_util",
                                 "completion_rate", "invalid_actions"]]
          .mean().round(3))


if __name__ == "__main__":
    main()
