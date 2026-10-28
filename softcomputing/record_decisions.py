"""record_decisions.py -- export the fuzzy reasoning behind real decisions.

The replay traces in ``results/traces/`` only store aggregate utilization, so
they cannot explain WHY the FIS picked a server.  This script re-runs the same
seeded episodes (same env, same seed, same policy objects) and records the
full audit trail produced by ``FuzzyScheduler.explain``:

  * the head job and the queue pressure,
  * every candidate server's crisp inputs and suitability score,
  * the chosen server's membership degrees and fired rules,
  * the weighted-average numerator/denominator.

The output feeds the dashboard's "Fuzzy decision inspector" and
``softcomputing/demo.py``.  It never changes a policy or the CSV results.

    python -m softcomputing.record_decisions
    python -m softcomputing.record_decisions --scenarios heavy --max-decisions 80
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (BASE_SEED, FUZZY_INPUTS, FUZZY_RULES, N_SERVERS,  # noqa: E402
                    RESULTS_DIR, SOFT_POLICY_NAMES)

from env.scheduling_env import CloudSchedulingEnv  # noqa: E402
from softcomputing.fis_policy import FuzzyScheduler  # noqa: E402
from softcomputing.ga import decode, select_best_ga  # noqa: E402

SCENARIOS = ["light", "heavy", "bursty"]
OUT_DIR = os.path.join(RESULTS_DIR, "softcomputing")
PATHS = {
    "fis": os.path.join(OUT_DIR, "decisions_fis_{scenario}.json"),
    "ga_fis": os.path.join(OUT_DIR, "decisions_ga_fis_{scenario}.json"),
}


def _sample(records: list, limit: int) -> list:
    """Evenly spaced sample that always keeps the first and last decision.

    WHY not the first N only: a demo should also show the steady state, not
    just the empty-cluster warm-up.
    """
    if limit <= 0 or len(records) <= limit:
        return records
    idx = np.linspace(0, len(records) - 1, limit)
    keep = sorted({int(round(i)) for i in idx})
    return [records[i] for i in keep]


def run_episode(params, scenario: str, limit: int) -> dict:
    """One seeded eval episode; returns the sampled audit trail + metrics."""
    env = CloudSchedulingEnv(mode="eval", scenario=scenario)
    policy = FuzzyScheduler(params)
    obs, _ = env.reset(seed=BASE_SEED)
    records, info = [], {}
    done = False
    while not done:
        rec = policy.explain(obs, env)
        records.append(rec)
        obs, _, term, trunc, info = env.step(int(rec["action"]))
        done = term or trunc

    sampled = _sample(records, limit)
    # WHY policy.params: passing None means "use the hand-designed defaults",
    # so the concrete cuts only exist after construction.
    cuts = {name: [float(policy.params.cuts[2 * i]),
                   float(policy.params.cuts[2 * i + 1])]
            for i, name in enumerate(FUZZY_INPUTS)}
    return {
        "scenario": scenario,
        "seed": BASE_SEED,
        "metrics": {k: (round(float(v), 4) if isinstance(v, float) else v)
                    for k, v in info.items()},
        "total_decisions": len(records),
        "sampled": len(sampled),
        "cuts": cuts,
        "rule_table": [list(r) for r in FUZZY_RULES],
        "decisions": sampled,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS,
                    choices=SCENARIOS)
    ap.add_argument("--policies", nargs="+", default=["fis", "ga_fis"],
                    choices=["fis", "ga_fis"])
    ap.add_argument("--max-decisions", type=int, default=150,
                    help="evenly-spaced sample size per file (0 = keep all)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    selected = select_best_ga()
    if "ga_fis" in args.policies and selected is None:
        raise SystemExit("no GA chromosome found: run "
                         "`python -m softcomputing.train_ga` first")

    for policy_key in args.policies:
        params = decode(selected["chromosome"]) if policy_key == "ga_fis" else None
        label = "GA-FIS" if policy_key == "ga_fis" else "FIS"
        for scen in args.scenarios:
            payload = run_episode(params, scen, args.max_decisions)
            payload["policy"] = policy_key
            payload["label"] = label
            payload["policy_name"] = SOFT_POLICY_NAMES[policy_key]
            if policy_key == "ga_fis":
                payload["ga_seed"] = selected["seed"]
            path = PATHS[policy_key].format(scenario=scen)
            with open(path, "w") as f:
                json.dump(payload, f, separators=(",", ":"))
            print(f"{label:6s} {scen:6s} -> {path} "
                  f"({payload['sampled']}/{payload['total_decisions']} decisions, "
                  f"avg_wait {payload['metrics']['avg_wait']:.1f}s)")


if __name__ == "__main__":
    main()
