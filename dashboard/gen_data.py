"""gen_data.py -- build dashboard/data.json for the React replay dashboard.

The dashboard REPLAYS precomputed traces (plan 9.1: no live RL inference in
the UI). This script collects, in ONE manifest:
  - summary_table.csv rows (results section table)
  - the trace file for every (scenario, policy) in results/traces/
  - the report chart PNGs (results/charts/)
  - the soft-computing panel data (membership plots + default/tuned rule tables)

Trace choice per policy: heuristics and soft-computing policies are
deterministic (single run); RL uses seed 0 per algorithm. WHY seed 0: it is one
valid trained run of each algo (results/analysis.md).

Run BEFORE opening the dashboard, then serve the repo root:
  python dashboard/gen_data.py
  python -m http.server 8000  # -> open http://localhost:8000/dashboard/
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FUZZY_CUTS_DEFAULT, FUZZY_RULES, RESULTS_DIR  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TRACE_DIR = os.path.join(RESULTS_DIR, "traces")
SOFT_DIR = os.path.join(RESULTS_DIR, "softcomputing")
GA_LOG_DIR = os.path.join(RESULTS_DIR, "ga_logs")

# (dash_key, full name in summary CSV, display label, family)
POLICIES = [
    ("round_robin", "heuristic.round_robin", "Round Robin", "heuristic"),
    ("first_fit", "heuristic.first_fit", "First-Fit", "heuristic"),
    ("best_fit", "heuristic.best_fit", "Best-Fit", "heuristic"),
    ("fis", "soft.fis", "FIS", "soft"),
    ("ga_fis", "soft.ga_fis", "GA-FIS", "soft"),
    ("dqn", "rl.dqn", "DQN", "rl"),
    ("ddqn", "rl.ddqn", "Double DQN", "rl"),
    ("ppo", "rl.ppo", "PPO", "rl"),
    ("a2c", "rl.a2c", "A2C", "rl"),
]
SCENARIOS = ["light", "heavy", "bursty"]
BEST_RL_SEED = 0
FAMILY_LABELS = {
    "heuristic": "Heuristics",
    "soft": "Soft computing",
    "rl": "Reinforcement learning",
}

METRICS = ["avg_wait", "p95_wait", "avg_cpu_util", "completion_rate",
           "invalid_actions", "makespan"]


def trace_path(scen: str, key: str, deterministic: bool) -> str | None:
    """Relative path (from dashboard/) of the trace for this policy+scenario.

    Trace files follow agents/evaluate.py naming:
      heuristic / soft computing: {key}_{scen}_0.json
      RL:                        {key}_seed{N}_{scen}_0.json
    """
    probe = (os.path.join(TRACE_DIR, f"{key}_{scen}_0.json") if deterministic
             else os.path.join(TRACE_DIR,
                               f"{key}_seed{BEST_RL_SEED}_{scen}_0.json"))
    if not os.path.exists(probe):
        return None
    # relpath already includes the leading ".." out of dashboard/.
    return os.path.relpath(probe, HERE).replace("\\", "/")


def _rel(path: str) -> str:
    return os.path.relpath(path, HERE).replace("\\", "/")


def _decision_path(policy_key: str, scenario: str) -> str | None:
    """Recorded fuzzy audit trail for one policy+scenario (may not exist yet)."""
    path = os.path.join(SOFT_DIR, f"decisions_{policy_key}_{scenario}.json")
    return _rel(path) if os.path.exists(path) else None


def _softcomputing_block() -> dict:
    """Membership plots + default/tuned rule tables for the Soft computing tab."""
    from softcomputing.ga import decode, select_best_ga

    memberships = [
        os.path.join(SOFT_DIR, "fis_default_memberships.png"),
        os.path.join(SOFT_DIR, "ga_fis_memberships.png"),
    ]
    convergence = os.path.join(SOFT_DIR, "ga_convergence.png")

    default_rules = [[leftover, load, pressure, float(z)]
                     for leftover, load, pressure, z in FUZZY_RULES]
    default_cuts = {name: [float(a), float(b)]
                    for name, (a, b) in FUZZY_CUTS_DEFAULT.items()}

    block = {
        "default_rules": default_rules,
        "default_cuts": default_cuts,
        "memberships": [_rel(p) for p in memberships if os.path.exists(p)],
        "convergence": _rel(convergence) if os.path.exists(convergence) else None,
        "decisions": {
            label: {scen: _decision_path(key, scen) for scen in SCENARIOS}
            for key, label in (("fis", "FIS"), ("ga_fis", "GA-FIS"))
        },
        "selected_seed": None,
        "train_fitness": None,
        "default_fitness": None,
        "tuned_rules": None,
        "tuned_cuts": None,
        "seed_summary": [],
        "robustness": [],
        "missing": None,
    }

    best = select_best_ga()
    if best is None:
        block["missing"] = ("No GA artifacts yet. Run "
                            "`python -m softcomputing.train_ga --seed 0`.")
    else:
        params = decode(best["chromosome"])
        block["selected_seed"] = best["seed"]
        block["train_fitness"] = best["fitness"]
        block["default_fitness"] = best.get("default_fitness")
        block["tuned_rules"] = [[leftover, load, pressure, float(z)]
                                for (leftover, load, pressure, _z), z
                                in zip(FUZZY_RULES, params.z)]
        block["tuned_cuts"] = best.get("cuts", {})

    summary = os.path.join(GA_LOG_DIR, "ga_seed_summary.csv")
    if os.path.exists(summary):
        df = pd.read_csv(summary)
        block["seed_summary"] = df.to_dict(orient="records")
    robust = os.path.join(GA_LOG_DIR, "ga_seed_robustness.csv")
    if os.path.exists(robust):
        df = pd.read_csv(robust)
        block["robustness"] = df.round(3).to_dict(orient="records")

    return block


def main() -> None:
    summ = pd.read_csv(os.path.join(RESULTS_DIR, "summary_table.csv"))

    table = {
        "columns": ["policy", "scenario"] + METRICS,
        "rows": [],
    }
    for _key, pol_full, _lbl, _fam in POLICIES:
        for scen in SCENARIOS:
            row = summ[(summ.policy == pol_full) & (summ.scenario == scen)]
            if row.empty:
                continue
            vals = []
            for c in ["policy", "scenario"] + METRICS:
                v = row[c].iloc[0]
                if isinstance(v, float):
                    v = round(v, 3)
                vals.append(v)
            table["rows"].append(vals)

    traces, families = {}, {}
    for scen in SCENARIOS:
        traces[scen] = {}
        for key, _pf, lbl, fam in POLICIES:
            deterministic = fam in ("heuristic", "soft")
            path = trace_path(scen, key, deterministic)
            if path is None:
                print(f"WARNING: no trace for {key}/{scen}")
                continue
            traces[scen][lbl] = path
            families[lbl] = FAMILY_LABELS[fam]

    charts = sorted(
        os.path.join("../results/charts", f).replace("\\", "/")
        for f in os.listdir(os.path.join(RESULTS_DIR, "charts"))
        if f.endswith(".png"))

    out = {"scenarios": SCENARIOS, "policies": list(traces["light"].keys()),
           "families": families, "traces": traces, "summary": table,
           "charts": charts, "softcomputing": _softcomputing_block()}
    dst = os.path.join(HERE, "data.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"wrote {dst}: {len(out['charts'])} charts, "
          f"{sum(len(v) for v in traces.values())} traces, "
          f"soft tab {'ready' if out['softcomputing']['tuned_rules'] else 'incomplete'}")


if __name__ == "__main__":
    main()
