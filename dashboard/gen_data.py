"""gen_data.py -- build dashboard/data.json for the React replay dashboard.

The dashboard REPLAYS precomputed traces (plan 9.1: no live RL inference in
the UI). This script collects, in ONE manifest:
  - summary_table.csv rows (results section table)
  - the trace file for every (scenario, policy) in results/traces/
  - the four chart PNGs (shown alongside)

Trace choice per policy: heuristics are deterministic (single run); RL uses
seed 0 per algorithm. WHY seed 0: it is one valid trained run of each algo;
all valid seeds are within 1 seed-spread of each other on the reported
metrics (results/analysis.md).

Run BEFORE opening the dashboard, then serve the repo root:
  python dashboard/gen_data.py
  python -m http.server 8000  # -> open http://localhost:8000/dashboard/
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TRACE_DIR = os.path.join(RESULTS_DIR, "traces")

POLICIES = [  # dash_key -> (full name in summary CSV, display label)
    ("round_robin", "heuristic.round_robin", "Round Robin"),
    ("first_fit", "heuristic.first_fit", "First-Fit"),
    ("best_fit", "heuristic.best_fit", "Best-Fit"),
    ("dqn", "rl.dqn", "DQN"),
    ("ddqn", "rl.ddqn", "Double DQN"),
    ("ppo", "rl.ppo", "PPO"),
    ("a2c", "rl.a2c", "A2C"),
]
SCENARIOS = ["light", "heavy", "bursty"]
BEST_RL_SEED = 0

METRICS = ["avg_wait", "p95_wait", "avg_cpu_util", "completion_rate",
           "invalid_actions", "makespan"]


def trace_path(scen: str, key: str, heuristic: bool) -> str | None:
    """Relative path (from dashboard/) of the trace for this policy+scenario.

    Trace files follow agents/evaluate.py naming:
      heuristic: {key}_{scen}_0.json ; RL: {key}_seed{N}_{scen}_0.json
    """
    probe = (os.path.join(TRACE_DIR, f"{key}_{scen}_0.json") if heuristic
             else os.path.join(TRACE_DIR, f"{key}_seed{BEST_RL_SEED}_{scen}_0.json"))
    if not os.path.exists(probe):
        return None
    rel = os.path.relpath(probe, HERE).replace("\\", "/")
    return "../" + rel


def main() -> None:
    summ = pd.read_csv(os.path.join(RESULTS_DIR, "summary_table.csv"))

    table = {
        "columns": ["policy", "scenario"] + METRICS,
        "rows": [],
    }
    for pol_full, _lbl in [(pf, l) for _k, pf, l in POLICIES]:
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

    traces = {}
    for scen in SCENARIOS:
        traces[scen] = {}
        for key, _pf, lbl in POLICIES:
            heuristic = key in ("round_robin", "first_fit", "best_fit")
            path = trace_path(scen, key, heuristic)
            if path is None:
                print(f"WARNING: no trace for {key}/{scen}")
                continue
            traces[scen][lbl] = path

    charts = sorted(
        os.path.join("../results/charts", f).replace("\\", "/")
        for f in os.listdir(os.path.join(RESULTS_DIR, "charts"))
        if f.endswith(".png"))

    out = {"scenarios": SCENARIOS, "policies": list(traces["light"].keys()),
           "traces": traces, "summary": table, "charts": charts}
    dst = os.path.join(HERE, "data.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"wrote {dst}: {len(out['charts'])} charts, "
          f"{sum(len(v) for v in traces.values())} traces")


if __name__ == "__main__":
    main()
