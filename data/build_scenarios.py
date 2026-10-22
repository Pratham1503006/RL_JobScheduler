"""build_scenarios.py -- jobs_clean.csv -> scenario CSVs + exploration stats.

Steps (per plan section 4.6, CRITICAL):
1. REQ_SCALE: multiply CPU and memory requests so that p95(scaled cpu) is
   about 0.5 of one server's capacity (capacity = 1.0). WHY: raw trace
   requests are normalized to a 12,000-machine cluster, so a typical task
   asks for ~2.5% of a machine; on 10 servers packing would be trivial and
   every policy would look identical.
2. Load calibration: for each scenario, rescale arrival times by a factor s
   (arrival *= s, durations unchanged) so that
       rho = sum(cpu_request_i * duration_i) / (T * N * capacity)
   hits the target. WHY: the raw trace arrives far faster than 10 servers
   could serve; we tune the OFFERED LOAD, not the job sizes.
3. Scenarios: light (rho=0.40), heavy (rho=0.90), bursty (mean rho=0.60 but
   arrivals taken from the window with the highest per-minute coefficient of
   variation, so there are sharp spikes). Each has SCENARIO_JOBS jobs.
4. Train/test split by time: first 70% of jobs_clean.csv by arrival time is
   the training pool; scenarios are built ONLY from the last 30%. WHY: agents
   must never train on the exact jobs they are evaluated on.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (CLEAN_CSV, N_SERVERS, REQ_SCALE, RHO_BURSTY, RHO_HEAVY,
                    RHO_LIGHT, SCEN_DIR, SCENARIO_JOBS, TARGET_P95_REQ,
                    TRAIN_POOL)  # noqa: E402

# Scenario CSV columns (arrival times ALREADY rescaled to the target rho).
COLS = ["job_id", "arrival_time", "duration", "cpu_request", "mem_request"]


def compute_req_scale(df: pd.DataFrame) -> float:
    """Scale factor so p95(cpu) maps to TARGET_P95_REQ of a server."""
    return float(TARGET_P95_REQ / df["cpu_request"].quantile(0.95))


def apply_req_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Apply config.REQ_SCALE; verify it achieves the p95 target.

    WHY config-first: the plan forbids magic numbers outside config.py, so
    the scale is a recorded constant and this script only CHECKS it.
    """
    expected = compute_req_scale(df)
    assert abs(expected - REQ_SCALE) < 0.05, (
        f"config.REQ_SCALE={REQ_SCALE} stale; re-measured {expected:.4f}")
    out = df.copy()
    out["cpu_request"] = (out["cpu_request"] * REQ_SCALE).clip(upper=1.0)
    out["mem_request"] = (out["mem_request"] * REQ_SCALE).clip(upper=1.0)
    return out


def rho_of(jobs: pd.DataFrame, n_servers: int = N_SERVERS) -> float:
    """Offered load: cpu-seconds demanded / cpu-seconds available."""
    span = jobs["arrival_time"].max() - jobs["arrival_time"].min()
    work = float((jobs["cpu_request"] * jobs["duration"]).sum())
    return work / (span * n_servers)


def rescale_arrivals(jobs: pd.DataFrame, target_rho: float) -> pd.DataFrame:
    """Return a copy with arrivals stretched/compressed so rho == target.

    WHY multiply arrivals only: durations are real runtimes; stretching time
    lowers the arrival RATE, which is exactly the load knob we want.
    """
    cur = rho_of(jobs)
    s = cur / target_rho  # >1 means stretch out arrivals (lower load)
    out = jobs.copy()
    out["arrival_time"] = out["arrival_time"] * s
    return out


def pick_bursty_window(pool: pd.DataFrame, n_jobs: int) -> pd.DataFrame:
    """Choose the n_jobs-long slice with the most bursty arrivals.

    WHY: 'bursty' must differ from 'heavy' in ARRIVAL VARIANCE, not mean
    load; the per-minute coefficient of variation is a direct measure of it.
    """
    cand = pool.reset_index(drop=True)
    if len(cand) <= n_jobs:
        return cand
    minute = (cand["arrival_time"] // 60).astype(int)
    counts = minute.value_counts().sort_index()
    cv = counts.std() / counts.mean()  # overall CV of per-minute counts
    # Slide a window over consecutive jobs; score each by CV of its arrivals.
    best_i, best_cv = 0, -1.0
    step = max(1, len(cand) // 400)  # ~400 candidate windows: fast + thorough
    for i in range(0, len(cand) - n_jobs + 1, step):
        w = cand.iloc[i:i + n_jobs]
        m = (w["arrival_time"] // 60).astype(int).value_counts()
        if len(m) < 2:
            continue
        c = m.std() / m.mean()
        if c > best_cv:
            best_cv, best_i = c, i
    win = cand.iloc[best_i:best_i + n_jobs].copy()
    win.attrs["window_cv"] = float(best_cv)
    win.attrs["pool_cv"] = float(cv)
    return win


def main() -> None:
    df = pd.read_csv(CLEAN_CSV)

    # ---- step 1: request scaling (constant from config.py, verified) ----
    df = apply_req_scale(df)
    print(f"REQ_SCALE = {REQ_SCALE:.4f}  "
          f"(p95 cpu: {df['cpu_request'].quantile(0.95):.4f} after scaling)")

    # ---- step 4 (first half): train/test split by time ----
    split = int(len(df) * 0.7)
    train_pool = df.iloc[:split].reset_index(drop=True)
    test_pool = df.iloc[split:].reset_index(drop=True)
    os.makedirs(SCEN_DIR, exist_ok=True)
    train_pool.to_csv(TRAIN_POOL, index=False)
    print(f"train pool: {len(train_pool)} jobs | test pool: {len(test_pool)} jobs")

    # ---- steps 2+3: build the three scenarios from the TEST pool ----
    report = {"req_scale": REQ_SCALE,
              "train_pool_jobs": len(train_pool),
              "test_pool_jobs": len(test_pool),
              "scenarios": {}}
    targets = {"light": RHO_LIGHT, "heavy": RHO_HEAVY}
    for name, target in targets.items():
        raw = test_pool.iloc[-SCENARIO_JOBS:].reset_index(drop=True)
        sc = rescale_arrivals(raw, target)
        sc.to_csv(os.path.join(SCEN_DIR, f"{name}.csv"), index=False)
        r = rho_of(sc)
        report["scenarios"][name] = {"target_rho": target, "actual_rho": round(r, 4),
                                     "jobs": len(sc),
                                     "arrival_factor": round(rho_of(raw) / target, 4)}
        print(f"{name}: rho={r:.4f} (target {target}) | span={sc['arrival_time'].max():.0f}s")

    # Bursty: pick the highest-CV window, THEN rescale to its (lower) target.
    raw = pick_bursty_window(test_pool, SCENARIO_JOBS)
    sc = rescale_arrivals(raw, RHO_BURSTY)
    sc.to_csv(os.path.join(SCEN_DIR, "bursty.csv"), index=False)
    r = rho_of(sc)
    report["scenarios"]["bursty"] = {
        "target_rho": RHO_BURSTY, "actual_rho": round(r, 4), "jobs": len(sc),
        "window_cv": raw.attrs.get("window_cv"), "pool_cv": raw.attrs.get("pool_cv"),
        "arrival_factor": round(rho_of(raw) / RHO_BURSTY, 4)}
    print(f"bursty: rho={r:.4f} (target {RHO_BURSTY}) | "
          f"window CV={raw.attrs.get('window_cv'):.3f} vs pool CV={raw.attrs.get('pool_cv'):.3f}")

    with open(os.path.join(SCEN_DIR, "calibration.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("calibration saved ->", os.path.join(SCEN_DIR, "calibration.json"))


if __name__ == "__main__":
    main()
