"""explore.py -- the 3 required exploration plots -> results/data_exploration/.

1. arrival rate over time
2. request distribution (CPU and memory)
3. duration distribution (log-scale x)

WHY these plots: they document the workload the simulator is fed and let a
reader sanity-check the cleaning + scaling decisions at a glance.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # WHY: no display on CI/agent runs; files only.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLEAN_CSV, REQ_SCALE, RESULTS_DIR  # noqa: E402

OUT = os.path.join(RESULTS_DIR, "data_exploration")


def main() -> None:
    df = pd.read_csv(CLEAN_CSV)
    # Plot requests in SIMULATOR units (server capacity = 1.0), i.e. after the
    # REQ_SCALE calibration, so the chart matches what policies are fed.
    df["cpu_request"] = (df["cpu_request"] * REQ_SCALE).clip(upper=1.0)
    df["mem_request"] = (df["mem_request"] * REQ_SCALE).clip(upper=1.0)
    os.makedirs(OUT, exist_ok=True)

    # ---- 1. arrival rate over time (jobs per 10-minute bucket) ----
    fig, ax = plt.subplots(figsize=(9, 4))
    bucket = df["arrival_time"] // 600
    rate = bucket.value_counts().sort_index() / 600.0  # jobs/second
    ax.plot(rate.index * 600 / 3600.0, rate.values, lw=0.8)
    ax.set_xlabel("time (h, window shifted to 0)")
    ax.set_ylabel("arrival rate (jobs/s)")
    ax.set_title("Job arrival rate over time (10-min buckets)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "arrival_rate.png"), dpi=150)
    plt.close(fig)

    # ---- 2. request distributions (CPU and memory, scaled) ----
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=False)
    for ax, col, label in ((axes[0], "cpu_request", "CPU request"),
                           (axes[1], "mem_request", "memory request")):
        ax.hist(df[col], bins=60, color="#4878cf")
        ax.set_xlabel(f"{label} (server capacity = 1.0)")
        ax.set_ylabel("tasks")
        ax.set_title(f"{label} distribution")
    fig.suptitle("Resource requests (after REQ_SCALE calibration)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "request_distribution.png"), dpi=150)
    plt.close(fig)

    # ---- 3. duration distribution (log x) ----
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["duration"], bins=np.logspace(0, np.log10(df["duration"].max()), 60),
            color="#6acc64")
    ax.set_xscale("log")
    ax.set_xlabel("duration (s, log scale; winsorized at p99.9)")
    ax.set_ylabel("tasks")
    ax.set_title("Task duration distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "duration_distribution.png"), dpi=150)
    plt.close(fig)

    print("plots written:")
    for f in sorted(os.listdir(OUT)):
        print(" -", os.path.join(OUT, f))


if __name__ == "__main__":
    main()
