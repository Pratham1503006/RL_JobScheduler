"""charts.py -- the four report charts (plan 9.1). CPU-only, matplotlib.

Every number plotted comes straight from results/ files:
  1+2: results/summary_table.csv   (mean + std across seeds)
  3  : results/traces/*.json       (replay traces, heavy scenario)
  4  : results/train_logs/qvalues_*_seed*.csv

Output: results/charts/*.png (200 dpi). WHY matplotlib for the report
charts: static PNGs are what the professor reads; the interactive dashboard
is a separate display of the same data files.
"""

import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")  # no display available; write PNGs directly
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR  # noqa: E402

CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")
TRACES_DIR = os.path.join(RESULTS_DIR, "traces")
TRAIN_LOGS = os.path.join(RESULTS_DIR, "train_logs")

# Display order + short labels: heuristics first, then RL (report convention).
POLICIES = [
    ("heuristic.round_robin", "RR"),
    ("heuristic.first_fit", "FF"),
    ("heuristic.best_fit", "BF"),
    ("rl.dqn", "DQN"),
    ("rl.ddqn", "DDQN"),
    ("rl.ppo", "PPO"),
    ("rl.a2c", "A2C"),
]
SCENARIOS = ["light", "heavy", "bursty"]
# WHY trace/chrome colors: consistent policy colors across charts 1-3 so the
# professor can match bars to policies without re-reading legends.
COLORS = ["#888888", "#bbbbbb", "#555555", "#1f77b4", "#2ca02c",
          "#9467bd", "#ff7f0e"]


def _summary() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS_DIR, "summary_table.csv"))


def grouped_bar(metric: str, ylabel: str, fname: str,
                log_y: bool = False) -> None:
    """Charts 1 and 2: per-policy bars grouped by scenario, std as error bars."""
    summ = _summary()
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, scen in enumerate(SCENARIOS):
        rows = summ[summ.scenario == scen]
        means, stds = [], []
        for pol, _ in POLICIES:
            row = rows[rows.policy == pol]
            means.append(float(row[metric].iloc[0]))
            stds.append(float(row[f"{metric}_std"].iloc[0]))
        x = [j + i * width for j in range(len(POLICIES))]
        ax.bar(x, means, width, yerr=stds, capsize=3, label=scen,
               color=COLORS[i], alpha=0.55 + 0.2 * i, edgecolor="k",
               linewidth=0.4)
    labels = [lbl for _, lbl in POLICIES]
    ax.set_xticks([j + width for j in range(len(POLICIES))])
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Policy")
    if log_y:
        # WHY log scale: RL stalls (10x-100x waits) would compress the
        # heuristic bars into invisibility on a linear axis.
        ax.set_yscale("log")
    ax.legend(title="Scenario")
    ax.set_title(f"{ylabel} per policy per scenario (mean, error=std over seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, fname), dpi=200)
    plt.close(fig)


def chart3_utilization_timeline() -> None:
    """Best heuristic vs best RL agent, cluster CPU utilization vs time.

    Steps-post plot: each trace event is a state change held until the next
    event, so the drawn curve equals the time-weighted utilization.
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    cases = [
        ("round_robin_heavy_0.json", "Round Robin (best heuristic)", "#888888"),
        ("ddqn_seed0_heavy_0.json", "Double DQN (best RL)", "#2ca02c"),
    ]
    for fname, label, color in cases:
        with open(os.path.join(TRACES_DIR, fname)) as f:
            trace = json.load(f)
        t = [ev["time"] for ev in trace]
        util = [sum(ev["server_cpu_used"]) / len(ev["server_cpu_used"])
                for ev in trace]
        ax.step(t, util, where="post", label=label, color=color, lw=1.2)
    ax.set_xlabel("Simulated time (s)")
    ax.set_ylabel("Cluster CPU utilization (0-1)")
    ax.set_title("Cluster CPU utilization over time, heavy scenario")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "3_utilization_timeline_heavy.png"),
                dpi=200)
    plt.close(fig)


def chart4_qvalue_overestimation() -> None:
    """Mean max-Q over training, DQN vs Double DQN (mean over the 3 seeds)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for algo, label, color in [("dqn", "DQN", "#1f77b4"),
                               ("ddqn", "Double DQN", "#2ca02c")]:
        curves = []
        for path in sorted(os.listdir(TRAIN_LOGS)):
            m = re.fullmatch(rf"qvalues_{algo}_seed\d+\.csv", path)
            if not m:
                continue
            df = pd.read_csv(os.path.join(TRAIN_LOGS, path))
            curves.append(df.groupby("timesteps")["mean_max_q"].mean())
        mean = pd.concat(curves, axis=1).mean(axis=1)
        ax.plot(mean.index, mean.values, label=label, color=color,
                marker="o", ms=3, lw=1.2)
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Mean max-Q over 500 random observations")
    ax.set_title("Mean max-Q during training: DQN vs Double DQN")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "4_qvalue_overestimation.png"),
                dpi=200)
    plt.close(fig)


def main() -> None:
    os.makedirs(CHARTS_DIR, exist_ok=True)
    grouped_bar("avg_wait", "Average wait (s)", "1_avg_wait_per_scenario.png",
                log_y=True)
    grouped_bar("avg_cpu_util", "Average CPU utilization (0-1)",
                "2_avg_cpu_util_per_scenario.png")
    chart3_utilization_timeline()
    chart4_qvalue_overestimation()
    for f in sorted(os.listdir(CHARTS_DIR)):
        print(os.path.join(CHARTS_DIR, f),
              os.path.getsize(os.path.join(CHARTS_DIR, f)), "bytes")


if __name__ == "__main__":
    main()
