"""charts.py -- the six report charts (plan 9.1 + addendum 5.2). CPU-only.

Every number plotted comes straight from results/ files:
  1+2: results/summary_table.csv   (mean + std across seeds)
  3  : results/traces/*.json       (replay traces, heavy scenario)
  4  : results/train_logs/qvalues_*_seed*.csv
  5  : results/ga_logs/ga_seed*_history.csv   (GA convergence)
  6  : results/ga_logs/ga_seed*_best.json     (default vs tuned memberships)

Output: results/charts/*.png (200 dpi). WHY matplotlib for the report
charts: static PNGs are what the professor reads; the interactive dashboard
is a separate display of the same data files.
"""

import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")  # no display available; write PNGs directly
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR  # noqa: E402

CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")
TRACES_DIR = os.path.join(RESULTS_DIR, "traces")
TRAIN_LOGS = os.path.join(RESULTS_DIR, "train_logs")
GA_LOGS = os.path.join(RESULTS_DIR, "ga_logs")

# Display order + short labels + family: heuristics, soft computing, then RL.
# WHY one colour per family (addendum 5.2): the reader should be able to see
# "soft computing vs heuristics vs RL" without decoding nine unrelated hues.
POLICIES = [
    ("heuristic.round_robin", "RR", "heuristic"),
    ("heuristic.first_fit", "FF", "heuristic"),
    ("heuristic.best_fit", "BF", "heuristic"),
    ("soft.fis", "FIS", "soft"),
    ("soft.ga_fis", "GA-FIS", "soft"),
    ("rl.dqn", "DQN", "rl"),
    ("rl.ddqn", "DDQN", "rl"),
    ("rl.ppo", "PPO", "rl"),
    ("rl.a2c", "A2C", "rl"),
]
SCENARIOS = ["light", "heavy", "bursty"]
FAMILY_COLORS = {
    "heuristic": "#4d4d4d",   # neutral grey
    "soft": "#1b9e77",        # teal-green
    "rl": "#7570b3",          # muted purple
}
FAMILY_LABELS = {
    "heuristic": "Heuristics",
    "soft": "Soft computing",
    "rl": "Reinforcement learning",
}
HATCHES = ["", "//", ".."]    # scenario encoding (colour is reserved for family)


def _summary() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS_DIR, "summary_table.csv"))


def grouped_bar(metric: str, ylabel: str, fname: str,
                log_y: bool = False) -> None:
    """Charts 1 and 2: per-policy bars grouped by scenario.

    Colour = policy family (same colour for every member), hatch = scenario.
    Policies missing from the summary (e.g. before the soft-computing
    evaluation has run) are skipped instead of crashing the plot.
    """
    summ = _summary()
    available = set(summ["policy"])
    policies = [(p, lbl, fam) for p, lbl, fam in POLICIES if p in available]
    if not policies:
        raise SystemExit("summary_table.csv has none of the expected policies")

    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, scen in enumerate(SCENARIOS):
        rows = summ[summ.scenario == scen]
        means, stds = [], []
        for pol, _lbl, _fam in policies:
            row = rows[rows.policy == pol]
            means.append(float(row[metric].iloc[0]))
            stds.append(float(row[f"{metric}_std"].iloc[0]))
        xs = [j + (i - 1) * width for j in range(len(policies))]
        ax.bar(xs, means, width, yerr=stds, capsize=2,
               color=[FAMILY_COLORS[fam] for _p, _l, fam in policies],
               alpha=[0.45, 0.72, 0.95][i], hatch=HATCHES[i],
               edgecolor="k", linewidth=0.4)

    labels = [lbl for _p, lbl, _f in policies]
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Policy")
    if log_y:
        # WHY log scale: RL stalls (10x-100x waits) would compress the
        # heuristic/soft-computing bars into invisibility on a linear axis.
        ax.set_yscale("log")

    # family separators + labels, so the three-way story is visible
    families = [fam for _p, _l, fam in policies]
    for j in range(1, len(families)):
        if families[j] != families[j - 1]:
            ax.axvline(j - 0.5, color="#999999", ls=":", lw=0.9)
    start = 0
    for j in range(1, len(families) + 1):
        if j == len(families) or families[j] != families[start]:
            ax.text((start + j - 1) / 2, 1.01, FAMILY_LABELS[families[start]],
                    transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=8, color=FAMILY_COLORS[families[start]])
            start = j

    handles = [Patch(facecolor=c, edgecolor="k", label=FAMILY_LABELS[f])
               for f, c in FAMILY_COLORS.items()
               if f in families]
    handles += [Patch(facecolor="white", edgecolor="k", hatch=HATCHES[i],
                      label=scen) for i, scen in enumerate(SCENARIOS)]
    ax.legend(handles=handles, ncol=2, fontsize=8)
    ax.set_title(f"{ylabel} per policy per scenario (mean, error=std over seeds)",
                 pad=16)  # pad: keep the title clear of the family labels above the axes
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
        ("round_robin_heavy_0.json", "Round Robin (best heuristic)", "#4d4d4d"),
        ("ddqn_seed0_heavy_0.json", "Double DQN (best RL)", "#7570b3"),
    ]
    for fname, label, color in cases:
        path = os.path.join(TRACES_DIR, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
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


def chart5_ga_convergence() -> None:
    """Chart 5: best/mean GA fitness per generation, all three seeds."""
    paths = sorted(glob.glob(os.path.join(GA_LOGS, "ga_seed*_history.csv")))
    if not paths:
        print("chart 5 skipped: no results/ga_logs/ga_seed*_history.csv")
        return
    from softcomputing.plots import plot_convergence

    histories = {}
    for path in paths:
        seed = int(re.search(r"ga_seed(\d+)_history", path).group(1))
        histories[seed] = pd.read_csv(path)
    default_fitness = None
    summary = os.path.join(GA_LOGS, "ga_seed_summary.csv")
    if os.path.exists(summary):
        default_fitness = float(pd.read_csv(summary)["default_fitness"].iloc[0])
    plot_convergence(histories,
                     os.path.join(CHARTS_DIR, "5_ga_convergence.png"),
                     default_fitness=default_fitness,
                     title="GA-FIS training convergence (best and mean)")


def chart6_fuzzy_memberships() -> None:
    """Chart 6: default vs GA-tuned membership functions, one panel per input."""
    from softcomputing.fuzzy import default_params
    from softcomputing.ga import decode, select_best_ga
    from softcomputing.plots import plot_overlay

    best = select_best_ga()
    if best is None:
        print("chart 6 skipped: no results/ga_logs/ga_seed*_best.json")
        return
    plot_overlay(default_params(), decode(best["chromosome"]),
                 os.path.join(CHARTS_DIR, "6_fuzzy_memberships.png"),
                 title=f"Default FIS vs GA-FIS (selected seed {best['seed']})")


def main() -> None:
    os.makedirs(CHARTS_DIR, exist_ok=True)
    grouped_bar("avg_wait", "Average wait (s)", "1_avg_wait_per_scenario.png",
                log_y=True)
    grouped_bar("avg_cpu_util", "Average CPU utilization (0-1)",
                "2_avg_cpu_util_per_scenario.png")
    chart3_utilization_timeline()
    chart4_qvalue_overestimation()
    chart5_ga_convergence()
    chart6_fuzzy_memberships()
    for f in sorted(os.listdir(CHARTS_DIR)):
        print(os.path.join(CHARTS_DIR, f),
              os.path.getsize(os.path.join(CHARTS_DIR, f)), "bytes")


if __name__ == "__main__":
    main()
