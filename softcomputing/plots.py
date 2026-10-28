"""plots.py -- static figures for the soft-computing part of the report.

Kept separate from ``fuzzy.py`` so the inference module stays pure math.  All
functions write PNGs to an explicit path (callers decide which results/
artifact they belong to) and return the path.
"""

import os

import matplotlib
matplotlib.use("Agg")  # CPU-only report rendering, no display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from config import FUZZY_INPUTS, FUZZY_SETS  # noqa: E402
from softcomputing.fuzzy import membership_params, set_memberships  # noqa: E402

COLORS = {
    "leftover": ["#c0392b", "#e67e22", "#27ae60"],
    "load": ["#2980b9", "#8e44ad", "#d35400"],
    "pressure": ["#16a085", "#c0392b"],
}
INPUT_LABEL = {
    "leftover": "leftover slack after placement",
    "load": "server load",
    "pressure": "queue pressure",
}


def _prepare(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def plot_memberships(params, path: str,
                     title: str = "Fuzzy membership functions") -> str:
    """The 8 default membership functions (SC1 acceptance check 5)."""
    _prepare(path)
    x = np.linspace(0.0, 1.0, 501)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, name in zip(axes, FUZZY_INPUTS):
        mus = set_memberships(name, x, params)
        for color, set_name in zip(COLORS[name], FUZZY_SETS[name]):
            ax.plot(x, mus[set_name], label=set_name, color=color, lw=1.8)
        # mark the two cut points that define this input's shape
        p1, p2 = (params.cuts[2 * FUZZY_INPUTS.index(name)],
                  params.cuts[2 * FUZZY_INPUTS.index(name) + 1])
        for p in (p1, p2):
            ax.axvline(float(p), color="#777777", ls=":", lw=0.8)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(INPUT_LABEL[name], fontsize=8)
        ax.set_ylabel(r"$\mu$", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_overlay(default, tuned, path: str,
                 title: str = "GA-FIS membership functions vs default") -> str:
    """Default (dashed grey) vs GA-tuned (solid colour), one panel per input."""
    _prepare(path)
    x = np.linspace(0.0, 1.0, 501)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, name in zip(axes, FUZZY_INPUTS):
        base = set_memberships(name, x, default)
        new = set_memberships(name, x, tuned)
        for color, set_name in zip(COLORS[name], FUZZY_SETS[name]):
            ax.plot(x, base[set_name], color="#9aa0a6", ls="--", lw=1.2,
                    label=f"{set_name} default")
            ax.plot(x, new[set_name], color=color, lw=1.9,
                    label=f"{set_name} tuned")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(INPUT_LABEL[name], fontsize=8)
        ax.set_ylabel(r"$\mu$", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6, ncol=2, loc="best")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_convergence(histories: dict, path: str,
                     default_fitness: float | None = None,
                     title: str = "GA convergence") -> str:
    """Best (solid) and mean (dashed) training fitness per GA generation.

    ``histories`` maps GA seed -> DataFrame with generation/best/mean/worst.
    """
    _prepare(path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    palette = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    for i, (seed, hist) in enumerate(sorted(histories.items())):
        color = palette[i % len(palette)]
        ax.plot(hist["generation"], hist["best"], color=color, lw=1.8,
                marker="o", ms=3, label=f"seed {seed} best")
        ax.plot(hist["generation"], hist["mean"], color=color, lw=1.0,
                ls="--", alpha=0.7, label=f"seed {seed} mean")
    if default_fitness is not None:
        ax.axhline(default_fitness, color="#444444", ls=":", lw=1.2,
                   label="default FIS fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Training fitness (higher is better)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
