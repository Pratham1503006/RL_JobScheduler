"""train_ga.py -- evolve the FIS membership breakpoints and rule outputs (SC2).

    python -m softcomputing.train_ga --seed 0
    python -m softcomputing.train_ga --seed 0 --generations 3 --population 6  # smoke
    python -m softcomputing.train_ga --finalize      # plots + selected seed

DATA LEAKAGE RULE (addendum 7): training reads ``data/scenarios/train_pool.csv``
ONLY.  ``_assert_train_only`` fails loudly if any other path is passed, so the
held-out light/heavy/bursty scenarios can never reach the GA.

Fitness (maximised):
    -(mean avg_wait) - GA_PENALTY_INVALID * mean(invalid_actions)
on ``GA_EVAL_EPISODES`` fixed 500-job windows drawn once per run (same
episodes for every chromosome AND every GA seed, so fitness values are
comparable and differences come from the chromosome only).
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (BASE_SEED, FUZZY_INPUTS, FUZZY_RULES,  # noqa: E402
                    GA_EPISODE_JOBS, GA_EVAL_EPISODES, GA_GENERATIONS,
                    GA_PENALTY_INVALID, GA_POP_SIZE, GA_RHO_RANGE, GA_SEEDS,
                    GA_WORKERS, N_SERVERS, RESULTS_DIR, TRAIN_POOL)

from env.scheduling_env import CloudSchedulingEnv, JobSet  # noqa: E402
from softcomputing.fis_policy import FuzzyScheduler  # noqa: E402
from softcomputing.ga import (GA_LOG_DIR, best_json_path,  # noqa: E402
                              decode, default_chromosome, history_path,
                              load_best, run_ga)
from softcomputing.fuzzy import default_params, membership_params  # noqa: E402
from softcomputing.plots import plot_convergence, plot_overlay  # noqa: E402

SOFT_DIR = os.path.join(RESULTS_DIR, "softcomputing")


# ------------------------------------------------------- training data ------
def _assert_train_only(path) -> str:
    """Hard guard: the GA may only ever read the training pool."""
    got = os.path.abspath(str(path))
    allowed = os.path.abspath(TRAIN_POOL)
    assert got == allowed, (
        f"GA training may read ONLY {TRAIN_POOL!r}, refusing {path!r}. "
        "light/heavy/bursty are the held-out test set (addendum 7).")
    return got


class FixedEpisodeEnv(CloudSchedulingEnv):
    """Eval-mode env that replays one pre-built JobSet (never reads a CSV)."""

    def __init__(self, jobs: JobSet):
        super().__init__(mode="eval", scenario=None)
        self._fixed_jobs = jobs

    def _sample_jobs(self, rng):  # noqa: ARG002 - no sampling by design
        return self._fixed_jobs


def build_episodes(seed: int = BASE_SEED, n_jobs: int | None = None,
                   n_episodes: int | None = None,
                   rho_range=None) -> list[JobSet]:
    """Contiguous 500-job windows rescaled to rho targets spread over [0.6, 1].

    WHY contiguous windows: they keep the real arrival-time structure.  WHY a
    fixed seed: every chromosome sees exactly the same episodes.
    """
    n_jobs = GA_EPISODE_JOBS if n_jobs is None else int(n_jobs)
    n_episodes = GA_EVAL_EPISODES if n_episodes is None else int(n_episodes)
    rho_range = GA_RHO_RANGE if rho_range is None else rho_range

    pool_df = pd.read_csv(_assert_train_only(TRAIN_POOL))
    rng = np.random.default_rng(seed)
    # WHY evenly spaced: guarantees the five episodes sample low/mid/high load
    # instead of accidentally clustering on one rho.
    rho_targets = np.linspace(float(rho_range[0]), float(rho_range[1]),
                              n_episodes)

    episodes = []
    for rho_target in rho_targets:
        start = int(rng.integers(0, len(pool_df) - n_jobs))
        df = pool_df.iloc[start:start + n_jobs].copy()
        arrival = df["arrival_time"].to_numpy(np.float64)
        span = arrival.max() - arrival.min()
        rho_cur = float((df["cpu_request"] * df["duration"]).sum()
                        / (span * N_SERVERS))
        df["arrival_time"] = arrival * (rho_cur / rho_target)
        df["arrival_time"] -= df["arrival_time"].min()
        episodes.append(JobSet(df.sort_values("arrival_time", kind="stable")))
    return episodes


# ------------------------------------------------------------- fitness ------
def evaluate_chromosome(chrom, episodes) -> float:
    """Run the FIS encoded by ``chrom`` on every episode (deterministic)."""
    policy = FuzzyScheduler(decode(chrom))
    waits, invalids = [], []
    for jobs in episodes:
        env = FixedEpisodeEnv(jobs)
        obs, _ = env.reset(seed=0)
        done, info = False, {}
        while not done:
            obs, _, terminated, truncated, info = env.step(
                int(policy(obs, env)))
            done = terminated or truncated
        waits.append(float(info["avg_wait"]))
        invalids.append(float(info["invalid_actions"]))
    return (-float(np.mean(waits))
            - GA_PENALTY_INVALID * float(np.mean(invalids)))


# multiprocessing plumbing: the worker + initializer must be module-level so
# Windows' spawn start method can re-import them.
_WORKER_EPISODES: list[JobSet] | None = None


def _init_worker(episodes):
    global _WORKER_EPISODES
    _WORKER_EPISODES = episodes


def _worker_fitness(chrom):
    return evaluate_chromosome(chrom, _WORKER_EPISODES)


class EpisodeFitness:
    """``fitness_fn(population) -> list[float]``, optionally parallel."""

    def __init__(self, episodes, parallel: bool = True):
        self.episodes = episodes
        self._pool = None
        if parallel:
            self._pool = Pool(processes=GA_WORKERS,
                              initializer=_init_worker,
                              initargs=(episodes,))

    def __call__(self, population):
        if self._pool is None:
            return [evaluate_chromosome(c, self.episodes) for c in population]
        return self._pool.map(_worker_fitness, population)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------- outputs ----
def save_seed_outputs(seed: int, result: dict, default_fitness: float) -> dict:
    os.makedirs(GA_LOG_DIR, exist_ok=True)
    result["history"].to_csv(history_path(seed), index=False)

    params = decode(result["best_chromosome"])
    record = {
        "seed": int(seed),
        "fitness": float(result["best_fitness"]),
        "default_fitness": float(default_fitness),
        "chromosome": [float(v) for v in result["best_chromosome"]],
        "cuts": {name: [float(params.cuts[2 * i]),
                        float(params.cuts[2 * i + 1])]
                 for i, name in enumerate(FUZZY_INPUTS)},
        "memberships": {name: {set_name: [float(v) for v in corners]
                               for set_name, corners in sets.items()}
                        for name, sets in membership_params(params).items()},
        "rules": [[leftover, load, pressure, float(z)]
                  for (leftover, load, pressure, _z), z
                  in zip(FUZZY_RULES, params.z)],
        "settings": {
            "population": GA_POP_SIZE,
            "generations": GA_GENERATIONS,
            "episodes": GA_EVAL_EPISODES,
            "episode_jobs": GA_EPISODE_JOBS,
            "rho_range": list(GA_RHO_RANGE),
            "train_pool": TRAIN_POOL,
        },
    }
    with open(best_json_path(seed), "w") as f:
        json.dump(record, f, indent=1)
    return record


def write_rule_tables(best: dict) -> str:
    """Side-by-side default vs tuned rule table (addendum 4.5)."""
    tuned = {(a, b, c): z for a, b, c, z in best["rules"]}
    lines = [
        "Zero-order Sugeno rule base: default FIS vs GA-FIS "
        f"(selected seed {best['seed']})",
        "leftover | load   | pressure | default z | tuned z",
        "---------+--------+----------+-----------+--------",
    ]
    for leftover, load, pressure, z in FUZZY_RULES:
        lines.append(f"{leftover:<8} | {load:<6} | {pressure:<8} | "
                     f"{z:9.2f} | {tuned[(leftover, load, pressure)]:7.2f}")
    path = os.path.join(GA_LOG_DIR, "rule_table_default_vs_ga_fis.txt")
    os.makedirs(GA_LOG_DIR, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def finalize() -> dict:
    """Select the best training seed and write the shared GA artifacts."""
    records = [r for r in (load_best(s) for s in GA_SEEDS) if r is not None]
    if not records:
        raise SystemExit("no GA runs found; run "
                         "`python -m softcomputing.train_ga --seed N` first")
    best = max(records, key=lambda r: r["fitness"])

    os.makedirs(SOFT_DIR, exist_ok=True)
    tuned = decode(best["chromosome"])
    plot_overlay(default_params(), tuned,
                 os.path.join(SOFT_DIR, "ga_fis_memberships.png"),
                 title=f"GA-FIS (seed {best['seed']}) vs default FIS")

    histories = {}
    for rec in records:
        path = history_path(rec["seed"])
        if os.path.exists(path):
            histories[rec["seed"]] = pd.read_csv(path)
    plot_convergence(histories,
                     os.path.join(SOFT_DIR, "ga_convergence.png"),
                     default_fitness=best.get("default_fitness"),
                     title="GA-FIS training convergence (3 seeds)")

    summary = pd.DataFrame([
        {"seed": r["seed"], "fitness": r["fitness"],
         "default_fitness": r.get("default_fitness"),
         "selected": r["seed"] == best["seed"]}
        for r in records
    ]).sort_values("seed")
    summary.to_csv(os.path.join(GA_LOG_DIR, "ga_seed_summary.csv"), index=False)
    with open(os.path.join(GA_LOG_DIR, "ga_selected.json"), "w") as f:
        json.dump({"seed": best["seed"], "fitness": best["fitness"]}, f,
                  indent=1)
    write_rule_tables(best)
    print(f"selected GA seed {best['seed']} (training fitness "
          f"{best['fitness']:.3f}); artifacts in {SOFT_DIR} and {GA_LOG_DIR}")
    return best


# ------------------------------------------------------------------ main ----
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0, choices=list(GA_SEEDS))
    ap.add_argument("--generations", type=int, default=GA_GENERATIONS)
    ap.add_argument("--population", type=int, default=GA_POP_SIZE)
    ap.add_argument("--episodes", type=int, default=GA_EVAL_EPISODES)
    ap.add_argument("--jobs", type=int, default=GA_EPISODE_JOBS)
    ap.add_argument("--serial", action="store_true",
                    help="evaluate the population in-process (debugging)")
    ap.add_argument("--finalize", action="store_true",
                    help="write plots/selection from the saved seed runs")
    args = ap.parse_args()

    if args.finalize:
        finalize()
        return

    episodes = build_episodes(seed=BASE_SEED, n_jobs=args.jobs,
                              n_episodes=args.episodes)
    default_fitness = evaluate_chromosome(default_chromosome(), episodes)
    print(f"seed {args.seed}: {len(episodes)} episodes, default FIS fitness "
          f"{default_fitness:.3f}")

    with EpisodeFitness(episodes, parallel=not args.serial) as fitness_fn:
        result = run_ga(fitness_fn, seed=args.seed,
                        population_size=args.population,
                        generations=args.generations)

    rec = save_seed_outputs(args.seed, result, default_fitness)
    os.makedirs(SOFT_DIR, exist_ok=True)
    plot_overlay(default_params(), decode(rec["chromosome"]),
                 os.path.join(SOFT_DIR, f"ga_seed{args.seed}_memberships.png"),
                 title=f"GA-FIS seed {args.seed} vs default FIS")
    print(f"seed {args.seed}: best fitness {rec['fitness']:.3f} "
          f"(default {default_fitness:.3f}) -> {best_json_path(args.seed)}")


if __name__ == "__main__":
    main()
