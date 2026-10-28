"""ga.py -- real-valued genetic algorithm over the FIS chromosome (SC2).

Chromosome (24 genes):
  genes 0..5   two cut points per input (leftover, load, pressure)
  genes 6..23  the 18 zero-order Sugeno consequents z_r, in config.FUZZY_RULES
               order

``repair`` is the safety net: for ANY input vector (out of range, swapped,
even non-finite) it returns sorted, clipped cut points with a minimum gap and
consequents in [0, 1].  ``decode`` then maps the repaired chromosome onto
ordered, full-coverage membership functions.  This is what makes the GA
search safe: no chromosome can produce an ill-formed fuzzy system.

The GA loop is fitness-agnostic -- callers pass ``fitness_fn(population) ->
list[float]`` (maximised).  ``train_ga.py`` supplies the episode-based fitness
and the multiprocessing pool; tests supply cheap synthetic objectives.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (FUZZY_CUTS_DEFAULT, FUZZY_INPUTS, FUZZY_MIN_GAP,  # noqa: E402
                    FUZZY_RULES, GA_BLX_ALPHA, GA_CROSSOVER_PROB, GA_ELITES,
                    GA_GENERATIONS, GA_INIT_SIGMA, GA_MUTATION_PROB,
                    GA_MUTATION_SIGMA, GA_POP_SIZE, GA_SEEDS, GA_TOURNAMENT,
                    RESULTS_DIR)

from softcomputing.fuzzy import FISParams, membership_params  # noqa: E402

N_MEMBRANE_GENES = 2 * len(FUZZY_INPUTS)
CHROMOSOME_LEN = N_MEMBRANE_GENES + len(FUZZY_RULES)
GA_LOG_DIR = os.path.join(RESULTS_DIR, "ga_logs")


# --------------------------------------------------------------- encoding ----
def default_chromosome() -> np.ndarray:
    """The hand-designed FIS encoded as a chromosome (the GA's seed)."""
    cuts = np.concatenate([FUZZY_CUTS_DEFAULT[name] for name in FUZZY_INPUTS])
    z = np.array([rule[3] for rule in FUZZY_RULES], dtype=float)
    return np.concatenate([cuts, z]).astype(float)


def repair(chrom) -> np.ndarray:
    """Sort/clip a chromosome into a valid FIS encoding (see module docstring)."""
    c = np.asarray(chrom, dtype=float).copy()
    if c.ndim != 1 or c.size != CHROMOSOME_LEN:
        raise ValueError(f"chromosome must have {CHROMOSOME_LEN} genes, "
                         f"got shape {c.shape}")
    # WHY neutral 0.5: keeps the result finite/valid even if a caller feeds
    # NaN/inf; the property test exercises this path.
    c = np.nan_to_num(c, nan=0.5, posinf=1.0, neginf=0.0)
    c = np.clip(c, 0.0, 1.0)

    lo, hi = FUZZY_MIN_GAP, 1.0 - FUZZY_MIN_GAP
    for i in range(0, N_MEMBRANE_GENES, 2):
        p1, p2 = sorted((float(c[i]), float(c[i + 1])))
        p1 = min(max(p1, lo), hi)
        p2 = min(max(p2, lo), hi)
        # WHY the -1e-12 slack: repair() must be idempotent; a gap of exactly
        # FUZZY_MIN_GAP can come back 1 ulp below it after the subtraction.
        if p2 - p1 < FUZZY_MIN_GAP - 1e-12:
            p2 = min(p1 + FUZZY_MIN_GAP, hi)
            p1 = p2 - FUZZY_MIN_GAP
        c[i], c[i + 1] = p1, p2

    c[N_MEMBRANE_GENES:] = np.clip(c[N_MEMBRANE_GENES:], 0.0, 1.0)
    return c


def decode(chrom) -> FISParams:
    """Validated chromosome -> a ready-to-use fuzzy system."""
    c = repair(chrom)
    return FISParams(cuts=c[:N_MEMBRANE_GENES].copy(),
                     z=c[N_MEMBRANE_GENES:].copy())


# -------------------------------------------------------------- operators ----
def tournament(population, fitness, size: int, rng) -> int:
    """Pick ``size`` random individuals, return the index of the fittest."""
    idx = rng.integers(0, len(population), size=size)
    best = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if fitness[i] > fitness[best]:
            best = i
    return best


def blx(p1, p2, alpha: float, rng):
    """BLX-alpha blend crossover: sample each gene around the parents' range."""
    a = np.asarray(p1, dtype=float)
    b = np.asarray(p2, dtype=float)
    low = np.minimum(a, b)
    high = np.maximum(a, b)
    spread = high - low
    c1 = rng.uniform(low - alpha * spread, high + alpha * spread)
    c2 = rng.uniform(low - alpha * spread, high + alpha * spread)
    return c1, c2


def mutate(chrom, sigma: float, prob: float, rng) -> np.ndarray:
    """Per-gene Gaussian mutation; caller applies repair() afterwards."""
    c = np.asarray(chrom, dtype=float)
    mask = rng.random(c.shape) < prob
    return c + mask * rng.normal(0.0, sigma, size=c.shape)


# ------------------------------------------------------------------ loop ----
def run_ga(fitness_fn, seed: int = 0, population_size: int | None = None,
           generations: int | None = None,
           init_chromosome=None) -> dict:
    """Maximise ``fitness_fn`` over the FIS chromosome.

    Elitism (GA_ELITES) carries the best individuals over unchanged, and the
    hand-designed FIS is individual 0 of generation 0, so the best training
    fitness can never drop below the default FIS -- by construction.
    """
    pop_size = GA_POP_SIZE if population_size is None else int(population_size)
    gens = GA_GENERATIONS if generations is None else int(generations)
    if pop_size < GA_ELITES:
        raise ValueError("population_size must be >= GA_ELITES")
    rng = np.random.default_rng(seed)
    base = (default_chromosome() if init_chromosome is None
            else repair(init_chromosome))

    population = [base.copy()]
    for _ in range(pop_size - 1):
        population.append(
            repair(base + rng.normal(0.0, GA_INIT_SIGMA, size=base.size)))

    history = []

    def evaluate(pop):
        fit = np.asarray(fitness_fn(pop), dtype=float)
        assert fit.shape == (len(pop),), "fitness_fn must return one value each"
        order = np.argsort(fit)[::-1]
        history.append({
            "generation": len(history),
            "best": float(fit[order[0]]),
            "mean": float(fit.mean()),
            "worst": float(fit.min()),
        })
        return fit, order

    for _gen in range(gens):
        fit, order = evaluate(population)
        # elitism: copy (not reference) so later mutation cannot alter them
        next_pop = [population[i].copy() for i in order[:GA_ELITES]]
        while len(next_pop) < pop_size:
            i = tournament(population, fit, GA_TOURNAMENT, rng)
            j = tournament(population, fit, GA_TOURNAMENT, rng)
            if rng.random() < GA_CROSSOVER_PROB:
                c1, c2 = blx(population[i], population[j], GA_BLX_ALPHA, rng)
            else:
                c1, c2 = population[i].copy(), population[j].copy()
            next_pop.append(repair(mutate(c1, GA_MUTATION_SIGMA,
                                          GA_MUTATION_PROB, rng)))
            if len(next_pop) < pop_size:
                next_pop.append(repair(mutate(c2, GA_MUTATION_SIGMA,
                                              GA_MUTATION_PROB, rng)))
        population = next_pop

    # one final evaluation so the returned best is the fitness of the
    # chromosome we actually return (and the history stays non-decreasing).
    fit, order = evaluate(population)
    return {
        "history": pd.DataFrame(history),
        "best_chromosome": population[int(order[0])].copy(),
        "best_fitness": float(fit[order[0]]),
        "population": population,
        "fitness": fit,
    }


# ------------------------------------------------------------------ logs ----
def history_path(seed: int) -> str:
    return os.path.join(GA_LOG_DIR, f"ga_seed{seed}_history.csv")


def best_json_path(seed: int) -> str:
    return os.path.join(GA_LOG_DIR, f"ga_seed{seed}_best.json")


def load_best(seed: int) -> dict | None:
    path = best_json_path(seed)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def select_best_ga() -> dict | None:
    """The saved chromosome with the best TRAINING fitness across GA seeds.

    WHY training fitness: the light/heavy/bursty scenarios are the held-out
    test set; selecting on them would leak test information into the policy
    (addendum 4.6 / 7).
    """
    best = None
    for seed in GA_SEEDS:
        rec = load_best(seed)
        if rec is not None and (best is None or rec["fitness"] > best["fitness"]):
            best = rec
    return best
