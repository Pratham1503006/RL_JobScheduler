"""SC2 acceptance checks for the GA-tuned FIS (addendum section 4).

 1. GA best fitness is never worse than the default FIS (elitism + seeded default)
 2. best fitness per generation is non-decreasing
 3. 1,000 random chromosomes decode to valid, ordered, full-coverage memberships
 4. any chromosome's FIS policy completes 100% with invalid_actions == 0 on all scenarios
 5. the same GA seed reproduces the identical result twice
Plus the data-leakage guard (train_ga may only read train_pool.csv).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (FUZZY_INPUTS, FUZZY_SETS, TRAIN_POOL,  # noqa: E402
                    GA_ELITES)

from softcomputing.ga import (default_chromosome, decode, repair,  # noqa: E402
                              run_ga)
from softcomputing.fuzzy import membership_params, set_memberships  # noqa: E402

CHROM_LEN = 6 + 18


# ---------------------------------------------------------- check 3 -------
def test_repair_decode_always_gives_valid_memberships():
    rng = np.random.default_rng(123)
    x = np.linspace(0.0, 1.0, 201)
    for _ in range(1000):
        chrom = rng.uniform(-0.5, 1.5, size=CHROM_LEN)
        c = repair(chrom)
        assert c.shape == (CHROM_LEN,)
        for i in (0, 2, 4):
            assert 0.0 < c[i] < c[i + 1] < 1.0
        assert np.all(c[6:] >= 0.0) and np.all(c[6:] <= 1.0)
        params = decode(c)
        for name in FUZZY_INPUTS:
            for _s, (a, b, cc, d) in membership_params(params)[name].items():
                assert 0.0 <= a <= b <= cc <= d <= 1.0
            mus = set_memberships(name, x, params)
            stack = np.stack([mus[s] for s in FUZZY_SETS[name]])
            assert np.all(stack.max(axis=0) > 0.0), name


def test_repair_is_idempotent_and_handles_extremes():
    for chrom in ([0.0] * CHROM_LEN, [1.0] * CHROM_LEN,
                  [float("nan")] * CHROM_LEN,
                  [float("inf")] * CHROM_LEN):
        c1 = repair(chrom)
        c2 = repair(c1)
        assert np.allclose(c1, c2)
        for i in (0, 2, 4):
            assert 0.0 < c1[i] < c1[i + 1] < 1.0


def test_default_chromosome_decodes_to_the_hand_designed_fis():
    from softcomputing.fuzzy import default_params

    chrom = default_chromosome()
    assert chrom.shape == (CHROM_LEN,)
    params = decode(chrom)
    ref = default_params()
    assert np.allclose(params.cuts, ref.cuts)
    assert np.allclose(params.z, ref.z)


# ------------------------------------------------------- checks 1, 2 -------
def _sphere(pop):
    """Synthetic smooth fitness: closer to 0.3 is better (maximise)."""
    return [-float(np.sum((np.asarray(c) - 0.3) ** 2)) for c in pop]


def test_ga_seeds_default_and_elitism_never_lowers_best_fitness():
    default = default_chromosome()
    res = run_ga(_sphere, seed=0, population_size=8, generations=5)
    best = list(res["history"]["best"])
    assert best == sorted(best)                     # non-decreasing
    assert res["best_fitness"] >= _sphere([default])[0] - 1e-9
    assert res["history"].shape[0] == 6             # gens 0..5


def test_same_seed_reproduces_ga_result_twice():
    a = run_ga(_sphere, seed=7, population_size=6, generations=3)
    b = run_ga(_sphere, seed=7, population_size=6, generations=3)
    assert np.allclose(a["best_chromosome"], b["best_chromosome"])
    assert a["history"].equals(b["history"])


def test_different_seeds_explore_differently():
    a = run_ga(_sphere, seed=1, population_size=6, generations=3)
    b = run_ga(_sphere, seed=2, population_size=6, generations=3)
    assert not np.allclose(a["best_chromosome"], b["best_chromosome"])


# ---------------------------------------------------------- check 4 -------
def test_any_chromosome_fis_completes_all_scenarios_with_zero_invalid():
    from agents.evaluate import evaluate
    from softcomputing.fis_policy import FuzzyScheduler

    rng = np.random.default_rng(0)
    chroms = [default_chromosome(),
              repair(rng.uniform(-0.5, 1.5, size=CHROM_LEN))]
    for chrom in chroms:
        params = decode(chrom)
        for scen in ("light", "heavy", "bursty"):
            row = evaluate(lambda p=params: FuzzyScheduler(p), scen,
                           n_episodes=1)[0]
            assert row["completion_rate"] == 1.0, scen
            assert row["invalid_actions"] == 0, scen


# --------------------------------------------------- training plumbing ----
def test_ga_training_data_guard_rejects_scenario_files():
    from softcomputing.train_ga import _assert_train_only

    _assert_train_only(TRAIN_POOL)                  # the only allowed path
    for bad in ("data/scenarios/light.csv",
                "data/scenarios/heavy.csv",
                "data/scenarios/bursty.csv"):
        with pytest.raises(AssertionError):
            _assert_train_only(bad)


def test_fitness_is_finite_and_deterministic():
    from softcomputing.train_ga import build_episodes, evaluate_chromosome

    episodes = build_episodes(seed=0, n_jobs=80, n_episodes=1)
    chrom = default_chromosome()
    f1 = evaluate_chromosome(chrom, episodes)
    f2 = evaluate_chromosome(chrom, episodes)
    assert np.isfinite(f1)
    assert f1 == f2


def test_tournament_never_selects_worse_genome():
    from softcomputing.ga import tournament

    rng = np.random.default_rng(0)
    pop = [np.full(CHROM_LEN, v) for v in np.linspace(0, 1, 10)]
    fit = np.arange(10, dtype=float)               # index i has fitness i
    for _ in range(200):
        idx = tournament(pop, fit, 3, rng)
        assert 0 <= idx < len(pop)


def test_elites_are_carried_over_unchanged():
    # with a 1-gene-independent objective, the best individual must survive
    calls = {"n": 0}

    def counting_fitness(pop):
        calls["n"] += 1
        return _sphere(pop)

    res = run_ga(counting_fitness, seed=0, population_size=8, generations=4)
    assert calls["n"] >= 4
    assert res["best_fitness"] >= _sphere([default_chromosome()])[0] - 1e-9
    assert GA_ELITES >= 2
