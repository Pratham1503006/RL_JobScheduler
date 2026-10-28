"""SC1 acceptance checks for the Fuzzy Inference Scheduler (addendum section 3).

Every check in addendum section 3 is a test here:
  1. membership coverage / support
  2. infer() is bounded and never NaN
  3. FIS completes 100% on `light` with invalid_actions == 0
  4. high pressure -> tighter server, low pressure -> emptier server
  5. the 8 default membership functions are plotted
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FUZZY_INPUTS, FUZZY_SETS, N_SERVERS, RESULTS_DIR  # noqa: E402

from softcomputing.fis_policy import FuzzyScheduler  # noqa: E402
from softcomputing.fuzzy import (default_params, infer, infer_batch,  # noqa: E402
                                 membership_params, set_memberships)


# ---------------------------------------------------------- check 1 -------
def test_memberships_cover_unit_interval_and_vanish_outside_support():
    params = default_params()
    x = np.linspace(0.0, 1.0, 1001)
    for name in FUZZY_INPUTS:
        mu = set_memberships(name, x, params)
        assert set(mu) == set(FUZZY_SETS[name]), name
        stack = np.stack([mu[s] for s in FUZZY_SETS[name]])
        assert np.all(np.isfinite(stack))
        assert stack.min() >= -1e-12
        assert stack.max() <= 1.0 + 1e-12
        # full coverage: at every x at least one linguistic set is active.
        assert np.all(stack.max(axis=0) > 0.0), name
        # nothing outside the [0, 1] support.
        outside = set_memberships(name, np.array([-1.0, 2.0]), params)
        for s in FUZZY_SETS[name]:
            assert np.allclose(outside[s], 0.0), (name, s)


# ---------------------------------------------------------- check 2 -------
def test_infer_is_bounded_and_never_nan():
    params = default_params()
    rng = np.random.default_rng(0)
    sample = rng.random((1000, 3))
    vals = np.array([infer(lo, ld, pr, params) for lo, ld, pr in sample])
    assert np.all(np.isfinite(vals))
    assert vals.min() >= 0.0 and vals.max() <= 1.0
    # the vectorised path used in the hot loop must agree with the scalar one.
    batch = infer_batch(sample[:, 0], sample[:, 1], sample[:, 2], params)
    assert batch.shape == (1000,)
    assert np.allclose(batch, vals)


# ---------------------------------------------------------- check 3 -------
def test_fis_completes_light_with_no_invalid_actions():
    from agents.evaluate import evaluate

    rows = evaluate(lambda: FuzzyScheduler(), "light", n_episodes=1)
    assert rows[0]["completion_rate"] == 1.0
    assert rows[0]["invalid_actions"] == 0


# ---------------------------------------------------------- check 4 -------
class _Server:
    def __init__(self, free_cpu, free_mem):
        self.free_cpu = free_cpu
        self.free_mem = free_mem


class _FakeEnv:
    """Only the surface the FIS policy uses (can_fit / head_job / servers)."""

    def __init__(self, servers, cpu, mem):
        self.cluster = SimpleNamespace(servers=servers)
        self._cpu, self._mem = cpu, mem
        # Surface used only by explain() (audit metadata, not the decision).
        self.now = 0.0
        self._jobs_done = 0
        self.queue = [0]

    def can_fit(self, k):
        if k >= len(self.cluster.servers):
            return False
        s = self.cluster.servers[k]
        return (s.free_cpu + 1e-9 >= self._cpu
                and s.free_mem + 1e-9 >= self._mem)

    def head_job(self):
        return self._cpu, self._mem


def _obs(pressure: float) -> np.ndarray:
    obs = np.zeros(2 * N_SERVERS + 3, dtype=np.float32)
    obs[2 * N_SERVERS + 2] = pressure
    return obs


def test_fis_prefers_tight_server_under_high_pressure():
    # server 0 is nearly full, server 1 is nearly empty; request is tiny.
    env = _FakeEnv([_Server(0.05, 0.05), _Server(0.95, 0.95)], 0.02, 0.02)
    action = FuzzyScheduler()(_obs(0.90), env)
    assert action == 0


def test_fis_prefers_empty_server_under_low_pressure():
    env = _FakeEnv([_Server(0.05, 0.05), _Server(0.95, 0.95)], 0.02, 0.02)
    action = FuzzyScheduler()(_obs(0.10), env)
    assert action == 1


def test_fis_returns_wait_when_nothing_fits():
    env = _FakeEnv([_Server(0.01, 0.01), _Server(0.02, 0.02)], 0.5, 0.5)
    assert FuzzyScheduler()(_obs(0.5), env) == N_SERVERS


# ------------------------------------------------ explanation audit trail ----
def test_explain_agrees_with_the_policy_and_the_arithmetic():
    """The demo must show the SAME decision the policy actually makes."""
    env = _FakeEnv([_Server(0.05, 0.05), _Server(0.95, 0.95),
                    _Server(0.55, 0.55)], 0.02, 0.02)
    policy = FuzzyScheduler()
    obs = _obs(0.10)

    rec = policy.explain(obs, env)
    assert rec["action"] == policy(obs, env) == rec["server"]
    assert rec["server"] in (0, 1, 2)

    # Every reported membership is a real degree in [0, 1].
    all_mu = [v for sets in rec["memberships"].values() for v in sets.values()]
    assert all(0.0 <= v <= 1.0 for v in all_mu)

    # Fired rules are exactly those whose strength equals min(memberships).
    assert rec["rules"], "at least one rule must fire for a crisp input"
    for r in rec["rules"]:
        expected = min(rec["memberships"]["leftover"][r["leftover"]],
                       rec["memberships"]["load"][r["load"]],
                       rec["memberships"]["pressure"][r["pressure"]])
        assert r["strength"] == pytest.approx(expected, abs=1e-4)
        assert r["contribution"] == pytest.approx(r["strength"] * r["z"],
                                                  abs=1e-4)

    # Weighted average is what the policy scored for the chosen candidate.
    assert rec["score"] == pytest.approx(rec["num"] / rec["den"], abs=1e-3)
    chosen = [c for c in rec["candidates"] if c["server"] == rec["server"]][0]
    assert rec["score"] == pytest.approx(chosen["score"], abs=1e-3)


def test_explain_uses_tuned_rule_consequents():
    """GA-FIS tunes z; the audit trail must report the TUNED value, not config."""
    params = default_params()
    params.z = params.z.copy()
    params.z[12] = 0.123            # FUZZY_RULES[12] = LOOSE / LOW / CALM
    env = _FakeEnv([_Server(0.95, 0.95)], 0.02, 0.02)
    rec = FuzzyScheduler(params).explain(_obs(0.10), env)
    fired = {f"{r['leftover']}/{r['load']}/{r['pressure']}": r["z"]
             for r in rec["rules"]}
    assert fired["LOOSE/LOW/CALM"] == pytest.approx(0.123)
    assert rec["score"] == pytest.approx(0.123, abs=1e-3)


def test_explain_wait_when_nothing_fits():
    env = _FakeEnv([_Server(0.01, 0.01)], 0.5, 0.5)
    rec = FuzzyScheduler().explain(_obs(0.5), env)
    assert rec["action"] == N_SERVERS
    assert rec["server"] is None
    assert rec["candidates"] == []


# ---------------------------------------------------------- check 5 -------
def test_membership_plot_is_written():
    from softcomputing.plots import plot_memberships

    path = os.path.join(RESULTS_DIR, "softcomputing", "fis_default_memberships.png")
    plot_memberships(default_params(), path)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


# ------------------------------------------------------- regression -------
def test_default_rule_table_is_complete_and_unique():
    from config import FUZZY_RULES

    assert len(FUZZY_RULES) == 18
    combos = {(a, b, c) for a, b, c, _z in FUZZY_RULES}
    assert len(combos) == 18
    for leftover, load, pressure, z in FUZZY_RULES:
        assert leftover in FUZZY_SETS["leftover"]
        assert load in FUZZY_SETS["load"]
        assert pressure in FUZZY_SETS["pressure"]
        assert 0.0 <= z <= 1.0
    # design logic: BUSY prefers packing, CALM prefers balancing.
    z = {(a, b, c): v for a, b, c, v in FUZZY_RULES}
    assert z[("TIGHT", "HIGH", "BUSY")] > z[("LOOSE", "LOW", "BUSY")]
    assert z[("LOOSE", "LOW", "CALM")] > z[("TIGHT", "HIGH", "CALM")]


def test_trap_handles_triangles_and_both_shoulders():
    from softcomputing.fuzzy import trap

    # triangle (a, b, b, d): peak at b
    assert trap(0.35, 0.1, 0.35, 0.35, 0.6) == pytest.approx(1.0)
    assert trap(0.0, 0.1, 0.35, 0.35, 0.6) == pytest.approx(0.0)
    # left shoulder: 1 up to c, falls to 0 at d
    assert trap(0.0, 0.0, 0.0, 0.1, 0.4) == pytest.approx(1.0)
    assert trap(0.4, 0.0, 0.0, 0.1, 0.4) == pytest.approx(0.0)
    # right shoulder: rises from a to b, 1 thereafter
    assert trap(1.0, 0.3, 0.7, 1.0, 1.0) == pytest.approx(1.0)
    assert trap(0.0, 0.3, 0.7, 1.0, 1.0) == pytest.approx(0.0)
