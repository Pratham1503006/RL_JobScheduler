"""tests for the CloudSchedulingEnv (plan section 5 acceptance checks)."""

import os
import sys

import gymnasium as gym
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CPU_CAP, MEM_CAP, N_SERVERS  # noqa: E402

from env.cluster import Server  # noqa: E402
from env.scheduling_env import CloudSchedulingEnv, _rho_of_jobs  # noqa: E402


# ---------------------------------------------------------- check 1 -------
def test_server_place_release_restores_capacity_exactly():
    s = Server()
    before = (s.free_cpu, s.free_mem)
    s.place(0.3, 0.2)
    s.place(0.1, 0.05)
    s.release(0.3, 0.2)
    s.release(0.1, 0.05)
    assert s.free_cpu == pytest.approx(before[0], abs=1e-9)
    assert s.free_mem == pytest.approx(before[1], abs=1e-9)


# ---------------------------------------------------------- check 2 -------
def test_capacity_never_negative_and_util_in_range():
    env = CloudSchedulingEnv(mode="eval", scenario="light")
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(300):
        action = int(rng.integers(0, N_SERVERS + 1))
        obs, r, term, trunc, info = env.step(action)
        assert obs.min() >= 0.0 and obs.max() <= 1.0 + 1e-6
        for s in env.cluster.servers:
            assert s.free_cpu >= -1e-9 and s.free_mem >= -1e-9
            assert 0.0 <= s.used_cpu / s.cpu_cap <= 1.0 + 1e-9
        if term or trunc:
            assert 0.0 <= info["avg_cpu_util"] <= 1.0
            obs, _ = env.reset(seed=0)


# ---------------------------------------------------------- check 3 -------
def test_random_policy_completes_20_light_episodes():
    for ep in range(20):
        env = CloudSchedulingEnv(mode="eval", scenario="light")
        obs, _ = env.reset(seed=ep)
        rng = np.random.default_rng(ep)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(int(rng.integers(0, N_SERVERS + 1)))
            done = term or trunc
        assert term, f"episode {ep} truncated instead of finishing"
        assert info["completion_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------- check 4 -------
def test_first_fit_policy_completes_100_percent():
    env = CloudSchedulingEnv(mode="eval", scenario="light")
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        action = next((k for k in range(N_SERVERS) if env.can_fit(k)), N_SERVERS)
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc
    assert term
    assert info["completion_rate"] == pytest.approx(1.0)
    assert info["invalid_actions"] == 0


# ---------------------------------------------------------- check 5 -------
def test_invalid_action_increments_counter_and_advances_time():
    # WHY heavy scenario: light load never fills a server at t=0; under heavy
    # load a full server appears quickly, giving a deterministic invalid pick.
    env = CloudSchedulingEnv(mode="eval", scenario="heavy")
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    bad = None
    for _ in range(500):
        head_cpu, head_mem = env.head_job()
        bad = next((k for k in range(N_SERVERS) if not env.can_fit(k)), None)
        if bad is not None:
            break
        obs, r, term, trunc, info = env.step(int(rng.integers(0, N_SERVERS + 1)))
        if term or trunc:
            env.reset(seed=0)
    assert bad is not None, "no full server appeared in 500 steps of heavy"
    t_before = env.now
    invalid_before = env._invalid
    env.step(bad)
    assert env._invalid == invalid_before + 1
    assert env.now > t_before, "invalid action did not advance time"


# ---------------------------------------------------------- check 6 -------
def test_same_seed_gives_identical_metrics():
    results = []
    for _ in range(2):
        env = CloudSchedulingEnv(mode="eval", scenario="heavy")
        obs, _ = env.reset(seed=123)
        rng = np.random.default_rng(123)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(int(rng.integers(0, N_SERVERS + 1)))
            done = term or trunc
        results.append(info)
    a, b = results
    for key in a:
        assert a[key] == pytest.approx(b[key], abs=1e-9), key


# ---------------------------------------------------------- check 7 -------
def test_env_checker_passes():
    from gymnasium.utils.env_checker import check_env
    env = CloudSchedulingEnv(mode="eval", scenario="light")
    check_env(env, skip_render_check=True)


# ----------------------------------------------- deadlock impossibility ---
def test_job_always_fits_empty_server():
    # WHY: requests are clipped to <= 1.0 and capacity is exactly 1.0, so an
    # empty server fits any job -> no deadlock (plan 5.1).
    env = CloudSchedulingEnv(mode="eval", scenario="heavy")
    env.reset(seed=0)
    assert env.jobs.cpu.max() <= CPU_CAP + 1e-12
    assert env.jobs.mem.max() <= MEM_CAP + 1e-12


def test_train_episode_sampling_deterministic_and_load_range():
    env = CloudSchedulingEnv(mode="train")
    obs1, _ = env.reset(seed=7)
    rho1 = _rho_of_jobs(env.jobs.arrival, env.jobs.duration, env.jobs.cpu)
    obs2, _ = env.reset(seed=7)
    assert np.allclose(obs1, obs2)
    assert rho1 == _rho_of_jobs(env.jobs.arrival, env.jobs.duration, env.jobs.cpu)
    assert 0.30 - 1e-9 <= rho1 <= 1.00 + 1e-9
