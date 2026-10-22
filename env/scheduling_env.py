"""scheduling_env.py -- CloudSchedulingEnv, an event-driven gymnasium.Env.

One agent step = one decision about the job at the HEAD of the FIFO queue.
Time only moves when the agent WAITs, makes an invalid choice, or the queue
empties (fast-forward).

WHY event-driven instead of fixed ticks: a real cluster scheduler acts at
event instants (arrivals/completions); fixed ticks would multiply the number
of decisions by the tick rate without adding any scheduling freedom.

WHY the reward looks like this (plan 5.2): summed over an episode, the
-W_WAIT * queue_len * dt / WAIT_NORM term equals -(total wait time) /
WAIT_NORM, because each waiting job accumulates queue_len * dt waiting-seconds
while time dt passes (Little's law style). So maximizing return directly
minimizes total waiting time.

WHY numpy arrays instead of pandas rows in the hot loop: pandas 3.0.6
segfaults when Series views from per-episode DataFrames are held across
resets; plain arrays are also much faster for the millions of RL steps.
"""

import os
from collections import deque

import gymnasium as gym
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (MAX_STEPS_FACTOR, N_SERVERS, QUEUE_NORM,
                    RHO_TRAIN_RANGE, SCEN_DIR, TRAIN_EPISODE_JOBS, TRAIN_POOL,
                    WAIT_NORM, W_INVALID, W_PLACE, W_UTIL, W_WAIT)  # noqa: E402

from env.cluster import Cluster  # noqa: E402

_CSV_CACHE: dict[str, pd.DataFrame] = {}


def _load_csv(path: str) -> pd.DataFrame:
    if path not in _CSV_CACHE:
        _CSV_CACHE[path] = pd.read_csv(path)
    return _CSV_CACHE[path]


def _rho_of_jobs(arr: np.ndarray, dur: np.ndarray, cpu: np.ndarray) -> float:
    span = arr.max() - arr.min()
    return float((cpu * dur).sum() / (span * N_SERVERS))


class JobSet:
    """Immutable numpy view of one episode's jobs, sorted by arrival time."""

    def __init__(self, df: pd.DataFrame):
        self.arrival = df["arrival_time"].to_numpy(np.float64)
        self.duration = df["duration"].to_numpy(np.float64)
        self.cpu = df["cpu_request"].to_numpy(np.float64)
        self.mem = df["mem_request"].to_numpy(np.float64)
        self.n = len(df)


class CloudSchedulingEnv(gym.Env):
    """Event-driven cluster scheduling environment (plan section 5)."""

    metadata = {"render_modes": []}

    def __init__(self, mode: str = "train", scenario: str | None = None,
                 record_trace: bool = False):
        super().__init__()
        assert mode in ("train", "eval")
        self.mode = mode
        self.scenario = scenario
        self.record_trace = record_trace
        n_obs = 2 * N_SERVERS + 3
        # Observations are fractions in [0, 1] by construction (free capacity
        # fractions, request sizes clipped to 1, queue length normalized).
        self.observation_space = gym.spaces.Box(0.0, 1.0, (n_obs,), np.float32)
        # Actions 0..N-1 = assign head job to that server; N = WAIT.
        self.action_space = gym.spaces.Discrete(N_SERVERS + 1)

    # ------------------------------------------------------------ helpers ----
    def _sample_jobs(self, rng: np.random.Generator) -> JobSet:
        if self.mode == "eval":
            df = _load_csv(os.path.join(SCEN_DIR, f"{self.scenario}.csv"))
        else:
            pool = _load_csv(TRAIN_POOL)
            # WHY a random contiguous window: episodes stay i.i.d.-ish while
            # covering different workload mixes; scattered sampling would
            # destroy the arrival-time structure.
            start = int(rng.integers(0, len(pool) - TRAIN_EPISODE_JOBS))
            df = pool.iloc[start:start + TRAIN_EPISODE_JOBS]
            # Domain randomization: rescale arrival times so rho ~ U[0.3, 1.0]
            # (plan 5.3). WHY: the agent must not memorize one load level.
            arr = df["arrival_time"].to_numpy(np.float64)
            rho_target = float(rng.uniform(*RHO_TRAIN_RANGE))
            span = arr.max() - arr.min()
            rho_cur = float((df["cpu_request"] * df["duration"]).sum() / (span * N_SERVERS))
            df = df.assign(arrival_time=arr * (rho_cur / rho_target))
        # WHY shift to 0: episodes should not start with an arbitrary offset.
        # WHY .copy(): detach from the cached CSV frame entirely.
        df = df.copy()
        df["arrival_time"] = df["arrival_time"] - df["arrival_time"].min()
        return JobSet(df.sort_values("arrival_time", kind="stable"))

    def _observe(self) -> np.ndarray:
        obs = np.zeros(2 * N_SERVERS + 3, dtype=np.float32)
        obs[:N_SERVERS] = self.cluster.free_cpu_array()
        obs[N_SERVERS:2 * N_SERVERS] = self.cluster.free_mem_array()
        if self.queue:
            h = self.queue[0]
            obs[2 * N_SERVERS] = self.jobs.cpu[h]
            obs[2 * N_SERVERS + 1] = self.jobs.mem[h]
        obs[2 * N_SERVERS + 2] = min(1.0, len(self.queue) / QUEUE_NORM)
        return obs

    def _advance_time(self, t_target: float) -> None:
        """Move simulated time to t_target, processing events in order and
        accumulating time-weighted CPU utilization (area under the curve)."""
        assert t_target >= self.now - 1e-9
        while True:
            next_c = self.cluster.next_completion
            next_a = (self.jobs.arrival[self.next_arr]
                      if self.next_arr < self.jobs.n else float("inf"))
            t = min(next_c, next_a, t_target)
            if t > t_target + 1e-9:
                break
            # accumulate utilization over [now, t]
            self._util_area += self.cluster.mean_cpu_util() * (t - self.now)
            self.now = t
            if next_c <= next_a and next_c <= t_target + 1e-9:
                self._jobs_done += len(self.cluster.pop_completed(t))
            elif next_a <= t_target + 1e-9:
                self.queue.append(self.next_arr)
                self.next_arr += 1
            else:
                break
        self._util_area += self.cluster.mean_cpu_util() * (t_target - self.now)
        self.now = t_target
        if self.record_trace:
            self._trace.append({
                "time": round(float(self.now), 3),
                "server_cpu_used": [round(float(s.used_cpu), 4) for s in self.cluster.servers],
                "server_mem_used": [round(float(s.used_mem), 4) for s in self.cluster.servers],
                "queue_len": len(self.queue),
                "jobs_done": int(self._jobs_done),
            })

    def _wait_step(self) -> float:
        """Advance to the next completion (or next arrival if idle) and return
        the wait/utilization shaping reward for the elapsed interval."""
        if self.cluster.n_running > 0:
            t = self.cluster.next_completion
        elif self.next_arr < self.jobs.n:
            t = self.jobs.arrival[self.next_arr]
        else:
            # Nothing running and nothing left to arrive: time cannot advance.
            # No deadlock is possible (the head job always fits an empty
            # cluster), so this only wastes the agent's step.
            return 0.0
        dt = t - self.now
        r = -W_WAIT * len(self.queue) * dt / WAIT_NORM
        r += W_UTIL * self.cluster.mean_cpu_util() * dt / WAIT_NORM
        self._advance_time(t)
        return r

    def _fast_forward_no_decisions(self) -> None:
        """Advance while nothing is decidable: to the next arrival, or -- if
        none remain -- through completions until the queue refills or the
        episode can end. WHY a loop: one completion may leave other jobs
        running with an empty queue, which would strand the episode mid-way."""
        while not self.queue:
            if self.next_arr < self.jobs.n:
                self._advance_time(self.jobs.arrival[self.next_arr])
            elif self.cluster.n_running > 0:
                self._advance_time(self.cluster.next_completion)
            else:
                break

    # --------------------------------------------------------- gymnasium ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        self.jobs = self._sample_jobs(rng)
        self.cluster = Cluster()
        self.queue: deque = deque()   # holds job indices into self.jobs
        self.next_arr = 0
        self.now = 0.0
        self._util_area = 0.0
        self._jobs_done = 0
        self._placed = 0
        self._invalid = 0
        self._wait_times: list[float] = []
        self._steps = 0
        self._trace: list | None = [] if self.record_trace else None
        self._fast_forward_no_decisions()
        return self._observe(), {}

    def step(self, action):
        assert self.queue, "step() called with an empty queue (episode done?)"
        self._steps += 1
        reward = 0.0
        h = self.queue[0]

        if 0 <= action < N_SERVERS:
            if self.cluster.can_fit(action, self.jobs.cpu[h], self.jobs.mem[h]):
                self.cluster.place(action, self.jobs.cpu[h], self.jobs.mem[h],
                                   self.now + self.jobs.duration[h])
                self._wait_times.append(self.now - self.jobs.arrival[h])
                self._placed += 1
                self.queue.popleft()
                reward += W_PLACE
                # WHY fast-forward here: an empty queue means no decision is
                # possible; time jumping forward is not an agent step.
                if not self.queue:
                    self._fast_forward_no_decisions()
            else:
                # WHY penalize AND advance: an invalid action must never be a
                # way to loop in zero time; it costs the same as WAIT plus a
                # penalty (plan 5.1 transition 2).
                self._invalid += 1
                reward -= W_INVALID
                reward += self._wait_step()
        elif action == N_SERVERS:  # WAIT
            if self.cluster.n_running > 0:
                reward += self._wait_step()
            else:
                # WAIT with nothing running is meaningless -> invalid.
                self._invalid += 1
                reward -= W_INVALID
                self._fast_forward_no_decisions()
        else:
            raise ValueError(f"invalid action {action}")

        terminated = (self.next_arr == self.jobs.n and not self.queue
                      and self.cluster.n_running == 0)
        truncated = (not terminated
                     and self._steps >= MAX_STEPS_FACTOR * self.jobs.n)
        info = {}
        if terminated or truncated:
            info = self._metrics()
        return self._observe(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------ metrics ----
    def _metrics(self) -> dict:
        makespan = self.now
        # WHY makespan only: _util_area already integrates the mean over ALL
        # servers (cluster.mean_cpu_util), so dividing by N again would
        # under-report utilization by 1/N.
        avg_util = self._util_area / makespan if makespan > 0 else 0.0
        waits = np.array(self._wait_times)
        return {
            "avg_wait": float(waits.mean()) if len(waits) else 0.0,
            "p95_wait": float(np.percentile(waits, 95)) if len(waits) else 0.0,
            "avg_cpu_util": float(np.clip(avg_util, 0.0, 1.0)),
            "completion_rate": self._placed / self.jobs.n,
            "invalid_actions": int(self._invalid),
            "makespan": float(makespan),
        }

    # For heuristics (capacity-aware policies need env introspection).
    def head_job(self):
        h = self.queue[0]
        return self.jobs.cpu[h], self.jobs.mem[h]

    def can_fit(self, k: int) -> bool:
        h = self.queue[0]
        return self.cluster.can_fit(k, self.jobs.cpu[h], self.jobs.mem[h])
