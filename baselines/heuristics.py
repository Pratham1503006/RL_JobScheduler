"""heuristics.py -- Round Robin, First-Fit, Best-Fit (plan section 6).

All policies are callables `policy(obs, env) -> action` operating on the SAME
env as the RL agents. WHY capacity-aware: a heuristic that assigns jobs to
full servers would be a strawman; the fair comparison requires all policies
to respect capacity and use WAIT (action N) when nothing fits.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_SERVERS  # noqa: E402

WAIT = N_SERVERS


class RoundRobin:
    """WHY a pointer: round robin spreads load evenly regardless of request
    size; the pointer starts each scan at the last server used."""

    def __init__(self):
        self.ptr = 0

    def __call__(self, obs, env):
        for i in range(N_SERVERS):
            k = (self.ptr + i) % N_SERVERS
            if env.can_fit(k):
                self.ptr = (k + 1) % N_SERVERS
                return k
        return WAIT


class FirstFit:
    """WHY lowest index: deterministic, cache-friendly simple baseline."""

    def __call__(self, obs, env):
        for k in range(N_SERVERS):
            if env.can_fit(k):
                return k
        return WAIT


class BestFit:
    """WHY smallest leftover: packs big jobs with small jobs to reduce
    fragmentation; the classic bin-packing heuristic."""

    def __call__(self, obs, env):
        cpu, mem = env.head_job()
        best_k, best_leftover = WAIT, float("inf")
        for k in range(N_SERVERS):
            if env.can_fit(k):
                s = env.cluster.servers[k]
                leftover = (s.free_cpu - cpu) + (s.free_mem - mem)
                if leftover < best_leftover:
                    best_leftover, best_k = leftover, k
        return best_k


# Factories (not singletons): RoundRobin carries state, so a fresh instance
# per episode keeps heuristic runs independent.
HEURISTIC_FACTORIES = {
    "round_robin": lambda: RoundRobin(),
    "first_fit": lambda: FirstFit(),
    "best_fit": lambda: BestFit(),
}
