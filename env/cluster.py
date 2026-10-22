"""cluster.py -- Server and Cluster state classes for the simulator.

WHY a tiny explicit class instead of raw arrays everywhere: the acceptance
tests must be able to verify that place+release restores capacity EXACTLY
(float tolerance 1e-9), and a Server makes that state auditable in one place.
"""

import heapq
import itertools

import numpy as np

from config import CPU_CAP, MEM_CAP, N_SERVERS

# WHY an epsilon: repeated place/release can drift used capacity by ~1 ulp,
# and event times differing by <1ns are effectively simultaneous. EPS keeps
# both behaviors stable without ever making a real overflow pass.
EPS = 1e-9


class Server:
    """One identical server with normalized capacities (cpu=1.0, mem=1.0)."""

    def __init__(self, cpu_cap: float = CPU_CAP, mem_cap: float = MEM_CAP):
        self.cpu_cap = float(cpu_cap)
        self.mem_cap = float(mem_cap)
        self.used_cpu = 0.0
        self.used_mem = 0.0

    def can_fit(self, cpu: float, mem: float) -> bool:
        # EPS absorbs the ~1 ulp drift from repeated place/release cycles so
        # a fully-released server always fits a capacity-sized request again.
        return (self.used_cpu + cpu <= self.cpu_cap + EPS
                and self.used_mem + mem <= self.mem_cap + EPS)

    def place(self, cpu: float, mem: float) -> None:
        assert self.can_fit(cpu, mem), "place() called on a server that cannot fit"
        self.used_cpu += cpu
        self.used_mem += mem

    def release(self, cpu: float, mem: float) -> None:
        self.used_cpu -= cpu
        self.used_mem -= mem

    @property
    def free_cpu(self) -> float:
        return self.cpu_cap - self.used_cpu

    @property
    def free_mem(self) -> float:
        return self.mem_cap - self.used_mem


class Cluster:
    """N identical servers plus the running-jobs completion heap."""

    def __init__(self, n_servers: int = N_SERVERS):
        self.n = n_servers
        self.servers = [Server() for _ in range(n_servers)]
        self._running = []          # heap of (completion_time, seq, server_idx, cpu, mem)
        self._seq = itertools.count()

    # ---------------------------------------------------------- capacity ----
    def can_fit(self, k: int, cpu: float, mem: float) -> bool:
        return self.servers[k].can_fit(cpu, mem)

    def place(self, k: int, cpu: float, mem: float, completion_time: float) -> None:
        self.servers[k].place(cpu, mem)
        heapq.heappush(self._running, (completion_time, next(self._seq), k, cpu, mem))

    def release(self, k: int, cpu: float, mem: float) -> None:
        self.servers[k].release(cpu, mem)

    # ----------------------------------------------------------- running ----
    @property
    def next_completion(self) -> float:
        return self._running[0][0] if self._running else float("inf")

    def pop_completed(self, now: float) -> list:
        """Pop every job whose completion time is <= now; release capacity.

        WHY completions are processed before same-time arrivals: freeing
        capacity first means an arriving job can be placed immediately, which
        matches how a real scheduler behaves at an event instant.
        """
        done = []
        while self._running and self._running[0][0] <= now + EPS:
            _, _, k, cpu, mem = heapq.heappop(self._running)
            self.release(k, cpu, mem)
            done.append(k)
        return done

    @property
    def n_running(self) -> int:
        return len(self._running)

    # -------------------------------------------------------- observation ----
    def free_cpu_array(self) -> np.ndarray:
        return np.clip(np.array([s.free_cpu / s.cpu_cap for s in self.servers],
                                dtype=np.float32), 0.0, 1.0)

    def free_mem_array(self) -> np.ndarray:
        return np.clip(np.array([s.free_mem / s.mem_cap for s in self.servers],
                                dtype=np.float32), 0.0, 1.0)

    def mean_cpu_util(self) -> float:
        return float(np.mean([s.used_cpu / s.cpu_cap for s in self.servers]))
