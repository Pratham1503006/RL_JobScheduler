"""fis_policy.py -- the Fuzzy Inference Scheduler as ``policy(obs, env)``.

For every server that can fit the head-of-queue job, score its suitability
from the three fuzzy inputs and pick the highest score; WAIT (action N) only
when no server fits.  Because only ``env.can_fit(k)`` servers are candidates,
the policy can never make an invalid choice -- ``invalid_actions`` stays 0.

Ties are broken by the lowest server index (``np.argmax`` returns the first
maximum and ``fitting`` is built in ascending index order).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FUZZY_INPUTS, FUZZY_RULES, N_SERVERS  # noqa: E402

from softcomputing.fuzzy import (default_params, infer_batch,  # noqa: E402
                                 set_memberships)

WAIT = N_SERVERS


class FuzzyScheduler:
    """SC1: interpretable, capacity-aware fuzzy scheduler (no learning)."""

    def __init__(self, params=None):
        self.params = default_params() if params is None else params

    # ------------------------------------------------------------ helpers ----
    def candidate_features(self, obs, env):
        """Crisp inputs for every server that can fit the head job.

        WHY one helper for both ``__call__`` and ``explain``: the recorded
        explanation must be produced by the SAME arithmetic that makes the
        decision, otherwise the demo could drift from the policy.
        """
        # pressure is the same for every server and already lives in the obs.
        pressure = float(obs[2 * N_SERVERS + 2])
        req_cpu, req_mem = env.head_job()

        fitting, leftovers, loads = [], [], []
        for k in range(N_SERVERS):
            if not env.can_fit(k):
                continue
            s = env.cluster.servers[k]
            fitting.append(k)
            leftovers.append(
                ((s.free_cpu - req_cpu) + (s.free_mem - req_mem)) / 2.0)
            loads.append(
                ((1.0 - s.free_cpu) + (1.0 - s.free_mem)) / 2.0)
        return (pressure, fitting,
                np.asarray(leftovers, dtype=float),
                np.asarray(loads, dtype=float))

    def __call__(self, obs, env):
        pressure, fitting, leftovers, loads = self.candidate_features(obs, env)
        if not fitting:
            return WAIT

        scores = infer_batch(np.maximum(leftovers, 0.0), loads, pressure,
                             self.params)
        return int(fitting[int(np.argmax(scores))])

    # -------------------------------------------------------- explanation ----
    def explain(self, obs, env) -> dict:
        """Full audit trail for the current decision (for demos/teaching).

        Returns the three crisp inputs per candidate, the fuzzy memberships,
        every rule that fired (strength, consequent), the weighted average and
        the chosen server.  It changes no state and is never used by
        ``__call__``, so recording cannot affect the measured policy.
        """
        pressure, fitting, leftovers, loads = self.candidate_features(obs, env)
        req_cpu, req_mem = env.head_job()
        cuts = {
            name: [float(self.params.cuts[2 * i]),
                   float(self.params.cuts[2 * i + 1])]
            for i, name in enumerate(FUZZY_INPUTS)
        }
        rec = {
            "time": round(float(env.now), 3),
            "queue_len": int(len(env.queue)),
            "jobs_done": int(env._jobs_done),
            "job": [round(float(req_cpu), 4), round(float(req_mem), 4)],
            "pressure": round(pressure, 4),
            "cuts": cuts,
        }
        if not fitting:
            rec.update({"action": WAIT, "server": None, "reason":
                        "no server can fit the head job -> WAIT",
                        "candidates": []})
            return rec

        left = np.maximum(leftovers, 0.0)
        scores = infer_batch(left, loads, pressure, self.params)
        best = int(np.argmax(scores))
        action = int(fitting[best])

        # Memberships of the CHOSEN server: what the audience needs to see.
        crisp = {"leftover": float(left[best]),
                 "load": float(loads[best]),
                 "pressure": pressure}
        memberships = {
            name: {s: round(float(v), 4) for s, v in
                   set_memberships(name, crisp[name], self.params).items()}
            for name in FUZZY_INPUTS
        }

        # Every rule with a non-zero firing strength, strongest first.
        # WHY params.z[i]: the GA tunes the consequents, so the config's
        # default z is only correct for the hand-designed FIS.
        rules = []
        for i, (lo_set, ld_set, pr_set, _z) in enumerate(FUZZY_RULES):
            z = float(self.params.z[i])
            strength = min(memberships["leftover"][lo_set],
                           memberships["load"][ld_set],
                           memberships["pressure"][pr_set])
            if strength > 1e-9:
                rules.append({
                    "i": i,
                    "leftover": lo_set, "load": ld_set, "pressure": pr_set,
                    "z": float(z),
                    "strength": round(float(strength), 4),
                    "contribution": round(float(strength * z), 4),
                })
        rules.sort(key=lambda r: r["strength"], reverse=True)
        den = float(sum(r["strength"] for r in rules))
        num = float(sum(r["contribution"] for r in rules))

        rec.update({
            "action": action,
            "server": action,
            "chosen_index": best,
            "crisp": {k: round(v, 4) for k, v in crisp.items()},
            "candidates": [
                {"server": int(k),
                 "leftover": round(float(lo), 4),
                 "load": round(float(ld), 4),
                 "score": round(float(sc), 4)}
                for k, lo, ld, sc in zip(fitting, left, loads, scores)
            ],
            "memberships": memberships,
            "rules": rules,
            "num": round(num, 4),
            "den": round(den, 4),
            "score": round(float(scores[best]), 4),
        })
        return rec
