"""fuzzy.py -- membership functions and zero-order Sugeno inference (SC1).

The whole fuzzy system is deliberately small and dependency-free (NumPy only):

  inputs   leftover, load, pressure   (all in [0, 1])
  sets     leftover: TIGHT / MEDIUM / LOOSE
           load    : LOW / MEDIUM / HIGH
           pressure: CALM / BUSY
  rules    18 = 3 x 3 x 2, one per input combination (table in config.py)
  output   crisp suitability in [0, 1] = weighted average of the rule
           consequents z_r, weighted by the rule strength
           min(mu_leftover, mu_load, mu_pressure)

WHY hand-written memberships instead of a fuzzy library: the GA chromosome has
to map directly onto the breakpoints.  One ``trap`` function covers triangles
and both shoulder shapes, so the chromosome is just six cut points plus the
eighteen consequents.

Chromosome -> memberships mapping (two cut points p1 < p2 per input):
  three-set input : left shoulder (0,0,p1,mid), triangle (p1,mid,mid,p2),
                    right shoulder (mid,p2,1,1), with mid=(p1+p2)/2
  two-set input   : left shoulder (0,0,p1,p2), right shoulder (p1,p2,1,1)

That mapping is full-coverage and ordered for ANY p1 < p2 in (0, 1), and it
reproduces the hand-designed defaults exactly (see
``config.FUZZY_CUTS_DEFAULT``).
"""

from dataclasses import dataclass

import numpy as np

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (FUZZY_CUTS_DEFAULT, FUZZY_INPUTS, FUZZY_RULES,  # noqa: E402
                    FUZZY_SETS)


@dataclass
class FISParams:
    """One concrete fuzzy system: 6 cut points + 18 rule consequents."""

    cuts: np.ndarray   # shape (6,): leftover p1,p2 | load p1,p2 | pressure p1,p2
    z: np.ndarray      # shape (18,): consequent of each rule in FUZZY_RULES


def trap(x, a: float, b: float, c: float, d: float):
    """Trapezoid membership: 0 below ``a``, 1 on ``[b, c]``, 0 above ``d``.

    Triangles are ``trap(a, b, b, d)``.  A zero-width rising edge (``b == a``)
    is a LEFT shoulder (1 from ``a`` upwards); a zero-width falling edge
    (``d == c``) is a RIGHT shoulder (1 up to ``c``).  Inputs outside the
    support get exactly 0, which is what the coverage test asserts.
    """
    x = np.asarray(x, dtype=float)
    if b > a:
        left = np.clip((x - a) / (b - a), 0.0, 1.0)
    else:
        left = np.where(x >= a, 1.0, 0.0)
    if d > c:
        right = np.clip((d - x) / (d - c), 0.0, 1.0)
    else:
        right = np.where(x <= c, 1.0, 0.0)
    return np.minimum(left, right)


def _sets_for_input(name: str, p1: float, p2: float) -> dict:
    """Map the two cut points of one input to ordered, full-coverage sets."""
    mid = 0.5 * (p1 + p2)
    if name == "pressure":
        calm, busy = FUZZY_SETS["pressure"]
        return {calm: (0.0, 0.0, p1, p2), busy: (p1, p2, 1.0, 1.0)}
    low, middle, high = FUZZY_SETS[name]
    return {
        low: (0.0, 0.0, p1, mid),
        middle: (p1, mid, mid, p2),
        high: (mid, p2, 1.0, 1.0),
    }


def membership_params(params: FISParams) -> dict:
    """``{input: {set: (a, b, c, d)}}`` -- the trapezoid corners to plot/audit."""
    out = {}
    for i, name in enumerate(FUZZY_INPUTS):
        p1, p2 = float(params.cuts[2 * i]), float(params.cuts[2 * i + 1])
        out[name] = _sets_for_input(name, p1, p2)
    return out


def set_memberships(input_name: str, x, params: FISParams) -> dict:
    """Membership of every linguistic set of ``input_name`` at ``x``."""
    sets = membership_params(params)[input_name]
    return {name: trap(x, *corners) for name, corners in sets.items()}


def default_params() -> FISParams:
    """The hand-designed FIS from addendum section 3.3/3.4."""
    cuts = np.concatenate([FUZZY_CUTS_DEFAULT[name] for name in FUZZY_INPUTS])
    z = np.array([rule[3] for rule in FUZZY_RULES], dtype=float)
    return FISParams(cuts=cuts.astype(float), z=z)


def infer_batch(leftovers, loads, pressure, params: FISParams) -> np.ndarray:
    """Score many candidate servers at once (the hot path used by the policy).

    ``leftovers`` / ``loads`` are 1-D arrays (one entry per candidate server);
    ``pressure`` is a scalar or an array broadcastable to them.  Returns one
    crisp suitability in [0, 1] per candidate; 0 when no rule fires.
    """
    lo = np.atleast_1d(np.asarray(leftovers, dtype=float))
    ld = np.atleast_1d(np.asarray(loads, dtype=float))
    pr = np.atleast_1d(np.asarray(pressure, dtype=float))

    mu = {
        "leftover": set_memberships("leftover", lo, params),
        "load": set_memberships("load", ld, params),
        "pressure": set_memberships("pressure", pr, params),
    }
    num = np.zeros_like(lo, dtype=float)
    den = np.zeros_like(lo, dtype=float)
    for i, (leftover_set, load_set, pressure_set, _z) in enumerate(FUZZY_RULES):
        strength = np.minimum(
            np.minimum(mu["leftover"][leftover_set], mu["load"][load_set]),
            mu["pressure"][pressure_set],
        )
        num += strength * params.z[i]
        den += strength

    out = np.zeros_like(num)
    active = den > 0.0
    out[active] = num[active] / den[active]
    return out


def infer(leftover: float, load: float, pressure: float,
          params: FISParams) -> float:
    """Crisp suitability of one server (scalar wrapper around infer_batch)."""
    return float(infer_batch(leftover, load, pressure, params)[0])
