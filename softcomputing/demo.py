"""demo.py -- live, step-by-step simulation of the fuzzy schedulers.

This is the terminal counterpart of the dashboard's "Fuzzy decision inspector":
it runs one real seeded episode and prints, for selected decisions,

  * the head job, queue pressure and candidate servers,
  * the crisp leftovers/loads fed to the fuzzy system,
  * the membership degree of every linguistic set,
  * every rule that fired (strength, consequent, contribution),
  * the weighted-average score and the chosen server.

It is a genuine simulation: the policy drives ``CloudSchedulingEnv`` exactly as
in the evaluation harness.  Output is sampled so a full episode is readable.

    python -m softcomputing.demo
    python -m softcomputing.demo --scenario bursty --every 40 --max-print 12
    python -m softcomputing.demo --policy ga_fis --scenario heavy --every 1 --max-print 5
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_SEED, N_SERVERS  # noqa: E402

from env.scheduling_env import CloudSchedulingEnv  # noqa: E402
from softcomputing.fis_policy import WAIT, FuzzyScheduler  # noqa: E402
from softcomputing.ga import decode, select_best_ga  # noqa: E402


def _fmt_memberships(mu: dict) -> str:
    parts = []
    for name, sets in mu.items():
        parts.append(name + ": " + " ".join(
            f"{s}={v:.2f}" for s, v in sets.items()))
    return " | ".join(parts)


def _fmt_candidates(cands: list) -> str:
    return "  ".join(
        f"s{c['server']}={c['score']:.2f}"
        + ("*" if c.get("chosen") else "")
        for c in cands)


def print_decision(step: int, rec: dict) -> None:
    job = rec["job"]
    t = rec["time"]
    if rec["action"] == WAIT:
        print(f"[{step:4d}] t={t:9.1f}s queue={rec['queue_len']:3d} "
              f"job cpu={job[0]:.3f} mem={job[1]:.3f} -> WAIT "
              f"({rec['reason']})")
        return

    cands = [dict(c, chosen=(c["server"] == rec["server"]))
             for c in rec["candidates"]]
    print(f"[{step:4d}] t={t:9.1f}s queue={rec['queue_len']:3d} "
          f"job cpu={job[0]:.3f} mem={job[1]:.3f} -> SERVER {rec['server']} "
          f"(score {rec['score']:.3f})")
    print(f"       crisp: leftover={rec['crisp']['leftover']:.3f} "
          f"load={rec['crisp']['load']:.3f} "
          f"pressure={rec['crisp']['pressure']:.3f}")
    print(f"       {_fmt_memberships(rec['memberships'])}")
    print("       fired: " + " | ".join(
        f"{r['leftover']}/{r['load']}/{r['pressure']} z={r['z']:.2f} "
        f"w={r['strength']:.2f} -> {r['contribution']:.3f}"
        for r in rec["rules"]))
    print(f"       score = {rec['num']:.3f} / {rec['den']:.3f} = "
          f"{rec['score']:.3f}")
    print(f"       candidates: {_fmt_candidates(cands)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="heavy",
                    choices=["light", "heavy", "bursty"])
    ap.add_argument("--policy", default="fis", choices=["fis", "ga_fis"])
    ap.add_argument("--every", type=int, default=50,
                    help="print one decision every N steps")
    ap.add_argument("--max-print", type=int, default=10,
                    help="stop printing after this many decisions")
    args = ap.parse_args()

    params = None
    if args.policy == "ga_fis":
        selected = select_best_ga()
        if selected is None:
            raise SystemExit("no GA chromosome found: run "
                             "`python -m softcomputing.train_ga` first")
        params = decode(selected["chromosome"])
        label = f"GA-FIS (seed {selected['seed']}, training fitness " \
                f"{selected['fitness']:.3f})"
    else:
        label = "FIS (hand-designed, addendum defaults)"

    print(f"Soft-computing simulation | {label}")
    print(f"scenario={args.scenario}  seed={BASE_SEED}  "
          f"servers={N_SERVERS}  actions=0..{N_SERVERS - 1} place, "
          f"{N_SERVERS}=WAIT")
    print("score = sum(strength * z) / sum(strength); "
          "candidate '*' = chosen\n")

    env = CloudSchedulingEnv(mode="eval", scenario=args.scenario)
    policy = FuzzyScheduler(params)
    obs, _ = env.reset(seed=BASE_SEED)

    step, printed, info = 0, 0, {}
    done = False
    while not done:
        rec = policy.explain(obs, env)
        if step % max(1, args.every) == 0 and printed < args.max_print:
            print_decision(step, rec)
            printed += 1
        obs, _, term, trunc, info = env.step(int(rec["action"]))
        done = term or trunc
        step += 1

    print(f"\nepisode finished: {step} decisions, "
          f"{info.get('invalid_actions', 0)} invalid, "
          f"completion {info.get('completion_rate', 0):.3f}, "
          f"avg_wait {info.get('avg_wait', 0):.1f}s, "
          f"p95 {info.get('p95_wait', 0):.1f}s, "
          f"util {info.get('avg_cpu_util', 0):.3f}, "
          f"makespan {info.get('makespan', 0):.1f}s")
    print("(the same numbers appear in results/summary_table.csv)")


if __name__ == "__main__":
    main()
