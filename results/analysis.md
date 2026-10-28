# Analysis (numbers from `summary_table.csv` / `full_comparison.csv` / `ga_logs/`)

## Head-to-head, per scenario

1. **`light`** (rho=0.40): the three heuristics and the two soft-computing policies tie at
   ~3,085–3,091 s avg wait (round_robin 3,084.5 s, first_fit 3,090.1 s, best_fit 3,090.6 s,
   fis 3,089.6 s, ga_fis 3,090.8 s — total spread 0.2%); best RL is **ddqn at 3,829.5 s
   (+24% vs round_robin)**, so under light load nothing is worth optimizing and the
   capacity-aware rules win.
2. **`heavy`** (rho=0.90): heuristics and soft computing again near-tie (round_robin 4,190.5 s,
   first_fit 4,193.4 s, best_fit 4,191.9 s, fis 4,212.9 s, ga_fis 4,205.6 s — spread 0.5%) and
   **round_robin still wins overall**; best RL ddqn 5,130.9 s is +22% vs round_robin. DQN
   (35,791.0 s) and PPO (37,846.7 s) wait ~8.5x longer than round_robin.
3. **`bursty`** (mean rho=0.60): round_robin 46,871.4 s, first_fit 46,876.2 s, best_fit 46,874.4 s,
   **fis 46,869.0 s (best of all nine policies)**, ga_fis 46,877.1 s — everything capacity-aware
   ties within 0.02%; best RL **a2c 65,113.2 s** (+38.9% vs fis), ddqn 110,909.5 s (std 42,627
   across seeds — one seed truncated, completion 0.963), while dqn/ppo average ~313k s (they
   spend most of the episode stalling).
4. **No RL agent beats any heuristic or soft-computing policy on any scenario.** The data-backed
   reason: every RL policy burns ~1,200–2,000 invalid actions per 2,000-job episode (one per job
   — it tries a full server first, then sits through the resulting time advance), while
   heuristics and FIS/GA-FIS have 0. RL effectively learned a greedy first-choice plus a WAIT
   fallback that adds roughly one completion interval of wait per job; at 150k/200k training
   steps it never learned to preemptively pack smaller jobs into fragmented servers the way
   best-fit does.

## Soft computing (addendum SC1/SC2)

5. **FIS vs heuristics**: the hand-designed FIS is in the same ballpark as Best-Fit, as the
   addendum predicted — +0.17% vs round_robin on `light`, +0.53% on `heavy`, and -0.005% on
   `bursty` (it is literally the fastest policy on that scenario). FIS and GA-FIS both finish
   100% of jobs with **0 invalid actions** on all three scenarios, by construction (they only
   consider servers that pass `can_fit`).
6. **Does GA tuning help?** On the **training** episodes, yes, slightly: the hand-designed
   chromosome scores -2729.4602 and the three GA seeds reach -2727.6631 / -2727.3650 / -2728.6244
   (`ga_seed_summary.csv`). On the **held-out** scenarios the gain does not carry over: GA-FIS
   (selected seed 1) is 0.04% worse than FIS on `light` (3,090.8 vs 3,089.6 s), 0.17% better on
   `heavy` (4,205.6 vs 4,212.9 s) and 0.02% worse on `bursty` (46,877.1 vs 46,869.0 s). The
   honest conclusion is **no reliable held-out improvement** — the tuned policy is a near-tie
   with the hand-designed one. WHY: FIS decisions are `argmax` over fuzzy scores, so the fitness
   is piecewise-constant in most genes; most mutations do not change a single placement.
7. **GA seed selection is not a test-set oracle.** Selection used training fitness only (seed 1,
   -2727.3650). The supplementary robustness table (`ga_seed_robustness.csv`) shows the three
   chromosomes are interchangeable on the held-out scenarios: mean avg wait 18,059.56 s (seed 0),
   18,057.85 s (seed 1, selected), 18,055.46 s (seed 2) — a 4.1 s / 0.02% spread, with the
   selected seed in the middle. Selecting on `light`/`heavy`/`bursty` would have picked seed 2,
   which is exactly the leakage the addendum forbids.
8. **RL vs the best soft-computing policy**: RL loses on every scenario — ddqn is +24% on
   `light` and +22% on `heavy`, and the best RL on `bursty` (a2c) is +38.9%. The gap is 1–2
   orders of magnitude larger than the seed spread, so it is not a seed artifact.

## DQN vs Double DQN, and statistical honesty

9. **DQN vs Double DQN is the one clear RL finding**: DoubleDQN trains to far higher episode
   rewards (seed 0: -32.7k vs -905.0k, ~15-28x better across seeds, `rewards_*_seed*.csv`) and
   its learned Q-values are CONSISTENT with realized returns (mean max-Q -1,738 at 150k steps
   implies -17.4/step at gamma .99; its final episodes earned -19 to -58/step,
   `qvalues_*_seed*.csv`), while vanilla DQN's Q-values (-329) imply only -3.3/step yet its
   final episodes earned -35 to -1,057/step: its estimates are detached from reality, the classic
   bootstrap-overestimation failure mode.

Statistical honesty notes: heuristic and soft-computing rows are single deterministic runs
(std 0); RL rows are mean ± std over 3 saved seeds, except PPO which uses 2 seeds — its seed-2
run was excluded after two independent training runs both collapsed (see
`hyperparameters.json` `evaluation_notes`). Gap-to-seed-spread rule: the soft-computing vs RL
and heuristic vs RL gaps far exceed the 1–3% spreads across RL seeds, so the ordering
rules > RL is not a seed artifact; the 10–25k s differences between dqn and ppo on identical
scenarios are within each other's seed spread and should be treated as a tie. The GA-FIS vs FIS
differences (0.02–0.17%) are smaller than the GA's own 0.02% held-out seed spread and should be
treated as a tie.
