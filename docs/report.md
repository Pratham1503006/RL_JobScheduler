# RL-Based Dynamic Cloud Job Scheduling — Project Report

Every number below is traceable to a file under `results/` or `data/` — nothing
is hand-reported outside those artifacts.

## 1. Problem statement and motivation

Cloud operators queue jobs and must choose, for the job at the head of the
queue, which server should run it now, or hold it. Bad choices waste capacity
(a 1-CPU job on an idle server while a 0.95-CPU job waits elsewhere) or force
long stalls (a 0.95-CPU job blocking the queue's head). Heuristic schedulers
(Round Robin, First-Fit, Best-Fit) are fixed rules: they never trade off
*waiting now* against *packing for later*. We test whether RL learns that
trade-off from workload experience and outperforms classical rules. The metric
is **average waiting time**, i.e., how long the user-visible queue takes.

## 2. Related work

Classical bin-packing schedulers (Round Robin, First-Fit, Best-Fit) are the
standard text-book baselines for multi-resource placement. RL for scheduling
has used DQN variants for discrete placement decisions and policy-gradient
methods for load balancing; Double DQN (van Hasselt et al., 2016) reduces the
max-operator's overestimation bias, which is exactly the bias a scheduling
environment (many similar actions, reward-heavy penalties) exacerbates.

## 3. Dataset

Google **ClusterData2011_2**, `task_events` table, first 10 of 500 shards
(~1.83M events). One trace *task* = one "job". Cleaning
(`data/clean_trace.py`, full audit in `data/cleaning_report.md`):

| # | rule | before | removed/changed |
|---|---|---|---|
| 1 | sentinel timestamps (0 / 2^63-1) | 144,339 | dropped |
| 2 | partial/reconstructed records | 0 | dropped |
| 3 | exact duplicate events | 0 | dedup'd |
| 4 | non-clean lifecycle (EVICT/FAIL/KILL/LOST or multi-SUBMIT) | 79,216 | dropped |
| 5 | no FINISH inside window / no SCHEDULE row | 33,503 + 0 | dropped |
| 6 | blank CPU or mem request | 0 | dropped |
| 7 | zero or negative request | 1,233 | dropped |
| 8 | request > 1.0 | 0 | clipped to 1.0 |
| 9 | non-positive duration | 0 | dropped |
| 10 | sub-second duration | 0 | clipped to 1s |
| 11 | extreme durations > p99.9 | 206 | winsorized at 22,879.9 s |
| 12 | same-timestamp bursts | 0 | kept as-is (real bursts) |

Result: **205,713 jobs** (`data/jobs_clean.csv`). Kept lifecycle favors
normally-finishing tasks — survivorship bias, accepted and stated.
First **70%** by arrival = training pool (143,999 jobs); the last 30%
(61,714 jobs) is held out, scenarios are built ONLY from it
(`calibration.json`), so **agents never train on test jobs**.

## 4. Environment design

`env/scheduling_env.py`: event-driven simulator, no fixed clock ticks.
- **State (23 floats in [0,1]):** free CPU + free memory of the 10 servers,
  head-job CPU/mem request, queue length / `QUEUE_NORM=50` clipped to 1.
  *Why [0,1]:* NN inputs normalized; matches the trace's normalized requests.
- **Action `Discrete(11)`:** assign head job to server k=0..9, or WAIT.
- **Reward** (weights in `config.py`): `+W_PLACE(1)` on placement,
  `-W_WAIT(1) * queue_len * dt / 60` per simulated second-pass,
  `-W_INVALID(20)` for an invalid pick, `+W_UTIL(0.1) * util * dt / 60`.
  *Why:* the queue-length x time term sums to total waiting time (Little's
  law), so maximizing return == minimizing average wait; `W_INVALID` was
  raised once (5 -> 20) after full training showed >20% invalid-action rates
  (`results/hyperparameters.json` -> `reward_weight_change_note`).
- **Truncation:** 10 x n_jobs agent steps guard against stalling.

## 5. Algorithms

All four: `stable-baselines3`, `MlpPolicy`, `net_arch=[128,128]`, CPU-only,
seeded (0/1/2). Full per-algorithm hyperparameters in
`results/hyperparameters.json` (`per_algorithm`). Headlines:
- **DQN:** replay buffer 50k, target update 500, eps 0.3->0.05,
  150k steps. *Plan-B tuning applied after a reward collapse:* lr
  1e-4 -> 3e-5 + grad clip 1.0 (`dqn_note`), still ~50% invalid picks in eval.
- **Double DQN:** same as DQN, exact same budget; subclass of SB3's DQN
  replacing only target-Q computation: pick next action with the ONLINE net,
  evaluate with the TARGET net.
- **PPO:** lr 3e-4, clip 0.2, ent 0.01, 200k steps. **A2C:** lr 7e-4,
  n_steps 16, 200k steps.

### 5.1 SC1 — Fuzzy Inference Scheduler (FIS)

A hand-designed, interpretable, non-learning scheduler (`softcomputing/fuzzy.py`,
`softcomputing/fis_policy.py`). For every server that can fit the head job it
computes a suitability score in [0, 1] and picks the highest-scoring server;
it returns WAIT only when no server fits, so it is capacity-aware exactly like
the heuristics (`invalid_actions` is always 0).

**Inputs (per candidate server, all in [0, 1])**

| input | definition | linguistic sets |
|---|---|---|
| `leftover` | `((free_cpu - req_cpu) + (free_mem - req_mem)) / 2` | TIGHT, MEDIUM, LOOSE |
| `load` | `((1 - free_cpu) + (1 - free_mem)) / 2` | LOW, MEDIUM, HIGH |
| `pressure` | `min(queue_len / QUEUE_NORM, 1)` | CALM, BUSY |

**Membership functions** (trapezoids/shoulders, `trap(x,a,b,c,d)`; triangles are
`trap(a,b,b,d)`). Default breakpoints, from `config.FUZZY_CUTS_DEFAULT`:

| input | set | trapezoid |
|---|---|---|
| leftover | TIGHT | (0.0, 0.0, 0.1, 0.35) |
| leftover | MEDIUM | (0.1, 0.35, 0.35, 0.6) |
| leftover | LOOSE | (0.35, 0.6, 1.0, 1.0) |
| load | LOW | (0.0, 0.0, 0.2, 0.5) |
| load | MEDIUM | (0.2, 0.5, 0.5, 0.8) |
| load | HIGH | (0.5, 0.8, 1.0, 1.0) |
| pressure | CALM | (0.0, 0.0, 0.2, 0.6) |
| pressure | BUSY | (0.2, 0.6, 1.0, 1.0) |

The sets are full-coverage: at every point in [0,1] at least one set is active
(`tests/test_fuzzy.py`). Functions: `results/softcomputing/fis_default_memberships.png`.

**Rule base (18 = 3 x 3 x 2, zero-order Sugeno).** Rule strength is
`min(mu_leftover, mu_load, mu_pressure)`; the final score is
`sum(w_r z_r) / sum(w_r)` (0 if no rule fires). The design logic is the
pressure switch: under **BUSY** prefer packing (keep big free blocks for large
waiting jobs, so TIGHT+HIGH scores 1.00 and LOOSE+LOW only 0.25); under
**CALM** prefer balancing (spread jobs and leave headroom, so LOOSE+LOW scores
1.00 and TIGHT+HIGH only 0.20); MEDIUM leftover + HIGH load is the
fragmentation trap and gets the lowest BUSY score (0.15).

| leftover | load | z (CALM) | z (BUSY) |
|---|---|---|---|
| TIGHT | LOW | 0.50 | 0.75 |
| TIGHT | MEDIUM | 0.35 | 0.90 |
| TIGHT | HIGH | 0.20 | 1.00 |
| MEDIUM | LOW | 0.70 | 0.50 |
| MEDIUM | MEDIUM | 0.65 | 0.40 |
| MEDIUM | HIGH | 0.50 | 0.15 |
| LOOSE | LOW | 1.00 | 0.25 |
| LOOSE | MEDIUM | 0.80 | 0.30 |
| LOOSE | HIGH | 0.60 | 0.35 |

### 5.2 SC2 — GA-Tuned Fuzzy Scheduler (GA-FIS)

The same FIS structure and the same 18 rules; a real-valued genetic algorithm
(`softcomputing/ga.py`, `softcomputing/train_ga.py`) evolves the membership
breakpoints and the rule consequents.

- **Chromosome (24 genes):** two cut points `(p1, p2)`, `0 < p1 < p2 < 1`, for
  each of the three inputs, plus the 18 consequents `z_r` in [0,1]. The cut
  points are mapped to ordered, full-coverage sets: for a three-set input,
  `(0,0,p1,mid)`, `(p1,mid,mid,p2)`, `(mid,p2,1,1)` with `mid=(p1+p2)/2`; for
  the two-set `pressure`, `(0,0,p1,p2)` and `(p1,p2,1,1)`. `repair()` sorts
  and clips every chromosome (minimum gap 0.02, consequents in [0,1]), so no
  chromosome can produce an ill-formed fuzzy system.
- **Initialisation:** individual 0 is the hand-designed FIS chromosome; the
  other 29 are that chromosome plus Gaussian noise (sigma 0.10). GA-FIS can
  therefore never start worse than FIS.
- **GA settings (`config.py`):** population 30, generations 30, tournament
  selection of size 3, BLX-alpha crossover (alpha 0.3, probability 0.8),
  Gaussian mutation (sigma 0.05 per gene, probability 0.1), elitism 2,
  `multiprocessing.Pool` population evaluation.
- **Fitness (maximised):** `-(mean avg_wait) - 1e5 * mean(invalid_actions)`
  on five fixed 500-job windows from `data/scenarios/train_pool.csv`,
  rescaled to rho targets evenly spaced over [0.6, 1.0]. The same episodes are
  used for every chromosome and every GA seed, so fitness differences come
  from the chromosome, not episode luck.
- **Leakage rule:** `_assert_train_only()` refuses any path other than
  `train_pool.csv`; the light/heavy/bursty scenarios are never shown to the
  GA. The final chromosome is selected by *training* fitness only; all three
  seeds are reported separately as a robustness check.


## 6. Experimental setup

- **Request scaling:** raw requests are ~2-10% of a machine; `REQ_SCALE=10`
  makes scaled p95 CPU = 1.0 (`calibration.json`) — at p95=0.5 every policy
  packed trivially and nothing separated (documented in config.py).
- **Load calibration:** arrivals stretched by factor s per scenario so
  rho = sum(cpu*duration)/(T*N) hits targets exactly: light 0.40, heavy 0.90,
  bursty 0.60 mean from the highest-CV arrival window
  (window CV 3.07 vs pool CV 3.33) — all in `data/scenarios/calibration.json`.
- **Scenarios:** 2,000 jobs each, all from the held-out 30%.
- **Evaluation:** heuristics are deterministic (1 episode, invalid_actions=0,
  completion 1.0); RL acts `deterministic=True`, one episode per trained seed
  (DQN/DDQN/A2C x3 seeds, PPO x2 — seed 2 excluded, see results/analysis.md).
- **Soft-computing evaluation:** FIS and GA-FIS are deterministic (one episode
  per scenario, invalid_actions=0), run through the same `agents.evaluate`
  loop. GA-FIS uses the chromosome with the best *training* fitness across the
  three GA seeds; all three seed chromosomes are also evaluated on the
  held-out scenarios in a supplementary robustness table
  (`results/ga_logs/ga_seed_robustness.csv`), never for selection.
- **GA training:** population 30 x 30 generations x 5 fixed episodes x 3 seeds,
  CPU-only with a multiprocessing pool. Episodes are 500-job contiguous
  windows from `data/scenarios/train_pool.csv` rescaled to rho targets evenly
  spaced over [0.6, 1.0]; the same episodes are shared by every chromosome and
  every GA seed.

## 7. Results

Full table `results/summary_table.csv` (mean ± std across seeds),
plots `results/charts/1-6*.png`. **Average wait (s):**

| policy | light | heavy | bursty | util heavy | invalid/ep (heavy) |
|---|---|---|---|---|---|
| Round Robin | 3,084 | 4,190 | 46,871 | 0.882 | 0 |
| First-Fit | 3,090 | 4,193 | 46,876 | 0.881 | 0 |
| Best-Fit | 3,091 | 4,192 | 46,874 | 0.881 | 0 |
| **FIS** (soft) | 3,090 | 4,213 | **46,869** | 0.875 | 0 |
| **GA-FIS** (soft) | 3,091 | 4,206 | 46,877 | 0.877 | 0 |
| DQN | 34,236 | 35,791 | 317,668 | 0.181 | 1,993 |
| Double DQN | 3,829 | 5,131 | 110,909 | 0.752 | 1,404 |
| PPO | 36,281 | 37,847 | 312,593 | 0.166 | 1,990 |
| A2C | 4,931 | 6,247 | 65,113 | 0.684 | 184 |

Round Robin wins `light` and `heavy` by a hair (0.2% / 0.5% over the
capacity-aware field); the FIS is the single fastest policy on `bursty`
(46,869 s vs 46,871 s for Round Robin). The two soft-computing schedulers are
within 0.5% of the best heuristic on every scenario and both complete 100% of
jobs with 0 invalid actions. GA tuning improved the *training* fitness
(-2729.4602 -> -2727.3650) but the held-out difference vs the hand-designed FIS
is ±0.17% per scenario: a tie (see 8.2). Statistical honesty: RL seed spreads
are 1-3% on average but explode on bursty (DDQN std 42,627 on 110,909 mean, one
truncated seed with completion 0.963), while the heuristic/soft-vs-RL gaps
(~22% to >7x) dwarf seed noise; DQN vs PPO differences on identical scenarios
are within each other's spread — treat as tie.

## 8. Discussion

**Who won where and why.** All three heuristics and both soft-computing
policies tie within ~0.5%; RL never beats any of them. Data-backed reason:
every RL agent makes ~1,400-2,000 invalid picks per
2,000-job episode, and each invalid pick costs one completion-interval of
advanced time; the measured gap to Round Robin is +745 s (light, DDQN) and
+940 s (heavy, DDQN), while the far larger DQN/PPO gaps (31-271k s) reflect
policies that mostly stall (util 0.07-0.18, wait 7-12x heuristics on average). Deeper: `baseline_sanity_check.md`
shows packing in this environment is essentially trivial (below-50% items
never create unavoidable fragmentation), so the only remaining learning
signal is queue discipline — where holding the head job costs more relative
  to the tiny placement gains.

**Does GA tuning help the fuzzy scheduler?** On the training episodes the
three GA runs all beat the hand-designed chromosome: default -2729.4602 vs
-2727.6631 / -2727.3650 / -2728.6244 (`results/ga_logs/ga_seed_summary.csv`).
On the held-out scenarios the improvement does not carry over. GA-FIS
(selected by training fitness, seed 1) is +0.04% on `light` (3,090.8 vs
3,089.6 s), -0.17% on `heavy` (4,205.6 vs 4,212.9 s) and +0.02% on `bursty`
(46,877.1 vs 46,869.0 s) relative to FIS. The honest answer is **no reliable
held-out gain**. WHY: the FIS action is an `argmax` over fuzzy scores, so the
fitness is piecewise-constant in most of the 24 genes — many mutations change
no placement at all, and the 30x30 search mostly finds tiny decision-boundary
shifts. The tuned membership functions *are* visibly different (chart 6), so
the GA is working; the objective is just nearly flat.

**GA seed selection is not a test-set oracle.** Because selection uses
training fitness only (addendum 4.6), the robustness table is the honest check:
the three chromosomes are interchangeable on the held-out scenarios (mean avg
wait 18,059.56 / 18,057.85 / 18,055.46 s for seeds 0/1/2, a 0.02% spread), and
the selected seed 1 is in the middle. Picking the best test-scenario seed
(seed 2) would have been leakage; the fact that it would have looked 2.4 s
better is exactly why the rule exists.

**Soft computing vs heuristics.** FIS lands in the same ballpark as Best-Fit,
as the addendum expected: +0.17% vs Round Robin on `light`, +0.53% on `heavy`,
and -0.005% on `bursty` (fastest of all nine policies there). The three
capacity-aware families are therefore statistically indistinguishable on these
workloads, and the 0.2-0.5% differences are far below any seed noise. GA-FIS
is also within 0.36% of Round Robin everywhere. Both soft policies have
`completion_rate=1.0` and `invalid_actions=0` on every scenario by
construction (`softcomputing/fis_policy.py` only considers `can_fit` servers).

**RL vs the best soft-computing policy.** RL loses on every scenario: Double
DQN is +24% on `light` (3,829 vs 3,090 s) and +22% on `heavy` (5,131 vs
4,213 s), and the best RL agent on `bursty` (A2C) is +38.9% (65,113 vs
46,869 s). The gaps are 1-2 orders of magnitude larger than the RL seed
spreads, so this is not a seed artifact: the fuzzy policies get the capacity
discipline for free, while every RL agent spends ~1,400-2,000 actions per
2,000-job episode on invalid placements (section 8, first paragraph).

**DQN vs Double DQN (the controlled experiment).** Same hyperparameters, same
budget, different target computation only:
- DDQN final-episode rewards: -32.7k/-107.9k/-57.8k (seeds 0-2); DQN:
  -905.0k/-68.8k/-1,979.7k
  (`results/train_logs/rewards_{dqn,ddqn}_seed*.csv`) ~ 15-28x worse.
- At 150k steps mean max-Q: DDQN **-1,738**, DQN **-329**
  (`results/train_logs/qvalues_{dqn,ddqn}_seed*.csv`). With gamma=0.99, a
  Q value implies a steady per-step reward of Q*(1-gamma): DDQN implies
  **-17.4/step** vs realized -19 to -58/step (consistent); DQN implies
  **-3.3/step** vs realized -35 to -1,057/step — its value estimates are
  10-320x detached from what its own policy earns: the value/return
  decoupling signature of max-operator overestimation. Both curves:
  `results/charts/4_qvalue_overestimation.png`.
- Consequence: DDQN is the best RL agent (3,829/5,131 s on light/heavy vs
  34,236/35,791 for DQN), and the only RL agent that approaches the
  heuristics.

## 9. Limitations

1. 10 identical servers (a real cluster is heterogeneous; the 12k-machine
   Google cluster is compressed to 10 nodes by design).
2. `REQ_SCALE=10` and arrival stretching are modeling choices; different
   scaling (e.g. p95 0.5) collapses policy differences
   (`baseline_sanity_check.md`).
3. Survivorship bias in the cleaning rules (only clean lifecycles).
4. Single resource-class view: no priorities, no preemption, no backfilling.
5. No action masking; RL pays invalid-action penalties heuristics never incur
   (they are capacity-aware rules).
6. 3 seeds (2 for PPO; its seed-2 run collapsed and is excluded —
   `hyperparameters.json` -> `evaluation_notes`) is a small sample.
7. Heuristic tie means the experiment mostly differentiates *how badly* RL
   handles the env, not which classical rule is better.
8. The FIS inputs (three) and its 18-rule base are hand-designed; a different
   rule set would change FIS and therefore the GA's starting point.
9. The GA trains on five fixed 500-job windows from a limited load range
   (rho 0.6-1.0). Its fitness is piecewise-constant because FIS decisions are
   `argmax`, so tuning gains are small and do not reliably transfer to the
   held-out scenarios; the three GA seeds are reported as a robustness check.
10. Soft-computing policies, like the heuristics, are capacity-aware by
    construction and cannot make invalid choices, so part of their advantage
    over RL is the absence of an invalid-action failure mode rather than
    better placement.

## 10. Future work

Action masking (`sb3_contrib.MaskablePPO` — removes the invalid-action class
entirely and typically speeds learning), heterogeneous server capacities,
priority queues + preemption, continuous resource allocation
(black-box placement), and curriculum on load (train rho-major scenarios
separately). For the soft-computing side: a smoother fitness (e.g. a softmax
over server scores instead of a hard `argmax`) would give the GA a gradient to
climb, and more training episodes / a wider rho range would test whether the
tuned FIS generalizes.
