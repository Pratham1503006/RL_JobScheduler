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

## 7. Results

Full table `results/summary_table.csv` (mean ± std across seeds),
plots `results/charts/1-4*.png`. **Average wait (s):**

| policy | light | heavy | bursty | util heavy | invalid/ep (heavy) |
|---|---|---|---|---|---|
| Round Robin | 3,084 | 4,190 | 46,871 | 0.882 | 0 |
| First-Fit | 3,090 | 4,193 | 46,876 | 0.881 | 0 |
| Best-Fit | 3,091 | 4,192 | 46,874 | 0.881 | 0 |
| DQN | 34,236 | 35,791 | 317,668 | 0.181 | 1,993 |
| Double DQN | 3,829 | 5,131 | 110,909 | 0.752 | 1,404 |
| PPO | 36,281 | 37,847 | 312,593 | 0.166 | 1,990 |
| A2C | 4,931 | 6,247 | 65,113 | 0.684 | 184 |

Round Robin wins every scenario; the LLN-style "clean winner" is not RL here —
see discussion. Statistical honesty: RL seed spreads are 1-3% on average but
explode on bursty (DDQN std 42,627 on 110,909 mean, one truncated seed with
completion 0.963), while the heuristic-vs-RL gaps (~22% to >7x) dwarf seed
noise; DQN vs PPO differences on identical scenarios are within each other's
spread — treat as tie.

## 8. Discussion

**Who won where and why.** All three heuristics tie within ~0.2%; RL never
beats them. Data-backed reason: every RL agent makes ~1,400-2,000 invalid picks per
2,000-job episode, and each invalid pick costs one completion-interval of
advanced time; the measured gap to Round Robin is +745 s (light, DDQN) and
+940 s (heavy, DDQN), while the far larger DQN/PPO gaps (31-271k s) reflect
policies that mostly stall (util 0.07-0.18, wait 7-12x heuristics on average). Deeper: `baseline_sanity_check.md`
shows packing in this environment is essentially trivial (below-50% items
never create unavoidable fragmentation), so the only remaining learning
signal is queue discipline — where holding the head job costs more relative
  to the tiny placement gains.

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

## 10. Future work

Action masking (`sb3_contrib.MaskablePPO` — removes the invalid-action class
entirely and typically speeds learning), heterogeneous server capacities,
priority queues + preemption, continuous resource allocation
(black-box placement), and curriculum on load (train rho-major scenarios
separately).
