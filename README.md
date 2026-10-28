# RL-Based Dynamic Cloud Job Scheduler

Deep-RL job scheduling on an event-driven 10-server cluster simulator, with the
workload taken from the real **Google ClusterData2011_2** trace, and compared
head-to-head with capacity-aware classical heuristics and two soft-computing
schedulers.

## Problem

Decide, for the job at the head of a FIFO queue, which of 10 identical servers
(CPU and memory, capacity 1.0 each) should run it — or wait. Goal: minimize
**total waiting time** while keeping the cluster busy. Four RL algorithms
(DQN, Double DQN, PPO, A2C) face three capacity-aware heuristics
(Round Robin, First-Fit, Best-Fit) and two soft-computing schedulers
(**FIS** and **GA-FIS**) under three load scenarios
(light rho=0.40, heavy rho=0.90, bursty mean rho=0.60), on **held-out test
jobs** the agents never trained on.

## Approach

- **Data:** first 10 shards of `task_events` (1.83M events -> 205,713 clean
  task lifecycles, 12 documented cleaning rules in `data/cleaning_report.md`).
- **Simulator:** `gymnasium.Env`, event-driven (no clock ticks), obs =
  free CPU/mem of 10 servers + head-job request + queue length (23 floats),
  action = Discrete(11) (which server / wait). Reward ~ minus total wait.
- **Calibration:** raw requests scaled by `REQ_SCALE=10` (p95 CPU = 1.0);
  arrival times stretched so each scenario hits its target rho exactly
  (factors 37.6 / 16.7 / 34.8 — see `data/scenarios/calibration.json`).
- **Soft computing:** a hand-designed Fuzzy Inference Scheduler (three inputs
  — leftover slack, server load, queue pressure — 18 zero-order Sugeno rules)
  and a GA-tuned version of the same FIS; the GA evolves 24 genes on
  `data/scenarios/train_pool.csv` only.
- **Fairness:** same env, same scenarios, same metrics harness
  (`agents/evaluate.py`), same net `[128,128]`, held-out test jobs, fixed seeds.
  FIS/GA-FIS use the identical `policy(obs, env) -> action` interface.

## Repo layout

```
config.py              ALL constants (servers, reward weights, hyperparameters)
data/                  raw shards, clean_trace.py, build_scenarios.py, explore.py
data/scenarios/        light.csv heavy.csv bursty.csv train_pool.csv calibration.json
env/                   cluster.py (Server/Cluster), scheduling_env.py (gym env)
baselines/heuristics.py
softcomputing/         fuzzy.py fis_policy.py ga.py train_ga.py plots.py evaluate_soft.py
agents/                train.py evaluate.py double_dqn.py callbacks.py policy_loader.py
models/                {algo}_seed{N}.zip (12 saved models)
results/               summary_table.csv full_comparison.csv analysis.md
results/charts/        the 6 report charts
results/traces/        per-(policy,scenario,seed) replay traces (JSON)
results/ga_logs/       GA history, best chromosomes, rule tables, robustness
results/softcomputing/ FIS/GA-FIS membership plots + convergence curve
dashboard/             React replay dashboard (no build step; vendored React)
tests/                 pytest suite for env + baselines + soft computing
```

## Reproduce (in order)

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # or venv/bin/pip
# 1) data (shards land in data/raw/; see docs/trace-schema.md for the schema)
python -m data.clean_trace          # raw -> data/jobs_clean.csv + cleaning_report.md
python -m data.build_scenarios      # jobs_clean -> scenario CSVs + calibration.json
python -m data.explore              # 3 exploration plots -> results/data_exploration/
# 2) sanity + baselines
python -m agents.evaluate --policies heuristic --no-traces    # -> results/baseline_results.csv
# 3) train (3 seeds each; parallelize in separate terminals)
python -m agents.train --algo dqn  --seed 0
python -m agents.train --algo ddqn --seed 0
python -m agents.train --algo ppo  --seed 0
python -m agents.train --algo a2c  --seed 0
# 3b) soft computing: GA-tuned FIS (3 GA seeds) + held-out evaluation
python -m softcomputing.train_ga --seed 0
python -m softcomputing.train_ga --seed 1
python -m softcomputing.train_ga --seed 2
python -m softcomputing.train_ga --finalize    # selection + convergence/membership plots
python -m softcomputing.evaluate_soft          # FIS + GA-FIS -> full_comparison.csv
python -m softcomputing.record_decisions        # per-decision fuzzy audit trail
python -m softcomputing.demo --scenario heavy   # terminal step-by-step simulation
# 4) full evaluation + summary
python -m agents.evaluate           # all 7 learning/heuristic policies x 3 scenarios
# 5) charts + dashboard
python charts.py                    # -> results/charts/*.png (6 charts)
python dashboard/gen_data.py        # -> dashboard/data.json (manifest)
python -m http.server 8000          # then open http://localhost:8000/dashboard/
# 6) tests
python -m pytest tests/ -q
```

Download shards (only once, ~250 MB):
`https://storage.googleapis.com/clusterdata-2011-2/task_events/part-0000{0-9}-of-00500.csv.gz`
into `data/raw/`.

## Results (avg wait in seconds; bold = winner per column)

| Policy | light (rho .40) | heavy (rho .90) | bursty (rho .60 mean) |
|---|---|---|---|
| Round Robin | **3,084** | **4,190** | 46,871 |
| First-Fit | 3,090 | 4,193 | 46,876 |
| Best-Fit | 3,091 | 4,192 | 46,874 |
| FIS (soft) | 3,090 | 4,213 | **46,869** |
| GA-FIS (soft) | 3,091 | 4,206 | 46,877 |
| DQN | 34,236 | 35,791 | 317,668 |
| Double DQN | 3,829 | 5,131 | 110,909 |
| PPO (2 seeds) | 36,281 | 37,847 | 312,593 |
| A2C | 4,931 | 6,247 | 65,113 |

Full metrics (`p95_wait`, utilization, invalid actions, makespan, std over
seeds) in `results/summary_table.csv`; per-episode rows in
`results/full_comparison.csv`; interpretation in `results/analysis.md`.

## Key findings (honest)

1. **No RL agent beats the heuristics or the soft-computing policies.** RL
   burns ~1,200-2,000 invalid actions per episode (one per job: it tries a
   full server first, then pays the time advance), so it adds roughly one
   completion interval of wait per job. The environment's packing itself is
   provably easy (see `results/baseline_sanity_check.md`), so the decision
   space rewards queue discipline more than clever placing — valid, reportable
   result.
2. **Double DQN is clearly better than DQN** and *sane*: DDQN's learned
   Q-values track its realized returns; vanilla DQN's Q-values are detached
   from reality (overestimation collapse). Trained episode rewards ~ -30k vs
   ~ -900k (`results/train_logs/rewards_*.csv`).
3. Heuristics **and FIS/GA-FIS** tie within ~0.5% everywhere
   (`results/baseline_sanity_check.md` shows why this is real behavior, not a
   bug); the FIS is the fastest policy on `bursty` (46,869 s).
4. **GA tuning helped on the training episodes but not on the held-out
   scenarios**: default FIS fitness -2729.4602 vs -2727.3650 for the selected
   GA seed, yet GA-FIS is within ±0.17% of FIS per scenario — a tie. FIS
   decisions are `argmax` over fuzzy scores, so the fitness is piecewise
   constant and most mutations change no placement.

## Limitations

- 10 identical servers; no heterogeneity, priorities, or preemption.
- Request scaling (`REQ_SCALE=10`) and arrival stretching are modeling
  choices that shape the difficulty — documented in `calibration.json`.
- Survivorship bias: only tasks completing cleanly inside the 10-shard window
  are kept; favors shorter tasks.
- No action masking: RL can pick invalid actions that heuristics cannot.
- 3 seeds (PPO 2, its seed-2 run collapsed; see
  `results/hyperparameters.json` -> `evaluation_notes`) is a small sample.
- Heuristic scenarios tie because packing is trivial at these request
  distributions; differences come only from queue behavior.
- The fuzzy rule base is hand-designed (18 rules, three inputs); a different
  rule set would change FIS and the GA's starting point.
- The GA is trained on five fixed 500-job windows sampled from a limited
  load range (rho 0.6-1.0) and can overfit those episodes; the three GA seeds
  are reported as a robustness check.
- Soft-computing policies (like the heuristics) are capacity-aware by
  construction and cannot make invalid choices, while the RL agents can.

## Dashboard

The React replay dashboard replays the committed traces next to the summary
table and the report charts. It has two tabs: **Replay & comparison**
(scenario/policy replay, where the policy dropdown is grouped into
heuristics / soft computing / RL) and **Soft computing** (a step-by-step
*fuzzy decision inspector*, default vs GA-tuned membership plots, both
18-rule tables and the GA seed robustness tables).
Serve the repo root with `python -m http.server 8000` and open
http://localhost:8000/dashboard/ (the tabs are deep-linkable:
`dashboard/#soft`). Screenshots of the rendered page are in
`results/dashboard_screenshot.png` (replay tab) and
`results/dashboard_softcomputing.png` (soft-computing tab).

For a viva/demo, `docs/soft_computing_presentation.md` contains the full
spoken script, a demo run sheet, a real worked example, a numbers cheat
sheet and a Q&A bank. For a terminal-only demo:

```bash
python -m softcomputing.demo --scenario heavy --every 800 --max-print 4
python -m softcomputing.demo --policy ga_fis --scenario heavy --every 800
```
