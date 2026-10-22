# RL-Based Dynamic Cloud Job Scheduler

Deep-RL job scheduling on an event-driven 10-server cluster simulator, with the
workload taken from the real **Google ClusterData2011_2** trace, and compared
head-to-head with capacity-aware classical heuristics.

## Problem

Decide, for the job at the head of a FIFO queue, which of 10 identical servers
(CPU and memory, capacity 1.0 each) should run it — or wait. Goal: minimize
**total waiting time** while keeping the cluster busy. Four RL algorithms
(DQN, Double DQN, PPO, A2C) face three capacity-aware heuristics
(Round Robin, First-Fit, Best-Fit) under three load scenarios
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
- **Fairness:** same env, same scenarios, same metrics harness
  (`agents/evaluate.py`), same net `[128,128]`, held-out test jobs, fixed seeds.

## Repo layout

```
config.py              ALL constants (servers, reward weights, hyperparameters)
data/                  raw shards, clean_trace.py, build_scenarios.py, explore.py
data/scenarios/        light.csv heavy.csv bursty.csv train_pool.csv calibration.json
env/                   cluster.py (Server/Cluster), scheduling_env.py (gym env)
baselines/heuristics.py
agents/                train.py evaluate.py double_dqn.py callbacks.py policy_loader.py
models/                {algo}_seed{N}.zip (12 saved models)
results/               summary_table.csv full_comparison.csv analysis.md
results/charts/        the 4 report charts
results/traces/        per-(policy,scenario,seed) replay traces (JSON)
dashboard/             React replay dashboard (no build step; vendored React)
tests/                 pytest suite for env + baselines
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
# 4) full evaluation + summary
python -m agents.evaluate           # all 7 policies x 3 scenarios -> full_comparison.csv
# 5) charts + dashboard
python charts.py                    # -> results/charts/*.png
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
| Round Robin | **3,084** | **4,190** | **46,871** |
| First-Fit | 3,090 | 4,193 | 46,876 |
| Best-Fit | 3,091 | 4,192 | 46,874 |
| DQN | 34,236 | 35,791 | 317,668 |
| Double DQN | 3,829 | 5,131 | 110,909 |
| PPO (2 seeds) | 36,281 | 37,847 | 312,593 |
| A2C | 4,931 | 6,247 | 65,113 |

Full metrics (`p95_wait`, utilization, invalid actions, makespan, std over
seeds) in `results/summary_table.csv`; per-episode rows in
`results/full_comparison.csv`; interpretation in `results/analysis.md`.

## Key findings (honest)

1. **No RL agent beats the heuristics.** RL burns ~1,200-2,000 invalid actions
   per episode (one per job: it tries a full server first, then pays the time
   advance), so it adds roughly one completion interval of wait per job.
   The environment's packing itself is provably easy (see
   `results/baseline_sanity_check.md`), so the decision space rewards queue
   discipline more than clever placing — valid, reportable result.
2. **Double DQN is clearly better than DQN** and *sane*: DDQN's learned
   Q-values track its realized returns; vanilla DQN's Q-values are detached
   from reality (overestimation collapse). Trained episode rewards ~ -30k vs
   ~ -900k (`results/train_logs/rewards_*.csv`).
3. Heuristics tie within ~0.2% everywhere (`results/baseline_sanity_check.md`
   shows why this is real behavior, not a bug).

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

## Dashboard

The React replay dashboard replays the committed traces next to the summary
table and the report charts: serve the repo root with
`python -m http.server 8000` and open http://localhost:8000/dashboard/
(heavy + Double DQN vs Round Robin). A screenshot of the rendered page is in
`results/dashboard_screenshot.png`.
