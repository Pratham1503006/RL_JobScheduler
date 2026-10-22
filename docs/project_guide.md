# Project guide — everything, from architecture to dataset to demo

A self-contained explanation of the RL-Based Dynamic Cloud Job Scheduler.
Every number here is traceable to a file under `results/` or `data/`
(the formal report in `docs/report.md` uses the same sources).

---

## 1. The problem in one paragraph

A cloud cluster receives jobs and must decide, **for the job at the head of a
FIFO queue**, which of 10 identical servers should run it right now — or
whether to wait. Each server has capacity 1.0 CPU + 1.0 memory; each job
demands a fraction of both and runs for a fixed duration. Premature waiting
wastes capacity; premature placement fragments it and blocks big jobs. The
objective is **minimize total waiting time** (the user-visible metric) while
keeping the cluster busy. Four RL algorithms (DQN, Double DQN, PPO, A2C) are
compared head-to-head against three capacity-aware classical heuristics
(Round Robin, First-Fit, Best-Fit) under three load scenarios, on held-out
test jobs neither the agents nor the tuning ever saw.

---

## 2. Architecture — the big picture

```
             DATA PIPELINE                          RUNTIME
 ┌───────────────────────────────────┐   ┌────────────────────────────────┐
 │ Google ClusterData2011_2 shards   │   │ env/cluster.py                 │
 │ (10 of 500, task_events)          │   │   Server x10 + completion heap │
 │          │ download               │   │        ^                       │
 │          v                        │   │        │ state                 │
 │ data/clean_trace.py               │   │ env/scheduling_env.py          │
 │   12 rules, audited counts        │   │   CloudSchedulingEnv (gymnasium│
 │          │                        │   │   event-driven, obs 23 floats, │
 │          v                        │   │   action Discrete(11))         │
 │ data/jobs_clean.csv (205,713)     │   │        ^            ^         │
 │          │ 70/30 time split       │   │        │            │         │
 │          v                        │   │ baselines/       agents/       │
 │ data/build_scenarios.py           │   │ heuristics.py    train.py      │
 │   REQ_SCALE=10, rho calibration   │   │ RR/FF/BF         evaluate.py   │
 │          │                        │   │ (policy(obs,env)) (SB3, [128]²)│
 │          v                        │   └────────────────────────────────┘
 │ data/scenarios/{light,heavy,      │                │
 │                 bursty}.csv       │                v
 └───────────────────────────────────┘   ┌────────────────────────────────┐
                                          │ results/                       │
                                          │  full_comparison.csv           │
                                          │  summary_table.csv  analysis.md│
                                          │  traces/*.json (42 replays)    │
                                          │  train_logs/  charts/          │
                                          │        │                       │
                                          │        v                       │
                                          │ dashboard/ (React, no build)   │
                                          │  gen_data.py -> data.json      │
                                          │  index.html + app.js replay    │
                                          └────────────────────────────────┘
```

Five layers, each with one job:

| Layer | Files | Responsibility |
|---|---|---|
| Config | `config.py` | **Every** constant: paths, server count, reward weights, ρ targets, seeds, timesteps. No magic numbers anywhere else. |
| Data | `data/*.py` | Raw shards → cleaned jobs → calibrated scenario CSVs |
| Simulator | `env/cluster.py`, `env/scheduling_env.py` | Event-driven gymnasium env shared by ALL policies |
| Policies + harness | `baselines/heuristics.py`, `agents/*.py` | 3 heuristics + 4 RL algos, trained and evaluated through one loop |
| Reporting | `charts.py`, `dashboard/`, `results/`, `docs/` | Tables, 4 charts, replay dashboard, this doc |

Key architectural rule: **one environment, one harness.** Heuristics are
callables `policy(obs, env) -> action` on the exact same env class the RL
agents see; everything runs through `agents/evaluate.py`. The "same
environment, same scenarios, same metrics" fairness claim is structural, not
aspirational.

---

## 3. Dataset — from Google trace to scenario CSVs

### 3.1 Source

Google **ClusterData2011_2**, `task_events` table — the standard public
cluster-trace benchmark. We use the **first 10 of 500 shards** (~250 MB gz,
~1.83M events). Schema (no header in the raw files, column order fixed in
`data/clean_trace.py`):

```
timestamp, missing_info, job_id, task_index, machine_id, event_type,
user, scheduling_class, priority, cpu_request, memory_request,
disk_request, different_machine_constraint
```

Timestamps are **microseconds**; event types: 0=SUBMIT, 1=SCHEDULE,
4=FINISH, and the "dirty" set 2/3/5/6 = EVICT/FAIL/KILL/LOST.
Two sentinels matter: `ts=0` (submitted before the window) and
`ts=2^63-1` (happened after the window) — both mean "censored, unusable".

### 3.2 Unit of work

One trace **task** = one **job** in our simulator, with a row:

```
job_id, arrival_time (=SUBMIT ts), duration (=FINISH-SCHEDULE),
cpu_request, mem_request
```

Events are aggregated per `(job_id, task_index)` **once**, then the cleaning
rules are applied as sequential row filters — so every rule's removal count
is attributable to exactly one reason (no double counting).

### 3.3 Cleaning — 12 rules, all audited

Full audit table with before/after counts: `data/cleaning_report.md`.
Pipeline: **1,827,856 events → 464,004 unique tasks → 205,713 kept.**

| # | rule | removed/changed | why |
|---|---|---|---|
| 1 | sentinel timestamps (0 / 2^63-1) | 144,339 tasks | arrival/end unknown |
| 2 | partial/reconstructed records (`missing_info`) | 0 | record unreliable |
| 3 | exact duplicate events | deduped | same (job, task, event, ts) |
| 4 | non-clean lifecycle (EVICT/FAIL/KILL/LOST, multi-SUBMIT) | 79,216 | duration censored |
| 5 | no FINISH in the 10-shard window | 33,503 | end unknowable |
| 6 | blank CPU/mem request | 0 | demand unknown |
| 7 | zero/negative request | 1,233 | unschedulable |
| 8 | request > 1.0 | clipped to 1.0 | normalized to largest machine |
| 9 | non-positive duration | 0 | clock artifacts |
| 10 | sub-second duration | clipped to 1 s | avoid zero-duration jobs |
| 11 | duration > p99.9 | 206 winsorized at 22,879.9 s | keep hard cases, cap extremes |
| 12 | same-timestamp bursts | **kept as-is** | real workload signal |

**Known bias (accepted & documented):** keeping only tasks submitted,
scheduled, *and* finished inside the window favors short tasks
(survivorship bias). The kept distribution (final percentiles in the
cleaning report): p50 duration 462 s, p90 4,566 s, p99 11,651 s;
p50 CPU request 0.025 (i.e., 2.5% of a machine *before* scaling).

### 3.4 Train/test split — leak-proof by construction

Split **by time**: first **70% by arrival → training pool (143,999 jobs)**
(`data/scenarios/train_pool.csv`); last **30% (61,714 jobs) is held out**.
Scenario CSVs are built **only from the held-out 30%**, so agents never
train on the exact jobs they are evaluated on. RL training episodes sample
random 1,000-job contiguous windows from the *pool* only.

### 3.5 Scenario construction & calibration (`data/build_scenarios.py`)

Two calibrations, both recorded in `data/scenarios/calibration.json`:

1. **Request scaling — `REQ_SCALE = 10.0`.** Raw requests are normalized to
   a 12,000-machine cluster, so a typical task asks for ~2.5% of *our*
   machine; on 10 servers packing would be trivial and every policy would
   tie. `REQ_SCALE` is chosen so **p95(scaled CPU) = 1.0** (one full server).
   *(config.py records why not 0.5: at p95=0.5 the phase-3 sanity check tied
   every heuristic within 2 s.)*
2. **Load calibration — arrival stretching.** Raw arrivals are far faster
   than 10 servers can serve, so each scenario's arrival times are scaled
   by a factor `s` (durations untouched) such that offered load
   `ρ = Σ(cpu·duration) / (T·N)` **hits the target exactly**:

| scenario | target ρ | achieved ρ | arrival factor | jobs |
|---|---|---|---|---|
| light | 0.40 | 0.40 | ×37.6083 | 2,000 |
| heavy | 0.90 | 0.90 | ×16.7148 | 2,000 |
| bursty | 0.60 mean | 0.60 | ×34.7873 | 2,000 |

Bursty additionally takes arrivals from the highest-CV window (CV 3.07 vs
pool 3.33) so it has genuine spikes, not just average load.

`data/explore.py` produces the three sanity plots (arrival rate, request
distribution, duration distribution) → `results/data_exploration/`.

---

## 4. The simulator

### 4.1 `env/cluster.py` — physical state

- `Server`: `used/free` CPU & memory against caps (1.0 each). A `EPS=1e-9`
  tolerance absorbs float drift so place/release restores capacity exactly
  (acceptance-tested in `tests/`).
- `Cluster`: 10 servers + a **min-heap of running jobs keyed by completion
  time**. `pop_completed(t)` frees capacity for everything finishing by `t`.
  Completions are processed **before** same-time arrivals — free first,
  place immediately, like a real scheduler.

### 4.2 `env/scheduling_env.py` — `CloudSchedulingEnv`

**Event-driven, no clock ticks** (chosen because a real scheduler acts at
event instants; ticks would multiply decisions without adding scheduling
freedom). One agent step = one decision about the head job. Time advances
only on WAIT, an invalid pick, or when the queue empties
(`_fast_forward_no_decisions`).

- **Observation — 23 floats in [0,1]:**
  `[free_cpu ×10, free_mem ×10, head_cpu, head_mem, min(1, queue_len/50)]`
- **Action — `Discrete(11)`:** 0–9 = place head job on server k, 10 = WAIT.
- **Reward** (weights in `config.py`, each justified there):
  - `+W_PLACE(1.0)` per placement
  - `−W_WAIT(1.0) × queue_len × dt / WAIT_NORM(60)` per simulated second —
    summed over an episode this equals −(total wait)/60 (Little's-law
    argument), so **maximizing return ≡ minimizing total waiting time**
  - `−W_INVALID(20.0)` per invalid pick — must dominate one WAIT step so
    stalling is never profitable. Raised 5→20 once after training showed
    >20% invalid rates (`results/hyperparameters.json` →
    `reward_weight_change_note`); no further tuning after that.
  - `+W_UTIL(0.1) × util × dt / 60` — small utilization shaping
- **Invalid picks penalized AND advance time** — an invalid action can never
  loop in zero time.
- **Episode end:** terminated when all jobs arrived + queue empty + nothing
  running; truncated at `MAX_STEPS_FACTOR(10) × n_jobs` steps (anti-stall
  guard).
- **Metrics** on termination: `avg_wait, p95_wait, avg_cpu_util`
  (time-integral of cluster utilization / makespan), `completion_rate`,
  `invalid_actions`, `makespan`.
- **Trace recording** (`record_trace=True`): every state change appends
  `{time, server_cpu_used[10], server_mem_used[10], queue_len, jobs_done}`
  → the JSON files the dashboard replays (schema: `docs/trace-schema.md`).

**Train vs eval modes.** `mode="train"`: random 1,000-job contiguous window
from the pool + **domain randomization of ρ ~ U(0.30, 1.00)** (arrival times
rescaled) so the agent can't memorize one load level.
`mode="eval"`: loads a fixed scenario CSV — deterministic job set.
Implementation note: numpy arrays instead of pandas rows in the hot loop
(pandas views across resets segfaulted; also ~faster over millions of steps).

---

## 5. Policies

### 5.1 Heuristics (`baselines/heuristics.py`)

All **capacity-aware** (a heuristic that assigns to full servers would be a
strawman) and all fall back to WAIT when nothing fits:

| policy | rule |
|---|---|
| Round Robin | rotating pointer; first server (from pointer) that fits |
| First-Fit | lowest-index server that fits |
| Best-Fit | server minimizing leftover (free_cpu−cpu)+(free_mem−mem) |

Constructed via factories each episode (Round Robin carries pointer state).

### 5.2 RL algorithms (`agents/train.py`)

SB3, `MlpPolicy`, **`[128,128]` for all four (fairness)**, CPU, seeded.
One CLI: `python -m agents.train --algo {dqn,ddqn,ppo,a2c} --seed N`.

| algo | steps | headline hyperparameters |
|---|---|---|
| DQN | 150k | lr 3e-5 (Plan-B tuning from 1e-4 after a reward collapse), buffer 50k, batch 64, γ .99, eps .3→.05, target update 500, grad clip 1.0 |
| Double DQN | 150k | **identical to DQN** — only the target-Q computation differs |
| PPO | 200k | lr 3e-4, n_steps 1024, clip .2, ent .01, GAE λ .95 |
| A2C | 200k | lr 7e-4, n_steps 16, ent .01, vf .5 |

Full per-algorithm dicts live in `results/hyperparameters.json`.

**Double DQN (`agents/double_dqn.py`)** is the one piece of original
algorithm code: a subclass of SB3's `DQN` that copies `train()` and changes
exactly one block — vanilla DQN *selects and evaluates* the next action
with the target network; DDQN **selects with the ONLINE net and evaluates
with the TARGET net** (van Hasselt et al., 2016). This decouples selection
error from evaluation error, reducing max-operator overestimation.

**Seeds:** 0/1/2 per algorithm → `models/{algo}_seed{N}.zip`. PPO seed 2
collapsed twice independently on light (completion 0.002/0.017) and was
**excluded rather than tuned** (plan rule: no hyperparameter changes after
Phase 4) — documented in `hyperparameters.json → evaluation_notes`. Final
counts: 12 model files on disk (PPO seed 2 included but excluded from
evaluation), PPO evaluated on 2 seeds.

**Logging (`agents/callbacks.py`):**
- `QOverestimationCallback` — collects a **fixed 500-observation pool**
  (seeded RNG) before training; every 5,000 steps records
  `mean(max_a Q(s,a))` on *those exact observations* → zero measurement
  noise in the DQN-vs-DDQN plot → `results/train_logs/qvalues_*.csv`.
- `EpisodeRewardCallback` — per-episode reward rows →
  `results/train_logs/rewards_*.csv` (training curves without tensorboard).

---

## 6. Evaluation harness (`agents/evaluate.py`)

The single fairness chokepoint — every policy type runs this exact loop:

- **Heuristics:** deterministic, 1 episode per scenario (invalid=0,
  completion=1.0).
- **RL:** wrapped as `model.predict(obs, deterministic=True)` (measure the
  learned policy, not exploration noise), **one episode per saved seed** —
  the seed dimension is carried by *which trained model* is loaded
  (`agents/policy_loader.py`), since re-running the same deterministic
  model on the same fixed job set would be an identical row.
- **Outputs:**
  - `results/full_comparison.csv` — per-episode rows (append)
  - `results/summary_table.csv` — mean/std per (policy, scenario)
    (generated by `summarize()`)
  - `results/traces/*.json` — replay traces when tracing is on. Naming:
    heuristics `{key}_{scenario}_0.json`, RL
    `{key}_seed{N}_{scenario}_0.json` → **42 files** (9 heuristic + 33 RL).
    *(An earlier version omitted the scenario from the name, so the
    3-scenario loop overwrote itself — fixed in Phase 6 and all traces
    regenerated.)*
- **CLI flags:** `--policies {heuristic,rl}`, `--scenarios`,
  `--seeds`, `--episodes`, `--no-traces`, `--no-csv` (trace-only rerun that
  never touches the result CSVs).

---

## 7. Results & findings

Headline table (avg wait, seconds; full metrics incl. p95, utilization,
invalid counts, std in `results/summary_table.csv`):

| policy | light (ρ .40) | heavy (ρ .90) | bursty (ρ .60 mean) |
|---|---|---|---|
| **Round Robin** | **3,084** | **4,190** | **46,871** |
| First-Fit | 3,090 | 4,193 | 46,876 |
| Best-Fit | 3,091 | 4,192 | 46,874 |
| Double DQN | 3,829 | 5,131 | 110,909 |
| A2C | 4,931 | 6,247 | 65,113 |
| DQN | 34,236 | 35,791 | 317,668 |
| PPO (2 seeds) | 36,281 | 37,847 | 312,593 |

**Finding 1 — no RL agent beats the heuristics** (honest negative result).
Data-backed reason: every RL policy burns ~1,200–2,000 invalid picks per
2,000-job episode (≈1 per job — try a full server, pay the time advance),
each costing a completion interval of wait. Meanwhile
`results/baseline_sanity_check.md` *proves* packing is trivial in this env
(items below 50% of capacity can't create unavoidable fragmentation), so
the only remaining skill is queue discipline — exactly where RL pays its
invalid-action tax. The heuristic tie (<0.2%) is real behavior, not a bug.

**Finding 2 — Double DQN is clearly better than DQN and *sane*.** The
controlled experiment: same hyperparameters, same budget, different target
computation only. Episode rewards ~−30k (DDQN) vs ~−900k (DQN); and the
value-estimate signature: DDQN's mean max-Q (−1,738 @150k) implies ≈−17.4
per step against realized −19..−58/step (consistent), while DQN's (−329)
implies −3.3/step against realized −35..−1,057/step (estimates 10–320×
detached from reality) — the classic overestimation collapse Double DQN
exists to fix. Plot: `results/charts/4_qvalue_overestimation.png`.

**Finding 3 — statistical honesty.** RL seed spreads are 1–3% on average
but explode on bursty (DDQN std 42,627 on mean 110,909); heuristic-vs-RL
gaps (22% to >7×) dwarf seed noise; DQN-vs-PPO differences are within each
other's spread → treated as a tie. Heuristic rows are single deterministic
runs (std 0 by definition).

---

## 8. Visualization

### 8.1 Report charts (`charts.py` → `results/charts/`, 4 PNGs)

| # | chart | data source |
|---|---|---|
| 1 | avg wait per scenario (grouped bars, **log y**) | `summary_table.csv` |
| 2 | avg CPU utilization per scenario | `summary_table.csv` |
| 3 | utilization timeline, heavy: RR vs DDQN | `results/traces/*.json` |
| 4 | mean max-Q over training: DQN vs DDQN | `train_logs/qvalues_*.csv` |

Chart 1 uses a log axis because RL's 100k+ stall waits would otherwise
flatten the heuristics into invisibility. Error bars = std across seeds.
Consistent policy colors across charts 1–3.

### 8.2 Replay dashboard (`dashboard/`)

**React 18 with zero build step:** `index.html` + `app.js`, React UMD + htm
**vendored locally** in `dashboard/vendor/` (no CDN, no npm, works offline).
Architecture:

- `dashboard/gen_data.py` → `data.json` manifest: the trace file for every
  **(scenario × policy)** pair — 3 × 7 = **21 replays** — plus the summary
  table rows and the 4 chart PNGs. RL traces use seed 0 per algorithm
  (documented in the UI); heuristics are deterministic single runs.
- `app.js` components: `App` (dropdowns) → `ReplayArea` (trace fetch +
  playback) and `SummaryTable`; charts rendered as plain `<img>`.
- **Replay mechanic:** one tick = one trace event (a simulator state
  change), cadence from the speed slider → replaying the events in order
  reproduces the exact episode that produced the CSV row.
- **Plan rule: no live RL inference in the browser** — everything is
  precomputed, so the replay is instant and cannot fail at runtime.
- UI: scenario dropdown, policy dropdown, Play/Pause/Restart, time slider
  (scrub any event), speed slider (5–200 events/s), live counters
  (simulated time / queue length / jobs completed), 10 CPU bars
  (red = >85%), embedded summary table, all 4 charts.

Serve from the **repo root** (paths in `data.json` are relative to it):

```bash
python -m http.server 8000     # -> http://localhost:8000/dashboard/
```

---

## 9. Testing & verification

- `tests/` — pytest suite: env mechanics (place/release restores capacity
  to 1e-9, wait-accounting equals ground truth, truncation guard) and
  heuristic validity (never overfills a server, 0 invalids).
  **10/10 passing** (`.pytest_cache` lastfailed = `{}`).
- Baseline sanity gate: `results/baseline_sanity_check.md` (Phase 3
  acceptance) — the documented proof that heuristic ties are inherent.
- Report-number audit: every figure in README/report is asserted against
  `summary_table.csv` by a check script (Phase 7).
- Dashboard verification: headless Chrome DOM inspection — 10 server bars,
  counters, both dropdowns, table, charts present; **0 JS console errors**;
  screenshot at `results/dashboard_screenshot.png`.

---

## 10. Reproduction (the exact command order)

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
# 1) data — download shards first (see README for the 10 URLs)
python -m data.clean_trace        # -> jobs_clean.csv + cleaning_report.md
python -m data.build_scenarios    # -> scenario CSVs + calibration.json
python -m data.explore            # -> results/data_exploration/
# 2) sanity + baselines
python -m agents.evaluate --policies heuristic --no-traces
# 3) train (each algo x seeds 0/1/2, separate terminals)
python -m agents.train --algo dqn  --seed 0     # ... repeat for all
# 4) full evaluation (all 7 policies x 3 scenarios)
python -m agents.evaluate
# 5) report artifacts
python charts.py                  # -> results/charts/*.png
python dashboard/gen_data.py      # -> dashboard/data.json
python -m http.server 8000        # -> dashboard at /dashboard/
# 6) tests
python -m pytest tests/ -q
```

All tunables: `config.py`. All per-run hyperparameters:
`results/hyperparameters.json`.

---

## 11. Limitations (stated, not hidden)

1. 10 identical servers — no heterogeneity, priorities, or preemption.
2. `REQ_SCALE=10` and arrival stretching are modeling choices that shape
   difficulty (documented in `calibration.json` / `config.py`).
3. Survivorship bias: only cleanly-finishing tasks kept (favors short jobs).
4. No action masking: RL *can* pick invalid actions; heuristics cannot —
   this asymmetry is real but is also exactly what the RL agents must learn
   to avoid.
5. 3 seeds (PPO 2) is a small sample; one PPO seed collapsed and was
   excluded, not tuned.
6. Heuristics tie because packing is trivial at these request
   distributions; all differentiation comes from queue behavior.
7. Dashboard replays are precomputed (plan rule), not live inference.

## 12. Where everything lives

| I want… | Open… |
|---|---|
| every constant/weight | `config.py` |
| cleaning audit | `data/cleaning_report.md` |
| calibration factors | `data/scenarios/calibration.json` |
| env contract & reward math | `env/scheduling_env.py` docstring |
| hyperparameters | `results/hyperparameters.json` |
| per-episode results | `results/full_comparison.csv` |
| mean±std table | `results/summary_table.csv` |
| interpretation | `results/analysis.md` |
| why heuristics tie | `results/baseline_sanity_check.md` |
| formal report | `docs/report.md` |
| trace JSON schema | `docs/trace-schema.md` |
