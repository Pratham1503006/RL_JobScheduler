"""config.py -- ALL tunable values live here.

Every constant used anywhere in the project must be imported from this file
(no magic numbers in other files).
"""

import os

# ---------------------------------------------------------------- paths ----
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
CLEAN_CSV = os.path.join(DATA_DIR, "jobs_clean.csv")
REPORT_PATH = os.path.join(DATA_DIR, "cleaning_report.md")
SCEN_DIR = os.path.join(DATA_DIR, "scenarios")
TRAIN_POOL = os.path.join(SCEN_DIR, "train_pool.csv")
RESULTS_DIR = "results"
MODELS_DIR = "models"

# ------------------------------------------------------- trace download ----
TRACE_BASE = "https://storage.googleapis.com/clusterdata-2011-2/task_events"
SHARDS = list(range(10))                          # first 10 of 500 shards
GZ_PATHS = {i: os.path.join(RAW_DIR, f"part-{i:05d}-of-00500.csv.gz") for i in SHARDS}

# ------------------------------------------------------------ raw schema ----
RAW_TIME_UNIT = "us"              # trace timestamps are microseconds
END_SENTINEL_US = 2**63 - 1       # event after the trace window ended
EPOCH_0_US = 0.0                  # timestamp 0 = submitted before the window

# --------------------------------------------------------------- servers ----
N_SERVERS = 10                    # identical servers, single cluster slice
CPU_CAP = 1.0                     # per-server capacity, normalized
MEM_CAP = 1.0

# --------------------------------------------------- calibration targets ----
RHO_LIGHT = 0.40                  # target utilization, light scenario
RHO_HEAVY = 0.90                  # target utilization, heavy scenario
RHO_BURSTY = 0.60                 # mean load, bursty scenario
RHO_TRAIN_RANGE = (0.30, 1.00)    # load-range domain randomization for training
TARGET_P95_REQ = 1.00             # REQ_SCALE tunes p95(scaled cpu) to this capacity upper bound
# WHY REQ_SCALE=10 (p95 cpu=1.0, mean ~0.154): with p95=0.50 (mean 0.077) the
# phase-3 sanity test FAILED -- First/Best/RR tied within 2s of 1837s wait on
# heavy (packing trivial: ~13 median-jobs fit per server; ties from the trace's
# quantized 0.025/0.1/0.125 catalogue). p95=1.0 doubles contention.
REQ_SCALE = 10.0                  # SET BY PHASE-1 CALIBRATION (measured on jobs_clean.csv)
SCENARIO_JOBS = 2000              # jobs per scenario file
CAL_TOL = 0.03                    # allowed |rho - target| tolerance
TRAIN_SPLIT_FRAC = 0.70           # first 70% by arrival time -> training pool

# ---------------------------------------------------------------- seeds -----
N_SEEDS = 3                       # RL seeds per algorithm (0, 1, 2)
BASE_SEED = 42                    # master seed for data pipeline reproducibility

# ============================================================== P2: ENV =====
# WHY these reward weights: the wait term sums to (approximately) the TOTAL
# waiting time of all jobs divided by WAIT_NORM, so maximizing return ==
# minimizing total wait (Little's-law-style argument). W_PLACE is a small
# constant bonus for making progress; W_INVALID must dominate one WAIT step
# so the agent never profits from stalling on an invalid action.
W_PLACE = 1.0                     # reward for a successful placement
W_WAIT = 1.0                      # per waiting-job per simulated second (scaled)
# W_INVALID raised ONCE per plan 7.6 (2026-10-12): after full training,
# DQN/DDQN picked invalid actions >20% of eval steps and A2C collapsed to
# ~100% invalid, so the original 5.0 penalty was too weak. 5.0 -> 20.0.
W_INVALID = 20.0                  # penalty for an invalid action
W_UTIL = 0.1                      # optional utilization shaping (kept small)
WAIT_NORM = 60.0                  # seconds of waiting-job-seconds per penalty unit
QUEUE_NORM = 50.0                 # queue length divided by this, clipped to 1, in obs
MAX_STEPS_FACTOR = 10             # episode truncates at this * n_jobs agent steps
TRAIN_EPISODE_JOBS = 1000         # jobs per training episode (random window)

# ============================================================== P4: RL ======
DQN_TIMESTEPS = 150_000
PPO_TIMESTEPS = 200_000
A2C_TIMESTEPS = 200_000
EVAL_EPISODES_RL = 5              # stochastic re-runs per RL policy per scenario
Q_LOG_INTERVAL = 5_000            # timesteps between mean-max-Q measurements
Q_OBS_POOL = 500                  # fixed observation set for the Q experiment
