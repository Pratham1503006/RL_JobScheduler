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

# ==================================================== P6: SOFT COMPUTING =====
# Two non-learning soft-computing schedulers, plugging into the SAME
# `policy(obs, env) -> action` interface as the heuristics:
#   SC1 FIS     -- hand-designed zero-order Sugeno fuzzy inference scheduler
#   SC2 GA-FIS  -- the same FIS, membership breakpoints + rule outputs evolved
#                  by a genetic algorithm on data/scenarios/train_pool.csv ONLY.
#
# WHY these three inputs: `leftover` captures fit (how much slack the server
# keeps after the placement), `load` captures how busy that server already is,
# and `pressure` (the queue) decides whether the scheduler should PACK (busy
# queue -> keep big free blocks) or BALANCE (calm queue -> spread jobs).
FUZZY_INPUTS = ("leftover", "load", "pressure")
FUZZY_SETS = {
    "leftover": ("TIGHT", "MEDIUM", "LOOSE"),
    "load": ("LOW", "MEDIUM", "HIGH"),
    "pressure": ("CALM", "BUSY"),
}

# Chromosome encoding: two cut points per input, mapped to full-coverage
# trapezoids by softcomputing.fuzzy.membership_params().  These defaults ARE
# the hand-designed membership functions from the addendum:
#   leftover(0.1, 0.6) -> TIGHT (0,0,.1,.35), MEDIUM (.1,.35,.35,.6), LOOSE (.35,.6,1,1)
#   load    (0.2, 0.8) -> LOW (0,0,.2,.5), MEDIUM (.2,.5,.5,.8), HIGH (.5,.8,1,1)
#   pressure(0.2, 0.6) -> CALM (0,0,.2,.6), BUSY (.2,.6,1,1)
FUZZY_CUTS_DEFAULT = {
    "leftover": (0.1, 0.6),
    "load": (0.2, 0.8),
    "pressure": (0.2, 0.6),
}

# The 18-rule zero-order Sugeno base (3 x 3 x 2), in leftover-major order.
# z is the crisp consequent "suitability" in [0, 1] used in the weighted
# average score.  Design logic (auditable table, addendum 3.4):
#   * BUSY  -> prefer PACKING: TIGHT leftover + HIGH load scores highest
#              (keep the big free blocks for large waiting jobs), LOOSE+LOW
#              scores lowest (do not burn an empty server on a small job).
#   * CALM  -> prefer BALANCING: LOOSE leftover + LOW load scores highest,
#              TIGHT+HIGH scores lowest (leave headroom, spread the jobs).
#   * MEDIUM leftover + HIGH load is penalised (0.15 busy / 0.50 calm):
#              half-full busy servers are exactly the fragmentation trap.
FUZZY_RULES = [
    # -- leftover TIGHT -------------------------------------------------
    ("TIGHT", "LOW", "CALM", 0.50),
    ("TIGHT", "LOW", "BUSY", 0.75),
    ("TIGHT", "MEDIUM", "CALM", 0.35),
    ("TIGHT", "MEDIUM", "BUSY", 0.90),
    ("TIGHT", "HIGH", "CALM", 0.20),
    ("TIGHT", "HIGH", "BUSY", 1.00),
    # -- leftover MEDIUM ------------------------------------------------
    ("MEDIUM", "LOW", "CALM", 0.70),
    ("MEDIUM", "LOW", "BUSY", 0.50),
    ("MEDIUM", "MEDIUM", "CALM", 0.65),
    ("MEDIUM", "MEDIUM", "BUSY", 0.40),
    ("MEDIUM", "HIGH", "CALM", 0.50),
    ("MEDIUM", "HIGH", "BUSY", 0.15),
    # -- leftover LOOSE -------------------------------------------------
    ("LOOSE", "LOW", "CALM", 1.00),
    ("LOOSE", "LOW", "BUSY", 0.25),
    ("LOOSE", "MEDIUM", "CALM", 0.80),
    ("LOOSE", "MEDIUM", "BUSY", 0.30),
    ("LOOSE", "HIGH", "CALM", 0.60),
    ("LOOSE", "HIGH", "BUSY", 0.35),
]

# --- GA (addendum 4.4) ------------------------------------------------------
FUZZY_MIN_GAP = 0.02              # smallest distance between a pair of cut points
GA_POP_SIZE = 30
GA_GENERATIONS = 30
GA_TOURNAMENT = 3                 # tournament selection size
GA_CROSSOVER_PROB = 0.8           # BLX-alpha applied with this probability
GA_BLX_ALPHA = 0.3                # blend-crossover expansion factor
GA_MUTATION_PROB = 0.1            # per-gene Gaussian mutation probability
GA_MUTATION_SIGMA = 0.05          # Gaussian mutation standard deviation
GA_INIT_SIGMA = 0.10              # noise around the seeded default individual
GA_ELITES = 2                     # top individuals carried over unchanged
GA_SEEDS = (0, 1, 2)              # independent GA runs
# --- GA fitness (addendum 4.3) ----------------------------------------------
GA_EVAL_EPISODES = 5              # training episodes per chromosome
GA_EPISODE_JOBS = 500             # jobs per training episode window
GA_RHO_RANGE = (0.6, 1.0)         # only these loads separate good policies
GA_PENALTY_INVALID = 1.0e5        # safety net; invalid actions should be 0
GA_WORKERS = min(8, os.cpu_count() or 1)

# Policy names used in results/full_comparison.csv and the dashboard.
SOFT_POLICY_NAMES = {"fis": "soft.fis", "ga_fis": "soft.ga_fis"}
