"""clean_trace.py -- raw Google trace shards -> data/jobs_clean.csv.

Reads the 10 raw task_events shards, builds one row per task, applies the
12 cleaning rules from the plan IN ORDER (every rule logs a count), and
writes data/jobs_clean.csv plus data/cleaning_report.md.

WHY one row per task: our simulator's unit of work is a single task with a
clean lifecycle (SUBMIT -> SCHEDULE -> FINISH); any other event pattern has an
unknown or censored duration and cannot be scheduled fairly in the simulator.

WHY a task table: aggregating events per (job_id, task_index) ONCE, then
applying the rules as sequential row filters, keeps every rule's count
attributable to exactly one reason (no double counting) and runs in seconds.
"""

import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLEAN_CSV, GZ_PATHS, RAW_DIR, REPORT_PATH  # noqa: E402

# Raw column order per the trace schema (no header row in the files).
RAW_COLS = [
    "timestamp", "missing_info", "job_id", "task_index", "machine_id",
    "event_type", "user", "scheduling_class", "priority",
    "cpu_request", "memory_request", "disk_request", "different_machine_constraint",
]
COLS = ["job_id", "arrival_time", "duration", "cpu_request", "mem_request"]
KEY = ["job_id", "task_index"]

# Sentinel meaning "event happened after the trace window ended" (2^63-1 us).
END_SENTINEL = 2**63 - 1
EV_SUBMIT, EV_SCHED, EV_FINISH = 0, 1, 4
DIRTY_EVENTS = (2, 3, 5, 6)  # EVICT, FAIL, KILL, LOST

_rule_counts: list[tuple[str, int, str, str]] = []


def log_rule(rule: str, n: int, action: str, reason: str) -> None:
    _rule_counts.append((rule, n, action, reason))


def drop_tasks(tasks: pd.DataFrame, mask: pd.Series, rule: str, action: str,
               reason: str) -> pd.DataFrame:
    """Log `mask.sum()` removals, then drop them. WHY: count must be taken
    BEFORE the drop so the report shows how many each rule removed."""
    n = int(mask.sum())
    log_rule(rule, n, action, reason)
    return tasks[~mask]


def load_events() -> pd.DataFrame:
    files = [p for i, p in sorted(GZ_PATHS.items()) if os.path.exists(p)]
    if not files:
        files = sorted(glob.glob(os.path.join(RAW_DIR, "part-*.csv.gz")))
    frames = [
        pd.read_csv(
            f, header=None, names=RAW_COLS, compression="gzip",
            usecols=["timestamp", "missing_info", "job_id", "task_index",
                     "event_type", "cpu_request", "memory_request"],
            dtype={"missing_info": str},
        )
        for f in files
    ]
    return pd.concat(frames, ignore_index=True)


def first_per_task(ev: pd.DataFrame, et: int) -> pd.DataFrame:
    """Earliest row of event type `et` per task.

    WHY earliest: matches the plan's lifecycle definition (first SCHEDULE /
    first FINISH) when a task has repeated rows of the same event type.
    """
    rows = ev[ev["event_type"] == et].sort_values("timestamp", kind="stable")
    return rows.drop_duplicates(subset=KEY, keep="first").set_index(KEY)


def build_task_table(events: pd.DataFrame) -> pd.DataFrame:
    sub = first_per_task(events, EV_SUBMIT)
    sch = first_per_task(events, EV_SCHED)
    fin = first_per_task(events, EV_FINISH)
    tasks = sub[[ "timestamp", "cpu_request", "memory_request"]].rename(
        columns={"timestamp": "submit_ts"})
    tasks = tasks.join(sch[["timestamp"]].rename(columns={"timestamp": "sched_ts"}),
                       how="outer")
    tasks = tasks.join(fin[["timestamp"]].rename(columns={"timestamp": "fin_ts"}),
                       how="outer")
    agg = events.groupby(KEY, observed=True).agg(
        n_submit=("event_type", lambda s: int((s == EV_SUBMIT).sum())),
        n_dirty=("event_type", lambda s: int(s.isin(DIRTY_EVENTS).sum())),
    )
    return tasks.join(agg, how="left").fillna({"n_submit": 0, "n_dirty": 0})


def main() -> pd.DataFrame:
    events = load_events()
    n_raw_events = len(events)
    print(f"raw events loaded: {n_raw_events}")

    # ------------------------------------------------ rule 3: duplicates ----
    # WHY first: duplicate SUBMIT/FINISH rows would otherwise fake a
    # "more than one SUBMIT" violation in rule 4 for perfectly clean tasks.
    dup_mask = events.duplicated(subset=KEY + ["event_type", "timestamp"], keep="first")
    log_rule("3 exact duplicate events", int(dup_mask.sum()),
             "dropped duplicate rows", "same (job, task, event_type, timestamp)")
    events = events[~dup_mask]

    tasks = build_task_table(events)
    n_tasks_seen = len(tasks)

    # ------------------------------------- rule 1: sentinel timestamps ----
    # WHY: timestamp 0 = submitted before the window, 2^63-1 = finished after
    # it; either means arrival or end time is unknown -> duration unusable.
    sent = (tasks["submit_ts"] == 0) | (tasks["fin_ts"] == END_SENTINEL) \
        | (tasks["sched_ts"] == END_SENTINEL)
    tasks = drop_tasks(tasks, sent.fillna(False), "1 sentinel timestamps",
                       "tasks dropped",
                       "SUBMIT or FINISH has ts==0 or 2^63-1; arrival/end unknown")

    # ------------------------------- rule 2: partial / reconstructed recs --
    # WHY: missing_info non-blank means Google reconstructed the row; its
    # timestamps/requests are estimates, so we cannot trust the duration.
    # Applied on the SUBMIT/SCHEDULE/FINISH rows of the task.
    key_of = events.set_index(KEY)
    miss_ev = key_of["missing_info"].notna() & (key_of["missing_info"].str.len() > 0)
    miss_task = miss_ev.groupby(level=[0, 1]).any()
    tasks = drop_tasks(tasks, miss_task.reindex(tasks.index).fillna(False),
                       "2 partial/reconstructed records", "tasks dropped",
                       "missing_info non-blank on any event of the task")

    # --------------------------------------- rule 4: clean lifecycle ------
    # WHY drop: evictions/failures split runtime across attempts, and a
    # resubmitted task has ambiguous arrival. Documented as survivorship bias.
    dirty = (tasks["n_dirty"] > 0) | (tasks["n_submit"] != 1)
    tasks = drop_tasks(tasks, dirty, "4 non-clean lifecycle", "tasks dropped",
                       "EVICT/FAIL/KILL/LOST present, or SUBMIT count != 1 "
                       "(limitation: survivorship bias toward completed tasks)")

    # ------------------------------------------- rule 5: no FINISH --------
    # WHY drop: the task ran past the loaded shards; duration unknowable.
    no_fin = tasks["fin_ts"].isna()
    tasks = drop_tasks(tasks, no_fin, "5 no FINISH in window", "tasks dropped",
                       "task ran past the loaded shards; duration unknowable")
    # Submitted but never scheduled: no runtime start either.
    no_sched = tasks["sched_ts"].isna()
    tasks = drop_tasks(tasks, no_sched, "5b no SCHEDULE row", "tasks dropped",
                       "submitted but never scheduled; runtime start unknown")

    # ------------------------------------------------ rule 6/7: requests --
    # WHY drop: unknown or non-positive requests are not schedulable demands.
    blank = tasks["cpu_request"].isna() | tasks["memory_request"].isna()
    tasks = drop_tasks(tasks, blank, "6 blank CPU or memory request",
                       "tasks dropped", "request unknown on the SUBMIT row")
    nonpos = (tasks["cpu_request"] <= 0) | (tasks["memory_request"] <= 0)
    tasks = drop_tasks(tasks, nonpos, "7 zero or negative request", "tasks dropped",
                       "value <= 0 is not a schedulable demand")

    # ------------------------------------------------------ timing (s) ----
    tasks = tasks.copy()
    tasks["arrival_time"] = tasks["submit_ts"] / 1e6
    tasks["duration"] = (tasks["fin_ts"] - tasks["sched_ts"]) / 1e6

    # ------------------------------------------- rule 9: bad durations ----
    nonpos_dur = tasks["duration"] <= 0
    tasks = drop_tasks(tasks, nonpos_dur, "9 non-positive duration", "tasks dropped",
                       "FINISH <= SCHEDULE (clock artifacts)")

    # ------------------------------------------- rule 10: sub-second ------
    # WHY clip, not drop: short tasks are common and legitimate; dropping them
    # would bias the workload longer.
    subsec = tasks["duration"] < 1.0
    tasks.loc[subsec, "duration"] = 1.0
    log_rule("10 sub-second durations", int(subsec.sum()),
             "durations clipped to 1s",
             "short tasks are legitimate; clipping avoids zero-duration jobs")

    # ------------------------------------------- rule 8: requests > 1 -----
    # WHY clip: requests are normalized to the largest machine, so >1 is an
    # artifact rather than a physical demand.
    over = (tasks["cpu_request"] > 1.0) | (tasks["memory_request"] > 1.0)
    tasks["cpu_request"] = tasks["cpu_request"].clip(upper=1.0)
    tasks["mem_request"] = tasks["memory_request"].clip(upper=1.0)
    log_rule("8 request above 1.0", int(over.sum()), "values clipped to 1.0",
             "requests are normalized to the largest machine; >1 is an artifact")

    # --------------------------------------- rule 11: winsorize tails -----
    # WHY winsorize, not delete: heavy tails are the realistic hard cases for
    # a scheduler; cap outliers but keep the tasks.
    p999 = tasks["duration"].quantile(0.999)
    wins = tasks["duration"] > p999
    tasks.loc[wins, "duration"] = p999
    log_rule("11 extreme long durations", int(wins.sum()),
             f"durations winsorized at p99.9={p999:.1f}s",
             "keep heavy tasks (realistic hard cases), cap the value")

    # ------------------------------------------- rule 12: bursts ----------
    log_rule("12 same-timestamp bursts", 0, "kept as is",
             "bursts are real workload signal; stable sort only, no jitter")

    # Shift the window to start at t=0 and sort by (arrival, job_id) stably.
    out = tasks["arrival_time"].to_frame()
    out["duration"] = tasks["duration"]
    out["cpu_request"] = tasks["cpu_request"]
    out["mem_request"] = tasks["mem_request"]
    out = out.reset_index()  # moves job_id/task_index from index to columns
    out["job_id"] = out["job_id"].astype(str) + "_" + out["task_index"].astype(str)
    out["arrival_time"] = out["arrival_time"] - out["arrival_time"].min()
    out = out[COLS].sort_values(["arrival_time", "job_id"],
                                kind="stable").reset_index(drop=True)

    os.makedirs(os.path.dirname(CLEAN_CSV), exist_ok=True)
    out.to_csv(CLEAN_CSV, index=False)
    write_report(n_raw_events, n_tasks_seen, len(out), out, p999)
    print(f"tasks seen: {n_tasks_seen} | kept: {len(out)}  ->  {CLEAN_CSV}")
    print(out[COLS[1:]].quantile([0.5, 0.9, 0.95, 0.99]).to_string())
    return out


def write_report(n_raw: int, n_seen: int, n_kept: int, df: pd.DataFrame,
                 p999: float) -> None:
    pct = df[["cpu_request", "mem_request", "duration"]].quantile(
        [0.5, 0.9, 0.95, 0.99])
    lines = [
        "# Data cleaning report",
        "",
        f"- Raw task events loaded: {n_raw}",
        f"- Tasks seen (unique (job_id, task_index) groups): {n_seen}",
        f"- Tasks kept with a clean lifecycle: {n_kept}",
        "",
        "| rule | rows/tasks before | removed or changed | reason |",
        "|---|---|---|---|",
    ]
    for rule, n, action, reason in _rule_counts:
        lines.append(f"| {rule} | {n} | {action} | {reason} |")
    lines += [
        "",
        "## Known bias",
        "",
        "Keeping only tasks that were submitted, scheduled and finished inside",
        "the loaded 10-shard window favors short tasks (survivorship bias).",
        "We accept this and use the result as the workload model.",
        "",
        "## Final percentiles",
        "",
        "| metric | p50 | p90 | p95 | p99 |",
        "|---|---|---|---|---|",
    ]
    for col in ("cpu_request", "mem_request", "duration"):
        lines.append(
            f"| {col} | {pct.loc[0.5, col]:.4f} | {pct.loc[0.9, col]:.4f} | "
            f"{pct.loc[0.95, col]:.4f} | {pct.loc[0.99, col]:.4f} |")
    lines.append(f"\n(p99.9 of duration before winsorization: {p999:.1f}s)")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
