# Data cleaning report

- Raw task events loaded: 1827856
- Tasks seen (unique (job_id, task_index) groups): 464004
- Tasks kept with a clean lifecycle: 205713

| rule | rows/tasks before | removed or changed | reason |
|---|---|---|---|
| 3 exact duplicate events | 0 | dropped duplicate rows | same (job, task, event_type, timestamp) |
| 1 sentinel timestamps | 144339 | tasks dropped | SUBMIT or FINISH has ts==0 or 2^63-1; arrival/end unknown |
| 2 partial/reconstructed records | 0 | tasks dropped | missing_info non-blank on any event of the task |
| 4 non-clean lifecycle | 79216 | tasks dropped | EVICT/FAIL/KILL/LOST present, or SUBMIT count != 1 (limitation: survivorship bias toward completed tasks) |
| 5 no FINISH in window | 33503 | tasks dropped | task ran past the loaded shards; duration unknowable |
| 5b no SCHEDULE row | 0 | tasks dropped | submitted but never scheduled; runtime start unknown |
| 6 blank CPU or memory request | 0 | tasks dropped | request unknown on the SUBMIT row |
| 7 zero or negative request | 1233 | tasks dropped | value <= 0 is not a schedulable demand |
| 9 non-positive duration | 0 | tasks dropped | FINISH <= SCHEDULE (clock artifacts) |
| 10 sub-second durations | 0 | durations clipped to 1s | short tasks are legitimate; clipping avoids zero-duration jobs |
| 8 request above 1.0 | 0 | values clipped to 1.0 | requests are normalized to the largest machine; >1 is an artifact |
| 11 extreme long durations | 206 | durations winsorized at p99.9=22879.9s | keep heavy tasks (realistic hard cases), cap the value |
| 12 same-timestamp bursts | 0 | kept as is | bursts are real workload signal; stable sort only, no jitter |

## Known bias

Keeping only tasks that were submitted, scheduled and finished inside
the loaded 10-shard window favors short tasks (survivorship bias).
We accept this and use the result as the workload model.

## Final percentiles

| metric | p50 | p90 | p95 | p99 |
|---|---|---|---|---|
| cpu_request | 0.0250 | 0.1000 | 0.1000 | 0.1250 |
| mem_request | 0.0219 | 0.0334 | 0.0477 | 0.0796 |
| duration | 462.1898 | 4566.3936 | 8238.6525 | 11650.5517 |

(p99.9 of duration before winsorization: 22879.9s)
