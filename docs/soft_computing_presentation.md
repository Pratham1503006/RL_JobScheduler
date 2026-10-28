# Soft Computing — Submission Script

Everything you need to present the FIS / GA-FIS work: a spoken script, a live
demo run sheet, a real worked example, a numbers cheat sheet, and a Q&A bank.
Numbers are copied from `results/summary_table.csv`, `results/full_comparison.csv`
and `results/ga_logs/` — nothing here is invented.

- **Audience:** faculty / project review
- **Time:** 12–15 minutes (script ~13 min + demo)
- **Deliverables to mention:** `softcomputing/`, `results/softcomputing/`,
  `results/ga_logs/`, dashboard → **Soft computing** tab

---

## 0. Thesis — say this in one sentence

> "We added a non-learning soft-computing family — a fuzzy rule-based scheduler
> and a genetic algorithm that tunes it — and ran it through the exact same
> environment and evaluation harness as the heuristics and the RL agents, so we
> can answer the question: **is reinforcement learning actually necessary for
> this scheduling problem?**"

That sentence is the spine of the whole presentation. Every detail after it
supports either "what the method is" or "the comparison is fair".

---

## 1. Spoken script

### 1.1 Opening (0:00–1:30)

> "The project compares three families of schedulers on the same cloud-scheduling
> environment: classical heuristics, reinforcement learning, and — the part I am
> presenting today — soft computing.
>
> Soft computing is not machine learning. It is a set of techniques for
> approximate reasoning and search: fuzzy logic, genetic algorithms, neural
> networks, and so on. We use two of them:
>
> 1. a **Fuzzy Inference Scheduler**, or FIS, which encodes scheduling intuition
>    as linguistic rules, and
> 2. a **genetic algorithm**, or GA, that tunes the FIS's membership functions
>    and rule conclusions on training data only.
>
> The reason this matters is fairness. If only RL and heuristics are compared,
> a good result for heuristics can be dismissed as 'the baseline was weak'. A
> third, independently designed family is a stronger control. It also lets us
> show exactly **what the RL agent has to beat** in order to justify its
> complexity."

### 1.2 Why fuzzy logic fits this problem (1:30–3:00)

> "Scheduling a job onto a server is full of soft boundaries. When is a server
> 'full'? When is the queue 'busy'? A hard rule says 'if free CPU is below 0.2,
> do not use the server'. But 0.21 and 0.19 are almost the same situation, and
> the hard rule treats them as opposites.
>
> Fuzzy logic replaces that cliff with graded membership. A server at 0.21 can
> be 0.9 'low load' and 0.1 'medium load' at the same time. The scheduler then
> combines several such judgements through rules, and the result is a smooth,
> interpretable decision.
>
> There is a second reason: interpretability. Every decision can be traced back
> to a small rule table. For a cloud operator, 'why did you put this job there?'
> has an answer that fits on one screen. A neural network cannot offer that."

### 1.3 The FIS, step by step (3:00–6:00)

> "The scheduler makes one decision at a time, for the job at the head of the
> FIFO queue. The steps are:
>
> **Step 1 — candidates.** It only considers servers that can actually fit the
> job. If none can fit, it emits WAIT. This is a structural guarantee: the FIS
> can never make an invalid placement. That is why you will see zero invalid
> actions in the results — not because we filtered them out, but because they
> are impossible by construction.
>
> **Step 2 — three crisp inputs per candidate.**
> - *leftover*: the free CPU and memory left on that server **after** placing
>   the job, averaged. This is the slack.
> - *load*: how utilized the server already is, averaged over CPU and memory.
> - *pressure*: the queue length, normalised. It is the same number for every
>   server; it tells the scheduler whether the cluster is calm or busy.
>
> **Step 3 — fuzzify.** Each number is converted into membership degrees:
> leftover is TIGHT / MEDIUM / LOOSE, load is LOW / MEDIUM / HIGH, pressure is
> CALM / BUSY. We use trapezoids; the hand-designed breakpoints are
> leftover (0.10, 0.60), load (0.20, 0.80), pressure (0.20, 0.60).
>
> **Step 4 — apply 18 rules.** There is one rule for every combination, 3 × 3 × 2.
> A rule's strength is the **minimum** of its three membership degrees. Example:
> 'IF leftover is MEDIUM AND load is MEDIUM AND pressure is CALM, THEN
> suitability is 0.65'.
>
> The rule table is not arbitrary. It encodes a policy:
> - When the queue is **BUSY**, prefer **packing**: a tight leftover on a hot
>   server scores highest, because you want to preserve large free blocks for
>   big jobs that are still waiting.
> - When the queue is **CALM**, prefer **balancing**: a loose leftover on a
>   cold server scores highest, because there is no urgency and you want to
>   spread load.
> - A half-full server that is already busy — MEDIUM leftover and HIGH load —
>   is penalised. That is the fragmentation trap.
>
> **Step 5 — defuzzify.** The final suitability is the strength-weighted average
> of the rule conclusions:
> `score = Σ(strength × z) / Σ(strength)`.
> This is a zero-order Sugeno model: the output of each rule is a constant, so
> the whole system is fast and needs no integration.
>
> **Step 6 — choose the maximum score.** Ties go to the lowest server index.
> That is the entire policy: 6 breakpoints and 18 numbers."

### 1.4 The genetic algorithm (6:00–8:00)

> "Those 24 numbers — 6 breakpoints and 18 conclusions — were hand-designed
> first. The GA asks whether a search can find better ones.
>
> The chromosome is exactly those 24 values. The GA uses a population of 30,
> runs for 30 generations, and uses tournament selection of size 3, BLX-alpha
> crossover with probability 0.8, Gaussian mutation with probability 0.1 per
> gene, and 2 elites. It is run with three independent seeds, 0, 1 and 2.
>
> The important part is the fitness function and the data rule.
> - Fitness is the negative mean average wait over **five fixed 500-job
>   episodes**.
> - Those episodes are drawn from `data/scenarios/train_pool.csv` **only**.
>   The code asserts the path; light, heavy and bursty are never exposed to the
>   GA. That is the anti-leakage guard.
> - The five episodes are built once with a fixed seed and are shared by every
>   chromosome and every GA run, so fitness values are comparable.
> - The hand-designed chromosome is seeded as individual 0, and elitism keeps
>   the best solution, so the GA cannot silently drift away from a valid FIS.
>
> After training, we select the chromosome with the best **training** fitness.
> We do not look at the test scenarios to pick a winner. That is deliberate."

### 1.5 How we kept the comparison fair (8:00–9:00)

> "The FIS and GA-FIS implement the same `policy(obs, env) -> action` interface
> as the heuristics and the RL agents. They run through `agents.evaluate`, the
> same loop, the same scenarios, the same seed, and the same metrics. Their
> traces are saved with the same naming convention.
>
> They are deterministic, so one run per scenario is the full result — there is
> no seed noise to average away. RL rows are mean ± standard deviation over the
> saved seeds.
>
> The GA never sees the test data. The evaluation never re-tunes anything. And
> the soft policies never make an invalid action, so their zero is structural,
> not a post-hoc filter."

### 1.6 Results (9:00–11:00)

> "Here is the headline table, average wait in seconds, lower is better.

| Policy | light | heavy | bursty |
|---|---:|---:|---:|
| Round Robin | **3,084** | **4,190** | 46,871 |
| First-Fit | 3,090 | 4,193 | 46,876 |
| Best-Fit | 3,091 | 4,192 | 46,874 |
| **FIS** | 3,090 | 4,213 | **46,869** |
| **GA-FIS** | 3,091 | 4,206 | 46,877 |
| Double DQN | 3,829 | 5,131 | 110,909 |
| A2C | 4,931 | 6,247 | 65,113 |

> Three observations.
>
> First, the FIS lands in the same ballpark as the best heuristics. It is 0.17%
> slower than Round Robin on light, 0.53% slower on heavy, and 0.005% **faster**
> on bursty — it is literally the fastest of all nine policies on that scenario.
> That is exactly what the addendum predicted: a well-designed fuzzy scheduler
> should match, not beat, a strong greedy baseline on this workload.
>
> Second, GA tuning improved the training fitness from minus 2,729.46 to minus
> 2,727.37, but the held-out difference is within ±0.17% per scenario. On light
> it is 0.04% slower, on heavy it is 0.17% faster, on bursty it is 0.02% slower.
> **The honest conclusion is a tie.** We do not claim a win.
>
> Third, RL loses to both. Double DQN is 22–24% slower than the best
> capacity-aware policy, and the best RL agent on bursty, A2C, is 38.9% slower
> than the FIS. The gap is one to two orders of magnitude larger than the RL
> seed spread, so it is not noise."

### 1.7 Honest interpretation (11:00–12:30)

> "Why didn't the GA transfer?
>
> The FIS picks the server with the highest score, an `argmax`. That means the
> fitness is piecewise constant: small changes to a chromosome usually do not
> change a single placement. Most mutations are invisible to the objective. The
> GA does move the membership functions visibly — you can see that in the
> overlay plot — but the search landscape is nearly flat, so there is very little
> to climb.
>
> We also checked whether seed selection was hiding a lucky win. The three GA
> seeds, evaluated on the held-out scenarios, have mean waits of 18,059.6,
> 18,057.9 and 18,055.5 seconds — a 4.1-second, 0.02% spread. The selected seed
> is the middle one on test. The seed that would have looked best on test is
> seed 2, and we did **not** pick it, because selection uses training fitness
> only. That is the point of the rule.
>
> Why is FIS so close to the heuristics? Because all of them are capacity-aware
> greedy policies. Under light load there is nothing to optimise; under heavy
> load the packing problem is easy. The differences between them are far below
> the RL seed noise. This is a real property of the workload, and we document it
> in `results/baseline_sanity_check.md`.
>
> And why does RL lose? The data-backed reason is invalid actions. Every RL
> policy spends roughly 1,400–2,000 actions per 2,000-job episode on placements
> that fail, because it tries a full server and then pays the time advance. That
> adds about one completion interval of wait per job. The capacity-aware
> policies never make that mistake."

### 1.8 Close (12:30–13:00)

> "So the answer to the question is: on this workload, RL is not necessary.
> A 24-parameter fuzzy scheduler with no training matches the best heuristic,
> and a GA cannot reliably improve it. That is a valid, reportable negative
> result for RL — and the soft-computing family is what makes the claim
> credible, because it is a third, independently designed control.
>
> Everything is reproducible with the commands in the README, and every number
> in this talk comes from a file under `results/`."

---

## 2. Live demo run sheet

### Option A — dashboard (best for a projector)

```powershell
python dashboard/gen_data.py
python -m http.server 8000
# open http://localhost:8000/dashboard/#soft
```

Then, in order:

1. **Soft computing tab → Fuzzy decision inspector.**
   - Keep policy = FIS, scenario = heavy.
   - Press **Play**; let it step.
   - Point at the three stats: time, queue, chosen server and score.
   - Drag the slider to a decision where different servers have different
     scores (the first decision is an all-tie warm-up: every empty server
     scores 1.0 and the tie goes to server 0).
2. **Point at the left panel** — the three membership bars. Say: "this is the
   fuzzification step; the crisp input is at the top, the degrees are the bars."
3. **Point at the right panel** — the candidate scores. Say: "the chosen server
   is starred; the grey bars are the alternatives."
4. **Point at the rule table** — "only the rules that fired are shown; strength
   is the min of the three memberships, and the score is the weighted average."
5. Switch **Policy** to GA-FIS and repeat one decision. Say: "same interface,
   different numbers; the GA only changed the breakpoints and the 18
   consequents."
6. Scroll down to **Rule base**, **GA convergence**, **GA seed robustness**.
   This is where you show that the GA did move the parameters but the held-out
   result is a tie.

> If the browser is unavailable, use `results/dashboard_softcomputing.png`
> and `results/charts/5_ga_convergence.png`,
> `results/charts/6_fuzzy_memberships.png`.

### Option B — terminal (best for a code review / viva)

```powershell
python -m softcomputing.demo --scenario heavy --every 800 --max-print 4
python -m softcomputing.demo --policy ga_fis --scenario heavy --every 800 --max-print 4
```

The output shows the whole chain for each printed decision: crisp inputs →
membership degrees → fired rules with strengths → weighted score → chosen
server, then the episode's final metrics. The final metrics match the CSV
(`avg_wait 4212.9 s` for FIS/heavy, `4205.6 s` for GA-FIS/heavy).

---

## 3. Worked example (real numbers, use it on the whiteboard)

From `results/softcomputing/decisions_fis_heavy.json`, at `t = 12,723.3 s`,
queue length 1 (pressure 0.02):

- Head job: CPU 0.006, memory 0.040 (a small job).
- Chosen server 5: leftover 0.419, load 0.558.

**Fuzzify**

| input | value | memberships |
|---|---:|---|
| leftover | 0.419 | TIGHT 0.00, MEDIUM 0.73, LOOSE 0.27 |
| load | 0.558 | LOW 0.00, MEDIUM 0.81, HIGH 0.19 |
| pressure | 0.02 | CALM 1.00, BUSY 0.00 |

**Rules that fire** (strength = min of the three)

| leftover | load | pressure | z | strength | contribution |
|---|---|---|---:|---:|---:|
| MEDIUM | MEDIUM | CALM | 0.65 | 0.726 | 0.472 |
| LOOSE | MEDIUM | CALM | 0.80 | 0.274 | 0.219 |
| MEDIUM | HIGH | CALM | 0.50 | 0.195 | 0.097 |
| LOOSE | HIGH | CALM | 0.60 | 0.195 | 0.117 |

**Defuzzify**

`score = (0.472 + 0.219 + 0.097 + 0.117) / (0.726 + 0.274 + 0.195 + 0.195)
= 0.905 / 1.390 = 0.652`

**Why this server**

- Server 1 has leftover 0.025 and load 0.952 → it is the fragmentation trap.
  Its only strong rule is TIGHT/HIGH/CALM, z = 0.20 → score 0.200.
- Server 0 scores 0.640, server 9 scores 0.639 — close, but lower.
- Server 5 scores 0.652 and wins.

One sentence for the audience: *"The queue was calm, so the rules preferred the
server with the most room, and the nearly-full server was explicitly penalised
by the TIGHT/HIGH/CALM rule."*

---

## 4. Numbers cheat sheet

| Quantity | Value |
|---|---|
| FIS avg wait — light / heavy / bursty | 3,089.6 / 4,212.9 / **46,869.0** s |
| GA-FIS avg wait — light / heavy / bursty | 3,090.8 / 4,205.6 / 46,877.1 s |
| Round Robin — light / heavy / bursty | 3,084.5 / 4,190.5 / 46,871.4 s |
| FIS vs Round Robin | +0.17% / +0.53% / −0.005% |
| GA-FIS vs FIS | +0.04% / −0.17% / +0.02% |
| Double DQN vs FIS | +24% / +22% / +137% |
| A2C vs FIS (bursty) | +38.9% |
| FIS / GA-FIS invalid actions | 0 / 0 on every scenario |
| FIS / GA-FIS completion rate | 1.0 / 1.0 on every scenario |
| Default FIS training fitness | −2,729.4602 |
| GA seed 0 / 1 / 2 training fitness | −2,727.6631 / **−2,727.3650** / −2,728.6244 |
| Selected GA seed | 1 (best **training** fitness) |
| Held-out mean across scenarios, seeds 0/1/2 | 18,059.56 / 18,057.85 / 18,055.46 s |
| Seed spread | 4.1 s = 0.02% |
| FIS parameters | 6 breakpoints + 18 rule consequents = 24 genes |
| GA settings | pop 30, 30 generations, tournament 3, BLX-α 0.3, mutation 0.1/σ0.05, 2 elites, 3 seeds |
| Training episodes | 5 × 500 jobs from `train_pool.csv`, rho 0.6–1.0 |

---

## 5. Q&A bank

**Q: Is fuzzy logic machine learning?**
No. It is approximate reasoning over hand-designed rules. There is no gradient
descent and no learned weights in the FIS. The GA is a search algorithm, not
learning from gradients; it tunes 24 numbers against a fixed objective.

**Q: Why did you add soft computing at all?**
Two reasons: (1) the addendum requires it, and (2) it is a third, independent
control family. Without it, "RL loses to heuristics" can be dismissed as a weak
baseline. With it, the conclusion is "RL loses to two independently designed
non-learning families".

**Q: If the GA didn't improve the test result, was it worth it?**
Yes, as a controlled negative result. It shows the FIS is already near the
ceiling of what this rule structure can do on this workload. We report the
training gain and the held-out tie separately instead of hiding the tie.

**Q: Did you tune the GA on the test scenarios?**
No. `_assert_train_only` refuses any path except `train_pool.csv`. Selection
uses training fitness only. We even report the seed that would have looked best
on test (seed 2) and show that we did not select it.

**Q: Why is the FIS deterministic and RL not?**
The FIS has no random component: same state, same memberships, same argmax.
RL has random initialisation, exploration and mini-batch sampling, so we report
mean ± std over seeds.

**Q: Why zero invalid actions for the FIS?**
Because only `env.can_fit(k)` servers are candidates, and WAIT is emitted only
when none fit. Invalid placements are impossible by construction, not filtered.

**Q: Why is the FIS worse than Round Robin on light/heavy?**
It is within 0.2–0.5%, which is far below the RL seed noise. Under light load
there is nothing to optimise; under heavy load the packing is easy. These are
ties in practice, and we report them as such.

**Q: Why is FIS the fastest on bursty?**
The pressure input changes the rule weighting: under a busy queue the
TIGHT/HIGH/BUSY rule (z = 1.00) dominates, which packs jobs onto already-hot
servers and preserves large free blocks. That is exactly the behaviour that
helps under bursty load.

**Q: What exactly does the GA change?**
The six membership breakpoints and the eighteen rule consequents. It does not
change the rule structure, the inputs, the environment or the reward.

**Q: Why 18 rules and not more?**
Because the input space is 3 × 3 × 2 and we use one rule per combination. That
keeps the table auditable on one screen and makes the chromosome small enough
for a 30 × 30 GA. A larger rule base is future work.

**Q: How do you know the GA chromosome is a valid FIS?**
`repair()` enforces ordered cut points with a minimum gap; the default
chromosome is seeded as individual 0; and the test suite checks that every
member of the population decodes to a full-coverage membership system.

**Q: What are the limitations?**
Hand-designed inputs and rule base; five fixed training windows; a piecewise
constant fitness landscape; and one deterministic run per soft policy. We list
these in `docs/report.md` §9.

**Q: Is the simulation real or a video?**
It is a real replay of the same seeded episode that produced the CSV, plus a
recorded audit trail from the same policy arithmetic. The dashboard does not run
inference live; `record_decisions.py` recomputes each decision with the same
`FuzzyScheduler` object.

**Q: Can I see the code?**
Yes: `softcomputing/fuzzy.py` (membership + inference), `fis_policy.py`
(policy + explanation), `ga.py` (chromosome/GA), `train_ga.py` (fitness and
data guard), `evaluate_soft.py` (held-out evaluation), `record_decisions.py`
(audit trail), `demo.py` (terminal simulation).

---

## 6. File map

| File | What it does |
|---|---|
| `config.py` (`# soft computing`) | inputs, 18-rule table, cut points, GA settings |
| `softcomputing/fuzzy.py` | trapezoid memberships, Sugeno inference |
| `softcomputing/fis_policy.py` | `FuzzyScheduler` + `explain()` audit trail |
| `softcomputing/ga.py` | 24-gene encoding, repair, crossover, mutation, GA loop |
| `softcomputing/train_ga.py` | training guard, fixed episodes, fitness, finalize |
| `softcomputing/evaluate_soft.py` | held-out evaluation + CSV merge |
| `softcomputing/record_decisions.py` | exports per-decision audit trails |
| `softcomputing/demo.py` | terminal step-by-step simulation |
| `softcomputing/plots.py` | membership and convergence figures |
| `results/ga_logs/` | histories, best chromosomes, seed summary, robustness |
| `results/softcomputing/` | membership plots, convergence, decision JSONs |
| `dashboard/app.js` | Soft computing tab + decision inspector |
| `tests/test_fuzzy.py`, `tests/test_ga.py` | 33 tests total |

---

## 7. Reproduce from scratch

```powershell
# 1) train the GA (three independent seeds, then select by training fitness)
python -m softcomputing.train_ga --seed 0
python -m softcomputing.train_ga --seed 1
python -m softcomputing.train_ga --seed 2
python -m softcomputing.train_ga --finalize

# 2) evaluate on the held-out scenarios (FIS + selected GA-FIS)
python -m softcomputing.evaluate_soft

# 3) record the fuzzy audit trail + run the terminal simulation
python -m softcomputing.record_decisions
python -m softcomputing.demo --scenario heavy --every 800 --max-print 4

# 4) charts + dashboard
python charts.py
python dashboard/gen_data.py
python -m http.server 8000   # open http://localhost:8000/dashboard/#soft

# 5) tests
python -m pytest tests/ -q
```

---

## 8. If the demo breaks

- **Dashboard blank / "Cannot load":** you opened `dashboard/index.html`
  directly. Serve the repo root (`python -m http.server 8000`) and use
  `http://localhost:8000/dashboard/`.
- **No decision data:** run `python -m softcomputing.record_decisions`, then
  `python dashboard/gen_data.py` again.
- **No GA artifacts:** run the three seeds and `--finalize` (step 1 above).
- **Projector/browser failure:** use the saved screenshots
  `results/dashboard_softcomputing.png` and the PNGs in `results/charts/`.
- **Faculty asks for raw numbers:** open `results/summary_table.csv` and
  `results/full_comparison.csv` — they are the only sources used in the talk.

---

## 9. Submission checklist

- [ ] `python -m pytest tests/ -q` → 33 passed
- [ ] `results/summary_table.csv` has 27 rows, no NaN
- [ ] `results/full_comparison.csv` has 9 policies
- [ ] soft rows have `completion_rate=1.0`, `invalid_actions=0`
- [ ] `results/ga_logs/ga_selected.json` says seed 1
- [ ] `results/ga_logs/ga_seed_summary.csv` shows every seed ≥ default
- [ ] `results/softcomputing/decisions_*.json` exist (6 files)
- [ ] dashboard loads both tabs, no console errors
- [ ] README + `docs/report.md` numbers match the CSVs
- [ ] no commit/push unless your supervisor asks for it

---

## 10. What we did **not** do (say this before they ask)

- We did not change the environment, reward weights, scenarios, or any RL model.
- We did not tune on the test scenarios, and we did not pick the GA seed by test
  performance.
- We did not force the GA to win by changing the fitness function.
- We did not filter invalid actions; they are structurally impossible for the
  soft policies.
- We did not claim a statistically meaningful GA improvement: the held-out
  difference is a tie and is reported as a tie.
