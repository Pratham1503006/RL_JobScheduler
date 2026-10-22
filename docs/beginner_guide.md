# Beginner's guide — the whole project in plain English

No prior knowledge assumed. Read this once and you can explain the project to
anyone and demo the dashboard.

## The one-line jist

It's a **cloud scheduling simulator**. Think of a restaurant with **10 tables**
(servers). Customers (**jobs**) keep arriving, each needs a table plus some
resources and occupies it for a while. You're the host: for the customer at the
front of the line, do you seat them at a table — and which one — or tell them
to wait? The goal is to **make everyone wait as little as possible while
keeping the tables busy.**

The project tests two kinds of "hosts": 4 AI agents (RL) vs 3 simple
rule-of-thumb policies (heuristics), and asks: who reduces total waiting time
best?

## The pieces, in order

1. **Data** (`data/`) — real logs from Google's datacenter (2011). Raw = 1.83
   million messy events. Scripts clean it down to ~205,713 usable jobs (each job
   = arrival time, how long it runs, how much CPU, how much memory) and build 3
   test scenarios. Analogy: turning a pile of messy receipts into a tidy list of
   customers.
2. **The simulator** (`env/`) — the fake restaurant. `cluster.py` is the 10
   tables and their capacity (each table holds 1.0 CPU + 1.0 memory).
   `scheduling_env.py` is the rules of the game: it shows the AI what's free,
   accepts a decision, advances time, and keeps score.
3. **Policies** (`baselines/`, `agents/`) — the decision-makers. 3 hand-coded
   rules + 4 RL algorithms trained with Stable-Baselines3.
4. **Evaluation** (`agents/evaluate.py`) — one fair scoreboard. Every policy
   plays the same 3 scenarios, and the same metrics are recorded.
5. **Reporting** (`results/`, `charts.py`, `dashboard/`) — tables, 4 charts, and
   the replay dashboard you'll demo.

## The game the AI plays

- **What it sees** (23 numbers): free CPU and memory on each of the 10 servers,
  the size of the job at the front of the line, and how long the line is. All
  numbers are scaled to 0–1.
- **What it can do** (11 choices): put the front job on server 0…9, **or** wait
  (10).
- **What it wants**: the reward is basically **minus the total waiting time** (a
  small penalty for every waiting job, every simulated second, plus a bonus for
  finishing jobs). So the AI learns "maximize reward = minimize waiting."
- **Invalid moves** (trying a full server, or waiting when nothing is running)
  cost a **20-point penalty** and still burn time. This detail ends up being the
  whole story (see the result below).

## How the 4 RL algorithms differ

They all use the same brain size (a `[128,128]` neural net) for fairness, but
learn differently:

- **DQN** — learns the *value* of each action (Q-value = "if I pick this, how
  much reward will I get?"). Picks the highest. Weakness: it tends to
  **overestimate** values (think of an optimist who always thinks things will
  go great).
- **Double DQN** — *identical* to DQN except it uses **two networks**: one to
  *choose* the best next action, the other to *judge* it. This fixes the
  overestimation. It's the one bit of original code here (`double_dqn.py`).
  Result: clearly the best RL agent.
- **PPO** — a *policy-gradient* method. Instead of scoring actions, it directly
  nudges its action probabilities, only changing them a little per step
  ("clipped") so training stays stable. Good generally, but here it stalls a
  lot.
- **A2C** — also policy-gradient (Advantage Actor-Critic): one part picks
  actions (actor), another judges how good they were (critic), and it updates
  every few steps. Faster/looser than PPO.

The 3 heuristics are simple: **Round Robin** (take turns), **First-Fit**
(lowest-numbered free server), **Best-Fit** (leave the least leftover space).

## The honest headline result

**None of the AIs beat the simple heuristics.** Why? Every RL agent makes ~1
dumb move per job (it tries a full server first, then eats the time penalty),
which adds roughly one wait-interval per job. Since packing jobs is easy in this
setup, the only real skill is queue discipline — and that's exactly where the RL
agents burn themselves. The clear *internal* win: **Double DQN crushes plain
DQN** (its value estimates match reality; DQN's don't).

---

## How to explain the dashboard when you hit Play

First start it: `python dashboard/gen_data.py` then `python -m http.server 8000`
and open `http://localhost:8000/dashboard/`.

**What you're looking at:** a *replay* of precomputed simulations (no AI runs
live — nothing can fail at runtime). Layout from top to bottom:

- **Two dropdowns**: **Scenario** (light / heavy / bursty = low / high / spiky
  traffic) and **Policy** (the 7 policies).
- **Play / Pause / Restart** buttons, a **Speed** slider (5–200 events per
  second), and a **time slider** to scrub to any moment.
- **Three big counters**: `t` = simulated seconds elapsed, `queuing jobs` = line
  length right now, `jobs completed`. (Completed flashes green.)
- **10 horizontal bars = the 10 servers, showing CPU used** by each. Blue =
  normal, **red = over 85% full**. Memory is also saved in the trace but only
  CPU is drawn.
- Below that: the **results table** (same as `summary_table.csv`) and the **4
  report charts**.

**What to narrate while it plays:**

1. Pick **heavy + Round Robin**, hit **Play**. Say: *"The rule takes turns; load
   spreads evenly across servers and the queue drains steadily."* Point at the
   bars filling and emptying, and the completed counter climbing.
2. Switch to **heavy + Double DQN**, hit **Play**. Say: *"This is the best RL
   agent, but watch — it stalls."* Pause on a stall: the **queuing-jobs counter
   gets stuck while some server bars sit idle**. That's the invalid-action tax
   (one per job).
3. If you want the extreme, pick **DQN or PPO**: bars mostly idle, queue stuck,
   because those agents learned to stall badly.
4. Scroll down and point at the **table**: heuristics tie within ~0.2%, no RL
   agent wins, Double DQN is the best RL.
5. Scroll to **chart 4 (Q-value overestimation)**: *"Double DQN's estimates
   match reality; DQN's don't — that's the bug Double DQN exists to fix."*

**Tip:** at 25 events/s a full 2,000-job episode takes ~80 seconds, so pause and
scrub rather than replaying whole episodes end to end.
