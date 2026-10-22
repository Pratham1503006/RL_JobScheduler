# Phase 5 analysis (auto-generated from `summary_table.csv` / `full_comparison.csv`)

1. **`light`** (rho=0.40): all three heuristics tie at ~3,090 s avg wait (round_robin 3,084 s,
   first_fit 3,090 s, best_fit 3,091 s — spread 0.2%); best RL is **ddqn at 3,829 s (+24% vs
   round_robin)**, so under light load nothing is worth optimizing and the heuristics win.
2. **`heavy`** (rho=0.90): heuristics again near-tie (round_robin 4,190 s, first_fit 4,193 s,
   best_fit 4,192 s) and **round_robin still wins overall**; best RL ddqn 5,131 s is +22% vs
   round_robin. DQN (35,791 s) and PPO (37,847 s) wait ~8.5x longer than round_robin.
3. **`bursty`** (mean rho=0.60): round_robin 46,871 s, first_fit 46,876 s, best_fit 46,874 s
   — heuristics tie within 0.01%; best RL **a2c 65,113 s** (+39% vs round_robin), ddqn
   110,909 s (std 42,627 across seeds — one seed truncated, completion 0.963), while
   dqn/ppo average ~313k s (they spend most of the episode stalling).
4. **No RL agent beats any heuristic on any scenario.** The data-backed reason: every RL
   policy burns ~1,200–2,000 invalid actions per 2,000-job episode (one per job — it tries a
   full server first, then sits through the resulting time advance), while heuristics have
   0. RL effectively learned a greedy first-choice plus a WAIT fallback that adds roughly one
   completion interval of wait per job; at 150k/200k training steps it never learned to
   preemptively pack smaller jobs into fragmented servers the way best-fit does.
5. **DQN vs Double DQN is the one clear RL finding**: DoubleDQN trains to far higher
   episode rewards (seed 0: -32.7k vs -905.0k, ~15-28x better across seeds,
   `rewards_*_seed*.csv`) and its learned Q-values are CONSISTENT with realized returns
   (mean max-Q -1,738 at 150k steps implies -17.4/step at gamma .99; its final episodes
   earned -19 to -58/step; `qvalues_*_seed*.csv`), while vanilla DQN's Q-values (-329)
   imply only -3.3/step yet its final episodes earned -35 to -1,057/step: its estimates
   are detached from reality, the classic bootstrap-overestimation failure mode.

Statistical honesty notes: heuristic rows are single deterministic runs (std 0); RL rows are
mean ± std over 3 saved seeds, except PPO which uses 2 seeds — its seed-2 run was excluded
after two independent training runs both collapsed (see `hyperparameters.json`
`evaluation_notes`). Gap-to-seed-spread rule: the heuristic-vs-RL gaps above far exceed the
1–3% spreads across RL seeds, so the ordering heuristics > RL is not a seed artifact; the
10–25k s differences between dqn and ppo on identical scenarios are within each other's
seed spread and should be treated as a tie.
