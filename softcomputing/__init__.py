"""Soft-computing schedulers (addendum: FIS + GA-tuned FIS).

These policies are plain callables ``policy(obs, env) -> action`` exactly like
``baselines.heuristics``; the environment and the evaluation harness are
shared, so the three-way comparison (heuristics vs soft computing vs RL) is
fair by construction.
"""
