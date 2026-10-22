"""double_dqn.py -- Double DQN as a DQN subclass (plan 7.2).

Copied SB3 2.3.2 DQN.train(), patching ONLY the target-Q computation:
vanilla DQN picks the next action with the target network (argmax over
q_net_target). Double DQN picks it with the ONLINE network and evaluates it
with the target network, which decouples selection error from evaluation
error and reduces the well-documented Q-value overestimation bias
(van Hasselt et al., 2016).

WHY a subclass and not QRDQN: QR-DQN is a different algorithm (distributional
RL), not Double DQN. SB3 2.3.2 DQN has no 'double_q' switch.
"""

import numpy as np
import torch as th
import torch.nn.functional as F

from stable_baselines3 import DQN


class DoubleDQN(DQN):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # --- Double-DQN patch (the ONLY change vs vanilla DQN) ---
                # 1. SELECT the next action greedily with the ONLINE network.
                next_actions = self.q_net(
                    replay_data.next_observations).argmax(dim=1, keepdim=True)
                # 2. EVALUATE that action with the TARGET network.
                next_q_values = th.gather(
                    self.q_net_target(replay_data.next_observations),
                    1, next_actions)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = (replay_data.rewards
                                   + (1 - replay_data.dones) * self.gamma * next_q_values)

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(
                current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                        self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates,
                           exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
