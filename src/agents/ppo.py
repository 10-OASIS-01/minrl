"""
Proximal Policy Optimization (PPO) Implementation
Created by: 10-OASIS-01
Date: 2025-02-09 08:06:50 UTC

A clean implementation of PPO algorithm with the following features:
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Value function clipping
- Mini-batch training
- Entropy bonus for exploration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import deque

from ..environment.grid_world import GridWorld, Action


class PPOActorNetwork(nn.Module):
    """
    Actor network for PPO that outputs action probabilities.
    Uses tanh activation for better gradient flow.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(PPOActorNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.network(x)
        return torch.distributions.Categorical(logits=logits)


class PPOCriticNetwork(nn.Module):
    """
    Critic network for PPO that estimates state values.
    Uses tanh activation for better gradient flow.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super(PPOCriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PPOMemory:
    """
    Memory buffer for storing trajectories and computing advantages.
    """

    def __init__(self):
        self.states: List[int] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[torch.Tensor] = []
        self.dones: List[bool] = []

    def clear(self) -> None:
        """Clear all stored trajectories."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def compute_advantages(self, last_value: float, gamma: float, gae_lambda: float) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).

        Args:
            last_value: Value estimate for the final state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            Tuple containing advantages and returns tensors
        """
        with torch.no_grad():
            rewards = torch.tensor(self.rewards + [last_value])
            values = torch.tensor(self.values + [last_value])
            dones = torch.tensor(self.dones + [True])

            advantages = []
            gae = 0

            for t in range(len(self.rewards) - 1, -1, -1):
                mask = 1.0 - float(dones[t])
                delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
                gae = delta + gamma * gae_lambda * mask * gae
                advantages.insert(0, gae)

            advantages = torch.tensor(advantages)
            returns = advantages + torch.tensor(self.values)

            return advantages, returns


class PPOAgent:
    """
    PPO agent implementation using separate actor and critic networks.
    Features clipped surrogate objective, GAE, and entropy bonus.
    """

    def __init__(self,
                 env: GridWorld,
                 learning_rate: float = 0.0003,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 critic_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 num_epochs: int = 4,
                 batch_size: int = 32,
                 value_clip_ratio: float = 0.2):
        """
        Initialize the PPO agent.

        Args:
            env: The GridWorld environment
            learning_rate: Learning rate for both actor and critic
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_ratio: PPO clipping parameter
            critic_loss_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            num_epochs: Number of epochs to train on each batch of data
            batch_size: Batch size for training
            value_clip_ratio: Clipping parameter for value function
        """
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.critic_loss_coef = critic_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.value_clip_ratio = value_clip_ratio

        # Initialize networks
        self.state_dim = env.size * env.size
        self.action_dim = env.get_action_space_size()

        self.actor = PPOActorNetwork(self.state_dim, self.action_dim)
        self.critic = PPOCriticNetwork(self.state_dim)

        # Initialize optimizer with shared parameters
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate
        )

        # Initialize memory
        self.memory = PPOMemory()

        # Training statistics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.entropy_losses: List[float] = []

    def state_to_tensor(self, state: int) -> torch.Tensor:
        """Convert state number to one-hot tensor"""
        state_tensor = torch.zeros(self.state_dim)
        state_tensor[state] = 1.0
        return state_tensor

    def select_action(self, state: int) -> Tuple[Action, torch.Tensor, float]:
        """
        Select action using the current policy.

        Args:
            state: Current state number

        Returns:
            Tuple containing:
            - Selected action
            - Log probability of the action
            - Value estimate for the state
        """
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state)

            # Get action distribution from actor
            dist = self.actor(state_tensor)

            # Get value estimate from critic
            value = self.critic(state_tensor).item()

            # Mask invalid actions
            valid_actions = self.env.get_valid_actions(state)
            if not valid_actions:
                return Action(0), torch.tensor(0.0), value

            mask = torch.zeros(self.action_dim)
            mask[list(valid_actions)] = 1

            # Apply mask to distribution
            masked_probs = dist.probs * mask
            masked_sum = masked_probs.sum()

            if masked_sum > 0:
                masked_probs = masked_probs / masked_sum
            else:
                masked_probs = mask / len(valid_actions)

            masked_dist = torch.distributions.Categorical(probs=masked_probs)

            # Sample action
            action = masked_dist.sample()
            log_prob = masked_dist.log_prob(action)

            return Action(action.item()), log_prob, value

    def update(self) -> Tuple[float, float, float]:
        """
        Update policy and value functions using the collected data.

        Returns:
            Tuple containing average policy loss, value loss, and entropy loss
        """
        # Convert collected data to tensors
        states = torch.stack([self.state_to_tensor(s) for s in self.memory.states])
        actions = torch.tensor(self.memory.actions)
        old_log_probs = torch.stack(self.memory.log_probs)

        # Compute advantages and returns
        with torch.no_grad():
            advantages, returns = self.memory.compute_advantages(
                last_value=self.critic(self.state_to_tensor(self.memory.states[-1])).item(),
                gamma=self.gamma,
                gae_lambda=self.gae_lambda
            )

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Get old value predictions for value clipping
            old_values = torch.tensor(self.memory.values)

        # Training loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0

        for _ in range(self.num_epochs):
            # Generate random permutation of indices
            indices = torch.randperm(len(states))

            # Mini-batch training
            for start_idx in range(0, len(states), self.batch_size):
                idx = indices[start_idx:start_idx + self.batch_size]

                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_old_values = old_values[idx]

                # Get current policy distribution and values
                dist = self.actor(batch_states)
                values = self.critic(batch_states).squeeze()

                # Calculate ratio of new and old policies
                new_log_probs = dist.log_prob(batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Calculate surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio,
                                    1 - self.clip_ratio,
                                    1 + self.clip_ratio) * batch_advantages

                # Calculate policy loss
                policy_loss = -torch.min(surr1, surr2).mean()

                # Calculate value loss with clipping
                value_pred_clipped = batch_old_values + torch.clamp(
                    values - batch_old_values,
                    -self.value_clip_ratio,
                    self.value_clip_ratio
                )
                value_losses = (values - batch_returns).pow(2)
                value_losses_clipped = (value_pred_clipped - batch_returns).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                # Calculate entropy loss
                entropy_loss = -dist.entropy().mean()

                # Calculate total loss
                loss = (policy_loss +
                        self.critic_loss_coef * value_loss +
                        self.entropy_coef * entropy_loss)

                # Perform update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                # Accumulate losses
                total_policy_loss += policy_loss.detach().item()
                total_value_loss += value_loss.detach().item()
                total_entropy_loss += entropy_loss.detach().item()

        # Calculate average losses
        n_batches = (len(states) + self.batch_size - 1) // self.batch_size
        total_batches = n_batches * self.num_epochs

        avg_policy_loss = total_policy_loss / total_batches
        avg_value_loss = total_value_loss / total_batches
        avg_entropy_loss = total_entropy_loss / total_batches

        return avg_policy_loss, avg_value_loss, avg_entropy_loss

    def train(self,
              n_episodes: int = 1000,
              max_steps: int = 100,
              update_interval: int = 2048) -> Tuple[List[float], List[int]]:
        """
        Train the agent using PPO.

        Args:
            n_episodes: Number of episodes to train
            max_steps: Maximum steps per episode
            update_interval: Number of steps between updates

        Returns:
            Tuple containing lists of episode rewards and lengths
        """
        steps_since_update = 0

        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0

            for step in range(max_steps):
                # Select action
                action, log_prob, value = self.select_action(state)

                # Take action in environment
                next_state, reward, done, _ = self.env.step(action)

                # Store transition in memory
                self.memory.states.append(state)
                self.memory.actions.append(action)
                self.memory.rewards.append(reward)
                self.memory.values.append(value)
                self.memory.log_probs.append(log_prob)
                self.memory.dones.append(done)

                episode_reward += reward
                steps += 1
                steps_since_update += 1

                # Update if we have collected enough steps
                if steps_since_update >= update_interval:
                    policy_loss, value_loss, entropy_loss = self.update()
                    self.policy_losses.append(policy_loss)
                    self.value_losses.append(value_loss)
                    self.entropy_losses.append(entropy_loss)
                    self.memory.clear()
                    steps_since_update = 0

                if done:
                    break

                state = next_state

            # Record episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)

            # Print progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                print(f"Episode {episode + 1}/{n_episodes} - "
                      f"Avg Reward: {avg_reward:.2f} - "
                      f"Avg Length: {avg_length:.2f}")

        return self.episode_rewards, self.episode_lengths

    def get_optimal_policy(self) -> Dict[int, List[float]]:
        """
        Extract the optimal policy from the actor network.

        Returns:
            Dictionary mapping states to action probability distributions
        """
        policy = {}

        with torch.no_grad():
            for state in range(self.env.get_state_space_size()):
                state_tensor = self.state_to_tensor(state)

                if state in self.env.terminal_states:
                    # Uniform random policy for terminal states
                    policy[state] = [1.0 / self.action_dim] * self.action_dim
                else:
                    # Get action distribution from actor
                    dist = self.actor(state_tensor)

                    # Mask invalid actions
                    valid_actions = self.env.get_valid_actions(state)
                    mask = torch.zeros_like(dist.probs)
                    mask[list(valid_actions)] = 1
                    masked_probs = dist.probs * mask

                    # Normalize probabilities
                    masked_sum = masked_probs.sum()
                    if masked_sum > 0:
                        masked_probs = masked_probs / masked_sum
                    else:
                        # If all probabilities are zero, use uniform distribution over valid actions
                        mask = torch.zeros_like(dist.probs)
                        mask[list(valid_actions)] = 1.0 / len(valid_actions)
                        masked_probs = mask

                    policy[state] = masked_probs.tolist()

        return policy

    def save(self, path: str) -> None:
        """
        Save the agent's networks to disk.

        Args:
            path: Path to save the model
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropy_losses': self.entropy_losses,
        }, path)

    def load(self, path: str) -> None:
        """
        Load the agent's networks from disk.

        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.policy_losses = checkpoint['policy_losses']
        self.value_losses = checkpoint['value_losses']
        self.entropy_losses = checkpoint['entropy_losses']

    def evaluate(self, n_episodes: int = 10) -> Tuple[float, float]:
        """
        Evaluate the agent's performance without training.

        Args:
            n_episodes: Number of episodes to evaluate

        Returns:
            Tuple containing average reward and average episode length
        """
        eval_rewards = []
        eval_lengths = []

        for _ in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0
            done = False

            while not done and steps < 1000:  # Add step limit to prevent infinite loops
                with torch.no_grad():
                    action, _, _ = self.select_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    episode_reward += reward
                    steps += 1
                    state = next_state

            eval_rewards.append(episode_reward)
            eval_lengths.append(steps)

        avg_reward = np.mean(eval_rewards)
        avg_length = np.mean(eval_lengths)

        return avg_reward, avg_length