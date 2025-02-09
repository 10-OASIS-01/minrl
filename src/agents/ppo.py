"""
Proximal Policy Optimization (PPO) Module
Created by: 10-OASIS-01
Date: 2025-02-09 04:57:01 UTC

Implements PPO algorithm with clipped objective function and value function estimation.
This implementation maintains MinRL's focus on clarity while providing a complete
PPO solution for the GridWorld environment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict
from collections import deque

from ..environment.grid_world import GridWorld, Action

class PPONetwork(nn.Module):
    """
    Combined actor-critic network for PPO.
    Maps states to both action probabilities (actor) and state values (critic).
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(PPONetwork, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value head (critic)
        self.value = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        return self.policy(features), self.value(features)

class PPOMemory:
    """Memory buffer for PPO training"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state.numpy())  # Convert tensor to numpy array
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def get_batch(self) -> Tuple[torch.Tensor, ...]:
        states = torch.stack([torch.from_numpy(s) for s in self.states])
        actions = torch.tensor(self.actions)
        rewards = torch.tensor(self.rewards)
        values = torch.tensor(self.values)
        log_probs = torch.tensor(self.log_probs)
        dones = torch.tensor(self.dones)
        return states, actions, rewards, values, log_probs, dones

class PPOAgent:
    """
    PPO agent implementation using clipped surrogate objective.
    """
    
    def __init__(self,
                 env: GridWorld,
                 learning_rate: float = 0.0003,
                 gamma: float = 0.99,
                 clip_ratio: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 batch_size: int = 64,
                 n_epochs: int = 10):
        """
        Initialize the PPO agent.
        
        Args:
            env: The environment
            learning_rate: Learning rate for optimization
            gamma: Discount factor
            clip_ratio: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            batch_size: Minibatch size for updates
            n_epochs: Number of epochs to optimize on each update
        """
        self.env = env
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        # Initialize dimensions
        self.state_dim = env.size * env.size
        self.action_dim = env.get_action_space_size()
        
        # Initialize network and optimizer
        self.network = PPONetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Initialize memory
        self.memory = PPOMemory()
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
    
    def state_to_tensor(self, state: int) -> torch.Tensor:
        """Convert state number to one-hot tensor"""
        state_tensor = torch.zeros(self.state_dim)
        state_tensor[state] = 1.0
        return state_tensor

    def select_action(self, state: int) -> Tuple[Action, float, float]:
        """
        Select action using the current policy.
        Returns action, log probability, and value estimate.
        """
        state_tensor = self.state_to_tensor(state)

        with torch.no_grad():
            # Get action probabilities and state value
            probs, value = self.network(state_tensor)

            # Mask invalid actions
            valid_actions = self.env.get_valid_actions(state)
            mask = torch.zeros_like(probs)
            mask[list(valid_actions)] = 1
            masked_probs = probs * mask

            # Normalize probabilities
            masked_probs = masked_probs / masked_probs.sum()

            # Sample action
            m = torch.distributions.Categorical(masked_probs)
            action = m.sample()
            log_prob = m.log_prob(action)

            return Action(action.item()), log_prob.item(), value.item()

    def compute_returns_and_advantages(self,
                                       rewards: torch.Tensor,
                                       values: torch.Tensor,
                                       dones: torch.Tensor,
                                       next_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute returns and advantages for PPO update"""
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        next_return = next_value
        next_advantage = 0

        for t in reversed(range(len(rewards))):
            returns[t] = rewards[t] + self.gamma * next_return * (~dones[t])
            td_error = rewards[t] + self.gamma * next_value * (~dones[t]) - values[t]
            advantages[t] = td_error + self.gamma * next_advantage * (~dones[t])

            next_return = returns[t]
            next_advantage = advantages[t]

        return returns, advantages

    def update(self) -> Tuple[float, float, float]:
        """Update policy and value function using PPO"""
        states, actions, rewards, values, old_log_probs, dones = self.memory.get_batch()
        
        # Get final value for return computation
        with torch.no_grad():
            _, last_value = self.network(self.state_to_tensor(self.env.reset()))
            last_value = last_value.item()
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(
            rewards, values, dones, last_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        for _ in range(self.n_epochs):
            # Get action probabilities and values
            probs, new_values = self.network(states)
            
            # Calculate new log probabilities
            m = torch.distributions.Categorical(probs)
            new_log_probs = m.log_prob(actions)
            
            # Calculate ratio and clipped ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            
            # Calculate losses
            policy_loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages
            ).mean()
            
            value_loss = 0.5 * (returns - new_values.squeeze()).pow(2).mean()
            
            entropy_loss = -m.entropy().mean()
            
            # Combined loss
            loss = (policy_loss + 
                   self.value_coef * value_loss + 
                   self.entropy_coef * entropy_loss)
            
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
        
        # Clear memory
        self.memory.clear()
        
        return (total_policy_loss / self.n_epochs,
                total_value_loss / self.n_epochs,
                total_entropy_loss / self.n_epochs)
    
    def train(self, n_episodes: int = 1000, max_steps: int = 100) -> Tuple[List[float], List[int]]:
        """Train the agent"""
        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0
            
            for step in range(max_steps):
                # Select and take action
                action, log_prob, value = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                # Store transition
                self.memory.add(
                    self.state_to_tensor(state),
                    action,
                    reward,
                    value,
                    log_prob,
                    done
                )
                
                episode_reward += reward
                steps += 1
                state = next_state
                
                # Update if memory is full or episode ends
                if len(self.memory.states) >= self.batch_size or done:
                    policy_loss, value_loss, entropy_loss = self.update()
                    self.policy_losses.append(policy_loss)
                    self.value_losses.append(value_loss)
                    self.entropy_losses.append(entropy_loss)
                
                if done:
                    break
            
            # Record statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)
            
            # Print progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                avg_policy_loss = np.mean(self.policy_losses[-100:])
                avg_value_loss = np.mean(self.value_losses[-100:])
                print(f"Episode {episode + 1}/{n_episodes} - "
                      f"Avg Reward: {avg_reward:.2f} - "
                      f"Avg Length: {avg_length:.2f} - "
                      f"Policy Loss: {avg_policy_loss:.4f} - "
                      f"Value Loss: {avg_value_loss:.4f}")
        
        return self.episode_rewards, self.episode_lengths
    
    def get_optimal_policy(self) -> Dict[int, List[float]]:
        """Extract the optimal policy from the current network"""
        policy = {}
        
        with torch.no_grad():
            for state in range(self.env.get_state_space_size()):
                policy[state] = [0.0] * self.action_dim
                
                if state in self.env.terminal_states:
                    # Uniform random policy for terminal states
                    policy[state] = [1.0 / self.action_dim] * self.action_dim
                else:
                    # Get action probabilities
                    state_tensor = self.state_to_tensor(state)
                    probs, _ = self.network(state_tensor)
                    
                    # Mask invalid actions
                    valid_actions = self.env.get_valid_actions(state)
                    mask = torch.zeros_like(probs)
                    mask[list(valid_actions)] = 1
                    masked_probs = probs * mask
                    
                    # Normalize probabilities
                    masked_probs = masked_probs / masked_probs.sum()
                    
                    policy[state] = masked_probs.tolist()
        
        return policy
