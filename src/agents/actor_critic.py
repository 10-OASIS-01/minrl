import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict
from collections import deque

from ..environment.grid_world import GridWorld, Action

class ActorNetwork(nn.Module):
    """
    Actor network that maps states to action probabilities.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(ActorNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class CriticNetwork(nn.Module):
    """
    Critic network that estimates state values.
    """
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ActorCriticAgent:
    """
    Actor-Critic agent implementation using separate actor and critic networks.
    """
    
    def __init__(self,
                 env: GridWorld,
                 actor_lr: float = 0.001,
                 critic_lr: float = 0.001,
                 gamma: float = 0.99):
        """
        Initialize the Actor-Critic agent.
        
        Args:
            env (GridWorld): The environment
            actor_lr (float): Learning rate for the actor
            critic_lr (float): Learning rate for the critic
            gamma (float): Discount factor
        """
        self.env = env
        self.gamma = gamma
        
        # State and action dimensions
        self.state_dim = env.size * env.size
        self.action_dim = env.get_action_space_size()
        
        # Initialize networks
        self.actor = ActorNetwork(self.state_dim, self.action_dim)
        self.critic = CriticNetwork(self.state_dim)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
    
    def state_to_tensor(self, state: int) -> torch.Tensor:
        """Convert state number to one-hot tensor"""
        state_tensor = torch.zeros(self.state_dim)
        state_tensor[state] = 1.0
        return state_tensor
    
    def select_action(self, state: int) -> Tuple[Action, torch.Tensor]:
        """
        Select action using the actor network.
        Returns both the action and the log probability.
        """
        state_tensor = self.state_to_tensor(state)
        
        # Get action probabilities from actor
        probs = self.actor(state_tensor)
        
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
        
        return Action(action.item()), log_prob
    
    def update(self, state: int, action: Action, reward: float, next_state: int, done: bool) -> Tuple[float, float]:
        """
        Update both actor and critic networks.
        Returns the actor and critic losses.
        """
        # Convert states to tensors
        state_tensor = self.state_to_tensor(state)
        next_state_tensor = self.state_to_tensor(next_state)
        
        # Get value estimates
        value = self.critic(state_tensor)
        next_value = self.critic(next_state_tensor)
        
        # Compute TD error
        if done:
            td_target = torch.tensor([reward], dtype=torch.float32)
        else:
            td_target = torch.tensor([reward + self.gamma * next_value.item()], dtype=torch.float32)
        td_error = td_target - value
        
        # Update critic
        critic_loss = td_error.pow(2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Get action probabilities and log probability of taken action
        probs = self.actor(state_tensor)
        m = torch.distributions.Categorical(probs)
        log_prob = m.log_prob(torch.tensor(action))
        
        # Update actor using policy gradient
        actor_loss = -log_prob * td_error.detach()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()
    
    def train(self, n_episodes: int = 1000, max_steps: int = 100) -> Tuple[List[float], List[int]]:
        """Train the agent"""
        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_actor_loss = 0
            episode_critic_loss = 0
            steps = 0
            
            for step in range(max_steps):
                # Select and take action
                action, _ = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                # Update networks
                actor_loss, critic_loss = self.update(state, action, reward, next_state, done)
                
                episode_actor_loss += actor_loss
                episode_critic_loss += critic_loss
                episode_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            # Record statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)
            self.actor_losses.append(episode_actor_loss / steps)
            self.critic_losses.append(episode_critic_loss / steps)
            
            # Print progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                avg_actor_loss = np.mean(self.actor_losses[-100:])
                avg_critic_loss = np.mean(self.critic_losses[-100:])
                print(f"Episode {episode + 1}/{n_episodes} - "
                      f"Avg Reward: {avg_reward:.2f} - "
                      f"Avg Length: {avg_length:.2f} - "
                      f"Avg Actor Loss: {avg_actor_loss:.4f} - "
                      f"Avg Critic Loss: {avg_critic_loss:.4f}")
        
        return self.episode_rewards, self.episode_lengths
    
    def get_optimal_policy(self) -> Dict[int, List[float]]:
        """Extract the optimal policy from the actor network"""
        policy = {}
        
        with torch.no_grad():
            for state in range(self.env.get_state_space_size()):
                policy[state] = [0.0] * self.action_dim
                
                if state in self.env.terminal_states:
                    # Uniform random policy for terminal states
                    policy[state] = [1.0 / self.action_dim] * self.action_dim
                else:
                    # Get action probabilities from actor
                    state_tensor = self.state_to_tensor(state)
                    probs = self.actor(state_tensor)
                    
                    # Mask invalid actions
                    valid_actions = self.env.get_valid_actions(state)
                    mask = torch.zeros_like(probs)
                    mask[list(valid_actions)] = 1
                    masked_probs = probs * mask
                    
                    # Normalize probabilities
                    masked_probs = masked_probs / masked_probs.sum()
                    
                    policy[state] = masked_probs.tolist()
        
        return policy
