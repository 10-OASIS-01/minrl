import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Optional

from ..environment.grid_world import GridWorld, Action


class ActorNetwork(nn.Module):
    """
    Actor network that outputs action probabilities.
    Uses the same architecture as PPO for consistency.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(ActorNetwork, self).__init__()
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


class CriticNetwork(nn.Module):
    """
    Critic network that estimates state values.
    Uses the same architecture as PPO for consistency.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ActorCriticAgent:
    """
    Actor-Critic agent implementation using separate actor and critic networks.
    Features TD(0) learning and policy gradient updates.
    """

    def __init__(self,
                 env: GridWorld,
                 learning_rate_actor: float = 0.0003,
                 learning_rate_critic: float = 0.001,
                 gamma: float = 0.99,
                 entropy_coef: float = 0.01):
        """
        Initialize the Actor-Critic agent.

        Args:
            env: The GridWorld environment
            learning_rate_actor: Learning rate for the actor
            learning_rate_critic: Learning rate for the critic
            gamma: Discount factor
            entropy_coef: Entropy bonus coefficient
        """
        self.env = env
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        # Initialize networks
        self.state_dim = env.size * env.size
        self.action_dim = env.get_action_space_size()

        self.actor = ActorNetwork(self.state_dim, self.action_dim)
        self.critic = CriticNetwork(self.state_dim)

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate_critic)

        # Training statistics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.actor_losses: List[float] = []
        self.critic_losses: List[float] = []
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

    def update(self, state: int, action: Action, reward: float, next_state: int, done: bool) -> Tuple[float, float, float]:
        """
        Update policy and value functions using TD(0) learning.

        Args:
            state: Current state
            action: Taken action
            reward: Received reward
            next_state: Next state
            done: Whether the episode is finished

        Returns:
            Tuple containing actor loss, critic loss, and entropy loss
        """
        # Convert states to tensors
        state_tensor = self.state_to_tensor(state)
        next_state_tensor = self.state_to_tensor(next_state)

        # Get value predictions
        value = self.critic(state_tensor)
        next_value = self.critic(next_state_tensor)

        # Compute TD target and error
        td_target = reward + (1 - int(done)) * self.gamma * next_value.detach()
        td_error = td_target - value

        # Update critic
        critic_loss = td_error.pow(2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Get action distribution and log probability
        dist = self.actor(state_tensor)
        log_prob = dist.log_prob(torch.tensor(action))

        # Compute actor loss
        actor_loss = -log_prob * td_error.detach()
        entropy_loss = -dist.entropy()
        total_actor_loss = actor_loss + self.entropy_coef * entropy_loss

        # Update actor
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        self.actor_optimizer.step()

        return (actor_loss.item(),
                critic_loss.item(),
                entropy_loss.item())

    def train(self, n_episodes: int = 1000, max_steps: int = 100) -> Tuple[List[float], List[int]]:
        """
        Train the agent using Actor-Critic.

        Args:
            n_episodes: Number of episodes to train
            max_steps: Maximum steps per episode

        Returns:
            Tuple containing lists of episode rewards and lengths
        """
        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0

            actor_loss_sum = 0
            critic_loss_sum = 0
            entropy_loss_sum = 0

            for step in range(max_steps):
                # Select action
                action, log_prob, value = self.select_action(state)

                # Take action in environment
                next_state, reward, done, _ = self.env.step(action)

                # Update networks
                actor_loss, critic_loss, entropy_loss = self.update(
                    state, action, reward, next_state, done
                )

                actor_loss_sum += actor_loss
                critic_loss_sum += critic_loss
                entropy_loss_sum += entropy_loss

                episode_reward += reward
                steps += 1

                if done:
                    break

                state = next_state

            # Record episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)
            self.actor_losses.append(actor_loss_sum / steps)
            self.critic_losses.append(critic_loss_sum / steps)
            self.entropy_losses.append(entropy_loss_sum / steps)

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
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
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
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.actor_losses = checkpoint['actor_losses']
        self.critic_losses = checkpoint['critic_losses']
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