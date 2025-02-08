import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from typing import List, Tuple, Dict
from ..environment.grid_world import GridWorld, Action

# Define experience tuple structure
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNetwork(nn.Module):
    """
    Neural network for Deep Q-Learning.
    Maps state representations to Q-values for each action.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        """
        Initialize the Q-Network.
        
        Args:
            state_dim (int): Dimension of state input
            action_dim (int): Number of possible actions
            hidden_dim (int): Size of hidden layers
        """
        super(DQNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.network(x)

class ReplayBuffer:
    """Experience replay buffer to store and sample transitions"""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity (int): Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience: Experience):
        """Add an experience to the buffer"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)

class DQNAgent:
    """
    Deep Q-Learning agent implementation with experience replay and target network.
    """
    
    def __init__(self,
                 env: GridWorld,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 min_epsilon: float = 0.01,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 target_update_freq: int = 100):
        """
        Initialize the DQN agent.
        
        Args:
            env (GridWorld): The environment
            learning_rate (float): Learning rate for the optimizer
            gamma (float): Discount factor
            epsilon (float): Initial exploration rate
            epsilon_decay (float): Rate at which to decay epsilon
            min_epsilon (float): Minimum exploration rate
            buffer_size (int): Size of replay buffer
            batch_size (int): Size of training batches
            target_update_freq (int): Frequency of target network updates
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # State dimension (one-hot encoded grid position)
        self.state_dim = env.size * env.size
        self.action_dim = env.get_action_space_size()
        
        # Initialize networks
        self.q_network = DQNetwork(self.state_dim, self.action_dim)
        self.target_network = DQNetwork(self.state_dim, self.action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
    
    def state_to_tensor(self, state: int) -> torch.Tensor:
        """Convert state number to one-hot tensor"""
        state_tensor = torch.zeros(self.state_dim)
        state_tensor[state] = 1.0
        return state_tensor
    
    def select_action(self, state: int) -> Action:
        """Select action using ε-greedy policy"""
        if random.random() < self.epsilon:
            return random.choice(self.env.get_valid_actions(state))
        
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state)
            q_values = self.q_network(state_tensor)
            
            # Mask invalid actions with large negative values
            valid_actions = self.env.get_valid_actions(state)
            mask = torch.ones_like(q_values) * float('-inf')
            mask[list(valid_actions)] = 0
            
            q_values = q_values + mask
            return Action(q_values.argmax().item())
    
    def train_step(self) -> float:
        """Perform one training step using a batch of experiences"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
            
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Prepare batch tensors
        states = torch.stack([self.state_to_tensor(exp.state) for exp in batch])
        actions = torch.tensor([exp.action for exp in batch])
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32)
        next_states = torch.stack([self.state_to_tensor(exp.next_state) for exp in batch])
        dones = torch.tensor([exp.done for exp in batch], dtype=torch.float32)
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def train(self, n_episodes: int = 1000, max_steps: int = 100) -> Tuple[List[float], List[int]]:
        """Train the agent"""
        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            steps = 0

            for step in range(max_steps):
                # Select and take action
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                # Store experience
                experience = Experience(state, action, reward, next_state, done)
                self.replay_buffer.add(experience)

                # Train
                loss = self.train_step()
                episode_loss += loss

                # Update target network
                if step % self.target_update_freq == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())

                episode_reward += reward
                steps += 1
                state = next_state

                if done:
                    break

            # Record statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)
            if episode_loss > 0:
                self.losses.append(episode_loss / steps)

            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            # Print progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                avg_loss = np.mean(self.losses[-100:]) if self.losses else 0
                print(f"Episode {episode + 1}/{n_episodes} - "
                      f"Avg Reward: {avg_reward:.2f} - "
                      f"Avg Length: {avg_length:.2f} - "
                      f"Avg Loss: {avg_loss:.4f} - "
                      f"Epsilon: {self.epsilon:.3f}")

        return self.episode_rewards, self.episode_lengths
    
    def get_optimal_policy(self) -> Dict[int, List[float]]:
        """Extract the optimal policy from the learned Q-network"""
        policy = {}
        
        with torch.no_grad():
            for state in range(self.env.get_state_space_size()):
                policy[state] = [0.0] * self.action_dim
                
                if state in self.env.terminal_states:
                    # Uniform random policy for terminal states
                    policy[state] = [1.0 / self.action_dim] * self.action_dim
                else:
                    # Get Q-values from network
                    state_tensor = self.state_to_tensor(state)
                    q_values = self.q_network(state_tensor)
                    
                    # Mask invalid actions
                    valid_actions = self.env.get_valid_actions(state)
                    mask = torch.ones_like(q_values) * float('-inf')
                    mask[list(valid_actions)] = 0
                    q_values = q_values + mask
                    
                    # Get best action
                    best_action = q_values.argmax().item()
                    policy[state][best_action] = 1.0
        
        return policy