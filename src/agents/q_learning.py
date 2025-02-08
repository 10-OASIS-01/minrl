import numpy as np
from typing import Dict, List, Tuple, Optional
from ..environment.grid_world import GridWorld, Action
import random
from collections import defaultdict

class QLearningAgent:
    """
    Q-Learning agent implementation for the GridWorld environment.
    Uses tabular Q-learning with ε-greedy exploration strategy.
    """
    
    def __init__(self, 
                 env: GridWorld, 
                 learning_rate: float = 0.1,
                 gamma: float = 0.99,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 min_epsilon: float = 0.01):
        """
        Initialize the Q-Learning agent.
        
        Args:
            env (GridWorld): The GridWorld environment
            learning_rate (float): Learning rate (α)
            gamma (float): Discount factor (γ)
            epsilon (float): Initial exploration rate (ε)
            epsilon_decay (float): Rate at which to decay epsilon
            min_epsilon (float): Minimum exploration rate
        """
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Initialize Q-table
        self.q_table = defaultdict(lambda: np.zeros(env.get_action_space_size()))
        
        # Statistics for evaluation
        self.episode_rewards = []
        self.episode_lengths = []
    
    def select_action(self, state: int) -> Action:
        """
        Select an action using ε-greedy strategy.
        
        Args:
            state (int): Current state
            
        Returns:
            Action: Selected action
        """
        valid_actions = self.env.get_valid_actions(state)
        
        # Explore: random action
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Exploit: best action based on Q-values
        q_values = self.q_table[state]
        valid_q_values = [(action, q_values[action]) for action in valid_actions]
        return max(valid_q_values, key=lambda x: x[1])[0]
    
    def update(self, state: int, action: Action, reward: float, next_state: int):
        """
        Update Q-value using the Q-learning update rule.
        
        Args:
            state (int): Current state
            action (Action): Taken action
            reward (float): Received reward
            next_state (int): Resulting state
        """
        # Get maximum Q-value for next state
        next_q_value = max([self.q_table[next_state][a] for a in self.env.get_valid_actions(next_state)])
        
        # Q-learning update rule
        current_q = self.q_table[state][action]
        new_q = current_q + self.learning_rate * (
            reward + self.gamma * next_q_value - current_q
        )
        self.q_table[state][action] = new_q
    
    def decay_epsilon(self):
        """Decay the exploration rate"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def train(self, n_episodes: int = 1000, max_steps: int = 100) -> Tuple[List[float], List[int]]:
        """
        Train the agent using Q-learning.
        
        Args:
            n_episodes (int): Number of training episodes
            max_steps (int): Maximum steps per episode
            
        Returns:
            Tuple[List[float], List[int]]: Episode rewards and lengths
        """
        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0
            
            for step in range(max_steps):
                # Select and take action
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                # Update Q-table
                self.update(state, action, reward, next_state)
                
                episode_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            # Record statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)
            
            # Decay exploration rate
            self.decay_epsilon()
            
            # Print progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                print(f"Episode {episode + 1}/{n_episodes} - "
                      f"Avg Reward: {avg_reward:.2f} - "
                      f"Avg Length: {avg_length:.2f} - "
                      f"Epsilon: {self.epsilon:.3f}")
        
        return self.episode_rewards, self.episode_lengths
    
    def get_optimal_policy(self) -> Dict[int, List[float]]:
        """
        Extract the optimal policy from the learned Q-values.
        
        Returns:
            Dict[int, List[float]]: The optimal policy
        """
        policy = {}
        
        for state in range(self.env.get_state_space_size()):
            policy[state] = [0.0] * self.env.get_action_space_size()
            
            if state in self.env.terminal_states:
                # Uniform random policy for terminal states
                policy[state] = [1.0 / self.env.get_action_space_size()] * self.env.get_action_space_size()
            else:
                # Greedy policy for non-terminal states
                valid_actions = self.env.get_valid_actions(state)
                best_action = max(valid_actions, key=lambda a: self.q_table[state][a])
                policy[state][best_action] = 1.0
        
        return policy
    
    def print_q_values(self):
        """Print Q-values in a grid format"""
        for i in range(self.env.size):
            for j in range(self.env.size):
                state = self.env._pos_to_state((i, j))
                print(f"\nState ({i},{j}):")
                for action in Action:
                    print(f"{action.name}: {self.q_table[state][action]:.2f}")