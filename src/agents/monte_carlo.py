"""
Monte Carlo Methods Module
Created by: 10-OASIS-01
Date: 2025-02-09

Implements Monte Carlo methods for policy evaluation and control.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from ..environment.grid_world import GridWorld, Action

class MonteCarloEvaluator:
    """
    A class to evaluate state values and action values using Monte Carlo methods.
    Implements both first-visit and every-visit Monte Carlo policy evaluation.
    """
    
    def __init__(self, env: GridWorld, gamma: float = 0.99):
        """
        Initialize the Monte Carlo evaluator.
        
        Args:
            env (GridWorld): The GridWorld environment
            gamma (float): Discount factor for future rewards
        """
        self.env = env
        self.gamma = gamma
        self.state_values = np.zeros(env.get_state_space_size())
        self.returns = {state: [] for state in range(env.get_state_space_size())}
        
    def generate_episode(self, policy: Dict[int, List[float]]) -> List[Tuple[int, int, float]]:
        """
        Generate an episode using the given policy.
        
        Args:
            policy (Dict[int, List[float]]): Dictionary mapping states to action probabilities
            
        Returns:
            List[Tuple[int, int, float]]: List of (state, action, reward) tuples
        """
        episode = []
        state = self.env.reset()
        done = False
        
        while not done:
            # Choose action based on policy
            action_probs = policy[state]
            action = np.random.choice(len(action_probs), p=action_probs)
            
            # Take action in environment
            next_state, reward, done, _ = self.env.step(action)
            
            # Record state, action, reward
            episode.append((state, action, reward))
            state = next_state
            
        return episode
    
    def evaluate_policy(self, 
                       policy: Dict[int, List[float]], 
                       num_episodes: int = 1000,
                       first_visit: bool = True) -> np.ndarray:
        """
        Evaluate a policy using Monte Carlo policy evaluation.
        
        Args:
            policy (Dict[int, List[float]]): Dictionary mapping states to action probabilities
            num_episodes (int): Number of episodes to generate
            first_visit (bool): If True, use first-visit MC. If False, use every-visit MC
            
        Returns:
            np.ndarray: The computed state values
        """
        # Reset returns for all states
        self.returns = {state: [] for state in range(self.env.get_state_space_size())}
        
        # Generate episodes and compute returns
        for episode in range(num_episodes):
            episode_history = self.generate_episode(policy)
            G = 0  # Initialize return
            
            # Process episode backwards to compute returns
            for t in range(len(episode_history) - 1, -1, -1):
                state, _, reward = episode_history[t]
                G = reward + self.gamma * G
                
                # For first-visit MC, only update if this is the first visit to the state
                if first_visit:
                    # Check if state was visited earlier in the episode
                    if not any(s == state for s, _, _ in episode_history[:t]):
                        self.returns[state].append(G)
                else:
                    # For every-visit MC, update for all visits
                    self.returns[state].append(G)
            
            # Update state values after each episode
            for state in range(self.env.get_state_space_size()):
                if self.returns[state]:  # Only update if we have returns for this state
                    self.state_values[state] = np.mean(self.returns[state])
        
        return self.state_values
    
    def print_values(self):
        """Print the state values in a grid format"""
        for i in range(self.env.size):
            row_values = []
            for j in range(self.env.size):
                state = self.env._pos_to_state((i, j))
                row_values.append(f"{self.state_values[state]:6.2f}")
            print(" ".join(row_values))
