import numpy as np
from typing import Dict, List, Optional
from ..environment.grid_world import GridWorld, Action

class PolicyEvaluator:
    """
    A class to evaluate state values for a given policy in the GridWorld environment
    using the Bellman expectation equation.
    """
    
    def __init__(self, env: GridWorld, gamma: float = 0.99):
        """
        Initialize the policy evaluator.
        
        Args:
            env (GridWorld): The GridWorld environment
            gamma (float): Discount factor for future rewards
        """
        self.env = env
        self.gamma = gamma
        self.state_values = np.zeros(env.get_state_space_size())
        
    def evaluate_policy(self, 
                       policy: Dict[int, List[float]], 
                       theta: float = 1e-6, 
                       max_iterations: int = 1000) -> np.ndarray:
        """
        Evaluate a given policy using the Bellman expectation equation.
        
        Args:
            policy (Dict[int, List[float]]): Dictionary mapping states to action probabilities
            theta (float): Convergence threshold
            max_iterations (int): Maximum number of iterations
            
        Returns:
            np.ndarray: The computed state values
        """
        # Initialize state values
        self.state_values = np.zeros(self.env.get_state_space_size())
        
        for iteration in range(max_iterations):
            delta = 0  # Track maximum change in value
            
            # Iterate through all states
            for state in range(self.env.get_state_space_size()):
                if state in self.env.terminal_states:
                    self.state_values[state] = self.env.terminal_states[state]
                    continue
                    
                old_value = self.state_values[state]
                new_value = 0
                
                # Calculate expected value for the state based on policy
                state_pos = self.env._state_to_pos(state)
                
                for action in self.env.get_valid_actions(state):
                    action_prob = policy[state][action]
                    if action_prob > 0:
                        # Simulate the action
                        self.env.current_pos = state_pos
                        next_state, reward, done, _ = self.env.step(action)
                        
                        # Calculate value using Bellman expectation equation
                        new_value += action_prob * (reward + self.gamma * self.state_values[next_state])
                
                # Update state value
                self.state_values[state] = new_value
                delta = max(delta, abs(old_value - new_value))
            
            # Check for convergence
            if delta < theta:
                print(f"Policy evaluation converged after {iteration + 1} iterations")
                break
                
            if iteration == max_iterations - 1:
                print("Warning: Policy evaluation did not converge within the maximum iterations")
        
        return self.state_values
    
    def print_values(self):
        """Print the state values in a grid format"""
        for i in range(self.env.size):
            row_values = []
            for j in range(self.env.size):
                state = self.env._pos_to_state((i, j))
                row_values.append(f"{self.state_values[state]:6.2f}")
            print(" ".join(row_values))
