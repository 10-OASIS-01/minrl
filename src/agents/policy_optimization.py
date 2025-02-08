import numpy as np
from typing import Dict, List, Tuple, Optional
from ..environment.grid_world import GridWorld, Action
from .policy_evaluation import PolicyEvaluator

class PolicyOptimizer:
    """
    A class implementing both Value Iteration and Policy Iteration algorithms
    for finding optimal policies in the GridWorld environment.
    """
    
    def __init__(self, env: GridWorld, gamma: float = 0.99):
        """
        Initialize the policy optimizer.
        
        Args:
            env (GridWorld): The GridWorld environment
            gamma (float): Discount factor for future rewards
        """
        self.env = env
        self.gamma = gamma
        self.state_values = np.zeros(env.get_state_space_size())
        self.policy = self._create_random_policy()
        self.policy_evaluator = PolicyEvaluator(env, gamma)
        
    def _create_random_policy(self) -> Dict[int, List[float]]:
        """Create an initial random policy"""
        policy = {}
        n_actions = self.env.get_action_space_size()
        
        for state in range(self.env.get_state_space_size()):
            valid_actions = self.env.get_valid_actions(state)
            probs = [1.0 / len(valid_actions) if action in valid_actions else 0.0 
                    for action in range(n_actions)]
            policy[state] = probs
            
        return policy
    
    def value_iteration(self, theta: float = 1e-6, max_iterations: int = 1000) -> Tuple[Dict[int, List[float]], np.ndarray]:
        """
        Implement the Value Iteration algorithm.
        
        Args:
            theta (float): Convergence threshold
            max_iterations (int): Maximum number of iterations
            
        Returns:
            Tuple[Dict, np.ndarray]: Optimal policy and state values
        """
        # Initialize state values
        self.state_values = np.zeros(self.env.get_state_space_size())
        
        for iteration in range(max_iterations):
            delta = 0
            
            # Update each state
            for state in range(self.env.get_state_space_size()):
                if state in self.env.terminal_states:
                    self.state_values[state] = self.env.terminal_states[state]
                    continue
                    
                old_value = self.state_values[state]
                state_pos = self.env._state_to_pos(state)
                
                # Find maximum value over all actions
                action_values = []
                for action in self.env.get_valid_actions(state):
                    self.env.current_pos = state_pos
                    next_state, reward, done, _ = self.env.step(action)
                    action_values.append(reward + self.gamma * self.state_values[next_state])
                
                self.state_values[state] = max(action_values) if action_values else 0
                delta = max(delta, abs(old_value - self.state_values[state]))
            
            # Check convergence
            if delta < theta:
                print(f"Value iteration converged after {iteration + 1} iterations")
                break
        
        # Extract optimal policy
        optimal_policy = self._extract_policy_from_values()
        return optimal_policy, self.state_values
    
    def policy_iteration(self, theta: float = 1e-6, max_iterations: int = 1000) -> Tuple[Dict[int, List[float]], np.ndarray]:
        """
        Implement the Policy Iteration algorithm.
        
        Args:
            theta (float): Convergence threshold
            max_iterations (int): Maximum number of iterations
            
        Returns:
            Tuple[Dict, np.ndarray]: Optimal policy and state values
        """
        policy_stable = False
        iteration = 0
        
        while not policy_stable and iteration < max_iterations:
            # Policy Evaluation
            self.state_values = self.policy_evaluator.evaluate_policy(self.policy, theta)
            
            # Policy Improvement
            policy_stable = True
            for state in range(self.env.get_state_space_size()):
                if state in self.env.terminal_states:
                    continue
                    
                old_action = np.argmax(self.policy[state])
                
                # Find best action
                state_pos = self.env._state_to_pos(state)
                action_values = []
                valid_actions = self.env.get_valid_actions(state)
                
                for action in valid_actions:
                    self.env.current_pos = state_pos
                    next_state, reward, done, _ = self.env.step(action)
                    action_values.append(reward + self.gamma * self.state_values[next_state])
                
                # Update policy
                best_action = valid_actions[np.argmax(action_values)]
                new_policy = [0.0] * self.env.get_action_space_size()
                new_policy[best_action] = 1.0
                self.policy[state] = new_policy
                
                if best_action != old_action:
                    policy_stable = False
            
            iteration += 1
            if policy_stable:
                print(f"Policy iteration converged after {iteration} iterations")
            elif iteration == max_iterations:
                print("Warning: Policy iteration did not converge within the maximum iterations")
        
        return self.policy, self.state_values
    
    def _extract_policy_from_values(self) -> Dict[int, List[float]]:
        """Extract deterministic policy from state values"""
        policy = {}
        
        for state in range(self.env.get_state_space_size()):
            if state in self.env.terminal_states:
                policy[state] = [1.0 / self.env.get_action_space_size()] * self.env.get_action_space_size()
                continue
                
            state_pos = self.env._state_to_pos(state)
            valid_actions = self.env.get_valid_actions(state)
            action_values = []
            
            for action in valid_actions:
                self.env.current_pos = state_pos
                next_state, reward, done, _ = self.env.step(action)
                action_values.append(reward + self.gamma * self.state_values[next_state])
            
            best_action = valid_actions[np.argmax(action_values)]
            probs = [0.0] * self.env.get_action_space_size()
            probs[best_action] = 1.0
            policy[state] = probs
        
        return policy
    
    def print_policy(self, policy: Dict[int, List[float]]):
        """Print the policy in a grid format using arrows"""
        action_symbols = ['↑', '→', '↓', '←']
        
        for i in range(self.env.size):
            row = []
            for j in range(self.env.size):
                state = self.env._pos_to_state((i, j))
                if state in self.env.terminal_states:
                    row.append('T')
                else:
                    action_idx = np.argmax(policy[state])
                    row.append(action_symbols[action_idx])
            print(' '.join(row))