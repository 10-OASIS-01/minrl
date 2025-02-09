"""
Monte Carlo Methods Example
Created by: 10-OASIS-01
Date: 2025-02-09 06:17:26 UTC

Demonstrates the usage of Monte Carlo methods for policy evaluation in the GridWorld environment.
Compares first-visit and every-visit Monte Carlo methods with traditional policy evaluation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import GridWorld
from src.agents import MonteCarloEvaluator, PolicyEvaluator
from src.utils.visualization import Visualizer

import numpy as np
from typing import Dict, List, Tuple

def create_test_policies(env: GridWorld) -> Tuple[Dict[int, List[float]], Dict[int, List[float]]]:
    """
    Create test policies for demonstration.
    
    Returns:
        Tuple containing random and simple deterministic policies
    """
    # Create random policy
    random_policy = {}
    for state in range(env.get_state_space_size()):
        valid_actions = env.get_valid_actions(state)
        probs = np.zeros(env.get_action_space_size())
        probs[list(valid_actions)] = 1.0 / len(valid_actions)
        random_policy[state] = probs.tolist()
    
    # Create simple deterministic policy (always move right or down)
    simple_policy = {}
    for state in range(env.get_state_space_size()):
        probs = np.zeros(env.get_action_space_size())
        i, j = env._state_to_pos(state)
        
        if state in env.terminal_states:
            probs[:] = 1.0 / env.get_action_space_size()
        elif i < env.size - 1:  # Can move down
            probs[2] = 1.0  # Down action
        else:  # Move right
            probs[1] = 1.0  # Right action
            
        simple_policy[state] = probs.tolist()
    
    return random_policy, simple_policy

def compare_evaluation_methods(env: GridWorld, 
                             policy: Dict[int, List[float]], 
                             num_episodes: int = 1000) -> Dict[str, np.ndarray]:
    """
    Compare different policy evaluation methods.
    
    Returns:
        Dictionary mapping method names to their computed state values
    """
    # Initialize evaluators
    mc_evaluator_first = MonteCarloEvaluator(env, gamma=0.99)
    mc_evaluator_every = MonteCarloEvaluator(env, gamma=0.99)
    dp_evaluator = PolicyEvaluator(env, gamma=0.99)
    
    # Evaluate policy using different methods
    print("\nEvaluating policy using different methods...")
    
    values_dict = {
        "First-visit MC": mc_evaluator_first.evaluate_policy(
            policy, num_episodes=num_episodes, first_visit=True),
        "Every-visit MC": mc_evaluator_every.evaluate_policy(
            policy, num_episodes=num_episodes, first_visit=False),
        "Dynamic Programming": dp_evaluator.evaluate_policy(policy)
    }
    
    # Visualize the comparison
    comparison_fig = Visualizer.plot_value_comparison(
        values_dict,
        env.size,
        "Comparison of Policy Evaluation Methods"
    )
    comparison_fig.show()
    
    return values_dict

def analyze_convergence(env: GridWorld, 
                       policy: Dict[int, List[float]], 
                       max_episodes: int = 5000, 
                       step: int = 100) -> Tuple[List[int], Dict[str, List[float]]]:
    """
    Analyze convergence of Monte Carlo methods.
    
    Returns:
        Tuple of episodes range and errors dictionary
    """
    episodes_range = range(step, max_episodes + step, step)
    errors_dict = {
        "First-visit MC": [],
        "Every-visit MC": []
    }
    
    # Get "true" values using dynamic programming
    dp_evaluator = PolicyEvaluator(env, gamma=0.99)
    true_values = dp_evaluator.evaluate_policy(policy)
    
    print("\nAnalyzing convergence...")
    for num_episodes in episodes_range:
        # First-visit MC
        mc_first = MonteCarloEvaluator(env, gamma=0.99)
        values_first = mc_first.evaluate_policy(
            policy, num_episodes=num_episodes, first_visit=True)
        errors_dict["First-visit MC"].append(
            np.mean(np.abs(values_first - true_values)))
        
        # Every-visit MC
        mc_every = MonteCarloEvaluator(env, gamma=0.99)
        values_every = mc_every.evaluate_policy(
            policy, num_episodes=num_episodes, first_visit=False)
        errors_dict["Every-visit MC"].append(
            np.mean(np.abs(values_every - true_values)))
    
    # Plot convergence analysis
    convergence_fig = Visualizer.plot_convergence_analysis(
        list(episodes_range),
        errors_dict,
        "Convergence of Monte Carlo Methods"
    )
    convergence_fig.show()
    
    return list(episodes_range), errors_dict

def run_monte_carlo_demo():
    """Run a comprehensive demonstration of Monte Carlo methods"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create environment with interesting terminal states
    env = GridWorld(size=4)
    # Convert coordinate tuples to state indices for terminal states
    goal_state = env._pos_to_state((0, 3))  # Convert goal position to state index
    trap_state = env._pos_to_state((1, 1))  # Convert trap position to state index

    # Set terminal states using the terminal_states dictionary
    env.terminal_states[goal_state] = 1.0   # Goal state
    env.terminal_states[trap_state] = -1.0  # Trap state

    # Create test policies
    random_policy, simple_policy = create_test_policies(env)

    # Compare methods with random policy
    print("\nEvaluating random policy...")
    random_policy_values = compare_evaluation_methods(env, random_policy)

    # Compare methods with simple policy
    print("\nEvaluating simple policy...")
    simple_policy_values = compare_evaluation_methods(env, simple_policy)

    # Analyze convergence with simple policy
    print("\nAnalyzing convergence with simple policy...")
    episodes, errors = analyze_convergence(env, simple_policy)

    return {
        'random_policy_values': random_policy_values,
        'simple_policy_values': simple_policy_values,
        'convergence_analysis': (episodes, errors)
    }

if __name__ == "__main__":
    results = run_monte_carlo_demo()