"""
Monte Carlo Methods Example
Created by: 10-OASIS-01
Date: 2025-02-09 05:17:23 UTC

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
import matplotlib.pyplot as plt
from typing import Dict, List

def create_random_policy(env: GridWorld) -> Dict[int, List[float]]:
    """Create a random policy for the environment"""
    policy = {}
    for state in range(env.get_state_space_size()):
        valid_actions = env.get_valid_actions(state)
        probs = np.zeros(env.get_action_space_size())
        probs[list(valid_actions)] = 1.0 / len(valid_actions)
        policy[state] = probs.tolist()
    return policy

def create_simple_policy(env: GridWorld) -> Dict[int, List[float]]:
    """Create a simple deterministic policy that always moves right or down"""
    policy = {}
    for state in range(env.get_state_space_size()):
        probs = np.zeros(env.get_action_space_size())
        i, j = env._state_to_pos(state)
        
        if state in env.terminal_states:
            probs[:] = 1.0 / env.get_action_space_size()
        elif i < env.size - 1:  # Can move down
            probs[2] = 1.0  # Down action
        else:  # Move right
            probs[1] = 1.0  # Right action
            
        policy[state] = probs.tolist()
    return policy

def compare_evaluation_methods(env: GridWorld, policy: Dict[int, List[float]], num_episodes: int = 1000):
    """Compare different policy evaluation methods"""
    # Initialize evaluators
    mc_evaluator_first = MonteCarloEvaluator(env, gamma=0.99)
    mc_evaluator_every = MonteCarloEvaluator(env, gamma=0.99)
    dp_evaluator = PolicyEvaluator(env, gamma=0.99)
    
    # Evaluate policy using different methods
    print("\nEvaluating policy using different methods...")
    
    mc_values_first = mc_evaluator_first.evaluate_policy(
        policy, num_episodes=num_episodes, first_visit=True)
    print("\nFirst-visit Monte Carlo State Values:")
    mc_evaluator_first.print_values()
    
    mc_values_every = mc_evaluator_every.evaluate_policy(
        policy, num_episodes=num_episodes, first_visit=False)
    print("\nEvery-visit Monte Carlo State Values:")
    mc_evaluator_every.print_values()
    
    dp_values = dp_evaluator.evaluate_policy(policy)
    print("\nDynamic Programming State Values:")
    dp_evaluator.print_values()
    
    # Visualize the results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    Visualizer.plot_state_values(mc_values_first, env.size, 
                               title="First-visit MC", ax=axes[0])
    Visualizer.plot_state_values(mc_values_every, env.size, 
                               title="Every-visit MC", ax=axes[1])
    Visualizer.plot_state_values(dp_values, env.size, 
                               title="Dynamic Programming", ax=axes[2])
    
    plt.tight_layout()
    plt.show()
    
    return mc_values_first, mc_values_every, dp_values

def analyze_convergence(env: GridWorld, policy: Dict[int, List[float]], 
                       max_episodes: int = 5000, step: int = 100):
    """Analyze convergence of Monte Carlo methods"""
    episodes_range = range(step, max_episodes + step, step)
    mc_first_errors = []
    mc_every_errors = []
    
    # Get "true" values using dynamic programming
    dp_evaluator = PolicyEvaluator(env, gamma=0.99)
    true_values = dp_evaluator.evaluate_policy(policy)
    
    print("\nAnalyzing convergence...")
    for num_episodes in episodes_range:
        # First-visit MC
        mc_first = MonteCarloEvaluator(env, gamma=0.99)
        values_first = mc_first.evaluate_policy(policy, num_episodes=num_episodes, first_visit=True)
        mc_first_errors.append(np.mean(np.abs(values_first - true_values)))
        
        # Every-visit MC
        mc_every = MonteCarloEvaluator(env, gamma=0.99)
        values_every = mc_every.evaluate_policy(policy, num_episodes=num_episodes, first_visit=False)
        mc_every_errors.append(np.mean(np.abs(values_every - true_values)))
    
    # Plot convergence
    plt.figure(figsize=(10, 5))
    plt.plot(episodes_range, mc_first_errors, label='First-visit MC')
    plt.plot(episodes_range, mc_every_errors, label='Every-visit MC')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Mean Absolute Error')
    plt.title('Convergence of Monte Carlo Methods')
    plt.legend()
    plt.grid(True)
    plt.show()

def run_monte_carlo_demo():
    """Run a demonstration of Monte Carlo methods"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create environment
    env = GridWorld(size=4)
    
    # Create and evaluate random policy
    print("\nEvaluating random policy...")
    random_policy = create_random_policy(env)
    compare_evaluation_methods(env, random_policy, num_episodes=1000)
    
    # Create and evaluate simple policy
    print("\nEvaluating simple policy...")
    simple_policy = create_simple_policy(env)
    compare_evaluation_methods(env, simple_policy, num_episodes=1000)
    
    # Analyze convergence
    print("\nAnalyzing convergence with simple policy...")
    analyze_convergence(env, simple_policy)

if __name__ == "__main__":
    run_monte_carlo_demo()
