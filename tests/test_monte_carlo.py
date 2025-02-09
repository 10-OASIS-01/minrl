"""
Test Module for Monte Carlo Policy Evaluation
Created by: 10-OASIS-01
Date: 2025-02-09 04:37:11
"""

import numpy as np
from src.environment.grid_world import GridWorld, Action
from src.agents.monte_carlo import MonteCarloEvaluator
from src.agents.policy_evaluation import PolicyEvaluator
from .test_policy_evaluation import create_random_policy

def test_monte_carlo_evaluation():
    """Test Monte Carlo evaluation with a random policy"""
    # Create a 3x3 grid world
    env = GridWorld(size=3)
    
    # Create a random policy
    random_policy = create_random_policy(env)
    
    # Create Monte Carlo evaluator
    mc_evaluator = MonteCarloEvaluator(env, gamma=0.99)
    
    # Evaluate the policy using both first-visit and every-visit MC
    print("Evaluating random policy with first-visit Monte Carlo...")
    first_visit_values = mc_evaluator.evaluate_policy(
        random_policy,
        num_episodes=1000,
        first_visit=True
    )
    
    print("\nFirst-visit MC state values:")
    mc_evaluator.print_values()
    
    print("\nEvaluating random policy with every-visit Monte Carlo...")
    every_visit_values = mc_evaluator.evaluate_policy(
        random_policy,
        num_episodes=1000,
        first_visit=False
    )
    
    print("\nEvery-visit MC state values:")
    mc_evaluator.print_values()
    
    # Verify values for terminal states
    print("\nVerifying terminal state values:")
    for state, reward in env.terminal_states.items():
        print(f"Terminal state {state}:")
        print(f"Expected = {reward}")
        print(f"First-visit MC = {first_visit_values[state]:.2f}")
        print(f"Every-visit MC = {every_visit_values[state]:.2f}")

def test_monte_carlo_vs_dp():
    """Compare Monte Carlo evaluation with dynamic programming"""
    # Create a 3x3 grid world
    env = GridWorld(size=3)
    
    # Create a deterministic policy (always go right if possible, else go down)
    deterministic_policy = {}
    for state in range(env.get_state_space_size()):
        probs = [0.0] * env.get_action_space_size()
        valid_actions = env.get_valid_actions(state)
        
        if Action.RIGHT in valid_actions:
            probs[Action.RIGHT] = 1.0
        elif Action.DOWN in valid_actions:
            probs[Action.DOWN] = 1.0
        elif valid_actions:
            probs[valid_actions[0]] = 1.0
            
        deterministic_policy[state] = probs
    
    # Create both evaluators
    mc_evaluator = MonteCarloEvaluator(env, gamma=0.99)
    dp_evaluator = PolicyEvaluator(env, gamma=0.99)
    
    # Evaluate using Monte Carlo
    print("\nEvaluating deterministic policy with Monte Carlo...")
    mc_values = mc_evaluator.evaluate_policy(
        deterministic_policy,
        num_episodes=5000,  # More episodes for better convergence
        first_visit=True
    )
    
    # Evaluate using Dynamic Programming
    print("\nEvaluating deterministic policy with Dynamic Programming...")
    dp_values = dp_evaluator.evaluate_policy(deterministic_policy, theta=1e-6)
    
    # Compare results
    print("\nMonte Carlo state values:")
    mc_evaluator.print_values()
    
    print("\nDynamic Programming state values:")
    dp_evaluator.print_values()
    
    print("\nAbsolute differences between MC and DP:")
    diff = np.abs(mc_values - dp_values)
    for i in range(env.size):
        row_values = []
        for j in range(env.size):
            state = env._pos_to_state((i, j))
            row_values.append(f"{diff[state]:6.2f}")
        print(" ".join(row_values))

def test_monte_carlo_convergence():
    """Test Monte Carlo convergence with increasing episodes"""
    # Create a 3x3 grid world
    env = GridWorld(size=3)
    random_policy = create_random_policy(env)
    mc_evaluator = MonteCarloEvaluator(env, gamma=0.99)
    
    episode_counts = [10, 100, 1000, 5000]
    previous_values = None
    
    for episodes in episode_counts:
        print(f"\nEvaluating with {episodes} episodes...")
        state_values = mc_evaluator.evaluate_policy(
            random_policy,
            num_episodes=episodes,
            first_visit=True
        )
        
        mc_evaluator.print_values()
        
        if previous_values is not None:
            mean_diff = np.mean(np.abs(state_values - previous_values))
            print(f"Mean absolute difference from previous iteration: {mean_diff:.4f}")
        
        previous_values = state_values.copy()

if __name__ == "__main__":
    print("Testing Monte Carlo evaluation with random policy:")
    test_monte_carlo_evaluation()
    
    print("\nComparing Monte Carlo with Dynamic Programming:")
    test_monte_carlo_vs_dp()
    
    print("\nTesting Monte Carlo convergence:")
    test_monte_carlo_convergence()
