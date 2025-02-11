"""
Monte Carlo Policy Evaluation Example
Created by: 10-OASIS-01
Demonstrates the usage of Monte Carlo methods for policy evaluation in the GridWorld environment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import GridWorld, Action
from src.agents.monte_carlo import MonteCarloEvaluator
from src.utils.visualization import Visualizer

import numpy as np
import torch

def run_monte_carlo_demo():
    """Run a demonstration of Monte Carlo policy evaluation"""
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Create environment with interesting terminal states
    env = GridWorld(size=5)

    # Set terminal states with different rewards
    goal_state = env._pos_to_state((4, 4))  # Bottom-right corner
    trap_state = env._pos_to_state((2, 2))  # Center of the grid

    env.terminal_states.clear()  # Clear default terminal states
    env.terminal_states[goal_state] = 5.0    # High reward for reaching goal
    env.terminal_states[trap_state] = -2.0   # Penalty for falling into trap

    # Create Monte Carlo evaluator
    mc_evaluator = MonteCarloEvaluator(env, gamma=0.9)

    # Create both random and simple policies using the evaluator methods
    random_policy = mc_evaluator.create_random_policy()
    simple_policy = mc_evaluator.create_simple_policy()

    # Parameters for evaluation
    num_episodes = 1000

    print("Evaluating random policy...")
    random_values = mc_evaluator.evaluate_policy(
        random_policy,
        num_episodes=num_episodes,
        first_visit=True
    )

    print("\nRandom Policy State Values:")
    mc_evaluator.print_values()

    # Reset evaluator for the simple policy
    mc_evaluator = MonteCarloEvaluator(env, gamma=0.9)

    print("\nEvaluating simple policy...")
    simple_values = mc_evaluator.evaluate_policy(
        simple_policy,
        num_episodes=num_episodes,
        first_visit=True
    )

    print("\nSimple Policy State Values:")
    mc_evaluator.print_values()

    # Create visualizer
    viz = Visualizer()

    # Plot value functions
    fig_random = viz.plot_value_function(
        values=random_values,
        size=env.size,
        title="Random Policy Value Function"
    )
    fig_random.show()

    fig_simple = viz.plot_value_function(
        values=simple_values,
        size=env.size,
        title="Simple Policy Value Function"
    )
    fig_simple.show()

    # Generate and visualize episodes
    print("\nGenerating sample episode with simple policy...")
    episode = mc_evaluator.generate_episode(simple_policy)

    # Extract episode data for visualization
    episode_data = [(state, action, reward) for state, action, reward in episode]

    fig_trajectory = viz.visualize_episode_trajectory(
        env,
        episode_data,
        "Sample Episode Trajectory"
    )
    fig_trajectory.show()

    # Print episode statistics
    total_reward = sum(reward for _, _, reward in episode)
    print(f"Episode length: {len(episode)}")
    print(f"Total reward: {total_reward:.2f}")

    return mc_evaluator, random_values, simple_values

if __name__ == "__main__":
    evaluator, random_values, simple_values = run_monte_carlo_demo()