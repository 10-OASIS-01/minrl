"""
Monte Carlo Methods Example
Created by: 10-OASIS-01
Date: 2025-02-09

Demonstrates Monte Carlo policy evaluation in a GridWorld environment.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.environment import GridWorld, Action
from src.agents.monte_carlo import MonteCarloEvaluator
from src.utils.visualization import Visualizer
import matplotlib.pyplot as plt


def create_random_policy(env: GridWorld) -> dict:
    """
    Create a random policy for the environment.

    Args:
        env (GridWorld): The environment

    Returns:
        dict: Mapping from states to action probabilities
    """
    n_actions = env.get_action_space_size()
    n_states = env.get_state_space_size()

    return {
        state: [1.0 / n_actions] * n_actions
        for state in range(n_states)
    }


def create_simple_policy(env: GridWorld) -> dict:
    """
    Create a simple deterministic policy that tends to move towards corners.

    Args:
        env (GridWorld): The environment

    Returns:
        dict: Mapping from states to action probabilities
    """
    policy = {}
    size = env.size

    for i in range(size):
        for j in range(size):
            state = env._pos_to_state((i, j))

            # Default to equal probabilities
            action_probs = [0.0] * env.get_action_space_size()

            if i < size // 2:  # Upper half - tend to go up
                if j < size // 2:  # Upper-left quadrant
                    action_probs[Action.LEFT] = 0.5  # Left
                    action_probs[Action.UP] = 0.5  # Up
                else:  # Upper-right quadrant
                    action_probs[Action.RIGHT] = 0.5  # Right
                    action_probs[Action.UP] = 0.5  # Up
            else:  # Lower half - tend to go down
                if j < size // 2:  # Lower-left quadrant
                    action_probs[Action.LEFT] = 0.5  # Left
                    action_probs[Action.DOWN] = 0.5  # Down
                else:  # Lower-right quadrant
                    action_probs[Action.RIGHT] = 0.5  # Right
                    action_probs[Action.DOWN] = 0.5  # Down

            policy[state] = action_probs

    return policy


def run_monte_carlo_example():
    """Run Monte Carlo policy evaluation example"""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create environment and evaluator
    env = GridWorld(size=9)
    evaluator = MonteCarloEvaluator(env, gamma=0.99)

    # Create policies to evaluate
    random_policy = create_random_policy(env)
    simple_policy = create_simple_policy(env)

    # Create visualizer
    viz = Visualizer()

    # Evaluate policies using both first-visit and every-visit MC
    print("Evaluating random policy...")
    random_first_visit = evaluator.evaluate_policy(
        random_policy,
        num_episodes=1000,
        first_visit=True
    )

    print("Evaluating simple policy...")
    simple_first_visit = evaluator.evaluate_policy(
        simple_policy,
        num_episodes=1000,
        first_visit=True
    )

    # Compare policy values
    value_comparison = {
        "Random Policy": random_first_visit,
        "Simple Policy": simple_first_visit
    }

    # Visualize results
    print("\nPolicy Comparison:")
    print("Random Policy Values:")
    evaluator.state_values = random_first_visit
    evaluator.print_values()

    print("\nSimple Policy Values:")
    evaluator.state_values = simple_first_visit
    evaluator.print_values()

    # Plot value functions
    fig_values = viz.plot_value_comparison(
        value_comparison,
        env.size,
        "Monte Carlo Policy Evaluation Comparison"
    )
    plt.show()

    # Visualize policies
    fig_random = viz.plot_policy(
        random_policy,
        env.size,
        "Random Policy"
    )
    plt.show()

    fig_simple = viz.plot_policy(
        simple_policy,
        env.size,
        "Simple Policy"
    )
    plt.show()

    # Generate and visualize example episodes
    print("\nGenerating example episodes...")

    # Random policy episode
    random_episode = evaluator.generate_episode(random_policy)
    fig_random_episode = viz.visualize_episode_trajectory(
        env,
        random_episode,
        "Random Policy Episode Trajectory"
    )
    plt.show()

    # Simple policy episode
    simple_episode = evaluator.generate_episode(simple_policy)
    fig_simple_episode = viz.visualize_episode_trajectory(
        env,
        simple_episode,
        "Simple Policy Episode Trajectory"
    )
    plt.show()

    # Print episode statistics
    print("\nEpisode Statistics:")
    print(f"Random Policy Episode Length: {len(random_episode)}")
    print(f"Simple Policy Episode Length: {len(simple_episode)}")

    random_total_reward = sum(r for _, _, r in random_episode)
    simple_total_reward = sum(r for _, _, r in simple_episode)

    print(f"Random Policy Total Reward: {random_total_reward:.2f}")
    print(f"Simple Policy Total Reward: {simple_total_reward:.2f}")


if __name__ == "__main__":
    run_monte_carlo_example()