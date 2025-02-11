"""
Value Iteration Algorithm Example
Created by: 10-OASIS-01
Demonstrates the use of Value Iteration in a GridWorld environment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.environment import GridWorld
from src.agents import PolicyOptimizer
from src.utils.visualization import Visualizer


def evaluate_policy(env, policy, n_episodes=100):
    """Evaluate the performance of a policy"""
    total_rewards = []

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = np.argmax(policy[state])
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state

        total_rewards.append(episode_reward)

    return np.mean(total_rewards), np.std(total_rewards)


def run_value_iteration_example():
    """Run a demonstration of the Value Iteration algorithm"""
    # Create environment with interesting terminal states
    env = GridWorld(size=9)

    # Set terminal states that don't conflict with starting position (0,0)
    goal_state = env._pos_to_state((4, 6))  # Set goal position
    trap_states = [
        env._pos_to_state((2, 5)),
        env._pos_to_state((2, 4)),
        env._pos_to_state((3, 4)),
        env._pos_to_state((4, 4)),
        env._pos_to_state((5, 4))
    ]

    # Clear default terminal states and set new ones
    env.terminal_states.clear()  # Clear default terminal states
    env.terminal_states[goal_state] = 3.0    # High reward for reaching goal
    for trap_state in trap_states:
        env.terminal_states[trap_state] = -1.0   # Penalty for falling into trap

    # Create policy optimizer
    optimizer = PolicyOptimizer(env, gamma=0.99)

    # Run value iteration
    print("Running Value Iteration...")
    optimal_policy, state_values = optimizer.value_iteration(theta=1e-6, max_iterations=1000)

    # Print the optimal policy
    print("\nOptimal Policy (↑:up, →:right, ↓:down, ←:left):")
    optimizer.print_policy(optimal_policy)

    # Initialize visualizer
    viz = Visualizer()

    # Visualize the policy
    viz.plot_policy(
        optimal_policy,
        env.size,
        title='Optimal Policy from Value Iteration'
    )
    plt.show()

    # Evaluate the optimal policy
    print("\nEvaluating the optimal policy...")
    mean_reward, std_reward = evaluate_policy(env, optimal_policy)
    print(f"Policy Evaluation Results:")
    print(f"Average Reward: {mean_reward:.2f} ± {std_reward:.2f}")

    return optimal_policy, state_values, mean_reward, std_reward


if __name__ == "__main__":
    # Run Value Iteration example
    optimal_policy, state_values, mean_reward, std_reward = run_value_iteration_example()