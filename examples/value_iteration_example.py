"""
Value Iteration Algorithm Example
Created by: 10-OASIS-01
Demonstrates the use of Value Iteration in a GridWorld environment.
"""


import numpy as np
from src.environment import GridWorld
from src.agents import PolicyOptimizer
import matplotlib.pyplot as plt


def run_value_iteration_example():
    # Create a 5x5 GridWorld environment
    env = GridWorld(size=9)

    # Set terminal states with different rewards by modifying terminal_states dictionary directly
    env.terminal_states[(0, 4)] = 1.0  # Goal state with positive reward
    env.terminal_states[(2, 2)] = -1.0  # Trap state with negative reward

    # Create policy optimizer
    optimizer = PolicyOptimizer(env, gamma=0.99)

    # Run value iteration
    optimal_policy, state_values = optimizer.value_iteration(theta=1e-6, max_iterations=1000)

    # Print the optimal policy
    print("\nOptimal Policy (↑:up, →:right, ↓:down, ←:left):")
    optimizer.print_policy(optimal_policy)

    # Visualize state values
    plt.figure(figsize=(8, 6))
    state_values_grid = state_values.reshape((env.size, env.size))
    plt.imshow(state_values_grid, cmap='RdYlGn')
    plt.colorbar(label='State Value')
    plt.title('State Values after Value Iteration')

    # Add value annotations
    for i in range(env.size):
        for j in range(env.size):
            plt.text(j, i, f'{state_values_grid[i, j]:.2f}',
                     ha='center', va='center')

    plt.show()

    return optimal_policy, state_values


def evaluate_policy(env, policy, n_episodes=100):
    total_rewards = []

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Choose action based on the policy
            action = np.argmax(policy[state])
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state

        total_rewards.append(episode_reward)

    return np.mean(total_rewards), np.std(total_rewards)


if __name__ == "__main__":
    # Run the example
    optimal_policy, state_values = run_value_iteration_example()

    # Create environment for evaluation
    env = GridWorld(size=5)
    env.terminal_states[(0, 4)] = 1.0
    env.terminal_states[(2, 2)] = -1.0

    # Evaluate the optimal policy
    mean_reward, std_reward = evaluate_policy(env, optimal_policy)
    print(f"\nPolicy Evaluation Results:")
    print(f"Average Reward: {mean_reward:.2f} ± {std_reward:.2f}")