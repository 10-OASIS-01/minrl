"""
Basic navigation example using static GridWorld
Created by: 10-OASIS-01
"""

import sys
import os

from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from src.environment import GridWorld
from src.agents import DQNAgent
from src.utils.visualization import Visualizer


def run_basic_navigation():
    """Run basic navigation example with static environment"""
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Create environment with interesting terminal states
    env = GridWorld(size=9)

    # Set terminal states that don't conflict with starting position (0,0)
    goal_state = env._pos_to_state((4, 6))  # Bottom-right corner
    trap_states = [
        env._pos_to_state((2, 5)),
        env._pos_to_state((2, 4)),
        env._pos_to_state((3, 4)),
        env._pos_to_state((4, 4)),
        env._pos_to_state((5, 4))
    ]

    # Clear default terminal states and set new ones
    env.terminal_states.clear()  # Clear default terminal states
    env.terminal_states[goal_state] = 3.0
    for trap_state in trap_states:
        env.terminal_states[trap_state] = -1.0  # Trap states with negative reward

    # Create agent
    agent = DQNAgent(
        env,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01
    )

    # Training parameters
    n_episodes = 1000
    max_steps = 100

    # Train agent
    print("Starting training...")
    rewards, lengths = agent.train(n_episodes=n_episodes, max_steps=max_steps)

    # Create visualizer
    viz = Visualizer()

    # Plot training results
    fig_training = viz.plot_training_results(
        rewards=rewards,
        lengths=lengths,
        title="Basic Navigation Training Results"
    )
    plt.show()

    # Extract and visualize learned policy
    optimal_policy = agent.get_optimal_policy()
    fig_policy = viz.plot_policy(
        policy=optimal_policy,
        size=env.size,
        title="Learned Policy"
    )
    plt.show()

    # Extract and visualize value function
    value_function = np.zeros(env.size * env.size)
    with torch.no_grad():
        for state in range(env.size * env.size):
            state_tensor = agent.state_to_tensor(state)
            q_values = agent.q_network(state_tensor)
            value_function[state] = q_values.max().item()

    fig_value = viz.plot_value_function(
        values=value_function,
        size=env.size,
        title="State Value Function"
    )
    plt.show()

    # Test trained agent
    print("\nTesting trained agent...")
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

    print(f"Test episode completed with total reward: {total_reward:.2f}")


if __name__ == "__main__":
    run_basic_navigation()