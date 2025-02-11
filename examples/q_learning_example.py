"""
Q-Learning Algorithm Example
Created by: 10-OASIS-01
Demonstrates the use of Q-Learning in a GridWorld environment.
"""

import numpy as np
from src.environment import GridWorld
from src.agents import QLearningAgent
from src.utils.visualization import Visualizer
from src.agents import PolicyOptimizer  # Added missing import
import matplotlib.pyplot as plt


def run_q_learning_example():
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

    # Create Q-Learning agent with optimized parameters
    agent = QLearningAgent(
        env,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01
    )

    # Train the agent
    print("Training Q-Learning Agent...")
    n_episodes = 1000
    rewards, lengths = agent.train(
        n_episodes=n_episodes,
        max_steps=100
    )

    # Visualize results
    viz = Visualizer()

    # Plot training progress
    viz.plot_training_results(
        rewards=rewards,
        lengths=lengths,
        title='Q-Learning Training Progress'
    )
    plt.show()

    # Print final Q-values
    print("\nFinal Q-values for each state:")
    agent.print_q_values()

    # Get and visualize optimal policy
    optimal_policy = agent.get_optimal_policy()
    viz.plot_policy(
        optimal_policy,
        env.size,
        title='Learned Policy from Q-Learning'
    )
    plt.show()

    return agent, optimal_policy


def compare_algorithms():
    """Compare Value Iteration and Q-Learning approaches."""
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

    # Get Value Iteration policy
    vi_optimizer = PolicyOptimizer(env)
    vi_policy, vi_values = vi_optimizer.value_iteration()

    # Get Q-Learning policy
    ql_agent = QLearningAgent(env)
    ql_agent.train(n_episodes=1000)
    ql_policy = ql_agent.get_optimal_policy()

    # Visualize results
    viz = Visualizer()

    viz.plot_policy(vi_policy, env.size, 'Value Iteration Policy')
    plt.show()

    viz.plot_policy(ql_policy, env.size, 'Q-Learning Policy')
    plt.show()

    viz.plot_value_function(vi_values, env.size, 'Value Function from Value Iteration')
    plt.show()


if __name__ == "__main__":
    # Run Q-Learning example
    agent, q_learning_policy = run_q_learning_example()

    # Compare with Value Iteration
    print("\nComparing Value Iteration and Q-Learning policies:")
    compare_algorithms()