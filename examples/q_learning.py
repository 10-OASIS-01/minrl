import numpy as np
from src.environment import GridWorld
from src.agents import QLearningAgent
from src.utils.visualization import Visualizer
from src.agents import PolicyOptimizer  # Added missing import


def run_q_learning_example():
    """
    Demonstrates the use of Q-Learning in a GridWorld environment.
    """
    # Create the same environment as in the Value Iteration example for comparison
    env = GridWorld(size=5)
    env.terminal_states[(0, 4)] = 1.0  # Goal state with positive reward
    env.terminal_states[(2, 2)] = -1.0  # Trap state with negative reward

    # Create Q-Learning agent
    agent = QLearningAgent(
        env,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,  # Start with full exploration
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

    # Visualize training progress using the Visualizer class
    training_fig = Visualizer.plot_training_results(
        rewards=rewards,
        lengths=lengths,
        title='Q-Learning Training Progress'
    )
    training_fig.show()

    # Print Q-values using the built-in method
    print("\nFinal Q-values for each state:")
    agent.print_q_values()

    # Get optimal policy
    optimal_policy = agent.get_optimal_policy()

    # Visualize the learned policy using the Visualizer class
    policy_fig = Visualizer.plot_policy(
        optimal_policy,
        env.size,
        title='Learned Policy from Q-Learning'
    )
    policy_fig.show()

    return agent, optimal_policy


def compare_algorithms():
    """
    Compare Value Iteration and Q-Learning approaches with visualizations.
    """
    env = GridWorld(size=5)
    env.terminal_states[(0, 4)] = 1.0
    env.terminal_states[(2, 2)] = -1.0

    # Get Value Iteration policy
    vi_optimizer = PolicyOptimizer(env)
    vi_policy, vi_values = vi_optimizer.value_iteration()

    # Get Q-Learning policy
    ql_agent = QLearningAgent(env)
    ql_agent.train(n_episodes=1000)
    ql_policy = ql_agent.get_optimal_policy()

    # Visualize both policies side by side
    vi_fig = Visualizer.plot_policy(
        vi_policy,
        env.size,
        title='Value Iteration Policy'
    )
    vi_fig.show()

    ql_fig = Visualizer.plot_policy(
        ql_policy,
        env.size,
        title='Q-Learning Policy'
    )
    ql_fig.show()

    # Visualize value function for Value Iteration
    value_fig = Visualizer.plot_value_function(
        vi_values,
        env.size,
        title='Value Function from Value Iteration'
    )
    value_fig.show()


if __name__ == "__main__":
    # Run Q-Learning example
    agent, q_learning_policy = run_q_learning_example()

    # Compare with Value Iteration
    print("\nComparing Value Iteration and Q-Learning policies:")
    compare_algorithms()