"""
MCTS Example
Created by: 10-OASIS-01
Date: 2025-02-09

Demonstrates Monte Carlo Tree Search in the GridWorld environment.
"""

import sys
import os
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.environment import GridWorld, Action
from src.agents.mcts import MCTSAgent
from src.utils.visualization import Visualizer
import matplotlib.pyplot as plt


def convert_policy_for_visualization(policy: Dict[Action, float]) -> List[float]:
    """Convert policy dict to list format for visualization"""
    return [policy[action] for action in Action]


def run_mcts_example():
    """Run Monte Carlo Tree Search example"""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create environment and agent
    env = GridWorld(size=5)
    agent = MCTSAgent(
        env,
        num_simulations=100,
        exploration_constant=1.41,
        max_rollout_steps=50
    )

    # Create visualizer
    viz = Visualizer()

    # Run episodes and collect statistics
    num_episodes = 100
    episode_rewards = []
    episode_lengths = []
    policies = {}

    print(f"Running {num_episodes} episodes...")

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            # Get action from MCTS
            action = agent.select_action(state)

            # Store policy for visualization
            if episode == num_episodes - 1:  # Store last episode's policies
                policy = agent.get_policy(state)
                policies[state] = convert_policy_for_visualization(policy)

            # Take action in environment
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1
            state = next_state

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"Average Reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
            print(f"Average Length (last 10): {np.mean(episode_lengths[-10:]):.2f}")
            print()

    # Plot results
    plt.figure(figsize=(12, 4))

    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    # Plot episode lengths
    plt.subplot(1, 2, 2)
    plt.plot(episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')

    plt.tight_layout()
    plt.show()

    # Visualize final policy
    policy_grid = viz.plot_policy(policies, env.size, "MCTS Final Policy")
    plt.show()

    # Run and visualize one final episode
    print("\nRunning final demonstration episode...")
    state = env.reset()
    done = False
    episode_data = []
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        episode_data.append((state, action, reward))
        total_reward += reward
        state = next_state

    # Visualize the episode trajectory
    trajectory_fig = viz.visualize_episode_trajectory(
        env,
        episode_data,
        "MCTS Final Episode Trajectory"
    )
    plt.show()

    print(f"Final episode reward: {total_reward}")

    return agent, policies, episode_rewards


if __name__ == "__main__":
    agent, final_policies, rewards = run_mcts_example()