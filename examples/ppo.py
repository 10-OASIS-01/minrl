"""
PPO Algorithm Example
Created by: 10-OASIS-01
Date: 2025-02-09 08:01:41 UTC

Demonstrates the usage of the PPO algorithm in the GridWorld environment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import GridWorld
from src.agents.ppo import PPOAgent
from src.utils.visualization import Visualizer

import torch
import numpy as np

def run_ppo_demo():
    """Run a demonstration of the PPO algorithm"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create environment with interesting terminal states
    env = GridWorld(size=5)

    # Set terminal states that don't conflict with starting position (2,2)
    goal_state = env._pos_to_state((0, 4))  # Top-right corner
    trap_state = env._pos_to_state((4, 0))  # Bottom-left corner

    # Clear default terminal states and set new ones
    env.terminal_states.clear()  # Clear default terminal states
    env.terminal_states[goal_state] = 1.0
    env.terminal_states[trap_state] = -1.0  # Trap state with negative reward

    # Create PPO agent with custom hyperparameters
    agent = PPOAgent(
        env,
        learning_rate=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        critic_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        num_epochs=4,
        batch_size=32
    )

    # Train the agent
    print("Training PPO agent...")
    n_episodes = 1000
    max_steps = 100
    update_interval = 2048
    rewards, lengths = agent.train(
        n_episodes=n_episodes,
        max_steps=max_steps,
        update_interval=update_interval
    )

    # Plot training results using the Visualizer
    training_fig = Visualizer.plot_ppo_training_results(
        rewards=rewards,
        lengths=lengths,
        policy_losses=agent.policy_losses,
        value_losses=agent.value_losses,
        entropy_losses=agent.entropy_losses,
        title='PPO Training Progress'
    )
    training_fig.show()

    # Get and display the learned policy
    policy = agent.get_optimal_policy()
    policy_fig = Visualizer.plot_policy(
        policy,
        env.size,
        "PPO Learned Policy"
    )
    policy_fig.show()

    # Run and visualize a test episode
    print("\nRunning test episode with learned policy...")
    state = env.reset()
    episode_data = []
    done = False
    total_reward = 0

    while not done:
        action, _, _ = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        episode_data.append((state, action, reward))
        total_reward += reward
        state = next_state

    # Visualize the test episode
    episode_fig = Visualizer.visualize_episode_trajectory(
        env,
        episode_data,
        "PPO Test Episode Trajectory"
    )
    episode_fig.show()

    print(f"Test episode finished with total reward: {total_reward}")

    return agent, policy, total_reward


if __name__ == "__main__":
    agent, policy, final_reward = run_ppo_demo()