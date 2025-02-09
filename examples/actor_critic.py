"""
Actor-Critic Example
Created by: 10-OASIS-01
Date: 2025-02-09 06:15:52 UTC

Demonstrates the usage of the Actor-Critic algorithm in the GridWorld environment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import GridWorld
from src.agents import ActorCriticAgent
from src.utils.visualization import Visualizer

import torch
import numpy as np

def run_actor_critic_demo():
    """Run a demonstration of the Actor-Critic algorithm"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create environment with interesting terminal states
    env = GridWorld(size=9)

    # Set terminal states that don't conflict with starting position (2,2)
    goal_state = env._pos_to_state((0, 4))  # Top-right corner
    trap_state = env._pos_to_state((4, 0))  # Bottom-left corner

    # Clear default terminal states and set new ones
    env.terminal_states.clear()  # Clear default terminal states
    env.terminal_states[goal_state] = 1.0  # Goal state with positive reward
    env.terminal_states[trap_state] = -1.0  # Trap state with negative reward

    # Create agent
    agent = ActorCriticAgent(env)

    # Train the agent
    print("Training Actor-Critic agent...")
    rewards, lengths = agent.train(n_episodes=1000)

    # Plot training results using the Visualizer
    training_fig = Visualizer.plot_training_results(
        rewards=rewards,
        lengths=lengths,
        actor_losses=agent.actor_losses,
        critic_losses=agent.critic_losses,
        title='Actor-Critic Training Progress'
    )
    training_fig.show()

    # Get and display the learned policy
    policy = agent.get_optimal_policy()
    policy_fig = Visualizer.plot_policy(
        policy,
        env.size,
        "Actor-Critic Learned Policy"
    )
    policy_fig.show()

    # Run and visualize a test episode
    print("\nRunning test episode with learned policy...")
    state = env.reset()
    states, actions, rewards = [state], [], []
    done = False
    total_reward = 0

    while not done:
        action, _ = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        states.append(next_state)
        actions.append(action)
        rewards.append(reward)
        total_reward += reward
        state = next_state

    # Visualize the test episode
    episode_fig = Visualizer.visualize_episode(
        env,
        states,
        actions,
        rewards,
        "Test Episode Trajectory"
    )
    episode_fig.show()

    print(f"Test episode finished with total reward: {total_reward}")

    return agent, policy, total_reward

if __name__ == "__main__":
    agent, policy, final_reward = run_actor_critic_demo()