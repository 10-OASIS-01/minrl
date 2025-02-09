"""
PPO Example
Created by: 10-OASIS-01
Date: 2025-02-09 06:18:57 UTC

Demonstrates the usage of the PPO algorithm in the GridWorld environment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import GridWorld
from src.agents import PPOAgent
from src.utils.visualization import Visualizer

import torch
import numpy as np
from typing import List, Tuple

def create_test_environment() -> GridWorld:
    """Create a test environment with interesting terminal states"""
    env = GridWorld(size=5)
    env.set_terminal_state((0, 4), 1.0)   # Goal state with positive reward
    env.set_terminal_state((2, 2), -1.0)  # Trap state with negative reward
    return env

def run_test_episode(agent: PPOAgent, env: GridWorld) -> Tuple[float, List[Tuple[int, int, float]]]:
    """
    Run a test episode with the trained agent.
    
    Returns:
        Tuple containing total reward and episode data
    """
    state = env.reset()
    episode_data = []
    total_reward = 0
    done = False
    
    while not done:
        action, _, _ = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        episode_data.append((state, action, reward))
        total_reward += reward
        state = next_state
    
    # Add final state
    episode_data.append((state, None, None))
    return total_reward, episode_data

def run_ppo_demo():
    """Run a comprehensive demonstration of the PPO algorithm"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create environment and agent
    env = create_test_environment()
    agent = PPOAgent(
        env,
        learning_rate=0.0003,
        gamma=0.99,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01
    )
    
    # Train the agent
    print("Training PPO agent...")
    rewards, lengths = agent.train(n_episodes=1000)
    
    # Visualize training results
    training_fig = Visualizer.plot_ppo_training_results(
        rewards=rewards,
        lengths=lengths,
        policy_losses=agent.policy_losses,
        value_losses=agent.value_losses,
        entropy_losses=agent.entropy_losses,
        title="PPO Training Progress"
    )
    training_fig.show()
    
    # Get and visualize the learned policy
    policy = agent.get_optimal_policy()
    policy_fig = Visualizer.plot_policy(
        policy,
        env.size,
        "PPO Learned Policy"
    )
    policy_fig.show()
    
    # Run and visualize a test episode
    print("\nRunning test episode with learned policy...")
    test_reward, episode_data = run_test_episode(agent, env)
    
    trajectory_fig = Visualizer.visualize_episode_trajectory(
        env,
        episode_data,
        f"PPO Test Episode (Reward: {test_reward:.2f})"
    )
    trajectory_fig.show()
    
    print(f"Test episode finished with total reward: {test_reward}")
    
    return {
        'agent': agent,
        'final_policy': policy,
        'training_rewards': rewards,
        'training_lengths': lengths,
        'test_reward': test_reward,
        'test_trajectory': episode_data
    }

if __name__ == "__main__":
    results = run_ppo_demo()
