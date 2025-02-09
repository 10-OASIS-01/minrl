"""
Actor-Critic Example
Created by: 10-OASIS-01
Date: 2025-02-09 04:56:06 UTC

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
import matplotlib.pyplot as plt

def run_actor_critic_demo():
    """Run a demonstration of the Actor-Critic algorithm"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create environment and agent
    env = GridWorld(size=4)
    agent = ActorCriticAgent(env)
    
    # Train the agent
    print("Training Actor-Critic agent...")
    rewards, lengths = agent.train(n_episodes=1000)
    
    # Plot training results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.tight_layout()
    plt.show()
    
    # Display learned policy
    policy = agent.get_optimal_policy()
    Visualizer.plot_policy(policy, env.size, "Actor-Critic Learned Policy")
    
    # Run an episode with the learned policy
    print("\nRunning episode with learned policy...")
    state = env.reset()
    env.render()
    
    done = False
    total_reward = 0
    while not done:
        action, _ = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
    
    print(f"Episode finished with total reward: {total_reward}")

if __name__ == "__main__":
    run_actor_critic_demo()
