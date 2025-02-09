"""
PPO Tests
Created by: 10-OASIS-01
Date: 2025-02-09 04:57:01 UTC

Test suite for the PPO implementation, including:
- Training visualization
- Policy comparison
- Hyperparameter experiments
- Integration with GridWorld environment
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

from src.environment.grid_world import GridWorld
from src.agents.ppo import PPOAgent
from src.agents.policy_optimization import PolicyOptimizer

def plot_training_results(rewards: List[float], lengths: List[int],
                         policy_losses: List[float], value_losses: List[float],
                         entropy_losses: List[float]):
    """Plot training results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot rewards and lengths
    ax1.plot(rewards, label='Rewards')
    ax1.plot(lengths, label='Lengths')
    ax1.set_title('Training Progress')
    ax1.set_xlabel('Episode')
    ax1.legend()
    
    # Plot policy losses
    ax2.plot(policy_losses)
    ax2.set_title('Policy Loss')
    ax2.set_xlabel('Update')
    ax2.set_ylabel('Loss')
    
    # Plot value losses
    ax3.plot(value_losses)
    ax3.set_title('Value Loss')
    ax3.set_xlabel('Update')
    ax3.set_ylabel('Loss')
    
    # Plot entropy losses
    ax4.plot(entropy_losses)
    ax4.set_title('Entropy Loss')
    ax4.set_xlabel('Update')
    ax4.set_ylabel('Loss')
    
    plt.tight_layout()
    plt.show()

def compare_methods(env: GridWorld):
    """Compare PPO with Value Iteration"""
    # Train PPO agent
    ppo_agent = PPOAgent(env)
    ppo_rewards, ppo_lengths = ppo_agent.train(n_episodes=1000)
    ppo_policy = ppo_agent.get_optimal_policy()
    # Get Value Iteration policy
    optimizer = PolicyOptimizer(env)
    vi_policy, _ = optimizer.value_iteration()
    
    # Print policies
    methods = [("PPO", ppo_policy), ("Value Iteration", vi_policy)]
    action_symbols = ['↑', '→', '↓', '←']
    
    for name, policy in methods:
        print(f"\n{name} Policy:")
        for i in range(env.size):
            row = []
            for j in range(env.size):
                state = env._pos_to_state((i, j))
                action_idx = max(range(len(policy[state])), 
                               key=lambda i: policy[state][i])
                row.append(action_symbols[action_idx])
            print(' '.join(row))

def run_ppo_experiments():
    """Run experiments with different hyperparameters"""
    grid_sizes = [3, 5]
    learning_rates = [0.0003, 0.0001]
    clip_ratios = [0.1, 0.2]
    
    results = {}
    
    for size in grid_sizes:
        for lr in learning_rates:
            for clip in clip_ratios:
                print(f"\nExperiment: size={size}, lr={lr}, clip={clip}")
                
                env = GridWorld(size=size)
                agent = PPOAgent(env, learning_rate=lr, clip_ratio=clip)
                
                # Train agent
                rewards, lengths = agent.train(n_episodes=1000)
                
                # Store results
                results[(size, lr, clip)] = {
                    'avg_reward': np.mean(rewards[-100:]),
                    'avg_length': np.mean(lengths[-100:])
                }
    
    # Print results summary
    print("\nExperiment Results Summary:")
    for params, metrics in results.items():
        print(f"Size={params[0]}, LR={params[1]}, Clip={params[2]}:")
        print(f"  Avg Reward: {metrics['avg_reward']:.2f}")
        print(f"  Avg Length: {metrics['avg_length']:.2f}")

def test_ppo():
    """Test PPO implementation"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create environment
    env = GridWorld(size=3)
    
    # Create and train PPO agent
    agent = PPOAgent(env)
    rewards, lengths = agent.train(n_episodes=1000)
    
    # Plot training results
    plot_training_results(rewards, lengths,
                         agent.policy_losses,
                         agent.value_losses,
                         agent.entropy_losses)
    
    # Compare with Value Iteration
    compare_methods(env)
    
    # Run experiments with different parameters
    run_ppo_experiments()

if __name__ == "__main__":
    test_ppo()
