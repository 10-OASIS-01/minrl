import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

from src.environment.grid_world import GridWorld
from src.agents.deep_q_learning import DQNAgent, DQNetwork
from src.agents.policy_optimization import PolicyOptimizer

def plot_training_results(rewards: List[float], lengths: List[int], losses: List[float]):
    """Plot training results"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot rewards
    ax1.plot(rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    
    # Plot episode lengths
    ax2.plot(lengths)
    ax2.set_title('Episode Lengths')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    
    # Plot losses
    ax3.plot(losses)
    ax3.set_title('Training Loss')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    
    plt.tight_layout()
    plt.show()

def compare_methods(env: GridWorld):
    """Compare DQN with Value Iteration and Q-Learning"""
    # Train DQN agent
    dqn_agent = DQNAgent(env)
    dqn_rewards, dqn_lengths = dqn_agent.train(n_episodes=1000)
    dqn_policy = dqn_agent.get_optimal_policy()
    
    # Get Value Iteration policy
    optimizer = PolicyOptimizer(env)
    vi_policy, _ = optimizer.value_iteration()
    
    # Print policies
    methods = [("DQN", dqn_policy), ("Value Iteration", vi_policy)]
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

def run_dqn_experiments():
    """Run experiments with different hyperparameters"""
    grid_sizes = [3, 5]
    learning_rates = [0.001, 0.0001]
    hidden_dims = [32, 64]
    
    results = {}
    
    for size in grid_sizes:
        for lr in learning_rates:
            for hidden_dim in hidden_dims:
                print(f"\nExperiment: size={size}, lr={lr}, hidden_dim={hidden_dim}")
                
                env = GridWorld(size=size)
                agent = DQNAgent(env, learning_rate=lr)
                agent.q_network = DQNetwork(agent.state_dim, agent.action_dim, hidden_dim)
                agent.target_network = DQNetwork(agent.state_dim, agent.action_dim, hidden_dim)
                agent.target_network.load_state_dict(agent.q_network.state_dict())
                
                # Train agent
                rewards, lengths = agent.train(n_episodes=1000)
                
                # Store results
                results[(size, lr, hidden_dim)] = {
                    'avg_reward': np.mean(rewards[-100:]),
                    'avg_length': np.mean(lengths[-100:])
                }
    
    # Print results summary
    print("\nExperiment Results Summary:")
    for params, metrics in results.items():
        print(f"Size={params[0]}, LR={params[1]}, Hidden={params[2]}:")
        print(f"  Avg Reward: {metrics['avg_reward']:.2f}")
        print(f"  Avg Length: {metrics['avg_length']:.2f}")

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create environment
    env = GridWorld(size=3)
    
    # Create and train DQN agent
    agent = DQNAgent(env)
    rewards, lengths = agent.train(n_episodes=1000)
    
    # Plot training results
    plot_training_results(rewards, lengths, agent.losses)
    
    # Compare with other methods
    compare_methods(env)
    
    # Run experiments with different parameters
    run_dqn_experiments()

if __name__ == "__main__":
    main()