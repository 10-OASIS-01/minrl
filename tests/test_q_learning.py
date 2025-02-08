import numpy as np
from typing import List, Dict

import matplotlib.pyplot as plt
from src.environment.grid_world import GridWorld
from src.agents.q_learning import QLearningAgent
from src.agents.policy_optimization import PolicyOptimizer

def plot_training_results(rewards: List[float], lengths: List[int]):
    """Plot training results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
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
    
    plt.tight_layout()
    plt.show()

def compare_with_value_iteration(env: GridWorld, q_learning_policy: Dict[int, List[float]]):
    """Compare Q-learning policy with Value Iteration policy"""
    # Get optimal policy using value iteration
    optimizer = PolicyOptimizer(env)
    vi_policy, _ = optimizer.value_iteration()
    
    # Print both policies
    print("\nQ-Learning Policy:")
    action_symbols = ['↑', '→', '↓', '←']
    for i in range(env.size):
        row = []
        for j in range(env.size):
            state = env._pos_to_pos((i, j))
            action_idx = max(range(len(q_learning_policy[state])), 
                           key=lambda i: q_learning_policy[state][i])
            row.append(action_symbols[action_idx])
        print(' '.join(row))
    
    print("\nValue Iteration Policy:")
    for i in range(env.size):
        row = []
        for j in range(env.size):
            state = env._pos_to_pos((i, j))
            action_idx = max(range(len(vi_policy[state])), 
                           key=lambda i: vi_policy[state][i])
            row.append(action_symbols[action_idx])
        print(' '.join(row))

def run_q_learning_experiments():
    """Run experiments with different hyperparameters"""
    # Test different grid sizes
    grid_sizes = [3, 5]
    learning_rates = [0.1, 0.01]
    gammas = [0.9, 0.99]
    
    results = {}
    
    for size in grid_sizes:
        for lr in learning_rates:
            for gamma in gammas:
                print(f"\nExperiment: size={size}, lr={lr}, gamma={gamma}")
                
                env = GridWorld(size=size)
                agent = QLearningAgent(env, learning_rate=lr, gamma=gamma)
                
                # Train agent
                rewards, lengths = agent.train(n_episodes=1000)
                
                # Store results
                results[(size, lr, gamma)] = {
                    'avg_reward': np.mean(rewards[-100:]),
                    'avg_length': np.mean(lengths[-100:])
                }
    
    # Print results summary
    print("\nExperiment Results Summary:")
    for params, metrics in results.items():
        print(f"Size={params[0]}, LR={params[1]}, Gamma={params[2]}:")
        print(f"  Avg Reward: {metrics['avg_reward']:.2f}")
        print(f"  Avg Length: {metrics['avg_length']:.2f}")

def main():
    # Create environment
    env = GridWorld(size=3)
    
    # Create and train Q-learning agent
    agent = QLearningAgent(env)
    rewards, lengths = agent.train(n_episodes=1000)
    
    # Plot training results
    plot_training_results(rewards, lengths)
    
    # Get and print learned policy
    optimal_policy = agent.get_optimal_policy()
    print("\nLearned Q-values:")
    agent.print_q_values()
    
    # Compare with value iteration
    compare_with_value_iteration(env, optimal_policy)
    
    # Run experiments with different parameters
    run_q_learning_experiments()

if __name__ == "__main__":
    main()