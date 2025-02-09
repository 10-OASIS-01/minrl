import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

from src.environment.grid_world import GridWorld
from src.agents.actor_critic import ActorCriticAgent
from src.agents.policy_optimization import PolicyOptimizer

def plot_training_results(rewards: List[float], lengths: List[int], 
                         actor_losses: List[float], critic_losses: List[float]):
    """Plot training results"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot rewards and lengths
    ax1.plot(rewards, label='Rewards')
    ax1.plot(lengths, label='Lengths')
    ax1.set_title('Training Progress')
    ax1.set_xlabel('Episode')
    ax1.legend()
    
    # Plot actor losses
    ax2.plot(actor_losses)
    ax2.set_title('Actor Loss')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    
    # Plot critic losses
    ax3.plot(critic_losses)
    ax3.set_title('Critic Loss')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    
    plt.tight_layout()
    plt.show()

def compare_methods(env: GridWorld):
    """Compare Actor-Critic with Value Iteration"""
    # Train Actor-Critic agent
    ac_agent = ActorCriticAgent(env)
    ac_rewards, ac_lengths = ac_agent.train(n_episodes=1000)
    ac_policy = ac_agent.get_optimal_policy()
    
    # Get Value Iteration policy
    optimizer = PolicyOptimizer(env)
    vi_policy, _ = optimizer.value_iteration()
    
    # Print policies
    methods = [("Actor-Critic", ac_policy), ("Value Iteration", vi_policy)]
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

def run_ac_experiments():
    """Run experiments with different hyperparameters"""
    grid_sizes = [3, 5]
    actor_lrs = [0.001, 0.0001]
    critic_lrs = [0.001, 0.0001]
    
    results = {}
    
    for size in grid_sizes:
        for actor_lr in actor_lrs:
            for critic_lr in critic_lrs:
                print(f"\nExperiment: size={size}, actor_lr={actor_lr}, critic_lr={critic_lr}")
                
                env = GridWorld(size=size)
                agent = ActorCriticAgent(env, actor_lr=actor_lr, critic_lr=critic_lr)
                
                # Train agent
                rewards, lengths = agent.train(n_episodes=1000)
                
                # Store results
                results[(size, actor_lr, critic_lr)] = {
                    'avg_reward': np.mean(rewards[-100:]),
                    'avg_length': np.mean(lengths[-100:])
                }
    
    # Print results summary
    print("\nExperiment Results Summary:")
    for params, metrics in results.items():
        print(f"Size={params[0]}, Actor LR={params[1]}, Critic LR={params[2]}:")
        print(f"  Avg Reward: {metrics['avg_reward']:.2f}")
        print(f"  Avg Length: {metrics['avg_length']:.2f}")

def test_actor_critic():
    """Test Actor-Critic implementation"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create environment
    env = GridWorld(size=3)
    
    # Create and train Actor-Critic agent
    agent = ActorCriticAgent(env)
    rewards, lengths = agent.train(n_episodes=1000)
    
    # Plot training results
    plot_training_results(rewards, lengths, agent.actor_losses, agent.critic_losses)
    
    # Compare with Value Iteration
    compare_methods(env)
    
    # Run experiments with different parameters
    run_ac_experiments()

if __name__ == "__main__":
    test_actor_critic()
