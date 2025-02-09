"""
MCTS Example
Created by: 10-OASIS-01
Date: 2025-02-09 05:22:17 UTC

Demonstrates the usage of Monte Carlo Tree Search in the GridWorld environment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import GridWorld
from src.agents.mcts import MCTSAgent
from src.utils.visualization import Visualizer
from src.agents.policy_optimization import PolicyOptimizer

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def run_episode(env: GridWorld, agent: MCTSAgent) -> Tuple[float, int]:
    """Run a single episode with the MCTS agent"""
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        action = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
    
    return total_reward, steps

def compare_with_value_iteration(env: GridWorld, mcts_agent: MCTSAgent):
    """Compare MCTS policy with Value Iteration policy"""
    # Get MCTS policy
    state = env.reset()
    mcts_agent.select_action(state)  # Run MCTS to build the tree
    mcts_policy = mcts_agent.get_optimal_policy()
    
    # Get Value Iteration policy
    optimizer = PolicyOptimizer(env)
    vi_policy, _ = optimizer.value_iteration()
    
    # Visualize both policies
    Visualizer.plot_policy(mcts_policy, env.size, "MCTS Policy")
    Visualizer.plot_policy(vi_policy, env.size, "Value Iteration Policy")

def run_mcts_experiments():
    """Run experiments with different parameters"""
    print("Running MCTS experiments...")
    
    # Test different grid sizes and simulation counts
    grid_sizes = [3, 4]
    simulation_counts = [50, 100, 200]
    exploration_weights = [1.0, 1.4, 2.0]
    
    results = {}
    
    for size in grid_sizes:
        env = GridWorld(size=size)
        
        for sims in simulation_counts:
            for weight in exploration_weights:
                print(f"\nGrid size: {size}x{size}, "
                      f"Simulations: {sims}, "
                      f"Exploration weight: {weight}")
                
                agent = MCTSAgent(env, 
                                exploration_weight=weight,
                                n_simulations=sims)
                
                # Run multiple episodes
                n_episodes = 20
                rewards = []
                lengths = []
                
                for episode in range(n_episodes):
                    reward, steps = run_episode(env, agent)
                    rewards.append(reward)
                    lengths.append(steps)
                
                results[(size, sims, weight)] = {
                    'avg_reward': np.mean(rewards),
                    'avg_length': np.mean(lengths),
                    'std_reward': np.std(rewards)
                }
    
    # Print results summary
    print("\nExperiment Results Summary:")
    for params, metrics in results.items():
        print(f"\nSize={params[0]}, Simulations={params[1]}, "
              f"Exploration weight={params[2]}:")
        print(f"  Avg Reward: {metrics['avg_reward']:.2f} "
              f"± {metrics['std_reward']:.2f}")
        print(f"  Avg Length: {metrics['avg_length']:.2f}")

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create environment and agent
    env = GridWorld(size=4)
    agent = MCTSAgent(env)
    
    # Run and visualize a single episode
    print("Running single episode demonstration...")
    reward, steps = run_episode(env, agent)
    print(f"Episode finished with reward {reward} in {steps} steps")
    
    # Compare with Value Iteration
    print("\nComparing MCTS with Value Iteration...")
    compare_with_value_iteration(env, agent)
    
    # Run parameter experiments
    run_mcts_experiments()

if __name__ == "__main__":
    main()
