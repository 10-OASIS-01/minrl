"""
MCTS Example
Created by: 10-OASIS-01
Date: 2025-02-09 06:37:03 UTC

Demonstrates the usage of Monte Carlo Tree Search in the GridWorld environment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.grid_world import GridWorld, Action
from src.agents.mcts import MCTSAgent
from src.utils.visualization import Visualizer
from src.agents.policy_optimization import PolicyOptimizer

import numpy as np
from typing import Dict, List, Tuple


def setup_environment() -> GridWorld:
    """Create and set up the GridWorld environment"""
    env = GridWorld(size=4)
    # Add interesting terminal states
    env.set_terminal_state((0, 3), 1.0)   # Goal state
    env.set_terminal_state((1, 1), -1.0)  # Trap state
    return env


def demonstrate_single_episode(agent: MCTSAgent, visualizer: Visualizer):
    """Run and visualize a single episode"""
    print("\nRunning single episode demonstration...")
    states, actions, rewards = agent.run_episode()
    total_reward = sum(rewards)
    
    print(f"Episode finished with reward {total_reward:.2f} in {len(states)} steps")
    
    # Visualize the episode trajectory
    visualizer.visualize_episode(
        env=agent.env,
        states=states,
        actions=actions,
        rewards=rewards,
        title="MCTS Example Episode"
    )
    return states, actions, rewards


def compare_with_value_iteration(agent: MCTSAgent, 
                               visualizer: Visualizer):
    """Compare MCTS policy with Value Iteration"""
    print("\nComparing MCTS with Value Iteration...")
    
    # Get MCTS policy after some exploration
    initial_state = agent.env.reset()
    agent.select_action(initial_state)  # Build the search tree
    mcts_policy = agent.get_optimal_policy()
    
    # Get Value Iteration policy
    optimizer = PolicyOptimizer(agent.env)
    vi_policy, vi_values = optimizer.value_iteration()
    
    # Visualize policies side by side
    visualizer.plot_policy(
        mcts_policy, 
        agent.env.size, 
        "MCTS Policy"
    )
    
    visualizer.plot_policy(
        vi_policy, 
        agent.env.size, 
        "Value Iteration Policy"
    )
    
    # Visualize value function from Value Iteration
    visualizer.plot_value_function(
        vi_values, 
        agent.env.size, 
        "Value Function (VI)"
    )
    
    return mcts_policy, vi_policy, vi_values


def run_parameter_study(env: GridWorld, visualizer: Visualizer):
    """Run and visualize parameter study"""
    print("\nRunning parameter study...")
    
    # Create base agent for parameter search
    agent = MCTSAgent(env)
    
    # Define parameter ranges
    simulation_counts = [50, 100, 200]
    exploration_weights = [1.0, 1.4, 2.0]
    
    # Run parameter search
    results = agent.parameter_search(
        simulation_counts=simulation_counts,
        exploration_weights=exploration_weights,
        episodes_per_config=20
    )
    
    # Prepare data for visualization
    config_results = {}
    for (sims, weight), metrics in results.items():
        label = f'Sims={sims}, c={weight:.1f}'
        config_results[label] = {
            'rewards': metrics['rewards'],
            'lengths': metrics['lengths']
        }
        
        print(f"\nConfiguration: {label}")
        print(f"  Mean Reward: {metrics['mean_reward']:.2f} "
              f"± {metrics['std_reward']:.2f}")
        print(f"  Mean Length: {metrics['mean_length']:.2f} "
              f"± {metrics['std_length']:.2f}")
    
    # Visualize learning curves
    episodes = list(range(len(next(iter(config_results.values()))['rewards'])))
    
    # Plot reward curves for each configuration
    reward_curves = {
        label: data['rewards'] 
        for label, data in config_results.items()
    }
    
    visualizer.plot_convergence_analysis(
        episodes=episodes,
        errors_dict=reward_curves,
        title="MCTS Performance Across Configurations",
        xlabel="Episode",
        ylabel="Total Reward"
    )
    
    return results


def demonstrate_convergence(agent: MCTSAgent, visualizer: Visualizer):
    """Demonstrate and visualize MCTS convergence"""
    print("\nAnalyzing MCTS convergence...")
    
    # Run multiple episodes with the same configuration
    results = agent.run_experiments(n_episodes=100, reset_tree=True)
    
    # Visualize convergence
    visualizer.plot_training_results(
        rewards=results['rewards'],
        lengths=results['lengths'],
        title="MCTS Convergence Analysis"
    )
    
    return results


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create environment and visualization tools
    env = setup_environment()
    visualizer = Visualizer()
    
    # Create MCTS agent with default parameters
    agent = MCTSAgent(env, exploration_weight=1.4, n_simulations=100)
    
    # Demonstrate single episode
    episode_results = demonstrate_single_episode(agent, visualizer)
    
    # Compare with Value Iteration
    policy_comparison = compare_with_value_iteration(agent, visualizer)
    
    # Run parameter study
    param_study_results = run_parameter_study(env, visualizer)
    
    # Analyze convergence
    convergence_results = demonstrate_convergence(agent, visualizer)
    
    print("\nMCTS demonstration completed successfully!")


if __name__ == "__main__":
    main()
