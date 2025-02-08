import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import GridWorld
from src.agents import (PolicyEvaluator, PolicyOptimizer, 
                       QLearningAgent, DQNAgent)
from src.utils.visualization import Visualizer

import numpy as np
import torch
import random
from datetime import datetime

def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def run_all_experiments():
    """Run experiments with all implemented algorithms"""
    # Configuration
    grid_sizes = [3, 5]
    experiment_configs = {
        'Value Iteration': {
            'gamma': [0.9, 0.99],
        },
        'Q-Learning': {
            'learning_rate': [0.1, 0.01],
            'gamma': [0.9, 0.99],
        },
        'DQN': {
            'learning_rate': [0.001, 0.0001],
            'hidden_dim': [32, 64],
        }
    }
    
    results = {}
    
    for size in grid_sizes:
        print(f"\nRunning experiments for {size}x{size} grid...")
        env = GridWorld(size=size)
        
        # Value Iteration
        print("\nRunning Value Iteration...")
        for gamma in experiment_configs['Value Iteration']['gamma']:
            optimizer = PolicyOptimizer(env, gamma=gamma)
            policy, values = optimizer.value_iteration()
            
            # Visualize results
            Visualizer.plot_value_function(values, size, 
                f"Value Function (γ={gamma})")
            Visualizer.plot_policy(policy, size, 
                f"Value Iteration Policy (γ={gamma})")
        
        # Q-Learning
        print("\nRunning Q-Learning...")
        for lr in experiment_configs['Q-Learning']['learning_rate']:
            for gamma in experiment_configs['Q-Learning']['gamma']:
                agent = QLearningAgent(env, learning_rate=lr, gamma=gamma)
                rewards, lengths = agent.train(n_episodes=1000)
                
                # Visualize results
                Visualizer.plot_training_results(
                    rewards, lengths, 
                    title=f"Q-Learning (lr={lr}, γ={gamma})")
        
        # DQN
        print("\nRunning DQN...")
        for lr in experiment_configs['DQN']['learning_rate']:
            for hidden_dim in experiment_configs['DQN']['hidden_dim']:
                agent = DQNAgent(env, learning_rate=lr)
                rewards, lengths = agent.train(n_episodes=1000)
                
                # Visualize results
                Visualizer.plot_training_results(
                    rewards, lengths, agent.losses,
                    title=f"DQN (lr={lr}, hidden={hidden_dim})")

def main():
    # Set random seeds
    set_random_seeds()
    
    # Record start time
    start_time = datetime.utcnow()
    print(f"Starting experiments at {start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # Run experiments
    run_all_experiments()
    
    # Record end time
    end_time = datetime.utcnow()
    print(f"\nExperiments completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Total duration: {end_time - start_time}")

if __name__ == "__main__":
    main()