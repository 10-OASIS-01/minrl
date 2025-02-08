import numpy as np

from src.environment.grid_world import GridWorld
from src.agents.policy_optimization import PolicyOptimizer


def test_optimization_methods():
    # Create environment
    env = GridWorld(size=3)
    optimizer = PolicyOptimizer(env)
    
    # Test Value Iteration
    print("Running Value Iteration...")
    vi_policy, vi_values = optimizer.value_iteration()
    
    print("\nOptimal Policy (Value Iteration):")
    optimizer.print_policy(vi_policy)
    print("\nOptimal State Values (Value Iteration):")
    for i in range(env.size):
        row_values = []
        for j in range(env.size):
            state = env._pos_to_state((i, j))
            row_values.append(f"{vi_values[state]:6.2f}")
        print(" ".join(row_values))
    
    # Test Policy Iteration
    print("\nRunning Policy Iteration...")
    pi_policy, pi_values = optimizer.policy_iteration()
    
    print("\nOptimal Policy (Policy Iteration):")
    optimizer.print_policy(pi_policy)
    print("\nOptimal State Values (Policy Iteration):")
    for i in range(env.size):
        row_values = []
        for j in range(env.size):
            state = env._pos_to_state((i, j))
            row_values.append(f"{pi_values[state]:6.2f}")
        print(" ".join(row_values))
    
    # Compare results
    print("\nComparing methods:")
    value_diff = np.max(np.abs(vi_values - pi_values))
    print(f"Maximum difference in state values: {value_diff:.6f}")

def run_comparison_experiment():
    """Run experiments with different grid sizes and parameters"""
    grid_sizes = [3, 5]
    gamma_values = [0.9, 0.99]
    
    for size in grid_sizes:
        for gamma in gamma_values:
            print(f"\nExperiment with grid size {size}x{size}, gamma={gamma}")
            env = GridWorld(size=size)
            optimizer = PolicyOptimizer(env, gamma=gamma)
            
            # Time and iterate value iteration
            vi_policy, vi_values = optimizer.value_iteration()
            
            # Time and iterate policy iteration
            pi_policy, pi_values = optimizer.policy_iteration()
            
            # Compare results
            value_diff = np.max(np.abs(vi_values - pi_values))
            print(f"Maximum difference in state values: {value_diff:.6f}")

if __name__ == "__main__":
    print("Running basic test...")
    test_optimization_methods()
    
    print("\nRunning comparison experiments...")
    run_comparison_experiment()