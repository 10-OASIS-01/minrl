from src.environment.grid_world import GridWorld, Action
from src.agents.policy_evaluation import PolicyEvaluator

def create_random_policy(env: GridWorld) -> dict:
    """
    Create a random policy for testing.
    
    Args:
        env (GridWorld): The environment
        
    Returns:
        dict: A dictionary mapping states to action probabilities
    """
    policy = {}
    n_actions = env.get_action_space_size()
    
    for state in range(env.get_state_space_size()):
        # For terminal states, action probabilities don't matter
        if state in env.terminal_states:
            policy[state] = [0.25] * n_actions
            continue
            
        valid_actions = env.get_valid_actions(state)
        probs = []
        
        # Create probability distribution over actions
        for action in range(n_actions):
            if action in valid_actions:
                probs.append(1.0 / len(valid_actions))
            else:
                probs.append(0.0)
                
        policy[state] = probs
        
    return policy

def test_policy_evaluation():
    # Create a 3x3 grid world
    env = GridWorld(size=3)
    
    # Create a random policy
    random_policy = create_random_policy(env)
    
    # Create policy evaluator
    evaluator = PolicyEvaluator(env, gamma=0.99)
    
    # Evaluate the policy
    print("Evaluating random policy...")
    state_values = evaluator.evaluate_policy(random_policy, theta=1e-6)
    
    print("\nFinal state values:")
    evaluator.print_values()
    
    # Verify values for terminal states
    print("\nVerifying terminal state values:")
    for state, reward in env.terminal_states.items():
        print(f"Terminal state {state}: Expected = {reward}, Got = {state_values[state]:.2f}")

def test_deterministic_policy():
    # Create a 3x3 grid world
    env = GridWorld(size=3)
    
    # Create a deterministic policy (always go right if possible, else go down)
    deterministic_policy = {}
    for state in range(env.get_state_space_size()):
        probs = [0.0] * env.get_action_space_size()
        valid_actions = env.get_valid_actions(state)
        
        if Action.RIGHT in valid_actions:
            probs[Action.RIGHT] = 1.0
        elif Action.DOWN in valid_actions:
            probs[Action.DOWN] = 1.0
        elif valid_actions:
            probs[valid_actions[0]] = 1.0
            
        deterministic_policy[state] = probs
    
    # Create policy evaluator
    evaluator = PolicyEvaluator(env, gamma=0.99)
    
    # Evaluate the policy
    print("\nEvaluating deterministic policy...")
    state_values = evaluator.evaluate_policy(deterministic_policy, theta=1e-6)
    
    print("\nFinal state values for deterministic policy:")
    evaluator.print_values()

if __name__ == "__main__":
    print("Testing with random policy:")
    test_policy_evaluation()
    
    print("\nTesting with deterministic policy:")
    test_deterministic_policy()