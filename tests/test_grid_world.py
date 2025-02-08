from src.environment.grid_world import GridWorld, Action

def test_grid_world():
    # Create a 3x3 grid world
    env = GridWorld(size=3)
    
    # Reset the environment
    initial_state = env.reset()
    print("Initial state:", initial_state)
    print("\nInitial grid:")
    print(env.render())
    
    # Try some actions
    print("\nTaking some actions:")
    actions = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]
    
    for action in actions:
        print(f"\nTaking action: {action.name}")
        next_state, reward, done, _ = env.step(action)
        print(f"Next state: {next_state}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(env.render())
        
        if done:
            print("\nEpisode finished!")
            break

if __name__ == "__main__":
    test_grid_world()