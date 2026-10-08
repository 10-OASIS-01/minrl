"""Lesson 5: use model-based lookahead at every decision."""

from minrl import GridWorld, MCTSAgent


def main():
    env = GridWorld()
    agent = MCTSAgent(env, num_simulations=200, seed=0)
    state = env.reset()
    for step in range(100):
        action = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        print(f"step={step + 1}, state={state}, reward={reward:+.1f}")
        if done:
            break


if __name__ == "__main__":
    main()
