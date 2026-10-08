"""Lesson 3: learn a table by interacting, without querying the transition model."""

from minrl import GridWorld, QLearningAgent


def main():
    env = GridWorld()
    agent = QLearningAgent(env, seed=0)
    rewards, lengths = agent.train(n_episodes=1000, max_steps=100)
    print(f"Last 100 episodes: mean return {sum(rewards[-100:]) / 100:.2f}")
    state = env.reset()
    for _ in range(100):
        action = agent.select_action(state, deterministic=True)
        state, reward, done, _ = env.step(action)
        print(f"state={state}, reward={reward:+.1f}")
        if done:
            break


if __name__ == "__main__":
    main()
