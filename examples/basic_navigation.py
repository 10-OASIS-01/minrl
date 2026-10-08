"""Lesson 1: an environment maps (state, action) to (next state, reward, done)."""

from minrl import GridWorld, Action


def main():
    env = GridWorld(terminal_states={8: 1.0})
    state = env.reset()
    for action in [Action.RIGHT, Action.DOWN, Action.RIGHT, Action.DOWN]:
        next_state, reward, done, _ = env.step(action)
        print(f"s={state}, a={action.name}, r={reward:+.1f}, s'={next_state}, done={done}")
        state = next_state
        if done:
            break
    print(env.render())


if __name__ == "__main__":
    main()
