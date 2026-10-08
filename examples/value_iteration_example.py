"""Lesson 2: with a transition model, compute values before acting."""

from minrl import GridWorld, PolicyOptimizer


def main():
    env = GridWorld()
    optimizer = PolicyOptimizer(env, gamma=0.99)
    policy, values = optimizer.value_iteration()
    print("V*(s):\n", values.reshape(env.size, env.size))
    optimizer.print_policy(policy)


if __name__ == "__main__":
    main()
