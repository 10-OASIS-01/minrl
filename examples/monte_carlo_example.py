"""Lesson 4: average complete returns under a fixed policy."""

import numpy as np
from minrl import GridWorld, MonteCarloEvaluator, PolicyEvaluator


def main():
    env = GridWorld()
    mc = MonteCarloEvaluator(env, seed=0)
    policy = mc.create_random_policy()
    estimated = mc.evaluate_policy(policy, num_episodes=3000, max_steps=1000)
    exact = PolicyEvaluator(env).evaluate_policy(policy)
    print("Monte Carlo:\n", estimated.reshape(3, 3))
    print("Bellman evaluation:\n", exact.reshape(3, 3))
    print(f"Mean absolute error: {np.abs(estimated - exact).mean():.4f}")
    print(f"Discarded truncated episodes: {mc.truncated_episodes}")


if __name__ == "__main__":
    main()
