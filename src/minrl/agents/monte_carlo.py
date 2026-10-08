"""Monte Carlo prediction learns from complete sampled returns."""

import numpy as np
from .policy_evaluation import random_policy


class MonteCarloEvaluator:
    def __init__(self, env, gamma=0.99, seed=0):
        self.env, self.gamma = env, gamma
        self.rng = np.random.default_rng(seed)
        self.state_values = np.zeros(env.n_states)
        self.truncated_episodes = 0

    def create_random_policy(self):
        return random_policy(self.env)

    def generate_episode(self, policy, max_steps=1000):
        state = self.env.reset()
        episode = []
        for _ in range(max_steps):
            action = self.rng.choice(self.env.n_actions, p=policy[state])
            nxt, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = nxt
            if done:
                return episode, False
        return episode, True

    def evaluate_policy(self, policy, num_episodes=1000, first_visit=True, max_steps=1000):
        self.state_values.fill(0)
        counts = np.zeros(self.env.n_states)
        self.truncated_episodes = 0
        for _ in range(num_episodes):
            episode, truncated = self.generate_episode(policy, max_steps)
            if truncated:
                self.truncated_episodes += 1
                continue  # A partial return is not the episodic Monte Carlo target.
            returns = np.zeros(len(episode))
            total = 0.0
            for t in reversed(range(len(episode))):
                total = episode[t][2] + self.gamma * total
                returns[t] = total
            visited = set()
            for (state, _, _), total in zip(episode, returns):
                if first_visit and state in visited:
                    continue
                visited.add(state)
                counts[state] += 1
                self.state_values[state] += (total - self.state_values[state]) / counts[state]
        return self.state_values.copy()

    def print_values(self):
        print(self.state_values.reshape(self.env.size, self.env.size))
