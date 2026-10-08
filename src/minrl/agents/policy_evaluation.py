"""Prediction: evaluate a fixed policy using the Bellman expectation equation."""

import numpy as np


def random_policy(env):
    policy = {}
    for state in range(env.n_states):
        probs = np.zeros(env.n_actions)
        actions = env.get_valid_actions(state)
        if actions:
            probs[actions] = 1 / len(actions)
        policy[state] = probs
    return policy


class PolicyEvaluator:
    def __init__(self, env, gamma=0.99):
        self.env, self.gamma = env, gamma
        self.state_values = np.zeros(env.n_states)

    def evaluate_policy(self, policy, theta=1e-8, max_iterations=10000):
        self.state_values.fill(0)
        for _ in range(max_iterations):
            old_values = self.state_values.copy()
            for state in range(self.env.n_states):
                if state in self.env.terminal_states:
                    continue
                probs = np.asarray(policy[state])
                if (
                    probs.shape != (self.env.n_actions,)
                    or np.any(probs < 0)
                    or not np.isclose(probs.sum(), 1)
                ):
                    raise ValueError(
                        "each nonterminal state needs an action probability distribution"
                    )
                value = 0.0
                # Include boundary actions too: GridWorld defines them as self-loops.
                for action, probability in enumerate(probs):
                    nxt, reward, done = self.env.transition(state, action)
                    value += probability * (reward + self.gamma * (not done) * old_values[nxt])
                self.state_values[state] = value
            if np.max(np.abs(old_values - self.state_values)) < theta:
                break
        return self.state_values.copy()

    def print_values(self):
        print(self.state_values.reshape(self.env.size, self.env.size))
