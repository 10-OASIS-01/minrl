"""Control with a known model: value iteration and policy iteration."""

import numpy as np
from .policy_evaluation import PolicyEvaluator, random_policy


class PolicyOptimizer:
    def __init__(self, env, gamma=0.99):
        self.env, self.gamma = env, gamma
        self.state_values = np.zeros(env.n_states)
        self.policy = random_policy(env)

    def action_values(self, state, values):
        result = np.full(self.env.n_actions, -np.inf)
        for action in self.env.get_valid_actions(state):
            nxt, reward, done = self.env.transition(state, action)
            result[action] = reward + self.gamma * (not done) * values[nxt]
        return result

    def _extract_policy_from_values(self):
        policy = {}
        for state in range(self.env.n_states):
            policy[state] = np.zeros(self.env.n_actions)
            if state not in self.env.terminal_states:
                policy[state][np.argmax(self.action_values(state, self.state_values))] = 1.0
        return policy

    def value_iteration(self, theta=1e-8, max_iterations=10000):
        self.state_values.fill(0)
        for _ in range(max_iterations):
            old_values = self.state_values.copy()
            for state in range(self.env.n_states):
                if state not in self.env.terminal_states:
                    self.state_values[state] = self.action_values(state, old_values).max()
            if np.max(np.abs(old_values - self.state_values)) < theta:
                break
        self.policy = self._extract_policy_from_values()
        return self.policy, self.state_values.copy()

    def policy_iteration(self, theta=1e-8, max_iterations=1000):
        evaluator = PolicyEvaluator(self.env, self.gamma)
        for _ in range(max_iterations):
            # 1. Evaluate the entire policy, including stochastic probabilities.
            self.state_values = evaluator.evaluate_policy(self.policy, theta)
            # 2. Improve it greedily. Compare distributions, not just their argmax.
            new_policy = self._extract_policy_from_values()
            stable = all(np.array_equal(self.policy[s], new_policy[s]) for s in new_policy)
            self.policy = new_policy
            if stable:
                break
        self.state_values = evaluator.evaluate_policy(self.policy, theta)
        return self.policy, self.state_values.copy()

    def print_policy(self, policy):
        symbols = ["↑", "→", "↓", "←"]
        for row in range(self.env.size):
            print(
                " ".join(
                    "T"
                    if row * self.env.size + col in self.env.terminal_states
                    else symbols[np.argmax(policy[row * self.env.size + col])]
                    for col in range(self.env.size)
                )
            )
