"""Tabular Q-learning: observe, act, receive reward, update one table entry."""

import json
from collections import deque
import numpy as np


class QLearningAgent:
    def __init__(
        self,
        env,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01,
        seed=0,
        replay=False,
        buffer_size=10000,
        batch_size=32,
        learning_rate_decay=False,
    ):
        self.env = env
        self.learning_rate, self.gamma = learning_rate, gamma
        self.epsilon, self.epsilon_decay, self.min_epsilon = epsilon, epsilon_decay, min_epsilon
        self.learning_rate_decay = learning_rate_decay
        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((env.n_states, env.n_actions))
        self.replay, self.batch_size = replay, batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.episode_rewards, self.episode_lengths = [], []
        self.config = dict(
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon,
            seed=seed,
            replay=replay,
            buffer_size=buffer_size,
            batch_size=batch_size,
            learning_rate_decay=learning_rate_decay,
        )

    def select_action(self, state, deterministic=False):
        actions = self.env.get_valid_actions(state)
        if not actions:
            raise ValueError("cannot act from a terminal state")
        if not deterministic and self.rng.random() < self.epsilon:
            return int(self.rng.choice(actions))
        return int(max(actions, key=lambda a: self.q_table[state, a]))

    def update(self, state, action, reward, next_state, done=None):
        done = next_state in self.env.terminal_states if done is None else done
        future = (
            0.0
            if done
            else max(self.q_table[next_state, a] for a in self.env.get_valid_actions(next_state))
        )
        # Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]
        target = reward + self.gamma * future
        self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])

    def train(self, n_episodes=1000, max_steps=100):
        if n_episodes < 1 or max_steps < 1:
            raise ValueError("episode and step budgets must be positive")
        initial_lr = self.learning_rate
        for episode in range(n_episodes):
            state, total = self.env.reset(), 0.0
            if self.learning_rate_decay:
                self.learning_rate = initial_lr * (1 - episode / n_episodes)
            for step in range(max_steps):
                action = self.select_action(state)
                nxt, reward, done, _ = self.env.step(action)
                transition = (state, action, reward, nxt, done)
                self.update(*transition)
                # Optional extension: reuse old transitions after the online update.
                if self.replay:
                    self.buffer.append(transition)
                    if len(self.buffer) >= self.batch_size:
                        for i in self.rng.choice(len(self.buffer), self.batch_size, replace=False):
                            self.update(*self.buffer[i])
                total += reward
                state = nxt
                if done:
                    break
            self.episode_rewards.append(total)
            self.episode_lengths.append(step + 1)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return self.episode_rewards, self.episode_lengths

    def get_optimal_policy(self):
        policy = {}
        for state in range(self.env.n_states):
            policy[state] = np.zeros(self.env.n_actions)
            if state not in self.env.terminal_states:
                policy[state][self.select_action(state, deterministic=True)] = 1
        return policy

    def save(self, path):
        # JSON avoids executable pickle data and is easy for students to inspect.
        with open(path, "w") as f:
            json.dump(
                dict(
                    config=self.config,
                    q_table=self.q_table.tolist(),
                    epsilon=self.epsilon,
                    rewards=self.episode_rewards,
                    lengths=self.episode_lengths,
                ),
                f,
            )

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.__init__(self.env, **data["config"])
        table = np.asarray(data["q_table"])
        if table.shape != self.q_table.shape:
            raise ValueError("checkpoint grid dimensions do not match")
        self.q_table, self.epsilon = table, data["epsilon"]
        self.episode_rewards, self.episode_lengths = data["rewards"], data["lengths"]

    def print_q_values(self):
        print(self.q_table)
