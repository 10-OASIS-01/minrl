"""Optional replay extensions. Start with uniform replay before studying PER."""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=50000, seed=0, prioritized=False, alpha=0.6):
        if capacity < 1 or not 0 <= alpha <= 1:
            raise ValueError("invalid replay capacity or alpha")
        self.capacity, self.prioritized, self.alpha = capacity, prioritized, alpha
        self.rng = np.random.default_rng(seed)
        self.data = []
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.position = 0

    def add(self, transition):
        priority = self.priorities[: len(self.data)].max() if self.data else 1.0
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.position] = transition
        self.priorities[self.position] = max(priority, 1e-6)
        self.position = (self.position + 1) % self.capacity

    def probabilities(self):
        if not self.data:
            raise ValueError("cannot sample empty replay")
        priorities = self.priorities[: len(self.data)] ** self.alpha
        return (
            priorities / priorities.sum()
            if self.prioritized
            else np.full(len(self.data), 1 / len(self.data))
        )

    def sample(self, batch_size, beta=0.4):
        if not self.data:
            raise ValueError("cannot sample empty replay")
        if not self.prioritized:
            indices = self.rng.integers(len(self.data), size=batch_size)
            return [self.data[i] for i in indices], indices, np.ones(batch_size, dtype=np.float32)
        probabilities = self.probabilities()
        indices = self.rng.choice(len(self.data), batch_size, replace=True, p=probabilities)
        # Correct the bias introduced by prioritized sampling; weights <= 1.
        weights = (len(self.data) * probabilities[indices]) ** (-beta)
        weights /= (len(self.data) * probabilities.min()) ** (-beta)
        return [self.data[i] for i in indices], indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        # For repeated sampled indices, keep the largest error, independent of order.
        for index in np.unique(indices):
            self.priorities[index] = np.max(np.abs(np.asarray(td_errors)[indices == index])) + 1e-6

    def __len__(self):
        return len(self.data)
