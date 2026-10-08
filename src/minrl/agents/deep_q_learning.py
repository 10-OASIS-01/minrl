"""DQN: collect a transition, sample replay, regress onto a target-network value."""

import numpy as np
import torch
from torch import nn
from .common import as_gym_env, Observations, mlp, set_learning_rate, neural_policy
from .replay import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        env,
        learning_rate=1e-3,
        gamma=0.99,
        hidden_dim=64,
        buffer_size=50000,
        batch_size=64,
        learning_starts=1000,
        train_freq=4,
        target_update_freq=500,
        target_update="hard",
        tau=0.005,
        epsilon=1.0,
        min_epsilon=0.05,
        exploration_steps=20000,
        prioritized=False,
        alpha=0.6,
        beta_start=0.4,
        learning_rate_decay=False,
        seed=0,
    ):
        self.config = {k: v for k, v in locals().items() if k not in ("self", "env")}
        if (
            target_update not in ("hard", "soft")
            or min(batch_size, train_freq, target_update_freq, exploration_steps) < 1
        ):
            raise ValueError("invalid update settings")
        self.env = as_gym_env(env)
        self.observations = Observations(self.env)
        self.gamma, self.batch_size = gamma, batch_size
        self.q_network = mlp(
            self.observations.dimension, self.observations.n_actions, hidden_dim, nn.ReLU
        )
        self.target_network = mlp(
            self.observations.dimension, self.observations.n_actions, hidden_dim, nn.ReLU
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size, seed, prioritized, alpha)
        self.rng = np.random.default_rng(seed)
        self.epsilon = epsilon
        self.history = []

    def select_action(self, observation, deterministic=False, info=None):
        mask = self.observations.mask(observation, info)
        if not deterministic and self.rng.random() < self.epsilon:
            return int(self.rng.choice(np.flatnonzero(mask)))
        with torch.no_grad():
            state = torch.from_numpy(self.observations.encode(observation))
            values = self.q_network(state).masked_fill(~torch.as_tensor(mask), -torch.inf)
            return int(values.argmax())

    def train_step(self, beta=0.4):
        if len(self.replay_buffer) < self.batch_size:
            return {}
        batch, indices, weights = self.replay_buffer.sample(self.batch_size, beta)
        # Each item is (state, action, reward, next_state, terminated, truncated, next_mask).
        states, actions, rewards, next_states, terminated, _, next_masks = zip(*batch)
        states = torch.as_tensor(np.stack(states))
        next_states = torch.as_tensor(np.stack(next_states))
        with torch.no_grad():
            next_q = self.target_network(next_states).masked_fill(
                ~torch.as_tensor(np.stack(next_masks)), -torch.inf
            )
            # Time-limit truncations still bootstrap. True terminal states do not.
            target = (
                torch.tensor(rewards, dtype=torch.float32)
                + self.gamma * (~torch.tensor(terminated)) * next_q.max(dim=1).values
            )
        prediction = self.q_network(states).gather(1, torch.tensor(actions)[:, None]).squeeze(1)
        td_error = target - prediction
        loss = (torch.as_tensor(weights) * td_error.square()).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        if self.replay_buffer.prioritized:
            self.replay_buffer.update_priorities(indices, td_error.detach().numpy())
        if self.config["target_update"] == "soft":
            self.sync_target(self.config["tau"])
        return {"loss": float(loss.detach()), "td_error": float(td_error.detach().abs().mean())}

    def sync_target(self, tau=1.0):
        with torch.no_grad():
            for target, source in zip(
                self.target_network.parameters(), self.q_network.parameters()
            ):
                target.lerp_(source, tau)

    def train(self, total_timesteps=100000, seed=0, callback=None):
        if total_timesteps < 1:
            raise ValueError("total_timesteps must be positive")
        observation, info = self.env.reset(seed=seed)
        episode_return, episode_length = 0.0, 0
        for step in range(1, total_timesteps + 1):
            # 1. Act and observe the environment's response.
            action = self.select_action(observation, info=info)
            nxt, reward, terminated, truncated, next_info = self.env.step(action)
            # 2. Store the transition, including the true final observation.
            self.replay_buffer.add(
                (
                    self.observations.encode(observation),
                    action,
                    reward,
                    self.observations.encode(nxt),
                    terminated,
                    truncated,
                    self.observations.mask(nxt, next_info),
                )
            )
            row = {"step": step}
            # 3. Learn from old transitions instead of only this latest sample.
            set_learning_rate(
                self.optimizer,
                self.config["learning_rate"],
                step - 1,
                total_timesteps,
                self.config["learning_rate_decay"],
            )
            if step >= self.config["learning_starts"] and step % self.config["train_freq"] == 0:
                beta = (
                    self.config["beta_start"]
                    + (1 - self.config["beta_start"]) * step / total_timesteps
                )
                row.update(self.train_step(beta))
            if (
                self.config["target_update"] == "hard"
                and step % self.config["target_update_freq"] == 0
            ):
                self.sync_target()
            fraction = min(1, step / self.config["exploration_steps"])
            self.epsilon = self.config["epsilon"] + fraction * (
                self.config["min_epsilon"] - self.config["epsilon"]
            )
            row["epsilon"] = self.epsilon
            episode_return += reward
            episode_length += 1
            observation, info = nxt, next_info
            if terminated or truncated:
                row.update(episode_return=episode_return, episode_length=episode_length)
                self.history.append(row.copy())
                observation, info = self.env.reset()
                episode_return, episode_length = 0.0, 0
            if callback is not None and callback(self, row) is False:
                break
        return self.history

    def get_optimal_policy(self):
        return neural_policy(self)

    def save(self, path):
        torch.save(
            dict(
                config=self.config,
                q=self.q_network.state_dict(),
                target=self.target_network.state_dict(),
                optimizer=self.optimizer.state_dict(),
                epsilon=self.epsilon,
                history=self.history,
            ),
            path,
        )

    def load(self, path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        self.__init__(self.env, **data["config"])
        self.q_network.load_state_dict(data["q"])
        self.target_network.load_state_dict(data["target"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.epsilon, self.history = data["epsilon"], data["history"]
