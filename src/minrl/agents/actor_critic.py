"""One-step actor-critic: the TD error teaches both the actor and the critic."""

import torch
from .common import (
    as_gym_env,
    Observations,
    mlp,
    masked_distribution,
    neural_policy,
    set_learning_rate,
)


class ActorCriticAgent:
    def __init__(
        self,
        env,
        learning_rate=3e-4,
        gamma=0.99,
        entropy_coef=0.01,
        hidden_dim=64,
        learning_rate_decay=False,
    ):
        self.config = {k: v for k, v in locals().items() if k not in ("self", "env")}
        self.env = as_gym_env(env)
        self.observations = Observations(self.env)
        self.actor = mlp(self.observations.dimension, self.observations.n_actions, hidden_dim)
        self.critic = mlp(self.observations.dimension, 1, hidden_dim)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=learning_rate
        )
        self.history = []

    def select_action(self, observation, deterministic=False, info=None):
        with torch.no_grad():
            distribution = masked_distribution(
                self.actor,
                torch.from_numpy(self.observations.encode(observation)),
                torch.as_tensor(self.observations.mask(observation, info)),
            )
            return int(distribution.probs.argmax() if deterministic else distribution.sample())

    def update(self, state, action, reward, next_state, terminated, info=None):
        state_tensor = torch.from_numpy(self.observations.encode(state))
        value = self.critic(state_tensor).squeeze(-1)
        with torch.no_grad():
            next_value = self.critic(
                torch.from_numpy(self.observations.encode(next_state))
            ).squeeze(-1)
            target = reward + self.config["gamma"] * (not terminated) * next_value
        advantage = target - value
        distribution = masked_distribution(
            self.actor, state_tensor, torch.as_tensor(self.observations.mask(state, info))
        )
        policy_loss = -distribution.log_prob(torch.tensor(action)) * advantage.detach()
        value_loss = advantage.square()
        entropy = distribution.entropy()
        loss = policy_loss + 0.5 * value_loss - self.config["entropy_coef"] * entropy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return dict(
            policy_loss=float(policy_loss.detach()),
            value_loss=float(value_loss.detach()),
            entropy=float(entropy.detach()),
        )

    def train(self, total_timesteps=10000, seed=0, callback=None):
        if total_timesteps < 1:
            raise ValueError("total_timesteps must be positive")
        state, info = self.env.reset(seed=seed)
        total, length = 0.0, 0
        for step in range(1, total_timesteps + 1):
            action = self.select_action(state, info=info)
            nxt, reward, terminated, truncated, next_info = self.env.step(action)
            set_learning_rate(
                self.optimizer,
                self.config["learning_rate"],
                step - 1,
                total_timesteps,
                self.config["learning_rate_decay"],
            )
            row = dict(step=step, **self.update(state, action, reward, nxt, terminated, info))
            total, length = total + reward, length + 1
            state, info = nxt, next_info
            if terminated or truncated:
                row.update(episode_return=total, episode_length=length)
                self.history.append(row.copy())
                state, info = self.env.reset()
                total, length = 0.0, 0
            if callback is not None and callback(self, row) is False:
                break
        return self.history

    def get_optimal_policy(self):
        return neural_policy(self)

    def save(self, path):
        torch.save(
            dict(
                config=self.config,
                actor=self.actor.state_dict(),
                critic=self.critic.state_dict(),
                optimizer=self.optimizer.state_dict(),
                history=self.history,
            ),
            path,
        )

    def load(self, path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        self.__init__(self.env, **data["config"])
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.history = data["history"]
