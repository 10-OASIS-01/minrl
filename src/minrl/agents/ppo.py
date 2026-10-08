"""PPO in two phases: collect on-policy data, then optimize the clipped objective."""

import numpy as np
import torch
from torch import nn
from .common import (
    as_gym_env,
    Observations,
    mlp,
    masked_distribution,
    set_learning_rate,
    neural_policy,
)


def compute_gae(rewards, values, next_values, terminated, truncated, gamma=0.99, gae_lambda=0.95):
    """Bootstrap across a time limit, but never propagate GAE into another episode."""
    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (not terminated[t]) * next_values[t] - values[t]
        same_episode = not (terminated[t] or truncated[t])
        gae = delta + gamma * gae_lambda * same_episode * gae
        advantages[t] = gae
    return advantages, advantages + np.asarray(values, dtype=np.float32)


class PPOAgent:
    def __init__(
        self,
        env,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        critic_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        num_epochs=10,
        batch_size=64,
        hidden_dim=64,
        rollout_steps=2048,
        value_clip_ratio=0.2,
        learning_rate_decay=False,
    ):
        self.config = {k: v for k, v in locals().items() if k not in ("self", "env")}
        if min(num_epochs, batch_size, rollout_steps) < 1:
            raise ValueError("epochs, batch size and rollout steps must be positive")
        self.env = as_gym_env(env)
        self.observations = Observations(self.env)
        self.actor = mlp(self.observations.dimension, self.observations.n_actions, hidden_dim)
        self.critic = mlp(self.observations.dimension, 1, hidden_dim)
        # Orthogonal initialization keeps the initial policy close to uniform.
        for network in (self.actor, self.critic):
            for layer in network:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, np.sqrt(2))
                    nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.orthogonal_(self.critic[-1].weight, 1.0)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate,
            eps=1e-5,
        )
        self.memory, self.history = [], []

    def select_action(self, observation, deterministic=False, info=None):
        action, _, _ = self.sample_action(observation, deterministic, info)
        return action

    def sample_action(self, observation, deterministic=False, info=None):
        with torch.no_grad():
            state = torch.from_numpy(self.observations.encode(observation))
            mask = torch.as_tensor(self.observations.mask(observation, info))
            distribution = masked_distribution(self.actor, state, mask)
            action = distribution.probs.argmax() if deterministic else distribution.sample()
            return (
                int(action),
                float(distribution.log_prob(action)),
                float(self.critic(state).squeeze(-1)),
            )

    def update(self):
        if not self.memory:
            return {}
        # Rollout tuples keep the math visible rather than hiding it in a trainer.
        states, actions, rewards, values, next_values, log_probs, ends, cuts, masks = zip(
            *self.memory
        )
        advantages, returns = compute_gae(
            rewards,
            values,
            next_values,
            ends,
            cuts,
            self.config["gamma"],
            self.config["gae_lambda"],
        )
        states = torch.as_tensor(np.stack(states))
        masks = torch.as_tensor(np.stack(masks))
        actions = torch.tensor(actions)
        old_log_probs, old_values = torch.tensor(log_probs), torch.tensor(values)
        advantages, returns = torch.from_numpy(advantages), torch.from_numpy(returns)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        metrics = []
        for _ in range(self.config["num_epochs"]):
            order = torch.randperm(len(states))
            for indices in order.split(self.config["batch_size"]):
                distribution = masked_distribution(self.actor, states[indices], masks[indices])
                new_values = self.critic(states[indices]).squeeze(-1)
                # ratio = pi_new(a|s) / pi_old(a|s), with identical action masks.
                log_ratio = distribution.log_prob(actions[indices]) - old_log_probs[indices]
                ratio = log_ratio.exp()
                unclipped = ratio * advantages[indices]
                clipped = (
                    ratio.clamp(1 - self.config["clip_ratio"], 1 + self.config["clip_ratio"])
                    * advantages[indices]
                )
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                clipped_values = old_values[indices] + (new_values - old_values[indices]).clamp(
                    -self.config["value_clip_ratio"], self.config["value_clip_ratio"]
                )
                value_loss = (
                    0.5
                    * torch.maximum(
                        (new_values - returns[indices]).square(),
                        (clipped_values - returns[indices]).square(),
                    ).mean()
                )
                entropy = distribution.entropy().mean()
                loss = (
                    policy_loss
                    + self.config["critic_loss_coef"] * value_loss
                    - self.config["entropy_coef"] * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.config["max_grad_norm"],
                )
                self.optimizer.step()
                metrics.append(
                    [
                        float(policy_loss.detach()),
                        float(value_loss.detach()),
                        float(entropy.detach()),
                        float(((ratio - 1) - log_ratio).mean().detach()),
                    ]
                )
        self.memory.clear()  # PPO must not reuse trajectories from older policies.
        return dict(
            zip(
                ("policy_loss", "value_loss", "entropy", "approx_kl"),
                np.mean(metrics, axis=0).tolist(),
            )
        )

    def train(self, total_timesteps=100000, seed=0, callback=None):
        if total_timesteps < 1:
            raise ValueError("total_timesteps must be positive")
        observation, info = self.env.reset(seed=seed)
        episode_return, episode_length = 0.0, 0
        for step in range(1, total_timesteps + 1):
            # 1. Collect experience with the current (unchanged) policy.
            action, log_prob, value = self.sample_action(observation, info=info)
            nxt, reward, terminated, truncated, next_info = self.env.step(action)
            with torch.no_grad():
                next_value = float(
                    self.critic(torch.from_numpy(self.observations.encode(nxt))).squeeze(-1)
                )
            self.memory.append(
                (
                    self.observations.encode(observation),
                    action,
                    reward,
                    value,
                    next_value,
                    log_prob,
                    terminated,
                    truncated,
                    self.observations.mask(observation, info),
                )
            )
            episode_return += reward
            episode_length += 1
            row = {"step": step}
            # 2. Compute GAE and update only when a rollout is ready (including the tail).
            if len(self.memory) >= self.config["rollout_steps"] or step == total_timesteps:
                set_learning_rate(
                    self.optimizer,
                    self.config["learning_rate"],
                    step - 1,
                    total_timesteps,
                    self.config["learning_rate_decay"],
                )
                row.update(self.update())
            observation, info = nxt, next_info
            if terminated or truncated:
                row.update(episode_return=episode_return, episode_length=episode_length)
                self.history.append(row.copy())
                observation, info = self.env.reset()
                episode_return, episode_length = 0.0, 0
            if callback is not None and callback(self, row) is False:
                self.update()  # Also consume the partial rollout when a callback stops training.
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
