"""Only the small pieces shared by neural agents; learning rules stay in each agent."""

import random
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from ..environment import GridWorld, GridWorldEnv


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def as_gym_env(env):
    return GridWorldEnv(env) if isinstance(env, GridWorld) else env


class Observations:
    """Encode discrete states as one-hot vectors; leave vector observations unchanged."""

    def __init__(self, env):
        self.env = env
        space = env.observation_space
        if not isinstance(env.action_space, gym.spaces.Discrete) or env.action_space.start != 0:
            raise ValueError("only zero-based Discrete actions are supported")
        self.discrete = isinstance(space, gym.spaces.Discrete)
        if self.discrete:
            if space.start != 0:
                raise ValueError("discrete observations must be zero-based")
            self.dimension = space.n
        elif isinstance(space, gym.spaces.Box) and len(space.shape) == 1:
            self.dimension = space.shape[0]
        else:
            raise ValueError("use Discrete or one-dimensional Box observations")
        self.n_actions = env.action_space.n
        self.eye = np.eye(self.dimension, dtype=np.float32) if self.discrete else None

    def encode(self, observation):
        return (
            self.eye[int(observation)].copy()
            if self.discrete
            else np.array(observation, dtype=np.float32, copy=True)
        )

    def mask(self, observation, info=None):
        if info is not None and "action_mask" in info:
            mask = np.asarray(info["action_mask"], dtype=bool)
        elif hasattr(self.env.unwrapped, "action_mask"):
            mask = self.env.unwrapped.action_mask(observation)
        else:
            mask = np.ones(self.n_actions, dtype=bool)
        if mask.shape != (self.n_actions,) or not mask.any():
            raise ValueError("action mask must allow at least one action")
        return mask.copy()


def mlp(input_dim, output_dim, hidden_dim=64, activation=nn.Tanh):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        activation(),
        nn.Linear(hidden_dim, hidden_dim),
        activation(),
        nn.Linear(hidden_dim, output_dim),
    )


def masked_distribution(actor, states, masks):
    logits = actor(states).masked_fill(~masks, -1e9)
    return torch.distributions.Categorical(logits=logits)


def set_learning_rate(optimizer, initial_lr, step, total_steps, decay):
    rate = initial_lr * max(0.0, 1 - step / total_steps) if decay else initial_lr
    for group in optimizer.param_groups:
        group["lr"] = rate


def neural_policy(agent):
    if not agent.observations.discrete:
        raise ValueError("a policy table is only defined for discrete observations")
    grid = getattr(agent.env.unwrapped, "grid", None)
    result = {}
    for state in range(agent.observations.dimension):
        probabilities = np.zeros(agent.observations.n_actions)
        if grid is None or state not in grid.terminal_states:
            probabilities[agent.select_action(state, deterministic=True)] = 1.0
        result[state] = probabilities
    return result
