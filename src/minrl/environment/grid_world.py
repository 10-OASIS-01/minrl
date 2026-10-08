"""A deterministic grid: rewards are paid on entry, terminal values are zero."""

from enum import IntEnum
import copy

import gymnasium as gym
import numpy as np


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class GridWorld:
    def __init__(
        self, size=3, random_seed=None, start_state=0, terminal_states=None, step_reward=-0.1
    ):
        if not isinstance(size, int) or size < 3:
            raise ValueError("size must be an integer >= 3")
        self.size, self.n_states, self.n_actions = size, size * size, 4
        self.start_state = start_state
        self.terminal_states = (
            dict(terminal_states)
            if terminal_states is not None
            else {size - 1: -1.0, size * (size - 1): -1.0, size * size - 1: 1.0}
        )
        self.step_reward = float(step_reward)
        self.rng = np.random.default_rng(random_seed)
        self.action_effects = {
            Action.UP: (-1, 0),
            Action.RIGHT: (0, 1),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
        }
        self.reset()

    def _check_state(self, state):
        if not isinstance(state, (int, np.integer)) or not 0 <= state < self.n_states:
            raise ValueError(f"invalid state: {state}")

    def _pos_to_state(self, pos):
        row, col = pos
        if not 0 <= row < self.size or not 0 <= col < self.size:
            raise ValueError("position outside grid")
        return row * self.size + col

    def _state_to_pos(self, state):
        self._check_state(state)
        return divmod(int(state), self.size)

    def reset(self):
        self._check_state(self.start_state)
        for state, reward in self.terminal_states.items():
            self._check_state(state)
            if not np.isfinite(reward):
                raise ValueError("rewards must be finite")
        if self.start_state in self.terminal_states:
            raise ValueError("start_state must not be terminal")
        if not np.isfinite(self.step_reward):
            raise ValueError("step_reward must be finite")
        self.current_pos = self._state_to_pos(self.start_state)
        return self.start_state

    def transition(self, state, action):
        """Return (next_state, reward, terminated), without changing current_pos."""
        row, col = self._state_to_pos(state)
        dr, dc = self.action_effects[Action(action)]
        if state in self.terminal_states:
            return int(state), 0.0, True
        pos = (max(0, min(row + dr, self.size - 1)), max(0, min(col + dc, self.size - 1)))
        nxt = self._pos_to_state(pos)
        return (
            nxt,
            float(self.terminal_states.get(nxt, self.step_reward)),
            nxt in self.terminal_states,
        )

    def step(self, action):
        nxt, reward, done = self.transition(self._pos_to_state(self.current_pos), action)
        self.current_pos = self._state_to_pos(nxt)
        return nxt, reward, done, {}

    def get_valid_actions(self, state=None):
        state = self._pos_to_state(self.current_pos) if state is None else state
        row, col = self._state_to_pos(state)
        if state in self.terminal_states:
            return []
        return [
            a
            for a, (dr, dc) in self.action_effects.items()
            if 0 <= row + dr < self.size and 0 <= col + dc < self.size
        ]

    def clone(self):
        return copy.deepcopy(self)

    def get_state_space_size(self):
        return self.n_states

    def get_action_space_size(self):
        return self.n_actions

    def render(self):
        return "\n".join(
            " ".join(
                "A"
                if (r, c) == self.current_pos
                else "T"
                if r * self.size + c in self.terminal_states
                else "."
                for c in range(self.size)
            )
            for r in range(self.size)
        )


class GridWorldEnv(gym.Env):
    """Thin Gymnasium adapter. Reaching the step limit truncates the episode."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, grid=None, max_episode_steps=100, render_mode=None, **kwargs):
        self.grid = grid if grid is not None else GridWorld(**kwargs)
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Discrete(self.grid.n_states)
        self.action_space = gym.spaces.Discrete(4)
        self.render_mode = render_mode
        self.elapsed_steps = 0

    def action_mask(self, state):
        mask = np.zeros(4, dtype=bool)
        mask[self.grid.get_valid_actions(int(state))] = True
        # The mask is immaterial after termination; avoid all-infinite logits.
        if not mask.any():
            mask[:] = True
        return mask

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.elapsed_steps = 0
        state = self.grid.reset()
        return state, {"action_mask": self.action_mask(state)}

    def step(self, action):
        state, reward, terminated, _ = self.grid.step(action)
        self.elapsed_steps += 1
        truncated = self.elapsed_steps >= self.max_episode_steps and not terminated
        return state, reward, terminated, truncated, {"action_mask": self.action_mask(state)}

    def render(self):
        return self.grid.render()
