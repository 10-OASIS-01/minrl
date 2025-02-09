"""
MinRL: Minimal, Clean Code for Reinforcement Learning
Created by: 10-OASIS-01
Date: 2025-02-08 10:38:23 UTC

This module provides implementation examples of various reinforcement learning
algorithms for robot navigation tasks. Each example demonstrates different
approaches to solving navigation problems using reinforcement learning techniques.

Available Examples:
    - Actor-Critic: Implementation of the Actor-Critic algorithm
    - Deep Q-Learning: Deep Q-Network (DQN) implementation
    - Monte Carlo Tree Search (MCTS): Tree search based planning
    - Monte Carlo Methods: Monte Carlo policy evaluation and control
    - Proximal Policy Optimization (PPO): Implementation of PPO algorithm
    - Q-Learning: Traditional Q-learning implementation
    - Value Iteration: Dynamic programming approach for optimal policies

Each example can be run independently and includes detailed documentation
on its implementation and usage.
"""

from . import (
    actor_critic,
    deep_qL,
    mcts,
    monte_carlo,
    ppo,
    q_learning,
    value_iteration
)

# Make commonly used classes and functions available at package level
from .actor_critic import ActorCritic
from .deep_qL import DeepQLearning
from .mcts import MCTS
from .monte_carlo import MonteCarlo
from .ppo import PPO
from .q_learning import QLearning
from .value_iteration import ValueIteration

__all__ = [
    'actor_critic',
    'deep_qL',
    'mcts',
    'monte_carlo',
    'ppo',
    'q_learning',
    'value_iteration',
    'ActorCritic',
    'DeepQLearning',
    'MCTS',
    'MonteCarlo',
    'PPO',
    'QLearning',
    'ValueIteration',
]

__version__ = '1.0.0'