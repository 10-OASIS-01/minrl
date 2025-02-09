"""
MinRL: Minimal, Clean Code for Reinforcement Learning
Created by: 10-OASIS-01
Date: 2025-02-09 06:59:44 UTC

Main package for dynamic robot navigation using reinforcement learning.
This package provides implementations of fundamental RL algorithms and
a customizable GridWorld environment for educational and research purposes.
"""

from .environment import GridWorld
from .agents import (
    PolicyEvaluator,
    PolicyOptimizer,
    QLearningAgent,
    DQNAgent,
    DQNetwork,
    MonteCarloEvaluator,
    MCTSAgent,
    PPOAgent,
    ActorCriticAgent
)
from .utils import Visualizer

__version__ = '1.0.0'
__author__ = '10-OASIS-01'
__created__ = '2025-02-09 06:59:44 UTC'
__description__ = 'MinRL: Minimal Reinforcement Learning Implementation'

__all__ = [
    # Environment
    'GridWorld',
    # Agents
    'PolicyEvaluator',
    'PolicyOptimizer',
    'QLearningAgent',
    'DQNAgent',
    'DQNetwork',
    'MonteCarloEvaluator',
    'MCTSAgent',
    'PPOAgent',
    'ActorCriticAgent',
    # Utils
    'Visualizer',
]