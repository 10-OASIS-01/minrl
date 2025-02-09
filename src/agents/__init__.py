"""
Agents Module
Created by: 10-OASIS-01
Date: 2025-02-09 06:59:44 UTC

Contains implementations of various reinforcement learning agents:
- Policy-based methods (Policy Evaluation and Optimization)
- Value-based methods (Q-Learning, Deep Q-Network)
- Monte Carlo methods (MC Evaluation, MCTS)
"""

from .policy_evaluation import PolicyEvaluator
from .policy_optimization import PolicyOptimizer
from .q_learning import QLearningAgent
from .deep_q_learning import DQNetwork, DQNAgent
from .monte_carlo import MonteCarloEvaluator
from .mcts import MCTSAgent
# from .ppo import PPOAgent
from .actor_critic import ActorCriticAgent

__all__ = [
    # Policy-based methods
    'PolicyEvaluator',
    'PolicyOptimizer',
    # Value-based methods
    'QLearningAgent',
    'DQNetwork',
    'DQNAgent',
    # Monte Carlo methods
    'MonteCarloEvaluator',
    'MCTSAgent',
    # Actor-Critic methods
    # 'PPOAgent',
    'ActorCriticAgent',
]