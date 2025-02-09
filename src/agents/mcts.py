"""
Monte Carlo Tree Search Implementation
Created by: 10-OASIS-01
Date: 2025-02-09

Implements Monte Carlo Tree Search (MCTS) for decision making in the GridWorld environment.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from ..environment import GridWorld, Action
import math

class MCTSNode:
    """
    Node class for Monte Carlo Tree Search
    """
    def __init__(self, state: int, parent: Optional['MCTSNode'] = None, action: Optional[Action] = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: Dict[Action, 'MCTSNode'] = {}  # action -> node mapping
        self.visits = 0
        self.value = 0.0
        self.untried_actions: List[Action] = []

    def add_child(self, action: Action, state: int) -> 'MCTSNode':
        """Add a child node"""
        child = MCTSNode(state=state, parent=self, action=action)
        self.children[action] = child
        return child

    def is_fully_expanded(self) -> bool:
        """Check if all possible actions have been tried"""
        return len(self.untried_actions) == 0

    def get_ucb_value(self, exploration_constant: float) -> float:
        """Calculate the UCB value for node selection"""
        if self.visits == 0:
            return float('inf')

        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration


class MCTSAgent:
    """
    Monte Carlo Tree Search agent for GridWorld environment
    """
    def __init__(self,
                 env: GridWorld,
                 num_simulations: int = 100,
                 exploration_constant: float = 1.41,
                 max_rollout_steps: int = 100):
        """
        Initialize the MCTS agent.

        Args:
            env (GridWorld): The GridWorld environment
            num_simulations (int): Number of simulations per action selection
            exploration_constant (float): Exploration constant for UCB1
            max_rollout_steps (int): Maximum steps in rollout phase
        """
        self.env = env
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.max_rollout_steps = max_rollout_steps

    def select_action(self, state: int) -> Action:
        """
        Select the best action for the current state using MCTS.

        Args:
            state (int): Current state

        Returns:
            Action: Selected action
        """
        root = MCTSNode(state=state)
        root.untried_actions = self.env.get_valid_actions(state)

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            sim_env = self.env.clone()  # Create a copy of environment for simulation

            # Selection
            while node.is_fully_expanded() and node.children:
                node = self._select_ucb(node)
                sim_env.step(node.action)

            # Expansion
            if not node.is_fully_expanded():
                action = np.random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                next_state, _, _, _ = sim_env.step(action)
                node = node.add_child(action, next_state)

            # Simulation (Rollout)
            cumulative_reward = self._rollout(sim_env)

            # Backpropagation
            while node is not None:
                node.visits += 1
                node.value += cumulative_reward
                node = node.parent

        # Select best action based on highest average value
        return max(root.children.items(),
                  key=lambda x: x[1].value / x[1].visits)[0]

    def _select_ucb(self, node: MCTSNode) -> MCTSNode:
        """Select child node using UCB1"""
        return max(node.children.values(),
                  key=lambda n: n.get_ucb_value(self.exploration_constant))

    def _rollout(self, env: GridWorld) -> float:
        """
        Perform rollout from current state using random policy.

        Returns:
            float: Cumulative reward from rollout
        """
        cumulative_reward = 0.0
        discount = 1.0
        steps = 0
        done = False

        while not done and steps < self.max_rollout_steps:
            valid_actions = env.get_valid_actions()
            action = np.random.choice(valid_actions)
            _, reward, done, _ = env.step(action)
            cumulative_reward += discount * reward
            discount *= 0.99  # discount factor
            steps += 1

        return cumulative_reward

    def get_policy(self, state: int) -> Dict[Action, float]:
        """
        Get the policy (action probabilities) for a state after running MCTS.

        Args:
            state (int): The state to get policy for

        Returns:
            Dict[Action, float]: Mapping from actions to their probabilities
        """
        root = MCTSNode(state=state)
        root.untried_actions = self.env.get_valid_actions(state)

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            sim_env = self.env.clone()

            # Selection
            while node.is_fully_expanded() and node.children:
                node = self._select_ucb(node)
                sim_env.step(node.action)

            # Expansion
            if not node.is_fully_expanded():
                action = np.random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                next_state, _, _, _ = sim_env.step(action)
                node = node.add_child(action, next_state)

            # Simulation
            cumulative_reward = self._rollout(sim_env)

            # Backpropagation
            while node is not None:
                node.visits += 1
                node.value += cumulative_reward
                node = node.parent

        # Convert visit counts to probabilities
        total_visits = sum(child.visits for child in root.children.values())
        policy = {action: child.visits / total_visits
                 for action, child in root.children.items()}

        # Add zero probability for unused actions
        for action in Action:
            if action not in policy:
                policy[action] = 0.0

        return policy