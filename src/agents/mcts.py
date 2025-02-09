"""
Monte Carlo Tree Search (MCTS) Implementation
Created by: 10-OASIS-01
Date: 2025-02-09 05:22:17 UTC

Implements the MCTS algorithm for the GridWorld environment.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import math

class Node:
    def __init__(self, state: int, parent: Optional['Node'] = None):
        self.state = state
        self.parent = parent
        self.children: Dict[int, 'Node'] = {}  # action -> Node
        self.visits = 0
        self.value = 0.0
        self.untried_actions: List[int] = []

class MCTSAgent:
    def __init__(self, env, exploration_weight: float = 1.4, n_simulations: int = 100):
        """
        Initialize the MCTS agent.
        
        Args:
            env: The GridWorld environment
            exploration_weight: The exploration constant (c) in UCT formula
            n_simulations: Number of simulations per action selection
        """
        self.env = env
        self.exploration_weight = exploration_weight
        self.n_simulations = n_simulations
        self.root: Optional[Node] = None
    
    def select_action(self, state: int) -> int:
        """Select the best action using MCTS"""
        self.root = Node(state)
        self.root.untried_actions = list(range(self.env.action_dim))
        
        # Run simulations
        for _ in range(self.n_simulations):
            node = self.root
            sim_env = self.env.clone()  # Create a copy of environment for simulation
            
            # Selection
            while not node.untried_actions and node.children:
                node = self._select_uct(node)
                
            # Expansion
            if node.untried_actions:
                action = np.random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                
                next_state, _, done, _ = sim_env.step(action)
                child = Node(next_state, parent=node)
                child.untried_actions = [] if done else list(range(self.env.action_dim))
                node.children[action] = child
                node = child
            
            # Simulation (Rollout)
            value = self._rollout(sim_env)
            
            # Backpropagation
            while node:
                node.visits += 1
                node.value += value
                node = node.parent
        
        # Select best action from root
        return max(self.root.children.items(),
                  key=lambda x: x[1].value / (x[1].visits + 1e-6))[0]
    
    def _select_uct(self, node: Node) -> Node:
        """Select child node using Upper Confidence Bound for Trees (UCT)"""
        log_parent_visits = math.log(node.visits)
        
        def uct_score(action: int, child: Node) -> float:
            exploitation = child.value / (child.visits + 1e-6)
            exploration = self.exploration_weight * math.sqrt(
                log_parent_visits / (child.visits + 1e-6)
            )
            return exploitation + exploration
        
        return max(node.children.items(),
                  key=lambda x: uct_score(x[0], x[1]))[1]
    
    def _rollout(self, env) -> float:
        """Perform a random rollout from the current state"""
        done = False
        total_reward = 0
        max_steps = self.env.size * 4  # Reasonable maximum steps for rollout
        step = 0
        
        while not done and step < max_steps:
            action = np.random.randint(env.action_dim)
            _, reward, done, _ = env.step(action)
            total_reward += reward
            step += 1
        
        return total_reward
    
    def get_optimal_policy(self) -> Dict[int, List[float]]:
        """Return the optimal policy based on visited states"""
        policy = {}
        
        def get_state_policy(node: Node) -> List[float]:
            actions_value = [0.0] * self.env.action_dim
            total_visits = sum(child.visits for child in node.children.values())
            
            for action, child in node.children.items():
                actions_value[action] = child.visits / (total_visits + 1e-6)
            
            return actions_value
        
        def traverse(node: Node):
            if node:
                policy[node.state] = get_state_policy(node)
                for child in node.children.values():
                    traverse(child)
        
        traverse(self.root)
        return policy
