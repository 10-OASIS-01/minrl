"""
Monte Carlo Tree Search (MCTS) Implementation
Created by: 10-OASIS-01
Date: 2025-02-09 06:35:38 UTC

Implements the MCTS algorithm for the GridWorld environment.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import math
from ..environment.grid_world import GridWorld, Action


class Node:
    """A node in the MCTS tree"""
    def __init__(self, state: int, parent: Optional['Node'] = None):
        self.state = state
        self.parent = parent
        self.children: Dict[int, 'Node'] = {}  # action -> Node
        self.visits = 0
        self.value = 0.0
        self.untried_actions: List[int] = []


class MCTSAgent:
    """Monte Carlo Tree Search Agent with integrated experiment capabilities"""
    def __init__(self, env: GridWorld, exploration_weight: float = 1.4, n_simulations: int = 100):
        """
        Initialize the MCTS agent.
        
        Args:
            env (GridWorld): The GridWorld environment
            exploration_weight (float): The exploration constant (c) in UCT formula
            n_simulations (int): Number of simulations per action selection
        """
        self.env = env
        self.exploration_weight = exploration_weight
        self.n_simulations = n_simulations
        self.root: Optional[Node] = None

    def select_action(self, state: int) -> int:
        """
        Select the best action using MCTS

        Args:
            state (int): Current state

        Returns:
            int: Selected action
        """
        self.root = Node(state)
        self.root.untried_actions = [action.value for action in Action]

        # Run simulations
        for _ in range(self.n_simulations):
            node = self.root
            sim_env = self.env.clone()
            sim_env.current_pos = self.env._state_to_pos(state)  # Set correct initial state
            current_state = state
            done = False

            # Selection
            while not node.untried_actions and node.children and not done:
                node = self._select_uct(node)
                action = next(a for a, n in node.parent.children.items() if n == node)
                current_state, _, done, _ = sim_env.step(action)

            # Expansion
            if not done and node.untried_actions:
                action = np.random.choice(node.untried_actions)
                node.untried_actions.remove(action)

                next_state, _, done, _ = sim_env.step(action)
                child = Node(next_state, parent=node)
                child.untried_actions = [] if done else [action.value for action in Action]
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
        if not self.root.children:
            # If no children were created (e.g., all simulations ended immediately),
            # return a random action
            return np.random.choice([action.value for action in Action])

        return max(self.root.children.items(),
                   key=lambda x: x[1].value / (x[1].visits + 1e-6))[0]

    def _select_uct(self, node: Node) -> Node:
        """
        Select child node using Upper Confidence Bound for Trees (UCT)

        Args:
            node (Node): Current node

        Returns:
            Node: Selected child node
        """
        log_parent_visits = math.log(node.visits + 1e-6)  # Add small epsilon to avoid log(0)

        def uct_score(action: int, child: Node) -> float:
            exploitation = child.value / (child.visits + 1e-6)
            exploration = self.exploration_weight * math.sqrt(
                log_parent_visits / (child.visits + 1e-6)
            )
            return exploitation + exploration

        return max(node.children.items(),
                   key=lambda x: uct_score(x[0], x[1]))[1]

    def _rollout(self, env: GridWorld) -> float:
        """
        Perform a random rollout from the current state

        Args:
            env (GridWorld): Environment for simulation

        Returns:
            float: Total reward from rollout
        """
        done = False
        total_reward = 0
        max_steps = self.env.size * 4  # Reasonable maximum steps for rollout
        step = 0

        while not done and step < max_steps:
            valid_actions = env.get_valid_actions()  # Get valid actions for current state
            action = np.random.choice([action.value for action in valid_actions])
            _, reward, done, _ = env.step(action)
            total_reward += reward
            step += 1

        return total_reward
    
    def run_episode(self) -> Tuple[List[int], List[int], List[float]]:
        """
        Run a single episode with MCTS
        
        Returns:
            Tuple[List[int], List[int], List[float]]: States, actions, and rewards
        """
        states, actions, rewards = [], [], []
        state = self.env.reset()
        done = False
        
        while not done:
            states.append(state)
            action = self.select_action(state)
            actions.append(action)
            state, reward, done, _ = self.env.step(action)
            rewards.append(reward)
        
        return states, actions, rewards
    
    def run_experiments(self,
                       n_episodes: int = 20,
                       reset_tree: bool = True) -> Dict[str, List[float]]:
        """
        Run multiple episodes and collect performance metrics
        
        Args:
            n_episodes (int): Number of episodes to run
            reset_tree (bool): Whether to reset the search tree between episodes
            
        Returns:
            Dict[str, List[float]]: Dictionary containing episode rewards and lengths
        """
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(n_episodes):
            if reset_tree:
                self.root = None
            states, actions, rewards = self.run_episode()
            episode_rewards.append(sum(rewards))
            episode_lengths.append(len(states))
        
        return {
            'rewards': episode_rewards,
            'lengths': episode_lengths,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths)
        }
    
    def parameter_search(self,
                        simulation_counts: List[int] = [50, 100, 200],
                        exploration_weights: List[float] = [1.0, 1.4, 2.0],
                        episodes_per_config: int = 20) -> Dict:
        """
        Search over different parameter configurations
        
        Args:
            simulation_counts (List[int]): Different numbers of simulations to try
            exploration_weights (List[float]): Different exploration weights to try
            episodes_per_config (int): Number of episodes per configuration
            
        Returns:
            Dict: Results for each parameter configuration
        """
        results = {}
        original_sims = self.n_simulations
        original_weight = self.exploration_weight
        
        try:
            for sims in simulation_counts:
                for weight in exploration_weights:
                    self.n_simulations = sims
                    self.exploration_weight = weight
                    
                    result = self.run_experiments(n_episodes=episodes_per_config)
                    results[(sims, weight)] = result
        
        finally:
            # Restore original parameters
            self.n_simulations = original_sims
            self.exploration_weight = original_weight
        
        return results

    def get_optimal_policy(self) -> Dict[int, List[float]]:
        """
        Return the optimal policy based on visited states

        Returns:
            Dict[int, List[float]]: Mapping from states to action probabilities
        """
        policy = {}
        n_actions = self.env.get_action_space_size()

        # Initialize policy with uniform distribution for all states
        for state in range(self.env.get_state_space_size()):
            policy[state] = [1.0 / n_actions] * n_actions

        def get_state_policy(node: Node) -> List[float]:
            if not node.children:
                # Return uniform policy for unvisited states
                return [1.0 / n_actions] * n_actions

            actions_value = [0.0] * n_actions
            total_visits = sum(child.visits for child in node.children.values())

            if total_visits == 0:
                return [1.0 / n_actions] * n_actions

            for action, child in node.children.items():
                actions_value[action] = child.visits / total_visits

            return actions_value

        def traverse(node: Node):
            if node:
                policy[node.state] = get_state_policy(node)
                for child in node.children.values():
                    traverse(child)

        if self.root:
            traverse(self.root)

        # Handle terminal states with uniform policy
        for terminal_state in self.env.terminal_states:
            policy[terminal_state] = [1.0 / n_actions] * n_actions

        return policy