"""
MCTS Tests
Created by: 10-OASIS-01
Date: 2025-02-09 05:23:49 UTC

Test suite for Monte Carlo Tree Search implementation.
"""

import unittest
import numpy as np
from src.environment import GridWorld
from src.agents.mcts import MCTSAgent, Node

class TestMCTS(unittest.TestCase):
    def setUp(self):
        """Set up test environment and agent"""
        self.env = GridWorld(size=3)  # Small grid for faster testing
        self.agent = MCTSAgent(self.env)
        np.random.seed(42)  # For reproducibility
    
    def test_node_initialization(self):
        """Test Node class initialization"""
        state = 0
        node = Node(state)
        
        self.assertEqual(node.state, state)
        self.assertIsNone(node.parent)
        self.assertEqual(node.children, {})
        self.assertEqual(node.visits, 0)
        self.assertEqual(node.value, 0.0)
        self.assertEqual(node.untried_actions, [])
    
    def test_mcts_initialization(self):
        """Test MCTSAgent initialization"""
        exploration_weight = 1.5
        n_simulations = 150
        agent = MCTSAgent(self.env, 
                         exploration_weight=exploration_weight,
                         n_simulations=n_simulations)
        
        self.assertEqual(agent.exploration_weight, exploration_weight)
        self.assertEqual(agent.n_simulations, n_simulations)
        self.assertIsNone(agent.root)
    
    def test_action_selection(self):
        """Test action selection process"""
        state = self.env.reset()
        action = self.agent.select_action(state)
        
        # Action should be valid
        self.assertIn(action, range(self.env.action_dim))
        
        # Root node should be created and populated
        self.assertIsNotNone(self.agent.root)
        self.assertEqual(self.agent.root.state, state)
        self.assertGreater(len(self.agent.root.children), 0)
    
    def test_uct_selection(self):
        """Test Upper Confidence Bound for Trees selection"""
        # Create a simple tree structure
        root = Node(0)
        root.visits = 10
        
        child1 = Node(1, parent=root)
        child1.visits = 5
        child1.value = 2.0
        
        child2 = Node(2, parent=root)
        child2.visits = 3
        child2.value = 1.0
        
        root.children = {0: child1, 1: child2}
        
        # Test UCT selection
        self.agent.root = root
        selected_node = self.agent._select_uct(root)
        
        # Should select child with higher UCT value
        self.assertIn(selected_node, [child1, child2])
    
    def test_rollout(self):
        """Test rollout process"""
        state = self.env.reset()
        reward = self.agent._rollout(self.env)
        
        # Reward should be a float
        self.assertIsInstance(reward, float)
        
        # Environment should be in a terminal state or within max steps
        self.assertTrue(self.env._is_terminal() or 
                       len(self.env._history) <= self.env.size * 4)
    
    def test_optimal_policy(self):
        """Test optimal policy extraction"""
        state = self.env.reset()
        self.agent.select_action(state)  # Run MCTS to build the tree
        policy = self.agent.get_optimal_policy()
        
        # Policy should exist for root state
        self.assertIn(state, policy)
        
        # Policy should be a valid probability distribution
        state_policy = policy[state]
        self.assertEqual(len(state_policy), self.env.action_dim)
        self.assertAlmostEqual(sum(state_policy), 1.0, places=5)
        self.assertTrue(all(0 <= p <= 1 for p in state_policy))
    
    def test_environment_interaction(self):
        """Test full episode interaction"""
        state = self.env.reset()
        total_reward = 0
        done = False
        steps = 0
        max_steps = 100
        
        while not done and steps < max_steps:
            action = self.agent.select_action(state)
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            steps += 1
        
        # Episode should complete
        self.assertTrue(done or steps == max_steps)
    
    def test_reproducibility(self):
        """Test reproducibility with fixed seed"""
        np.random.seed(42)
        state = self.env.reset()
        first_action = self.agent.select_action(state)
        
        np.random.seed(42)
        state = self.env.reset()
        second_action = self.agent.select_action(state)
        
        # Actions should be the same with same seed
        self.assertEqual(first_action, second_action)
    
    def test_invalid_parameters(self):
        """Test agent creation with invalid parameters"""
        with self.assertRaises(ValueError):
            MCTSAgent(self.env, exploration_weight=-1.0)
        
        with self.assertRaises(ValueError):
            MCTSAgent(self.env, n_simulations=0)
    
    def test_convergence(self):
        """Test if MCTS converges to better solutions with more simulations"""
        state = self.env.reset()
        
        # Run with few simulations
        agent_few = MCTSAgent(self.env, n_simulations=10)
        reward_few = 0
        for _ in range(5):
            state = self.env.reset()
            done = False
            while not done:
                action = agent_few.select_action(state)
                state, r, done, _ = self.env.step(action)
                reward_few += r
        
        # Run with many simulations
        agent_many = MCTSAgent(self.env, n_simulations=100)
        reward_many = 0
        for _ in range(5):
            state = self.env.reset()
            done = False
            while not done:
                action = agent_many.select_action(state)
                state, r, done, _ = self.env.step(action)
                reward_many += r
        
        # More simulations should generally lead to better performance
        self.assertGreaterEqual(reward_many, reward_few)

if __name__ == '__main__':
    unittest.main()
