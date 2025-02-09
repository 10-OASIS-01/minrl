# MinRL: Minimal, Clean Code for Reinforcement Learning

MinRL provides clean, minimal implementations of fundamental reinforcement learning algorithms in a customizable GridWorld environment. The project focuses on educational clarity and implementation simplicity while maintaining production-quality code standards.

## 🌟 Key Features

- **Modular GridWorld Environment**: Customizable grid sizes (3×3, 5×5) with configurable rewards and dynamics
- **Clean Implementation** of core RL algorithms:
  - Policy Evaluation (Bellman Expectation)
  - Monte Carlo Methods (First-visit and Every-visit)
  - Monte Carlo Tree Search (MCTS)
  - Policy Iteration & Value Iteration
  - Tabular Q-Learning
  - Deep Q-Learning (DQN)
  - Actor-Critic Methods
  - Proximal Policy Optimization (PPO)
- **Visualization Tools**: Built-in plotting and state-value visualization
- **Comprehensive Tests**: 100% test coverage with pytest
- **Educational Focus**: Well-documented code with step-by-step examples

## 📁 Project Structure

```
minrl/
├── src/
│   ├── environment/       
│   │   └── grid_world.py  # Core grid world logic, state transitions, rewards, and environment dynamics
│   ├── agents/           
│   │   ├── policy_evaluation.py  # Implements Bellman Expectation for policy evaluation
│   │   ├── monte_carlo.py  # Monte Carlo methods for policy evaluation
│   │   ├── mcts.py  # Monte Carlo Tree Search implementation
│   │   ├── policy_optimization.py  # Policy iteration and value iteration implementations
│   │   ├── q_learning.py  # Tabular Q-Learning algorithm
│   │   ├── deep_q_learning.py  # Deep Q-Learning (DQN) implementation using neural networks
│   │   ├── actor_critic.py  # Actor-Critic implementation with separate networks
│   │   └── ppo.py  # Proximal Policy Optimization implementation
│   └── utils/             
│       └── visualization.py  # Visualizes state values, learned policies, and rewards over episodes
├── tests/  # Comprehensive test suite ensuring correct functionality
├── examples/              
│   ├── basic_navigation.py  # Basic navigation example using a static GridWorld
│   ├── run_experiments.py  # Runs experiments with all implemented RL algorithms
│   ├── mcts_example.py  # Monte Carlo Tree Search example
│   ├── actor_critic_example.py  # Actor-Critic implementation example
│   └── ppo_example.py  # PPO implementation example
└── docs/  # Implementation Guide for Beginners
```
## 🎓 Implemented Algorithms

1. **Policy-based Methods**
- `PolicyEvaluator`: Implements policy evaluation using the Bellman expectation equation
- `PolicyOptimizer`: Implements both Policy Iteration and Value Iteration algorithms
  - Policy Iteration combines policy evaluation and policy improvement
  - Value Iteration uses the Bellman optimality equation

2. **Value-based Methods**
- `QLearningAgent`: Implements tabular Q-learning with ε-greedy exploration
  - Features: Experience replay, ε-greedy exploration with decay
  - Maintains a Q-table for state-action values
  
- `DQNAgent`: Implements Deep Q-Network (DQN) with several modern improvements
  - Features:
    - Neural network function approximation
    - Experience replay buffer
    - Target network for stability
    - ε-greedy exploration with decay
    - Batch training
    - Prioritized experience replay

3. **Monte Carlo Methods**
- `MonteCarloEvaluator`: Implements Monte Carlo policy evaluation
  - Supports both first-visit and every-visit MC methods
  - Model-free learning from episodes
  
- `MCTSAgent`: Implements Monte Carlo Tree Search
  - Features:
    - UCT (Upper Confidence Bound for Trees) selection
    - Tree expansion and backpropagation
    - Random rollout simulations
    - Exploration parameter tuning
    - Optimal policy extraction

4. **Actor-Critic Methods**
- `ActorCriticAgent`: Implements the Actor-Critic architecture
  - Separate networks for policy (actor) and value function (critic)
  - Policy gradient with baseline
  - Value function approximation
  
- `PPOAgent`: Implements Proximal Policy Optimization (PPO)
  - Features:
    - Clipped surrogate objective
    - Combined actor-critic architecture
    - Generalized Advantage Estimation (GAE)
    - Value function clipping
    - Entropy regularization
    - Mini-batch training

## 🛠️ Dependencies

```bash
- Python 3.7+
- NumPy >= 1.19.0
- PyTorch >= 1.8.0
- Matplotlib >= 3.3.0
- Seaborn >= 0.11.0
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/10-OASIS-01/minrl.git
cd minrl

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install the package
pip install -e .
```

### **Basic Usage**

Here's how you can start using the basic functionality of the library:

```python
from minrl.environment import GridWorld
from minrl.agents import DQNAgent, ActorCriticAgent, PPOAgent

# Create a 3x3 GridWorld environment
env = GridWorld(size=3)

# Example 1: Deep Q-Learning
dqn_agent = DQNAgent(env)
dqn_rewards = dqn_agent.train(episodes=1000)
dqn_agent.visualize_policy()

# Example 2: Actor-Critic
ac_agent = ActorCriticAgent(env)
ac_rewards = ac_agent.train(episodes=1000)
ac_agent.visualize_policy()

# Example 3: PPO
ppo_agent = PPOAgent(env)
ppo_rewards = ppo_agent.train(episodes=1000)
ppo_agent.visualize_policy()
```

### **Run Advanced Policy Gradient Examples**

```bash
# Run the Actor-Critic example
python -m minrl.examples.actor_critic_example

# Run the PPO example
python -m minrl.examples.ppo_example

# Run the Deep Q Learning. example      
python -m minrl.examples.basic_navigation
```

## 🤝 Contributing

Contributions are welcome! MinRL aims to be an educational and clean implementation of RL algorithms. Before submitting your contribution, please consider the following guidelines:

### Getting Started
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Priority Areas for Contribution

#### 1. Environment Enhancements
- **Monte Carlo-Friendly Environments**:
  - Implementing intermediate rewards to address sparse reward issues
  - Adjusting reward structures to balance exploration-exploitation
  - Adding state-dependent reward scaling
  - Introducing progressive difficulty levels
  - Implementing curriculum learning support
  - Adding option for reversible terminal states

- **Advanced Environment Features**:
  - Adding support for partially observable states (POMDP)
  - Implementing more complex reward structures: Sparse rewards and Multi-objective rewards
  - Adding support for continuous action spaces
  - Implementing dynamic obstacles
  - Custom reward shaping tools for different learning algorithms

#### 2. Project Structure Improvements
- Improving code modularity and reusability
- Enhancing documentation with theory explanations
- Streamlining configuration management
- Adding new examples and tutorials
- Implementing logging and experiment tracking

#### 3. New RL Algorithm Implementations  
- **Model-Based Methods**: Dyna-Q, Prioritized Sweeping, PILCO (Probabilistic Inference for Learning Control)
- **Multi-Agent RL**: Independent Q-Learning, MADDPG (Multi-Agent DDPG)
- **Hierarchical RL**: Options Framework, MAXQ

#### 4. Visualization Enhancements

### Contribution Guidelines
1. Follow the existing code style and project structure
2. Add comprehensive tests for new features
3. Update documentation accordingly
4. Ensure all tests pass before submitting PR
5. Include example usage in docstrings
6. Add relevant citations for implemented algorithms

### Code Quality Requirements
- Maintain clean, readable code
- Include type hints
- Follow PEP 8 guidelines
- Achieve 100% test coverage for new code
- Add detailed docstrings

For major changes, please open an issue first to discuss what you would like to change. This ensures your time is well spent and your contribution aligns with the project's goals.

## ✨ Acknowledgments

- Inspired by [simple_rl](https://github.com/david-abel/simple_rl), [CleanRL](https://github.com/vwxyzjn/cleanrl) and [RLCode](https://github.com/rlcode/reinforcement-learning)

- Special thanks to Professor Shiyu Zhao for his insightful course on the "Mathematical Foundations of Reinforcement Learning," which provided a solid foundation for my understanding of reinforcement learning. The course materials, including the textbook, PPTs, and code, can be found on his [GitHub repository](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning), and the English [lecture videos](https://www.youtube.com/playlist?list=PLEhdbSEZZbDaFWPX4gehhwB9vJZJ1DNm8) and Chinese [lecture videos](https://space.bilibili.com/2044042934/lists/748665?type=season) are available online.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Contact

Created by: Yibin Liu  

Email: [yibin.leon.liu@outlook.com](yibin.leon.liu@outlook.com)  

Last Updated: 2025-02-09 05:00:14
