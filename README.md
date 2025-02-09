# MinRL: Minimal, Clean Code for Reinforcement Learning

MinRL provides clean, minimal implementations of fundamental reinforcement learning algorithms in a customizable GridWorld environment. The project focuses on educational clarity and implementation simplicity while maintaining production-quality code standards.

## 🌟 Key Features

- **Modular GridWorld Environment**: Customizable grid sizes (3×3, 5×5) with configurable rewards and dynamics
- **Clean Implementation** of core RL algorithms:
  - Policy Evaluation (Bellman Expectation)
  - Monte Carlo Methods (First-visit and Every-visit)
  - Policy Iteration & Value Iteration
  - Tabular Q-Learning
  - Deep Q-Learning (DQN)
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
│   │   ├── policy_optimization.py  # Policy iteration and value iteration implementations
│   │   ├── q_learning.py  # Tabular Q-Learning algorithm
│   │   └── deep_q_learning.py  # Deep Q-Learning (DQN) implementation using neural networks
│   └── utils/             
│       └── visualization.py  # Visualizes state values, learned policies, and rewards over episodes
├── tests/  # Comprehensive test suite ensuring correct functionality
├── examples/              
│   ├── basic_navigation.py  # Basic navigation example using a static GridWorld
│   └── run_experiments.py  # Runs experiments with all implemented RL algorithms
└── docs/  # Implementation Guide for Beginners
```

## 🎓 Implemented Algorithms

1. **Policy Evaluation**
   - Implements Bellman expectation equation
   - Iterative value computation
   - Convergence validation

2. **Monte Carlo Methods**
   - First-visit Monte Carlo evaluation
   - Every-visit Monte Carlo evaluation
   - Episode-based learning
   - Model-free approach

3. **Policy/Value Iteration**
   - Value iteration with Bellman optimality
   - Policy iteration (evaluation + improvement)
   - Optimal policy extraction

4. **Q-Learning**
   - Tabular state-action value learning
   - ε-greedy exploration
   - Hyperparameter tuning support

5. **Deep Q-Learning (DQN)**
   - Neural network function approximation
   - Experience replay memory
   - Target network implementation
   - PyTorch backend

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
from minrl.agents import DQNAgent, MonteCarloEvaluator

# Create a 3x3 GridWorld environment
env = GridWorld(size=3)

# Example 1: Deep Q-Learning
dqn_agent = DQNAgent(env)
rewards = dqn_agent.train(episodes=1000)
dqn_agent.visualize_policy()

# Example 2: Monte Carlo Policy Evaluation
mc_evaluator = MonteCarloEvaluator(env, gamma=0.99)
policy = create_random_policy(env)  # Create a random policy
state_values = mc_evaluator.evaluate_policy(
    policy,
    num_episodes=1000,
    first_visit=True  # Use first-visit Monte Carlo
)
mc_evaluator.print_values()
```

### **Run Basic Navigation Example**

The **basic_navigation.py** example demonstrates the agent navigating a static GridWorld environment. To run this example:

```bash
# Run the basic navigation example
python -m minrl.examples.basic_navigation
```

### **Run Experiments with All Implemented Algorithms**

To compare different RL algorithms (Monte Carlo, Q-learning, DQN, etc.) on the same environment:

```bash
# Run experiments with all RL algorithms
python -m minrl.examples.run_experiments
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ✨ Acknowledgments

- Inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl) and [RLCode](https://github.com/rlcode/reinforcement-learning)
- Special thanks to Professor Shiyu Zhao for his insightful course on the "Mathematical Foundations of Reinforcement Learning"

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Contact

Created by: Yibin Liu  
Email: [yibin.leon.liu@outlook.com](yibin.leon.liu@outlook.com)  
Last Updated: 2025-02-09
