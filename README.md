# MinRL: Minimal, Clean Code for Reinforcement Learning

MinRL provides clean, minimal implementations of fundamental reinforcement learning algorithms in a customizable GridWorld environment. The project focuses on educational clarity and implementation simplicity while maintaining production-quality code standards.

## 🌟 Key Features

- **Modular GridWorld Environment**: Customizable grid sizes (3×3, 5×5) with configurable rewards and dynamics
- **Clean Implementation** of core RL algorithms:
  - Policy Evaluation (Bellman Expectation)
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
│   │   └── grid_world.py           # Implementation of Core grid world logic, state transitions, rewards, and environment dynamics
│   ├── agents/           
│   │   ├── policy_evaluation.py    # Implements Bellman Expectation for policy evaluation
│   │   ├── policy_optimization.py  # Policy iteration and value iteration implementations
│   │   ├── q_learning.py           # Tabular Q-Learning algorithm
│   │   └── deep_q_learning.py      # Deep Q-Learning (DQN) implementation using neural networks
│   └── utils/             
│       └── visualization.py        # Utilities for Visualizes state values, learned policies, and rewards over episodes
├── tests/                          # Comprehensive test suite ensuring correct functionality
├── examples/              
│   ├── basic_navigation.py         # Basic navigation example using a static GridWorld
│   └── run_experiments.py          # Runs experiments with all implemented RL algorithms
└── docs/                           # Implementation Guide for Beginners

```

## 🎓 Implemented Algorithms

1. **Policy Evaluation**
   - Implements Bellman expectation equation
   - Iterative value computation
   - Convergence validation

2. **Policy/Value Iteration**
   - Value iteration with Bellman optimality
   - Policy iteration (evaluation + improvement)
   - Optimal policy extraction

3. **Q-Learning**
   - Tabular state-action value learning
   - ε-greedy exploration
   - Hyperparameter tuning support

4. **Deep Q-Learning (DQN)**
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
---

### **Basic Usage**

Here’s how you can start using the basic functionality of the library:

```python
from minrl.environment import GridWorld
from minrl.agents import DQNAgent

# Create a 3x3 GridWorld environment
env = GridWorld(size=3)

# Initialize and train a DQN agent
agent = DQNAgent(env)
rewards = agent.train(episodes=1000)

# Visualize the learned policy
agent.visualize_policy()
```

This code sets up a simple **3x3 GridWorld**, trains a **DQN agent** for 1000 episodes, and then visualizes the learned policy.

---

### **Run Basic Navigation Example**

The **basic_navigation.py** example demonstrates the agent navigating a static GridWorld environment. To run this example:

```bash
# Run the basic navigation example
python -m minrl.examples.basic_navigation
```

This will execute a basic navigation task where the agent tries to learn how to move in the 3x3 GridWorld. It will show how to set up the environment and train a simple agent.

---

### **Run Experiments with All Implemented Algorithms**

To see how the different RL algorithms (like Q-learning, DQN, etc.) perform on the same environment, you can run **run_experiments.py**. This will execute a full set of experiments and compare the results.

```bash
# Run experiments with all RL algorithms
python -m minrl.examples.run_experiments
```

This command will automatically run experiments for all available RL algorithms, training agents using Q-Learning, Deep Q-Learning, and other approaches. The results will allow you to compare the performance of each algorithm in the same GridWorld environment.


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ✨ Acknowledgments

- Inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl) and [RLCode](https://github.com/rlcode/reinforcement-learning)

Certainly! Here's the updated **Contact** section with an added email address:

---

## 🔗 Contact

Created by: Yibin Liu  

Email: [yibin.leon.liu@outlook.com](yibin.leon.liu@outlook.com)  

Last Updated: 2025-02-08

