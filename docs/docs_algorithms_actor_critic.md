# Actor-Critic Implementation Guide
Created by: 10-OASIS-01
Date: 2025-02-09 04:56:06 UTC

## Overview
The Actor-Critic algorithm combines policy gradient methods with value function approximation,
providing a hybrid approach to reinforcement learning. This implementation uses two neural
networks:
- Actor: learns to select actions by outputting action probabilities
- Critic: learns to evaluate states by estimating their values

## Implementation Details

### Networks Architecture
- **Actor Network**: Maps states to action probabilities using softmax output
- **Critic Network**: Maps states to scalar value estimates

### Learning Process
1. Actor selects actions using learned policy
2. Critic evaluates states and computes TD error
3. TD error is used to:
   - Update critic network (minimize TD error)
   - Update actor network (policy gradient with TD error as advantage)

### Key Features
- Handles invalid actions through action masking
- Provides detailed training statistics
- Supports hyperparameter tuning
- Includes policy extraction functionality

## Usage Example
```python
from src.environment import GridWorld
from src.agents import ActorCriticAgent

# Create environment and agent
env = GridWorld(size=4)
agent = ActorCriticAgent(env)

# Train the agent
rewards, lengths = agent.train(n_episodes=1000)

# Get learned policy
policy = agent.get_optimal_policy()