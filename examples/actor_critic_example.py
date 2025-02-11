import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import GridWorld
from src.agents.actor_critic import ActorCriticAgent
from src.utils.visualization import Visualizer

import torch
import numpy as np

def run_actor_critic_demo():
    """Run a demonstration of the Actor-Critic algorithm"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create environment with interesting terminal states
    env = GridWorld(size=9)

    # Set terminal states that don't conflict with starting position (0,0)
    goal_state = env._pos_to_state((4, 6))  # Bottom-right corner
    trap_states = [
        env._pos_to_state((2, 5)),
        env._pos_to_state((2, 4)),
        env._pos_to_state((3, 4)),
        env._pos_to_state((4, 4)),
        env._pos_to_state((5, 4))
    ]

    # Clear default terminal states and set new ones
    env.terminal_states.clear()  # Clear default terminal states
    env.terminal_states[goal_state] = 10.0
    for trap_state in trap_states:
        env.terminal_states[trap_state] = -2.0  # Trap states with negative reward

    # Create Actor-Critic agent with custom hyperparameters
    agent = ActorCriticAgent(
        env,
        learning_rate_actor=0.0003,
        learning_rate_critic=0.001,
        gamma=0.99,
        entropy_coef=0.01
    )

    # Train the agent
    print("Training Actor-Critic agent...")
    n_episodes = 2000
    max_steps = 100
    rewards, lengths = agent.train(
        n_episodes=n_episodes,
        max_steps=max_steps
    )

    # Plot training results using the Visualizer
    training_fig = Visualizer.plot_training_results(
        rewards=rewards,
        lengths=lengths,
        actor_losses=agent.actor_losses,
        critic_losses=agent.critic_losses,
        title='Actor-Critic Training Progress'
    )
    training_fig.show()

    # Get and display the learned policy
    policy = agent.get_optimal_policy()
    policy_fig = Visualizer.plot_policy(
        policy,
        env.size,
        "Actor-Critic Learned Policy"
    )
    policy_fig.show()

    # Run and visualize a test episode
    print("\nRunning test episode with learned policy...")
    state = env.reset()
    episode_data = []
    done = False
    total_reward = 0

    while not done:
        action, _, _ = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        episode_data.append((state, action, reward))
        total_reward += reward
        state = next_state

    # Visualize the test episode
    episode_fig = Visualizer.visualize_episode_trajectory(
        env,
        episode_data,
        "Actor-Critic Test Episode Trajectory"
    )
    episode_fig.show()

    print(f"Test episode finished with total reward: {total_reward}")

    return agent, policy, total_reward


if __name__ == "__main__":
    agent, policy, final_reward = run_actor_critic_demo()