"""Lesson 6: replace the Q-table with a network. Read DQNAgent.train_step next."""

import gymnasium as gym
import torch
from minrl import DQNAgent
from minrl.agents.common import seed_everything


def main():
    seed_everything(0)
    torch.set_num_threads(1)
    env = gym.make("CartPole-v1")
    agent = DQNAgent(env)
    history = agent.train(total_timesteps=10000, seed=0)
    print(f"Completed {len(history)} episodes; last return: {history[-1]['episode_return']}")
    env.close()


if __name__ == "__main__":
    main()
