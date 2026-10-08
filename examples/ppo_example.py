"""Lesson 8: collect a rollout, compute advantages, optimize the clipped objective."""

import gymnasium as gym
import torch
from minrl import PPOAgent
from minrl.agents.common import seed_everything


def main():
    seed_everything(0)
    torch.set_num_threads(1)
    env = gym.make("CartPole-v1")
    agent = PPOAgent(env, rollout_steps=512, num_epochs=4)
    history = agent.train(total_timesteps=10000, seed=0)
    print(f"Completed {len(history)} episodes; last return: {history[-1]['episode_return']}")
    env.close()


if __name__ == "__main__":
    main()
