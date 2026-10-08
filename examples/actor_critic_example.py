"""Lesson 7: the critic's one-step TD error guides the actor."""

import torch
from minrl import GridWorld, ActorCriticAgent
from minrl.agents.common import seed_everything


def main():
    seed_everything(0)
    torch.set_num_threads(1)
    agent = ActorCriticAgent(GridWorld())
    history = agent.train(total_timesteps=5000, seed=0)
    print(f"Completed {len(history)} episodes; last return: {history[-1]['episode_return']}")


if __name__ == "__main__":
    main()
