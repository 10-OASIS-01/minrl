"""Reproduce the README's grid figures from a trained Q-table and exact values."""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from minrl import GridWorld, QLearningAgent, PolicyOptimizer
from minrl.utils.visualization import (
    plot_policy,
    plot_value_comparison,
    visualize_episode_trajectory,
    plot_training_results,
    save_figure,
)


def main(output):
    env = GridWorld(size=5, terminal_states={24: 1.0, 7: -1.0, 12: -1.0, 17: -1.0})
    agent = QLearningAgent(env, seed=0)
    rewards, lengths = agent.train(1500, 100)
    _, exact = PolicyOptimizer(env).value_iteration()
    learned = np.array(
        [
            max(agent.q_table[s, a] for a in env.get_valid_actions(s))
            if s not in env.terminal_states
            else 0
            for s in range(env.n_states)
        ]
    )
    state = env.reset()
    trajectory = []
    for step in range(100):
        action = agent.select_action(state, deterministic=True)
        nxt, reward, terminated, _ = env.step(action)
        trajectory.append(
            dict(
                state=state,
                action=action,
                reward=reward,
                next_state=nxt,
                terminated=terminated,
                truncated=step == 99 and not terminated,
            )
        )
        state = nxt
        if terminated:
            break
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), layout="constrained")
    plot_policy(agent.get_optimal_policy(), env, ax=axes[0], title="Q-learning · learned policy")
    visualize_episode_trajectory(env, trajectory, ax=axes[1], title="Greedy evaluation")
    save_figure(fig, output / "gridworld")
    plt.close(fig)
    fig = plot_value_comparison(
        {"Value iteration · exact": exact, "Q-learning · estimated": learned}, env
    )
    save_figure(fig, output / "values")
    plt.close(fig)
    steps = np.cumsum(lengths)
    rows = [
        dict(step=int(s), episode_return=r, episode_length=n)
        for s, r, n in zip(steps, rewards, lengths)
    ]
    fig = plot_training_results(rows)
    save_figure(fig, output / "q_learning")
    plt.close(fig)
    from minrl.experiments import write_json

    write_json(
        output / "gridworld_data.json",
        dict(
            seed=0,
            episodes=1500,
            size=5,
            terminal_states=env.terminal_states,
            rewards=rewards,
            lengths=lengths,
            values=exact.tolist(),
            learned_values=learned.tolist(),
            policy={s: p.tolist() for s, p in agent.get_optimal_policy().items()},
            trajectory=trajectory,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("figure"))
    main(parser.parse_args().output)
