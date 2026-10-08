# Migrating from 0.1 to 0.2

This revision prioritizes correct, readable lessons over backward compatibility.

| Before | Now |
| --- | --- |
| `from src.agents import ...` | `from minrl.agents import ...` or `from minrl import ...` |
| Default reset starts on a terminal cell | State 0 is a nonterminal start |
| Mutate `terminal_states` after construction | Prefer `GridWorld(terminal_states={...})` |
| Terminal values equal terminal rewards | Terminal future values are zero; reward is paid on entry |
| Neural `train(n_episodes=...)` | `train(total_timesteps=..., seed=...)` returns episode records |
| PPO/AC `select_action` returns a tuple | `select_action` returns an integer; PPO exposes `sample_action` for training details |
| Plot separate rewards/loss lists on implicit clocks | `plot_training_results(rows)` uses explicit environment steps |
| `plot_policy(policy, size)` | `plot_policy(policy, env)` identifies terminal states |
| Trajectory tuples omit the final next state | Dictionaries include `next_state`, `terminated`, `truncated` |
| Monte Carlo `generate_episode` returns a list | Returns `(episode, truncated)` so incomplete episodes cannot be mistaken for complete ones |

The small GridWorld `reset()` and four-element `step()` interface remains for tabular
lessons. `GridWorldEnv` wraps it in Gymnasium's reset pair and five-element step interface.
Neural agents accept either one. Old checkpoints are not compatible with the new architecture
and configuration format. The package version now matches project metadata: `0.2.0`.
