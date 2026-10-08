# Changes in 0.2

## Correctness

- The default GridWorld no longer resets into a terminal state.
- Planning no longer mutates the environment or counts a terminal reward twice.
- Q-learning stops bootstrap at true termination, independent of terminal Q initialization.
- DQN masks target actions and synchronizes on global rather than per-episode steps.
- Actor-critic and PPO use identical masked distributions for acting and learning.
- PPO uses the next observation's value, separates truncation from termination, handles
  one-sample updates, and learns from the final partial rollout.
- MCTS expands beyond the root and backs up the rewards of traversed edges.
- Monte Carlo resets statistics, bounds episodes and reports discarded truncations.
- Trajectory plots include the final transition; policy plots identify terminal cells.

## Completed teaching extensions

Prioritized DQN replay, optional tabular replay, hard/soft target updates, model save/load,
learning-rate decay, optional evaluation-based early stopping, basic navigation and experiment
examples are implemented. Advanced options remain separate from the initial lessons.

## Readability and presentation

Each algorithm keeps its own explicit training and update functions. Common helpers do not
hide learning rules. The learning guide links equations to these functions and explains
state transitions, collection, updates, and termination boundaries. README and algorithm
guides use real interfaces rather than unfinished skeletons.

Plots share a restrained palette and typography, separate loss scales, display real time
coordinates, and use common value color scales. Benchmarks retain all seeds and distinguish
variability across seeds from variation across episodes.

## Evidence rather than coverage claims

The previous test files were mainly print/plot demos, with no assertion statements and no
PPO tests. The replacement suite checks hand-computed targets, masks, boundary conditions,
replay math, save/load, Gymnasium compatibility, reproducibility, logging and plotting.
See the generated benchmark report for measured local test coverage and training results.
