# Experiments, after the lessons

The optional runner keeps file handling out of algorithm code. Agents work on their own;
their `callback(agent, metrics)` argument is only a small hook for logging or stopping.
Returning `False` stops training; `None` or `True` continues. There is no callback class hierarchy.

## Commands

```bash
pip install -e '.[experiments]'
minrl train --config configs/cartpole_dqn.yaml --seed 2 --steps 10000 --output runs/dqn-short
minrl evaluate --run runs/dqn-short --episodes 100
minrl plot --run runs/dqn-short
tensorboard --logdir runs
minrl benchmark --config configs/benchmark.yaml --output runs/cartpole
minrl plot --run runs/cartpole
```

Run commands from the repository root. Training creates a **new** output folder and fails
rather than overwrite an existing one. `--steps` and `--seed` override YAML; on a benchmark,
`--seed` restricts it to one seed. `plot --output path` selects a different figure directory.
The dedicated `configs/cartpole_per.yaml` changes replay, not the underlying DQN architecture.

## Reproducibility and evaluation

Each run seeds Python, NumPy, PyTorch and its environment. Neural-network initialization
uses the runner seed before construction. For direct use, call `seed_everything(seed)`
**before** constructing an agent; `train(seed=...)` seeds the environment, not existing weights.
Training is CPU-only with one PyTorch thread. Exact results are not guaranteed across
PyTorch versions, platforms or changes to the implementation.

Every 5,000 training steps, evaluate greedily on ten episodes with reset seeds 10000–10009.
At the end, evaluate the **final** checkpoint on 100 episodes with seeds 20000–20099.
The separately saved best checkpoint is selected using periodic evaluation and is not
substituted for the final checkpoint in the reported benchmark. The random baseline uses
the same final environment seeds and uniformly samples valid actions.

Evaluation owns a separate environment and restores Python, NumPy and PyTorch RNG states.
It does not alter replay, exploration, model parameters or the training episode. CartPole
has its native 500-step limit; the evaluator also has a 1,000-step safety bound.

## Files you can inspect

| File | Contents |
| --- | --- |
| `config.yaml` | Requested settings plus resolved agent defaults |
| `metadata.json` | Commit, dirty-tree status, dependency versions, CPU thread count |
| `metrics.csv` | Losses and episode statistics at their actual environment step |
| `evaluations.json` | Per-episode periodic evaluation returns and lengths |
| `result.json` | Final per-episode returns, seed, completion status and elapsed time |
| `final.pt`, `best.pt` | Networks, optimizers, constructor configuration and episode history |
| `tensorboard/` | Training and evaluation event files |
| `figures/` | Rebuildable SVG and 300 DPI PNG exports |

A benchmark adds `summary.json` (all seeds, including failures) and `random_baseline.json`.
Training losses are logged on update steps; episode returns on episode completion. The CSV
is sparse by design: an empty loss cell does not mean zero loss. Timing includes periodic
and final evaluation, and logging; it is not an isolated inference benchmark.

Plots aggregate per-seed evaluation means. Bands show **±1 population standard deviation
across seeds**, not confidence intervals. Curves only use evaluation steps shared by the
included runs, with no extrapolation. Incomplete runs are counted explicitly. The final
distribution plots one mean per seed instead of treating correlated episodes as independent runs.

## Optional extensions

- `learning_rate_decay: true` in the agent settings linearly reduces learning rate over the
  current training budget. It is off by default.
- DQN `target_update: soft` uses `tau: 0.005` after each optimizer update; the default is hard
  synchronization every `target_update_freq: 500` environment steps. The two modes are exclusive.
- `early_stopping: {threshold: 475, patience: 3}` at the experiment level stops after three
  consecutive periodic evaluations reach that score. It is off in the full benchmark.
- For Q-learning, `replay=True` enables a uniform buffer after the online update; `batch_size`
  controls how many additional table updates occur once enough transitions are available.
- Set `tensorboard: false` to use CSV/JSON only, without installing the TensorBoard extra.

## Checkpoints are for reloading and evaluation

```python
import gymnasium as gym
from minrl import DQNAgent

agent = DQNAgent(gym.make("CartPole-v1"))
agent.load("runs/dqn-short/final.pt")
```

Loading reconstructs the saved architecture, weights, optimizers and training statistics.
Replay, environment state and random-generator state are not checkpointed, so this is not
an exact training resumption mechanism. Use a matching environment. Q-learning uses readable
JSON instead of PyTorch checkpoints. There is no cloud logging or distributed execution.
