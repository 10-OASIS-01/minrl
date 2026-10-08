# DQN: the Q-learning target, fitted by a network

Read [`DQNAgent.train_step`](../src/minrl/agents/deep_q_learning.py), then `train`.
Run `python -m examples.deep_ql_example`.

## Trace one update

1. Sample `(state, action, reward, next_state, terminated, truncated, next_mask)` from replay.
2. The target network predicts next-state Q-values. Mask invalid actions before `max`.
3. Compute `target = reward + gamma * (1 - terminated) * max_next_q` without gradients.
4. Gather the online network's prediction for the action actually taken.
5. Minimize the mean squared TD error; update only the online network.
6. Periodically copy online weights to the target network.

The target network is a delayed copy, not a second independently trained predictor.
Hard copies happen every 500 **environment steps**, even when episodes are shorter.
For the optional soft update, each optimizer update applies
`target_parameter ← (1 − tau) target_parameter + tau online_parameter`.

## Replay before prioritized replay

Uniform replay samples old transitions with equal probability. The basic algorithm should
be understood first. `prioritized=True` enables the extension in
[`replay.py`](../src/minrl/agents/replay.py):

```text
priority_i = abs(td_error_i) + 1e-6
P(i)       = priority_i ** alpha / sum(priority ** alpha)
weight_i   = (N * P(i)) ** (-beta)
loss       = mean(normalized_weight_i * td_error_i ** 2)
```

Weights are normalized by the largest weight in the current buffer. Newly added transitions
start with the current maximum priority. Sampling is with replacement; repeated sampled
indices receive the maximum newly observed absolute TD error. Alpha defaults to 0.6; beta
anneals from 0.4 to 1 across the training budget.

This deliberately uses plain NumPy arrays instead of a segment tree: easier to read, but
priority sampling costs O(buffer size). PER can help or hurt depending on the task and settings;
the benchmark keeps it separate from uniform replay rather than assuming it is superior.

## Default teaching implementation

Two hidden layers of 64 ReLU units; Adam at 0.001; MSE loss; gradient norm limit 10;
50,000 replay capacity; batch size 64; 1,000 warm-up steps; one update every four steps.
Epsilon decreases linearly from 1 to 0.05 over 20,000 environment steps.
This is vanilla DQN, not Double DQN or dueling DQN.

Reference: [Schaul et al., Prioritized Experience Replay](https://arxiv.org/abs/1511.05952).
