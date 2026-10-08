# PPO: collect first, update second

Read [`ppo.py`](../src/minrl/agents/ppo.py) in this order:
`sample_action` → `train` → `compute_gae` → `update`.
Run `python -m examples.ppo_example`.

The implementation uses **separate actor and critic networks**, each with two 64-unit tanh
layers. The actor outputs logits, not pre-masked probabilities. A shared helper constructs
the same masked categorical distribution during both sampling and optimization.

## Phase 1: collect a rollout

For each step, save the encoded observation, action, reward, old value, next observation's
value, old log probability, termination flag, truncation flag and action mask.
Do not update the policy between these samples. At `rollout_steps`, stop collecting and learn.
The final partial rollout is also used when the training budget ends.

## Phase 2: compute advantages and returns

```text
delta[t] = reward[t] + gamma * (1 - terminated[t]) * next_value[t] - value[t]
adv[t]   = delta[t] + gamma * lambda * same_episode[t] * adv[t + 1]
return[t] = adv[t] + value[t]
```

`same_episode` is false for termination **or** truncation. At a time limit, `next_value`
comes from the final observation, not the reset state. At a rollout boundary, the recursion
stops, but the TD residual still includes the next state's value.

Normalize advantages using population standard deviation. For a single sample, skip
normalization: its variance is zero and centering would remove the learning signal.

## Phase 3: optimize, then discard the rollout

```text
ratio       = exp(new_log_prob - old_log_prob)
unclipped   = ratio * advantage
clipped     = clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage
policy_loss = -mean(min(unclipped, clipped))
```

The critic regresses to the fixed rollout returns, using the maximum of unclipped and
clipped squared value errors. Positive entropy is logged; its coefficient is subtracted
from the total loss to encourage exploration. Shuffle the rollout into minibatches and
repeat for `num_epochs`, then clear it. Old rollouts are not a replay buffer.

The default clip ratio is 0.2, GAE lambda 0.95, gamma 0.99, actor learning rate 0.0003,
value-loss coefficient 0.5, entropy coefficient 0.01 and maximum gradient norm 0.5.
Approximate KL is logged as a diagnostic; this implementation does not stop updates by KL.

Reference: [Schulman et al., Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
