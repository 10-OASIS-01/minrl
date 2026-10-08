# Actor-critic: learn a policy with a one-step critic

Read [`actor_critic.py`](../src/minrl/agents/actor_critic.py) and run
`python -m examples.actor_critic_example`.

The actor outputs action logits; the critic outputs one state value. They are separate
two-layer MLPs, optimized together. The policy distribution masks exactly the same actions
when sampling and when computing the update.

```text
target      = reward + gamma * (1 - terminated) * V(next_state)
advantage   = target - V(state)
actor_loss  = -log pi(action | state) * stop_gradient(advantage)
critic_loss = advantage ** 2
loss        = actor_loss + 0.5 * critic_loss - entropy_coef * entropy
```

`target` is calculated without gradients. The actor receives a detached advantage: it should
change action probabilities, not manipulate the critic to change its learning signal.
Positive advantage reinforces an action; negative advantage discourages it.

The loop is intentionally online: observe → sample → step → update → repeat. There is no
replay buffer or rollout reuse. A time limit resets the environment but still bootstraps
from its final observation. Learning-rate decay is optional and off by default.

```python
from minrl import GridWorld, ActorCriticAgent
from minrl.agents.common import seed_everything

seed_everything(0)
agent = ActorCriticAgent(GridWorld())
agent.train(total_timesteps=5000, seed=0)
policy = agent.get_optimal_policy()
```
