"""Monte Carlo tree search: select, expand, simulate, back up."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class MCTSNode:
    state: int
    untried_actions: list
    children: dict = field(default_factory=dict)
    visits: int = 0
    value: float = 0.0  # Sum of returns from this node's incoming edge.


class MCTSAgent:
    def __init__(
        self,
        env,
        num_simulations=100,
        exploration_constant=1.41,
        max_rollout_steps=100,
        gamma=0.99,
        seed=0,
    ):
        if num_simulations < 1 or max_rollout_steps < 1:
            raise ValueError("search budgets must be positive")
        self.env, self.num_simulations = env, num_simulations
        self.exploration_constant, self.max_rollout_steps = exploration_constant, max_rollout_steps
        self.gamma, self.rng = gamma, np.random.default_rng(seed)

    def _search(self, state):
        root = MCTSNode(state, list(self.env.get_valid_actions(state)))
        if not root.untried_actions:
            raise ValueError("cannot search from a terminal state")
        for _ in range(self.num_simulations):
            node, path, depth = root, [], 0
            # 1. Select visited edges by UCB; 2. expand one unseen edge.
            while node.state not in self.env.terminal_states and depth < self.max_rollout_steps:
                expand = bool(node.untried_actions)
                if expand:
                    action = int(self.rng.choice(node.untried_actions))
                    node.untried_actions.remove(action)
                    nxt, reward, _ = self.env.transition(node.state, action)
                    child = MCTSNode(nxt, list(self.env.get_valid_actions(nxt)))
                    node.children[action] = child
                else:
                    action, child = max(
                        node.children.items(),
                        key=lambda item: (
                            item[1].value / item[1].visits
                            + self.exploration_constant
                            * np.sqrt(np.log(max(1, node.visits)) / item[1].visits)
                        ),
                    )
                    _, reward, _ = self.env.transition(node.state, action)
                path.append((child, reward))
                node, depth = child, depth + 1
                if expand:
                    break
            # 3. Random rollout from the leaf, within a fixed total horizon.
            rollout_state, total, discount = node.state, 0.0, 1.0
            for _ in range(self.max_rollout_steps - depth):
                actions = self.env.get_valid_actions(rollout_state)
                if not actions:
                    break
                rollout_state, reward, _ = self.env.transition(
                    rollout_state, self.rng.choice(actions)
                )
                total += discount * reward
                discount *= self.gamma
            # 4. Every edge gets its own immediate reward plus discounted future return.
            for child, reward in reversed(path):
                total = reward + self.gamma * total
                child.visits += 1
                child.value += total
            root.visits += 1
        return root

    def select_action(self, state):
        root = self._search(state)
        return max(root.children, key=lambda a: root.children[a].visits)

    def get_policy(self, state):
        root = self._search(state)
        total = sum(child.visits for child in root.children.values())
        return {
            a: root.children[a].visits / total if a in root.children else 0.0
            for a in range(self.env.n_actions)
        }
