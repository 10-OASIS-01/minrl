import numpy as np
import pytest
import torch
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from minrl import (
    GridWorld,
    GridWorldEnv,
    PolicyEvaluator,
    PolicyOptimizer,
    QLearningAgent,
    MonteCarloEvaluator,
    MCTSAgent,
    DQNAgent,
    PPOAgent,
    ActorCriticAgent,
)
from minrl.agents.common import seed_everything, masked_distribution, set_learning_rate
from minrl.agents.ppo import compute_gae
from minrl.agents.replay import ReplayBuffer


@pytest.fixture(autouse=True)
def reproducible():
    seed_everything(7)
    torch.set_num_threads(1)


def test_grid_model_and_terminal_reward():
    env = GridWorld(terminal_states={2: 1.0})
    assert env.reset() == 0
    assert env.transition(1, 1) == (2, 1.0, True)
    assert env.current_pos == (0, 0)
    assert env.step(0) == (0, -0.1, False, {})
    env.step(1)
    assert env.step(1) == (2, 1.0, True, {})
    assert env.step(2) == (2, 0.0, True, {})
    assert env.get_valid_actions(2) == []
    with pytest.raises(ValueError):
        GridWorld(terminal_states={0: 1.0})
    with pytest.raises(ValueError):
        GridWorld(size=2)
    with pytest.raises(ValueError):
        env.transition(-1, 0)
    clone = env.clone()
    clone.terminal_states[2] = 99
    assert env.terminal_states[2] == 1


def test_gym_adapter_and_time_limit():
    env = GridWorldEnv(max_episode_steps=1)
    check_env(GridWorldEnv(), skip_render_check=True)
    env.reset(seed=4)
    assert env.step(1)[2:4] == (False, True)


def test_bellman_values_and_state_preservation():
    env = GridWorld(terminal_states={2: 1.0})
    position = env.current_pos
    optimizer = PolicyOptimizer(env, gamma=0.9)
    policy, values = optimizer.value_iteration()
    assert values[2] == 0
    assert values[1] == pytest.approx(1)
    assert values[0] == pytest.approx(0.8)
    _, pi_values = PolicyOptimizer(env, gamma=0.9).policy_iteration()
    np.testing.assert_allclose(values, pi_values, atol=1e-6)
    np.testing.assert_allclose(values, PolicyEvaluator(env, 0.9).evaluate_policy(policy), atol=1e-6)
    assert env.current_pos == position


def test_evaluation_boundary_self_loop():
    env = GridWorld()
    policy = {s: [1, 0, 0, 0] for s in range(9)}
    values = PolicyEvaluator(env, gamma=0.9).evaluate_policy(policy)
    assert values[0] == pytest.approx(-1, abs=1e-6)


def test_q_learning_target_training_replay_and_checkpoint(tmp_path):
    env = GridWorld(terminal_states={2: 1.0})
    agent = QLearningAgent(env, learning_rate=1, gamma=0.9, replay=True)
    agent.q_table[2] = 100
    agent.update(1, 1, 1, 2, True)
    assert agent.q_table[1, 1] == 1
    agent.train(200, 50)
    assert len(agent.buffer) > agent.batch_size
    assert np.mean(agent.episode_rewards[-20:]) > 0
    agent.save(tmp_path / "q.json")
    other = QLearningAgent(env)
    other.load(tmp_path / "q.json")
    np.testing.assert_array_equal(agent.q_table, other.q_table)
    assert other.config["replay"]


def test_monte_carlo_visits_and_truncation(monkeypatch):
    mc = MonteCarloEvaluator(GridWorld(), gamma=1)
    monkeypatch.setattr(mc, "generate_episode", lambda *args: ([(0, 1, 1), (0, 1, 2)], False))
    assert mc.evaluate_policy({}, 1, first_visit=True)[0] == 3
    assert mc.evaluate_policy({}, 1, first_visit=False)[0] == 2.5
    monkeypatch.setattr(mc, "generate_episode", lambda *args: ([(0, 1, 10)], True))
    assert mc.evaluate_policy({}, 2)[0] == 0
    assert mc.truncated_episodes == 2


def test_mcts_counts_immediate_reward_and_requested_state():
    env = GridWorld(terminal_states={2: 1.0, 0: -1.0}, start_state=4)
    agent = MCTSAgent(env, num_simulations=200, seed=3)
    assert agent.select_action(1) == 1
    assert env.current_pos == (1, 1)
    root = agent._search(1)
    assert root.children[1].value / root.children[1].visits == pytest.approx(1)
    assert any(child.children for child in root.children.values())
    assert sum(agent.get_policy(1).values()) == pytest.approx(1)
    with pytest.raises(ValueError):
        agent.select_action(2)


def test_replay_probabilities_weights_and_overwrite():
    buffer = ReplayBuffer(3, prioritized=True, alpha=1)
    for i in range(3):
        buffer.add(i)
    buffer.update_priorities(np.array([0, 1, 2]), np.array([1.0, 2.0, 7.0]))
    np.testing.assert_allclose(buffer.probabilities(), [0.1, 0.2, 0.7], atol=1e-6)
    _, indices, weights = buffer.sample(5000, beta=1)
    assert np.mean(indices == 2) == pytest.approx(0.7, abs=0.03)
    np.testing.assert_allclose(
        weights, buffer.probabilities().min() / buffer.probabilities()[indices]
    )
    buffer.update_priorities(np.array([1, 1]), np.array([10.0, 2.0]))
    assert buffer.priorities[1] == pytest.approx(10.000001)
    buffer.add(99)
    assert buffer.data[0] == 99 and len(buffer) == 3


def test_dqn_target_mask_terminal_truncation():
    agent = DQNAgent(GridWorld(), learning_rate=0, gamma=0.9, batch_size=1)
    for network in (agent.q_network, agent.target_network):
        for parameter in network.parameters():
            parameter.data.zero_()
    agent.target_network[-1].bias.data[:] = torch.tensor([100.0, 2.0, 3.0, 100.0])
    state = agent.observations.encode(0)
    mask = np.array([False, True, True, False])
    agent.replay_buffer.add((state, 1, 1.0, state, False, True, mask))
    assert agent.train_step()["loss"] == pytest.approx((1 + 0.9 * 3) ** 2)
    agent.replay_buffer = ReplayBuffer(1)
    agent.replay_buffer.add((state, 1, 1.0, state, True, False, mask))
    assert agent.train_step()["loss"] == pytest.approx(1)
    for parameter in agent.q_network.parameters():
        parameter.data.fill_(2)
    for parameter in agent.target_network.parameters():
        parameter.data.zero_()
    agent.sync_target(0.25)
    assert all(
        torch.allclose(p, torch.full_like(p, 0.5)) for p in agent.target_network.parameters()
    )


def test_dqn_global_sync_and_learning_rate(monkeypatch):
    agent = DQNAgent(
        GridWorldEnv(max_episode_steps=1),
        batch_size=1,
        learning_starts=1,
        target_update_freq=3,
        learning_rate_decay=True,
    )
    syncs = []
    monkeypatch.setattr(agent, "sync_target", lambda: syncs.append(True))
    agent.train(10)
    assert len(syncs) == 3
    assert agent.optimizer.param_groups[0]["lr"] < agent.config["learning_rate"]


def test_gae_hand_calculation_and_episode_boundary():
    advantages, returns = compute_gae(
        [1.0, 2.0],
        [0.5, 1.0],
        [1.0, 10.0],
        [False, True],
        [False, False],
        gamma=0.9,
        gae_lambda=0.8,
    )
    np.testing.assert_allclose(advantages, [2.12, 1.0])
    np.testing.assert_allclose(returns, [2.62, 2.0])
    advantages, _ = compute_gae(
        [1.0, 100.0], [0.5, 1.0], [2.0, 0.0], [False, True], [True, False], gamma=0.9
    )
    assert advantages[0] == pytest.approx(2.3)  # Includes final value, excludes next episode.


def test_ppo_mask_ratio_single_sample_and_tail():
    agent = PPOAgent(GridWorld(), batch_size=1, num_epochs=1, rollout_steps=64)
    observation = 0
    action, log_prob, _ = agent.sample_action(observation)
    state = torch.from_numpy(agent.observations.encode(observation))
    mask = torch.as_tensor(agent.observations.mask(observation))
    dist = masked_distribution(agent.actor, state, mask)
    assert dist.probs[0] == 0 and dist.probs[3] == 0
    assert torch.exp(dist.log_prob(torch.tensor(action)) - log_prob).item() == pytest.approx(1)
    before = [p.detach().clone() for p in agent.actor.parameters()]
    rows = []
    agent.train(1, callback=lambda a, row: rows.append(row))
    assert agent.memory == [] and "policy_loss" in rows[-1]
    assert all(np.isfinite(v) for v in rows[-1].values())
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.actor.parameters()))
    assert agent.update() == {}


@pytest.mark.parametrize("agent_type", [DQNAgent, PPOAgent, ActorCriticAgent])
@pytest.mark.parametrize("environment", ["CartPole-v1", "FrozenLake-v1", "GridWorld"])
def test_neural_train_save_load(agent_type, environment, tmp_path):
    env = GridWorldEnv() if environment == "GridWorld" else gym.make(environment)
    kwargs = dict(batch_size=2, learning_starts=2) if agent_type is DQNAgent else {}
    if agent_type is PPOAgent:
        kwargs = dict(batch_size=4, num_epochs=1, rollout_steps=8)
    agent = agent_type(env, **kwargs)
    rows = []
    agent.train(16, callback=lambda a, row: rows.append(row))
    assert all(np.isfinite(value) for row in rows for value in row.values())
    observation, info = env.reset(seed=42)
    action = agent.select_action(observation, deterministic=True, info=info)
    agent.save(tmp_path / "model.pt")
    restored = agent_type(env)
    restored.load(tmp_path / "model.pt")
    assert restored.select_action(observation, deterministic=True, info=info) == action
    assert restored.config == agent.config
    env.close()


def test_learning_rate_schedule():
    agent = PPOAgent(GridWorld())
    set_learning_rate(agent.optimizer, 0.1, 50, 100, True)
    assert agent.optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
