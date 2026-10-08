import json
import random
import subprocess
import sys
import numpy as np
import pytest
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from minrl import DQNAgent
from minrl.experiments import EarlyStopping, make_env, train_run, evaluate, main, benchmark
from minrl.agents.common import seed_everything


def small_config():
    return dict(
        algorithm="ppo",
        env="GridWorld",
        seed=3,
        total_timesteps=16,
        eval_interval=8,
        eval_episodes=1,
        final_eval_episodes=2,
        tensorboard=True,
        agent=dict(rollout_steps=8, num_epochs=1, batch_size=4),
    )


def test_run_artifacts_tensorboard_and_no_overwrite(tmp_path):
    folder = tmp_path / "run"
    result = train_run(small_config(), folder)
    assert result["status"] == "complete"
    assert result["training_steps"] == 16
    assert len(result["returns"]) == 2
    for file in (
        "config.yaml",
        "metadata.json",
        "metrics.csv",
        "evaluations.json",
        "final.pt",
        "best.pt",
    ):
        assert (folder / file).exists()
    events = EventAccumulator(str(folder / "tensorboard")).Reload()
    assert len(events.Scalars("eval/return")) == 2
    assert "train/policy_loss" in events.Tags()["scalars"]
    with pytest.raises(FileExistsError):
        train_run(small_config(), folder)


def test_evaluation_preserves_training_rng_and_state():
    seed_everything(3)
    env = make_env("CartPole-v1")
    agent = DQNAgent(env)
    env.reset(seed=8)
    state = env.unwrapped.state.copy()
    python_rng, numpy_rng, torch_rng = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    own_rng = agent.rng.bit_generator.state
    result = evaluate(agent, "CartPole-v1", episodes=2)
    assert result == evaluate(agent, "CartPole-v1", episodes=2)
    assert python_rng == random.getstate()
    np.testing.assert_array_equal(numpy_rng[1], np.random.get_state()[1])
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert own_rng == agent.rng.bit_generator.state
    np.testing.assert_array_equal(state, env.unwrapped.state)
    env.close()


def test_reproducible_runs_and_early_stopping(tmp_path):
    config = dict(small_config(), tensorboard=False)
    first = train_run(config, tmp_path / "first")
    second = train_run(config, tmp_path / "second")
    assert first["returns"] == second["returns"]
    a = torch.load(tmp_path / "first/final.pt", weights_only=True)
    b = torch.load(tmp_path / "second/final.pt", weights_only=True)
    assert all(torch.equal(a["actor"][k], b["actor"][k]) for k in a["actor"])
    config["early_stopping"] = dict(threshold=-100, patience=1)
    result = train_run(config, tmp_path / "early")
    assert result["stopped_early"] and result["training_steps"] == 8
    stopper = EarlyStopping(10, 2)
    assert not stopper(10)
    assert not stopper(9)
    assert not stopper(10)
    assert stopper(10)


def test_cli_overrides_and_installed_import(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(small_config()))
    folder = tmp_path / "cli"
    main(
        [
            "train",
            "--config",
            str(config_path),
            "--output",
            str(folder),
            "--steps",
            "8",
            "--seed",
            "9",
        ]
    )
    saved = yaml.safe_load((folder / "config.yaml").read_text())
    assert saved["total_timesteps"] == 8 and saved["seed"] == 9
    main(["evaluate", "--run", str(folder), "--episodes", "1"])
    completed = subprocess.run(
        [sys.executable, "-c", "import minrl; print(minrl.__version__)"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "0.2.0"


def test_failed_run_is_reported(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate test failure")

    from minrl import PPOAgent

    monkeypatch.setattr(PPOAgent, "train", fail)
    with pytest.raises(RuntimeError):
        train_run(small_config(), tmp_path / "bad")
    assert json.loads((tmp_path / "bad/result.json").read_text())["status"] == "failed"


def test_benchmark_with_one_seed_and_failure(tmp_path):
    config = dict(
        env="GridWorld",
        algorithms=["dqn", "invalid"],
        seeds=[0],
        total_timesteps=4,
        eval_interval=4,
        eval_episodes=1,
        final_eval_episodes=1,
        tensorboard=False,
    )
    results = benchmark(config, tmp_path / "benchmark")
    assert [r["status"] for r in results] == ["complete", "failed"]
    assert (tmp_path / "benchmark/figures/benchmark.svg").exists()
