"""Optional experiment script. Students can ignore this file while learning an agent."""

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import time

import gymnasium as gym
import numpy as np
import torch
import yaml

from . import GridWorldEnv, DQNAgent, PPOAgent, ActorCriticAgent
from .agents.common import seed_everything


def make_env(name, kwargs=None):
    kwargs = dict(kwargs or {})
    if name == "GridWorld":
        return GridWorldEnv(**kwargs)
    return gym.make(name, **kwargs)


def make_agent(config, env):
    name = config["algorithm"]
    parameters = dict(config.get("agent", {}))
    if name in ("dqn", "dqn_per"):
        parameters["prioritized"] = name == "dqn_per"
        parameters["seed"] = config.get("seed", 0)
        return DQNAgent(env, **parameters)
    if name == "ppo":
        return PPOAgent(env, **parameters)
    if name == "actor_critic":
        return ActorCriticAgent(env, **parameters)
    raise ValueError(f"unknown algorithm: {name}")


def evaluate(agent, env_name, env_kwargs=None, episodes=10, seed=10000, max_steps=1000):
    """Evaluation owns its environment and preserves all training RNG state."""
    python_rng, numpy_rng, torch_rng = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    env = make_env(env_name, env_kwargs)
    returns, lengths = [], []
    rng = np.random.default_rng(seed)
    try:
        for episode in range(episodes):
            observation, info = env.reset(seed=seed + episode)
            total = 0.0
            for step in range(max_steps):
                if agent is None:
                    mask = info.get("action_mask", np.ones(env.action_space.n, dtype=bool))
                    action = int(rng.choice(np.flatnonzero(mask)))
                else:
                    action = agent.select_action(observation, deterministic=True, info=info)
                observation, reward, terminated, truncated, info = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            returns.append(total)
            lengths.append(step + 1)
    finally:
        env.close()
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
    return dict(
        returns=returns,
        lengths=lengths,
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        mean_length=float(np.mean(lengths)),
    )


class EarlyStopping:
    """Optional stopping after a target score is met on consecutive evaluations."""

    def __init__(self, threshold, patience=3):
        if patience < 1:
            raise ValueError("patience must be positive")
        self.threshold, self.patience, self.count = threshold, patience, 0

    def __call__(self, score):
        self.count = self.count + 1 if score >= self.threshold else 0
        return self.count >= self.patience


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def metadata():
    def git(*args):
        result = subprocess.run(
            ["git", *args], text=True, capture_output=True, cwd=Path(__file__).resolve().parents[2]
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    return dict(
        commit=git("rev-parse", "HEAD"),
        working_tree=git("status", "--short"),
        dependencies={
            name: importlib.metadata.version(name)
            for name in ("minrl", "numpy", "torch", "gymnasium", "matplotlib")
        },
        device="cpu",
        torch_threads=torch.get_num_threads(),
    )


def train_run(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)  # Never silently overwrite an experiment.
    seed = int(config.get("seed", 0))
    total_steps = int(config.get("total_timesteps", 100000))
    interval = int(config.get("eval_interval", 5000))
    if min(total_steps, interval) < 1:
        raise ValueError("training and evaluation intervals must be positive")
    seed_everything(seed)
    torch.set_num_threads(1)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(output / "metadata.json", metadata())
    writer, env = None, None
    evaluations = []
    best_score, start, final_step = -np.inf, time.perf_counter(), 0
    stopper = EarlyStopping(**config["early_stopping"]) if config.get("early_stopping") else None
    fields = [
        "step",
        "episode_return",
        "episode_length",
        "loss",
        "td_error",
        "epsilon",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "steps_per_second",
    ]
    try:
        if config.get("tensorboard", True):
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(str(output / "tensorboard"))
        env = make_env(config["env"], config.get("env_kwargs"))
        agent = make_agent(config, env)
        # Materialize defaults so a run can be understood without opening the source.
        config = dict(config, agent=agent.config)
        (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        with (output / "metrics.csv").open("w", newline="") as file:
            csv_writer = csv.DictWriter(file, fieldnames=fields)
            csv_writer.writeheader()

            def record(current_agent, row):
                nonlocal best_score, final_step
                final_step = row["step"]
                if not all(np.isfinite(v) for v in row.values()):
                    raise FloatingPointError(f"non-finite training metric at step {final_step}")
                # Losses and episodes have different clocks; retain their actual step.
                if len(row) > 2 or final_step % 100 == 0:
                    row = dict(row, steps_per_second=final_step / (time.perf_counter() - start))
                    csv_writer.writerow(row)
                    if writer:
                        for key, value in row.items():
                            if key != "step":
                                writer.add_scalar("train/" + key, value, final_step)
                if final_step % interval == 0 or final_step == total_steps:
                    result = evaluate(
                        current_agent,
                        config["env"],
                        config.get("env_kwargs"),
                        episodes=config.get("eval_episodes", 10),
                        seed=10000,
                    )
                    result["step"] = final_step
                    evaluations.append(result)
                    write_json(output / "evaluations.json", evaluations)
                    file.flush()
                    if writer:
                        writer.add_scalar("eval/return", result["mean_return"], final_step)
                        writer.add_scalar("eval/length", result["mean_length"], final_step)
                        writer.flush()
                    if result["mean_return"] > best_score:
                        best_score = result["mean_return"]
                        current_agent.save(output / "best.pt")
                    print(
                        f"{config['algorithm']} seed={seed} step={final_step:,} eval={result['mean_return']:.1f}",
                        flush=True,
                    )
                    if stopper and stopper(result["mean_return"]):
                        return False
                return True

            agent.train(total_steps, seed=seed, callback=record)
        agent.save(output / "final.pt")
        result = evaluate(
            agent,
            config["env"],
            config.get("env_kwargs"),
            episodes=config.get("final_eval_episodes", 100),
            seed=20000,
        )
        result.update(
            status="complete",
            algorithm=config["algorithm"],
            seed=seed,
            training_steps=final_step,
            elapsed_seconds=time.perf_counter() - start,
            stopped_early=final_step < total_steps,
        )
        write_json(output / "result.json", result)
        return result
    except Exception as error:
        write_json(
            output / "result.json",
            dict(
                status="failed",
                algorithm=config["algorithm"],
                seed=seed,
                training_steps=final_step,
                error=repr(error),
            ),
        )
        raise
    finally:
        if env is not None:
            env.close()
        if writer:
            writer.close()


def benchmark(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for algorithm in config.get("algorithms", ["dqn", "dqn_per", "ppo"]):
        for seed in config.get("seeds", [0, 1, 2, 3, 4]):
            run_config = {
                k: v for k, v in config.items() if k not in ("algorithms", "seeds", "agents")
            }
            run_config.update(
                algorithm=algorithm, seed=seed, agent=config.get("agents", {}).get(algorithm, {})
            )
            folder = output / f"{algorithm}-seed{seed}"
            try:
                result = train_run(run_config, folder)
            except Exception as error:
                result = dict(status="failed", algorithm=algorithm, seed=seed, error=repr(error))
                print(result, flush=True)
            results.append(result)
            write_json(output / "summary.json", results)
    baseline = evaluate(
        None,
        config["env"],
        config.get("env_kwargs"),
        episodes=config.get("final_eval_episodes", 100),
        seed=20000,
    )
    write_json(output / "random_baseline.json", baseline)
    from .utils.visualization import plot_experiment

    plot_experiment(output)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate", "benchmark", "plot"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run", type=Path, help="existing run or benchmark directory")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args(argv)
    if args.command in ("train", "benchmark"):
        if args.config is None or args.output is None:
            parser.error("--config and --output are required")
        config = yaml.safe_load(args.config.read_text())
        if args.steps is not None:
            config["total_timesteps"] = args.steps
        if args.seed is not None:
            config["seed"] = args.seed
            config["seeds"] = [args.seed]
        result = (
            train_run(config, args.output)
            if args.command == "train"
            else benchmark(config, args.output)
        )
        if args.command == "benchmark" and any(r["status"] == "failed" for r in result):
            raise SystemExit(1)
    elif args.command == "plot":
        if args.run is None:
            parser.error("--run is required")
        from .utils.visualization import plot_experiment

        plot_experiment(args.run, args.output)
    else:
        if args.run is None:
            parser.error("--run is required")
        config = yaml.safe_load((args.run / "config.yaml").read_text())
        env = make_env(config["env"], config.get("env_kwargs"))
        agent = make_agent(config, env)
        agent.load(args.run / "final.pt")
        result = evaluate(agent, config["env"], config.get("env_kwargs"), args.episodes, seed=20000)
        env.close()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
