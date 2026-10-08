"""Small, composable research figures. All functions return figures, never call show()."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
import yaml


COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
LABELS = {"dqn": "DQN", "dqn_per": "DQN + PER", "ppo": "PPO"}


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor("white")
    ax.figure.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color("#D5DCE1")
    ax.tick_params(colors="#52616B", labelsize=9, length=3)
    ax.grid(axis="y", color="#E8EDF0", linewidth=0.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color="#20313C", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color="#52616B", fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color="#52616B", fontsize=10)


def save_figure(fig, path):
    """Export matching vector and high-resolution raster assets."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        fig.savefig(
            path.with_suffix("." + extension), dpi=300, bbox_inches="tight", facecolor="white"
        )


def plot_training_results(rows, axes=None, window=25):
    keys = [
        key
        for key in (
            "episode_return",
            "episode_length",
            "loss",
            "policy_loss",
            "value_loss",
            "entropy",
        )
        if any(key in row and np.isfinite(row[key]) for row in rows)
    ]
    keys = keys or ["episode_return"]
    if axes is None:
        fig, axes = plt.subplots(
            len(keys), 1, figsize=(8, 2.3 * len(keys)), squeeze=False, layout="constrained"
        )
    else:
        fig = np.atleast_1d(axes).flat[0].figure
    axes = np.asarray(axes).ravel()
    if len(axes) < len(keys):
        raise ValueError("provide an axis for each available metric")
    for ax, key in zip(axes, keys):
        selected = [row for row in rows if key in row and np.isfinite(row[key])]
        x, y = np.array([row["step"] for row in selected]), np.array([row[key] for row in selected])
        ax.plot(x, y, color=COLORS[0], alpha=0.25, linewidth=0.7, label="Raw")
        width = min(window, len(y))
        if width > 1:
            ax.plot(
                x[width - 1 :],
                np.convolve(y, np.ones(width) / width, mode="valid"),
                color=COLORS[0],
                linewidth=1.8,
                label=f"Mean · {width} observations",
            )
        elif len(y):
            ax.scatter(x, y, s=20, color=COLORS[0])
        else:
            ax.text(0.5, 0.5, "No observations recorded", ha="center", transform=ax.transAxes)
        style(ax, key.replace("_", " ").capitalize(), "Environment steps")
        if len(y):
            ax.legend(frameon=False, fontsize=8)
    return fig


def grid_background(env, ax):
    for state, reward in env.terminal_states.items():
        row, col = env._state_to_pos(state)
        ax.add_patch(
            Rectangle(
                (col - 0.5, row - 0.5),
                1,
                1,
                facecolor="#D9EFE6" if reward > 0 else "#FBE5D8",
                edgecolor="none",
                zorder=2,
            )
        )
        ax.text(
            col,
            row - 0.32,
            f"{'Goal' if reward > 0 else 'Trap'} {reward:+g}",
            ha="center",
            va="center",
            fontsize=8,
            color="#17634C" if reward > 0 else "#9D4318",
            zorder=5,
        )
    row, col = env._state_to_pos(env.start_state)
    ax.text(col, row - 0.32, "Start", ha="center", fontsize=8, color="#20313C", zorder=6)
    ax.scatter(col, row, marker="s", s=100, facecolors="none", edgecolors="#20313C", zorder=6)
    ax.set(xlim=(-0.5, env.size - 0.5), ylim=(env.size - 0.5, -0.5), aspect="equal")
    ax.set_xticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.size, 1), minor=True)
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.grid(False)
    ax.grid(which="minor", color="#DDE4E8", linewidth=0.7)
    ax.tick_params(which="minor", length=0)


def plot_policy(policy, env, ax=None, title="Greedy policy"):
    if ax is None:
        _, ax = plt.subplots(figsize=(5.3, 5.3), layout="constrained")
    style(ax, title, "Column", "Row")
    grid_background(env, ax)
    for state in range(env.n_states):
        if state in env.terminal_states:
            continue
        row, col = env._state_to_pos(state)
        probabilities = np.asarray(policy.get(state, []))
        if probabilities.size != env.n_actions or probabilities.sum() <= 0:
            ax.text(col, row, "?", ha="center", va="center", color="#89969E")
            continue
        best = int(probabilities.argmax())
        dr, dc = env.action_effects[best]
        ax.annotate(
            "",
            xy=(col + dc * 0.28, row + dr * 0.28),
            xytext=(col - dc * 0.18, row - dr * 0.18),
            arrowprops=dict(arrowstyle="-|>", color=COLORS[0], lw=1.6),
        )
        if env.size <= 9 and not np.isclose(probabilities[best], 1.0):
            ax.text(
                col + 0.25,
                row + 0.32,
                f"{probabilities[best]:.0%}",
                fontsize=6,
                color="#52616B",
                ha="center",
            )
    return ax.figure


def plot_value_comparison(values_dict, env, axes=None):
    if not values_dict:
        raise ValueError("at least one value function is required")
    if axes is None:
        fig, axes = plt.subplots(
            1,
            len(values_dict),
            figsize=(4.6 * len(values_dict), 4.4),
            squeeze=False,
            layout="constrained",
        )
    else:
        fig = np.atleast_1d(axes).flat[0].figure
    axes = np.asarray(axes).ravel()
    if len(axes) < len(values_dict):
        raise ValueError("one axis is required per value function")
    values = np.concatenate([np.asarray(v).ravel() for v in values_dict.values()])
    norm = Normalize(vmin=min(0, values.min()), vmax=max(1e-8, values.max()))
    for ax, (name, value) in zip(axes, values_dict.items()):
        style(ax, name, "Column", "Row")
        im = ax.imshow(np.asarray(value).reshape(env.size, env.size), cmap="cividis", norm=norm)
        for state, number in enumerate(value):
            if state not in env.terminal_states:
                row, col = env._state_to_pos(state)
                ax.text(
                    col,
                    row,
                    f"{number:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if norm(number) < 0.55 else "#20313C",
                )
        grid_background(env, ax)
    fig.colorbar(
        im, ax=list(axes[: len(values_dict)]), shrink=0.75, label="Expected discounted return"
    )
    return fig


def visualize_episode_trajectory(env, transitions, ax=None, title="Episode trajectory"):
    if ax is None:
        _, ax = plt.subplots(figsize=(5.3, 5.3), layout="constrained")
    total = sum(t["reward"] for t in transitions)
    style(ax, f"{title}\n{len(transitions)} steps · return {total:+.2f}", "Column", "Row")
    grid_background(env, ax)
    if transitions:
        states = [transitions[0]["state"]] + [t["next_state"] for t in transitions]
        positions = np.asarray([env._state_to_pos(s) for s in states])
        ax.plot(positions[:, 1], positions[:, 0], color=COLORS[0], lw=2, alpha=0.85, zorder=3)
        stride = max(1, len(transitions) // 25)
        for i in range(0, len(transitions), stride):
            start, end = positions[i], positions[i + 1]
            ax.annotate(
                "",
                xy=end[::-1],
                xytext=start[::-1],
                arrowprops=dict(arrowstyle="->", color=COLORS[0], lw=1.3),
                zorder=4,
            )
        ax.scatter(
            *positions[-1][::-1],
            s=180,
            marker="o",
            facecolors="none",
            edgecolors=COLORS[0],
            lw=2,
            zorder=6,
        )
    return ax.figure


def plot_experiment(directory, output=None):
    """Rebuild reports solely from saved JSON/CSV data; no training is performed."""
    directory = Path(directory)
    output = Path(output) if output is not None else directory / "figures"
    output.mkdir(parents=True, exist_ok=True)
    if (directory / "metrics.csv").exists():
        with (directory / "metrics.csv").open() as file:
            rows = [{k: float(v) for k, v in row.items() if v} for row in csv.DictReader(file)]
        fig = plot_training_results(rows)
        save_figure(fig, output / "training")
        plt.close(fig)
        return
    results = json.loads((directory / "summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), layout="constrained")
    algorithms = list(dict.fromkeys(r["algorithm"] for r in results))
    for index, algorithm in enumerate(algorithms):
        runs = [r for r in results if r["algorithm"] == algorithm and r["status"] == "complete"]
        color = COLORS[index % len(COLORS)]
        series = []
        for run in runs:
            path = directory / f"{algorithm}-seed{run['seed']}" / "evaluations.json"
            if path.exists():
                series.append({r["step"]: r["mean_return"] for r in json.loads(path.read_text())})
        if series:
            steps = sorted(set.intersection(*(set(s) for s in series)))
            if steps:
                values = np.array([[s[t] for t in steps] for s in series])
                mean, std = values.mean(axis=0), values.std(axis=0)
                axes[0].plot(
                    steps,
                    mean,
                    color=color,
                    lw=2,
                    linestyle=["-", "--", "-."][index % 3],
                    label=f"{LABELS.get(algorithm, algorithm)} · n={len(runs)}",
                )
                if len(runs) > 1:
                    axes[0].fill_between(steps, mean - std, mean + std, color=color, alpha=0.12)
        if runs:
            scores = np.array([r["mean_return"] for r in runs])
            axes[1].scatter(
                index + np.linspace(-0.09, 0.09, len(scores)), scores, color=color, s=35, alpha=0.7
            )
            axes[1].errorbar(
                index,
                scores.mean(),
                yerr=scores.std(),
                color="#20313C",
                marker="_",
                markersize=22,
                capsize=5,
                lw=1.5,
            )
    baseline_path = directory / "random_baseline.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())["mean_return"]
        axes[0].axhline(baseline, color="#89969E", ls=":", lw=1.3, label="Random policy")
        axes[1].axhline(baseline, color="#89969E", ls=":", lw=1.3)
    style(axes[0], "Learning curves", "Environment steps", "Evaluation return")
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))
    style(axes[1], "Final evaluation · individual seeds", "", "Per-seed mean return")
    axes[1].set_xticks(range(len(algorithms)), [LABELS.get(a, a) for a in algorithms])
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.95,
            fontsize=8,
            loc="lower right",
        )
    failures = sum(r["status"] != "complete" for r in results)
    configs = list(directory.glob("*/config.yaml"))
    config = yaml.safe_load(configs[0].read_text()) if configs else {}
    fig.suptitle(
        f"{config.get('env', 'Experiment')}  /  {config.get('total_timesteps', 0):,} steps per run  /  mean ± 1 SD across seeds"
        + (f"  /  {failures} incomplete" if failures else ""),
        fontsize=11,
        color="#52616B",
    )
    save_figure(fig, output / "benchmark")
    plt.close(fig)


class Visualizer:
    """Optional namespace for classroom notebooks; module functions work identically."""

    plot_training_results = staticmethod(plot_training_results)
    plot_policy = staticmethod(plot_policy)
    plot_value_comparison = staticmethod(plot_value_comparison)
    visualize_episode_trajectory = staticmethod(visualize_episode_trajectory)
