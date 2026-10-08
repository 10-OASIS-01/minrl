import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from minrl import GridWorld
from minrl.utils.visualization import (
    plot_training_results,
    plot_policy,
    plot_value_comparison,
    visualize_episode_trajectory,
    plot_experiment,
    save_figure,
)


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"step": 1, "episode_return": 2}],
        [{"step": 2, "policy_loss": 0.1}, {"step": 8, "value_loss": 5.0}],
        [{"step": 3, "entropy": float("nan")}],
    ],
)
def test_sparse_training_figures(rows, tmp_path):
    before = dict(plt.rcParams)
    figure = plot_training_results(rows)
    save_figure(figure, tmp_path / "training")
    assert (tmp_path / "training.svg").exists()
    assert (tmp_path / "training.png").stat().st_size > 0
    assert dict(plt.rcParams) == before
    plt.close(figure)


def test_grid_policy_and_last_transition():
    env = GridWorld(terminal_states={2: 1.0})
    figure, axes = plt.subplots(1, 2)
    assert plot_policy({0: [0, 1, 0, 0]}, env, ax=axes[0]) is figure
    labels = [text.get_text() for text in axes[0].texts]
    assert "?" in labels and "Goal +1" in labels
    trajectory = [
        dict(state=0, action=1, reward=-0.1, next_state=1, terminated=False, truncated=False),
        dict(state=1, action=1, reward=1.0, next_state=2, terminated=True, truncated=False),
    ]
    visualize_episode_trajectory(env, trajectory, ax=axes[1])
    assert axes[1].lines[0].get_xdata()[-1] == 2
    assert axes[1].lines[0].get_ydata()[-1] == 0
    plt.close(figure)


def test_value_figures_share_color_scale():
    env = GridWorld()
    figure = plot_value_comparison({"one": np.arange(9), "two": np.arange(9) * 2}, env)
    assert figure.axes[0].images[0].norm is figure.axes[1].images[0].norm
    plt.close(figure)
    figure = plot_value_comparison({"one": np.zeros(9)}, env)
    plt.close(figure)


def test_plot_incomplete_benchmark(tmp_path):
    (tmp_path / "summary.json").write_text(
        json.dumps([dict(algorithm="ppo", seed=0, status="failed")])
    )
    plot_experiment(tmp_path)
    assert (tmp_path / "figures/benchmark.png").exists()
