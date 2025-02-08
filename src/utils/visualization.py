import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
import seaborn as sns


class Visualizer:
    """Utility class for visualizing RL training results and policies"""

    @staticmethod
    def plot_training_results(rewards: List[float],
                              lengths: List[float],
                              losses: Optional[List[float]] = None,
                              title: str = "Training Results"):
        """Plot training metrics"""
        n_plots = 3 if losses is not None else 2
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))

        # Plot rewards
        axes[0].plot(rewards)
        axes[0].set_title(f'{title} - Episode Rewards')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')

        # Plot episode lengths
        axes[1].plot(lengths)
        axes[1].set_title('Episode Lengths')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Steps')

        # Plot losses if provided
        if losses is not None:
            axes[2].plot(losses)
            axes[2].set_title('Training Loss')
            axes[2].set_xlabel('Episode')
            axes[2].set_ylabel('Loss')

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_value_function(values: np.ndarray, size: int, title: str = "Value Function"):
        """Plot state values as a heatmap"""
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(values.reshape(size, size),
                    annot=True,
                    fmt='.2f',
                    cmap='RdYlBu')
        ax.set_title(title)
        return fig

    @staticmethod
    def plot_policy(policy: Dict[int, List[float]],
                    size: int,
                    title: str = "Policy"):
        """Plot policy as arrows in a grid"""
        fig, ax = plt.subplots(figsize=(8, 8))
        action_symbols = ['↑', '→', '↓', '←']

        for i in range(size):
            for j in range(size):
                state = i * size + j
                action_probs = policy[state]
                action_idx = np.argmax(action_probs)

                ax.text(j + 0.5, i + 0.5,
                        action_symbols[action_idx],
                        ha='center', va='center',
                        fontsize=20)

        ax.grid(True)
        ax.set_xlim(0, size)
        ax.set_ylim(size, 0)
        ax.set_title(title)

        return fig
