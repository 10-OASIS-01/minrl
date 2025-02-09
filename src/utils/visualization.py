import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Tuple
import seaborn as sns


class Visualizer:
    """Utility class for visualizing RL training results and policies"""

    @staticmethod
    def plot_training_results(rewards: List[float],
                            lengths: List[float],
                            losses: Optional[List[float]] = None,
                            title: str = "Training Results",
                            actor_losses: Optional[List[float]] = None,
                            critic_losses: Optional[List[float]] = None):
        """
        Plot training metrics with optional actor-critic specific losses.
        
        Args:
            rewards: List of episode rewards
            lengths: List of episode lengths
            losses: Optional generic losses
            title: Plot title
            actor_losses: Optional actor network losses
            critic_losses: Optional critic network losses
        """
        # Determine number of subplots needed
        n_plots = 2  # Always show rewards and lengths
        if losses is not None:
            n_plots += 1
        if actor_losses is not None and critic_losses is not None:
            n_plots += 1

        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))

        # Plot rewards
        axes[0].plot(rewards)
        axes[0].set_title(f'{title} - Episode Rewards')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        
        # Add moving average for rewards
        window_size = min(50, len(rewards))
        if window_size > 1:
            moving_avg = np.convolve(rewards, 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
            axes[0].plot(range(window_size-1, len(rewards)), 
                        moving_avg, 
                        'r--', 
                        label=f'{window_size}-Episode Moving Average')
            axes[0].legend()

        # Plot episode lengths
        axes[1].plot(lengths)
        axes[1].set_title('Episode Lengths')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Steps')

        # Plot generic losses if provided
        if losses is not None:
            axes[2].plot(losses)
            axes[2].set_title('Training Loss')
            axes[2].set_xlabel('Episode')
            axes[2].set_ylabel('Loss')

        # Plot actor-critic specific losses if provided
        if actor_losses is not None and critic_losses is not None:
            axes[-1].plot(actor_losses, label='Actor Loss')
            axes[-1].plot(critic_losses, label='Critic Loss')
            axes[-1].set_title('Actor-Critic Losses')
            axes[-1].set_xlabel('Episode')
            axes[-1].set_ylabel('Loss')
            axes[-1].legend()

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

        # Default to uniform policy for missing states
        default_policy = [1.0 / len(action_symbols)] * len(action_symbols)

        for i in range(size):
            for j in range(size):
                state = i * size + j
                action_probs = policy.get(state, default_policy)
                action_idx = np.argmax(action_probs)

                # Add different colors or markers for terminal states
                if state in policy and sum(action_probs) > 0:
                    color = 'black'
                    fontsize = 20
                else:
                    color = 'gray'  # Use gray for uniform/default policies
                    fontsize = 16

                ax.text(j + 0.5, i + 0.5,
                        action_symbols[action_idx],
                        ha='center', va='center',
                        fontsize=fontsize,
                        color=color)

        ax.grid(True)
        ax.set_xlim(0, size)
        ax.set_ylim(size, 0)
        ax.set_title(title)

        return fig

    @staticmethod
    def visualize_episode(env: 'GridWorld',
                         states: List[int],
                         actions: List[int],
                         rewards: List[float],
                         title: str = "Episode Visualization"):
        """
        Visualize a complete episode with states, actions, and rewards.
        
        Args:
            env: The GridWorld environment
            states: List of states visited
            actions: List of actions taken
            rewards: List of rewards received
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(title)
        
        # Plot trajectory
        for i, (state, action) in enumerate(zip(states, actions)):
            pos = env._state_to_pos(state)
            ax.plot(pos[1], pos[0], 'bo')  # Plot state
            
            if i < len(states) - 1:
                next_pos = env._state_to_pos(states[i + 1])
                ax.arrow(pos[1], pos[0],
                        next_pos[1] - pos[1],
                        next_pos[0] - pos[0],
                        head_width=0.1, head_length=0.1,
                        fc='blue', ec='blue')
        
        # Plot grid
        ax.grid(True)
        ax.set_xlim(-0.5, env.size - 0.5)
        ax.set_ylim(env.size - 0.5, -0.5)
        
        return fig
                             
    @staticmethod
    def plot_value_comparison(values_dict: Dict[str, np.ndarray], 
                            size: int,
                            suptitle: str = "Policy Evaluation Methods Comparison") -> plt.Figure:
        """
        Plot and compare different value functions side by side.
        
        Args:
            values_dict: Dictionary mapping method names to their value arrays
            size: Size of the grid world
            suptitle: Super title for the entire figure
            
        Returns:
            matplotlib.figure.Figure: The comparison figure
        """
        n_methods = len(values_dict)
        fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
        
        if n_methods == 1:
            axes = [axes]
            
        for ax, (method_name, values) in zip(axes, values_dict.items()):
            sns.heatmap(values.reshape(size, size),
                       annot=True,
                       fmt='.2f',
                       cmap='RdYlBu',
                       ax=ax)
            ax.set_title(method_name)
            
        plt.suptitle(suptitle)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_convergence_analysis(episodes: List[int],
                                errors_dict: Dict[str, List[float]],
                                title: str = "Convergence Analysis",
                                xlabel: str = "Number of Episodes",
                                ylabel: str = "Mean Absolute Error") -> plt.Figure:
        """
        Plot convergence analysis for different methods.
        
        Args:
            episodes: List of episode numbers
            errors_dict: Dictionary mapping method names to their error values
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            
        Returns:
            matplotlib.figure.Figure: The convergence plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for method_name, errors in errors_dict.items():
            ax.plot(episodes, errors, label=method_name)
            
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        return fig

    @staticmethod
    def plot_ppo_training_results(rewards: List[float],
                                lengths: List[float],
                                policy_losses: List[float],
                                value_losses: List[float],
                                entropy_losses: List[float],
                                title: str = "PPO Training Results") -> plt.Figure:
        """
        Plot PPO-specific training results including losses.
        
        Args:
            rewards: List of episode rewards
            lengths: List of episode lengths
            policy_losses: List of policy losses
            value_losses: List of value function losses
            entropy_losses: List of entropy losses
            title: Plot title
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot rewards with moving average
        axes[0, 0].plot(rewards, alpha=0.6, label='Raw Rewards')
        window_size = min(50, len(rewards))
        if window_size > 1:
            moving_avg = np.convolve(rewards, 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
            axes[0, 0].plot(range(window_size-1, len(rewards)), 
                          moving_avg, 
                          'r--', 
                          label=f'{window_size}-Episode Moving Average')
        axes[0, 0].set_title(f'{title}\nEpisode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Plot episode lengths
        axes[0, 1].plot(lengths)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True)

        # Plot policy and value losses
        axes[1, 0].plot(policy_losses, label='Policy Loss')
        axes[1, 0].plot(value_losses, label='Value Loss')
        axes[1, 0].set_title('Policy and Value Losses')
        axes[1, 0].set_xlabel('Update')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Plot entropy loss
        axes[1, 1].plot(entropy_losses, label='Entropy Loss')
        axes[1, 1].set_title('Entropy Loss')
        axes[1, 1].set_xlabel('Update')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        return fig

    @staticmethod
    def visualize_episode_trajectory(env: 'GridWorld',
                                   episode_data: List[Tuple[int, int, float]],
                                   title: str = "Episode Trajectory") -> plt.Figure:
        """
        Visualize a complete episode trajectory with actions and rewards.
        
        Args:
            env: The GridWorld environment
            episode_data: List of (state, action, reward) tuples
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot grid
        for i in range(env.size):
            for j in range(env.size):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False))
        
        # Plot terminal states
        for state, reward in env.terminal_states.items():
            pos = env._state_to_pos(state)
            color = 'green' if reward > 0 else 'red'
            ax.add_patch(plt.Rectangle((pos[1], pos[0]), 1, 1, 
                                     alpha=0.3, color=color))
        
        # Plot trajectory
        for i in range(len(episode_data) - 1):
            state, action, reward = episode_data[i]
            next_state = episode_data[i + 1][0]
            
            current_pos = env._state_to_pos(state)
            next_pos = env._state_to_pos(next_state)
            
            # Plot arrow
            ax.arrow(current_pos[1] + 0.5, current_pos[0] + 0.5,
                    next_pos[1] - current_pos[1],
                    next_pos[0] - current_pos[0],
                    head_width=0.1, head_length=0.1,
                    fc='blue', ec='blue', alpha=0.5)
        
        ax.set_title(title)
        ax.grid(True)
        ax.set_xlim(-0.5, env.size - 0.5)
        ax.set_ylim(env.size - 0.5, -0.5)
        
        return fig
