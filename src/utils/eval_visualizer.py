import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

"""Evaluation visualization helpers.

create_policy_heatmap previously failed with ValueError: Unknown format code 'd' for object of type 'float'
because seaborn's heatmap default annotation format was '%d' (via fmt='d') applied to float arrays.
We ensure arrays are float and specify fmt='.0f'.
"""
def create_policy_heatmap(expert_data, agent_data, battery_levels, temperatures, save_path=None):
    """Create heatmap comparing expert and agent policies"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Custom colormap for actions
    action_colors = ["green", "blue", "red"]
    action_cmap = LinearSegmentedColormap.from_list("action_cmap", action_colors, N=3)
    
    # Convert action values to float to avoid formatting errors
    expert_data = expert_data.astype(float)
    agent_data = agent_data.astype(float)
    
    # Create heatmaps
    sns.heatmap(expert_data, ax=axes[0], cmap=action_cmap, annot=True, 
                fmt=".0f", cbar=False, xticklabels=temperatures, yticklabels=battery_levels)
    sns.heatmap(agent_data, ax=axes[1], cmap=action_cmap, annot=True, 
                fmt=".0f", cbar=False, xticklabels=temperatures, yticklabels=battery_levels)
    
    # Add titles
    axes[0].set_title("Expert Policy")
    axes[1].set_title("Agent Policy")
    
    # Add labels
    for ax in axes:
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Battery Level")
    
    # Add legend manually
    legend_elements = [Patch(facecolor=color, label=label) 
                      for color, label in zip(action_colors, ["No_DDoS", "XGBoost", "TST"])]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        return fig
