#!/usr/bin/env python3
"""
Visualization utilities for the UAV DDoS defense system
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import json
from typing import Dict, List, Tuple, Optional

def visualize_policy_comparison(agent, environment, save_dir="visualizations"):
    """Visualize comparison between expert and learned policies"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize counters for statistics
    total_states = 0
    matching_states = 0
    mismatch_data = []
    
    # Generate all possible states
    battery_levels = environment.battery_levels
    temperatures = ["Safe", "Warning", "Critical"]
    threat_states = environment.threat_states
    
    # Compare actions for each state
    for battery in battery_levels:
        for temp in temperatures:
            for threat in threat_states:
                # Create state dictionary
                state = {
                    'battery': battery,
                    'temperature': temp,
                    'threat': threat
                }
                
                # Get actions from both policies
                agent_action = agent.make_decision(state)
                expert_action = environment.get_expert_action(state)
                
                # Record match/mismatch
                total_states += 1
                if agent_action == expert_action:
                    matching_states += 1
                else:
                    mismatch_data.append({
                        'battery': battery,
                        'temperature': temp,
                        'threat': threat,
                        'expert_action': expert_action,
                        'agent_action': agent_action
                    })
    
    # Calculate alignment percentage
    alignment_pct = (matching_states / total_states * 100) if total_states > 0 else 0
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Policy Comparison: {alignment_pct:.1f}% Alignment', fontsize=16)
    
    # Create heatmap for each threat state
    for i, threat in enumerate(["Normal", "Confirming", "Confirmed"]):
        # Filter mismatches for this threat
        threat_mismatches = [m for m in mismatch_data if m['threat'] == threat]
        
        # Create mismatch matrix
        mismatch_matrix = np.zeros((5, 3))  # 5 battery levels x 3 temp levels
        
        for m in threat_mismatches:
            # Map battery and temp to indices
            b_idx = battery_levels.index(m['battery'])
            t_idx = temperatures.index(m['temperature'])
            mismatch_matrix[b_idx, t_idx] = 1
        
        # Plot heatmap
        sns.heatmap(mismatch_matrix, 
                   cmap=sns.color_palette(["lightgreen", "red"]), 
                   annot=False, 
                   cbar=False,
                   xticklabels=temperatures,
                   yticklabels=battery_levels,
                   ax=axes[i])
        
        axes[i].set_title(f'Threat: {threat}')
        axes[i].set_xlabel('Temperature')
        axes[i].set_ylabel('Battery Level')
        
        # Add text annotations for mismatches
        for m in threat_mismatches:
            b_idx = battery_levels.index(m['battery'])
            t_idx = temperatures.index(m['temperature'])
            expert_label = environment.action_labels[m['expert_action']]
            agent_label = environment.action_labels[m['agent_action']]
            axes[i].text(t_idx + 0.5, b_idx + 0.5, 
                        f"E:{expert_label[0]}\nA:{agent_label[0]}", 
                        ha='center', va='center', color='black', fontsize=8)
    
    plt.tight_layout()
    filename = f"{save_dir}/policy_comparison.png"
    plt.savefig(filename)
    plt.close()
    
    print(f"Policy comparison visualization saved to {filename}")
    print(f"Alignment rate: {matching_states}/{total_states} states ({alignment_pct:.1f}%)")
    
    return alignment_pct, mismatch_data

def visualize_training_progress(metrics, save_dir="visualizations"):
    """Create visualizations for training metrics"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16)
    
    # Plot rewards
    if 'rewards' in metrics and 'episodes' in metrics:
        axes[0, 0].plot(metrics['episodes'], metrics['rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)
    
    # Plot expert alignment
    if 'expert_alignment' in metrics and 'episodes' in metrics:
        axes[0, 1].plot(metrics['episodes'], metrics['expert_alignment'])
        axes[0, 1].set_title('Expert Alignment')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Alignment Rate')
        axes[0, 1].set_ylim(0, 1.1)
        axes[0, 1].grid(True)
    
    # Plot power consumption
    if 'power_consumption' in metrics and 'episodes' in metrics:
        axes[1, 0].plot(metrics['episodes'], metrics['power_consumption'])
        axes[1, 0].set_title('Power Consumption')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Power (W)')
        axes[1, 0].grid(True)
    
    # Plot epsilon decay
    if 'epsilon_values' in metrics and 'episodes' in metrics:
        axes[1, 1].plot(metrics['episodes'], metrics['epsilon_values'])
        axes[1, 1].set_title('Exploration Rate (Epsilon)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    filename = f"{save_dir}/training_progress.png"
    plt.savefig(filename)
    plt.close()
    
    print(f"Training progress visualization saved to {filename}")

def create_action_distribution_chart(actions_taken, labels, save_dir="visualizations"):
    """Create pie chart showing distribution of actions taken"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate percentages
    total_actions = sum(actions_taken)
    percentages = [count/total_actions*100 for count in actions_taken]
    
    # Create pie chart
    plt.figure(figsize=(10, 7))
    colors = ['#1FB8CD', '#DB4545', '#2E8B57']  # Using consistent colors
    
    # Plot pie chart with percentages
    plt.pie(actions_taken, labels=[f"{label}\n({pct:.1f}%)" for label, pct in zip(labels, percentages)],
           autopct='%1.1f%%', startangle=90, colors=colors,
           wedgeprops={'edgecolor': 'white', 'linewidth': 1})
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.title('Distribution of Actions Taken', fontsize=16)
    
    filename = f"{save_dir}/action_distribution.png"
    plt.savefig(filename)
    plt.close()
    
    print(f"Action distribution chart saved to {filename}")

def load_lookup_table(filepath):
    """Load lookup table from JSON file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Extract lookup table
        lookup_table = {}
        for k, v in data['lookup_table'].items():
            parts = k.split('|')
            lookup_table[(parts[0], parts[1], parts[2])] = v
        
        # Extract state space and action labels
        state_space = data.get('state_space', {})
        action_labels = data.get('action_labels', ["No_DDoS", "XGBoost", "TST"])
        
        return lookup_table, state_space, action_labels
    except Exception as e:
        print(f"Error loading lookup table: {e}")
        return {}, {}, ["No_DDoS", "XGBoost", "TST"]


def visualize_expert_lookup_table(
    lookup_table_path: str = "models/lookup_table_expert.json",
    save_dir: str = "visualizations/expert",
    show: bool = False,
    save: bool = True,
    filename_prefix: str = "expert_policy"
) -> Optional[str]:
    """Visualize the expert lookup table as heatmaps per threat state.

    Parameters:
        lookup_table_path: Path to the JSON file containing the expert lookup table.
        save_dir: Directory to save the generated figures.
        show: If True, display the plots interactively.
        save: If True, save the plots to disk.
        filename_prefix: Prefix for saved filenames.

    Returns:
        Path to the combined figure if saved, else None.
    """
    if not os.path.exists(lookup_table_path):
        print(f"Lookup table not found at {lookup_table_path}")
        return None

    lookup_table, state_space, action_labels = load_lookup_table(lookup_table_path)
    if not lookup_table:
        print("Failed to load lookup table data")
        return None

    battery_levels = state_space.get("battery_levels", [])
    temperatures = state_space.get("temperatures", [])
    threat_states = state_space.get("threat_states", [])

    if not (battery_levels and temperatures and threat_states):
        print("State space incomplete in lookup table file")
        return None

    os.makedirs(save_dir, exist_ok=True)

    # Map actions to integers (already stored) and prepare matrices per threat state
    threat_matrices = {}
    for threat in threat_states:
        mat = np.zeros((len(battery_levels), len(temperatures)))
        for b_idx, b in enumerate(battery_levels):
            for t_idx, temp in enumerate(temperatures):
                key = (b, temp, threat)
                action = lookup_table.get(key, 0)
                mat[b_idx, t_idx] = action
        threat_matrices[threat] = mat

    # Colormap for three actions
    action_colors = ["#2E8B57", "#1F77B4", "#D62728"]  # green, blue, red
    cmap = LinearSegmentedColormap.from_list("expert_actions", action_colors, N=len(action_labels))

    # Create figure with one row per threat state
    fig, axes = plt.subplots(1, len(threat_states), figsize=(6 * len(threat_states), 6), sharey=True)
    if len(threat_states) == 1:
        axes = [axes]

    for idx, threat in enumerate(threat_states):
        data = threat_matrices[threat].astype(float)
        sns.heatmap(
            data,
            ax=axes[idx],
            cmap=cmap,
            annot=True,
            fmt=".0f",
            cbar=False,
            xticklabels=temperatures,
            yticklabels=battery_levels,
            linewidths=0.5,
            linecolor="white",
        )
        axes[idx].set_title(f"Expert Policy - {threat}")
        axes[idx].set_xlabel("Temperature")
        if idx == 0:
            axes[idx].set_ylabel("Battery Level")

    # Add legend
    legend_elements = [Patch(facecolor=action_colors[i], label=label) for i, label in enumerate(action_labels)]
    fig.legend(handles=legend_elements, loc="upper center", ncol=len(action_labels), bbox_to_anchor=(0.5, 0.04))
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    out_path = None
    if save:
        timestamp = os.path.splitext(os.path.basename(lookup_table_path))[0].split('_')[-1]
        out_path = os.path.join(save_dir, f"{filename_prefix}_{timestamp}.png")
        plt.savefig(out_path, dpi=150)
        print(f"Expert lookup table visualization saved to {out_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return out_path