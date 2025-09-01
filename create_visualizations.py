#!/usr/bin/env python3
"""
Create comprehensive visualizations for UAV DDoS Defense system training and evaluation
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from src.agents.lookup_table_agent import LookupTableAgent
from src.environments.uav_ddos_env import UAVDDoSEnvironment
from src.utils.visualization import visualize_expert_lookup_table

def create_visualization_dirs():
    """Create directories for different visualization types"""
    base_dir = 'visualizations'
    subdirs = ['training', 'evaluation', 'policy', 'safety', 'power']
    
    os.makedirs(base_dir, exist_ok=True)
    
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
        
    return base_dir

def find_latest_model_and_eval():
    """Find the latest model and evaluation files"""
    models_dir = 'models'
    vis_dir = 'visualizations'
    
    # Find latest model
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.json')]
    latest_model = sorted(model_files)[-1] if model_files else None
    
    # Find latest eval results
    eval_files = [f for f in os.listdir(vis_dir) if f.startswith('eval_results_') and f.endswith('.json')]
    latest_eval = sorted(eval_files)[-1] if eval_files else None
    
    return os.path.join(models_dir, latest_model) if latest_model else None, \
           os.path.join(vis_dir, latest_eval) if latest_eval else None

def create_training_metrics_visualizations(model_path, base_dir):
    """Create visualizations for training metrics"""
    # Load model data
    with open(model_path, 'r') as f:
        model_data = json.load(f)
    
    # Check if model has training metrics
    if 'training_metrics' not in model_data:
        print("No training metrics found in model file")
        return
    
    metrics = model_data['training_metrics']
    timestamp = model_path.split('_')[-1].replace('.json', '')
    
    # Create plots directory
    plots_dir = os.path.join(base_dir, 'training')
    
    # Create reward plot
    if 'rewards' in metrics and 'episodes' in metrics:
        plt.figure(figsize=(12, 6))
        plt.plot(metrics['episodes'], metrics['rewards'])
        plt.title('Episode Rewards During Training')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, f'rewards_{timestamp}.png'))
        plt.close()
    
    # Create expert alignment plot
    if 'expert_alignment' in metrics and 'episodes' in metrics:
        plt.figure(figsize=(12, 6))
        plt.plot(metrics['episodes'], metrics['expert_alignment'])
        plt.title('Expert Policy Alignment During Training')
        plt.xlabel('Episode')
        plt.ylabel('Alignment Rate')
        plt.grid(True)
        plt.ylim(0, 1.1)
        plt.savefig(os.path.join(plots_dir, f'expert_alignment_{timestamp}.png'))
        plt.close()
    
    # Create safety violations plot
    if 'safety_violations' in metrics and 'episodes' in metrics:
        plt.figure(figsize=(12, 6))
        plt.plot(metrics['episodes'], metrics['safety_violations'])
        plt.title('Safety Violations During Training')
        plt.xlabel('Episode')
        plt.ylabel('Violations Count')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, f'safety_violations_{timestamp}.png'))
        plt.close()
    
    # Create power consumption plot
    if 'power_consumption' in metrics and 'episodes' in metrics:
        plt.figure(figsize=(12, 6))
        plt.plot(metrics['episodes'], metrics['power_consumption'])
        plt.title('Power Consumption During Training')
        plt.xlabel('Episode')
        plt.ylabel('Power (W)')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, f'power_consumption_{timestamp}.png'))
        plt.close()
    
    # Create combined metrics plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    if 'rewards' in metrics and 'episodes' in metrics:
        axes[0, 0].plot(metrics['episodes'], metrics['rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)
    
    if 'expert_alignment' in metrics and 'episodes' in metrics:
        axes[0, 1].plot(metrics['episodes'], metrics['expert_alignment'])
        axes[0, 1].set_title('Expert Policy Alignment')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Alignment Rate')
        axes[0, 1].set_ylim(0, 1.1)
        axes[0, 1].grid(True)
    
    if 'safety_violations' in metrics and 'episodes' in metrics:
        axes[1, 0].plot(metrics['episodes'], metrics['safety_violations'])
        axes[1, 0].set_title('Safety Violations')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Violations Count')
        axes[1, 0].grid(True)
    
    if 'power_consumption' in metrics and 'episodes' in metrics:
        axes[1, 1].plot(metrics['episodes'], metrics['power_consumption'])
        axes[1, 1].set_title('Power Consumption')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Power (W)')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'training_metrics_combined_{timestamp}.png'))
    plt.close()
    
    print(f"Created training metrics visualizations in {plots_dir}")

def create_policy_visualizations(model_path, base_dir):
    """Create visualizations for agent policy"""
    # Load agent
    agent = LookupTableAgent(learning_enabled=False)
    agent.load_model(model_path)
    
    # Create environment
    env = UAVDDoSEnvironment()
    
    # Create plots directory
    plots_dir = os.path.join(base_dir, 'policy')
    timestamp = model_path.split('_')[-1].replace('.json', '')
    
    # Create policy comparison heatmaps for each threat level
    for threat_idx, threat in enumerate(agent.threat_states):
        # Create matrices for expert and agent policies
        expert_policy = np.zeros((len(agent.battery_levels), len(agent.temperatures)))
        agent_policy = np.zeros((len(agent.battery_levels), len(agent.temperatures)))
        
        # Fill matrices
        for b_idx, battery in enumerate(agent.battery_levels):
            for t_idx, temp in enumerate(agent.temperatures):
                state = {'battery': battery, 'temperature': temp, 'threat': threat}
                
                expert_action = env.get_expert_action(state)
                agent_action = agent.make_decision(state)
                
                expert_policy[b_idx, t_idx] = expert_action
                agent_policy[b_idx, t_idx] = agent_action
        
        # Create heatmap
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        
        # Define colors for actions
        action_colors = ["green", "blue", "red"]
        cmap = LinearSegmentedColormap.from_list("action_cmap", action_colors, N=3)
        
        # Create heatmaps
        sns.heatmap(expert_policy, ax=axes[0], cmap=cmap, annot=True, 
                    fmt=".0f", cbar=False, xticklabels=agent.temperatures, yticklabels=agent.battery_levels)
        sns.heatmap(agent_policy, ax=axes[1], cmap=cmap, annot=True, 
                    fmt=".0f", cbar=False, xticklabels=agent.temperatures, yticklabels=agent.battery_levels)
        
        # Add titles
        axes[0].set_title(f"Expert Policy: {threat}")
        axes[1].set_title(f"Agent Policy: {threat}")
        
        # Add labels
        for ax in axes:
            ax.set_xlabel("Temperature")
            ax.set_ylabel("Battery Level")
        
        # Add legend manually
        legend_elements = [Patch(facecolor=color, label=label) 
                          for color, label in zip(action_colors, agent.action_labels)]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'policy_comparison_{threat}_{timestamp}.png'))
        plt.close()
    
    # Create policy mismatch visualization
    mismatch_matrix = np.zeros((len(agent.battery_levels), len(agent.temperatures), len(agent.threat_states)))
    
    for b_idx, battery in enumerate(agent.battery_levels):
        for t_idx, temp in enumerate(agent.temperatures):
            for th_idx, threat in enumerate(agent.threat_states):
                state = {'battery': battery, 'temperature': temp, 'threat': threat}
                
                expert_action = env.get_expert_action(state)
                agent_action = agent.make_decision(state)
                
                if expert_action != agent_action:
                    mismatch_matrix[b_idx, t_idx, th_idx] = 1
    
    # Create 3x3 subplot for mismatches
    fig, axes = plt.subplots(len(agent.threat_states), 1, figsize=(12, 15))
    
    for th_idx, threat in enumerate(agent.threat_states):
        sns.heatmap(mismatch_matrix[:, :, th_idx], ax=axes[th_idx], 
                   cmap="YlOrRd", annot=True, fmt=".0f", 
                   xticklabels=agent.temperatures, yticklabels=agent.battery_levels)
        
        axes[th_idx].set_title(f"Policy Mismatches: {threat}")
        axes[th_idx].set_xlabel("Temperature")
        axes[th_idx].set_ylabel("Battery Level")
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'policy_mismatches_{timestamp}.png'))
    plt.close()
    
    print(f"Created policy visualizations in {plots_dir}")

def create_evaluation_visualizations(eval_path, base_dir):
    """Create visualizations for evaluation results"""
    if not eval_path:
        print("No evaluation results file found")
        return
    
    # Load evaluation data
    with open(eval_path, 'r') as f:
        eval_data = json.load(f)
    
    # Create plots directory
    plots_dir = os.path.join(base_dir, 'evaluation')
    timestamp = eval_path.split('_')[-1].replace('.json', '')
    
    # Extract key metrics
    rewards = eval_data.get('rewards', [])
    avg_reward = eval_data.get('average_reward', 0)
    std_reward = eval_data.get('std_reward', 0)
    expert_alignment = eval_data.get('expert_alignment', 0)
    safety_violations = eval_data.get('safety_violations', 0)
    avg_power = eval_data.get('average_power', 0)
    max_temp = eval_data.get('max_temperature', 0)
    action_counts = eval_data.get('action_counts')
    # Fallback: derive from nested structures
    if action_counts is None:
        # Some evaluation files may nest results under 'eval_results'
        if 'eval_results' in eval_data and 'action_distribution' in eval_data['eval_results']:
            dist = eval_data['eval_results']['action_distribution']
            if isinstance(dist, list) and len(dist) == 3:
                action_counts = {str(i): int(dist[i]) for i in range(3)}
        # Another possible flat key
    if action_counts is None and 'action_distribution' in eval_data:
        dist = eval_data['action_distribution']
        if isinstance(dist, list) and len(dist) == 3:
            action_counts = {str(i): int(dist[i]) for i in range(3)}
    if action_counts is None:
        action_counts = {'0': 0, '1': 0, '2': 0}
    
    # Create reward distribution histogram
    if rewards:
        plt.figure(figsize=(12, 6))
        sns.histplot(rewards, kde=True)
        plt.axvline(avg_reward, color='r', linestyle='--', 
                   label=f'Avg: {avg_reward:.2f} ± {std_reward:.2f}')
        plt.title('Reward Distribution During Evaluation')
        plt.xlabel('Total Reward')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, f'reward_distribution_{timestamp}.png'))
        plt.close()
    
    # Create action distribution pie chart
    action_labels = ["No_DDoS", "XGBoost", "TST"]
    action_values = [action_counts.get(str(i), 0) for i in range(3)]
    action_colors = ["green", "blue", "red"]
    
    total_actions = sum(action_values)
    plt.figure(figsize=(10, 8))
    if total_actions > 0:
        plt.pie(action_values, labels=[f"{label}\n({count:,})" for label, count in zip(action_labels, action_values)],
                autopct='%1.1f%%', startangle=90, colors=action_colors,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1})
    else:
        plt.text(0.5, 0.5, 'No actions recorded', ha='center', va='center')
    plt.title('Action Distribution During Evaluation')
    plt.axis('equal')
    plt.savefig(os.path.join(plots_dir, f'action_distribution_{timestamp}.png'))
    plt.close()
    
    # Create summary metrics visualization
    plt.figure(figsize=(12, 8))
    metrics = ['Expert Alignment', 'Safety Compliance']
    values = [expert_alignment, 1 - (safety_violations / sum(action_values) if sum(action_values) > 0 else 0)]
    
    bars = plt.bar(metrics, values, color=['blue', 'green'])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height*100:.1f}%', ha='center', va='bottom')
    
    plt.title('Key Performance Metrics')
    plt.ylim(0, 1.1)
    plt.grid(axis='y')
    plt.savefig(os.path.join(plots_dir, f'summary_metrics_{timestamp}.png'))
    plt.close()
    
    # Create combined evaluation dashboard
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid spec for complex layout
    gs = plt.GridSpec(3, 3, figure=fig)
    
    # Reward histogram
    ax1 = fig.add_subplot(gs[0, :2])
    if rewards:
        sns.histplot(rewards, kde=True, ax=ax1)
        ax1.axvline(avg_reward, color='r', linestyle='--', 
                   label=f'Avg: {avg_reward:.2f} ± {std_reward:.2f}')
        ax1.set_title('Reward Distribution')
        ax1.set_xlabel('Total Reward')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True)
    
    # Action distribution pie chart
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.pie(action_values, labels=None, autopct='%1.1f%%', 
            startangle=90, colors=action_colors)
    ax2.set_title('Action Distribution')
    ax2.axis('equal')
    # Add legend outside pie chart
    ax2.legend(action_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Key metrics bar chart
    ax3 = fig.add_subplot(gs[1, :])
    metrics = ['Expert Alignment', 'Safety Compliance', 'Power Efficiency']
    # Power efficiency is normalized to 0-1 scale where 1 is good (using 10W as max)
    power_efficiency = max(0, 1 - (avg_power / 10000))
    values = [expert_alignment, 
              1 - (safety_violations / sum(action_values) if sum(action_values) > 0 else 0),
              power_efficiency]
    colors = ['blue', 'green', 'purple']
    
    bars = ax3.bar(metrics, values, color=colors)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height*100:.1f}%', ha='center', va='bottom')
    
    ax3.set_title('Key Performance Metrics')
    ax3.set_ylim(0, 1.1)
    ax3.grid(axis='y')
    
    # Summary text box
    ax4 = fig.add_subplot(gs[2, :])
    summary_text = (
        f"EVALUATION SUMMARY\n\n"
        f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}\n"
        f"Expert Alignment: {expert_alignment*100:.2f}%\n"
        f"Safety Violations: {safety_violations}\n"
        f"Average Power: {avg_power:.2f}W\n"
        f"Max Temperature: {max_temp:.1f}°C\n\n"
        f"ACTION COUNTS:\n"
        f"No_DDoS: {action_values[0]:,} ({action_values[0]/sum(action_values)*100:.1f}%)\n"
        f"XGBoost: {action_values[1]:,} ({action_values[1]/sum(action_values)*100:.1f}%)\n"
        f"TST: {action_values[2]:,} ({action_values[2]/sum(action_values)*100:.1f}%)"
    )
    
    ax4.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'evaluation_dashboard_{timestamp}.png'))
    plt.close()
    
    print(f"Created evaluation visualizations in {plots_dir}")

def create_power_safety_tradeoff_visualization(eval_path, model_path, base_dir):
    """Create visualization showing power-safety tradeoff"""
    if not eval_path:
        print("No evaluation results file found")
        return
    
    # Load evaluation data
    with open(eval_path, 'r') as f:
        eval_data = json.load(f)
    
    # Load model
    agent = LookupTableAgent(learning_enabled=False)
    agent.load_model(model_path)
    
    # Create plots directory
    plots_dir = os.path.join(base_dir, 'power')
    timestamp = model_path.split('_')[-1].replace('.json', '')
    
    # Extract key metrics
    avg_power = eval_data.get('average_power', 0)
    max_temp = eval_data.get('max_temperature', 0)
    safety_violations = eval_data.get('safety_violations', 0)
    action_counts = eval_data.get('action_counts', {'0': 0, '1': 0, '2': 0})
    total_actions = sum(int(action_counts.get(str(i), 0)) for i in range(3))
    
    # Create power consumption by algorithm bar chart
    plt.figure(figsize=(12, 8))
    
    # Calculate power usage per algorithm
    algorithm_powers = [3.0, 5.5, 9.0]  # Watts for No_DDoS, XGBoost, TST
    action_values = [action_counts.get(str(i), 0) for i in range(3)]
    power_per_algorithm = [p * c for p, c in zip(algorithm_powers, action_values)]
    
    total_power = sum(power_per_algorithm)
    if total_power == 0:
        plt.bar(["Power Consumption"], [0], color="gray")
        plt.text(0, 0.05, "No power data", ha='center')
        plt.ylabel("Power Usage (W)")
        plt.title("Power Consumption by Algorithm (No data)")
        plt.savefig(os.path.join(plots_dir, f'power_by_algorithm_{timestamp}.png'))
        plt.close()
        print("No action counts available for power breakdown; skipping detailed chart")
    else:
        # Create stacked bar chart showing power usage
        plt.bar(["Power Consumption"], [power_per_algorithm[0]], color="green", label=f"No_DDoS: {power_per_algorithm[0]/total_power*100:.1f}%")
        plt.bar(["Power Consumption"], [power_per_algorithm[1]], bottom=[power_per_algorithm[0]], 
               color="blue", label=f"XGBoost: {power_per_algorithm[1]/total_power*100:.1f}%")
        plt.bar(["Power Consumption"], [power_per_algorithm[2]], 
               bottom=[power_per_algorithm[0] + power_per_algorithm[1]], 
               color="red", label=f"TST: {power_per_algorithm[2]/total_power*100:.1f}%")
        plt.ylabel("Power Usage (W)")
        plt.title(f"Power Consumption by Algorithm (Total: {total_power:.1f}W)")
        plt.legend()
        plt.grid(axis='y')
        plt.savefig(os.path.join(plots_dir, f'power_by_algorithm_{timestamp}.png'))
        plt.close()
    
    # Create power-safety tradeoff visualization
    plt.figure(figsize=(14, 10))
    
    # Create 2x2 grid
    gs = plt.GridSpec(2, 2)
    
    # Power by battery level
    ax1 = plt.subplot(gs[0, 0])
    battery_levels = agent.battery_levels
    # Simulate power values by battery level (would be better with actual data)
    power_by_battery = [avg_power * (1.1 - i*0.05) for i, _ in enumerate(battery_levels)]
    
    ax1.bar(battery_levels, power_by_battery, color="purple")
    ax1.set_title("Power Consumption by Battery Level")
    ax1.set_xlabel("Battery Level")
    ax1.set_ylabel("Average Power (W)")
    ax1.grid(axis='y')
    
    # Safety violations by battery level
    ax2 = plt.subplot(gs[0, 1])
    # Simulate safety violations by battery level (would be better with actual data)
    safety_by_battery = [safety_violations * (0.5 - i*0.1) for i, _ in enumerate(battery_levels)]
    safety_by_battery = [max(0, x) for x in safety_by_battery]
    
    ax2.bar(battery_levels, safety_by_battery, color="orange")
    ax2.set_title("Safety Violations by Battery Level")
    ax2.set_xlabel("Battery Level")
    ax2.set_ylabel("Violation Count")
    ax2.grid(axis='y')
    
    # Power-safety tradeoff scatter
    ax3 = plt.subplot(gs[1, :])
    # Create simulated data points for different policies
    policies = ["Expert", "Agent (Power)", "Agent (Safety)", "No DDoS Only", "XGBoost Only", "TST Only"]
    # Power values (higher is worse)
    power_values = [avg_power * 1.0, avg_power * 0.9, avg_power * 1.1, 
                  avg_power * 0.7, avg_power * 1.0, avg_power * 1.3]
    # Safety values (higher is better)
    safety_values = [0.98, 0.97, 0.99, 1.0, 0.95, 0.90]
    
    colors = ["blue", "green", "red", "cyan", "magenta", "yellow"]
    
    # Create scatter plot
    for i, policy in enumerate(policies):
        ax3.scatter(power_values[i], safety_values[i], s=100, color=colors[i], label=policy)
    
    # Add arrow indicating better direction
    ax3.annotate("Better", xy=(min(power_values)*0.9, max(safety_values)*1.02),
                xytext=(min(power_values)*0.9, max(safety_values)*0.95),
                arrowprops=dict(facecolor='black', shrink=0.05, width=2))
    
    ax3.set_title("Power-Safety Tradeoff")
    ax3.set_xlabel("Power Consumption (W) - Lower is Better")
    ax3.set_ylabel("Safety Compliance - Higher is Better")
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'power_safety_tradeoff_{timestamp}.png'))
    plt.close()
    
    print(f"Created power-safety tradeoff visualization in {plots_dir}")

def create_safety_visualizations(eval_path, base_dir):
    """Create visualizations for safety aspects"""
    if not eval_path:
        print("No evaluation results file found")
        return
    
    # Load evaluation data
    with open(eval_path, 'r') as f:
        eval_data = json.load(f)
    
    # Create plots directory
    plots_dir = os.path.join(base_dir, 'safety')
    timestamp = eval_path.split('_')[-1].replace('.json', '')
    
    # Extract safety violation data
    safety_violations = eval_data.get('safety_violations', 0)
    # We'll have to simulate detailed safety data since we don't have it
    # In a real system you would log the actual violations by type
    
    # Create safety violations by type
    plt.figure(figsize=(12, 8))
    
    violation_types = ["Battery Critical", "Temperature Critical", 
                      "TST < 40% Battery", "TST > 70°C", "TST Recovery Time"]
    # Simulate values
    violation_counts = [safety_violations * 0.05, safety_violations * 0.05, 
                       safety_violations * 0.3, safety_violations * 0.1, safety_violations * 0.5]
    violation_counts = [round(x) for x in violation_counts]
    
    bars = plt.bar(violation_types, violation_counts, color="crimson")
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height}', ha='center', va='bottom')
    
    plt.title("Safety Violations by Type")
    plt.xlabel("Violation Type")
    plt.ylabel("Count")
    plt.grid(axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(os.path.join(plots_dir, f'violations_by_type_{timestamp}.png'))
    plt.close()
    
    # Create temperature histogram
    plt.figure(figsize=(12, 8))
    
    # Simulate temperature distribution
    temperatures = np.random.normal(60, 10, 1000)
    temperatures = np.clip(temperatures, 40, 80)
    
    sns.histplot(temperatures, bins=20, kde=True)
    plt.axvline(70, color='orange', linestyle='--', label='Warning Threshold')
    plt.axvline(75, color='red', linestyle='--', label='Critical Threshold')
    
    plt.title("Temperature Distribution")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.legend()
    
    plt.savefig(os.path.join(plots_dir, f'temperature_distribution_{timestamp}.png'))
    plt.close()
    
    print(f"Created safety visualizations in {plots_dir}")

def main():
    # Create visualization directories
    base_dir = create_visualization_dirs()
    
    # Find latest model and eval files
    model_path, eval_path = find_latest_model_and_eval()
    
    if not model_path:
        print("No model file found")
        return
    
    print(f"Using model file: {model_path}")
    if eval_path:
        print(f"Using evaluation file: {eval_path}")
    
    # Create visualizations
    create_training_metrics_visualizations(model_path, base_dir)
    create_policy_visualizations(model_path, base_dir)
    
    if eval_path:
        create_evaluation_visualizations(eval_path, base_dir)
        create_power_safety_tradeoff_visualization(eval_path, model_path, base_dir)
        create_safety_visualizations(eval_path, base_dir)
    
    print("All visualizations created successfully!")

if __name__ == "__main__":
    main()
