#!/usr/bin/env python3
"""
Training script for Lookup Table-Based RL Agent for UAV DDoS defense
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import our modules
from src.agents.lookup_table_agent import LookupTableAgent
from src.environments.uav_ddos_env import UAVDDoSEnvironment

def plot_training_metrics(metrics, save_path=None):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Lookup Table-Based RL Training Metrics', fontsize=16)
    
    # Rewards
    axes[0, 0].plot(metrics['episodes'], metrics['rewards'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    
    # Expert alignment
    axes[0, 1].plot(metrics['episodes'], metrics['expert_alignment'])
    axes[0, 1].set_title('Expert Alignment')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Alignment Rate')
    axes[0, 1].set_ylim(0, 1.1)
    axes[0, 1].grid(True)
    
    # Power consumption
    axes[1, 0].plot(metrics['episodes'], metrics['power_consumption'])
    axes[1, 0].set_title('Power Consumption per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Power (W)')
    axes[1, 0].grid(True)
    
    # Epsilon decay
    axes[1, 1].plot(metrics['episodes'], metrics['epsilon_values'])
    axes[1, 1].set_title('Exploration Rate (Epsilon)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Epsilon')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Training metrics plot saved to: {save_path}")
    
    plt.show()

def evaluate_agent(agent, env, num_episodes=50):
    """Evaluate the trained agent"""
    print("\nEvaluating agent performance...")
    
    eval_metrics = {
        'rewards': [],
        'expert_alignment': [],
        'safety_violations': [],
        'power_consumption': [],
        'temperature_max': [],
        'action_distribution': [0, 0, 0]
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        expert_agreements = 0
        total_actions = 0
        max_temp = 0
        
        while True:
            # Get action without exploration
            action = agent.make_decision(state)
            expert_action = env.get_expert_action(state)
            
            if action == expert_action:
                expert_agreements += 1
            total_actions += 1
            eval_metrics['action_distribution'][action] += 1
            
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # Track max temperature
            current_temp = env.thermal_simulator.get_temperature()
            max_temp = max(max_temp, current_temp)
            
            state = next_state
            if done:
                break
        
        # Record metrics
        eval_metrics['rewards'].append(total_reward)
        eval_metrics['expert_alignment'].append(expert_agreements / total_actions if total_actions > 0 else 0)
        eval_metrics['safety_violations'].append(env.safety_violations)
        eval_metrics['power_consumption'].append(env.total_power_consumed)
        eval_metrics['temperature_max'].append(max_temp)
    
    # Calculate summary statistics
    print("\nEVALUATION RESULTS:")
    print(f"Average Reward: {np.mean(eval_metrics['rewards']):.2f} ± {np.std(eval_metrics['rewards']):.2f}")
    print(f"Expert Alignment: {np.mean(eval_metrics['expert_alignment']):.2%}")
    print(f"Safety Violations: {sum(eval_metrics['safety_violations'])}")
    print(f"Average Power: {np.mean(eval_metrics['power_consumption']):.2f}W")
    print(f"Max Temperature: {np.max(eval_metrics['temperature_max']):.1f}°C")
    
    print("\nACTION DISTRIBUTION:")
    action_names = ["No DDoS", "XGBoost", "TST"]
    total_actions = sum(eval_metrics['action_distribution'])
    for i, count in enumerate(eval_metrics['action_distribution']):
        print(f"{action_names[i]}: {count} ({count/total_actions:.1%})")
    
    return eval_metrics

def analyze_policy_alignment(agent, env):
    """Analyze alignment between expert policy and learned policy"""
    # Initialize counters
    total_states = 0
    matching_states = 0
    error_states = 0
    
    # Compare actions for all possible states
    print("\nAnalyzing policy alignment...")
    
    mismatches = []
    
    # Use consistent state definitions
    battery_levels = ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"]
    temperatures = ["Safe", "Warning", "Critical"]
    threat_states = ["Normal", "Confirming", "Confirmed"]
    
    for battery in battery_levels:
        for temp in temperatures:
            for threat in threat_states:
                # Create state
                state = {
                    'battery': battery,
                    'temperature': temp,
                    'threat': threat
                }
                
                # Get actions
                try:
                    agent_action = agent.make_decision(state)
                    expert_action = env.get_expert_action(state)
                    
                    # Check alignment
                    aligned = (agent_action == expert_action)
                    total_states += 1
                    if aligned:
                        matching_states += 1
                    else:
                        mismatches.append({
                            'state': state,
                            'expert_action': expert_action,
                            'agent_action': agent_action,
                            'expert_label': agent.action_labels[expert_action],
                            'agent_label': agent.action_labels[agent_action]
                        })
                except Exception as e:
                    error_states += 1
                    print(f"Error analyzing state {state}: {e}")
    
    # Print results
    if total_states > 0:
        alignment_rate = matching_states / total_states
        print(f"Policy alignment: {matching_states}/{total_states} states ({alignment_rate:.1%})")
    else:
        alignment_rate = 0
        print("No valid states analyzed")
    
    if error_states > 0:
        print(f"Errors encountered in {error_states} states")
    
    # Print mismatches if any
    if mismatches:
        print("\nPolicy mismatches:")
        print(f"{'State':<25} {'Expert':<10} {'Agent':<10}")
        print("-" * 45)
        for m in mismatches[:10]:  # Show top 10 mismatches
            state_str = f"{m['state']['battery']}, {m['state']['temperature']}, {m['state']['threat']}"
            print(f"{state_str:<25} {m['expert_label']:<10} {m['agent_label']:<10}")
        
        if len(mismatches) > 10:
            print(f"...and {len(mismatches) - 10} more mismatches")
    else:
        print("\nPerfect policy alignment with expert!")
    
    return alignment_rate, mismatches

def main():
    print("="*80)
    print("LOOKUP TABLE-BASED RL FOR UAV DDoS DEFENSE")
    print("="*80)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # Create directories for outputs
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # Initialize environment and agent
    env = UAVDDoSEnvironment()
    agent = LookupTableAgent(learning_enabled=True)
    
    # Training configuration
    num_episodes = 200  # Q-learning needs fewer episodes with expert initialization
    
    # Train the agent
    print(f"\nTraining for {num_episodes} episodes...")
    start_time = time.time()
    training_metrics = agent.train(env, num_episodes=num_episodes)  # Using named parameter
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save the model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    model_path = f"models/uav_ddos_lookup_model_{timestamp}.json"
    agent.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Plot training metrics
    plot_path = f"visualizations/training_metrics_{timestamp}.png"
    plot_training_metrics(training_metrics, plot_path)
    
    # Evaluate the trained agent
    eval_results = evaluate_agent(agent, env)
    
    # Analyze policy alignment
    alignment_rate, mismatches = analyze_policy_alignment(agent, env)
    
    # Save evaluation results
    eval_data = {
        'timestamp': timestamp,
        'training_time_seconds': training_time,
        'episodes': num_episodes,
        'final_reward': training_metrics['rewards'][-1],
        'expert_alignment': alignment_rate,
        'eval_results': {
            'avg_reward': float(np.mean(eval_results['rewards'])),
            'avg_expert_alignment': float(np.mean(eval_results['expert_alignment'])),
            'total_safety_violations': sum(eval_results['safety_violations']),
            'avg_power': float(np.mean(eval_results['power_consumption'])),
            'action_distribution': eval_results['action_distribution']
        },
        'policy_mismatches': len(mismatches)
    }
    
    eval_path = f"visualizations/eval_results_{timestamp}.json"
    with open(eval_path, 'w') as f:
        json.dump(eval_data, f, indent=2)
    
    print(f"\nEvaluation results saved to {eval_path}")
    print("\n✅ Training and evaluation complete!")

if __name__ == "__main__":
    main()
