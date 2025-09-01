#!/usr/bin/env python3
"""
Training script for the Hybrid RL Agent with enhanced power and thermal modeling
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from hybrid_rl_agent import HybridRLAgent
from uav_ddos_training import UAVDDoSEnvironment

def plot_training_metrics(metrics, save_path=None):
    """Plot training metrics"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Hybrid RL Agent Training Metrics', fontsize=16)
    
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
    
    # Temperature
    axes[1, 1].plot(metrics['episodes'], metrics['thermal_metrics'])
    axes[1, 1].axhline(y=70, color='r', linestyle='--', label='Warning Threshold')
    axes[1, 1].set_title('Temperature Progression')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Temperature (°C)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Epsilon decay
    axes[2, 0].plot(metrics['episodes'], metrics['epsilon_values'])
    axes[2, 0].set_title('Exploration Rate (Epsilon)')
    axes[2, 0].set_xlabel('Episode')
    axes[2, 0].set_ylabel('Epsilon')
    axes[2, 0].grid(True)
    
    # Neural usage
    axes[2, 1].plot(metrics['episodes'], metrics['neural_usage'])
    axes[2, 1].set_title('Neural Network Contribution')
    axes[2, 1].set_xlabel('Episode')
    axes[2, 1].set_ylabel('Neural Usage Rate')
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    if save_path:
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
            action = agent.get_action(state, training=False)
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

def main():
    print("="*80)
    print("HYBRID RL AGENT TRAINING WITH POWER & THERMAL OPTIMIZATION")
    print("="*80)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create directories for outputs
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Initialize environment and agent
    env = UAVDDoSEnvironment()
    agent = HybridRLAgent(env)
    
    # Training configuration
    num_episodes = 300  # More episodes for hybrid learning
    
    # Train the agent
    print(f"\nTraining for {num_episodes} episodes...")
    training_metrics = agent.train(num_episodes=num_episodes)
    
    # Save the model
    model_path = f"models/hybrid_uav_ddos_model_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    agent.save_model(model_path)
    
    # Plot training metrics
    plot_path = f"plots/hybrid_training_metrics_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
    plot_training_metrics(training_metrics, plot_path)
    
    # Evaluate the trained agent
    evaluate_agent(agent, env)
    
    print("\n✅ Training and evaluation complete!")

if __name__ == "__main__":
    main()
