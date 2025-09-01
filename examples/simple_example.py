#!/usr/bin/env python3
"""
Simple example showing how to use the UAV DDoS Defense System
"""

import os
import sys
import time
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from src.agents.lookup_table_agent import LookupTableAgent
from src.environments.uav_ddos_env import UAVDDoSEnvironment

def run_example():
    """Run a simple example of the UAV DDoS Defense System"""
    print("=== UAV DDoS Defense System - Simple Example ===")
    
    # Create environment and agent
    env = UAVDDoSEnvironment()
    agent = LookupTableAgent(learning_enabled=False)  # No learning for this example
    
    # Load expert lookup table
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'models', 'lookup_table_expert.json')
    agent.load_model(model_path)
    
    print(f"Loaded expert lookup table")
    
    # Run a simple simulation
    print("\n=== Running Simple Simulation ===")
    state = env.reset()
    
    # Initialize metrics
    total_reward = 0
    expert_agreements = 0
    total_decisions = 0
    
    # Simulate for 20 steps
    for step in range(20):
        # Print current state
        battery = state['battery']
        temperature = env.thermal_simulator.get_temperature()
        temp_category = env.thermal_simulator.get_temperature_category()
        threat = state['threat']
        threat_label = env.threat_states[threat] if isinstance(threat, int) else threat
        
        print(f"\nStep {step+1}:")
        print(f"  State: Battery={battery:.1f}%, Temperature={temperature:.1f}°C ({temp_category}), Threat={threat_label}")
        
        # Get action from agent
        action = agent.make_decision(state)
        action_label = agent.action_labels[action]
        
        # Get expert action for comparison
        expert_action = env.get_expert_action(state)
        expert_label = env.action_labels[expert_action]
        
        # Check agreement
        if action == expert_action:
            expert_agreements += 1
        
        # Take action in environment
        next_state, reward, done = env.step(action)
        
        # Update metrics
        total_reward += reward
        total_decisions += 1
        
        # Print decision and result
        print(f"  Decision: {action_label}")
        print(f"  Expert recommendation: {expert_label}")
        print(f"  Reward: {reward:.2f}")
        print(f"  New temperature: {env.thermal_simulator.get_temperature():.1f}°C")
        print(f"  Power consumption: {env.power_monitor.calculate_power(action_label, env.thermal_simulator):.2f}W")
        
        # Update state
        state = next_state
        
        # Check if episode is done
        if done:
            print(f"\nEpisode ended early at step {step+1}: {state}")
            break
        
        # Small delay for readability
        time.sleep(0.5)
    
    # Print final stats
    print("\n=== Simulation Complete ===")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Expert agreement: {expert_agreements}/{total_decisions} decisions ({expert_agreements/total_decisions:.1%})")
    print(f"Final temperature: {env.thermal_simulator.get_temperature():.1f}°C")
    print(f"Total power consumed: {env.total_power_consumed:.2f}W")
    print(f"Safety violations: {env.safety_violations}")

if __name__ == "__main__":
    run_example()
