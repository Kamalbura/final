#!/usr/bin/env python3
"""
Demonstration script for the Lookup Table-Based RL Agent
Shows deterministic decision-making with various scenarios
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
from lookup_table_rl_agent import LookupTableRLAgent

def create_expert_lookup_table():
    """Create lookup table from user-provided expert knowledge"""
    lookup = {}
    
    # Direct mapping from provided lookup table
    # Battery Level × Temperature × Threat State → Action
    entries = [
        # 0-20% battery entries
        (("0-20%", "Safe", "Normal"), 0),
        (("0-20%", "Safe", "Confirming"), 0),
        (("0-20%", "Safe", "Confirmed"), 0),
        (("0-20%", "Warning", "Normal"), 0),
        (("0-20%", "Warning", "Confirming"), 0),
        (("0-20%", "Warning", "Confirmed"), 0),
        (("0-20%", "Critical", "Normal"), 0),
        (("0-20%", "Critical", "Confirming"), 0),
        (("0-20%", "Critical", "Confirmed"), 0),
        
        # 21-40% battery entries
        (("21-40%", "Safe", "Normal"), 0),
        (("21-40%", "Safe", "Confirming"), 1),
        (("21-40%", "Safe", "Confirmed"), 1),
        (("21-40%", "Warning", "Normal"), 0),
        (("21-40%", "Warning", "Confirming"), 1),
        (("21-40%", "Warning", "Confirmed"), 1),
        (("21-40%", "Critical", "Normal"), 0),
        (("21-40%", "Critical", "Confirming"), 0),
        (("21-40%", "Critical", "Confirmed"), 0),
        
        # 41-60% battery entries
        (("41-60%", "Safe", "Normal"), 1),
        (("41-60%", "Safe", "Confirming"), 2),
        (("41-60%", "Safe", "Confirmed"), 1),
        (("41-60%", "Warning", "Normal"), 1),
        (("41-60%", "Warning", "Confirming"), 2),
        (("41-60%", "Warning", "Confirmed"), 1),
        (("41-60%", "Critical", "Normal"), 0),
        (("41-60%", "Critical", "Confirming"), 0),
        (("41-60%", "Critical", "Confirmed"), 0),
        
        # 61-80% battery entries
        (("61-80%", "Safe", "Normal"), 1),
        (("61-80%", "Safe", "Confirming"), 2),
        (("61-80%", "Safe", "Confirmed"), 1),
        (("61-80%", "Warning", "Normal"), 1),
        (("61-80%", "Warning", "Confirming"), 2),
        (("61-80%", "Warning", "Confirmed"), 1),
        (("61-80%", "Critical", "Normal"), 0),
        (("61-80%", "Critical", "Confirming"), 0),
        (("61-80%", "Critical", "Confirmed"), 0),
        
        # 81-100% battery entries
        (("81-100%", "Safe", "Normal"), 1),
        (("81-100%", "Safe", "Confirming"), 2),
        (("81-100%", "Safe", "Confirmed"), 1),
        (("81-100%", "Warning", "Normal"), 1),
        (("81-100%", "Warning", "Confirming"), 2),
        (("81-100%", "Warning", "Confirmed"), 1),
        (("81-100%", "Critical", "Normal"), 0),
        (("81-100%", "Critical", "Confirming"), 0),
        (("81-100%", "Critical", "Confirmed"), 0),
    ]
    
    for state_key, action in entries:
        lookup[state_key] = action
    
    return lookup

def visualize_lookup_table(lookup_table, action_labels):
    """Visualize the lookup table as a heatmap"""
    # Create directory for visualizations if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Define the dimensions
    battery_levels = ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"]
    temperatures = ["Safe", "Warning", "Critical"]
    threat_states = ["Normal", "Confirming", "Confirmed"]
    
    # Create heatmap data
    for threat in threat_states:
        data = np.zeros((len(battery_levels), len(temperatures)))
        
        for i, battery in enumerate(battery_levels):
            for j, temp in enumerate(temperatures):
                state_key = (battery, temp, threat)
                action = lookup_table.get(state_key, 0)
                data[i, j] = action
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.imshow(data, cmap='RdYlGn_r')
        plt.colorbar(ticks=[0, 1, 2], label='Action')
        
        # Add labels
        plt.title(f'Actions for Threat State: {threat}')
        plt.xlabel('Temperature')
        plt.ylabel('Battery Level')
        plt.xticks(np.arange(len(temperatures)), temperatures)
        plt.yticks(np.arange(len(battery_levels)), battery_levels)
        
        # Add text annotations
        for i in range(len(battery_levels)):
            for j in range(len(temperatures)):
                action = int(data[i, j])
                plt.text(j, i, action_labels[action], 
                         ha="center", va="center", 
                         color="white" if action > 0.5 else "black")
        
        # Save plot
        plt.tight_layout()
        plt.savefig(f'visualizations/lookup_heatmap_{threat}.png')
        plt.close()

def run_simulation(agent, scenarios, duration=20):
    """Run simulation with given scenarios"""
    results = []
    
    for scenario_name, scenario_states in scenarios.items():
        print(f"\n=== Running Scenario: {scenario_name} ===")
        
        # Reset for this scenario
        agent.thermal_simulator = agent.thermal_simulator.__class__()
        agent.power_tracker = agent.power_tracker.__class__()
        
        scenario_results = []
        
        # Run through the scenario
        for t in range(duration):
            # Get state for this timestep (could be dynamic)
            if callable(scenario_states):
                state = scenario_states(t, agent)
            else:
                state = scenario_states
            
            # Make decision
            action = agent.make_decision(state)
            
            # Get measurements
            temperature = agent.thermal_simulator.get_temperature()
            power_rate = agent.power_tracker.get_current_rate()
            
            # Record results
            scenario_results.append({
                'scenario': scenario_name,
                'timestep': t,
                'battery': state['battery'],
                'temperature': temperature,
                'temperature_category': agent.thermal_simulator.get_temperature_category(),
                'threat': state['threat'],
                'action': action,
                'action_label': agent.action_labels[action],
                'power_rate': power_rate
            })
            
            # Brief pause to simulate time passing (thermal effects)
            time.sleep(0.1)
            
            print(f"  Step {t}: {agent.action_labels[action]} at {temperature:.1f}°C")
        
        results.extend(scenario_results)
    
    return pd.DataFrame(results)

def plot_scenario_results(results):
    """Create plots visualizing scenario results"""
    scenarios = results['scenario'].unique()
    
    for scenario in scenarios:
        scenario_data = results[results['scenario'] == scenario]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        fig.suptitle(f'Scenario: {scenario}', fontsize=16)
        
        # 1. Plot actions
        action_map = {'No_DDoS': 0, 'XGBoost': 1, 'TST': 2}
        action_colors = ['green', 'blue', 'red']
        
        for action_label in action_map.keys():
            mask = scenario_data['action_label'] == action_label
            if mask.any():
                axes[0].scatter(
                    scenario_data.loc[mask, 'timestep'],
                    [action_map[action_label]] * mask.sum(),
                    color=action_colors[action_map[action_label]],
                    label=action_label,
                    s=100
                )
        
        axes[0].set_title('Actions Over Time')
        axes[0].set_ylabel('Action')
        axes[0].set_yticks(list(action_map.values()))
        axes[0].set_yticklabels(list(action_map.keys()))
        axes[0].legend()
        
        # 2. Plot temperature
        axes[1].plot(scenario_data['timestep'], scenario_data['temperature'], 'r-', label='Temperature')
        axes[1].axhline(y=55, color='y', linestyle='--', label='Warning Threshold')
        axes[1].axhline(y=70, color='r', linestyle='--', label='Critical Threshold')
        axes[1].set_title('Temperature Over Time')
        axes[1].set_ylabel('Temperature (°C)')
        axes[1].legend()
        
        # 3. Plot power consumption
        axes[2].plot(scenario_data['timestep'], scenario_data['power_rate'], 'b-', label='Power Rate')
        axes[2].set_title('Power Consumption Rate')
        axes[2].set_xlabel('Timestep')
        axes[2].set_ylabel('Power (W)')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(f'lookup_scenario_{scenario.replace(" ", "_")}.png')
        plt.show()

def main():
    # Create agent
    agent = LookupTableRLAgent(learning_enabled=False)
    
    # Override the lookup table with user-provided expert lookup table
    expert_lookup = create_expert_lookup_table()
    agent.lookup_table = expert_lookup
    
    print(f"Loaded user-defined lookup table with {len(expert_lookup)} entries")
    
    # Visualize the lookup table as heatmap
    visualize_lookup_table(expert_lookup, agent.action_labels)
    print("Lookup table visualizations saved to 'visualizations' directory")
    
    # Define test scenarios
    scenarios = {
        "Normal Operation": {
            'battery': 90, 
            'temperature': 40, 
            'threat': "Normal"
        },
        "Confirming Threat": {
            'battery': 80, 
            'temperature': 45, 
            'threat': "Confirming"
        },
        "Critical Battery": {
            'battery': 15, 
            'temperature': 50, 
            'threat': "Confirming"
        },
        "Temperature Rise": lambda t, agent: {
            'battery': 70,
            'temperature': min(85, 40 + t * 2.5),  # Temperature rises over time
            'threat': "Confirming"
        },
        "Threat Escalation": lambda t, agent: {
            'battery': 75,
            'temperature': 50,
            'threat': ["Normal", "Normal", "Confirming", "Confirming", "Confirmed", 
                      "Confirmed", "Confirmed", "Confirmed", "Confirmed", "Confirmed",
                      "Normal", "Normal", "Normal", "Normal", "Normal", "Normal",
                      "Normal", "Normal", "Normal", "Normal"][t]
        },
        "Power Optimization": lambda t, agent: {
            'battery': max(15, 80 - t * 4),  # Battery decreases over time
            'temperature': 50,
            'threat': "Confirming"
        }
    }
    
    # Run simulation
    results = run_simulation(agent, scenarios)
    
    # Plot results
    plot_scenario_results(results)
    
    # Generate lookup table summary
    action_counts = {0: 0, 1: 0, 2: 0}
    for action in expert_lookup.values():
        action_counts[action] = action_counts.get(action, 0) + 1
    
    print("\n=== EXPERT LOOKUP TABLE SUMMARY ===")
    print(f"Total entries: {len(expert_lookup)}")
    print(f"No DDoS actions: {action_counts[0]} ({action_counts[0]/len(expert_lookup)*100:.1f}%)")
    print(f"XGBoost actions: {action_counts[1]} ({action_counts[1]/len(expert_lookup)*100:.1f}%)")
    print(f"TST actions: {action_counts[2]} ({action_counts[2]/len(expert_lookup)*100:.1f}%)")
    
    # Print statistics
    print("\n=== LOOKUP TABLE AGENT PERFORMANCE ===")
    stats = agent.get_performance_stats()
    
    print(f"Total decisions made: {stats['total_decisions']}")
    print("\nAction distribution:")
    for action, count in stats['action_distribution'].items():
        print(f"  {action}: {count} ({stats['action_percentages'].get(action, 0):.1f}%)")
    
    print(f"\nAverage power rate: {stats['avg_power_rate']:.2f}W")
    print(f"Average temperature: {stats['avg_temperature']:.2f}°C")
    print(f"Maximum temperature: {stats['max_temperature']:.2f}°C")
    
    # Print example entries
    print("\nExample lookup table entries for each threat state:")
    for threat in agent.threat_states:
        print(f"\n  Threat state: {threat}")
        for battery in ["0-20%", "41-60%", "81-100%"]:
            for temp in ["Safe", "Critical"]:
                state_key = (battery, temp, threat)
                action = agent.lookup_table.get(state_key, -1)
                if action >= 0:
                    print(f"    {state_key} → {agent.action_labels[action]}")

if __name__ == "__main__":
    main()
