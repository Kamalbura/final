import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import itertools

# Define the state space dimensions
battery_levels = ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"]
temperatures = ["Safe", "Warning", "Critical"]  
threat_states = ["Normal", "Confirming", "Confirmed"]

# Generate the lookup table with expert decisions (from our previous analysis)
def generate_expert_lookup_table():
    """Generate the 45-entry expert lookup table based on our domain knowledge"""
    lookup_table = []
    
    for battery in battery_levels:
        for temp in temperatures:
            for threat in threat_states:
                # Expert decision logic
                if battery == "0-20%" or temp == "Critical":
                    action = 0  # No DDoS - protect system
                elif threat == "Normal":
                    if battery in ["0-20%", "21-40%"]:
                        action = 0  # No DDoS for low battery normal traffic
                    else:
                        action = 1  # XGBoost for normal traffic
                elif threat == "Confirming":
                    if battery in ["0-20%", "21-40%"]:
                        if battery == "0-20%":
                            action = 0  # Critical battery
                        else:
                            action = 1  # XGBoost for low battery
                    else:
                        if temp == "Critical":
                            action = 0  # No DDoS due to critical temp
                        else:
                            action = 2  # TST for confirmation
                elif threat == "Confirmed":
                    if battery == "0-20%":
                        action = 0  # Critical battery
                    else:
                        action = 1  # XGBoost for monitoring (already confirmed)
                
                lookup_table.append({
                    'battery': battery,
                    'temperature': temp, 
                    'threat': threat,
                    'action': action
                })
    
    return lookup_table

# State encoding function
def get_state_index(battery, temp, threat):
    """Map state tuple to index (0-44)"""
    battery_idx = battery_levels.index(battery)
    temp_idx = temperatures.index(temp)
    threat_idx = threat_states.index(threat)
    return battery_idx * 9 + temp_idx * 3 + threat_idx

# Initialize Q-table from expert lookup table
def initialize_q_table_from_expert():
    """Initialize Q-table using expert knowledge from lookup table"""
    num_states = 45
    num_actions = 3
    
    # Initialize with low default values
    Q = np.full((num_states, num_actions), -1.0)
    
    # Generate expert lookup table
    expert_table = generate_expert_lookup_table()
    
    # Set high Q-values for expert actions
    for entry in expert_table:
        state_idx = get_state_index(entry['battery'], entry['temperature'], entry['threat'])
        expert_action = entry['action']
        Q[state_idx, expert_action] = 10.0  # High initial value for expert action
    
    return Q, expert_table

# Initialize the Q-table
Q_table, expert_lookup = initialize_q_table_from_expert()

print("=== Q-TABLE INITIALIZATION FROM EXPERT LOOKUP TABLE ===")
print(f"Q-table shape: {Q_table.shape}")
print(f"Expert lookup table entries: {len(expert_lookup)}")
print("\nFirst 10 Q-table entries:")
print(Q_table[:10])

# Create a summary of expert actions
expert_actions = [entry['action'] for entry in expert_lookup]
action_counts = pd.Series(expert_actions).value_counts().sort_index()
print(f"\nExpert action distribution:")
print(f"Action 0 (No DDoS): {action_counts.get(0, 0)} states")
print(f"Action 1 (XGBoost): {action_counts.get(1, 0)} states") 
print(f"Action 2 (TST): {action_counts.get(2, 0)} states")