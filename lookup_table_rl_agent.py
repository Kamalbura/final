#!/usr/bin/env python3
"""
Pure Lookup Table-Based RL Agent for UAV DDoS Detection
Prioritizes deterministic expert knowledge with minimal learning updates
"""

import numpy as np
import json
import os
import time
import logging
from datetime import datetime
from collections import defaultdict

class ThermalSimulator:
    """Simulates thermal behavior of algorithms running on hardware"""
    
    def __init__(self, ambient_temp=25.0, time_constant=60.0):
        # Baseline thermal parameters
        self.ambient_temp = ambient_temp
        self.time_constant = time_constant
        
        # Thermal impact of each action (°C/s at steady state)
        self.action_thermal_impact = {
            0: 0.05,   # No DDoS: minimal heating
            1: 0.12,   # XGBoost: moderate heating
            2: 0.25    # TST: significant heating - can cause overheating
        }
        
        # Current state
        self.current_temp = ambient_temp
        self.last_update_time = time.time()
        self.current_action = 0
    
    def update_temperature(self, action=None):
        """Update temperature based on action and time elapsed"""
        current_time = time.time()
        delta_t = current_time - self.last_update_time
        
        if action is not None:
            self.current_action = action
        
        # Get steady state temperature for current action
        action_impact = self.action_thermal_impact.get(self.current_action, 0.0)
        steady_state_temp = self.ambient_temp + action_impact * 60
        
        # Newton's law of cooling/heating
        self.current_temp = steady_state_temp - (steady_state_temp - self.current_temp) * np.exp(-delta_t / self.time_constant)
        self.last_update_time = current_time
        
        return self.current_temp
    
    def get_temperature(self):
        """Get current temperature with update"""
        return self.update_temperature()
    
    def get_temperature_category(self):
        """Map temperature to category"""
        temp = self.get_temperature()
        if temp <= 55:
            return "Safe"
        elif temp <= 70:
            return "Warning"
        else:
            return "Critical"


class PowerTracker:
    """Tracks power consumption using V×I×Δt method"""
    
    def __init__(self):
        # Power consumption per action (Watts)
        self.power_consumption = {
            0: 3.0,  # No DDoS - baseline system
            1: 5.5,  # XGBoost - moderate
            2: 9.0   # TST - intensive
        }
        
        # Tracking
        self.start_time = time.time()
        self.total_power = 0.0
        self.current_action = None
        self.action_start_time = None
    
    def start_action(self, action):
        """Start tracking a new action"""
        # Finalize previous action if any
        if self.current_action is not None:
            self.update_power()
        
        self.current_action = action
        self.action_start_time = time.time()
    
    def update_power(self):
        """Update power consumption based on current action"""
        if self.current_action is None:
            return 0
        
        now = time.time()
        duration = now - self.action_start_time
        
        # Calculate power used: P = V×I×Δt
        power_rate = self.power_consumption.get(self.current_action, 0)
        power_used = power_rate * duration
        
        self.total_power += power_used
        self.action_start_time = now
        
        return power_used
    
    def get_total_power(self):
        """Get total power consumed so far"""
        self.update_power()
        return self.total_power
    
    def get_current_rate(self):
        """Get current power consumption rate in Watts"""
        if self.current_action is None:
            return 0
        return self.power_consumption.get(self.current_action, 0)


class LookupTableRLAgent:
    """
    Lookup Table-Based RL Agent for UAV DDoS Detection
    Uses deterministic expert knowledge with limited learning
    """
    
    def __init__(self, learning_enabled=True, learning_rate=0.05):
        self.learning_enabled = learning_enabled
        self.learning_rate = learning_rate
        
        # State space definitions
        self.battery_levels = ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"]
        self.temperatures = ["Safe", "Warning", "Critical"]
        self.threat_states = ["Normal", "Confirming", "Confirmed"]
        self.action_labels = ["No_DDoS", "XGBoost", "TST"]
        
        # Create expert lookup table - this is the foundation of our approach
        self.lookup_table = self._create_expert_lookup_table()
        
        # Create experience counter to track state visitation
        self.experience = defaultdict(int)
        
        # Decision history
        self.decisions = []
        
        # Hardware monitors
        self.thermal_simulator = ThermalSimulator()
        self.power_tracker = PowerTracker()
        
        # Initialize logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("lookup_agent.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("LookupTableRL")
        self.logger.info("Lookup Table RL Agent initialized")
    
    def _create_expert_lookup_table(self):
        """
        Create expert lookup table based on deterministic rules
        This is the core of the lookup-based approach
        """
        lookup = {}
        
        # Build comprehensive lookup table for all state combinations
        for battery in self.battery_levels:
            for temp in self.temperatures:
                for threat in self.threat_states:
                    state_key = (battery, temp, threat)
                    
                    # EXPERT DECISION LOGIC - Based on DFA rules and expert knowledge
                    
                    # RULE 1: Critical battery or temperature - always protect system
                    if battery == "0-20%" or temp == "Critical":
                        action = 0  # No DDoS - preserve critical systems
                    
                    # RULE 2: Normal threats - never use TST (power conservation)
                    elif threat == "Normal":
                        if battery == "21-40%":
                            action = 0  # Conservative for low battery
                        else:
                            action = 1  # XGBoost for routine monitoring
                    
                    # RULE 3: Confirming threats - resource-aware TST usage
                    elif threat == "Confirming":
                        if battery == "21-40%":
                            action = 1  # XGBoost for low battery
                        else:
                            action = 2  # TST for confirmation when resources allow
                    
                    # RULE 4: Confirmed threats - efficient monitoring
                    elif threat == "Confirmed":
                        action = 1  # XGBoost for ongoing monitoring (already confirmed)
                    
                    # Add to lookup table
                    lookup[state_key] = action
        
        return lookup
    
    def discretize_state(self, state):
        """Convert continuous state to discrete state for lookup table"""
        # Handle battery level
        battery = state.get('battery')
        if isinstance(battery, (int, float)):
            if battery <= 20:
                battery_level = "0-20%"
            elif battery <= 40:
                battery_level = "21-40%"
            elif battery <= 60:
                battery_level = "41-60%"
            elif battery <= 80:
                battery_level = "61-80%"
            else:
                battery_level = "81-100%"
        else:
            # Already discrete
            battery_level = battery
        
        # Handle temperature
        temp = state.get('temperature')
        if isinstance(temp, (int, float)):
            if temp <= 55:
                temp_level = "Safe"
            elif temp <= 70:
                temp_level = "Warning"
            else:
                temp_level = "Critical"
        else:
            # Already discrete
            temp_level = temp
        
        # Handle threat state
        threat = state.get('threat')
        if isinstance(threat, int):
            threat_level = ["Normal", "Confirming", "Confirmed"][min(threat, 2)]
        else:
            # Already discrete
            threat_level = threat
        
        return (battery_level, temp_level, threat_level)
    
    def make_decision(self, state):
        """
        Make decision based on current state
        Uses lookup table with optional minimal learning
        """
        # Discretize state
        discrete_state = self.discretize_state(state)
        
        # Get action from lookup table
        action = self.lookup_table.get(discrete_state, 1)  # Default to XGBoost if unknown
        
        # Safety check - critical conditions always override
        if discrete_state[0] == "0-20%" or discrete_state[1] == "Critical":
            action = 0  # Force No DDoS for critical conditions
        
        # Log decision
        self._log_decision(discrete_state, action)
        
        # Update experience counter
        self.experience[discrete_state] += 1
        
        # Update hardware monitors
        self.thermal_simulator.update_temperature(action)
        self.power_tracker.start_action(action)
        
        return action
    
    def update_lookup_table(self, state, action, reward, next_state):
        """
        Limited learning mechanism to update the lookup table
        Only makes small adjustments based on observed rewards
        """
        if not self.learning_enabled:
            return
        
        # Only update if we have enough experience with this state
        discrete_state = self.discretize_state(state)
        if self.experience[discrete_state] < 5:
            return  # Not enough experience to justify changing expert policy
        
        # Get current action from lookup
        current_action = self.lookup_table.get(discrete_state, 0)
        
        # Only consider updating if reward is significantly better or worse
        if action == current_action and reward < -50:
            # Consistently bad results with current policy
            # Consider alternatives
            alternative_actions = [a for a in range(3) if a != current_action]
            
            # Choose safest alternative (lowest number = safest)
            new_action = min(alternative_actions)
            
            # Small chance to update based on learning rate
            if np.random.random() < self.learning_rate:
                self.lookup_table[discrete_state] = new_action
                self.logger.info(f"Updated lookup table: {discrete_state} -> {new_action} (was {current_action})")
    
    def _log_decision(self, discrete_state, action):
        """Log decision for analysis"""
        power_rate = self.power_tracker.get_current_rate()
        temp = self.thermal_simulator.get_temperature()
        
        decision = {
            'timestamp': datetime.now().isoformat(),
            'state': {
                'battery': discrete_state[0],
                'temperature': discrete_state[1], 
                'threat': discrete_state[2]
            },
            'action': action,
            'action_label': self.action_labels[action],
            'power_rate': power_rate,
            'temperature': temp,
            'experience_count': self.experience[discrete_state]
        }
        
        self.decisions.append(decision)
        
        # Log decision
        self.logger.info(f"Decision: {self.action_labels[action]} for state {discrete_state}")
    
    def save_lookup_table(self, filepath='lookup_table.json'):
        """Save lookup table to file"""
        # Convert tuple keys to strings for JSON serialization
        serializable_table = {}
        for state_key, action in self.lookup_table.items():
            key_str = f"{state_key[0]}|{state_key[1]}|{state_key[2]}"
            serializable_table[key_str] = action
        
        # Include metadata
        data = {
            'lookup_table': serializable_table,
            'state_space': {
                'battery_levels': self.battery_levels,
                'temperatures': self.temperatures,
                'threat_states': self.threat_states
            },
            'action_labels': self.action_labels,
            'timestamp': datetime.now().isoformat()
        }
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Lookup table saved to: {filepath}")
    
    def load_lookup_table(self, filepath):
        """Load lookup table from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Convert string keys back to tuples
            lookup = {}
            for key_str, action in data['lookup_table'].items():
                parts = key_str.split('|')
                if len(parts) == 3:
                    state_key = (parts[0], parts[1], parts[2])
                    lookup[state_key] = action
            
            self.lookup_table = lookup
            self.logger.info(f"Loaded lookup table with {len(lookup)} entries from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading lookup table: {e}")
            return False
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.decisions:
            return {}
        
        # Action distribution
        action_counts = defaultdict(int)
        for decision in self.decisions:
            action_counts[decision['action_label']] += 1
        
        # Calculate percentages
        total_decisions = len(self.decisions)
        action_percentages = {action: count/total_decisions*100 
                             for action, count in action_counts.items()}
        
        # Power statistics
        total_power = self.power_tracker.get_total_power()
        avg_power_rate = sum(d['power_rate'] for d in self.decisions) / total_decisions
        
        # Temperature statistics
        avg_temp = sum(d['temperature'] for d in self.decisions) / total_decisions
        max_temp = max(d['temperature'] for d in self.decisions)
        
        # Check for critical states
        critical_states = sum(1 for d in self.decisions 
                             if d['state']['battery'] == "0-20%" or 
                             d['state']['temperature'] == "Critical")
        
        return {
            'total_decisions': total_decisions,
            'action_distribution': dict(action_counts),
            'action_percentages': action_percentages,
            'total_power': total_power,
            'avg_power_rate': avg_power_rate,
            'avg_temperature': avg_temp,
            'max_temperature': max_temp,
            'critical_states': critical_states,
            'critical_percentage': critical_states/total_decisions*100 if total_decisions else 0
        }
    
    def visualize_lookup_table(self):
        """Generate a representation of the lookup table for visualization"""
        # Group by threat state for better organization
        table_by_threat = defaultdict(dict)
        
        for state, action in self.lookup_table.items():
            battery, temp, threat = state
            key = (battery, temp)
            table_by_threat[threat][key] = action
        
        return {
            'table_by_threat': dict(table_by_threat),
            'action_labels': self.action_labels
        }


# Example usage
if __name__ == "__main__":
    # Create agent
    agent = LookupTableRLAgent(learning_enabled=True, learning_rate=0.05)
    
    print(f"Created lookup table RL agent with {len(agent.lookup_table)} entries")
    
    # Example decision making
    state = {
        'battery': 75,           # 75% battery
        'temperature': 45,       # 45°C (Safe)
        'threat': "Confirming"   # Potential threat
    }
    
    action = agent.make_decision(state)
    print(f"For state {state}, selected action: {agent.action_labels[action]}")
    
    # Save lookup table
    agent.save_lookup_table('models/lookup_table_expert.json')
    
    # Get performance stats
    print(agent.get_performance_stats())
