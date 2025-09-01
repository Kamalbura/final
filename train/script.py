# COMPREHENSIVE Q-LEARNING IMPLEMENTATION WITH EXPERT LOOKUP TABLE
# Focus on proper training, reward function, and decision-making

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import random

print("="*80)
print("POWER-AWARE DDoS-RL: COMPLETE IMPLEMENTATION WITH EXPERT GUIDANCE")
print("="*80)

# 1. DEFINE STATE SPACE AND EXPERT LOOKUP TABLE
class UAVDDoSEnvironment:
    def __init__(self):
        # State space definitions
        self.battery_levels = ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"]
        self.temperatures = ["Safe", "Warning", "Critical"]
        self.threat_states = ["Normal", "Confirming", "Confirmed"]
        self.action_labels = ["No DDoS", "XGBoost", "TST"]  # Fixed action label
        
        # Decision matrix based on battery level and threat state
        self.decision_matrix = [
            [0, 0, 0],  # Battery 0-20%: Always No DDoS
            [1, 2, 1],  # Battery 21-40%: XGBoost, TST, XGBoost
            [1, 2, 1],  # Battery 41-60%: XGBoost, TST, XGBoost
            [1, 2, 1],  # Battery 61-80%: XGBoost, TST, XGBoost
            [1, 2, 1]   # Battery 81-100%: XGBoost, TST, XGBoost
        ]
        
        # Power consumption per action (Watts)
        self.power_consumption = {0: 3.0, 1: 5.5, 2: 9.0}
        
        # Expert lookup table - THIS IS THE GROUND TRUTH
        self.expert_lookup = self._create_expert_lookup_table()
        print(f"✅ Expert lookup table created: {len(self.expert_lookup)} entries")
        
        self.reset()
    
    def _create_expert_lookup_table(self):
        """Create the expert lookup table based on the provided decision matrix"""
        lookup = {}
        
        for i, battery in enumerate(self.battery_levels):
            for j, temp in enumerate(self.temperatures):
                for k, threat in enumerate(self.threat_states):
                    state_key = (battery, temp, threat)
                    
                    # Safety first: Critical temperature always uses No DDoS
                    if temp == "Critical" or battery == "0-20%":
                        expert_action = 0  # Always No DDoS for critical conditions
                    else:
                        # Use the decision matrix for non-critical conditions
                        expert_action = self.decision_matrix[i][k]
                    
                    lookup[state_key] = expert_action
        
        # Print summary of lookup table
        actions_count = {0: 0, 1: 0, 2: 0}
        for action in lookup.values():
            actions_count[action] = actions_count.get(action, 0) + 1
        
        print(f"Lookup table distribution:")
        print(f"- {self.action_labels[0]}: {actions_count[0]} states")
        print(f"- {self.action_labels[1]}: {actions_count[1]} states")
        print(f"- {self.action_labels[2]}: {actions_count[2]} states")
        
        return lookup
    
    def get_state_index(self, battery, temp, threat):
        """Convert state to index for Q-table"""
        battery_idx = self.battery_levels.index(battery)
        temp_idx = self.temperatures.index(temp)
        threat_idx = self.threat_states.index(threat)
        return battery_idx * 9 + temp_idx * 3 + threat_idx
    
    def get_expert_action(self, state):
        """Get expert recommended action for current state"""
        state_key = (state['battery'], state['temperature'], state['threat'])
        return self.expert_lookup.get(state_key, 0)
    
    def reset(self):
        """Reset environment to random initial state"""
        self.current_state = {
            'battery': random.choice(self.battery_levels[1:]),  # Not starting with critical battery
            'temperature': random.choice(self.temperatures[:2]),  # Not starting critical temp
            'threat': 'Normal'
        }
        self.episode_step = 0
        self.total_power_consumed = 0.0
        self.total_detections = 0
        self.safety_violations = 0
        return self.current_state.copy()
    
    def step(self, action):
        """Execute action and return next state, reward, done"""
        self.episode_step += 1
        
        # Calculate reward based on action taken
        reward = self._calculate_reward(action, self.current_state)
        
        # Update metrics
        power_used = self.power_consumption[action]
        self.total_power_consumed += power_used
        
        # Simulate detection success
        detection_success = self._simulate_detection(action, self.current_state['threat'])
        if detection_success:
            self.total_detections += 1
    
        # Check for safety violations
        if self._is_safety_violation(action, self.current_state):
            self.safety_violations += 1
        
        # Simulate state transition
        next_state = self._simulate_state_transition()
        self.current_state = next_state
        
        # Episode termination conditions
        done = (self.episode_step >= 20 or 
                (self.current_state['battery'] == "0-20%" and action != 0) or
                (self.current_state['temperature'] == "Critical" and action != 0))
        
        return next_state.copy(), reward, done
    
    def _calculate_reward(self, action, state):
        """CRITICAL: Reward function that heavily penalizes deviations from expert policy"""
        expert_action = self.get_expert_action(state)
        
        # BASE REWARD COMPONENTS
        reward = 0.0
        
        # 1. EXPERT ALIGNMENT REWARD (MOST IMPORTANT)
        if action == expert_action:
            reward += 50.0  # BIG REWARD for following expert guidance
        else:
            reward -= 30.0  # PENALTY for deviating from expert
        
        # Add component weights and documentation for clearer understanding
        # Each component's contribution to final reward should be logged
        
        # Add reward normalization to ensure consistent scale across different state spaces
        
        # 2. SAFETY VIOLATION PENALTIES (HUGE PENALTIES)
        if self._is_safety_violation(action, state):
            reward -= 100.0  # MASSIVE penalty for unsafe actions
        
        # 3. SPECIFIC DANGEROUS ACTION PENALTIES
        # TST when not needed (Normal or Confirmed threats)
        if action == 2 and state['threat'] in ['Normal', 'Confirmed']:
            reward -= 75.0  # Heavy penalty for unnecessary TST
        
        # Any action other than No_DDoS in critical conditions
        if (state['battery'] == "0-20%" or state['temperature'] == "Critical") and action != 0:
            reward -= 200.0  # EXTREME penalty for ignoring critical conditions
        
        # 4. POWER EFFICIENCY BONUS/PENALTY
        power_used = self.power_consumption[action]
        if power_used <= 5.5:  # Efficient actions
            reward += 5.0
        elif power_used > 7.0:  # High power actions
            reward -= 10.0
        
        # 5. DETECTION CAPABILITY BONUS
        if action > 0 and state['threat'] in ['Confirming', 'Confirmed']:
            detection_prob = self._get_detection_probability(action, state['threat'])
            reward += detection_prob * 10.0  # Bonus for detection capability
        
        # 6. SMART RESOURCE USAGE BONUS
        if state['battery'] in ["61-80%", "81-100%"] and state['temperature'] in ["Safe", "Warning"]:
            if action == 2 and state['threat'] == "Confirming":
                reward += 20.0  # Bonus for using TST in good conditions
        
        return reward
    
    def _is_safety_violation(self, action, state):
        """Check if action violates safety constraints"""
        # Critical battery with any action other than No_DDoS
        if state['battery'] == "0-20%" and action != 0:
            return True
        
        # Critical temperature with any action other than No_DDoS
        if state['temperature'] == "Critical" and action != 0:
            return True
        
        return False
    
    def _get_detection_probability(self, action, threat):
        """Get detection probability for action-threat combination"""
        if action == 0:  # No_DDoS
            return 0.0
        elif action == 1:  # XGBoost
            probs = {'Normal': 0.8, 'Confirming': 0.6, 'Confirmed': 0.9}
        else:  # TST
            probs = {'Normal': 0.9, 'Confirming': 0.95, 'Confirmed': 0.95}
        
        return probs.get(threat, 0.0)
    
    def _simulate_detection(self, action, threat):
        """Simulate detection success"""
        prob = self._get_detection_probability(action, threat)
        return random.random() < prob
    
    def _simulate_state_transition(self):
        """Simulate realistic state transitions"""
        new_state = self.current_state.copy()
        
        # Threat evolution (30% chance of change)
        if random.random() < 0.3:
            if self.current_state['threat'] == 'Normal':
                new_state['threat'] = random.choice(['Normal', 'Confirming'])
            elif self.current_state['threat'] == 'Confirming':
                new_state['threat'] = random.choice(['Normal', 'Confirmed', 'Confirming'])
            else:  # Confirmed
                new_state['threat'] = random.choice(['Normal', 'Confirmed'])
        
        # Battery degradation (15% chance)
        if random.random() < 0.15:
            current_idx = self.battery_levels.index(self.current_state['battery'])
            if current_idx > 0:
                new_state['battery'] = self.battery_levels[max(0, current_idx - 1)]
        
        # Temperature changes (20% chance)
        if random.random() < 0.2:
            new_state['temperature'] = random.choice(self.temperatures)
        
        return new_state

# Initialize environment
env = UAVDDoSEnvironment()
print(f"✅ Environment initialized with {len(env.expert_lookup)} expert decisions")