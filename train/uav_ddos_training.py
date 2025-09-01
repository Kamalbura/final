#!/usr/bin/env python3
"""
Power-Aware DDoS-RL Training Script for UAV
Expert-Guided Q-Learning with Safety-First Reward Function
"""

import numpy as np
import pandas as pd
import json
import random
from datetime import datetime
import time  # For timestamp-based power measurement

class PowerTracker:
    """Precise power consumption tracker using V×I×Δt methodology"""
    
    def __init__(self):
        # Default UAV voltage (volts)
        self.voltage = 11.1  # 3S LiPo battery
        
        # Current draw for each algorithm (amps)
        self.current_draw = {
            0: 0.27,  # No DDoS: 3.0W / 11.1V ≈ 0.27A
            1: 0.50,  # XGBoost: 5.5W / 11.1V ≈ 0.50A
            2: 0.81   # TST: 9.0W / 11.1V ≈ 0.81A
        }
        
        # Tracking variables
        self.action_start_time = None
        self.current_action = None
        self.total_power_consumed = 0.0
        self.last_measurement = 0.0
    
    def start_action(self, action):
        """Start tracking a new action"""
        # First, finalize any ongoing measurement
        if self.action_start_time is not None:
            self.measure_current_consumption()
        
        self.action_start_time = time.time()
        self.current_action = action
    
    def measure_current_consumption(self):
        """Measure power consumption for current action"""
        if self.action_start_time is None or self.current_action is None:
            return 0.0
        
        # Calculate time delta
        now = time.time()
        delta_t = now - self.action_start_time
        
        # P = V × I × Δt (in watt-seconds or joules)
        current = self.current_draw.get(self.current_action, 0.0)
        power_consumed = self.voltage * current * delta_t
        
        # Add to total
        self.total_power_consumed += power_consumed
        self.last_measurement = power_consumed
        
        # Reset for next action
        self.action_start_time = now
        
        return power_consumed
    
    def get_total_consumption(self):
        """Get total power consumption with latest measurement"""
        self.measure_current_consumption()
        return self.total_power_consumed
    
    def get_last_consumption(self):
        """Get most recent power consumption"""
        return self.last_measurement

class ThermalSimulator:
    """Thermal model for algorithm heating effects"""
    
    def __init__(self, ambient_temp=25.0, time_constant=60.0):
        # Baseline thermal parameters
        self.ambient_temp = ambient_temp  # Ambient temperature in Celsius
        self.time_constant = time_constant  # Thermal time constant (seconds)
        
        # Thermal impact of each action (°C/s at steady state)
        self.action_thermal_impact = {
            0: 0.05,   # No DDoS: minimal heating
            1: 0.12,   # XGBoost: moderate heating
            2: 0.25    # TST: significant heating
        }
        
        # Current state
        self.current_temp = ambient_temp
        self.last_update_time = time.time()
        self.current_action = None
    
    def update_temperature(self, action=None):
        """Update temperature based on time elapsed and action"""
        current_time = time.time()
        delta_t = current_time - self.last_update_time
        
        if action is not None:
            self.current_action = action
        
        if self.current_action is not None:
            # Get steady state temperature for current action
            action_impact = self.action_thermal_impact.get(self.current_action, 0.0)
            steady_state_temp = self.ambient_temp + action_impact * 60
            
            # Newton's law of cooling/heating
            # T(t) = T_ambient + (T_initial - T_ambient) * e^(-t/tau)
            # Here we use it for heating, with steady state replacing ambient
            self.current_temp = steady_state_temp - (steady_state_temp - self.current_temp) * np.exp(-delta_t / self.time_constant)
        
        self.last_update_time = current_time
        return self.current_temp
    
    def get_temperature(self):
        """Get current temperature with update"""
        return self.update_temperature()
    
    def get_temperature_category(self):
        """Get temperature category based on current temperature"""
        temp = self.get_temperature()
        if temp <= 55:
            return "Safe"
        elif temp <= 70:
            return "Warning"
        else:
            return "Critical"

class UAVDDoSEnvironment:
    def __init__(self):
        # State space definitions
        self.battery_levels = ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"]
        self.temperatures = ["Safe", "Warning", "Critical"]
        self.threat_states = ["Normal", "Confirming", "Confirmed"]
        self.action_labels = ["No_DDoS", "XGBoost", "TST"]

        # Enhanced monitoring systems
        self.power_tracker = PowerTracker()
        self.thermal_simulator = ThermalSimulator()

        # Power consumption per action (Watts) - reference values
        self.power_consumption = {0: 3.0, 1: 5.5, 2: 9.0}

        # Expert lookup table - SAFETY-FIRST POLICY
        self.expert_lookup = self._create_expert_lookup_table()
        self.reset()

    def _create_expert_lookup_table(self):
        """Create expert lookup table with optimal safety-first decisions"""
        lookup = {}

        for battery in self.battery_levels:
            for temp in self.temperatures:
                for threat in self.threat_states:
                    state_key = (battery, temp, threat)

                    # EXPERT DECISION LOGIC - SAFETY FIRST
                    if battery == "0-20%" or temp == "Critical":
                        expert_action = 0  # Always No_DDoS for critical conditions
                    elif threat == "Normal":
                        expert_action = 1 if battery not in ["21-40%"] else 0  # Conservative
                    elif threat == "Confirming":
                        expert_action = 2 if battery not in ["21-40%"] else 1  # TST if resources allow
                    elif threat == "Confirmed":
                        expert_action = 1  # XGBoost monitoring (already confirmed)

                    lookup[state_key] = expert_action

        return lookup

    def get_state_index(self, battery, temp, threat):
        """Convert state to Q-table index"""
        battery_idx = self.battery_levels.index(battery)
        temp_idx = self.temperatures.index(temp)
        threat_idx = self.threat_states.index(threat)
        return battery_idx * 9 + temp_idx * 3 + threat_idx

    def get_expert_action(self, state):
        """Get expert recommended action"""
        state_key = (state['battery'], state['temperature'], state['threat'])
        return self.expert_lookup.get(state_key, 0)

    def reset(self):
        """Reset environment"""
        self.current_state = {
            'battery': random.choice(self.battery_levels[1:]),
            'temperature': random.choice(self.temperatures[:2]),
            'threat': 'Normal'
        }
        self.episode_step = 0
        self.total_power_consumed = 0.0
        self.safety_violations = 0
        return self.current_state.copy()

    def step(self, action):
        """Execute action and return next state, reward, done"""
        self.episode_step += 1

        # Start power tracking for the action
        self.power_tracker.start_action(action)
        
        # Update thermal model
        self.thermal_simulator.update_temperature(action)

        # Calculate reward
        reward = self._calculate_reward(action, self.current_state)

        # Measure actual power consumption (V×I×Δt method)
        power_used = self.power_tracker.measure_current_consumption()
        self.total_power_consumed += power_used

        if self._is_safety_violation(action, self.current_state):
            self.safety_violations += 1

        # State transition
        next_state = self._simulate_state_transition()
        self.current_state = next_state

        # Episode end conditions - enhanced with thermal condition
        done = (self.episode_step >= 20 or 
                (self.current_state['battery'] == "0-20%" and action != 0) or
                (self.current_state['temperature'] == "Critical" and action != 0) or
                self.thermal_simulator.get_temperature() > 80)  # Hard thermal limit

        return next_state.copy(), reward, done

    def _calculate_reward(self, action, state):
        """CRITICAL: Enhanced multi-objective reward function"""
        expert_action = self.get_expert_action(state)
        reward = 0.0

        # Component 1: Expert alignment (20% weight)
        if action == expert_action:
            expert_alignment_reward = 50.0  # BIG reward for following expert
        else:
            expert_alignment_reward = -30.0  # Penalty for deviation
        
        reward += 0.2 * expert_alignment_reward
        
        # Component 2: Power efficiency (30% weight)
        # Use actual measured power from V×I×Δt
        power_used = self.power_tracker.get_last_consumption()
        power_efficiency_reward = -power_used * 0.5  # Penalty proportional to power
        
        reward += 0.3 * power_efficiency_reward
        
        # Component 3: Thermal management (25% weight)
        current_temp = self.thermal_simulator.get_temperature()
        if current_temp <= 55:
            thermal_reward = 10.0  # Excellent thermal management
        elif current_temp <= 70:
            thermal_reward = 0.0  # Acceptable but no bonus
        else:
            thermal_reward = -50.0 * (current_temp - 70) / 10  # Heavy penalty above 70°C
        
        reward += 0.25 * thermal_reward
        
        # Component 4: Security effectiveness (25% weight)
        if action == 0:  # No DDoS
            if state['threat'] in ['Confirming', 'Confirmed']:
                security_reward = -20.0  # Penalty for missing detection
            else:
                security_reward = 10.0  # Appropriate for normal state
        elif action == 1:  # XGBoost
            if state['threat'] in ['Normal']:
                security_reward = 5.0  # Decent for monitoring
            else:
                security_reward = 15.0  # Good for detection
        else:  # TST
            if state['threat'] in ['Normal']:
                security_reward = -30.0  # Wasteful for normal traffic
            elif state['threat'] == 'Confirming':
                security_reward = 30.0  # Excellent for confirming
            else:  # Confirmed
                security_reward = 20.0  # Good but possibly overkill
        
        reward += 0.25 * security_reward
        
        # Safety violations override (massive penalties)
        if self._is_safety_violation(action, state):
            reward -= 100.0  # Massive penalty for safety violations
        
        return reward

    def _is_safety_violation(self, action, state):
        """Check for safety violations"""
        return ((state['battery'] == "0-20%" and action != 0) or
                (state['temperature'] == "Critical" and action != 0))

    def _get_detection_probability(self, action, threat):
        """Get detection probability"""
        if action == 0:
            return 0.0
        elif action == 1:  # XGBoost
            return {'Normal': 0.8, 'Confirming': 0.6, 'Confirmed': 0.9}[threat]
        else:  # TST
            return {'Normal': 0.9, 'Confirming': 0.95, 'Confirmed': 0.95}[threat]

    def _simulate_state_transition(self):
        """Simulate realistic state transitions with thermal feedback"""
        new_state = self.current_state.copy()
        
        # Threat evolution
        if random.random() < 0.3:
            transitions = {
                'Normal': ['Normal', 'Confirming'],
                'Confirming': ['Normal', 'Confirmed', 'Confirming'],
                'Confirmed': ['Normal', 'Confirmed']
            }
            new_state['threat'] = random.choice(transitions[self.current_state['threat']])
        
        # Battery degradation with power-aware depletion
        if random.random() < 0.15:
            current_idx = self.battery_levels.index(self.current_state['battery'])
            # More power consumption increases chance of battery degradation
            power_factor = min(1.0, self.power_tracker.get_last_consumption() / 10.0)
            if random.random() < power_factor and current_idx > 0:
                new_state['battery'] = self.battery_levels[max(0, current_idx - 1)]
        
        # Temperature changes based on thermal model
        # Update simulator and get new temperature category
        new_state['temperature'] = self.thermal_simulator.get_temperature_category()
        
        return new_state

class ExpertGuidedQAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, 
                 epsilon=0.3, epsilon_decay=0.995, epsilon_min=0.05):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.num_states = 45
        self.num_actions = 3
        self.q_table = self._initialize_q_table()

        self.training_metrics = {
            'episodes': [], 'rewards': [], 'expert_alignment': [],
            'safety_violations': [], 'power_consumption': [], 'epsilon_values': []
        }

    def _initialize_q_table(self):
        """Initialize Q-table with expert knowledge"""
        q_table = np.random.normal(0, 0.1, (self.num_states, self.num_actions))

        # Set HIGH values for expert actions
        for state_key, expert_action in self.env.expert_lookup.items():
            battery, temp, threat = state_key
            state_idx = self.env.get_state_index(battery, temp, threat)
            q_table[state_idx, expert_action] = 100.0  # High value for expert actions

            # Lower values for non-expert actions
            for action in range(self.num_actions):
                if action != expert_action:
                    q_table[state_idx, action] = -10.0

        return q_table

    def get_action(self, state, training=True):
        """Get action using epsilon-greedy with safety bias"""
        state_idx = self.env.get_state_index(state['battery'], state['temperature'], state['threat'])

        if training and random.random() < self.epsilon:
            return self._safe_random_action(state)
        else:
            return np.argmax(self.q_table[state_idx])

    def _safe_random_action(self, state):
        """Safe exploration strategy"""
        # Always No_DDoS for critical conditions
        if state['battery'] == "0-20%" or state['temperature'] == "Critical":
            return 0

        expert_action = self.env.get_expert_action(state)
        # 70% chance to follow expert even during exploration
        return expert_action if random.random() < 0.7 else random.choice([0, 1, 2])

    def update_q_value(self, state, action, reward, next_state):
        """Update Q-table using Q-learning"""
        state_idx = self.env.get_state_index(state['battery'], state['temperature'], state['threat'])
        next_state_idx = self.env.get_state_index(next_state['battery'], next_state['temperature'], next_state['threat'])

        old_q = self.q_table[state_idx, action]
        next_max_q = np.max(self.q_table[next_state_idx])
        new_q = old_q + self.lr * (reward + self.gamma * next_max_q - old_q)
        self.q_table[state_idx, action] = new_q

    def train(self, num_episodes=200):
        """Train agent with expert guidance"""
        print(f"Starting training: {num_episodes} episodes")

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            expert_agreements = 0
            total_actions = 0

            while True:
                action = self.get_action(state, training=True)
                expert_action = self.env.get_expert_action(state)

                if action == expert_action:
                    expert_agreements += 1
                total_actions += 1

                next_state, reward, done = self.env.step(action)
                total_reward += reward

                self.update_q_value(state, action, reward, next_state)
                state = next_state

                if done:
                    break

            # Decay exploration
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Track metrics
            self.training_metrics['episodes'].append(episode)
            self.training_metrics['rewards'].append(total_reward)
            self.training_metrics['expert_alignment'].append(expert_agreements / total_actions)
            self.training_metrics['safety_violations'].append(self.env.safety_violations)
            self.training_metrics['power_consumption'].append(self.env.total_power_consumed)
            self.training_metrics['epsilon_values'].append(self.epsilon)

            if episode % 20 == 0:
                print(f"Episode {episode}: Reward={total_reward:.1f}, "
                      f"Expert Align={expert_agreements/total_actions:.3f}, "
                      f"Safety Violations={self.env.safety_violations}")

        return self.training_metrics

    def save_model(self, filepath):
        """Save trained Q-table and metadata"""
        model_data = {
            'q_table': self.q_table.tolist(),
            'training_params': {
                'learning_rate': self.lr,
                'discount_factor': self.gamma,
                'epsilon': self.epsilon,
                'num_states': self.num_states,
                'num_actions': self.num_actions
            },
            'expert_lookup': self.env.expert_lookup,
            'training_metrics': self.training_metrics,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        print(f"Model saved to: {filepath}")

# MAIN TRAINING SCRIPT
if __name__ == "__main__":
    print("="*80)
    print("UAV POWER-AWARE DDoS-RL TRAINING")
    print("="*80)

    # Initialize environment and agent
    env = UAVDDoSEnvironment()
    agent = ExpertGuidedQAgent(env)

    # Train the agent
    training_results = agent.train(num_episodes=200)

    # Save the trained model
    agent.save_model('trained_uav_ddos_model.json')

    print("\n✅ Training completed successfully!")
    print(f"Final reward: {training_results['rewards'][-1]:.1f}")
    print(f"Final expert alignment: {training_results['expert_alignment'][-1]:.3f}")
    print(f"Total safety violations: {sum(training_results['safety_violations'])}")
