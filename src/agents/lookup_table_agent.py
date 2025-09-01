#!/usr/bin/env python3
"""
Lookup Table-Based RL Agent for UAV DDoS defense
Combines expert knowledge with Q-learning
"""

import numpy as np
import json
import os
import time
from collections import defaultdict
from typing import Dict, Tuple, List, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LookupTableAgent:
    """
    Lookup Table-Based RL Agent for UAV DDoS defense
    Uses expert knowledge for initialization and Q-learning for refinement
    """
    
    def __init__(self, learning_enabled=True):
        """Initialize the lookup table agent"""
        # State space definition
        self.battery_levels = ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"]
        self.temperatures = ["Safe", "Warning", "Critical"]
        self.threat_states = ["Normal", "Confirming", "Confirmed"]
        self.action_labels = ["No_DDoS", "XGBoost", "TST"]
        
        # Q-learning parameters
        self.learning_enabled = learning_enabled
        self.alpha = 0.1          # Learning rate
        self.gamma = 0.95         # Discount factor
        self.epsilon = 0.2        # Initial exploration rate
        self.epsilon_min = 0.01   # Minimum exploration rate
        self.epsilon_decay = 0.995  # Exploration decay rate
        
        # Initialize lookup table (state -> action)
        self.lookup_table = self._initialize_expert_lookup_table()
        
        # Initialize Q-table for learning
        self.q_table = self._initialize_q_table_from_expert()
        self.visit_count = defaultdict(lambda: defaultdict(int))
        
        # Runtime metrics
        self.performance_metrics = {
            'decisions_made': 0,
            'action_counts': {0: 0, 1: 0, 2: 0},
            'temperatures': [],
            'power_rates': []
        }
        
        # Add simulated components for demonstration
        self._initialize_simulators()
        
        logger.info("Lookup Table Agent initialized with expert knowledge")
    
    def _initialize_expert_lookup_table(self):
        """Initialize expert lookup with simplified rules:
        - Critical temp OR 0-20% battery => No_DDoS (0)
        - Else if threat == Confirming => TST (2)
        - Else => XGBoost (1)
        """
        lookup_table = {}
        for battery in self.battery_levels:
            for temp in self.temperatures:
                for threat in self.threat_states:
                    if temp == "Critical" or battery == "0-20%":
                        action = 0
                    elif threat == "Confirming":
                        action = 2
                    else:
                        action = 1
                    lookup_table[(battery, temp, threat)] = action
        logger.info(f"Expert lookup table initialized with {len(lookup_table)} entries (simplified policy)")
        return lookup_table
    
    def _initialize_q_table_from_expert(self):
        """Initialize Q-table with values from expert policy"""
        q_table = defaultdict(lambda: defaultdict(float))
        
        # Initialize Q-values based on expert policy
        for state, expert_action in self.lookup_table.items():
            # Set high value for expert-recommended action
            q_table[state][expert_action] = 10.0
            
            # Set lower values for other actions
            for action in range(3):
                if action != expert_action:
                    q_table[state][action] = 2.0
                    
                    # Apply safety constraints
                    battery, temp, threat = state
                    if (battery == "0-20%" or temp == "Critical") and action != 0:
                        q_table[state][action] = -100.0  # Strongly discourage unsafe actions
                    # Penalize TST anywhere it is not expert-recommended (keep it minimal)
                    if action == 2 and expert_action != 2:
                        q_table[state][action] = 0.5  # mild deterrent
                    # Mildly penalize No_DDoS when expert wants active detection (XGBoost/TST)
                    if action == 0 and expert_action in [1, 2] and not (battery == "0-20%" or temp == "Critical"):
                        q_table[state][action] = 1.0
        
        return q_table
    
    def _initialize_simulators(self):
        """Initialize thermal and power simulators for demonstration"""
        # Simple thermal simulator
        class ThermalSimulator:
            def __init__(self):
                self.temperature = 45.0
                self.temp_history = []
                self.algorithm_impact = {"No_DDoS": 0.1, "XGBoost": 0.5, "TST": 1.2}
            
            def update(self, algorithm):
                impact = self.algorithm_impact.get(algorithm, 0)
                # Temperature rises based on algorithm but naturally cools
                self.temperature += impact - max(0, (self.temperature - 45) * 0.05)
                self.temperature = max(40, min(85, self.temperature))
                self.temp_history.append(self.temperature)
                return self.temperature
            
            def get_temperature(self):
                return self.temperature
            
            def get_temperature_category(self):
                if self.temperature <= 55:
                    return "Safe"
                elif self.temperature <= 70:
                    return "Warning"
                else:
                    return "Critical"
        
        # Simple power tracker
        class PowerTracker:
            def __init__(self):
                self.power_rates = []
                self.algorithm_power = {"No_DDoS": 3.0, "XGBoost": 5.5, "TST": 9.0}
            
            def update(self, algorithm):
                power = self.algorithm_power.get(algorithm, 3.0)
                self.power_rates.append(power)
                return power
            
            def get_current_rate(self):
                return self.power_rates[-1] if self.power_rates else 0
            
            def get_total_consumption(self):
                return sum(self.power_rates)
            
            def get_average_consumption(self):
                return sum(self.power_rates) / len(self.power_rates) if self.power_rates else 0
        
        self.thermal_simulator = ThermalSimulator()
        self.power_tracker = PowerTracker()
    
    def _discretize_state(self, state):
        """Convert continuous state values to discrete state for lookup table"""
        # Extract state values
        battery = state.get('battery', 50.0)
        temperature = state.get('temperature', self.thermal_simulator.get_temperature())
        threat = state.get('threat', "Normal")
        
        # Discretize battery
        if isinstance(battery, str):
            # If battery is already a category string, use it directly
            battery_level = battery
        else:
            # Convert numerical battery to category
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
        
        # Discretize temperature
        if isinstance(temperature, str):
            temp_category = temperature  # Already categorical
        else:
            if temperature <= 55:
                temp_category = "Safe"
            elif temperature <= 70:
                temp_category = "Warning"
            else:
                temp_category = "Critical"
        
        # Ensure threat is a string (could be int or string)
        if isinstance(threat, int):
            threat_state = self.threat_states[min(threat, 2)]
        else:
            threat_state = threat
        
        return (battery_level, temp_category, threat_state)
    
    def make_decision(self, state):
        """Make a decision based on current state"""
        # Convert state to discrete format for lookup
        discrete_state = self._discretize_state(state)
        
        # Increment decision counter
        self.performance_metrics['decisions_made'] += 1
        
        # Check if we should explore (only during training)
        if self.learning_enabled and np.random.random() < self.epsilon:
            # Safe exploration - only consider safe actions (use raw state for recovery time)
            safe_actions = self._get_safe_actions(discrete_state, state)
            action = np.random.choice(safe_actions)
        else:
            # Use Q-table for exploitation (with safety mask including recovery)
            action = self._get_best_action(discrete_state, state)
        
        # Record action for metrics
        self.performance_metrics['action_counts'][action] += 1
        
        # Simulate effects for demonstration
        algorithm = self.action_labels[action]
        temperature = self.thermal_simulator.update(algorithm)
        power_rate = self.power_tracker.update(algorithm)
        
        # Record metrics
        self.performance_metrics['temperatures'].append(temperature)
        self.performance_metrics['power_rates'].append(power_rate)
        
        return action
    
    def _get_safe_actions(self, state, raw_state=None):
        """Return allowed action(s) implementing simplified deterministic policy:
        - If Critical temperature OR battery 0-20% -> [No_DDoS]
        - Else if threat == Confirming -> [TST]
        - Else -> [XGBoost]
        """
        if isinstance(state, tuple):
            battery_level, temp_category, threat_state = state
        else:
            try:
                battery_level, temp_category, threat_state = state
            except Exception:
                battery_level, temp_category, threat_state = "41-60%", "Safe", "Normal"
        if battery_level == "0-20%" or temp_category == "Critical":
            return [0]
        if threat_state == "Confirming":
            return [2]
        return [1]
    
    def _get_best_action(self, state, raw_state=None):
        """Get best action from Q-table (respects safety mask incl. recovery)."""
        # Get safe actions (include raw_state for recovery constraint)
        safe_actions = self._get_safe_actions(state, raw_state)

        # Get Q-values for safe actions
        q_values = {action: self.q_table[state][action] for action in safe_actions}

        # Return action with highest Q-value
        return max(q_values, key=q_values.get)
    
    def update(self, state, action, reward, next_state, done):
        """Update Q-values based on experience"""
        if not self.learning_enabled:
            return
        
        # Convert states to discrete format
        state = self._discretize_state(state)
        next_state = self._discretize_state(next_state)
        
        # Get current Q-value
        current_q = self.q_table[state][action]
        
        # Compute target Q-value
        if done:
            target_q = reward
        else:
            # Get best next action
            next_safe_actions = self._get_safe_actions(next_state)
            next_q_values = [self.q_table[next_state][a] for a in next_safe_actions]
            max_next_q = max(next_q_values) if next_q_values else 0
            
            target_q = reward + self.gamma * max_next_q
        
        # Expert guidance factor
        expert_action = self.lookup_table.get(state, 0)
        expert_factor = 2.0 if action == expert_action else 0.5
        target_q *= expert_factor
        
        # Safety constraints
        battery, temp, _ = state
        if (battery == "0-20%" or temp == "Critical") and action != 0:
            target_q = -100.0  # Strong penalty for unsafe actions
        
        # Power consumption penalty for TST in high battery states
        # Only apply this during training to refine the policy
        if action == 2 and expert_action == 1:  # If expert recommends XGBoost but action is TST
            if battery in ["61-80%", "81-100%"]:
                target_q -= 30.0  # Strong penalty to discourage unnecessary TST use
        
        # Penalty for recovery time violations
        if action == 2 and "TST needs recovery time" in self.performance_metrics.get('last_safety_violation', ''):
            target_q -= 50.0  # Very strong penalty for recovery time violations
        
        # Adaptive learning rate
        visit_count = self.visit_count[state][action]
        adaptive_alpha = self.alpha / (1.0 + 0.01 * visit_count)
        
        # Update Q-value
        self.q_table[state][action] = current_q + adaptive_alpha * (target_q - current_q)
        self.visit_count[state][action] += 1
        
        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, env, num_episodes=200, max_steps=300):
        """Train the agent on the environment"""
        if not self.learning_enabled:
            logger.warning("Training called but learning is disabled")
            return {}

        logger.info(f"Starting training for {num_episodes} episodes...")

        metrics = {
            'episodes': [],
            'rewards': [],
            'expert_alignment': [],
            'safety_violations': [],
            'power_consumption': [],
            'epsilon_values': []
        }

        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0.0
            expert_agreements = 0
            total_actions = 0

            for _ in range(max_steps):
                action = self.make_decision(state)
                expert_action = env.get_expert_action(state)
                if action == expert_action:
                    expert_agreements += 1
                total_actions += 1

                next_state, reward, done = env.step(action)
                self.update(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
                if done:
                    break

            metrics['episodes'].append(episode)
            metrics['rewards'].append(total_reward)
            metrics['expert_alignment'].append(expert_agreements / total_actions if total_actions > 0 else 0)
            metrics['safety_violations'].append(getattr(env, 'safety_violations', 0))
            metrics['power_consumption'].append(getattr(env, 'total_power_consumed', 0))
            metrics['epsilon_values'].append(self.epsilon)

            if episode % 10 == 0:
                logger.info(
                    f"Episode {episode}/{num_episodes}: Reward={total_reward:.1f}, "
                    f"Expert Alignment={expert_agreements / total_actions if total_actions > 0 else 0:.1%}"
                )

        logger.info(
            f"Training completed: Final expert alignment: {metrics['expert_alignment'][-1]:.1%}, "
            f"Final reward: {metrics['rewards'][-1]:.1f}"
        )
        self.last_training_metrics = metrics
        return metrics
    
    def save_model(self, filepath):
        """Save the lookup table and Q-table to file"""
        # Convert defaultdict to regular dict for JSON serialization
        q_table_dict = {str(k): {str(a): v for a, v in d.items()} 
                       for k, d in self.q_table.items()}
        
        visit_count_dict = {str(k): {str(a): v for a, v in d.items()} 
                           for k, d in self.visit_count.items()}
        
        # Create model data
        model_data = {
            'lookup_table': {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in self.lookup_table.items()},
            'q_table': q_table_dict,
            'visit_count': visit_count_dict,
            'params': {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon
            },
            'state_space': {
                'battery_levels': self.battery_levels,
                'temperatures': self.temperatures,
                'threat_states': self.threat_states
            },
            'action_labels': self.action_labels,
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S"),
            'training_metrics': getattr(self, 'last_training_metrics', None)
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load lookup table and Q-table from file"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        # Load lookup table
        self.lookup_table = {}
        for k, v in model_data['lookup_table'].items():
            parts = k.split('|')
            self.lookup_table[(parts[0], parts[1], parts[2])] = v
        
        # Load Q-table
        self.q_table = defaultdict(lambda: defaultdict(float))
        for state_str, actions in model_data['q_table'].items():
            state_parts = state_str.strip('()').replace("'", "").split(', ')
            state = (state_parts[0], state_parts[1], state_parts[2])
            for action_str, value in actions.items():
                self.q_table[state][int(action_str)] = value
        
        # Load visit counts
        self.visit_count = defaultdict(lambda: defaultdict(int))
        for state_str, actions in model_data.get('visit_count', {}).items():
            state_parts = state_str.strip('()').replace("'", "").split(', ')
            state = (state_parts[0], state_parts[1], state_parts[2])
            for action_str, value in actions.items():
                self.visit_count[state][int(action_str)] = value
        
        # Load parameters
        params = model_data.get('params', {})
        self.alpha = params.get('alpha', self.alpha)
        self.gamma = params.get('gamma', self.gamma)
        self.epsilon = params.get('epsilon', self.epsilon)
        
        # Optional: Load state space definitions if present
        if 'state_space' in model_data:
            self.battery_levels = model_data['state_space'].get('battery_levels', self.battery_levels)
            self.temperatures = model_data['state_space'].get('temperatures', self.temperatures)
            self.threat_states = model_data['state_space'].get('threat_states', self.threat_states)
        
        if 'action_labels' in model_data:
            self.action_labels = model_data['action_labels']
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_performance_stats(self):
        """Get performance statistics"""
        total_decisions = self.performance_metrics['decisions_made']
        
        # Calculate action percentages
        action_percentages = {}
        for action, count in self.performance_metrics['action_counts'].items():
            action_percentages[self.action_labels[action]] = count / total_decisions * 100 if total_decisions > 0 else 0
        
        return {
            'total_decisions': total_decisions,
            'action_distribution': {self.action_labels[a]: c for a, c in self.performance_metrics['action_counts'].items()},
            'action_percentages': action_percentages,
            'avg_temperature': np.mean(self.performance_metrics['temperatures']) if self.performance_metrics['temperatures'] else 0,
            'max_temperature': np.max(self.performance_metrics['temperatures']) if self.performance_metrics['temperatures'] else 0,
            'avg_power_rate': np.mean(self.performance_metrics['power_rates']) if self.performance_metrics['power_rates'] else 0,
            'total_power_consumption': np.sum(self.performance_metrics['power_rates']) if self.performance_metrics['power_rates'] else 0
        }
