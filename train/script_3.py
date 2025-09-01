# 4. CREATE DEPLOYABLE SCRIPTS FOR PRODUCTION USE

# Create main training script
training_script = '''#!/usr/bin/env python3
"""
Power-Aware DDoS-RL Training Script for UAV
Expert-Guided Q-Learning with Safety-First Reward Function
"""

import numpy as np
import pandas as pd
import json
import random
from datetime import datetime

class UAVDDoSEnvironment:
    def __init__(self):
        # State space definitions
        self.battery_levels = ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"]
        self.temperatures = ["Safe", "Warning", "Critical"]
        self.threat_states = ["Normal", "Confirming", "Confirmed"]
        self.action_labels = ["No_DDoS", "XGBoost", "TST"]
        
        # Power consumption per action (Watts)
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
        
        # Calculate reward
        reward = self._calculate_reward(action, self.current_state)
        
        # Update metrics
        self.total_power_consumed += self.power_consumption[action]
        
        if self._is_safety_violation(action, self.current_state):
            self.safety_violations += 1
        
        # State transition
        next_state = self._simulate_state_transition()
        self.current_state = next_state
        
        # Episode end conditions
        done = (self.episode_step >= 20 or 
                (self.current_state['battery'] == "0-20%" and action != 0) or
                (self.current_state['temperature'] == "Critical" and action != 0))
        
        return next_state.copy(), reward, done
    
    def _calculate_reward(self, action, state):
        """CRITICAL: Reward function with heavy penalties for unsafe actions"""
        expert_action = self.get_expert_action(state)
        reward = 0.0
        
        # 1. EXPERT ALIGNMENT (Primary reward component)
        if action == expert_action:
            reward += 50.0  # BIG reward for following expert
        else:
            reward -= 30.0  # Penalty for deviation
        
        # 2. SAFETY VIOLATIONS (Huge penalties)
        if self._is_safety_violation(action, state):
            reward -= 100.0  # Massive penalty for safety violations
        
        # 3. DANGEROUS ACTION PENALTIES
        if action == 2 and state['threat'] in ['Normal', 'Confirmed']:
            reward -= 75.0  # Heavy penalty for unnecessary TST
        
        if (state['battery'] == "0-20%" or state['temperature'] == "Critical") and action != 0:
            reward -= 200.0  # Extreme penalty for ignoring critical conditions
        
        # 4. EFFICIENCY BONUSES
        power_used = self.power_consumption[action]
        if power_used <= 5.5:
            reward += 5.0
        elif power_used > 7.0:
            reward -= 10.0
        
        # 5. DETECTION CAPABILITY BONUS
        if action > 0 and state['threat'] in ['Confirming', 'Confirmed']:
            detection_prob = self._get_detection_probability(action, state['threat'])
            reward += detection_prob * 10.0
        
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
        """Simulate realistic state transitions"""
        new_state = self.current_state.copy()
        
        # Threat evolution
        if random.random() < 0.3:
            transitions = {
                'Normal': ['Normal', 'Confirming'],
                'Confirming': ['Normal', 'Confirmed', 'Confirming'],
                'Confirmed': ['Normal', 'Confirmed']
            }
            new_state['threat'] = random.choice(transitions[self.current_state['threat']])
        
        # Battery degradation
        if random.random() < 0.15:
            current_idx = self.battery_levels.index(self.current_state['battery'])
            if current_idx > 0:
                new_state['battery'] = self.battery_levels[max(0, current_idx - 1)]
        
        # Temperature changes
        if random.random() < 0.2:
            new_state['temperature'] = random.choice(self.temperatures)
        
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
    
    print("\\n✅ Training completed successfully!")
    print(f"Final reward: {training_results['rewards'][-1]:.1f}")
    print(f"Final expert alignment: {training_results['expert_alignment'][-1]:.3f}")
    print(f"Total safety violations: {sum(training_results['safety_violations'])}")
'''

# Save training script
with open('uav_ddos_training.py', 'w') as f:
    f.write(training_script)

print("✅ Complete training script saved to: uav_ddos_training.py")