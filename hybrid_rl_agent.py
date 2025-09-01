#!/usr/bin/env python3
"""
Hybrid RL Agent for UAV DDoS Detection
Combines Q-table with Neural Network for exact + approximated learning
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import json
from datetime import datetime
import os
import random

class ActorCriticNetwork:
    """Neural approximation for state-action values"""
    
    def __init__(self, state_dim=4, action_dim=3):
        # Actor network (policy)
        self.actor = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(state_dim,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(action_dim, activation='softmax')
        ])
        
        # Critic network (value function)
        self.critic = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(state_dim,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)  # Single value output (state value)
        ])
        
        # Compile models
        self.actor.compile(optimizer='adam', loss='categorical_crossentropy')
        self.critic.compile(optimizer='adam', loss='mse')
    
    def encode_state(self, state):
        """Convert state dictionary to vector for neural network"""
        # Convert battery level string to numeric
        battery_map = {
            "0-20%": 0.1,
            "21-40%": 0.3,
            "41-60%": 0.5,
            "61-80%": 0.7,
            "81-100%": 0.9
        }
        
        # Convert temperature string to numeric
        temp_map = {
            "Safe": 0.2,
            "Warning": 0.6,
            "Critical": 0.9
        }
        
        # Convert threat state string to numeric
        threat_map = {
            "Normal": 0.1,
            "Confirming": 0.5,
            "Confirmed": 0.9
        }
        
        # Encode state as vector
        battery_val = battery_map.get(state['battery'], 0.5)
        temp_val = temp_map.get(state['temperature'], 0.5)
        threat_val = threat_map.get(state['threat'], 0.5)
        
        # Include time since last change if available
        time_val = state.get('time_since_change', 0.0) / 300.0  # Normalize to [0,1]
        
        return np.array([[battery_val, temp_val, threat_val, time_val]])
    
    def get_action_probs(self, state):
        """Get action probabilities from actor network"""
        state_encoded = self.encode_state(state)
        return self.actor.predict(state_encoded, verbose=0)[0]
    
    def get_value(self, state):
        """Get state value from critic network"""
        state_encoded = self.encode_state(state)
        return self.critic.predict(state_encoded, verbose=0)[0][0]
    
    def update(self, states, actions, rewards, next_states, dones):
        """Update both actor and critic networks with a batch of experiences"""
        # Encode all states and next_states
        encoded_states = np.vstack([self.encode_state(s) for s in states])
        encoded_next_states = np.vstack([self.encode_state(s) for s in next_states])
        
        # Get values for next states
        next_values = self.critic.predict(encoded_next_states, verbose=0).flatten()
        
        # Calculate targets for critic (TD targets)
        targets = np.array(rewards) + 0.99 * next_values * (1 - np.array(dones))
        
        # Update critic
        self.critic.train_on_batch(encoded_states, targets.reshape(-1, 1))
        
        # One-hot encode actions
        action_masks = np.zeros((len(actions), 3))
        for i, action in enumerate(actions):
            action_masks[i, action] = 1
        
        # Calculate advantages
        values = self.critic.predict(encoded_states, verbose=0).flatten()
        advantages = targets - values
        
        # Custom loss that incorporates advantages
        weighted_actions = action_masks * advantages.reshape(-1, 1)
        
        # Update actor
        self.actor.train_on_batch(encoded_states, weighted_actions)


class HybridRLAgent:
    """
    Hybrid RL Agent that combines:
    1. Q-table for exact state lookup
    2. Neural network for generalization
    3. Expert knowledge initialization
    """
    
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, 
                 epsilon=0.3, epsilon_decay=0.995, epsilon_min=0.05):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table with expert knowledge
        self.num_states = 45  # 5 battery × 3 temp × 3 threat
        self.num_actions = 3
        self.q_table = self._initialize_q_table()
        
        # Initialize neural approximator
        self.neural_net = ActorCriticNetwork()
        
        # Experience buffer for neural network training
        self.experience_buffer = []
        self.buffer_size = 1000
        self.batch_size = 32
        
        # Performance tracking
        self.training_metrics = {
            'episodes': [],
            'rewards': [],
            'expert_alignment': [],
            'safety_violations': [],
            'power_consumption': [],
            'epsilon_values': [],
            'thermal_metrics': [],
            'neural_usage': []
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
        """Hybrid action selection combining Q-table and neural net"""
        state_idx = self.env.get_state_index(state['battery'], state['temperature'], state['threat'])
        
        # Exploration with safety bias during training
        if training and random.random() < self.epsilon:
            return self._safe_random_action(state)
        
        # Try exact lookup in Q-table first
        try:
            # Get action from Q-table for known states
            q_values = self.q_table[state_idx]
            q_action = np.argmax(q_values)
            
            # Check confidence (are Q-values well separated?)
            q_confidence = q_values[q_action] - np.mean(q_values)
            
            # If high confidence in Q-table, use it
            if q_confidence > 20.0:  # Significant separation
                return q_action
            
            # Otherwise, blend with neural network
            neural_probs = self.neural_net.get_action_probs(state)
            
            # Weighted combination (75% Q-table, 25% neural)
            combined_values = 0.75 * q_values + 0.25 * neural_probs * 100.0
            
            return np.argmax(combined_values)
            
        except (IndexError, KeyError):
            # Fallback to neural network for unknown states
            neural_probs = self.neural_net.get_action_probs(state)
            return np.argmax(neural_probs)
    
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
        
        # Add to experience buffer for neural network training
        self.experience_buffer.append((state, action, reward, next_state, 1 if next_state == state else 0))
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)  # Remove oldest experience
    
    def train_neural_network(self):
        """Train neural network with batch from experience buffer"""
        if len(self.experience_buffer) < self.batch_size:
            return 0  # Not enough samples
            
        # Sample batch
        batch_indices = np.random.choice(len(self.experience_buffer), self.batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in batch_indices]
        
        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Update neural network
        self.neural_net.update(states, actions, rewards, next_states, dones)
        
        return 1  # Successfully trained
    
    def train(self, num_episodes=200):
        """Train agent with expert guidance"""
        print(f"Starting hybrid training: {num_episodes} episodes")
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            expert_agreements = 0
            total_actions = 0
            neural_usage = 0
            
            while True:
                # Get action
                action = self.get_action(state, training=True)
                expert_action = self.env.get_expert_action(state)
                
                # Track expert alignment
                if action == expert_action:
                    expert_agreements += 1
                total_actions += 1
                
                # Take step
                next_state, reward, done = self.env.step(action)
                total_reward += reward
                
                # Update Q-table
                self.update_q_value(state, action, reward, next_state)
                
                # Train neural network periodically
                if episode % 5 == 0:  # Every 5 episodes
                    neural_usage += self.train_neural_network()
                
                state = next_state
                
                if done:
                    break
            
            # Decay exploration
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Track metrics
            self.training_metrics['episodes'].append(episode)
            self.training_metrics['rewards'].append(total_reward)
            self.training_metrics['expert_alignment'].append(expert_agreements / total_actions if total_actions > 0 else 0)
            self.training_metrics['safety_violations'].append(self.env.safety_violations)
            self.training_metrics['power_consumption'].append(self.env.total_power_consumed)
            self.training_metrics['epsilon_values'].append(self.epsilon)
            self.training_metrics['thermal_metrics'].append(self.env.thermal_simulator.get_temperature())
            self.training_metrics['neural_usage'].append(neural_usage / total_actions if total_actions > 0 else 0)
            
            if episode % 20 == 0:
                print(f"Episode {episode}: Reward={total_reward:.1f}, "
                      f"Expert Align={expert_agreements/total_actions if total_actions > 0 else 0:.3f}, "
                      f"Safety Violations={self.env.safety_violations}, "
                      f"Temp={self.env.thermal_simulator.get_temperature():.1f}°C")
        
        return self.training_metrics
    
    def save_model(self, filepath):
        """Save hybrid model (Q-table + neural weights)"""
        model_dir = os.path.dirname(filepath)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # Save Q-table and training metrics
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
            
        # Save neural network weights
        neural_path = filepath.replace('.json', '_neural')
        self.neural_net.actor.save_weights(f"{neural_path}_actor.h5")
        self.neural_net.critic.save_weights(f"{neural_path}_critic.h5")
        
        print(f"Model saved to: {filepath}")
        print(f"Neural weights saved to: {neural_path}_actor.h5 and {neural_path}_critic.h5")
        
    def load_model(self, filepath):
        """Load hybrid model (Q-table + neural weights)"""
        try:
            with open(filepath, 'r') as f:
                model_data = json.load(f)
                
            self.q_table = np.array(model_data['q_table'])
            self.lr = model_data['training_params']['learning_rate']
            self.gamma = model_data['training_params']['discount_factor']
            self.epsilon = model_data['training_params']['epsilon']
            
            # Load neural network weights
            neural_path = filepath.replace('.json', '_neural')
            self.neural_net.actor.load_weights(f"{neural_path}_actor.h5")
            self.neural_net.critic.load_weights(f"{neural_path}_critic.h5")
            
            print(f"Model loaded from: {filepath}")
            print(f"Neural weights loaded from: {neural_path}_actor.h5 and {neural_path}_critic.h5")
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
