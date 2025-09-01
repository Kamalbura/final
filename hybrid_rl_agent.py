#!/usr/bin/env python3
"""
Hybrid RL Agent for UAV DDoS defense using lookup table with Q-learning
Combines expert knowledge with reinforcement learning
"""

import numpy as np
import tensorflow as tf
import json
import os
from collections import defaultdict
import itertools
from typing import Dict, Tuple, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridRLAgent:
    """
    Hybrid RL Agent that combines lookup table, Q-learning and neural network
    for UAV DDoS defense with power and thermal awareness
    """
    
    def __init__(self, env):
        # Store environment
        self.env = env
        
        # State discretization
        self.setup_state_discretization()
        
        # Q-learning parameters
        self.alpha = 0.1          # Learning rate
        self.gamma = 0.95         # Discount factor
        self.epsilon = 0.3        # Initial exploration rate
        self.epsilon_min = 0.01   # Minimum exploration rate
        self.epsilon_decay = 0.995 # Exploration decay rate
        
        # Initialize Q-table with expert knowledge
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.visit_count = defaultdict(lambda: defaultdict(int))
        self.warm_start_q_table()
        
        # Simple neural network for continuous state representation
        self.neural_net = self._build_neural_network()
        self.neural_buffer = []
        self.neural_batch_size = 64
        self.neural_update_frequency = 10
        self.neural_usage_rate = 0.0  # Tracks how often neural net is used
        
        # Metrics tracking
        self.metrics = {
            'episodes': [],
            'rewards': [],
            'expert_alignment': [],
            'power_consumption': [],
            'thermal_metrics': [],
            'epsilon_values': [],
            'neural_usage': []
        }
        
        logger.info("Hybrid RL Agent initialized with warm-start Q-table and neural network")
    
    def setup_state_discretization(self):
        """Setup state discretization for lookup table"""
        # Define state bins
        self.state_bins = {
            'temperature': [0, 55, 65, 70, 80, 100],     # 5 thermal zones
            'battery': [0, 20, 40, 70, 100],             # 4 battery levels
            'threat': [0, 1, 2],                         # 3 threat states
        }
        
        # Calculate state space size
        self.state_space_size = 1
        for key, bins in self.state_bins.items():
            self.state_space_size *= (len(bins) - 1)
        
        logger.info(f"State space discretized into {self.state_space_size} states")
    
    def warm_start_q_table(self):
        """Initialize Q-table with expert knowledge"""
        # Generate all possible discrete states
        temp_zones = len(self.state_bins['temperature']) - 1
        battery_zones = len(self.state_bins['battery']) - 1
        threat_states = len(self.state_bins['threat']) - 1
        
        all_states = list(itertools.product(range(temp_zones), range(battery_zones), range(threat_states)))
        
        # Initialize Q-table with expert values
        for state in all_states:
            # Convert to continuous state approximation for expert policy
            continuous_state = self._discrete_to_continuous(state)
            
            # Get expert action for this state
            expert_action = self.env.get_expert_action(continuous_state)
            
            # Initialize Q-values
            for action in range(3):  # 3 actions: No DDoS, XGBoost, TST
                if action == expert_action:
                    self.q_table[state][action] = 10.0  # High value for expert action
                else:
                    self.q_table[state][action] = 2.0   # Lower value for other actions
                
                # Apply safety constraints
                safe, _ = self.env.is_safe_action(continuous_state, action)
                if not safe:
                    self.q_table[state][action] = -100.0  # Strong negative value for unsafe actions
        
        logger.info(f"Q-table initialized with expert knowledge for {len(all_states)} states")
    
    def _build_neural_network(self):
        """Build a simple neural network for continuous state representation"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, activation='relu', input_shape=(3,)),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(3)  # 3 actions: No DDoS, XGBoost, TST
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def _discretize_state(self, state: Dict) -> Tuple:
        """Convert continuous state to discrete state tuple"""
        discrete = []
        
        # Temperature zone
        temp = state.get('temperature', 50)
        discrete.append(np.digitize(temp, self.state_bins['temperature']) - 1)
        
        # Battery level
        battery = state.get('battery', 80)
        discrete.append(np.digitize(battery, self.state_bins['battery']) - 1)
        
        # Threat level (already discrete)
        threat = state.get('threat', 0)
        discrete.append(min(threat, 2))
        
        return tuple(discrete)
    
    def _discrete_to_continuous(self, discrete_state: Tuple) -> Dict:
        """Convert discrete state back to continuous approximation"""
        temp_zone, battery_zone, threat_level = discrete_state
        
        # Use midpoints of bins
        temp_bins = self.state_bins['temperature']
        battery_bins = self.state_bins['battery']
        
        # Get midpoints of the bins
        temp = (temp_bins[temp_zone] + temp_bins[temp_zone + 1]) / 2
        battery = (battery_bins[battery_zone] + battery_bins[battery_zone + 1]) / 2
        
        return {
            'temperature': temp,
            'battery': battery,
            'threat': threat_level
        }
    
    def _state_to_array(self, state: Dict) -> np.ndarray:
        """Convert state dict to array for neural network"""
        return np.array([
            state.get('temperature', 50) / 100.0,  # Normalize to [0,1]
            state.get('battery', 50) / 100.0,      # Normalize to [0,1]
            state.get('threat', 0) / 2.0           # Normalize to [0,1]
        ]).reshape(1, -1)
    
    def get_action(self, state: Dict, training: bool = True) -> int:
        """Get action using hybrid approach (Q-table + neural network)"""
        # Discretize state for Q-table
        discrete_state = self._discretize_state(state)
        
        # Get valid actions (remove unsafe actions)
        valid_actions = []
        for action in range(3):
            safe, _ = self.env.is_safe_action(state, action)
            if safe:
                valid_actions.append(action)
        
        if not valid_actions:
            return 0  # Default to No DDoS if no safe actions
        
        # Exploration during training
        if training and np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        
        # Decide whether to use Q-table or neural network
        use_neural = False
        
        # Check if this state has been visited enough in the Q-table
        if sum(self.visit_count[discrete_state].values()) < 5:
            # Not enough Q-table data, try neural network
            if np.random.random() < 0.5 and training:
                use_neural = True
                self.neural_usage_rate = 0.9 * self.neural_usage_rate + 0.1 * 1.0
            else:
                self.neural_usage_rate = 0.9 * self.neural_usage_rate + 0.1 * 0.0
        
        if use_neural:
            # Use neural network prediction
            state_array = self._state_to_array(state)
            q_values = self.neural_net.predict(state_array, verbose=0)[0]
            
            # Filter to only valid actions
            valid_q_values = {action: q_values[action] for action in valid_actions}
            return max(valid_q_values, key=valid_q_values.get)
        else:
            # Use Q-table
            valid_q_values = {action: self.q_table[discrete_state][action] for action in valid_actions}
            return max(valid_q_values, key=valid_q_values.get)
    
    def update(self, state: Dict, action: int, reward: float, next_state: Dict, done: bool):
        """Update agent with experience"""
        # Update Q-table
        self._update_q_table(state, action, reward, next_state, done)
        
        # Store experience for neural network
        self.neural_buffer.append((state, action, reward, next_state, done))
        
        # Periodically train neural network
        if len(self.neural_buffer) >= self.neural_batch_size:
            self._update_neural_network()
    
    def _update_q_table(self, state: Dict, action: int, reward: float, next_state: Dict, done: bool):
        """Update Q-table with experience"""
        # Discretize states
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)
        
        # Get current Q-value
        current_q = self.q_table[discrete_state][action]
        
        # Expert alignment bonus
        expert_action = self.env.get_expert_action(state)
        expert_bonus = 2.0 if action == expert_action else -0.5
        
        # Compute target Q-value with expert bonus
        if done:
            target_q = reward + expert_bonus
        else:
            # Get best next action from Q-table
            max_next_q = max(self.q_table[discrete_next_state].values())
            target_q = reward + expert_bonus + self.gamma * max_next_q
        
        # Apply safety check (prevent learning unsafe actions)
        safe, _ = self.env.is_safe_action(state, action)
        if not safe:
            target_q = -100.0  # Strong penalty for unsafe actions
        
        # Update Q-value with adaptive learning rate
        visit_count = self.visit_count[discrete_state][action]
        adaptive_alpha = self.alpha / (1.0 + 0.01 * visit_count)  # Decay learning rate with visits
        
        self.q_table[discrete_state][action] = current_q + adaptive_alpha * (target_q - current_q)
        self.visit_count[discrete_state][action] += 1
    
    def _update_neural_network(self):
        """Update neural network with experiences from buffer"""
        if len(self.neural_buffer) < self.neural_batch_size:
            return
        
        # Sample batch from buffer
        indices = np.random.choice(len(self.neural_buffer), self.neural_batch_size, replace=False)
        batch = [self.neural_buffer[i] for i in indices]
        
        # Prepare training data
        states = np.zeros((self.neural_batch_size, 3))
        targets = np.zeros((self.neural_batch_size, 3))
        
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            # Convert state to array
            states[i] = self._state_to_array(state).flatten()
            
            # Get current Q-values prediction
            target = self.neural_net.predict(states[i].reshape(1, -1), verbose=0)[0]
            
            if done:
                target[action] = reward
            else:
                # Get next state Q-values
                next_state_array = self._state_to_array(next_state)
                next_q_values = self.neural_net.predict(next_state_array, verbose=0)[0]
                
                # Apply Bellman equation
                target[action] = reward + self.gamma * np.max(next_q_values)
            
            targets[i] = target
        
        # Train neural network
        self.neural_net.fit(states, targets, epochs=1, verbose=0)
        
        # Clear buffer
        self.neural_buffer = []
    
    def train(self, num_episodes=300, max_steps=300):
        """Train the agent"""
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            expert_agreements = 0
            total_actions = 0
            
            for step in range(max_steps):
                # Get action
                action = self.get_action(state, training=True)
                
                # Check if matches expert
                expert_action = self.env.get_expert_action(state)
                if action == expert_action:
                    expert_agreements += 1
                total_actions += 1
                
                # Take action
                next_state, reward, done = self.env.step(action)
                
                # Update agent
                self.update(state, action, reward, next_state, done)
                
                # Update metrics
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Update exploration rate
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Record metrics
            expert_alignment = expert_agreements / total_actions if total_actions > 0 else 0
            
            self.metrics['episodes'].append(episode)
            self.metrics['rewards'].append(total_reward)
            self.metrics['expert_alignment'].append(expert_alignment)
            self.metrics['power_consumption'].append(self.env.total_power_consumed)
            self.metrics['thermal_metrics'].append(self.env.thermal_simulator.get_temperature())
            self.metrics['epsilon_values'].append(self.epsilon)
            self.metrics['neural_usage'].append(self.neural_usage_rate)
            
            # Log progress
            if episode % 10 == 0:
                logger.info(f"Episode {episode}/{num_episodes}: "
                           f"Reward={total_reward:.2f}, "
                           f"Expert={expert_alignment:.2%}, "
                           f"Temperature={self.env.thermal_simulator.get_temperature():.1f}°C, "
                           f"Power={self.env.total_power_consumed:.2f}W")
        
        logger.info(f"Training completed after {num_episodes} episodes")
        return self.metrics
    
    def save_model(self, filepath):
        """Save the trained model"""
        # Save Q-table (convert to regular dict for JSON serialization)
        q_table_dict = {str(k): dict(v) for k, v in self.q_table.items()}
        
        # Save visit counts
        visit_count_dict = {str(k): dict(v) for k, v in self.visit_count.items()}
        
        # Save neural network
        nn_path = filepath.replace('.json', '_nn')
        self.neural_net.save(nn_path)
        
        # Create model data
        model_data = {
            'q_table': q_table_dict,
            'visit_count': visit_count_dict,
            'state_bins': self.state_bins,
            'neural_network_path': nn_path,
            'params': {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon
            }
        }
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"Model saved to {filepath} and {nn_path}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        # Load from JSON
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        # Load Q-table
        self.q_table = defaultdict(lambda: defaultdict(float))
        for state_str, actions in model_data['q_table'].items():
            state = tuple(map(int, state_str.strip('()').split(', ')))
            for action_str, value in actions.items():
                self.q_table[state][int(action_str)] = value
        
        # Load visit counts
        self.visit_count = defaultdict(lambda: defaultdict(int))
        for state_str, actions in model_data['visit_count'].items():
            state = tuple(map(int, state_str.strip('()').split(', ')))
            for action_str, value in actions.items():
                self.visit_count[state][int(action_str)] = value
        
        # Load state bins
        self.state_bins = model_data['state_bins']
        
        # Load parameters
        self.alpha = model_data['params']['alpha']
        self.gamma = model_data['params']['gamma']
        self.epsilon = model_data['params']['epsilon']
        
        # Load neural network
        nn_path = model_data['neural_network_path']
        self.neural_net = tf.keras.models.load_model(nn_path)
        
        logger.info(f"Model loaded from {filepath}")
