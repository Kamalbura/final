# Create a simple UAV DDoS detection environment simulator
class UAVDDoSEnvironment:
    """Simulated environment for UAV DDoS detection system"""
    
    def __init__(self):
        self.battery_levels = ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"]
        self.temperatures = ["Safe", "Warning", "Critical"]
        self.threat_states = ["Normal", "Confirming", "Confirmed"]
        
        # Power consumption per action (watts)
        self.power_consumption = {0: 3.0, 1: 5.5, 2: 9.0}  # No DDoS, XGBoost, TST
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_state = {
            'battery': np.random.choice(self.battery_levels[2:]),  # Start with decent battery
            'temperature': np.random.choice(self.temperatures[:2]),  # Start safe/warning
            'threat': 'Normal'
        }
        self.episode_power = 0.0
        self.episode_detections = 0
        self.timestep = 0
        return self.current_state
    
    def step(self, action):
        """Execute action and return next state, reward, done"""
        self.timestep += 1
        
        # Calculate immediate power consumption
        power_used = self.power_consumption[action]
        self.episode_power += power_used
        
        # Simulate threat detection success
        detection_success = self._simulate_detection(action, self.current_state['threat'])
        if detection_success:
            self.episode_detections += 1
        
        # Calculate reward
        reward = self._calculate_reward(action, power_used, detection_success)
        
        # Simulate state transitions
        next_state = self._simulate_state_transition()
        self.current_state = next_state
        
        # Episode ends after 20 timesteps or critical conditions
        done = (self.timestep >= 20 or 
                (self.current_state['battery'] == "0-20%" and action != 0) or
                (self.current_state['temperature'] == "Critical" and action != 0))
        
        return next_state, reward, done
    
    def _simulate_detection(self, action, threat):
        """Simulate detection success based on action and threat"""
        if action == 0:  # No DDoS
            return False
        elif action == 1:  # XGBoost
            success_rates = {'Normal': 0.8, 'Confirming': 0.6, 'Confirmed': 0.9}
        else:  # TST
            success_rates = {'Normal': 0.9, 'Confirming': 0.95, 'Confirmed': 0.95}
        
        return np.random.random() < success_rates[threat]
    
    def _calculate_reward(self, action, power_used, detection_success):
        """Calculate reward based on detection success, power efficiency, and safety"""
        reward = 0.0
        
        # Detection rewards
        if detection_success and self.current_state['threat'] != 'Normal':
            reward += 10.0  # Successful threat detection
        
        # Power efficiency penalty
        reward -= power_used * 0.5  # Penalize high power consumption
        
        # Safety rewards
        if self.current_state['battery'] == "0-20%" and action == 0:
            reward += 5.0  # Reward for protecting critical systems
        
        if self.current_state['temperature'] == "Critical" and action == 0:
            reward += 5.0  # Reward for thermal protection
        
        # Penalty for wrong actions in critical states
        if (self.current_state['battery'] == "0-20%" or 
            self.current_state['temperature'] == "Critical") and action != 0:
            reward -= 20.0  # Heavy penalty for unsafe actions
        
        return reward
    
    def _simulate_state_transition(self):
        """Simulate random state transitions"""
        new_state = self.current_state.copy()
        
        # Random threat evolution
        if np.random.random() < 0.3:  # 30% chance of threat change
            if self.current_state['threat'] == 'Normal':
                new_state['threat'] = np.random.choice(['Confirming', 'Normal'])
            elif self.current_state['threat'] == 'Confirming':
                new_state['threat'] = np.random.choice(['Normal', 'Confirmed', 'Confirming'])
            else:  # Confirmed
                new_state['threat'] = np.random.choice(['Normal', 'Confirmed'])
        
        # Random battery degradation (slight)
        if np.random.random() < 0.1:  # 10% chance
            current_idx = self.battery_levels.index(self.current_state['battery'])
            if current_idx > 0:
                new_state['battery'] = self.battery_levels[max(0, current_idx - 1)]
        
        # Random temperature changes
        if np.random.random() < 0.15:  # 15% chance
            new_state['temperature'] = np.random.choice(self.temperatures)
        
        return new_state

# Initialize environment and run minimal Q-learning training
env = UAVDDoSEnvironment()

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.2  # Exploration rate
num_episodes = 100

# Training metrics
episode_rewards = []
episode_powers = []
episode_detections = []

print("\n=== RUNNING MINIMAL Q-LEARNING TRAINING ===")
print(f"Training for {num_episodes} episodes...")

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    
    while True:
        # Get state index
        s_idx = get_state_index(state['battery'], state['temperature'], state['threat'])
        
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.choice(3)  # Explore
        else:
            action = np.argmax(Q_table[s_idx])  # Exploit
        
        # Take action
        next_state, reward, done = env.step(action)
        total_reward += reward
        
        # Get next state index
        s_next_idx = get_state_index(next_state['battery'], next_state['temperature'], next_state['threat'])
        
        # Q-learning update
        Q_table[s_idx, action] += alpha * (reward + gamma * np.max(Q_table[s_next_idx]) - Q_table[s_idx, action])
        
        state = next_state
        
        if done:
            break
    
    # Record episode metrics
    episode_rewards.append(total_reward)
    episode_powers.append(env.episode_power)
    episode_detections.append(env.episode_detections)
    
    # Decay epsilon
    epsilon = max(0.05, epsilon * 0.995)

print(f"Training completed!")
print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
print(f"Average power consumption: {np.mean(episode_powers):.2f}W ± {np.std(episode_powers):.2f}W")
print(f"Average detections per episode: {np.mean(episode_detections):.2f} ± {np.std(episode_detections):.2f}")