# 2. Q-LEARNING AGENT WITH EXPERT INITIALIZATION

class ExpertGuidedQAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, 
                 epsilon=0.3, epsilon_decay=0.995, epsilon_min=0.05):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table
        self.num_states = 45  # 5 * 3 * 3
        self.num_actions = 3
        self.q_table = self._initialize_q_table()
        
        # Performance tracking
        self.training_metrics = {
            'episodes': [],
            'rewards': [],
            'expert_alignment': [],
            'safety_violations': [],
            'power_consumption': [],
            'epsilon_values': []
        }
        
        print(f"✅ Q-Agent initialized with expert-guided Q-table")
        print(f"   Learning rate: {self.lr}")
        print(f"   Discount factor: {self.gamma}")
        print(f"   Initial epsilon: {self.epsilon}")
    
    def _initialize_q_table(self):
        """Initialize Q-table with expert knowledge"""
        # Start with small random values
        q_table = np.random.normal(0, 0.1, (self.num_states, self.num_actions))
        
        # Set HIGH values for expert actions
        for state_key, expert_action in self.env.expert_lookup.items():
            battery, temp, threat = state_key
            state_idx = self.env.get_state_index(battery, temp, threat)
            
            # Give expert actions very high initial Q-values
            q_table[state_idx, expert_action] = 100.0  # Much higher than random actions
            
            # Set lower values for non-expert actions
            for action in range(self.num_actions):
                if action != expert_action:
                    q_table[state_idx, action] = -10.0  # Discourage non-expert actions
        
        print(f"✅ Q-table initialized with expert knowledge")
        print(f"   Expert actions start with Q-value: 100.0")
        print(f"   Non-expert actions start with Q-value: -10.0")
        
        return q_table
    
    def get_action(self, state, training=True):
        """Get action using epsilon-greedy policy"""
        state_idx = self.env.get_state_index(state['battery'], state['temperature'], state['threat'])
        
        # During training: epsilon-greedy
        if training and random.random() < self.epsilon:
            # Explore: but bias towards safer actions
            action = self._safe_random_action(state)
        else:
            # Exploit: choose best Q-value action
            action = np.argmax(self.q_table[state_idx])
        
        return action
    
    def _safe_random_action(self, state):
        """Random action with safety bias"""
        # Always choose No_DDoS for critical conditions (safety first)
        if state['battery'] == "0-20%" or state['temperature'] == "Critical":
            return 0
        
        # For other conditions, random choice but weighted towards expert action
        expert_action = self.env.get_expert_action(state)
        
        # 70% chance to choose expert action even during exploration
        if random.random() < 0.7:
            return expert_action
        else:
            # 30% chance for true random exploration
            return random.choice([0, 1, 2])
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-table using Q-learning rule"""
        state_idx = self.env.get_state_index(state['battery'], state['temperature'], state['threat'])
        next_state_idx = self.env.get_state_index(next_state['battery'], next_state['temperature'], next_state['threat'])
        
        # Q-learning update rule
        old_q = self.q_table[state_idx, action]
        next_max_q = np.max(self.q_table[next_state_idx])
        
        new_q = old_q + self.lr * (reward + self.gamma * next_max_q - old_q)
        self.q_table[state_idx, action] = new_q
    
    def train(self, num_episodes=200):
        """Train the agent with expert guidance"""
        print(f"\n🚀 STARTING TRAINING: {num_episodes} episodes")
        print("="*60)
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            expert_agreements = 0
            total_actions = 0
            
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
                
                state = next_state
                
                if done:
                    break
            
            # Update epsilon (decay exploration over time)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Track metrics
            expert_alignment = expert_agreements / total_actions
            self.training_metrics['episodes'].append(episode)
            self.training_metrics['rewards'].append(total_reward)
            self.training_metrics['expert_alignment'].append(expert_alignment)
            self.training_metrics['safety_violations'].append(self.env.safety_violations)
            self.training_metrics['power_consumption'].append(self.env.total_power_consumed)
            self.training_metrics['epsilon_values'].append(self.epsilon)
            
            # Progress reporting
            if episode % 20 == 0:
                print(f"Episode {episode:3d}: Reward={total_reward:6.1f}, "
                      f"Expert Align={expert_alignment:.3f}, "
                      f"Safety Violations={self.env.safety_violations}, "
                      f"ε={self.epsilon:.3f}")
        
        print("\n✅ TRAINING COMPLETED")
        return self.training_metrics
    
    def evaluate_policy(self, num_eval_episodes=50):
        """Evaluate trained policy"""
        print(f"\n📊 EVALUATING POLICY: {num_eval_episodes} episodes")
        print("="*50)
        
        eval_metrics = {
            'rewards': [],
            'expert_alignment': [],
            'safety_violations': [],
            'power_efficiency': [],
            'action_distribution': [0, 0, 0]
        }
        
        # Test on all possible states
        state_performance = {}
        
        for episode in range(num_eval_episodes):
            state = self.env.reset()
            total_reward = 0
            expert_agreements = 0
            total_actions = 0
            
            while True:
                # Get best action (no exploration)
                action = self.get_action(state, training=False)
                expert_action = self.env.get_expert_action(state)
                
                # Track metrics
                if action == expert_action:
                    expert_agreements += 1
                total_actions += 1
                eval_metrics['action_distribution'][action] += 1
                
                # Store state-action performance
                state_key = (state['battery'], state['temperature'], state['threat'])
                if state_key not in state_performance:
                    state_performance[state_key] = {'actions': [], 'expert_actions': []}
                state_performance[state_key]['actions'].append(action)
                state_performance[state_key]['expert_actions'].append(expert_action)
                
                # Take step
                next_state, reward, done = self.env.step(action)
                total_reward += reward
                
                state = next_state
                if done:
                    break
            
            eval_metrics['rewards'].append(total_reward)
            eval_metrics['expert_alignment'].append(expert_agreements / total_actions)
            eval_metrics['safety_violations'].append(self.env.safety_violations)
            eval_metrics['power_efficiency'].append(self.env.total_power_consumed)
        
        # Calculate final metrics
        avg_reward = np.mean(eval_metrics['rewards'])
        avg_alignment = np.mean(eval_metrics['expert_alignment'])
        total_violations = sum(eval_metrics['safety_violations'])
        avg_power = np.mean(eval_metrics['power_efficiency'])
        
        print(f"\n📈 EVALUATION RESULTS:")
        print(f"   Average Reward: {avg_reward:.2f}")
        print(f"   Expert Alignment: {avg_alignment:.3f} ({avg_alignment*100:.1f}%)")
        print(f"   Total Safety Violations: {total_violations}")
        print(f"   Average Power Consumption: {avg_power:.1f}W")
        
        action_total = sum(eval_metrics['action_distribution'])
        print(f"\n🎯 ACTION DISTRIBUTION:")
        for i, count in enumerate(eval_metrics['action_distribution']):
            percentage = (count / action_total) * 100
            print(f"   {env.action_labels[i]}: {count} ({percentage:.1f}%)")
        
        return eval_metrics, state_performance

# Initialize and train the agent
agent = ExpertGuidedQAgent(env)
training_results = agent.train(num_episodes=200)

print(f"\n💾 Training completed with {len(training_results['episodes'])} episodes")