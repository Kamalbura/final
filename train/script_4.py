# Create deployment/inference script
deployment_script = '''#!/usr/bin/env python3
"""
UAV Power-Aware DDoS Detection - Production Deployment Script
Loads trained Q-table and makes real-time decisions
"""

import numpy as np
import json
import time
import logging
from datetime import datetime

class UAVDDoSAgent:
    def __init__(self, model_path):
        """Load trained model for production use"""
        self.load_model(model_path)
        self.setup_logging()
        
        # State space definitions (must match training)
        self.battery_levels = ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"]
        self.temperatures = ["Safe", "Warning", "Critical"]
        self.threat_states = ["Normal", "Confirming", "Confirmed"]
        self.action_labels = ["No_DDoS", "XGBoost", "TST"]
        self.power_consumption = {0: 3.0, 1: 5.5, 2: 9.0}
        
        # Runtime metrics
        self.decision_log = []
        self.power_consumed = 0.0
        self.start_time = time.time()
        
        print(f"✅ UAV DDoS Agent loaded and ready for deployment")
    
    def load_model(self, model_path):
        """Load trained Q-table from file"""
        try:
            with open(model_path, 'r') as f:
                model_data = json.load(f)
            
            self.q_table = np.array(model_data['q_table'])
            self.expert_lookup = {tuple(k.split('|')): v for k, v in model_data['expert_lookup'].items()} if isinstance(list(model_data['expert_lookup'].keys())[0], str) else model_data['expert_lookup']
            self.training_params = model_data['training_params']
            
            print(f"✅ Model loaded from: {model_path}")
            print(f"   Q-table shape: {self.q_table.shape}")
            print(f"   Training timestamp: {model_data.get('timestamp', 'Unknown')}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def setup_logging(self):
        """Setup logging for production monitoring"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('uav_ddos_decisions.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_state_index(self, battery, temp, threat):
        """Convert state to Q-table index"""
        try:
            battery_idx = self.battery_levels.index(battery)
            temp_idx = self.temperatures.index(temp)
            threat_idx = self.threat_states.index(threat)
            return battery_idx * 9 + temp_idx * 3 + threat_idx
        except ValueError as e:
            self.logger.error(f"Invalid state values: {battery}, {temp}, {threat}")
            raise
    
    def get_expert_action(self, state):
        """Get expert recommendation for comparison"""
        state_key = (state['battery'], state['temperature'], state['threat'])
        return self.expert_lookup.get(state_key, 0)
    
    def make_decision(self, current_state, log_decision=True):
        """Make real-time decision based on current UAV state"""
        try:
            # Validate input state
            self._validate_state(current_state)
            
            # Get state index
            state_idx = self.get_state_index(
                current_state['battery'],
                current_state['temperature'], 
                current_state['threat']
            )
            
            # Get best action from Q-table
            action = int(np.argmax(self.q_table[state_idx]))
            action_label = self.action_labels[action]
            power_cost = self.power_consumption[action]
            
            # Get expert recommendation for comparison
            expert_action = self.get_expert_action(current_state)
            expert_alignment = (action == expert_action)
            
            # Safety check
            safety_violation = self._check_safety_violation(action, current_state)
            
            # Log decision
            decision_info = {
                'timestamp': datetime.now().isoformat(),
                'state': current_state,
                'action': action,
                'action_label': action_label,
                'power_cost': power_cost,
                'expert_action': expert_action,
                'expert_alignment': expert_alignment,
                'safety_violation': safety_violation,
                'q_values': self.q_table[state_idx].tolist()
            }
            
            if log_decision:
                self.decision_log.append(decision_info)
                self.power_consumed += power_cost
                
                self.logger.info(f"DECISION: {action_label} | State: {current_state} | "
                               f"Power: {power_cost}W | Expert Align: {expert_alignment} | "
                               f"Safe: {not safety_violation}")
                
                if safety_violation:
                    self.logger.warning("⚠️  SAFETY VIOLATION DETECTED!")
                
                if not expert_alignment:
                    self.logger.warning(f"⚠️  Expert mismatch: Got {action_label}, "
                                      f"Expert suggests {self.action_labels[expert_action]}")
            
            return {
                'action': action,
                'action_label': action_label,
                'power_cost': power_cost,
                'confidence': float(np.max(self.q_table[state_idx])),
                'expert_alignment': expert_alignment,
                'safety_status': 'SAFE' if not safety_violation else 'VIOLATION',
                'decision_info': decision_info
            }
            
        except Exception as e:
            self.logger.error(f"Decision making failed: {e}")
            # Fallback to safest action
            return {
                'action': 0,
                'action_label': 'No_DDoS',
                'power_cost': 3.0,
                'confidence': 0.0,
                'expert_alignment': True,
                'safety_status': 'SAFE',
                'error': str(e)
            }
    
    def _validate_state(self, state):
        """Validate input state format"""
        required_keys = ['battery', 'temperature', 'threat']
        for key in required_keys:
            if key not in state:
                raise ValueError(f"Missing required state key: {key}")
        
        if state['battery'] not in self.battery_levels:
            raise ValueError(f"Invalid battery level: {state['battery']}")
        if state['temperature'] not in self.temperatures:
            raise ValueError(f"Invalid temperature: {state['temperature']}")
        if state['threat'] not in self.threat_states:
            raise ValueError(f"Invalid threat state: {state['threat']}")
    
    def _check_safety_violation(self, action, state):
        """Check if action violates safety constraints"""
        return ((state['battery'] == "0-20%" and action != 0) or
                (state['temperature'] == "Critical" and action != 0))
    
    def get_performance_summary(self):
        """Get runtime performance summary"""
        if not self.decision_log:
            return "No decisions made yet"
        
        total_decisions = len(self.decision_log)
        safety_violations = sum(1 for d in self.decision_log if d['safety_violation'])
        expert_alignments = sum(1 for d in self.decision_log if d['expert_alignment'])
        action_counts = [0, 0, 0]
        
        for decision in self.decision_log:
            action_counts[decision['action']] += 1
        
        runtime_hours = (time.time() - self.start_time) / 3600
        
        summary = f"""
UAV DDoS Agent Performance Summary
==================================
Runtime: {runtime_hours:.2f} hours
Total Decisions: {total_decisions}
Power Consumed: {self.power_consumed:.1f}W
Safety Violations: {safety_violations} ({safety_violations/total_decisions*100:.1f}%)
Expert Alignment: {expert_alignments} ({expert_alignments/total_decisions*100:.1f}%)

Action Distribution:
- No_DDoS: {action_counts[0]} ({action_counts[0]/total_decisions*100:.1f}%)
- XGBoost: {action_counts[1]} ({action_counts[1]/total_decisions*100:.1f}%)
- TST: {action_counts[2]} ({action_counts[2]/total_decisions*100:.1f}%)
"""
        return summary
    
    def export_decision_log(self, filepath):
        """Export decision log for analysis"""
        with open(filepath, 'w') as f:
            json.dump(self.decision_log, f, indent=2)
        print(f"✅ Decision log exported to: {filepath}")

# EXAMPLE USAGE AND TESTING
if __name__ == "__main__":
    print("="*80)
    print("UAV POWER-AWARE DDoS AGENT - DEPLOYMENT TEST")
    print("="*80)
    
    # Load trained model
    try:
        agent = UAVDDoSAgent('trained_uav_ddos_model.json')
        
        # Test scenarios
        test_scenarios = [
            {'battery': '81-100%', 'temperature': 'Safe', 'threat': 'Normal'},
            {'battery': '41-60%', 'temperature': 'Warning', 'threat': 'Confirming'},
            {'battery': '0-20%', 'temperature': 'Safe', 'threat': 'Confirmed'},
            {'battery': '61-80%', 'temperature': 'Critical', 'threat': 'Normal'},
            {'battery': '21-40%', 'temperature': 'Safe', 'threat': 'Confirming'}
        ]
        
        print("\\n🧪 TESTING DECISION MAKING:")
        print("-" * 60)
        
        for i, scenario in enumerate(test_scenarios, 1):
            result = agent.make_decision(scenario)
            print(f"Test {i}: {scenario}")
            print(f"   → Decision: {result['action_label']} ({result['power_cost']}W)")
            print(f"   → Safety: {result['safety_status']}")
            print(f"   → Expert Alignment: {result['expert_alignment']}")
            print()
        
        # Performance summary
        print(agent.get_performance_summary())
        
        # Export log
        agent.export_decision_log('test_decisions.json')
        
    except FileNotFoundError:
        print("❌ Model file not found. Please run training script first.")
    except Exception as e:
        print(f"❌ Error: {e}")
'''

# Save deployment script
with open('uav_ddos_deployment.py', 'w') as f:
    f.write(deployment_script)

print("✅ Complete deployment script saved to: uav_ddos_deployment.py")