import logging
import json
from datetime import datetime
import os

class UAVDDoSLogger:
    """Enhanced logging for UAV DDoS hybrid RL system"""
    
    def __init__(self, log_dir='logs', log_level=logging.INFO):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # File handler for all logs
        self.setup_file_logger(log_level)
        
        # Decision history
        self.decisions = []
        self.safety_violations = []
        self.power_history = []
        self.thermal_events = []
        
    def setup_file_logger(self, log_level):
        """Setup file logger with proper formatting"""
        self.logger = logging.getLogger('uav_ddos_rl')
        self.logger.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        log_file = os.path.join(self.log_dir, f'uav_ddos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        # Add handler to logger if not already added
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
        
        self.logger.info("UAV DDoS RL Logger initialized")
    
    def log_decision(self, state, action, expert_action, reward, power_used, decision_source=None, temperature=None):
        """Log a decision made by the agent with enhanced thermal and decision source tracking"""
        # Get temperature value
        if temperature is None and isinstance(state, dict):
            temperature = state.get('temperature')
            
            # Convert string temperature to numeric value if needed
            if temperature == "Safe":
                temp_value = 50
            elif temperature == "Warning":
                temp_value = 65
            elif temperature == "Critical":
                temp_value = 80
            elif isinstance(temperature, (int, float)):
                temp_value = temperature
            else:
                temp_value = 0
        else:
            temp_value = temperature if temperature is not None else 0
        
        # Determine temperature zone
        if temp_value < 55:
            temp_zone = "cold"
        elif temp_value < 60:
            temp_zone = "cool"
        elif temp_value < 65:
            temp_zone = "warm"
        elif temp_value < 70:
            temp_zone = "hot"
        else:
            temp_zone = "danger"
            
        # Check for thermal event
        thermal_event = temp_zone in ["hot", "danger"]
        if thermal_event:
            self.thermal_events.append({
                'timestamp': datetime.now().isoformat(),
                'temperature': temp_value,
                'zone': temp_zone
            })
            
        # Check for safety violations
        safety_violation = self._is_safety_violation(action, state)
        if safety_violation:
            self.safety_violations.append({
                'timestamp': datetime.now().isoformat(),
                'state': state,
                'action': action,
                'violation_type': 'safety'
            })
        
        # Create action label
        action_labels = ["No_DDoS", "XGBoost", "TST"]
        action_label = action_labels[action] if 0 <= action < len(action_labels) else f"Unknown-{action}"
        
        # Create decision record
        decision = {
            'timestamp': datetime.now().isoformat(),
            'state': state,
            'action': action,
            'action_label': action_label,
            'expert_action': expert_action,
            'expert_alignment': action == expert_action,
            'reward': reward,
            'power_cost': power_used,
            'safety_violation': safety_violation,
            'temperature_value': temp_value,
            'temperature_zone': temp_zone,
            'thermal_event': thermal_event,
            'decision_source': decision_source or ('expert_table' if action == expert_action else 'q_table')
        }
        
        self.decisions.append(decision)
        self.power_history.append(power_used)
        
        # Log based on importance
        if safety_violation:
            self.logger.warning(f"Safety violation: {action_label} in {state}")
        elif thermal_event:
            self.logger.warning(f"Thermal event: {temp_value}°C ({temp_zone}) with action {action_label}")
        else:
            self.logger.info(f"Decision: {action_label} in state {state}")
            
        return decision
    
    def _is_safety_violation(self, action, state):
        """Check if action violates safety constraints"""
        if not isinstance(state, dict):
            return False
            
        # Critical battery with any action other than No_DDoS
        if state.get('battery') == "0-20%" and action != 0:
            return True
        
        # Critical temperature with any action other than No_DDoS
        if state.get('temperature') == "Critical" and action != 0:
            return True
            
        return False
    
    def save_decision_log(self, filename='decision_log.json'):
        """Save decisions to JSON file"""
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.decisions, f, indent=2)
        self.logger.info(f"Decision log saved to {filepath}")
        return filepath
    
    def save_thermal_events(self, filename='thermal_events.json'):
        """Save thermal events to JSON file"""
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.thermal_events, f, indent=2)
        self.logger.info(f"Thermal events saved to {filepath}")
        return filepath
    
    def get_summary_stats(self):
        """Get summary statistics"""
        if not self.decisions:
            return {"error": "No decisions recorded"}
        
        total = len(self.decisions)
        expert_matches = sum(d['expert_alignment'] for d in self.decisions)
        violations = len(self.safety_violations)
        thermal_events = len(self.thermal_events)
        
        # Count decision sources
        decision_sources = {}
        for d in self.decisions:
            source = d.get('decision_source', 'unknown')
            decision_sources[source] = decision_sources.get(source, 0) + 1
        
        # Action distribution
        action_counts = {}
        for d in self.decisions:
            action_label = d['action_label']
            action_counts[action_label] = action_counts.get(action_label, 0) + 1
        
        # Temperature zone distribution
        temp_zones = {}
        for d in self.decisions:
            zone = d.get('temperature_zone', 'unknown')
            temp_zones[zone] = temp_zones.get(zone, 0) + 1
        
        return {
            'total_decisions': total,
            'expert_alignment_rate': expert_matches / total if total else 0,
            'safety_violations': violations,
            'thermal_events': thermal_events,
            'decision_sources': decision_sources,
            'action_distribution': action_counts,
            'temperature_zones': temp_zones,
            'total_power_consumed': sum(self.power_history),
            'avg_power_per_decision': sum(self.power_history) / total if total else 0
        }
