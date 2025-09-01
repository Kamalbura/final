#!/usr/bin/env python3
"""
Integration example showing how to use the UAV DDoS-RL agent
in a production environment with real hardware sensors
"""

import time
import json
import logging
import os
import argparse
import signal
import sys
from threading import Event

# Import our modules
from utils.logger import UAVDDoSLogger
from hardware.sensor_interface import HardwareSensorInterface

class UAVDDoSRLIntegration:
    """Integration of UAV DDoS-RL agent with hardware and external systems"""
    
    def __init__(self, model_path='trained_uav_ddos_model.json', log_dir='logs'):
        # Setup logging
        self.logger = UAVDDoSLogger(log_dir=log_dir)
        logging.info("Starting UAV DDoS-RL Integration")
        
        # Load the trained model
        self.load_model(model_path)
        
        # Initialize hardware interface
        self.hardware = HardwareSensorInterface(update_interval=1.0)
        
        # Current state tracking
        self.current_state = {
            'battery': "81-100%",
            'temperature': "Safe",
            'threat': "Normal",
            'time_since_change': 0
        }
        
        self.current_action = 0  # Default: No DDoS
        self.last_action_time = time.time()
        self.threat_change_time = time.time()
        
        # Control flags
        self.stop_event = Event()
        
        # Algorithm process handles
        self.current_algorithm_process = None
    
    def load_model(self, model_path):
        """Load the trained RL model"""
        try:
            with open(model_path, 'r') as f:
                model_data = json.load(f)
            
            self.q_table = model_data['q_table']
            self.expert_lookup = model_data['expert_lookup']
            logging.info(f"Model loaded from {model_path}")
            
            # Check model shape
            num_states = len(self.q_table)
            num_actions = len(self.q_table[0])
            logging.info(f"Model contains {num_states} states and {num_actions} actions")
            
            return True
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            return False
    
    def start(self):
        """Start the integration"""
        # Start hardware monitoring
        self.hardware.start()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Main loop
        try:
            self.run_loop()
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop all processes and clean up"""
        self.stop_event.set()
        
        # Stop hardware monitoring
        self.hardware.stop()
        
        # Stop any running algorithms
        self.stop_current_algorithm()
        
        # Save decision log
        self.logger.save_decision_log()
        
        logging.info("UAV DDoS-RL Integration stopped")
    
    def signal_handler(self, signum, frame):
        """Handle termination signals"""
        logging.info(f"Received signal {signum}, stopping...")
        self.stop()
        sys.exit(0)
    
    def run_loop(self):
        """Main monitoring and decision loop"""
        logging.info("Starting main loop")
        
        while not self.stop_event.is_set():
            # Get current hardware state
            hw_state = self.hardware.get_state_data()
            
            # Update state with hardware data
            self.update_state(hw_state)
            
            # Check for threat changes that would trigger a decision
            if self.detect_threat_change():
                # Make a new decision
                self.make_decision()
            
            # Sleep briefly
            time.sleep(0.5)
    
    def update_state(self, hw_state):
        """Update the current state with hardware data"""
        self.current_state['battery'] = hw_state['battery']
        self.current_state['temperature'] = hw_state['temperature']
        
        # Update time since last algorithm change
        current_time = time.time()
        self.current_state['time_since_change'] = current_time - self.last_action_time
    
    def detect_threat_change(self):
        """
        Detect if there's been a threat state change
        In a real system, this would come from network monitoring
        """
        # Simulate threat changes for demonstration
        # In a real system, this would be input from network monitoring
        
        # For demo purposes, generate some simulated threat changes
        current_time = time.time()
        
        # Generate a threat change every 60 seconds for demonstration
        if current_time - self.threat_change_time > 60:
            # Cycle through threat states: Normal -> Confirming -> Confirmed -> Normal
            if self.current_state['threat'] == "Normal":
                new_threat = "Confirming"
            elif self.current_state['threat'] == "Confirming":
                new_threat = "Confirmed"
            else:
                new_threat = "Normal"
            
            # Log the change
            logging.info(f"Threat state changed: {self.current_state['threat']} -> {new_threat}")
            
            # Update state
            self.current_state['threat'] = new_threat
            self.threat_change_time = current_time
            
            return True
        
        return False
    
    def make_decision(self):
        """Make a decision based on current state using the RL model"""
        # Convert state to index for Q-table lookup
        state_idx = self.get_state_index()
        
        # Get action from Q-table
        action = self.get_action(state_idx)
        
        # Get expert action for comparison
        expert_action = self.get_expert_action()
        
        # Log the decision
        power_used = [3.0, 5.5, 9.0][action]  # Power for each action type
        self.logger.log_decision(
            state=self.current_state.copy(),
            action=action,
            expert_action=expert_action,
            reward=0,  # No reward in production
            power_used=power_used
        )
        
        # Execute the decision
        self.execute_action(action)
        
        # Update last action time
        self.last_action_time = time.time()
        self.current_action = action
    
    def get_state_index(self):
        """Convert current state to Q-table index"""
        # Define the state space
        battery_levels = ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"]
        temperatures = ["Safe", "Warning", "Critical"]
        threat_states = ["Normal", "Confirming", "Confirmed"]
        
        # Get indices
        battery_idx = battery_levels.index(self.current_state['battery'])
        temp_idx = temperatures.index(self.current_state['temperature'])
        threat_idx = threat_states.index(self.current_state['threat'])
        
        # Compute state index
        return battery_idx * 9 + temp_idx * 3 + threat_idx
    
    def get_action(self, state_idx):
        """Get action from Q-table with safety checks"""
        # Default action is No DDoS (safest)
        default_action = 0
        
        try:
            # Get action from Q-table
            action = self.q_table[state_idx].index(max(self.q_table[state_idx]))
            
            # Safety check: Never run algorithms with critical battery or temperature
            if (self.current_state['battery'] == "0-20%" or 
                self.current_state['temperature'] == "Critical") and action != 0:
                logging.warning("Safety override: Critical conditions detected, forcing No DDoS")
                return default_action
                
            return action
        except Exception as e:
            logging.error(f"Error getting action: {e}")
            return default_action
    
    def get_expert_action(self):
        """Get the expert action for the current state"""
        state_key = (
            self.current_state['battery'], 
            self.current_state['temperature'], 
            self.current_state['threat']
        )
        
        # Try to get from expert lookup, default to 0 (No DDoS) for safety
        return self.expert_lookup.get(state_key, 0)
    
    def execute_action(self, action):
        """Execute the selected action by controlling the DDoS detection algorithms"""
        action_names = ["No_DDoS", "XGBoost", "TST"]
        logging.info(f"Executing action: {action_names[action]}")
        
        # Stop any currently running algorithm
        self.stop_current_algorithm()
        
        # Start the new algorithm if it's not No DDoS
        if action > 0:
            self.start_algorithm(action)
    
    def stop_current_algorithm(self):
        """Stop any currently running algorithm"""
        if self.current_algorithm_process is not None:
            try:
                # Send termination signal
                self.current_algorithm_process.terminate()
                # Wait for it to exit
                self.current_algorithm_process.wait(timeout=2.0)
                logging.info("Stopped running algorithm")
            except Exception as e:
                logging.error(f"Error stopping algorithm: {e}")
            finally:
                self.current_algorithm_process = None
    
    def start_algorithm(self, action):
        """Start a DDoS detection algorithm based on the selected action"""
        try:
            # In a real implementation, this would execute the actual algorithm
            # Here we simulate with a log message
            if action == 1:  # XGBoost
                logging.info("Started XGBoost DDoS detection algorithm")
                # self.current_algorithm_process = subprocess.Popen(['/path/to/xgboost_algorithm.py'])
            elif action == 2:  # TST
                logging.info("Started TST DDoS detection algorithm")
                # self.current_algorithm_process = subprocess.Popen(['/path/to/tst_algorithm.py'])
        except Exception as e:
            logging.error(f"Error starting algorithm: {e}")

# If run directly
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UAV DDoS-RL Integration')
    parser.add_argument('--model', type=str, default='trained_uav_ddos_model.json',
                        help='Path to the trained model file')
    parser.add_argument('--logs', type=str, default='logs',
                        help='Directory for log files')
    
    args = parser.parse_args()
    
    # Create and start the integration
    integration = UAVDDoSRLIntegration(
        model_path=args.model,
        log_dir=args.logs
    )
    
    integration.start()
