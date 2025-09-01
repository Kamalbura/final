#!/usr/bin/env python3
"""
Deployment script for the Lookup Table-Based UAV DDoS Defense System
"""

import os
import sys
import numpy as np
import json
import time
import logging
from datetime import datetime
import argparse

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import our modules
from src.agents.lookup_table_agent import LookupTableAgent

def setup_logging(log_dir="logs"):
    """Setup logging for the deployment"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = f"uav_ddos_defense_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("UAV-DDoS-Defense")
    logger.info(f"Logging initialized. Log file: {log_path}")
    return logger

def load_model(model_path):
    """Load a trained model"""
    logger = logging.getLogger("UAV-DDoS-Defense")
    
    try:
        agent = LookupTableAgent(learning_enabled=False)
        agent.load_model(model_path)
        logger.info(f"Model successfully loaded from {model_path}")
        return agent
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def make_prediction(agent, state):
    """Make a prediction for the given state"""
    logger = logging.getLogger("UAV-DDoS-Defense")
    
    try:
        # Make decision
        action = agent.make_decision(state)
        action_label = agent.action_labels[action]
        
        # Get expert action for comparison
        expert_state = agent._discretize_state(state)
        expert_action = agent.lookup_table.get(expert_state, 0)
        expert_aligned = (action == expert_action)
        
        # Log decision
        logger.info(f"Decision: {action_label} for state {state}")
        if not expert_aligned:
            logger.warning(f"Decision differs from expert recommendation: {agent.action_labels[expert_action]}")
        
        # Return result
        return {
            'action': action,
            'action_label': action_label,
            'expert_aligned': expert_aligned,
            'expert_action': expert_action,
            'expert_label': agent.action_labels[expert_action]
        }
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        # Default to safest action
        return {
            'action': 0,
            'action_label': 'No_DDoS',
            'error': str(e)
        }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='UAV DDoS Defense System Deployment')
    parser.add_argument('--model', '-m', type=str, default='models/lookup_table_expert.json',
                        help='Path to the model file')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Run in interactive mode')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting UAV DDoS Defense System")
    
    # Load model
    try:
        agent = load_model(args.model)
    except Exception as e:
        logger.error(f"Exiting due to model loading error: {e}")
        return
    
    logger.info(f"Model loaded successfully. Ready for predictions.")
    
    # Interactive mode
    if args.interactive:
        logger.info("Starting interactive mode. Type 'exit' to quit.")
        
        while True:
            try:
                print("\n=== UAV DDoS Defense System ===")
                print("Enter state values (or 'exit' to quit):")
                
                # Get battery level
                battery_input = input("Battery level (0-100 or category like '41-60%'): ")
                if battery_input.lower() == 'exit':
                    break
                
                # Parse battery
                if '%' in battery_input:
                    battery = battery_input
                else:
                    try:
                        battery = float(battery_input)
                    except ValueError:
                        print("Invalid battery value. Please enter a number (0-100) or category.")
                        continue
                
                # Get temperature
                temp_input = input("Temperature (numeric or 'Safe', 'Warning', 'Critical'): ")
                if temp_input.lower() == 'exit':
                    break
                
                # Parse temperature
                if temp_input in ['Safe', 'Warning', 'Critical']:
                    temperature = temp_input
                else:
                    try:
                        temperature = float(temp_input)
                    except ValueError:
                        print("Invalid temperature value. Please enter a number or category.")
                        continue
                
                # Get threat state
                threat_input = input("Threat (0-2, 'Normal', 'Confirming', or 'Confirmed'): ")
                if threat_input.lower() == 'exit':
                    break
                
                # Parse threat
                if threat_input in ['Normal', 'Confirming', 'Confirmed']:
                    threat = threat_input
                else:
                    try:
                        threat = int(threat_input)
                    except ValueError:
                        print("Invalid threat value. Please enter a number (0-2) or category.")
                        continue
                
                # Create state
                state = {
                    'battery': battery,
                    'temperature': temperature,
                    'threat': threat
                }
                
                # Make prediction
                result = make_prediction(agent, state)
                
                # Display result
                print("\n=== Decision ===")
                print(f"Action: {result['action_label']}")
                print(f"Expert recommendation: {result.get('expert_label', 'N/A')}")
                print(f"Expert aligned: {result.get('expert_aligned', 'N/A')}")
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, exiting...")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"Error: {e}")
    else:
        # Non-interactive mode - run predefined test cases
        logger.info("Running test scenarios...")
        
        test_scenarios = [
            {'battery': 90, 'temperature': 50, 'threat': 'Normal'},
            {'battery': 75, 'temperature': 'Warning', 'threat': 'Confirming'},
            {'battery': 15, 'temperature': 50, 'threat': 'Confirmed'},
            {'battery': '61-80%', 'temperature': 'Critical', 'threat': 1},
            {'battery': '21-40%', 'temperature': 'Safe', 'threat': 0}
        ]
        
        for i, scenario in enumerate(test_scenarios):
            print(f"\nScenario {i+1}: {scenario}")
            result = make_prediction(agent, scenario)
            print(f"Decision: {result['action_label']}")
            if 'expert_label' in result:
                print(f"Expert recommendation: {result['expert_label']}")
                print(f"Expert aligned: {result['expert_aligned']}")
    
    logger.info("UAV DDoS Defense System shutdown complete")

if __name__ == "__main__":
    main()
