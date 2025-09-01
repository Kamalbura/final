#!/usr/bin/env python3
"""
UAV DDoS Environment with thermal and power modeling
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AlgorithmProfile:
    """Algorithm performance characteristics"""
    cpu_avg: float
    cpu_variance: float
    power_factor: float
    thermal_coefficient: float
    effectiveness: float
    memory_impact: float
    cooling_factor: float

class ThermalSimulator:
    """High-fidelity thermal simulation for UAV system"""
    
    def __init__(self):
        # Thermal characteristics
        self.thermal_mass = 8.2  # J/°C
        self.thermal_resistance = 12.5  # °C/W
        self.ambient_temp = 25.0  # °C
        self.current_temp = 50.0  # °C starting temperature
        self.temp_history = []
        
        # Algorithm profiles with cooling factors
        self.algorithms = {
            'No_DDoS': AlgorithmProfile(8, 5, 1.0, 0.02, 0.0, 0.0, 1.5),    # Fastest cooling
            'XGBoost': AlgorithmProfile(35, 15, 1.4, 0.12, 0.85, 2.0, 1.0),  # Standard cooling
            'TST': AlgorithmProfile(85, 10, 2.1, 0.25, 0.95, 14.0, 0.5),     # Slowest cooling
        }
        
        # Thermal time constants
        self.heating_rate = 0.15  # °C/second at high load
        self.cooling_rate = 0.07  # °C/second at idle
        self.thermal_lag = 10     # seconds thermal time constant
        
        # Track algorithm transitions for cooling effects
        self.previous_algorithm = 'No_DDoS'
        self.algorithm_change_time = time.time()
        self.tst_recovery_time = 240  # seconds to recover from TST (measured)
        
        # Temperature setpoint (equilibrium temp for current algorithm)
        self.temp_setpoint = 50.0
    
    def update_temperature(self, algorithm: str, dt: float = 1.0) -> Tuple[float, float]:
        """Update temperature based on current algorithm with realistic cooling"""
        profile = self.algorithms.get(algorithm, self.algorithms['No_DDoS'])
        
        # Check for algorithm transition
        current_time = time.time()
        if algorithm != self.previous_algorithm:
            # Record algorithm change time for thermal lag calculations
            self.algorithm_change_time = current_time
            
            # If coming from TST, mark this for special cooling behavior
            if self.previous_algorithm == 'TST':
                logging.debug(f"Detected transition from TST to {algorithm}, starting cooling cycle")
            
            self.previous_algorithm = algorithm
        
        # Calculate time since algorithm change to model thermal lag
        time_since_change = current_time - self.algorithm_change_time
        transition_factor = min(1.0, time_since_change / self.thermal_lag)
        
        # Heat generation (algorithm-specific)
        cpu_usage = np.random.normal(profile.cpu_avg, profile.cpu_variance)
        cpu_usage = np.clip(cpu_usage, 0, 100)  # Clamp to valid range
        
        # Calculate temperature setpoint for current algorithm
        # This is the equilibrium temperature this algorithm would reach if run indefinitely
        algo_setpoint = self.ambient_temp + (cpu_usage * profile.thermal_coefficient * self.thermal_resistance)
        
        # Gradual transition of setpoint based on thermal lag
        if time_since_change < self.thermal_lag:
            # During transition period, blend old and new setpoints
            old_setpoint = self.temp_setpoint
            self.temp_setpoint = old_setpoint * (1 - transition_factor) + algo_setpoint * transition_factor
        else:
            # After transition period, use actual algorithm setpoint
            self.temp_setpoint = algo_setpoint
        
        # Apply special cooling behavior for TST → XGBoost transition (240s recovery)
        tst_transition_cooldown = False
        if self.previous_algorithm == 'TST' and time_since_change < self.tst_recovery_time:
            # Slow recovery from TST thermal buildup
            tst_transition_cooldown = True
            recovery_progress = time_since_change / self.tst_recovery_time
            # Gradual cooling from TST temperature to algorithm setpoint
            cooling_setpoint = self.temp_setpoint + (75.0 - self.temp_setpoint) * (1.0 - recovery_progress)
            self.temp_setpoint = cooling_setpoint
        
        # Newton's law of cooling (heat transfer proportional to temperature difference)
        # Direction and rate depends on whether we're heating up or cooling down
        if self.current_temp < self.temp_setpoint:
            # Heating up - use algorithm's thermal coefficient
            rate = self.heating_rate
        else:
            # Cooling down - use algorithm's cooling factor
            rate = self.cooling_rate * profile.cooling_factor
            
            # Special case: much slower cooling when coming down from TST
            if tst_transition_cooldown:
                # TST has severe thermal inertia, cooling is slower than normal
                rate *= 0.5  # 50% slower cooling during TST recovery period
        
        # Temperature change based on difference from setpoint and rate
        temp_diff = self.temp_setpoint - self.current_temp
        temp_change = temp_diff * rate * dt
        
        # Apply temperature change with some noise
        self.current_temp += temp_change + np.random.normal(0, 0.02)  # Small random fluctuations
        self.temp_history.append(self.current_temp)
        
        # Keep history manageable
        if len(self.temp_history) > 1000:
            self.temp_history = self.temp_history[-500:]
            
        return self.current_temp, cpu_usage
    
    def get_temperature(self) -> float:
        """Get current temperature"""
        return self.current_temp
    
    def get_temperature_category(self) -> str:
        """Get temperature category based on current temperature"""
        if self.current_temp <= 55:
            return "Safe"
        elif self.current_temp <= 70:
            return "Warning"
        else:
            return "Critical"
    
    def reset(self, start_temp: float = 50.0):
        """Reset thermal state"""
        self.current_temp = start_temp
        self.temp_history = []
        self.previous_algorithm = 'No_DDoS'
        self.algorithm_change_time = time.time()
        self.temp_setpoint = 50.0

class PowerMonitor:
    """Accurate power consumption tracking"""
    
    def __init__(self):
        self.voltage = 5.1  # RPi typical voltage
        self.baseline_current = 0.7  # Amps at idle
        self.current_consumption = self.baseline_current
        self.power_history = []
        
    def calculate_power(self, algorithm: str, thermal_sim: ThermalSimulator) -> float:
        """Calculate real-time power consumption"""
        profile = thermal_sim.algorithms.get(algorithm, thermal_sim.algorithms['No_DDoS'])
        
        # Current consumption based on algorithm and temperature
        base_current = self.baseline_current * profile.power_factor
        thermal_factor = 1.0 + (thermal_sim.current_temp - 50) * 0.01  # 1% per degree
        
        self.current_consumption = base_current * thermal_factor
        power = self.voltage * self.current_consumption
        
        self.power_history.append(power)
        
        return power
    
    def reset(self):
        """Reset power monitor"""
        self.current_consumption = self.baseline_current
        self.power_history = []

class UAVDDoSEnvironment:
    """UAV DDoS training environment with enhanced thermal and power modeling"""
    
    def __init__(self):
        # State space definitions
        self.battery_levels = ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"]
        self.threat_states = ["Normal", "Confirming", "Confirmed"]
        self.action_labels = ["No_DDoS", "XGBoost", "TST"]
        
        # Create thermal simulator and power monitor
        self.thermal_simulator = ThermalSimulator()
        self.power_monitor = PowerMonitor()
        
        # Expert lookup table
        self.expert_lookup = self._create_expert_lookup_table()
        
        # Current state
        self.current_state = {}
        self.last_tst_time = -1000  # Time since last TST use (for recovery)
        self.time_step = 0
        self.total_power_consumed = 0
        self.safety_violations = 0
        
        # Reset environment
        self.reset()
        
        logger.info(f"UAV DDoS Environment initialized with expert lookup table")
    
    def _create_expert_lookup_table(self):
        """Create expert lookup table using simplified rules:
        - If temperature is Critical OR battery in 0-20% => No_DDoS (0)
        - Else if threat state is Confirming => TST (2)
        - Else => XGBoost (1)
        """
        lookup = {}
        for temp_category in ["Safe", "Warning", "Critical"]:
            for battery_level in self.battery_levels:
                for threat_state in self.threat_states:
                    if temp_category == "Critical" or battery_level == "0-20%":
                        action = 0
                    elif threat_state == "Confirming":
                        action = 2
                    else:
                        action = 1
                    lookup[(battery_level, temp_category, threat_state)] = action
        return lookup
    
    def reset(self):
        """Reset the environment to initial state"""
        # Reset thermal and power simulators
        self.thermal_simulator.reset()
        self.power_monitor.reset()
        
        # Initial state
        self.current_state = {
            'temperature': self.thermal_simulator.get_temperature(),
            'battery': 90.0,  # Start with 90% battery
            'threat': 0,      # Start with Normal threat
            'time_since_tst': 1000  # Start with no recent TST usage
        }
        
        self.time_step = 0
        self.last_tst_time = -1000
        self.total_power_consumed = 0
        self.safety_violations = 0
        
        return self.current_state.copy()
    
    def step(self, action: int):
        """Take a step in the environment"""
        # Check safety
        safe, reason = self.is_safe_action(self.current_state, action)
        if not safe:
            logger.warning(f"Safety violation: {reason}")
            self.safety_violations += 1
        
        # Map action to algorithm name
        algorithm = self.action_labels[action]
        
        # Update thermal simulation
        self.thermal_simulator.update_temperature(algorithm)
        
        # Calculate power consumption
        power = self.power_monitor.calculate_power(algorithm, self.thermal_simulator)
        self.total_power_consumed += power
        
        # Track TST usage for recovery periods
        if action == 2:  # TST
            self.last_tst_time = self.time_step
        
        # Update battery level based on power consumption
        self.current_state['battery'] -= power * 0.01  # 0.01% drain per watt per step
        self.current_state['battery'] = max(0, min(100, self.current_state['battery']))  # Clamp
        
        # Update temperature
        self.current_state['temperature'] = self.thermal_simulator.get_temperature()
        
        # Update time since last TST usage
        self.current_state['time_since_tst'] = self.time_step - self.last_tst_time
        
        # Random threat state transitions (could make this more sophisticated)
        if np.random.random() < 0.05:  # 5% chance to change
            # Threat state transitions depend on current algorithm effectiveness
            if self.current_state['threat'] == 0:  # Normal
                # More likely to detect threat with better algorithms
                transition_probs = [0.8, 0.2, 0] if action == 0 else [0.6, 0.4, 0] if action == 1 else [0.4, 0.6, 0]
            elif self.current_state['threat'] == 1:  # Confirming
                # More likely to confirm with better algorithms
                transition_probs = [0.2, 0.7, 0.1] if action == 0 else [0.1, 0.6, 0.3] if action == 1 else [0.05, 0.45, 0.5]
            else:  # Confirmed
                # More likely to resolve with better algorithms
                transition_probs = [0.1, 0.2, 0.7] if action == 0 else [0.2, 0.3, 0.5] if action == 1 else [0.3, 0.4, 0.3]
            
            self.current_state['threat'] = np.random.choice([0, 1, 2], p=transition_probs)
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Increment time step
        self.time_step += 1
        
        # Check if episode is done (battery depleted or thermal emergency)
        done = self.current_state['battery'] <= 10 or self.current_state['temperature'] >= 85
        
        return self.current_state.copy(), reward, done
    
    def _calculate_reward(self, action: int):
        """Calculate reward for the current state-action pair"""
        reward = 0
        
        # Get current state
        temp = self.current_state['temperature']
        battery = self.current_state['battery']
        threat = self.current_state['threat']
        
        # 1. Power efficiency (30% weight)
        # Power penalties based on algorithm
        power_values = [3.0, 5.5, 9.0]  # Power for No_DDoS, XGBoost, TST
        power = power_values[action]
        power_reward = (8.0 - power) / 8.0  # Higher reward for lower power
        reward += 0.3 * power_reward
        
        # 2. Thermal management (25% weight)
        if temp > 80:
            thermal_reward = -10  # Severe penalty for critical temperature
        elif temp > 70:
            thermal_reward = -2   # Moderate penalty for high temperature
        else:
            thermal_reward = 0.5  # Bonus for normal temperature
        reward += 0.25 * thermal_reward
        
        # 3. Threat effectiveness (25% weight)
        # Higher reward for appropriate algorithm based on threat level
        effectiveness = [0.2, 0.7, 0.9]  # Effectiveness of each algorithm
        if threat == 0:  # Normal - no need for heavy algorithms
            threat_reward = 1.0 if action == 0 else 0.5 if action == 1 else -0.5
        elif threat == 1:  # Confirming - need good detection
            threat_reward = -0.5 if action == 0 else 0.8 if action == 1 else 1.0
        else:  # Confirmed - need balanced approach
            threat_reward = -1.0 if action == 0 else 1.0 if action == 1 else 0.5
        reward += 0.25 * threat_reward
        
        # 4. Expert agreement (20% weight)
        # Get categorical state for lookup
        battery_category = self._get_battery_category(battery)
        temp_category = self.thermal_simulator.get_temperature_category()
        threat_category = self.threat_states[threat] if isinstance(threat, int) else threat
        
        # Get expert action
        expert_state = (battery_category, temp_category, threat_category)
        expert_action = self.expert_lookup.get(expert_state, 0)
        expert_reward = 1.0 if action == expert_action else -0.5
        reward += 0.2 * expert_reward
        
        # 5. Safety penalties (override)
        if temp > 80 and action != 0:
            reward -= 15  # Severe penalty for using algorithms at critical temperature
        if battery < 20 and action != 0:
            reward -= 10  # Severe penalty for using algorithms with low battery
        
        return reward
    
    def get_expert_action(self, state: Dict) -> int:
        """Get expert action for a given state"""
        # Convert continuous state to categorical for lookup
        temp = state.get('temperature', 50)
        battery = state.get('battery', 80)
        threat = state.get('threat', 0)
        
        # Get temperature category
        if isinstance(temp, str):
            temp_category = temp  # Already categorical
        else:
            if temp <= 55:
                temp_category = "Safe"
            elif temp <= 70:
                temp_category = "Warning"
            else:
                temp_category = "Critical"
        
        # Get battery category
        battery_category = self._get_battery_category(battery)
        
        # Get threat category
        if isinstance(threat, int):
            threat_category = self.threat_states[min(threat, 2)]
        else:
            threat_category = threat
        
        # Look up expert action
        state_key = (battery_category, temp_category, threat_category)
        return self.expert_lookup.get(state_key, 0)  # Default to No_DDoS if not found
    
    def is_safe_action(self, state: Dict, action: int) -> Tuple[bool, str]:
        """Check if action is safe given current state"""
        temp = state.get('temperature', 50)
        battery = state.get('battery', 80)
        time_since_tst = state.get('time_since_tst', 1000)
        
        # Convert temp to numerical if it's categorical
        if isinstance(temp, str):
            if temp == "Critical":
                temp = 75  # Approximate critical temperature
            elif temp == "Warning":
                temp = 65  # Approximate warning temperature
            else:  # Safe
                temp = 50  # Approximate safe temperature
        
        # Convert battery to numerical if it's categorical
        if isinstance(battery, str):
            try:
                if "-" in battery:
                    # Take the lower end of the range for safety
                    battery = float(battery.split("-")[0])
                else:
                    battery = float(battery.strip("%"))
            except (ValueError, AttributeError):
                # Default to middle of range if parsing fails
                battery = 50.0
        
        # Safety constraints
        if temp >= 80 and action != 0:
            return False, f"Critical temperature ({temp:.1f}°C): must use No_DDoS"
        
        if battery <= 20 and action != 0:
            return False, f"Critical battery ({battery:.1f}%): must use No_DDoS"
        
        if action == 2:  # TST specific constraints
            if temp >= 70:
                return False, f"TST forbidden above 70°C (current: {temp:.1f}°C)"
            
            if battery <= 40:
                return False, f"TST forbidden below 40% battery (current: {battery:.1f}%)"
            
            if time_since_tst < 240:
                return False, f"TST needs recovery time (only {time_since_tst} steps since last use, need 240)"
        
        return True, "Safe"
    
    def _get_battery_category(self, battery: float) -> str:
        """Convert continuous battery level to categorical"""
        # Handle case where battery is already a string category
        if isinstance(battery, str):
            # If it's already a category string, return it if valid
            if battery in self.battery_levels:
                return battery
            # Otherwise try to extract numeric value if it has one
            try:
                # Try to extract number from string like "41-60%"
                if "-" in battery:
                    # Use the average of the range
                    start, end = battery.strip("%").split("-")
                    battery = (float(start) + float(end)) / 2
                else:
                    # Try to convert any numeric string
                    battery = float(battery.strip("%"))
            except (ValueError, AttributeError):
                # Default to middle category if we can't parse
                return "41-60%"
        
        # Now handle numeric battery value
        if battery <= 20:
            return "0-20%"
        elif battery <= 40:
            return "21-40%"
        elif battery <= 60:
            return "41-60%"
        elif battery <= 80:
            return "61-80%"
        else:
            return "81-100%"
