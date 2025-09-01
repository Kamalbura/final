#!/usr/bin/env python3
"""
Real-time monitoring system for UAV DDoS-RL agent
Tracks power consumption, temperature, and decision quality
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import queue
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("monitoring.log"),
        logging.StreamHandler()
    ]
)

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
    warmup_time: float  # Time to reach peak temp (seconds)

class ThermalSimulator:
    """High-fidelity thermal simulation for Raspberry Pi UAV system"""
    
    def __init__(self):
        # Thermal characteristics (from real data analysis)
        self.thermal_mass = 8.2  # J/°C
        self.thermal_resistance = 12.5  # °C/W
        self.ambient_temp = 25.0  # °C
        self.current_temp = 50.0  # °C starting temperature
        self.temp_history = []
        
        # Algorithm profiles with measured real-world thermal behavior
        self.algorithms = {
            'No_DDoS': AlgorithmProfile(
                cpu_avg=8,
                cpu_variance=5, 
                power_factor=1.0, 
                thermal_coefficient=0.02,
                effectiveness=0.0, 
                memory_impact=0.0, 
                cooling_factor=1.8,  # Fastest cooling
                warmup_time=30       # Quick stabilization
            ),
            'XGBoost': AlgorithmProfile(
                cpu_avg=35,
                cpu_variance=15, 
                power_factor=1.4, 
                thermal_coefficient=0.10,  # Reduced from 0.12
                effectiveness=0.85, 
                memory_impact=2.0, 
                cooling_factor=1.0,   # Standard cooling
                warmup_time=60        # Moderate warmup
            ),
            'TST': AlgorithmProfile(
                cpu_avg=85,           # High CPU usage
                cpu_variance=10, 
                power_factor=2.1, 
                thermal_coefficient=0.20,  # Adjusted for 2-minute warmup
                effectiveness=0.95, 
                memory_impact=14.0, 
                cooling_factor=0.5,    # Slower cooling
                warmup_time=120        # Takes 2 minutes to reach peak temp
            ),
        }
        
        # Thermal time constants calibrated to observed behavior
        self.heating_rate = 0.15      # Slightly increased for faster initial warmup
        self.cooling_rate = 0.07      # Adjusted for 30s cooldown
        self.thermal_lag = 10         # Reduced thermal lag for quicker response
        
        # Track algorithm transitions for cooling effects
        self.previous_algorithm = 'No_DDoS'
        self.algorithm_change_time = time.time()
        self.tst_recovery_time = 30   # 30 seconds to cool down from TST (observed value)
        
        # Temperature setpoint and tracking
        self.temp_setpoint = 50.0
        self.time_at_algorithm_start = time.time()
        self.algorithm_runtime = 0.0
        
    def update_temperature(self, algorithm_idx: int, dt: float = 1.0) -> Tuple[float, float]:
        """Update temperature based on current algorithm with realistic thermal behavior"""
        # Map algorithm index to name
        algorithm_names = ['No_DDoS', 'XGBoost', 'TST']
        algorithm = algorithm_names[algorithm_idx] if 0 <= algorithm_idx < 3 else 'No_DDoS'
        
        profile = self.algorithms.get(algorithm, self.algorithms['No_DDoS'])
        current_time = time.time()
        
        # Check for algorithm transition
        if algorithm != self.previous_algorithm:
            # Record algorithm change time for thermal lag calculations
            self.algorithm_change_time = current_time
            self.time_at_algorithm_start = current_time
            
            # If coming from TST, mark this for special cooling behavior
            if self.previous_algorithm == 'TST':
                logging.info(f"Detected transition from TST to {algorithm}, starting cooling cycle (30s)")
            
            self.previous_algorithm = algorithm
            self.algorithm_runtime = 0.0
        else:
            # Update runtime on current algorithm
            self.algorithm_runtime = current_time - self.time_at_algorithm_start
        
        # Calculate time since algorithm change for cooling behavior
        time_since_change = current_time - self.algorithm_change_time
        
        # Generate realistic CPU usage based on algorithm profile
        cpu_usage = np.random.normal(profile.cpu_avg, profile.cpu_variance)
        cpu_usage = np.clip(cpu_usage, 0, 100)  # Clamp to valid range
        
        # Apply warmup scaling factor - TST especially takes time to reach peak temperature
        warmup_factor = min(1.0, self.algorithm_runtime / profile.warmup_time)
        
        # For TST, follow observed heating curve (slower start, then accelerating)
        if algorithm == 'TST':
            # Non-linear warmup - slow at first, then exponential increase
            if warmup_factor < 0.5:
                # First half of warmup period: slow increase
                warmup_factor = warmup_factor * 0.8
            else:
                # Second half: accelerating increase
                warmup_factor = 0.4 + (warmup_factor - 0.5) * 1.2
        
        # Calculate target temperature for current algorithm at current runtime
        if algorithm == 'TST':
            # TST peaks at around 72-75°C after 120 seconds
            max_algo_temp = 75.0
        elif algorithm == 'XGBoost':
            # XGBoost stabilizes around 65°C
            max_algo_temp = 65.0
        else:
            # No_DDoS stays cooler
            max_algo_temp = 55.0
            
        # Calculate temperature setpoint based on warmup progression
        base_temp = 50.0  # Starting temperature
        algo_setpoint = base_temp + (max_algo_temp - base_temp) * warmup_factor
        
        # Special case for cooling from TST
        cooling_from_tst = False
        if self.previous_algorithm == 'TST' and time_since_change < self.tst_recovery_time:
            cooling_from_tst = True
            # Calculate cooling progress (0 = just switched, 1 = fully cooled)
            cooling_progress = time_since_change / self.tst_recovery_time
            
            # Start from the high TST temperature and cool toward algorithm setpoint
            # Use exponential cooling curve for more realistic behavior
            cooling_factor = 1.0 - np.exp(-3 * cooling_progress)  # Faster initial cooling
            high_temp = max(75.0, self.current_temp)  # Use actual temp if higher
            
            # Blend between high temp and target setpoint based on cooling progress
            algo_setpoint = high_temp - (high_temp - algo_setpoint) * cooling_factor
        
        # Apply temperature change using Newton's law of cooling/heating
        temp_diff = algo_setpoint - self.current_temp
        
        if temp_diff > 0:  # Heating up
            # Heating rate is influenced by algorithm's thermal coefficient
            effective_rate = self.heating_rate * (1.0 + profile.thermal_coefficient)
        else:  # Cooling down
            # Cooling affected by cooling factor of current algorithm
            effective_rate = self.cooling_rate * profile.cooling_factor
            
            # Faster cooling immediately after switching from TST
            if cooling_from_tst and time_since_change < 5.0:
                effective_rate *= 1.5  # Initial faster cooling
        
        # Apply temperature change with realistic physical model
        temp_change = temp_diff * effective_rate * dt
        
        # Add minor noise for realism
        temp_noise = np.random.normal(0, 0.05)
        self.current_temp += temp_change + temp_noise
        self.temp_history.append(self.current_temp)
        
        # Keep history manageable
        if len(self.temp_history) > 1000:
            self.temp_history = self.temp_history[-500:]
            
        return self.current_temp, cpu_usage
    
    def get_thermal_trend(self) -> float:
        """Calculate temperature derivative"""
        if len(self.temp_history) < 2:
            return 0.0
        return self.temp_history[-1] - self.temp_history[-2]
    
    def get_temperature_category(self) -> str:
        """Get temperature category based on current temperature"""
        if self.current_temp <= 55:
            return "Safe"
        elif self.current_temp <= 70:
            return "Warning"
        else:
            return "Critical"
    
    def is_thermal_emergency(self) -> bool:
        """Check if thermal emergency conditions exist"""
        return self.current_temp > 75.0  # Critical threshold
    
    def reset(self, start_temp: float = 50.0):
        """Reset thermal state"""
        self.current_temp = start_temp
        self.temp_history = []

class PowerMonitor:
    """Accurate power consumption tracking using V×I×Δt method"""
    
    def __init__(self):
        self.voltage = 5.1  # RPi typical voltage
        self.baseline_current = 0.7  # Amps at idle
        self.current_consumption = self.baseline_current
        self.power_history = []
        self.cumulative_energy = 0.0
        
    def calculate_power(self, algorithm_idx: int, thermal_sim: ThermalSimulator) -> float:
        """Calculate real-time power consumption"""
        # Map algorithm index to name
        algorithm_names = ['No_DDoS', 'XGBoost', 'TST']
        algorithm = algorithm_names[algorithm_idx] if 0 <= algorithm_idx < 3 else 'No_DDoS'
        
        profile = thermal_sim.algorithms.get(algorithm, thermal_sim.algorithms['No_DDoS'])
        
        # Current consumption based on algorithm and temperature
        base_current = self.baseline_current * profile.power_factor
        thermal_factor = 1.0 + (thermal_sim.current_temp - 50) * 0.01  # 1% per degree
        
        self.current_consumption = base_current * thermal_factor
        power = self.voltage * self.current_consumption
        
        self.power_history.append(power)
        self.cumulative_energy += power  # Accumulate energy over time
        
        return power
    
    def get_efficiency_metric(self, algorithm_effectiveness: float) -> float:
        """Calculate performance per watt"""
        current_power = self.power_history[-1] if self.power_history else 3.57
        return algorithm_effectiveness / current_power if current_power > 0 else 0

class MetricsCollector:
    """Collects real-time metrics from the UAV system"""
    
    def __init__(self, update_interval=1.0):
        self.update_interval = update_interval
        self.stop_event = threading.Event()
        self.metrics_queue = queue.Queue(maxsize=1000)
        
        # Initialize metrics storage
        self.power_history = []
        self.temp_history = []
        self.action_history = []
        self.battery_history = []
        self.threat_history = []
        self.cpu_history = []
        
        # Initialize timestamps
        self.timestamps = []
        self.start_time = time.time()
        
        # Enhanced thermal and power simulation
        self.thermal_simulator = ThermalSimulator()
        self.power_monitor = PowerMonitor()
        
        # Time since last algorithm switch
        self.last_algorithm_change = 0
        self.current_algorithm = 0
    
    def start(self):
        """Start metrics collection thread"""
        self.thread = threading.Thread(target=self._collection_loop)
        self.thread.daemon = True
        self.stop_event.clear()
        self.thread.start()
        logging.info("Metrics collection started")
    
    def stop(self):
        """Stop metrics collection"""
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        logging.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Main metrics collection loop"""
        while not self.stop_event.is_set():
            try:
                # Collect metrics from the system
                metrics = self._collect_system_metrics()
                
                # Add to queue for plotting
                self.metrics_queue.put(metrics)
                
                # Store in history
                self._update_history(metrics)
                
                # Sleep for update interval
                time.sleep(self.update_interval)
                
            except Exception as e:
                logging.error(f"Error collecting metrics: {e}")
    
    def _collect_system_metrics(self):
        """Collect metrics from system with enhanced thermal model"""
        # Determine current action based on state and algorithm switching rules
        action = self._determine_next_action()
        
        # Update thermal simulator with current algorithm
        temp, cpu_usage = self.thermal_simulator.update_temperature(action, dt=self.update_interval)
        
        # Calculate power consumption
        power = self.power_monitor.calculate_power(action, self.thermal_simulator)
        
        # Simulate battery level decreasing with power usage
        if not self.battery_history:
            battery = 90.0  # Start with 90% battery
        else:
            # More realistic battery drain based on power consumption
            battery_drain = power * 0.005  # Higher power = faster drain
            battery = max(0, self.battery_history[-1] - battery_drain)
        
        # Threat state changes occasionally (would come from network monitoring)
        if not self.threat_history:
            threat = 0  # Start with Normal
        elif np.random.random() < 0.05:  # 5% chance to change
            threat = np.random.choice([0, 1, 2])  # Random threat state
        else:
            threat = self.threat_history[-1]
        
        # Record timestamp
        timestamp = time.time() - self.start_time
        
        # Return collected metrics
        return {
            'timestamp': timestamp,
            'power': power,
            'temperature': temp,
            'cpu_usage': cpu_usage,
            'battery': battery,
            'threat': threat,
            'action': action,
            'thermal_category': self.thermal_simulator.get_temperature_category(),
            'thermal_trend': self.thermal_simulator.get_thermal_trend(),
            'thermal_emergency': self.thermal_simulator.is_thermal_emergency(),
            'time_since_algorithm_change': timestamp - self.last_algorithm_change
        }
    
    def _determine_next_action(self):
        """Determine next action based on current state using the decision rules"""
        # Get current state
        if not self.temp_history:
            temp = 50.0
            battery = 90.0
            threat = 0
        else:
            temp = self.temp_history[-1]
            battery = self.battery_history[-1]
            threat = self.threat_history[-1]
        
        # Get current algorithm
        current_algorithm = self.current_algorithm if self.action_history else 0
        
        # Check for TST thermal limit - emergency transition to XGBoost for cooling
        if current_algorithm == 2 and temp > 70:  # TST causing overheating
            new_action = 1  # Switch to XGBoost for cooling
            logging.warning(f"THERMAL EMERGENCY: Switching from TST to XGBoost for cooling, temp={temp:.1f}°C")
            self.last_algorithm_change = time.time() - self.start_time
            self.current_algorithm = new_action
            return new_action
        
        # Apply expert decision rules
        if battery < 20 or temp > 70:  # Critical conditions
            new_action = 0  # No DDoS - protect system
        elif threat == 0:  # Normal
            new_action = 1 if battery > 40 else 0  # XGBoost if sufficient battery
        elif threat == 1:  # Confirming
            # TST only if sufficient resources and temperature is safe
            new_action = 2 if (battery > 60 and temp < 60) else 1
        else:  # Confirmed
            new_action = 1  # XGBoost for monitoring
            
        # Check if algorithm is changing
        if new_action != current_algorithm:
            self.last_algorithm_change = time.time() - self.start_time
            self.current_algorithm = new_action
            
        return new_action
    
    def _update_history(self, metrics):
        """Update metrics history"""
        self.timestamps.append(metrics['timestamp'])
        self.power_history.append(metrics['power'])
        self.temp_history.append(metrics['temperature'])
        self.battery_history.append(metrics['battery'])
        self.threat_history.append(metrics['threat'])
        self.action_history.append(metrics['action'])
        self.cpu_history.append(metrics['cpu_usage'])
        
        # Keep history to a reasonable size
        max_history = 1000
        if len(self.timestamps) > max_history:
            self.timestamps = self.timestamps[-max_history:]
            self.power_history = self.power_history[-max_history:]
            self.temp_history = self.temp_history[-max_history:]
            self.battery_history = self.battery_history[-max_history:]
            self.threat_history = self.threat_history[-max_history:]
            self.action_history = self.action_history[-max_history:]
            self.cpu_history = self.cpu_history[-max_history:]
    
    def get_latest_metrics(self):
        """Get latest metrics"""
        try:
            return self.metrics_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_history(self):
        """Get full history of metrics"""
        return {
            'timestamps': self.timestamps,
            'power': self.power_history,
            'temperature': self.temp_history,
            'battery': self.battery_history,
            'threat': self.threat_history,
            'action': self.action_history,
            'cpu': self.cpu_history
        }
    
    def save_history(self, filepath):
        """Save metrics history to file"""
        history = self.get_history()
        
        # Convert to lists for JSON serialization
        history = {k: list(v) for k, v in history.items()}
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
        
        logging.info(f"Metrics history saved to {filepath}")


class RealtimeMonitor:
    """Real-time monitoring and visualization for UAV DDoS-RL system"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        
        # Create figure for plotting with CPU usage added
        self.fig, self.axes = plt.subplots(4, 1, figsize=(12, 12))
        self.fig.suptitle('UAV DDoS-RL Real-time Monitoring', fontsize=16)
        
        # Initialize plots
        self._setup_plots()
        
        # Animation setup
        self.ani = None
    
    def _setup_plots(self):
        """Set up the plots with enhanced visualizations"""
        # Power plot
        self.power_line, = self.axes[0].plot([], [], 'b-', label='Power (W)')
        self.axes[0].set_title('Power Consumption')
        self.axes[0].set_xlabel('Time (s)')
        self.axes[0].set_ylabel('Power (W)')
        self.axes[0].set_ylim(0, 12)
        self.axes[0].axhline(y=5.5, color='orange', linestyle='--', label='XGBoost Level')
        self.axes[0].axhline(y=9.0, color='red', linestyle='--', label='TST Level')
        self.axes[0].grid(True)
        self.axes[0].legend()
        
        # Temperature plot
        self.temp_line, = self.axes[1].plot([], [], 'r-', label='Temperature')
        self.warning_line = self.axes[1].axhline(y=65, color='orange', linestyle='--', label='Warning (65°C)')
        self.critical_line = self.axes[1].axhline(y=70, color='red', linestyle='--', label='Critical (70°C)')
        self.axes[1].set_title('Temperature')
        self.axes[1].set_xlabel('Time (s)')
        self.axes[1].set_ylabel('Temperature (°C)')
        self.axes[1].set_ylim(30, 80)
        self.axes[1].grid(True)
        self.axes[1].legend()
        
        # CPU usage plot (new)
        self.cpu_line, = self.axes[2].plot([], [], 'purple', label='CPU Usage')
        self.axes[2].set_title('CPU Usage')
        self.axes[2].set_xlabel('Time (s)')
        self.axes[2].set_ylabel('CPU (%)')
        self.axes[2].set_ylim(0, 100)
        self.axes[2].grid(True)
        self.axes[2].legend()
        
        # Battery and action plot
        self.battery_line, = self.axes[3].plot([], [], 'g-', label='Battery (%)')
        self.axes[3].set_title('Battery & Action')
        self.axes[3].set_xlabel('Time (s)')
        self.axes[3].set_ylabel('Battery (%)')
        self.axes[3].set_ylim(0, 100)
        self.axes[3].grid(True)
        
        # Secondary axis for actions
        self.ax2 = self.axes[3].twinx()
        self.action_scatter = self.ax2.scatter([], [], c=[], cmap='viridis', 
                                              marker='o', label='Action')
        self.ax2.set_ylabel('Action')
        self.ax2.set_ylim(-0.5, 2.5)
        self.ax2.set_yticks([0, 1, 2])
        self.ax2.set_yticklabels(['No DDoS', 'XGBoost', 'TST'])
        
        # Add legend for both y-axes
        lines, labels = self.axes[3].get_legend_handles_labels()
        lines2, labels2 = self.ax2.get_legend_handles_labels()
        self.ax2.legend(lines + lines2, labels + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, hspace=0.4)
    
    def _update_plots(self, frame):
        """Update plots with new data"""
        history = self.metrics_collector.get_history()
        if not history['timestamps']:
            return []
        
        # Update power plot
        self.power_line.set_data(history['timestamps'], history['power'])
        self.axes[0].set_xlim(0, max(history['timestamps']) + 1)
        if history['power']:
            max_power = max(history['power'])
            self.axes[0].set_ylim(0, max(max_power * 1.1, 12))
        
        # Update temperature plot
        self.temp_line.set_data(history['timestamps'], history['temperature'])
        self.axes[1].set_xlim(0, max(history['timestamps']) + 1)
        
        # Update CPU plot
        self.cpu_line.set_data(history['timestamps'], history['cpu'])
        self.axes[2].set_xlim(0, max(history['timestamps']) + 1)
        
        # Update battery and action plot
        self.battery_line.set_data(history['timestamps'], history['battery'])
        self.axes[3].set_xlim(0, max(history['timestamps']) + 1)
        
        # Update action scatter plot
        self.action_scatter.set_offsets(np.column_stack((history['timestamps'], history['action'])))
        self.action_scatter.set_array(np.array(history['action']))
        
        # Calculate current metrics for title update
        current_power = history['power'][-1] if history['power'] else 0
        current_temp = history['temperature'][-1] if history['temperature'] else 0
        current_battery = history['battery'][-1] if history['battery'] else 0
        current_action = history['action'][-1] if history['action'] else 0
        current_cpu = history['cpu'][-1] if history['cpu'] else 0
        action_names = ["No DDoS", "XGBoost", "TST"]
        
        # Update title with current metrics
        self.fig.suptitle(f'UAV DDoS-RL Monitoring - Power: {current_power:.1f}W, '
                         f'Temp: {current_temp:.1f}°C, CPU: {current_cpu:.1f}%, '
                         f'Action: {action_names[current_action]}', 
                         fontsize=14)
        
        return [self.power_line, self.temp_line, self.cpu_line, self.battery_line, self.action_scatter]
    
    def start(self):
        """Start monitoring"""
        self.metrics_collector.start()
        self.ani = FuncAnimation(self.fig, self._update_plots, interval=1000, blit=False)
        plt.show()
    
    def stop(self):
        """Stop monitoring"""
        if self.ani:
            self.ani.event_source.stop()
        self.metrics_collector.stop()
        
        # Save metrics history
        output_dir = 'monitoring_data'
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.metrics_collector.save_history(f"{output_dir}/metrics_{timestamp}.json")


def main():
    print("="*80)
    print("UAV DDoS-RL REAL-TIME MONITORING SYSTEM")
    print("="*80)
    print("Starting real-time monitoring. Close the plot window to exit.")
    
    try:
        # Create and start monitor
        monitor = RealtimeMonitor()
        monitor.start()
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
    finally:
        try:
            monitor.stop()
        except:
            pass
        
        print("Monitoring stopped. Data saved to monitoring_data directory.")

if __name__ == "__main__":
    main()
