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
        
        # Initialize timestamps
        self.timestamps = []
        self.start_time = time.time()
    
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
        """Collect metrics from system (would connect to real sensors)"""
        # In a real implementation, this would read from actual sensors
        # For demonstration, we'll simulate values
        
        # Simulate power consumption based on current action
        # This would read from actual power sensors in production
        current_action = self.action_history[-1] if self.action_history else 0
        power_base = [3.0, 5.5, 9.0][current_action]  # Base power for each action
        power_variation = np.random.normal(0, 0.2)  # Add some noise
        power = max(0, power_base + power_variation)
        
        # Simulate temperature based on power consumption
        # In reality, this would read from thermal sensors
        if not self.temp_history:
            temp = 30.0  # Starting temperature
        else:
            # Temperature increases with power and decreases with time
            temp_increase = power * 0.01  # Higher power = faster heating
            cooling = max(0, (self.temp_history[-1] - 30) * 0.01)  # Natural cooling
            temp = self.temp_history[-1] + temp_increase - cooling + np.random.normal(0, 0.1)
        
        # Simulate battery level decreasing with power usage
        if not self.battery_history:
            battery = 90.0  # Start with 90% battery
        else:
            battery_drain = power * 0.005  # Higher power = faster drain
            battery = max(0, self.battery_history[-1] - battery_drain)
        
        # Threat state changes occasionally (would come from network monitoring)
        if not self.threat_history:
            threat = 0  # Start with Normal
        elif np.random.random() < 0.05:  # 5% chance to change
            threat = np.random.choice([0, 1, 2])  # Random threat state
        else:
            threat = self.threat_history[-1]
        
        # Current action (would come from RL agent)
        # For simulation, we'll use a simple heuristic
        if battery < 20 or temp > 70:
            action = 0  # No DDoS for critical conditions
        elif threat == 0:  # Normal
            action = 1 if battery > 40 else 0  # XGBoost if sufficient battery
        elif threat == 1:  # Confirming
            action = 2 if battery > 60 and temp < 60 else 1  # TST if resources good
        else:  # Confirmed
            action = 1  # XGBoost for monitoring
        
        # Record timestamp
        timestamp = time.time() - self.start_time
        
        # Return collected metrics
        return {
            'timestamp': timestamp,
            'power': power,
            'temperature': temp,
            'battery': battery,
            'threat': threat,
            'action': action
        }
    
    def _update_history(self, metrics):
        """Update metrics history"""
        self.timestamps.append(metrics['timestamp'])
        self.power_history.append(metrics['power'])
        self.temp_history.append(metrics['temperature'])
        self.battery_history.append(metrics['battery'])
        self.threat_history.append(metrics['threat'])
        self.action_history.append(metrics['action'])
        
        # Keep history to a reasonable size
        max_history = 1000
        if len(self.timestamps) > max_history:
            self.timestamps = self.timestamps[-max_history:]
            self.power_history = self.power_history[-max_history:]
            self.temp_history = self.temp_history[-max_history:]
            self.battery_history = self.battery_history[-max_history:]
            self.threat_history = self.threat_history[-max_history:]
            self.action_history = self.action_history[-max_history:]
    
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
            'action': self.action_history
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
        
        # Create figure for plotting
        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 10))
        self.fig.suptitle('UAV DDoS-RL Real-time Monitoring', fontsize=16)
        
        # Initialize plots
        self._setup_plots()
        
        # Animation setup
        self.ani = None
    
    def _setup_plots(self):
        """Set up the plots"""
        # Power plot
        self.power_line, = self.axes[0].plot([], [], 'b-', label='Power (W)')
        self.axes[0].set_title('Power Consumption')
        self.axes[0].set_xlabel('Time (s)')
        self.axes[0].set_ylabel('Power (W)')
        self.axes[0].set_ylim(0, 10)
        self.axes[0].grid(True)
        self.axes[0].legend()
        
        # Temperature plot
        self.temp_line, = self.axes[1].plot([], [], 'r-', label='Temperature')
        self.warning_line = self.axes[1].axhline(y=70, color='orange', linestyle='--', label='Warning')
        self.critical_line = self.axes[1].axhline(y=80, color='red', linestyle='--', label='Critical')
        self.axes[1].set_title('Temperature')
        self.axes[1].set_xlabel('Time (s)')
        self.axes[1].set_ylabel('Temperature (°C)')
        self.axes[1].set_ylim(20, 90)
        self.axes[1].grid(True)
        self.axes[1].legend()
        
        # Battery and action plot
        self.battery_line, = self.axes[2].plot([], [], 'g-', label='Battery (%)')
        self.axes[2].set_title('Battery & Action')
        self.axes[2].set_xlabel('Time (s)')
        self.axes[2].set_ylabel('Battery (%)')
        self.axes[2].set_ylim(0, 100)
        self.axes[2].grid(True)
        
        # Secondary axis for actions
        self.ax2 = self.axes[2].twinx()
        self.action_scatter = self.ax2.scatter([], [], c=[], cmap='viridis', 
                                              marker='o', label='Action')
        self.ax2.set_ylabel('Action')
        self.ax2.set_ylim(-0.5, 2.5)
        self.ax2.set_yticks([0, 1, 2])
        self.ax2.set_yticklabels(['No DDoS', 'XGBoost', 'TST'])
        
        # Add legend for both y-axes
        lines, labels = self.axes[2].get_legend_handles_labels()
        lines2, labels2 = self.ax2.get_legend_handles_labels()
        self.ax2.legend(lines + lines2, labels + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, hspace=0.3)
    
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
            self.axes[0].set_ylim(0, max(max_power * 1.1, 10))
        
        # Update temperature plot
        self.temp_line.set_data(history['timestamps'], history['temperature'])
        self.axes[1].set_xlim(0, max(history['timestamps']) + 1)
        
        # Update battery and action plot
        self.battery_line.set_data(history['timestamps'], history['battery'])
        self.axes[2].set_xlim(0, max(history['timestamps']) + 1)
        
        # Update action scatter plot
        self.action_scatter.set_offsets(np.column_stack((history['timestamps'], history['action'])))
        self.action_scatter.set_array(np.array(history['action']))
        
        # Calculate current metrics for title update
        current_power = history['power'][-1] if history['power'] else 0
        current_temp = history['temperature'][-1] if history['temperature'] else 0
        current_battery = history['battery'][-1] if history['battery'] else 0
        current_action = history['action'][-1] if history['action'] else 0
        action_names = ["No DDoS", "XGBoost", "TST"]
        
        # Update title with current metrics
        self.fig.suptitle(f'UAV DDoS-RL Monitoring - Power: {current_power:.1f}W, '
                         f'Temp: {current_temp:.1f}°C, Battery: {current_battery:.1f}%, '
                         f'Action: {action_names[current_action]}', 
                         fontsize=14)
        
        return [self.power_line, self.temp_line, self.battery_line, self.action_scatter]
    
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
