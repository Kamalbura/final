import psutil
import time
from threading import Thread, Event
import logging

class HardwareSensorInterface:
    """Interface for reading hardware sensor data on Raspberry Pi"""
    
    def __init__(self, update_interval=1.0):
        self.update_interval = update_interval
        self.stop_event = Event()
        self.logger = logging.getLogger('uav_ddos_rl.hardware')
        
        # Initialize sensor values
        self.battery_level = 100.0  # %
        self.temperature = 45.0     # °C
        self.cpu_usage = 0.0        # %
        self.memory_usage = 0.0     # %
        self.network_activity = 0.0  # packets/s
        
        # For network monitoring
        self.prev_net_io = psutil.net_io_counters()
        self.prev_time = time.time()
        
        # Start monitoring thread
        self.thread = Thread(target=self._monitor_loop)
        self.thread.daemon = True
    
    def start(self):
        """Start the monitoring thread"""
        if not self.thread.is_alive():
            self.stop_event.clear()
            self.thread.start()
            self.logger.info("Hardware monitoring started")
    
    def stop(self):
        """Stop the monitoring thread"""
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
            self.logger.info("Hardware monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self.stop_event.is_set():
            try:
                self._update_measurements()
                time.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"Error in hardware monitoring: {e}")
    
    def _update_measurements(self):
        """Update all measurements"""
        self._update_battery()
        self._update_temperature()
        self._update_cpu_memory()
        self._update_network()
    
    def _update_battery(self):
        """Update battery level - on Raspberry Pi this would use a proper interface"""
        try:
            # Simulate battery drain based on CPU usage
            cpu_pct = psutil.cpu_percent(interval=None)
            # Battery drains faster with higher CPU
            drain_rate = 0.01 * (1 + cpu_pct / 50)
            self.battery_level = max(0, self.battery_level - drain_rate)
        except Exception as e:
            self.logger.warning(f"Failed to update battery: {e}")
    
    def _update_temperature(self):
        """Update CPU temperature"""
        try:
            # On Raspberry Pi, use the actual thermal sensors
            # For simulation, base it on CPU usage
            cpu_pct = psutil.cpu_percent(interval=None)
            # Temperature rises with CPU usage
            self.temperature = 40 + (cpu_pct / 2)
        except Exception as e:
            self.logger.warning(f"Failed to update temperature: {e}")
    
    def _update_cpu_memory(self):
        """Update CPU and memory usage"""
        try:
            self.cpu_usage = psutil.cpu_percent(interval=None)
            self.memory_usage = psutil.virtual_memory().percent
        except Exception as e:
            self.logger.warning(f"Failed to update CPU/memory: {e}")
    
    def _update_network(self):
        """Update network activity metrics"""
        try:
            current_time = time.time()
            current_net_io = psutil.net_io_counters()
            
            # Calculate packets per second
            time_diff = current_time - self.prev_time
            if time_diff > 0:
                packets_sent = current_net_io.packets_sent - self.prev_net_io.packets_sent
                packets_recv = current_net_io.packets_recv - self.prev_net_io.packets_recv
                
                self.network_activity = (packets_sent + packets_recv) / time_diff
            
            self.prev_net_io = current_net_io
            self.prev_time = current_time
        except Exception as e:
            self.logger.warning(f"Failed to update network stats: {e}")
    
    def get_battery_category(self):
        """Get battery level category"""
        if self.battery_level <= 20:
            return "0-20%"
        elif self.battery_level <= 40:
            return "21-40%"
        elif self.battery_level <= 60:
            return "41-60%"
        elif self.battery_level <= 80:
            return "61-80%"
        else:
            return "81-100%"
    
    def get_temperature_category(self):
        """Get temperature category"""
        if self.temperature <= 55:
            return "Safe"
        elif self.temperature <= 70:
            return "Warning"
        else:
            return "Critical"
    
    def get_state_data(self):
        """Get the current hardware state data"""
        return {
            'battery': self.get_battery_category(),
            'battery_raw': self.battery_level,
            'temperature': self.get_temperature_category(),
            'temperature_raw': self.temperature,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'network_activity': self.network_activity,
            'timestamp': time.time()
        }
