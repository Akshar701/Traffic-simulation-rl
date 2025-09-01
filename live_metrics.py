#!/usr/bin/env python3
"""
Live Metrics Collector - Real-time traffic data collection from SUMO
Collects and processes live traffic metrics during simulation
"""

import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque
from datetime import datetime
import threading

from traci_manager import TraciManager, TrafficState

@dataclass
class LiveMetrics:
    """Live traffic metrics data structure"""
    timestamp: float
    simulation_time: float
    vehicle_count: int
    waiting_vehicles: int
    avg_waiting_time: float
    avg_speed: float
    queue_length: int
    current_phase: int
    phase_duration: float
    efficiency_score: float
    congestion_level: str
    throughput: float
    density: float

class MetricsCollector:
    """Collects and processes live traffic metrics"""
    
    def __init__(self, traci_manager: TraciManager, history_size: int = 1000):
        self.traci_manager = traci_manager
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.is_collecting = False
        self.collection_thread = None
        self.collection_interval = 1.0  # seconds
        
        # Performance thresholds
        self.congestion_thresholds = {
            "low": 5,
            "moderate": 15,
            "high": 30,
            "severe": 50
        }
        
        # Baseline metrics for comparison
        self.baseline_metrics = None
        
    def start_collection(self, interval: float = 1.0):
        """Start collecting metrics in a separate thread"""
        if self.is_collecting:
            print("Metrics collection already running")
            return
            
        self.collection_interval = interval
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        print(f"Started metrics collection with {interval}s interval")
    
    def stop_collection(self):
        """Stop collecting metrics"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=2.0)
        print("Stopped metrics collection")
    
    def _collection_loop(self):
        """Main collection loop"""
        while self.is_collecting:
            try:
                # Get current traffic state
                traffic_state = self.traci_manager.get_traffic_state()
                if traffic_state:
                    # Process and store metrics
                    live_metrics = self._process_metrics(traffic_state)
                    self.metrics_history.append(live_metrics)
                
                # Wait for next collection interval
                time.sleep(self.collection_interval)
                
            except Exception as e:
                print(f"Error in metrics collection: {e}")
                time.sleep(self.collection_interval)
    
    def _process_metrics(self, traffic_state: TrafficState) -> LiveMetrics:
        """Process raw traffic state into live metrics"""
        # Calculate efficiency score (0-100)
        efficiency_score = self._calculate_efficiency_score(traffic_state)
        
        # Determine congestion level
        congestion_level = self._determine_congestion_level(traffic_state)
        
        # Calculate throughput (vehicles per hour)
        throughput = self._calculate_throughput(traffic_state)
        
        # Calculate density (vehicles per km)
        density = self._calculate_density(traffic_state)
        
        return LiveMetrics(
            timestamp=time.time(),
            simulation_time=traffic_state.timestamp,
            vehicle_count=traffic_state.vehicle_count,
            waiting_vehicles=traffic_state.waiting_vehicles,
            avg_waiting_time=traffic_state.avg_waiting_time,
            avg_speed=traffic_state.avg_speed,
            queue_length=traffic_state.queue_length,
            current_phase=traffic_state.current_phase,
            phase_duration=traffic_state.phase_duration,
            efficiency_score=efficiency_score,
            congestion_level=congestion_level,
            throughput=throughput,
            density=density
        )
    
    def _calculate_efficiency_score(self, traffic_state: TrafficState) -> float:
        """Calculate efficiency score (0-100) based on multiple factors"""
        score = 100.0
        
        # Penalize for waiting vehicles
        if traffic_state.waiting_vehicles > 0:
            waiting_penalty = min(traffic_state.waiting_vehicles * 2, 30)
            score -= waiting_penalty
        
        # Penalize for low average speed
        if traffic_state.avg_speed < 10:
            speed_penalty = (10 - traffic_state.avg_speed) * 3
            score -= speed_penalty
        
        # Penalize for long waiting times
        if traffic_state.avg_waiting_time > 30:
            time_penalty = min((traffic_state.avg_waiting_time - 30) * 0.5, 20)
            score -= time_penalty
        
        # Bonus for high throughput
        if traffic_state.vehicle_count > 50:
            throughput_bonus = min((traffic_state.vehicle_count - 50) * 0.1, 10)
            score += throughput_bonus
        
        return max(0.0, min(100.0, score))
    
    def _determine_congestion_level(self, traffic_state: TrafficState) -> str:
        """Determine congestion level based on queue length"""
        queue_length = traffic_state.queue_length
        
        if queue_length <= self.congestion_thresholds["low"]:
            return "LOW"
        elif queue_length <= self.congestion_thresholds["moderate"]:
            return "MODERATE"
        elif queue_length <= self.congestion_thresholds["high"]:
            return "HIGH"
        else:
            return "SEVERE"
    
    def _calculate_throughput(self, traffic_state: TrafficState) -> float:
        """Calculate throughput (vehicles per hour)"""
        # This is a simplified calculation
        # In a real system, you'd track vehicles that actually pass through
        return traffic_state.vehicle_count * 3.6  # Rough estimate
    
    def _calculate_density(self, traffic_state: TrafficState) -> float:
        """Calculate traffic density (vehicles per km)"""
        # Simplified calculation - assumes 1km total road length
        return traffic_state.vehicle_count / 1.0
    
    def get_current_metrics(self) -> Optional[LiveMetrics]:
        """Get the most recent metrics"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_metrics_history(self, count: int = None) -> List[LiveMetrics]:
        """Get recent metrics history"""
        if count is None:
            count = len(self.metrics_history)
        return list(self.metrics_history)[-count:]
    
    def get_average_metrics(self, window: int = 10) -> Optional[LiveMetrics]:
        """Get average metrics over a time window"""
        recent_metrics = self.get_metrics_history(window)
        if not recent_metrics:
            return None
        
        # Calculate averages
        avg_metrics = {}
        for field in LiveMetrics.__dataclass_fields__:
            if field == 'timestamp' or field == 'simulation_time':
                # Use the latest timestamp
                avg_metrics[field] = recent_metrics[-1].__getattribute__(field)
            elif field == 'congestion_level':
                # Use the most common congestion level
                levels = [m.__getattribute__(field) for m in recent_metrics]
                avg_metrics[field] = max(set(levels), key=levels.count)
            else:
                # Calculate average for numeric fields
                values = [m.__getattribute__(field) for m in recent_metrics]
                avg_metrics[field] = sum(values) / len(values)
        
        return LiveMetrics(**avg_metrics)
    
    def set_baseline(self, baseline_metrics: LiveMetrics):
        """Set baseline metrics for comparison"""
        self.baseline_metrics = baseline_metrics
        print("Baseline metrics set")
    
    def compare_to_baseline(self) -> Dict[str, float]:
        """Compare current metrics to baseline"""
        if not self.baseline_metrics:
            return {}
        
        current = self.get_current_metrics()
        if not current:
            return {}
        
        baseline = self.baseline_metrics
        
        return {
            "efficiency_improvement": current.efficiency_score - baseline.efficiency_score,
            "waiting_time_reduction": baseline.avg_waiting_time - current.avg_waiting_time,
            "speed_improvement": current.avg_speed - baseline.avg_speed,
            "queue_reduction": baseline.queue_length - current.queue_length,
            "throughput_improvement": current.throughput - baseline.throughput
        }
    
    def export_metrics(self, filename: str = None) -> str:
        """Export metrics history to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_export_{timestamp}.json"
        
        metrics_data = []
        for metrics in self.metrics_history:
            metrics_data.append(asdict(metrics))
        
        with open(filename, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"Metrics exported to {filename}")
        return filename
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of current performance"""
        current = self.get_current_metrics()
        if not current:
            return {}
        
        avg_metrics = self.get_average_metrics(10)
        
        summary = {
            "current": asdict(current),
            "average_10_steps": asdict(avg_metrics) if avg_metrics else {},
            "history_size": len(self.metrics_history),
            "collection_active": self.is_collecting
        }
        
        # Add baseline comparison if available
        if self.baseline_metrics:
            summary["baseline_comparison"] = self.compare_to_baseline()
        
        return summary

# Example usage
if __name__ == "__main__":
    # Create traci manager and metrics collector
    traci_manager = TraciManager()
    metrics_collector = MetricsCollector(traci_manager)
    
    # Start simulation
    if traci_manager.start_simulation():
        # Start collecting metrics
        metrics_collector.start_collection(interval=2.0)
        
        # Run simulation for a while
        for _ in range(50):
            traci_manager.step_simulation(10)
            time.sleep(0.1)
            
            # Print current metrics
            current = metrics_collector.get_current_metrics()
            if current:
                print(f"Time: {current.simulation_time:.1f}, "
                      f"Vehicles: {current.vehicle_count}, "
                      f"Efficiency: {current.efficiency_score:.1f}, "
                      f"Congestion: {current.congestion_level}")
        
        # Stop collection and simulation
        metrics_collector.stop_collection()
        traci_manager.stop_simulation()
        
        # Export metrics
        metrics_collector.export_metrics()
