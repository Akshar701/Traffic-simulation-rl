#!/usr/bin/env python3
"""
Enhanced Traffic Metrics System for Streamlit Dashboard
Features:
- Real-time metric updates
- Baseline performance comparison
- Comprehensive performance tracking
- Dashboard-ready data structures
"""

import pandas as pd
import numpy as np
import os
import glob
import xml.etree.ElementTree as ET
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import threading
import queue

@dataclass
class TrafficMetrics:
    """Structured traffic metrics for dashboard display"""
    timestamp: float
    total_vehicles: int
    avg_speed: float
    max_queue_length: float
    avg_queue_length: float
    total_waiting_time: float
    avg_waiting_time: float
    efficiency_score: float
    congestion_level: str
    throughput: float
    density: float
    scenario: str
    episode: int

class RealTimeMetricsCollector:
    """Real-time metrics collector for live dashboard updates"""
    
    def __init__(self, output_dir="Sumo_env/Single intersection lhd"):
        self.output_dir = output_dir
        self.metrics_queue = queue.Queue()
        self.is_collecting = False
        self.collection_thread = None
        self.baseline_metrics = None
        self.current_episode = 0
        self.current_scenario = "unknown"
        
    def start_collection(self, scenario: str = "unknown", episode: int = 0):
        """Start real-time metrics collection"""
        self.current_scenario = scenario
        self.current_episode = episode
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collect_metrics_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
    def stop_collection(self):
        """Stop real-time metrics collection"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=1)
            
    def _collect_metrics_loop(self):
        """Background thread for continuous metrics collection"""
        while self.is_collecting:
            try:
                metrics = self._get_current_metrics()
                if metrics:
                    self.metrics_queue.put(metrics)
                time.sleep(2)  # Update every 2 seconds
            except Exception as e:
                print(f"Error in metrics collection: {e}")
                time.sleep(5)
                
    def get_latest_metrics(self) -> Optional[TrafficMetrics]:
        """Get the latest metrics from the queue"""
        try:
            return self.metrics_queue.get_nowait()
        except queue.Empty:
            return None
            
    def _get_current_metrics(self) -> Optional[TrafficMetrics]:
        """Get current metrics from detector files"""
        try:
            analyzer = EnhancedTrafficAnalyzer(self.output_dir)
            metrics_dict = analyzer.calculate_essential_metrics()
            
            if not metrics_dict:
                return None
                
            return TrafficMetrics(
                timestamp=time.time(),
                total_vehicles=metrics_dict.get('total_vehicles', 0),
                avg_speed=metrics_dict.get('avg_speed', 0),
                max_queue_length=metrics_dict.get('max_queue_length', 0),
                avg_queue_length=metrics_dict.get('avg_queue_length', 0),
                total_waiting_time=metrics_dict.get('total_waiting_time', 0),
                avg_waiting_time=metrics_dict.get('avg_waiting_time', 0),
                efficiency_score=analyzer.calculate_efficiency_score(metrics_dict),
                congestion_level=analyzer.calculate_congestion_level(metrics_dict),
                throughput=metrics_dict.get('throughput', 0),
                density=metrics_dict.get('density', 0),
                scenario=self.current_scenario,
                episode=self.current_episode
            )
        except Exception as e:
            print(f"Error getting current metrics: {e}")
            return None

class EnhancedTrafficAnalyzer:
    """Enhanced traffic analyzer with dashboard-optimized features"""
    
    def __init__(self, output_dir="Sumo_env/Single intersection lhd"):
        self.output_dir = output_dir
        self.baseline_data = None
        self.metrics_history = []
        
    def load_baseline_data(self, baseline_file: str):
        """Load baseline performance data for comparison"""
        try:
            if os.path.exists(baseline_file):
                with open(baseline_file, 'r') as f:
                    self.baseline_data = json.load(f)
                print(f"✓ Baseline data loaded from {baseline_file}")
            else:
                print(f"⚠️ Baseline file not found: {baseline_file}")
        except Exception as e:
            print(f"Error loading baseline data: {e}")
            
    def save_baseline_data(self, baseline_file: str, scenario: str = "baseline"):
        """Save current performance as baseline data"""
        try:
            metrics = self.calculate_essential_metrics()
            baseline_data = {
                'scenario': scenario,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'efficiency_score': self.calculate_efficiency_score(metrics),
                'congestion_level': self.calculate_congestion_level(metrics)
            }
            
            with open(baseline_file, 'w') as f:
                json.dump(baseline_data, f, indent=2)
            
            print(f"✓ Baseline data saved to {baseline_file}")
            return baseline_data
        except Exception as e:
            print(f"Error saving baseline data: {e}")
            return None
            
    def compare_with_baseline(self, current_metrics: Dict) -> Dict:
        """Compare current metrics with baseline"""
        if not self.baseline_data:
            return {}
            
        baseline_metrics = self.baseline_data['metrics']
        
        comparison = {}
        for key in current_metrics:
            if key in baseline_metrics and baseline_metrics[key] != 0:
                current_val = current_metrics[key]
                baseline_val = baseline_metrics[key]
                change_pct = ((current_val - baseline_val) / baseline_val) * 100
                comparison[f"{key}_change_pct"] = change_pct
                comparison[f"{key}_baseline"] = baseline_val
                
        return comparison
        
    def analyze_flow_data(self, flow_file: str) -> Dict:
        """Enhanced flow data analysis with more metrics"""
        try:
            tree = ET.parse(flow_file)
            root = tree.getroot()
            
            total_vehicles = 0
            speeds = []
            flows = []
            densities = []
            
            for interval in root.findall('.//interval'):
                n_veh = int(interval.get('nVehContrib', 0))
                total_vehicles += n_veh
                
                speed = float(interval.get('speed', -1))
                if speed > 0:
                    speeds.append(speed)
                
                flow = float(interval.get('flow', 0))
                if flow > 0:
                    flows.append(flow)
                    
                density = float(interval.get('density', 0))
                if density > 0:
                    densities.append(density)
            
            return {
                'total_vehicles': total_vehicles,
                'avg_speed': np.mean(speeds) if speeds else 0,
                'avg_flow': np.mean(flows) if flows else 0,
                'avg_density': np.mean(densities) if densities else 0,
                'throughput': total_vehicles,
                'file': flow_file
            }
        except Exception as e:
            print(f"Error analyzing flow data {flow_file}: {e}")
            return {}
    
    def analyze_queue_data(self, queue_file: str) -> Dict:
        """Enhanced queue data analysis"""
        try:
            tree = ET.parse(queue_file)
            root = tree.getroot()
            
            queue_lengths = []
            waiting_times = []
            jam_lengths = []
            
            for interval in root.findall('.//interval'):
                queue_length = float(interval.get('queueing_length', 0))
                if queue_length > 0:
                    queue_lengths.append(queue_length)
                
                waiting_time = float(interval.get('waiting_time', 0))
                if waiting_time > 0:
                    waiting_times.append(waiting_time)
                    
                jam_length = float(interval.get('jam_length_vehicle', 0))
                if jam_length > 0:
                    jam_lengths.append(jam_length)
            
            return {
                'max_queue_length': max(queue_lengths) if queue_lengths else 0,
                'avg_queue_length': np.mean(queue_lengths) if queue_lengths else 0,
                'total_waiting_time': sum(waiting_times),
                'avg_waiting_time': np.mean(waiting_times) if waiting_times else 0,
                'max_jam_length': max(jam_lengths) if jam_lengths else 0,
                'file': queue_file
            }
        except Exception as e:
            print(f"Error analyzing queue data {queue_file}: {e}")
            return {}
    
    def analyze_wait_data(self, wait_file: str) -> Dict:
        """Enhanced wait time analysis"""
        try:
            tree = ET.parse(wait_file)
            root = tree.getroot()
            
            waiting_times = []
            stopped_times = []
            
            for interval in root.findall('.//interval'):
                waiting_time = float(interval.get('waiting_time', 0))
                if waiting_time > 0:
                    waiting_times.append(waiting_time)
                    
                stopped_time = float(interval.get('timeLoss', 0))
                if stopped_time > 0:
                    stopped_times.append(stopped_time)
            
            return {
                'total_waiting_time': sum(waiting_times),
                'avg_waiting_time': np.mean(waiting_times) if waiting_times else 0,
                'max_waiting_time': max(waiting_times) if waiting_times else 0,
                'vehicles_waiting': len(waiting_times),
                'total_stopped_time': sum(stopped_times),
                'avg_stopped_time': np.mean(stopped_times) if stopped_times else 0,
                'file': wait_file
            }
        except Exception as e:
            print(f"Error analyzing wait data {wait_file}: {e}")
            return {}
    
    def calculate_essential_metrics(self) -> Dict:
        """Calculate comprehensive essential metrics"""
        # Find detector output files
        flow_files = glob.glob(os.path.join(self.output_dir, "flow_*.out"))
        queue_files = glob.glob(os.path.join(self.output_dir, "queue_*.out"))
        wait_files = glob.glob(os.path.join(self.output_dir, "wait_*.out"))
        
        # Analyze data
        flow_metrics = [self.analyze_flow_data(f) for f in flow_files]
        queue_metrics = [self.analyze_queue_data(f) for f in queue_files]
        wait_metrics = [self.analyze_wait_data(f) for f in wait_files]
        
        # Calculate comprehensive metrics with safe defaults
        essential_metrics = {
            'total_vehicles': sum(m.get('total_vehicles', 0) for m in flow_metrics),
            'avg_speed': 0.0,
            'max_queue_length': 0.0,
            'avg_queue_length': 0.0,
            'total_waiting_time': 0.0,
            'avg_waiting_time': 0.0,
            'throughput': sum(m.get('throughput', 0) for m in flow_metrics),
            'density': 0.0,
            'max_waiting_time': 0.0,
            'total_stopped_time': 0.0,
            'avg_stopped_time': 0.0
        }
        
        # Safely calculate averages and max values
        avg_speeds = [m.get('avg_speed', 0) for m in flow_metrics if m.get('avg_speed', 0) > 0]
        if avg_speeds:
            essential_metrics['avg_speed'] = np.mean(avg_speeds)
            
        queue_lengths = [m.get('max_queue_length', 0) for m in queue_metrics if m.get('max_queue_length', 0) > 0]
        if queue_lengths:
            essential_metrics['max_queue_length'] = max(queue_lengths)
            
        avg_queue_lengths = [m.get('avg_queue_length', 0) for m in queue_metrics if m.get('avg_queue_length', 0) > 0]
        if avg_queue_lengths:
            essential_metrics['avg_queue_length'] = np.mean(avg_queue_lengths)
            
        total_waiting = sum(m.get('total_waiting_time', 0) for m in wait_metrics)
        essential_metrics['total_waiting_time'] = total_waiting
        
        avg_waiting_times = [m.get('avg_waiting_time', 0) for m in wait_metrics if m.get('avg_waiting_time', 0) > 0]
        if avg_waiting_times:
            essential_metrics['avg_waiting_time'] = np.mean(avg_waiting_times)
            
        densities = [m.get('avg_density', 0) for m in flow_metrics if m.get('avg_density', 0) > 0]
        if densities:
            essential_metrics['density'] = np.mean(densities)
            
        max_waiting_times = [m.get('max_waiting_time', 0) for m in wait_metrics if m.get('max_waiting_time', 0) > 0]
        if max_waiting_times:
            essential_metrics['max_waiting_time'] = max(max_waiting_times)
            
        total_stopped = sum(m.get('total_stopped_time', 0) for m in wait_metrics)
        essential_metrics['total_stopped_time'] = total_stopped
        
        avg_stopped_times = [m.get('avg_stopped_time', 0) for m in wait_metrics if m.get('avg_stopped_time', 0) > 0]
        if avg_stopped_times:
            essential_metrics['avg_stopped_time'] = np.mean(avg_stopped_times)
        
        return essential_metrics
    
    def calculate_efficiency_score(self, metrics: Dict) -> float:
        """Enhanced efficiency score calculation"""
        score = 100.0
        
        # Penalize for delays
        if metrics['avg_waiting_time'] > 0:
            score -= min(metrics['avg_waiting_time'] * 1.5, 25)
        
        # Penalize for long queues
        if metrics['avg_queue_length'] > 0:
            score -= min(metrics['avg_queue_length'] * 4, 20)
        
        # Penalize for low speeds
        if metrics['avg_speed'] < 10:
            score -= min((10 - metrics['avg_speed']) * 2, 20)
            
        # Penalize for stopped time
        if metrics['avg_stopped_time'] > 0:
            score -= min(metrics['avg_stopped_time'] * 0.5, 15)
        
        # Bonus for high throughput
        if metrics['throughput'] > 1000:
            score += min((metrics['throughput'] - 1000) * 0.01, 10)
        
        return max(0, min(100, score))
    
    def calculate_congestion_level(self, metrics: Dict) -> str:
        """Enhanced congestion level determination (RL-optimized thresholds)"""
        wait_score = metrics['avg_waiting_time']
        queue_score = metrics['avg_queue_length']
        density_score = metrics.get('density', 0)
        
        # Weighted scoring (adjusted for RL training)
        total_score = (wait_score * 0.4 + queue_score * 0.3 + density_score * 0.3)
        
        # More sensitive thresholds for RL training
        if total_score < 2:
            return "LOW"
        elif total_score < 8:
            return "MODERATE"
        elif total_score < 20:
            return "HIGH"
        else:
            return "SEVERE"
            
    def get_dashboard_data(self, scenario: str = "current") -> Dict:
        """Get formatted data for dashboard display"""
        metrics = self.calculate_essential_metrics()
        efficiency_score = self.calculate_efficiency_score(metrics)
        congestion_level = self.calculate_congestion_level(metrics)
        
        dashboard_data = {
            'scenario': scenario,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'efficiency_score': efficiency_score,
            'congestion_level': congestion_level,
            'comparison': self.compare_with_baseline(metrics) if self.baseline_data else {},
            'summary': {
                'status': '🟢 Good' if efficiency_score > 70 else '🟡 Moderate' if efficiency_score > 50 else '🔴 Poor',
                'recommendation': self._get_recommendation(efficiency_score, congestion_level)
            }
        }
        
        return dashboard_data
        
    def _get_recommendation(self, efficiency_score: float, congestion_level: str) -> str:
        """Get recommendation based on performance"""
        if efficiency_score > 80 and congestion_level == "LOW":
            return "Excellent performance! Consider optimizing for even better efficiency."
        elif efficiency_score > 60:
            return "Good performance. Minor optimizations could improve flow."
        elif efficiency_score > 40:
            return "Moderate performance. Traffic light timing adjustments recommended."
        else:
            return "Poor performance. Major traffic management improvements needed."

def create_baseline_performance():
    """Create baseline performance data for comparison"""
    analyzer = EnhancedTrafficAnalyzer()
    
    # Create baseline for each scenario
    scenarios = ["uniform", "tidal", "asymmetric"]
    
    for scenario in scenarios:
        baseline_file = f"baseline_{scenario}.json"
        analyzer.save_baseline_data(baseline_file, scenario)
        
    print("✓ Baseline performance data created for all scenarios")

if __name__ == "__main__":
    # Demo the enhanced metrics system
    print("Enhanced Traffic Metrics System Demo")
    print("=" * 40)
    
    analyzer = EnhancedTrafficAnalyzer()
    
    try:
        # Get current metrics
        dashboard_data = analyzer.get_dashboard_data("demo")
        
        print(f"Current Performance:")
        print(f"  Efficiency Score: {dashboard_data['efficiency_score']:.1f}/100")
        print(f"  Congestion Level: {dashboard_data['congestion_level']}")
        print(f"  Total Vehicles: {dashboard_data['metrics']['total_vehicles']:,}")
        print(f"  Average Speed: {dashboard_data['metrics']['avg_speed']:.1f} m/s")
        print(f"  Average Waiting Time: {dashboard_data['metrics']['avg_waiting_time']:.1f} seconds")
        print(f"  Status: {dashboard_data['summary']['status']}")
        print(f"  Recommendation: {dashboard_data['summary']['recommendation']}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Note: Requires detector output files from SUMO simulation")
