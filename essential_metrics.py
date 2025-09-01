#!/usr/bin/env python3
"""
Essential Performance Metrics for RL Training
Focused on core metrics: queue length, wait time, throughput
"""

import pandas as pd
import numpy as np
import os
import glob
import xml.etree.ElementTree as ET
from typing import Dict

class EssentialTrafficAnalyzer:
    def __init__(self, output_dir="Sumo_env/Single intersection lhd"):
        self.output_dir = output_dir
        
    def analyze_flow_data(self, flow_file: str) -> Dict:
        """Analyze flow detector data (XML format)"""
        try:
            tree = ET.parse(flow_file)
            root = tree.getroot()
            
            # Extract data from XML
            total_vehicles = 0
            speeds = []
            flows = []
            
            for interval in root.findall('.//interval'):
                n_veh = int(interval.get('nVehContrib', 0))
                total_vehicles += n_veh
                
                speed = float(interval.get('speed', -1))
                if speed > 0:
                    speeds.append(speed)
                
                flow = float(interval.get('flow', 0))
                if flow > 0:
                    flows.append(flow)
            
            return {
                'total_vehicles': total_vehicles,
                'avg_speed': np.mean(speeds) if speeds else 0,
                'avg_flow': np.mean(flows) if flows else 0,
                'file': flow_file
            }
        except Exception as e:
            print(f"Error analyzing flow data {flow_file}: {e}")
            return {}
    
    def analyze_queue_data(self, queue_file: str) -> Dict:
        """Analyze queue length detector data (XML format)"""
        try:
            tree = ET.parse(queue_file)
            root = tree.getroot()
            
            # Extract data from XML
            queue_lengths = []
            waiting_times = []
            
            for interval in root.findall('.//interval'):
                queue_length = float(interval.get('queueing_length', 0))
                if queue_length > 0:
                    queue_lengths.append(queue_length)
                
                waiting_time = float(interval.get('waiting_time', 0))
                if waiting_time > 0:
                    waiting_times.append(waiting_time)
            
            return {
                'max_queue_length': max(queue_lengths) if queue_lengths else 0,
                'avg_queue_length': np.mean(queue_lengths) if queue_lengths else 0,
                'total_waiting_time': sum(waiting_times),
                'avg_waiting_time': np.mean(waiting_times) if waiting_times else 0,
                'file': queue_file
            }
        except Exception as e:
            print(f"Error analyzing queue data {queue_file}: {e}")
            return {}
    
    def analyze_wait_data(self, wait_file: str) -> Dict:
        """Analyze wait time detector data (XML format)"""
        try:
            tree = ET.parse(wait_file)
            root = tree.getroot()
            
            # Extract data from XML
            waiting_times = []
            
            for interval in root.findall('.//interval'):
                waiting_time = float(interval.get('waiting_time', 0))
                if waiting_time > 0:
                    waiting_times.append(waiting_time)
            
            return {
                'total_waiting_time': sum(waiting_times),
                'avg_waiting_time': np.mean(waiting_times) if waiting_times else 0,
                'max_waiting_time': max(waiting_times) if waiting_times else 0,
                'vehicles_waiting': len(waiting_times),
                'file': wait_file
            }
        except Exception as e:
            print(f"Error analyzing wait data {wait_file}: {e}")
            return {}
    
    def calculate_essential_metrics(self) -> Dict:
        """Calculate essential performance metrics"""
        print("Analyzing essential traffic metrics...")
        
        # Find detector output files
        flow_files = glob.glob(os.path.join(self.output_dir, "flow_*.out"))
        queue_files = glob.glob(os.path.join(self.output_dir, "queue_*.out"))
        wait_files = glob.glob(os.path.join(self.output_dir, "wait_*.out"))
        
        print(f"Found {len(flow_files)} flow files, {len(queue_files)} queue files, {len(wait_files)} wait files")
        
        # Analyze data
        flow_metrics = [self.analyze_flow_data(f) for f in flow_files]
        queue_metrics = [self.analyze_queue_data(f) for f in queue_files]
        wait_metrics = [self.analyze_wait_data(f) for f in wait_files]
        
        # Calculate essential metrics
        essential_metrics = {
            'total_vehicles': sum(m.get('total_vehicles', 0) for m in flow_metrics),
            'avg_speed': np.mean([m.get('avg_speed', 0) for m in flow_metrics if m.get('avg_speed', 0) > 0]),
            'max_queue_length': max([m.get('max_queue_length', 0) for m in queue_metrics]),
            'avg_queue_length': np.mean([m.get('avg_queue_length', 0) for m in queue_metrics if m.get('avg_queue_length', 0) > 0]),
            'total_waiting_time': sum(m.get('total_waiting_time', 0) for m in wait_metrics),
            'avg_waiting_time': np.mean([m.get('avg_waiting_time', 0) for m in wait_metrics if m.get('avg_waiting_time', 0) > 0])
        }
        
        return essential_metrics
    
    def generate_essential_report(self, scenario_name: str = "traffic_simulation") -> str:
        """Generate essential performance report"""
        metrics = self.calculate_essential_metrics()
        
        report = f"""
ESSENTIAL TRAFFIC PERFORMANCE REPORT
====================================
Scenario: {scenario_name}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

ESSENTIAL METRICS
-----------------
Total Vehicles: {metrics['total_vehicles']:,}
Average Speed: {metrics['avg_speed']:.2f} m/s ({metrics['avg_speed']*3.6:.1f} km/h)
Maximum Queue Length: {metrics['max_queue_length']:.1f} vehicles
Average Queue Length: {metrics['avg_queue_length']:.2f} vehicles
Total Waiting Time: {metrics['total_waiting_time']:.2f} seconds
Average Waiting Time: {metrics['avg_waiting_time']:.2f} seconds

PERFORMANCE SUMMARY
-------------------
Efficiency Score: {self.calculate_efficiency_score(metrics):.2f}/100
Congestion Level: {self.calculate_congestion_level(metrics)}
"""
        
        # Save report
        report_file = os.path.join(self.output_dir, f"{scenario_name}_essential_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        
        return report_file
    
    def calculate_efficiency_score(self, metrics: Dict) -> float:
        """Calculate efficiency score (0-100)"""
        score = 100.0
        
        # Penalize for delays
        if metrics['avg_waiting_time'] > 0:
            score -= min(metrics['avg_waiting_time'] * 2, 30)
        
        # Penalize for long queues
        if metrics['avg_queue_length'] > 0:
            score -= min(metrics['avg_queue_length'] * 5, 20)
        
        # Penalize for low speeds
        if metrics['avg_speed'] < 10:
            score -= min((10 - metrics['avg_speed']) * 3, 25)
        
        return max(0, min(100, score))
    
    def calculate_congestion_level(self, metrics: Dict) -> str:
        """Determine congestion level"""
        if metrics['avg_waiting_time'] < 10 and metrics['avg_queue_length'] < 2:
            return "LOW"
        elif metrics['avg_waiting_time'] < 30 and metrics['avg_queue_length'] < 5:
            return "MODERATE"
        elif metrics['avg_waiting_time'] < 60 and metrics['avg_queue_length'] < 10:
            return "HIGH"
        else:
            return "SEVERE"

def demo_essential_metrics():
    """Demonstrate essential metrics analysis"""
    print("Essential Traffic Metrics Demo")
    print("=" * 35)
    
    analyzer = EssentialTrafficAnalyzer()
    
    try:
        report_file = analyzer.generate_essential_report("essential_demo")
        print(f"✓ Essential report generated: {os.path.basename(report_file)}")
        
        metrics = analyzer.calculate_essential_metrics()
        print(f"\nEssential Summary:")
        print(f"  Total vehicles: {metrics['total_vehicles']:,}")
        print(f"  Average speed: {metrics['avg_speed']:.1f} m/s")
        print(f"  Average waiting time: {metrics['avg_waiting_time']:.1f} seconds")
        print(f"  Efficiency score: {analyzer.calculate_efficiency_score(metrics):.1f}/100")
        print(f"  Congestion level: {analyzer.calculate_congestion_level(metrics)}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Note: Requires detector output files from SUMO simulation")

if __name__ == "__main__":
    demo_essential_metrics()
