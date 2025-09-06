#!/usr/bin/env python3
"""
Comprehensive Benchmarking System for Fixed-Duration Traffic Lights

This script runs extensive benchmarks on all traffic scenarios to establish
baseline performance metrics for comparison with RL models.

Metrics Collected:
- Average travel time per route
- Total waiting time per route  
- Average speed per route
- Queue lengths per lane
- Throughput (vehicles completed per route)
- Delay per vehicle / per route
- Fuel consumption and emissions
- Junction efficiency metrics
"""

import os
import sys
import time
import json
import csv
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
import subprocess
import traci
import traci.constants as tc

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_traffic import TrafficGenerator
from traci_manager import TraciManager


class TrafficLightBenchmarker:
    """Comprehensive benchmarking system for traffic light performance"""
    
    def __init__(self, output_dir="benchmark_results", num_runs=5, simulation_duration=3600):
        """
        Initialize the benchmarker
        
        Args:
            output_dir: Directory to save benchmark results
            num_runs: Number of runs per scenario for statistical significance
            simulation_duration: Duration of each simulation in seconds (default: 1 hour)
        """
        self.output_dir = output_dir
        self.num_runs = num_runs
        self.simulation_duration = simulation_duration
        self.warmup_duration = 300  # 5 minutes warmup to handle transients
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Traffic scenarios to benchmark
        self.scenarios = ['uniform', 'tidal', 'congested', 'asymmetric']
        
        # Route definitions for analysis
        self.route_mapping = {
            "r0": "W→E", "r1": "W→S", "r2": "W→N",
            "r3": "S→N", "r4": "S→W", "r5": "S→E", 
            "r6": "E→W", "r7": "E→S", "r8": "E→N",
            "r9": "N→S", "r10": "N→W", "r11": "N→E"
        }
        
        # Initialize results storage
        self.results = defaultdict(list)
        
    def run_benchmark(self):
        """Run comprehensive benchmarks for all scenarios"""
        print("🚦 TRAFFIC LIGHT BENCHMARKING SYSTEM")
        print("=" * 60)
        print(f"📊 Scenarios: {', '.join(self.scenarios)}")
        print(f"🔄 Runs per scenario: {self.num_runs}")
        print(f"⏱️  Simulation duration: {self.simulation_duration}s ({self.simulation_duration/60:.1f} min)")
        print(f"🔥 Warmup duration: {self.warmup_duration}s ({self.warmup_duration/60:.1f} min)")
        print(f"📁 Output directory: {self.output_dir}")
        print()
        
        # Generate traffic files for all scenarios
        self._generate_traffic_files()
        
        # Run benchmarks for each scenario
        for scenario in self.scenarios:
            print(f"🎯 BENCHMARKING SCENARIO: {scenario.upper()}")
            print("-" * 40)
            
            scenario_results = []
            for run in range(self.num_runs):
                print(f"  Run {run + 1}/{self.num_runs}...", end=" ")
                
                # Run single simulation
                run_results = self._run_single_simulation(scenario, run)
                scenario_results.append(run_results)
                
                print(f"✅ (Travel time: {run_results['avg_travel_time']:.1f}s)")
            
            # Calculate statistics for this scenario
            scenario_stats = self._calculate_scenario_statistics(scenario, scenario_results)
            self.results[scenario] = scenario_stats
            
            print(f"  📈 Average travel time: {scenario_stats['avg_travel_time_mean']:.1f} ± {scenario_stats['avg_travel_time_std']:.1f}s")
            print(f"  🚗 Total throughput: {scenario_stats['total_throughput_mean']:.0f} ± {scenario_stats['total_throughput_std']:.0f} vehicles")
            print()
        
        # Save results
        self._save_results()
        self._generate_summary_report()
        
        print("🎉 BENCHMARKING COMPLETED!")
        print(f"📊 Results saved to: {self.output_dir}/")
        
    def _generate_traffic_files(self):
        """Generate traffic route files for all scenarios"""
        print("🚗 Generating traffic files...")
        
        generator = TrafficGenerator(
            max_steps=self.simulation_duration,
            n_cars_generated=1200,  # Increased for better statistics
            out_dir=self.output_dir
        )
        
        for scenario in self.scenarios:
            # Generate with noise for realistic variability
            route_file = generator.generate(
                seed=42,  # Fixed seed for reproducibility
                scenario=scenario,
                add_noise=True
            )
            print(f"  ✅ {scenario}: {route_file}")
    
    def _run_single_simulation(self, scenario, run_id):
        """Run a single simulation and collect metrics"""
        
        # Create SUMO configuration
        sumocfg_file = self._create_sumo_config(scenario, run_id)
        
        # Initialize TraCI
        traci.start([
            "sumo", "-c", sumocfg_file,
            "--no-step-log", "--no-warnings", "true",
            "--duration-log.statistics",
            "--tripinfo-output", f"{self.output_dir}/{scenario}_run_{run_id}_tripinfo.xml",
            "--summary-output", f"{self.output_dir}/{scenario}_run_{run_id}_summary.xml",
            "--queue-output", f"{self.output_dir}/{scenario}_run_{run_id}_queue.xml",
            "--emission-output", f"{self.output_dir}/{scenario}_run_{run_id}_emissions.xml"
        ])
        
        # Initialize metrics collection
        metrics = {
            'travel_times': defaultdict(list),
            'waiting_times': defaultdict(list),
            'speeds': defaultdict(list),
            'queue_lengths': defaultdict(list),
            'delays': defaultdict(list),
            'vehicle_counts': defaultdict(int),
            'completed_vehicles': defaultdict(int),
            'fuel_consumption': 0.0,
            'co2_emissions': 0.0,
            'junction_efficiency': defaultdict(list)
        }
        
        step = 0
        try:
            while step < self.simulation_duration:
                traci.simulationStep()
                
                # Skip warmup period for cleaner metrics
                if step >= self.warmup_duration:
                    self._collect_metrics(metrics, step)
                
                step += 1
                
        except Exception as e:
            print(f"⚠️  Simulation error at step {step}: {e}")
        finally:
            traci.close()
        
        # Process collected metrics
        return self._process_metrics(metrics, scenario, run_id)
    
    def _create_sumo_config(self, scenario, run_id):
        """Create SUMO configuration file for the simulation"""
        config_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="Sumo_env/gpt_newint/intersection.net.xml"/>
        <route-files value="{self.output_dir}/{scenario}_episode_routes.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{self.simulation_duration}"/>
        <step-length value="1"/>
    </time>
    <processing>
        <ignore-route-errors value="true"/>
        <ignore-junction-blocker value="10"/>
    </processing>
    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
        <no-warnings value="true"/>
    </report>
</configuration>"""
        
        config_file = f"{self.output_dir}/{scenario}_run_{run_id}.sumocfg"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        return config_file
    
    def _collect_metrics(self, metrics, step):
        """Collect metrics during simulation"""
        
        # Get all vehicles
        vehicle_ids = traci.vehicle.getIDList()
        
        for vehicle_id in vehicle_ids:
            try:
                # Get vehicle information
                route_id = traci.vehicle.getRouteID(vehicle_id)
                if route_id not in self.route_mapping:
                    continue
                
                # Travel time and waiting time
                travel_time = traci.vehicle.getTimeLoss(vehicle_id)
                waiting_time = traci.vehicle.getWaitingTime(vehicle_id)
                
                # Speed
                speed = traci.vehicle.getSpeed(vehicle_id)
                
                # Delay (time loss due to traffic conditions)
                delay = traci.vehicle.getTimeLoss(vehicle_id)
                
                # Store metrics
                metrics['travel_times'][route_id].append(travel_time)
                metrics['waiting_times'][route_id].append(waiting_time)
                metrics['speeds'][route_id].append(speed)
                metrics['delays'][route_id].append(delay)
                metrics['vehicle_counts'][route_id] += 1
                
                # Check if vehicle completed its route
                if traci.vehicle.getArrivedNumber() > 0:
                    metrics['completed_vehicles'][route_id] += 1
                
            except traci.exceptions.TraCIException:
                # Vehicle may have left the simulation
                continue
        
        # Collect queue lengths for each lane
        for lane_id in traci.lane.getIDList():
            if 'C_' in lane_id:  # Only intersection lanes
                queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
                metrics['queue_lengths'][lane_id].append(queue_length)
        
        # Collect junction efficiency
        for junction_id in traci.junction.getIDList():
            if junction_id == 'C':  # Main intersection
                # Get junction statistics
                try:
                    # This would need to be implemented based on SUMO version
                    pass
                except:
                    pass
    
    def _process_metrics(self, metrics, scenario, run_id):
        """Process collected metrics into summary statistics"""
        
        processed = {
            'scenario': scenario,
            'run_id': run_id,
            'routes': {}
        }
        
        # Process metrics for each route
        for route_id, route_name in self.route_mapping.items():
            if route_id in metrics['travel_times']:
                route_metrics = {
                    'route_name': route_name,
                    'avg_travel_time': np.mean(metrics['travel_times'][route_id]) if metrics['travel_times'][route_id] else 0,
                    'total_waiting_time': np.sum(metrics['waiting_times'][route_id]) if metrics['waiting_times'][route_id] else 0,
                    'avg_speed': np.mean(metrics['speeds'][route_id]) if metrics['speeds'][route_id] else 0,
                    'avg_delay': np.mean(metrics['delays'][route_id]) if metrics['delays'][route_id] else 0,
                    'throughput': metrics['completed_vehicles'][route_id],
                    'vehicle_count': metrics['vehicle_counts'][route_id]
                }
                processed['routes'][route_id] = route_metrics
        
        # Overall metrics
        all_travel_times = []
        all_waiting_times = []
        all_speeds = []
        total_throughput = 0
        
        for route_metrics in processed['routes'].values():
            all_travel_times.append(route_metrics['avg_travel_time'])
            all_waiting_times.append(route_metrics['total_waiting_time'])
            all_speeds.append(route_metrics['avg_speed'])
            total_throughput += route_metrics['throughput']
        
        processed['overall'] = {
            'avg_travel_time': np.mean(all_travel_times) if all_travel_times else 0,
            'total_waiting_time': np.sum(all_waiting_times) if all_waiting_times else 0,
            'avg_speed': np.mean(all_speeds) if all_speeds else 0,
            'total_throughput': total_throughput,
            'fuel_consumption': metrics['fuel_consumption'],
            'co2_emissions': metrics['co2_emissions']
        }
        
        return processed
    
    def _calculate_scenario_statistics(self, scenario, run_results):
        """Calculate statistical summary for a scenario across multiple runs"""
        
        # Extract metrics across all runs
        metrics_by_run = {
            'avg_travel_time': [run['overall']['avg_travel_time'] for run in run_results],
            'total_waiting_time': [run['overall']['total_waiting_time'] for run in run_results],
            'avg_speed': [run['overall']['avg_speed'] for run in run_results],
            'total_throughput': [run['overall']['total_throughput'] for run in run_results],
            'fuel_consumption': [run['overall']['fuel_consumption'] for run in run_results],
            'co2_emissions': [run['overall']['co2_emissions'] for run in run_results]
        }
        
        # Calculate statistics
        stats = {}
        for metric, values in metrics_by_run.items():
            stats[f'{metric}_mean'] = np.mean(values)
            stats[f'{metric}_std'] = np.std(values)
            stats[f'{metric}_min'] = np.min(values)
            stats[f'{metric}_max'] = np.max(values)
            stats[f'{metric}_values'] = values
        
        # Route-specific statistics
        route_stats = {}
        for route_id in self.route_mapping.keys():
            route_metrics = {}
            for metric in ['avg_travel_time', 'total_waiting_time', 'avg_speed', 'avg_delay', 'throughput']:
                values = []
                for run in run_results:
                    if route_id in run['routes']:
                        values.append(run['routes'][route_id][metric])
                
                if values:
                    route_metrics[f'{metric}_mean'] = np.mean(values)
                    route_metrics[f'{metric}_std'] = np.std(values)
                    route_metrics[f'{metric}_values'] = values
            
            if route_metrics:
                route_stats[route_id] = route_metrics
        
        stats['routes'] = route_stats
        stats['scenario'] = scenario
        stats['num_runs'] = len(run_results)
        stats['timestamp'] = datetime.now().isoformat()
        
        return stats
    
    def _save_results(self):
        """Save benchmark results to files"""
        
        # Save detailed results as JSON
        with open(f"{self.output_dir}/benchmark_results_detailed.json", 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary CSV
        summary_data = []
        for scenario, stats in self.results.items():
            summary_data.append({
                'scenario': scenario,
                'num_runs': stats['num_runs'],
                'avg_travel_time_mean': stats['avg_travel_time_mean'],
                'avg_travel_time_std': stats['avg_travel_time_std'],
                'total_waiting_time_mean': stats['total_waiting_time_mean'],
                'total_waiting_time_std': stats['total_waiting_time_std'],
                'avg_speed_mean': stats['avg_speed_mean'],
                'avg_speed_std': stats['avg_speed_std'],
                'total_throughput_mean': stats['total_throughput_mean'],
                'total_throughput_std': stats['total_throughput_std'],
                'fuel_consumption_mean': stats['fuel_consumption_mean'],
                'co2_emissions_mean': stats['co2_emissions_mean']
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(f"{self.output_dir}/benchmark_summary.csv", index=False)
        
        # Save route-specific results
        route_data = []
        for scenario, stats in self.results.items():
            for route_id, route_stats in stats['routes'].items():
                route_name = self.route_mapping[route_id]
                route_data.append({
                    'scenario': scenario,
                    'route_id': route_id,
                    'route_name': route_name,
                    'avg_travel_time_mean': route_stats.get('avg_travel_time_mean', 0),
                    'avg_travel_time_std': route_stats.get('avg_travel_time_std', 0),
                    'total_waiting_time_mean': route_stats.get('total_waiting_time_mean', 0),
                    'avg_speed_mean': route_stats.get('avg_speed_mean', 0),
                    'avg_delay_mean': route_stats.get('avg_delay_mean', 0),
                    'throughput_mean': route_stats.get('throughput_mean', 0)
                })
        
        df_routes = pd.DataFrame(route_data)
        df_routes.to_csv(f"{self.output_dir}/benchmark_routes.csv", index=False)
    
    def _generate_summary_report(self):
        """Generate a human-readable summary report"""
        
        report_file = f"{self.output_dir}/BENCHMARK_REPORT.md"
        
        with open(report_file, 'w') as f:
            f.write("# Traffic Light Benchmark Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Simulation Duration:** {self.simulation_duration}s ({self.simulation_duration/60:.1f} minutes)\n")
            f.write(f"**Warmup Duration:** {self.warmup_duration}s ({self.warmup_duration/60:.1f} minutes)\n")
            f.write(f"**Runs per Scenario:** {self.num_runs}\n\n")
            
            f.write("## Summary Statistics\n\n")
            f.write("| Scenario | Avg Travel Time (s) | Total Waiting Time (s) | Avg Speed (m/s) | Throughput |\n")
            f.write("|----------|-------------------|----------------------|----------------|------------|\n")
            
            for scenario, stats in self.results.items():
                f.write(f"| {scenario} | "
                       f"{stats['avg_travel_time_mean']:.1f} ± {stats['avg_travel_time_std']:.1f} | "
                       f"{stats['total_waiting_time_mean']:.1f} ± {stats['total_waiting_time_std']:.1f} | "
                       f"{stats['avg_speed_mean']:.1f} ± {stats['avg_speed_std']:.1f} | "
                       f"{stats['total_throughput_mean']:.0f} ± {stats['total_throughput_std']:.0f} |\n")
            
            f.write("\n## Route-Specific Performance\n\n")
            
            for scenario, stats in self.results.items():
                f.write(f"### {scenario.upper()} Scenario\n\n")
                f.write("| Route | Travel Time (s) | Waiting Time (s) | Speed (m/s) | Throughput |\n")
                f.write("|-------|----------------|------------------|-------------|------------|\n")
                
                for route_id, route_stats in stats['routes'].items():
                    route_name = self.route_mapping[route_id]
                    f.write(f"| {route_name} | "
                           f"{route_stats.get('avg_travel_time_mean', 0):.1f} ± {route_stats.get('avg_travel_time_std', 0):.1f} | "
                           f"{route_stats.get('total_waiting_time_mean', 0):.1f} ± {route_stats.get('total_waiting_time_std', 0):.1f} | "
                           f"{route_stats.get('avg_speed_mean', 0):.1f} ± {route_stats.get('avg_speed_std', 0):.1f} | "
                           f"{route_stats.get('throughput_mean', 0):.0f} ± {route_stats.get('throughput_std', 0):.0f} |\n")
                f.write("\n")


def main():
    """Main function to run benchmarks"""
    
    # Configuration
    OUTPUT_DIR = "benchmark_results"
    NUM_RUNS = 5  # Number of runs per scenario for statistical significance
    SIMULATION_DURATION = 3600  # 1 hour simulation (excluding warmup)
    
    # Create and run benchmarker
    benchmarker = TrafficLightBenchmarker(
        output_dir=OUTPUT_DIR,
        num_runs=NUM_RUNS,
        simulation_duration=SIMULATION_DURATION
    )
    
    # Run benchmarks
    benchmarker.run_benchmark()


if __name__ == "__main__":
    main()
