#!/usr/bin/env python3
"""
Traffic API Interface for Dashboard Integration
Provides clean API methods for dashboard to control traffic generation and collect metrics
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from generate_traffic import TrafficGenerator
from dashboard_metrics import EnhancedTrafficAnalyzer, RealTimeMetricsCollector

class TrafficAPI:
    """API interface for dashboard integration"""
    
    def __init__(self, output_dir="Sumo_env/Single intersection lhd"):
        self.output_dir = output_dir
        self.generator = TrafficGenerator(1500, 1200, output_dir)
        self.analyzer = EnhancedTrafficAnalyzer(output_dir)
        self.metrics_collector = RealTimeMetricsCollector(output_dir)
        self.current_session = None
        
    def start_session(self, session_id: str) -> Dict:
        """Start a new simulation session"""
        self.current_session = {
            'session_id': session_id,
            'start_time': datetime.now().isoformat(),
            'episodes': [],
            'status': 'active'
        }
        
        return {
            'status': 'success',
            'session_id': session_id,
            'message': f'Session {session_id} started'
        }
        
    def end_session(self) -> Dict:
        """End current simulation session"""
        if self.current_session:
            self.current_session['end_time'] = datetime.now().isoformat()
            self.current_session['status'] = 'completed'
            
            # Save session data
            session_file = os.path.join(self.output_dir, f"session_{self.current_session['session_id']}.json")
            with open(session_file, 'w') as f:
                json.dump(self.current_session, f, indent=2)
                
            return {
                'status': 'success',
                'message': f'Session {self.current_session["session_id"]} ended',
                'total_episodes': len(self.current_session['episodes'])
            }
        else:
            return {'status': 'error', 'message': 'No active session'}
            
    def generate_traffic(self, scenario: str, episode: int = 0, add_noise: bool = True, use_traci: bool = False) -> Dict:
        """Generate traffic for specified scenario"""
        try:
            self.generator.current_episode = episode
            route_file = self.generator.generate(seed=episode, scenario=scenario, add_noise=add_noise)
            
            # Create sumocfg file
            sumocfg_file = os.path.join(self.output_dir, f"{scenario}_simulation.sumocfg")
            
            with open(sumocfg_file, "w") as cfg:
                cfg.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="cross_2lanes.net.xml"/>
        <route-files value="{os.path.basename(route_file)}"/>
        <additional-files value="essential_detectors.xml,essential_traffic_lights.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="1500"/>
    </time>
    <processing>
        <lateral-resolution value="0.64"/>
    </processing>
    <report>
        <verbose value="true"/>
        <no-step-log value="true"/>
    </report>""")
                
                if use_traci:
                    cfg.write("""
    <!-- Enable TraCI for RL control -->
    <traci_server>
        <remote-port value="8813"/>
    </traci_server>""")
                
                cfg.write("\n</configuration>")
            
            return {
                'status': 'success',
                'episode': episode,
                'scenario': scenario,
                'route_file': os.path.basename(route_file),
                'sumocfg_file': os.path.basename(sumocfg_file),
                'add_noise': add_noise
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'episode': episode,
                'scenario': scenario
            }
            
    def run_simulation(self, scenario: str, episode: int = 0, use_gui: bool = False, use_traci: bool = False) -> Dict:
        """Run SUMO simulation and collect metrics"""
        try:
            # Generate traffic first
            traffic_result = self.generate_traffic(scenario, episode, add_noise=True, use_traci=use_traci)
            if traffic_result['status'] != 'success':
                return traffic_result
                
            # Run simulation
            sumocfg_file = os.path.join(self.output_dir, f"{scenario}_simulation.sumocfg")
            result = self.generator.run_simulation(sumocfg_file, use_gui)
            
            # Add to session if active
            if self.current_session:
                self.current_session['episodes'].append({
                    'episode': episode,
                    'scenario': scenario,
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                })
                
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'episode': episode,
                'scenario': scenario
            }
            
    def get_current_metrics(self) -> Dict:
        """Get current performance metrics"""
        try:
            dashboard_data = self.analyzer.get_dashboard_data(
                self.generator.current_scenario
            )
            return {
                'status': 'success',
                'data': dashboard_data
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def start_real_time_collection(self, scenario: str, episode: int = 0) -> Dict:
        """Start real-time metrics collection"""
        try:
            self.metrics_collector.start_collection(scenario, episode)
            return {
                'status': 'success',
                'message': f'Real-time collection started for {scenario} episode {episode}'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def stop_real_time_collection(self) -> Dict:
        """Stop real-time metrics collection"""
        try:
            self.metrics_collector.stop_collection()
            return {
                'status': 'success',
                'message': 'Real-time collection stopped'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def get_latest_metrics(self) -> Dict:
        """Get latest real-time metrics"""
        try:
            metrics = self.metrics_collector.get_latest_metrics()
            if metrics:
                return {
                    'status': 'success',
                    'data': {
                        'timestamp': metrics.timestamp,
                        'total_vehicles': metrics.total_vehicles,
                        'avg_speed': metrics.avg_speed,
                        'avg_waiting_time': metrics.avg_waiting_time,
                        'efficiency_score': metrics.efficiency_score,
                        'congestion_level': metrics.congestion_level,
                        'scenario': metrics.scenario,
                        'episode': metrics.episode
                    }
                }
            else:
                return {
                    'status': 'no_data',
                    'message': 'No metrics available yet'
                }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def load_baseline(self, scenario: str) -> Dict:
        """Load baseline data for comparison"""
        try:
            baseline_file = f"baseline_{scenario}.json"
            self.analyzer.load_baseline_data(baseline_file)
            return {
                'status': 'success',
                'message': f'Baseline loaded for {scenario}'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def save_baseline(self, scenario: str) -> Dict:
        """Save current performance as baseline"""
        try:
            baseline_file = f"baseline_{scenario}.json"
            baseline_data = self.analyzer.save_baseline_data(baseline_file, scenario)
            return {
                'status': 'success',
                'message': f'Baseline saved for {scenario}',
                'data': baseline_data
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def get_session_summary(self) -> Dict:
        """Get summary of current session"""
        if not self.current_session:
            return {
                'status': 'error',
                'message': 'No active session'
            }
            
        episodes = self.current_session['episodes']
        successful_episodes = [ep for ep in episodes if ep['result']['status'] == 'success']
        
        summary = {
            'session_id': self.current_session['session_id'],
            'status': self.current_session['status'],
            'start_time': self.current_session['start_time'],
            'total_episodes': len(episodes),
            'successful_episodes': len(successful_episodes),
            'scenarios_used': list(set(ep['scenario'] for ep in episodes))
        }
        
        if successful_episodes:
            avg_efficiency = sum(
                ep['result'].get('metrics', {}).get('efficiency_score', 0) 
                for ep in successful_episodes
            ) / len(successful_episodes)
            summary['average_efficiency'] = round(avg_efficiency, 2)
            
        return {
            'status': 'success',
            'data': summary
        }
        
    def get_available_scenarios(self) -> Dict:
        """Get list of available traffic scenarios"""
        return {
            'status': 'success',
            'data': {
                'scenarios': [
                    {'id': 'uniform', 'name': 'Uniform Traffic', 'description': 'Balanced traffic from all directions'},
                    {'id': 'tidal', 'name': 'Tidal Traffic', 'description': 'Heavy East-West traffic (rush hour)'},
                    {'id': 'asymmetric', 'name': 'Asymmetric Traffic', 'description': 'Heavy North+East traffic'},
                    {'id': 'random', 'name': 'Random (RL Mode)', 'description': 'Random scenario with variability'}
                ]
            }
        }
        
    def get_system_status(self) -> Dict:
        """Get overall system status"""
        return {
            'status': 'success',
            'data': {
                'generator_ready': True,
                'analyzer_ready': True,
                'collector_ready': True,
                'current_session': self.current_session is not None,
                'real_time_collecting': self.metrics_collector.is_collecting,
                'last_update': datetime.now().isoformat()
            }
        }

# Example usage for dashboard integration
def demo_api():
    """Demonstrate API usage"""
    print("Traffic API Demo")
    print("=" * 30)
    
    api = TrafficAPI()
    
    # Start session
    print("1. Starting session...")
    result = api.start_session("demo_session_001")
    print(f"   {result}")
    
    # Get available scenarios
    print("\n2. Available scenarios:")
    scenarios = api.get_available_scenarios()
    for scenario in scenarios['data']['scenarios']:
        print(f"   - {scenario['name']}: {scenario['description']}")
    
    # Generate and run traffic
    print("\n3. Running simulation...")
    result = api.run_simulation("uniform", episode=1, use_gui=False)
    print(f"   Status: {result['status']}")
    
    # Get metrics
    print("\n4. Getting metrics...")
    metrics = api.get_current_metrics()
    print(f"   Status: {metrics['status']}")
    
    # Get session summary
    print("\n5. Session summary:")
    summary = api.get_session_summary()
    print(f"   {summary}")
    
    # End session
    print("\n6. Ending session...")
    result = api.end_session()
    print(f"   {result}")

if __name__ == "__main__":
    demo_api()
