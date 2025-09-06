import numpy as np
import math
import os
import sys
import subprocess
import time
import random
import json
from datetime import datetime
from typing import Dict, Optional, Tuple

class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated, out_dir="Sumo_env/gpt_newint"):
        self._n_cars_generated = n_cars_generated
        self._max_steps = max_steps
        self._out_dir = out_dir
        self.current_episode = 0
        self.current_scenario = "unknown"
        self.metrics_history = []
        os.makedirs(out_dir, exist_ok=True)
        
        # RL Training Mode: Increase vehicle count for better challenges
        if n_cars_generated > 1500:
            print(f"⚠️  Reducing vehicle count from {n_cars_generated} to 1500 for RL training")
            self._n_cars_generated = 1500
        elif n_cars_generated < 1200:
            print(f"🚀  Increasing vehicle count to 1200 for better RL challenges")
            self._n_cars_generated = 1200

    def _generate_depart_times(self, seed):
        """Generate stochastic vehicle departure times using a Weibull distribution."""
        np.random.seed(seed)
        random.seed(seed)  # For full reproducibility
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        min_old, max_old = math.floor(timings[1]), math.ceil(timings[-1])
        car_gen_steps = []
        for value in timings:
            # Linear normalization: spread departures smoothly over [0, max_steps]
            step = int(((value - min_old) / (max_old - min_old)) * self._max_steps)
            car_gen_steps.append(step)
        return car_gen_steps

    def generate(self, seed, scenario=None, add_noise=True):
        """
        Generate route file for a given scenario:
        - uniform: balanced traffic from all directions
        - tidal: heavy E-W, light N-S
        - asymmetric: heavy North+East, light South+West
        
        If scenario is None, randomly sample one with equal probability.
        If add_noise is True, add random scaling to route weights for variability.
        """
        # Balanced scenario sampling
        if scenario is None:
            scenario = random.choice(["uniform", "tidal", "asymmetric", "congested"])
        
        self.current_scenario = scenario
        car_gen_steps = self._generate_depart_times(seed)
        filename = os.path.join(self._out_dir, f"{scenario}_episode_routes.rou.xml")

        with open(filename, "w") as routes:
            # vehicle types - compatible with the existing environment
            print("""<routes>
    <vTypeDistribution id="mixed">
        <vType id="car" vClass="passenger" speedDev="0.2" latAlignment="center" probability="0.6"
               lcStrategic="1.0" lcCooperative="1.0" lcSpeedGain="1.0" lcKeepRight="1.0"
               lcLookaheadLeft="2.0" lcLookaheadRight="2.0" lcSpeedGainRight="0.1"
               lcPushy="0.0" lcPushyGap="0.0" lcAssertive="1.0" lcImpatience="0.0"
               lcTimeToImpatience="1.0" lcAccelLat="1.0" lcMaxSpeedLat="1.0"/>
        <vType id="moped" vClass="moped" speedDev="0.4" latAlignment="center" probability="0.4"
               lcStrategic="1.0" lcCooperative="1.0" lcSpeedGain="1.0" lcKeepRight="1.0"
               lcLookaheadLeft="2.0" lcLookaheadRight="2.0" lcSpeedGainRight="0.1"
               lcPushy="0.0" lcPushyGap="0.0" lcAssertive="1.0" lcImpatience="0.0"
               lcTimeToImpatience="1.0" lcAccelLat="1.0" lcMaxSpeedLat="1.0"/>
    </vTypeDistribution>""", file=routes)

            # define routes - compatible with gpt_newint network structure
            routes_def = {
                "r0": "W_in C_E",  # West to East
                "r1": "W_in C_S",  # West to South
                "r2": "W_in C_N",  # West to North
                "r3": "S_in C_N",  # South to North
                "r4": "S_in C_W",  # South to West
                "r5": "S_in C_E",  # South to East
                "r6": "E_in C_W",  # East to West
                "r7": "E_in C_S",  # East to South
                "r8": "E_in C_N",  # East to North
                "r9": "N_in C_S",  # North to South
                "r10": "N_in C_W", # North to West
                "r11": "N_in C_E"  # North to East
            }
            
            for rid, edges in routes_def.items():
                print(f'    <route id="{rid}" edges="{edges}"/>', file=routes)

            # scenario-specific route weights (enhanced for RL challenges)
            if scenario == "uniform":
                route_weights = [1.0] * 12  # Equal weights for all routes
            elif scenario == "tidal":
                # Heavy East-West traffic (more extreme for RL challenges)
                route_weights = [0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1]
            elif scenario == "asymmetric":
                # Heavy North+East traffic (more extreme for RL challenges)
                route_weights = [0.1, 0.1, 0.4, 0.4, 0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4]
            elif scenario == "congested":
                # High congestion scenario for RL training
                route_weights = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
            else:
                route_weights = [1.0] * 12

            # Traffic Variability: Add independent noise per route for true variability
            if add_noise:
                route_weights = [w * np.random.uniform(0.8, 1.2) for w in route_weights]
            
            # Lane mapping for proper lane usage (prevents last-minute lane changes)
            # Each route should use the appropriate lane for its movement type
            lane_mapping = {
                "r0": "0",   # W→E (right turn from rightmost lane)
                "r1": "1",   # W→S (straight from middle lane)
                "r2": "2",   # W→N (straight from middle lane)
                "r3": "0",   # S→N (right turn from rightmost lane)
                "r4": "1",   # S→W (straight from middle lane)
                "r5": "2",   # S→E (straight from middle lane)
                "r6": "0",   # E→W (right turn from rightmost lane)
                "r7": "1",   # E→S (straight from middle lane)
                "r8": "2",   # E→N (straight from middle lane)
                "r9": "0",   # N→S (right turn from rightmost lane)
                "r10": "1",  # N→W (straight from middle lane)
                "r11": "2"   # N→E (straight from middle lane)
            }
            
            # Generate vehicles
            for i, step in enumerate(car_gen_steps):
                route = random.choices(list(routes_def.keys()), weights=route_weights, k=1)[0]
                lane = lane_mapping[route]
                print(f'    <vehicle id="{route}_{i}" type="mixed" route="{route}" depart="{step}" departLane="{lane}" departPos="0"/>', file=routes)

            print("</routes>", file=routes)

        return filename

    def generate_for_rl_training(self, episode: int) -> Tuple[str, str]:
        """
        Generate traffic for RL training with balanced scenario sampling and variability.
        Returns (route_file_path, scenario_name)
        """
        self.current_episode = episode
        route_file = self.generate(seed=episode, scenario=None, add_noise=True)
        return route_file, self.current_scenario

    def run_simulation(self, sumocfg_file: str, use_gui: bool = True) -> Dict:
        """
        Run SUMO simulation and return performance metrics.
        Enhanced for dashboard compatibility.
        """
        try:
            # Determine SUMO command
            sumo_cmd = "sumo-gui" if use_gui else "sumo"
            
            # Run SUMO with absolute path for safer execution
            cmd = [sumo_cmd, "-c", os.path.abspath(sumocfg_file)]
            print(f"Running SUMO with command: {' '.join(cmd)}")
            
            # Run simulation
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("Simulation completed successfully!")
                
                # Collect metrics after simulation
                metrics = self._collect_simulation_metrics()
                self.metrics_history.append({
                    'episode': self.current_episode,
                    'scenario': self.current_scenario,
                    'timestamp': datetime.now().isoformat(),
                    'metrics': metrics
                })
                
                return {
                    'status': 'success',
                    'episode': self.current_episode,
                    'scenario': self.current_scenario,
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                print(f"Simulation failed with return code: {result.returncode}")
                return {
                    'status': 'failed',
                    'error': result.stderr,
                    'episode': self.current_episode,
                    'scenario': self.current_scenario
                }
                
        except subprocess.TimeoutExpired:
            print("Simulation timed out")
            return {
                'status': 'timeout',
                'episode': self.current_episode,
                'scenario': self.current_scenario
            }
        except Exception as e:
            print(f"Error running simulation: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'episode': self.current_episode,
                'scenario': self.current_scenario
            }

    def _collect_simulation_metrics(self) -> Dict:
        """
        Collect performance metrics from simulation output.
        Includes both dashboard-compatible and basic built-in metrics.
        """
        try:
            # Try dashboard metrics first
            from dashboard_metrics import EnhancedTrafficAnalyzer
            
            analyzer = EnhancedTrafficAnalyzer(self._out_dir)
            metrics = analyzer.calculate_essential_metrics()
            
            # Add episode-specific data
            metrics.update({
                'episode': self.current_episode,
                'scenario': self.current_scenario,
                'efficiency_score': analyzer.calculate_efficiency_score(metrics),
                'congestion_level': analyzer.calculate_congestion_level(metrics)
            })
            
            return metrics
            
        except ImportError:
            # Fallback: collect basic metrics from SUMO output files
            return self._collect_basic_metrics()
        except Exception as e:
            print(f"Error collecting dashboard metrics: {e}")
            return self._collect_basic_metrics()

    def _collect_basic_metrics(self) -> Dict:
        """
        Collect basic performance metrics from SUMO output files.
        Provides essential metrics without dashboard dependency.
        """
        try:
            metrics = {
                'episode': self.current_episode,
                'scenario': self.current_scenario,
                'total_vehicles': self._n_cars_generated,
                'simulation_duration': self._max_steps,
                'status': 'basic_metrics'
            }
            
            # Try to collect tripinfo data if available
            tripinfo_files = [f for f in os.listdir(self._out_dir) if f.endswith('.tripinfo.xml')]
            if tripinfo_files:
                # Basic tripinfo parsing (can be enhanced)
                metrics['tripinfo_available'] = True
                metrics['tripinfo_files'] = len(tripinfo_files)
            else:
                metrics['tripinfo_available'] = False
            
            # Calculate basic efficiency score based on vehicle count and duration
            efficiency_score = min(100, (self._n_cars_generated / self._max_steps) * 100)
            metrics['efficiency_score'] = efficiency_score
            
            # Basic congestion level estimation
            if self._n_cars_generated > 1000:
                congestion_level = "high"
            elif self._n_cars_generated > 600:
                congestion_level = "medium"
            else:
                congestion_level = "low"
            metrics['congestion_level'] = congestion_level
            
            return metrics
            
        except Exception as e:
            print(f"Error collecting basic metrics: {e}")
            return {
                'episode': self.current_episode,
                'scenario': self.current_scenario,
                'status': 'metrics_error',
                'error': str(e)
            }

    def get_dashboard_data(self) -> Dict:
        """
        Get current state data for dashboard integration.
        """
        return {
            'current_episode': self.current_episode,
            'current_scenario': self.current_scenario,
            'metrics_history': self.metrics_history,
            'total_episodes': len(self.metrics_history),
            'last_update': datetime.now().isoformat()
        }

    def save_episode_data(self, episode_data: Dict, filename: Optional[str] = None):
        """
        Save episode data for dashboard analysis.
        """
        if filename is None:
            filename = f"episode_{self.current_episode}_{self.current_scenario}.json"
        
        filepath = os.path.join(self._out_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(episode_data, f, indent=2)
            print(f"Episode data saved to {filepath}")
        except Exception as e:
            print(f"Error saving episode data: {e}")

def main():
    """Enhanced main function with dashboard compatibility"""
    print("SUMO Traffic Generator - Dashboard Compatible Mode")
    print("=" * 50)
    
    # Configuration
    max_steps = 1500
    n_cars = 1200
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "Sumo_env/gpt_newint")
    
    # Create generator
    generator = TrafficGenerator(max_steps, n_cars, out_dir)
    
    try:
        # Display scenario menu
        print("\nAvailable Traffic Scenarios:")
        print("1. Uniform - Balanced traffic from all directions")
        print("2. Tidal - Heavy East-West traffic, light North-South")
        print("3. Asymmetric - Heavy North+East traffic, light South+West")
        print("4. Congested - High congestion scenario for RL training")
        print("5. Random (RL Training Mode) - Random scenario selection")
        print()
        
        # Get user input with fallbacks for non-interactive use
        try:
            scenario_choice = input("Select scenario (1-4) or press Enter for uniform: ").strip()
        except EOFError:
            scenario_choice = "1"
        
        try:
            add_noise_input = input("Add traffic variability (noise)? (y/n, default: y): ").strip().lower()
        except EOFError:
            add_noise_input = "y"
        
        try:
            use_gui_input = input("Run with GUI? (y/n, default: y): ").strip().lower()
        except EOFError:
            use_gui_input = "y"
        
        # Process inputs
        if scenario_choice == "1" or scenario_choice == "":
            scenario = "uniform"
        elif scenario_choice == "2":
            scenario = "tidal"
        elif scenario_choice == "3":
            scenario = "asymmetric"
        elif scenario_choice == "4":
            scenario = "congested"
        elif scenario_choice == "5":
            scenario = None  # Random RL mode
        else:
            scenario = "uniform"
        
        add_noise = add_noise_input in ["y", "yes", ""]
        use_gui = use_gui_input in ["y", "yes", ""]
        
        print(f"\nGenerating traffic for scenario: {scenario if scenario else 'random'}")
        print(f"Traffic variability: {'Yes' if add_noise else 'No'}")
        print(f"GUI mode: {'Yes' if use_gui else 'No'}")
        print(f"Total vehicles: {n_cars}")
        print(f"Simulation duration: {max_steps} steps ({max_steps/60:.1f} minutes)")
        
        # Generate traffic
        route_file = generator.generate(seed=42, scenario=scenario, add_noise=add_noise)
        print(f"Generated {os.path.basename(route_file)}")
        
        # Create sumocfg file
        scenario_name = scenario if scenario is not None else "random"
        sumocfg_file = os.path.join(out_dir, f"{scenario_name}_simulation.sumocfg")
        
        # Ask user if they want TraCI (for RL control) or regular simulation
        try:
            use_traci_input = input("Enable TraCI for RL control? (y/n, default: n): ").strip().lower()
        except EOFError:
            use_traci_input = "n"
        
        use_traci = use_traci_input in ["y", "yes"]
        
        with open(sumocfg_file, "w") as cfg:
            cfg.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="intersection.net.xml"/>
        <route-files value="{os.path.basename(route_file)}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{max_steps}"/>
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
        
        # Run simulation with metrics collection
        print(f"Running SUMO with command: sumo-gui -c {os.path.basename(sumocfg_file)}")
        result = generator.run_simulation(sumocfg_file, use_gui)
        
        # Save episode data for dashboard
        if result['status'] == 'success':
            generator.save_episode_data(result)
            
            # Print dashboard-compatible summary
            print("\n" + "=" * 50)
            print("DASHBOARD-READY SUMMARY")
            print("=" * 50)
            print(f"Episode: {result['episode']}")
            print(f"Scenario: {result['scenario']}")
            print(f"Status: {result['status']}")
            print(f"Timestamp: {result['timestamp']}")
            
            if 'metrics' in result:
                metrics = result['metrics']
                print(f"Total Vehicles: {metrics.get('total_vehicles', 'N/A')}")
                print(f"Efficiency Score: {metrics.get('efficiency_score', 'N/A'):.1f}/100")
                print(f"Congestion Level: {metrics.get('congestion_level', 'N/A')}")
        
        return result
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        return {'status': 'interrupted'}
    except Exception as e:
        print(f"Error: {e}")
        return {'status': 'error', 'error': str(e)}

if __name__ == "__main__":
    main()
