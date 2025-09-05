#!/usr/bin/env python3
"""
Traffic Environment - Gym-compatible wrapper for SUMO simulation
===============================================================

A gym-like environment that wraps the SUMO traffic simulation
for reinforcement learning training.
"""

import gym
import numpy as np
import traci
import time
import os
import subprocess
from gym import spaces
from typing import Dict, List, Any, Optional, Tuple
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.state_utils import get_12d_state_vector, get_state_summary
from utils.reward_utils import calculate_reward, reset_reward_calculator

class SUMOError(Exception):
    """Custom exception for SUMO-related errors"""
    pass

class TrafficEnv(gym.Env):
    """
    Gym-compatible traffic signal control environment
    
    This environment wraps the SUMO simulation and provides a clean
    interface for RL algorithms to interact with the traffic system.
    """
    
    def __init__(self, 
                 config_file: str = "Sumo_env/gpt_newint/intersection.sumocfg",
                 max_steps: int = 1000,
                 yellow_time: int = 3,
                 min_green: int = 10,
                 max_green: int = 60):
        super(TrafficEnv, self).__init__()
        
        # Environment parameters
        self.config_file = config_file
        self.max_steps = max_steps
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        
        # Simulation state
        self.simulation_running = False
        self.current_step = 0
        self.traffic_light_id = "C"  # Traffic light ID for our intersection
        
        # Initialize traci manager for compatibility
        from traci_manager import TraciManager
        self.traci_manager = TraciManager(config_file)
        
        # Traffic light state
        self.current_phase = 0
        self.phase_duration = 0
        self.time_in_phase = 0
        
        # Action space: 4 discrete actions (one for each phase)
        # 0: NS_Left_Straight (North-South left-turn + straight lanes)
        # 1: NS_Yellow (North-South yellow transition)
        # 2: EW_Left_Straight (East-West left-turn + straight lanes)
        # 3: EW_Yellow (East-West yellow transition)
        self.action_space = spaces.Discrete(4)
        
        # Observation space: 12-dimensional state vector
        # [north_straight_left_queue, north_right_queue, east_straight_left_queue, east_right_queue,
        #  south_straight_left_queue, south_right_queue, west_straight_left_queue, west_right_queue,
        #  phase_0, phase_1, phase_2, phase_3]
        # for 4 approaches × 2 movement types + 4 phases (12 values total)
        self.observation_space = spaces.Box(
            low=0.0, 
            high=np.inf, 
            shape=(12,),  # 12-dimensional state vector
            dtype=np.float32
        )
        
        # Phase definitions matching our SUMO configuration
        self.phases = {
            0: {"name": "NS_Left_Straight", "duration": 30, "description": "North-South left-turn + straight lanes green"},
            1: {"name": "NS_Yellow", "duration": 3, "description": "North-South yellow transition"},
            2: {"name": "EW_Left_Straight", "duration": 30, "description": "East-West left-turn + straight lanes green"},
            3: {"name": "EW_Yellow", "duration": 3, "description": "East-West yellow transition"}
        }
        
        # Episode tracking
        self.episode_reward = 0.0
        self.episode_metrics = []
        
        # Validate SUMO installation and config before proceeding
        self._validate_setup()
        
    def reset(self) -> np.ndarray:
        """
        Reset the environment for a new episode
        
        Returns:
            np.ndarray: Initial state observation
        """
        # Close any existing simulation
        if self.simulation_running:
            self.close()
        
        # Start new simulation
        self._start_simulation()
        
        # Reset episode variables
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_metrics = []
        
        # Reset reward calculator
        reset_reward_calculator()
        
        # Get initial state
        initial_state = self._get_state()
        
        return initial_state
    
    def _validate_setup(self):
        """Validate SUMO installation and configuration files"""
        try:
            # Check SUMO installation
            self._check_sumo_installation()
            
            # Validate config file
            self._validate_config_file()
            
            # Check network and route files
            self._validate_simulation_files()
            
            print("✅ SUMO setup validation completed successfully")
            
        except Exception as e:
            print(f"❌ SUMO setup validation failed: {e}")
            raise
    
    def _check_sumo_installation(self):
        """Check if SUMO is properly installed and accessible"""
        try:
            # Check SUMO version
            import subprocess
            result = subprocess.run(
                ['sumo', '--version'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode != 0:
                raise SUMOError(f"SUMO command failed with return code {result.returncode}")
            
            # Extract version information
            version_line = result.stdout.split('\n')[0]
            print(f"✅ SUMO found: {version_line}")
            
            # Check if traci module is available
            try:
                import traci
                print("✅ traci module imported successfully")
            except ImportError as e:
                raise SUMOError(f"traci module not available: {e}")
                
        except FileNotFoundError:
            raise SUMOError(
                "SUMO not found in PATH. Please install SUMO and ensure it's in your system PATH.\n"
                "Installation instructions: https://sumo.dlr.de/docs/Installing/index.html"
            )
        except subprocess.TimeoutExpired:
            raise SUMOError("SUMO version check timed out. SUMO may be corrupted or not responding.")
        except Exception as e:
            raise SUMOError(f"Unexpected error checking SUMO installation: {e}")
    
    def _validate_config_file(self):
        """Validate the SUMO configuration file"""
        if not os.path.exists(self.config_file):
            raise SUMOError(
                f"SUMO config file not found: {self.config_file}\n"
                f"Current working directory: {os.getcwd()}\n"
                f"Available files in Sumo_env/Single intersection lhd/: {os.listdir('Sumo_env/Single intersection lhd/') if os.path.exists('Sumo_env/Single intersection lhd/') else 'Directory not found'}"
            )
        
        # Check if config file is readable
        try:
            with open(self.config_file, 'r') as f:
                config_content = f.read()
                if not config_content.strip():
                    raise SUMOError(f"Config file is empty: {self.config_file}")
        except Exception as e:
            raise SUMOError(f"Cannot read config file {self.config_file}: {e}")
        
        print(f"✅ Config file validated: {self.config_file}")
    
    def _validate_simulation_files(self):
        """Validate that required simulation files exist"""
        try:
            # Parse config file to find network and route files
            import xml.etree.ElementTree as ET
            
            tree = ET.parse(self.config_file)
            root = tree.getroot()
            
            # Find input files
            input_files = []
            for input_elem in root.findall('.//input'):
                for child in input_elem:
                    if child.tag in ['net-file', 'route-files', 'additional-files']:
                        file_path = child.get('value')
                        if file_path:
                            # Handle comma-separated file lists
                            if ',' in file_path:
                                # Split by comma and handle each file
                                individual_files = [f.strip() for f in file_path.split(',')]
                                for individual_file in individual_files:
                                    if individual_file:  # Skip empty strings
                                        # Resolve relative paths
                                        config_dir = os.path.dirname(os.path.abspath(self.config_file))
                                        full_path = os.path.join(config_dir, individual_file)
                                        input_files.append((child.tag, full_path))
                            else:
                                # Single file
                                # Resolve relative paths
                                config_dir = os.path.dirname(os.path.abspath(self.config_file))
                                full_path = os.path.join(config_dir, file_path)
                                input_files.append((child.tag, full_path))
            
            # Validate each input file
            for file_type, file_path in input_files:
                if not os.path.exists(file_path):
                    raise SUMOError(f"Required {file_type} not found: {file_path}")
                print(f"✅ {file_type} validated: {os.path.basename(file_path)}")
            
            if not input_files:
                print("⚠️ No input files found in config - this may cause issues")
                
        except ET.ParseError as e:
            raise SUMOError(f"Invalid XML in config file: {e}")
        except Exception as e:
            raise SUMOError(f"Error validating simulation files: {e}")
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute an action and return the next state
        
        Args:
            action: Integer action (0-3)
            
        Returns:
            Tuple containing:
            - observation: Current state
            - reward: Reward for this step
            - done: Whether episode is finished
            - info: Additional information
        """
        # Execute action
        self._execute_action(action)
        
        # Step simulation
        traci.simulationStep()
        self.current_step += 1
        
        # Get new state
        observation = self._get_state()
        
        # Calculate reward
        reward = self._calculate_reward()
        self.episode_reward += reward
        
        # Check if episode is done
        done = self._is_done()
        
        # Collect metrics
        metrics = self._get_metrics()
        self.episode_metrics.append(metrics)
        
        # Info dictionary
        info = {
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'action': action,
            'current_phase': self.current_phase,
            'metrics': metrics
        }
        
        return observation, reward, done, info
    
    def render(self, mode='human'):
        """
        Render the environment (optional)
        
        Args:
            mode: Rendering mode ('human' or 'rgb_array')
        """
        # For now, just print current state
        if mode == 'human':
            state_summary = get_state_summary()
            phase_name = self.phases.get(self.current_phase, {}).get('name', f'Phase_{self.current_phase}')
            print(f"Step {self.current_step}: "
                  f"Phase {self.current_phase} ({phase_name}), "
                  f"Vehicles {state_summary.get('total_vehicles', 0)}, "
                  f"Queue {state_summary.get('total_queue_length', 0)}, "
                  f"Waiting {state_summary.get('total_waiting_time', 0.0):.1f}s")
    
    def get_action_name(self, action: int) -> str:
        """Get human-readable name for an action"""
        return self.phases.get(action, {}).get('name', f'Action_{action}')
    
    def get_phase_info(self, phase_id: int) -> Dict[str, Any]:
        """Get information about a specific phase"""
        return self.phases.get(phase_id, {})
    
    def close(self):
        """Close the environment and clean up resources"""
        if self.simulation_running:
            traci.close()
            self.simulation_running = False
    
    def _start_simulation(self):
        """Start the SUMO simulation"""
        try:
            # Check if config file exists
            if not os.path.exists(self.config_file):
                raise FileNotFoundError(f"Config file not found: {self.config_file}")
            
            # Start SUMO with traci
            sumo_cmd = ["sumo", "-c", self.config_file]
            traci.start(sumo_cmd)
            
            self.simulation_running = True
            print(f"Simulation started with config: {self.config_file}")
            
        except Exception as e:
            print(f"Error starting simulation: {e}")
            raise
    
    def _execute_action(self, action: int):
        """Execute the given action"""
        if not self.simulation_running:
            return
        
        # Get current traffic light state
        self.current_phase = traci.trafficlight.getPhase(self.traffic_light_id)
        self.phase_duration = traci.trafficlight.getPhaseDuration(self.traffic_light_id)
        
        # Direct phase selection - each action corresponds to a specific phase
        if action == 0:  # NS_Left_Straight
            self._set_phase(0, self.phases[0]["duration"])
        elif action == 1:  # NS_Yellow
            self._set_phase(1, self.phases[1]["duration"])
        elif action == 2:  # EW_Left_Straight
            self._set_phase(2, self.phases[2]["duration"])
        elif action == 3:  # EW_Yellow
            self._set_phase(3, self.phases[3]["duration"])
    
    def _set_phase(self, phase_id: int, duration: float):
        """Set traffic light phase"""
        if not self.simulation_running:
            return
        
        try:
            traci.trafficlight.setPhase(self.traffic_light_id, phase_id)
            traci.trafficlight.setPhaseDuration(self.traffic_light_id, duration)
        except Exception as e:
            print(f"Error setting phase: {e}")
    
    
    def _get_state(self) -> np.ndarray:
        """Get current state observation"""
        try:
            # Get 12-dimensional state vector from state utils
            state_vector = get_12d_state_vector()
            return state_vector
        except Exception as e:
            print(f"Error getting state: {e}")
            # Return zero state if error occurs
            return np.zeros(12, dtype=np.float32)
    
    def _calculate_reward(self) -> float:
        """Calculate reward for current state"""
        try:
            # Get state summary for reward calculation
            state_summary = get_state_summary()
            
            current_waiting_time = state_summary.get('total_waiting_time', 0.0)
            current_queue_length = state_summary.get('total_queue_length', 0)
            avg_speed = state_summary.get('avg_speed', 0.0)
            vehicle_count = state_summary.get('total_vehicles', 0)
            
            # Calculate reward using reward utils
            reward = calculate_reward(
                current_waiting_time,
                current_queue_length,
                avg_speed,
                vehicle_count
            )
            
            return reward
            
        except Exception as e:
            print(f"Error calculating reward: {e}")
            return 0.0
    
    def _is_done(self) -> bool:
        """Check if episode is done"""
        return (
            self.current_step >= self.max_steps or
            not self.simulation_running or
            not traci.simulation.getMinExpectedNumber() > 0
        )
    
    def _get_metrics(self) -> Dict[str, Any]:
        """Get current episode metrics"""
        try:
            state_summary = get_state_summary()
            
            return {
                'step': self.current_step,
                'total_vehicles': state_summary.get('total_vehicles', 0),
                'total_queue_length': state_summary.get('total_queue_length', 0),
                'total_waiting_time': state_summary.get('total_waiting_time', 0.0),
                'avg_speed': state_summary.get('avg_speed', 0.0),
                'current_phase': self.current_phase,
                'episode_reward': self.episode_reward
            }
        except Exception as e:
            print(f"Error getting metrics: {e}")
            return {}
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of completed episode"""
        if not self.episode_metrics:
            return {}
        
        # Calculate episode statistics
        total_vehicles = sum(m.get('total_vehicles', 0) for m in self.episode_metrics)
        avg_queue_length = np.mean([m.get('total_queue_length', 0) for m in self.episode_metrics])
        avg_waiting_time = np.mean([m.get('total_waiting_time', 0.0) for m in self.episode_metrics])
        avg_speed = np.mean([m.get('avg_speed', 0.0) for m in self.episode_metrics])
        
        return {
            'episode_length': self.current_step,
            'total_reward': self.episode_reward,
            'total_vehicles': total_vehicles,
            'avg_queue_length': avg_queue_length,
            'avg_waiting_time': avg_waiting_time,
            'avg_speed': avg_speed,
            'metrics': self.episode_metrics
        }

# Example usage
if __name__ == "__main__":
    # Create environment
    env = TrafficEnv()
    
    # Test reset
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    # Test a few steps
    for i in range(10):
        action = env.action_space.sample()  # Random action
        state, reward, done, info = env.step(action)
        action_name = env.get_action_name(action)
        print(f"Step {i+1}: Action {action} ({action_name}), Reward {reward:.3f}, Done {done}")
        
        if done:
            break
    
    # Get episode summary
    summary = env.get_episode_summary()
    print(f"Episode summary: {summary}")
    
    env.close()
