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
from gym import spaces
from typing import Dict, List, Any, Optional, Tuple
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.state_utils import get_24d_state_vector, get_state_summary
from utils.reward_utils import calculate_reward, reset_reward_calculator

class TrafficEnv(gym.Env):
    """
    Gym-compatible traffic signal control environment
    
    This environment wraps the SUMO simulation and provides a clean
    interface for RL algorithms to interact with the traffic system.
    """
    
    def __init__(self, 
                 config_file: str = "Sumo_env/Single intersection lhd/uniform_simulation.sumocfg",
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
        self.traffic_light_id = "0"  # Default traffic light ID
        
        # Traffic light state
        self.current_phase = 0
        self.phase_duration = 0
        self.time_in_phase = 0
        
        # Action space: 4 discrete actions
        # 0: NS Green (North-South)
        # 1: EW Green (East-West)  
        # 2: Extend current phase
        # 3: Skip to next phase
        self.action_space = spaces.Discrete(4)
        
        # Observation space: 24-dimensional state vector
        # [queue_length_1, waiting_time_1, queue_length_2, waiting_time_2, ...]
        # for all 12 lanes (24 values total)
        self.observation_space = spaces.Box(
            low=0.0, 
            high=np.inf, 
            shape=(24,), 
            dtype=np.float32
        )
        
        # Phase definitions
        self.phases = {
            0: {"name": "NS_Green", "duration": 30},
            1: {"name": "NS_Yellow", "duration": 3},
            2: {"name": "EW_Green", "duration": 30},
            3: {"name": "EW_Yellow", "duration": 3}
        }
        
        # Episode tracking
        self.episode_reward = 0.0
        self.episode_metrics = []
        
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
            print(f"Step {self.current_step}: "
                  f"Phase {self.current_phase}, "
                  f"Vehicles {state_summary.get('total_vehicles', 0)}, "
                  f"Queue {state_summary.get('total_queue_length', 0)}")
    
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
        
        if action == 0:  # NS Green
            self._set_phase(0, 30)
        elif action == 1:  # EW Green
            self._set_phase(2, 30)
        elif action == 2:  # Extend current phase
            new_duration = min(self.phase_duration + 10, self.max_green)
            self._set_phase(self.current_phase, new_duration)
        elif action == 3:  # Skip to next phase
            next_phase = self._get_next_phase(self.current_phase)
            self._set_phase(next_phase, 30)
    
    def _set_phase(self, phase_id: int, duration: float):
        """Set traffic light phase"""
        if not self.simulation_running:
            return
        
        try:
            traci.trafficlight.setPhase(self.traffic_light_id, phase_id)
            traci.trafficlight.setPhaseDuration(self.traffic_light_id, duration)
        except Exception as e:
            print(f"Error setting phase: {e}")
    
    def _get_next_phase(self, current_phase: int) -> int:
        """Get the next phase in the sequence"""
        if current_phase == 0:  # NS Green -> NS Yellow
            return 1
        elif current_phase == 1:  # NS Yellow -> EW Green
            return 2
        elif current_phase == 2:  # EW Green -> EW Yellow
            return 3
        elif current_phase == 3:  # EW Yellow -> NS Green
            return 0
        else:
            return 0  # Default to NS Green
    
    def _get_state(self) -> np.ndarray:
        """Get current state observation"""
        try:
            # Get 24-dimensional state vector from state utils
            state_vector = get_24d_state_vector()
            return state_vector
        except Exception as e:
            print(f"Error getting state: {e}")
            # Return zero state if error occurs
            return np.zeros(24, dtype=np.float32)
    
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
        print(f"Step {i+1}: Action {action}, Reward {reward:.3f}, Done {done}")
        
        if done:
            break
    
    # Get episode summary
    summary = env.get_episode_summary()
    print(f"Episode summary: {summary}")
    
    env.close()
