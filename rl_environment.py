#!/usr/bin/env python3
"""
RL Environment for Traffic Signal Control
=========================================

A gym-compatible environment for training RL models on traffic signal control.
Provides proper state representation, action space, and reward function.
"""

import gym
import numpy as np
from gym import spaces
from typing import Dict, List, Any, Optional, Tuple
import time
import json

from traci_manager import TraciManager, TrafficState
from signal_controller import SignalController, SignalDecision

class TrafficSignalEnv(gym.Env):
    """Traffic Signal Control Environment for RL Training"""
    
    def __init__(self, config_file: str = None, max_steps: int = 1000):
        super(TrafficSignalEnv, self).__init__()
        
        # Initialize components
        self.traci_manager = TraciManager(config_file)
        self.signal_controller = SignalController(self.traci_manager)
        
        # Environment parameters
        self.max_steps = max_steps
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_metrics = []
        
        # State and action spaces
        self._setup_spaces()
        
        # Performance tracking
        self.baseline_metrics = None
        self.performance_history = []
        
    def _setup_spaces(self):
        """Setup observation and action spaces"""
        # Action space: 4 discrete actions (0: NS Green, 1: EW Green, 2: Extend, 3: Skip)
        self.action_space = spaces.Discrete(4)
        
        # Observation space: normalized state vector
        # Features: [vehicle_count, waiting_vehicles, avg_waiting_time, avg_speed, 
        #           queue_length, current_phase, phase_duration, 
        #           ns_queues, ew_queues, ns_waiting, ew_waiting,
        #           congestion_level_ns, congestion_level_ew]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(13,), dtype=np.float32
        )
    
    def reset(self) -> np.ndarray:
        """Reset environment for new episode"""
        # Stop current simulation if running
        if self.traci_manager.is_simulation_running():
            self.traci_manager.stop_simulation()
        
        # Start new simulation
        if not self.traci_manager.start_simulation():
            raise RuntimeError("Failed to start simulation")
        
        # Reset episode variables
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_metrics = []
        
        # Get initial state
        initial_state = self._get_state()
        
        return initial_state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return next state, reward, done, info"""
        # Execute action
        self._execute_action(action)
        
        # Step simulation
        self.traci_manager.step_simulation(1)
        self.current_step += 1
        
        # Get new state
        state = self._get_state()
        
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
            'metrics': metrics,
            'action': action
        }
        
        return state, reward, done, info
    
    def _execute_action(self, action: int):
        """Execute the given action"""
        traffic_state = self.traci_manager.get_traffic_state()
        if not traffic_state:
            return
        
        if action == 0:  # NS Green
            self.traci_manager.change_signal_phase(0, 30.0)
        elif action == 1:  # EW Green
            self.traci_manager.change_signal_phase(2, 30.0)
        elif action == 2:  # Extend current phase
            signal_info = self.signal_controller.get_current_signal_info()
            current_phase = signal_info["current_phase"]
            current_duration = signal_info["phase_duration"]
            self.traci_manager.change_signal_phase(current_phase, current_duration + 10)
        elif action == 3:  # Skip current phase
            signal_info = self.signal_controller.get_current_signal_info()
            current_phase = signal_info["current_phase"]
            next_phase = self.signal_controller._get_next_phase(current_phase)
            self.traci_manager.change_signal_phase(next_phase, 30.0)
    
    def _get_state(self) -> np.ndarray:
        """Get normalized state vector"""
        traffic_state = self.traci_manager.get_traffic_state()
        if not traffic_state:
            return np.zeros(13, dtype=np.float32)
        
        # Extract features
        vehicle_count = traffic_state.vehicle_count
        waiting_vehicles = traffic_state.waiting_vehicles
        avg_waiting_time = traffic_state.avg_waiting_time
        avg_speed = traffic_state.avg_speed
        queue_length = traffic_state.queue_length
        current_phase = traffic_state.current_phase
        phase_duration = traffic_state.phase_duration
        
        # Per-direction metrics
        ns_queues = self._get_ns_queue_length(traffic_state)
        ew_queues = self._get_ew_queue_length(traffic_state)
        ns_waiting = self._get_ns_waiting_time(traffic_state)
        ew_waiting = self._get_ew_waiting_time(traffic_state)
        
        # Congestion levels (encoded as numbers)
        congestion_ns = self._encode_congestion_level(self._get_congestion_level(ns_queues))
        congestion_ew = self._encode_congestion_level(self._get_congestion_level(ew_queues))
        
        # Normalize features
        state_vector = np.array([
            min(vehicle_count / 100.0, 1.0),  # Normalize vehicle count
            min(waiting_vehicles / 50.0, 1.0),  # Normalize waiting vehicles
            min(avg_waiting_time / 60.0, 1.0),  # Normalize waiting time
            min(avg_speed / 15.0, 1.0),  # Normalize speed
            min(queue_length / 30.0, 1.0),  # Normalize queue length
            current_phase / 7.0,  # Normalize phase
            min(phase_duration / 60.0, 1.0),  # Normalize phase duration
            min(ns_queues / 15.0, 1.0),  # Normalize NS queues
            min(ew_queues / 15.0, 1.0),  # Normalize EW queues
            min(ns_waiting / 60.0, 1.0),  # Normalize NS waiting
            min(ew_waiting / 60.0, 1.0),  # Normalize EW waiting
            congestion_ns,  # Already normalized
            congestion_ew   # Already normalized
        ], dtype=np.float32)
        
        return state_vector
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on traffic performance"""
        traffic_state = self.traci_manager.get_traffic_state()
        if not traffic_state:
            return 0.0
        
        # Base reward components
        efficiency_reward = self._calculate_efficiency_reward(traffic_state)
        waiting_time_penalty = self._calculate_waiting_time_penalty(traffic_state)
        queue_penalty = self._calculate_queue_penalty(traffic_state)
        throughput_reward = self._calculate_throughput_reward(traffic_state)
        
        # Combine rewards
        total_reward = (
            efficiency_reward * 0.4 +
            throughput_reward * 0.3 +
            waiting_time_penalty * 0.2 +
            queue_penalty * 0.1
        )
        
        return total_reward
    
    def _calculate_efficiency_reward(self, traffic_state: TrafficState) -> float:
        """Calculate reward based on overall efficiency"""
        # Efficiency score (0-100) converted to reward (-1 to 1)
        efficiency_score = self._calculate_efficiency_score(traffic_state)
        return (efficiency_score - 50.0) / 50.0  # Normalize to [-1, 1]
    
    def _calculate_waiting_time_penalty(self, traffic_state: TrafficState) -> float:
        """Calculate penalty for long waiting times"""
        # Penalize waiting times > 30 seconds
        if traffic_state.avg_waiting_time > 30.0:
            return -1.0
        elif traffic_state.avg_waiting_time > 20.0:
            return -0.5
        elif traffic_state.avg_waiting_time > 10.0:
            return -0.2
        else:
            return 0.0
    
    def _calculate_queue_penalty(self, traffic_state: TrafficState) -> float:
        """Calculate penalty for long queues"""
        # Penalize queues > 20 vehicles
        if traffic_state.queue_length > 20:
            return -1.0
        elif traffic_state.queue_length > 15:
            return -0.5
        elif traffic_state.queue_length > 10:
            return -0.2
        else:
            return 0.0
    
    def _calculate_throughput_reward(self, traffic_state: TrafficState) -> float:
        """Calculate reward based on throughput"""
        # Reward for high average speed
        if traffic_state.avg_speed > 10.0:
            return 1.0
        elif traffic_state.avg_speed > 7.0:
            return 0.5
        elif traffic_state.avg_speed > 5.0:
            return 0.2
        else:
            return -0.5
    
    def _calculate_efficiency_score(self, traffic_state: TrafficState) -> float:
        """Calculate efficiency score (0-100)"""
        # Base efficiency on multiple factors
        speed_factor = min(traffic_state.avg_speed / 15.0, 1.0) * 30
        waiting_factor = max(0, (60.0 - traffic_state.avg_waiting_time) / 60.0) * 30
        queue_factor = max(0, (30.0 - traffic_state.queue_length) / 30.0) * 20
        throughput_factor = min(traffic_state.vehicle_count / 50.0, 1.0) * 20
        
        return speed_factor + waiting_factor + queue_factor + throughput_factor
    
    def _is_done(self) -> bool:
        """Check if episode is done"""
        return (
            self.current_step >= self.max_steps or
            not self.traci_manager.is_simulation_running()
        )
    
    def _get_metrics(self) -> Dict[str, Any]:
        """Get current episode metrics"""
        traffic_state = self.traci_manager.get_traffic_state()
        if not traffic_state:
            return {}
        
        return {
            'step': self.current_step,
            'vehicle_count': traffic_state.vehicle_count,
            'waiting_vehicles': traffic_state.waiting_vehicles,
            'avg_waiting_time': traffic_state.avg_waiting_time,
            'avg_speed': traffic_state.avg_speed,
            'queue_length': traffic_state.queue_length,
            'efficiency_score': self._calculate_efficiency_score(traffic_state),
            'episode_reward': self.episode_reward
        }
    
    def _get_ns_queue_length(self, traffic_state: TrafficState) -> int:
        """Get North-South queue length"""
        if traffic_state.per_lane_queues:
            ns_queues = 0
            for lane_id, queue in traffic_state.per_lane_queues.items():
                if any(direction in lane_id for direction in ['3i', '4i']):
                    ns_queues += queue
            return ns_queues
        return 0
    
    def _get_ew_queue_length(self, traffic_state: TrafficState) -> int:
        """Get East-West queue length"""
        if traffic_state.per_lane_queues:
            ew_queues = 0
            for lane_id, queue in traffic_state.per_lane_queues.items():
                if any(direction in lane_id for direction in ['1i', '2i']):
                    ew_queues += queue
            return ew_queues
        return 0
    
    def _get_ns_waiting_time(self, traffic_state: TrafficState) -> float:
        """Get North-South average waiting time"""
        if traffic_state.per_lane_waiting_times:
            ns_times = []
            for lane_id, waiting_time in traffic_state.per_lane_waiting_times.items():
                if any(direction in lane_id for direction in ['3i', '4i']):
                    ns_times.append(waiting_time)
            return sum(ns_times) / len(ns_times) if ns_times else 0.0
        return 0.0
    
    def _get_ew_waiting_time(self, traffic_state: TrafficState) -> float:
        """Get East-West average waiting time"""
        if traffic_state.per_lane_waiting_times:
            ew_times = []
            for lane_id, waiting_time in traffic_state.per_lane_waiting_times.items():
                if any(direction in lane_id for direction in ['1i', '2i']):
                    ew_times.append(waiting_time)
            return sum(ew_times) / len(ew_times) if ew_times else 0.0
        return 0.0
    
    def _get_congestion_level(self, queue_length: int) -> str:
        """Get congestion level based on queue length"""
        if queue_length == 0:
            return "low"
        elif queue_length <= 3:
            return "moderate"
        elif queue_length <= 8:
            return "high"
        else:
            return "severe"
    
    def _encode_congestion_level(self, level: str) -> float:
        """Encode congestion level as number"""
        encoding = {"low": 0.0, "moderate": 0.33, "high": 0.66, "severe": 1.0}
        return encoding.get(level, 0.0)
    
    def close(self):
        """Clean up environment"""
        if self.traci_manager.is_simulation_running():
            self.traci_manager.stop_simulation()
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of completed episode"""
        if not self.episode_metrics:
            return {}
        
        # Calculate episode statistics
        avg_efficiency = np.mean([m['efficiency_score'] for m in self.episode_metrics])
        avg_waiting_time = np.mean([m['avg_waiting_time'] for m in self.episode_metrics])
        avg_speed = np.mean([m['avg_speed'] for m in self.episode_metrics])
        total_vehicles = sum([m['vehicle_count'] for m in self.episode_metrics])
        
        return {
            'episode_length': self.current_step,
            'total_reward': self.episode_reward,
            'avg_efficiency': avg_efficiency,
            'avg_waiting_time': avg_waiting_time,
            'avg_speed': avg_speed,
            'total_vehicles': total_vehicles,
            'metrics': self.episode_metrics
        }
