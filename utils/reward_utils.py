#!/usr/bin/env python3
"""
Reward Calculation Utilities
============================

Helper functions to calculate rewards for RL training based on
traffic performance metrics.
"""

import numpy as np
import pandas as pd
import csv
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass

@dataclass
class RewardComponents:
    """Components of the reward function"""
    waiting_time_change: float
    queue_penalty: float
    throughput_reward: float
    efficiency_reward: float
    total_reward: float

class RewardCalculator:
    """Calculates rewards based on traffic performance metrics"""
    
    def __init__(self, log_file: str = "reward_log.csv"):
        self.log_file = log_file
        self.previous_waiting_time = 0.0
        self.previous_queue_length = 0
        self.step_count = 0
        
        # Initialize reward log
        self._init_reward_log()
        
        # Reward weights
        self.weights = {
            'waiting_time_change': 0.4,
            'queue_penalty': 0.2,
            'throughput_reward': 0.25,
            'efficiency_reward': 0.15
        }
        
    def _init_reward_log(self):
        """Initialize the reward log CSV file"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'step',
                    'timestamp',
                    'waiting_time_change',
                    'queue_penalty',
                    'throughput_reward',
                    'efficiency_reward',
                    'total_reward',
                    'current_waiting_time',
                    'current_queue_length',
                    'avg_speed',
                    'vehicle_count'
                ])
    
    def reward_waiting_time_change(self, prev_wait: float, curr_wait: float) -> float:
        """
        Calculate reward based on change in cumulative waiting time
        
        Args:
            prev_wait: Previous step's cumulative waiting time
            curr_wait: Current step's cumulative waiting time
            
        Returns:
            float: Reward (positive if waiting time decreased)
        """
        # Reward is the decrease in waiting time
        # Positive reward if waiting time decreased
        # Negative reward if waiting time increased
        return prev_wait - curr_wait
    
    def reward_queue_penalty(self, queue_length: int) -> float:
        """
        Calculate penalty for long queues
        
        Args:
            queue_length: Total number of vehicles in queues
            
        Returns:
            float: Penalty (negative value)
        """
        if queue_length == 0:
            return 0.0
        elif queue_length <= 5:
            return -0.1
        elif queue_length <= 10:
            return -0.3
        elif queue_length <= 20:
            return -0.6
        else:
            return -1.0
    
    def reward_throughput(self, avg_speed: float, vehicle_count: int) -> float:
        """
        Calculate reward based on throughput (speed and vehicle count)
        
        Args:
            avg_speed: Average speed of vehicles
            vehicle_count: Number of vehicles in simulation
            
        Returns:
            float: Throughput reward
        """
        # Reward for high average speed
        speed_reward = 0.0
        if avg_speed > 10.0:
            speed_reward = 1.0
        elif avg_speed > 7.0:
            speed_reward = 0.5
        elif avg_speed > 5.0:
            speed_reward = 0.2
        elif avg_speed > 2.0:
            speed_reward = 0.0
        else:
            speed_reward = -0.5
        
        # Small reward for having vehicles (indicates activity)
        activity_reward = min(vehicle_count / 20.0, 0.2)
        
        return speed_reward + activity_reward
    
    def reward_efficiency(self, queue_length: int, waiting_time: float, avg_speed: float) -> float:
        """
        Calculate efficiency reward based on multiple factors
        
        Args:
            queue_length: Total queue length
            waiting_time: Cumulative waiting time
            avg_speed: Average vehicle speed
            
        Returns:
            float: Efficiency reward
        """
        # Efficiency score (0-100) converted to reward (-1 to 1)
        queue_factor = max(0, (30.0 - queue_length) / 30.0) * 30
        waiting_factor = max(0, (60.0 - waiting_time) / 60.0) * 40
        speed_factor = min(avg_speed / 15.0, 1.0) * 30
        
        efficiency_score = queue_factor + waiting_factor + speed_factor
        return (efficiency_score - 50.0) / 50.0  # Normalize to [-1, 1]
    
    def calculate_reward(self, 
                        current_waiting_time: float,
                        current_queue_length: int,
                        avg_speed: float,
                        vehicle_count: int) -> RewardComponents:
        """
        Calculate total reward from all components
        
        Args:
            current_waiting_time: Current cumulative waiting time
            current_queue_length: Current total queue length
            avg_speed: Average vehicle speed
            vehicle_count: Number of vehicles in simulation
            
        Returns:
            RewardComponents: All reward components and total
        """
        # Calculate waiting time change reward
        waiting_time_change = self.reward_waiting_time_change(
            self.previous_waiting_time, current_waiting_time
        )
        
        # Calculate queue penalty
        queue_penalty = self.reward_queue_penalty(current_queue_length)
        
        # Calculate throughput reward
        throughput_reward = self.reward_throughput(avg_speed, vehicle_count)
        
        # Calculate efficiency reward
        efficiency_reward = self.reward_efficiency(
            current_queue_length, current_waiting_time, avg_speed
        )
        
        # Calculate weighted total reward
        total_reward = (
            waiting_time_change * self.weights['waiting_time_change'] +
            queue_penalty * self.weights['queue_penalty'] +
            throughput_reward * self.weights['throughput_reward'] +
            efficiency_reward * self.weights['efficiency_reward']
        )
        
        # Create reward components object
        reward_components = RewardComponents(
            waiting_time_change=waiting_time_change,
            queue_penalty=queue_penalty,
            throughput_reward=throughput_reward,
            efficiency_reward=efficiency_reward,
            total_reward=total_reward
        )
        
        # Log reward to CSV
        self._log_reward(reward_components, current_waiting_time, 
                        current_queue_length, avg_speed, vehicle_count)
        
        # Update previous values for next step
        self.previous_waiting_time = current_waiting_time
        self.previous_queue_length = current_queue_length
        self.step_count += 1
        
        return reward_components
    
    def _log_reward(self, 
                   reward_components: RewardComponents,
                   current_waiting_time: float,
                   current_queue_length: int,
                   avg_speed: float,
                   vehicle_count: int):
        """Log reward components to CSV file for debugging"""
        try:
            with open(self.log_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    self.step_count,
                    datetime.now().isoformat(),
                    reward_components.waiting_time_change,
                    reward_components.queue_penalty,
                    reward_components.throughput_reward,
                    reward_components.efficiency_reward,
                    reward_components.total_reward,
                    current_waiting_time,
                    current_queue_length,
                    avg_speed,
                    vehicle_count
                ])
        except Exception as e:
            print(f"Error logging reward: {e}")
    
    def get_reward_summary(self) -> Dict[str, float]:
        """Get summary statistics of rewards"""
        try:
            df = pd.read_csv(self.log_file)
            
            if df.empty:
                return {}
            
            summary = {
                'total_steps': len(df),
                'mean_total_reward': df['total_reward'].mean(),
                'std_total_reward': df['total_reward'].std(),
                'min_total_reward': df['total_reward'].min(),
                'max_total_reward': df['total_reward'].max(),
                'mean_waiting_time_change': df['waiting_time_change'].mean(),
                'mean_queue_penalty': df['queue_penalty'].mean(),
                'mean_throughput_reward': df['throughput_reward'].mean(),
                'mean_efficiency_reward': df['efficiency_reward'].mean()
            }
            
            return summary
            
        except Exception as e:
            print(f"Error getting reward summary: {e}")
            return {}
    
    def reset(self):
        """Reset the reward calculator for new episode"""
        self.previous_waiting_time = 0.0
        self.previous_queue_length = 0
        self.step_count = 0

# Global instance for easy access
reward_calculator = RewardCalculator()

def calculate_reward(current_waiting_time: float,
                    current_queue_length: int,
                    avg_speed: float,
                    vehicle_count: int) -> float:
    """Convenience function to calculate total reward"""
    components = reward_calculator.calculate_reward(
        current_waiting_time, current_queue_length, avg_speed, vehicle_count
    )
    return components.total_reward

def reward_waiting_time_change(prev_wait: float, curr_wait: float) -> float:
    """Convenience function for waiting time change reward"""
    return reward_calculator.reward_waiting_time_change(prev_wait, curr_wait)

def get_reward_summary() -> Dict[str, float]:
    """Convenience function to get reward summary"""
    return reward_calculator.get_reward_summary()

def reset_reward_calculator():
    """Convenience function to reset reward calculator"""
    reward_calculator.reset()
