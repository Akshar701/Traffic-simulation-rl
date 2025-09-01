#!/usr/bin/env python3
"""
State Extraction Utilities
==========================

Helper functions to extract and compute traffic state information
from SUMO simulation for RL training.
"""

import traci
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class LaneState:
    """State information for a single lane"""
    queue_length: int
    cumulative_waiting_time: float
    vehicle_count: int
    avg_speed: float

class StateExtractor:
    """Extracts traffic state information from SUMO simulation"""
    
    def __init__(self):
        # Define lane IDs for the intersection
        # Based on the network structure in Sumo_env/Single intersection lhd/
        self.lane_ids = [
            # North-South lanes
            "3i_0", "3i_1", "4i_0", "4i_1",  # Input lanes
            "3o_0", "3o_1", "4o_0", "4o_1",  # Output lanes
            
            # East-West lanes  
            "1i_0", "1i_1", "2i_0", "2i_1",  # Input lanes
            "1o_0", "1o_1", "2o_0", "2o_1",  # Output lanes
            
            # Diagonal lanes (if they exist)
            "51o_0", "51o_1", "52o_0", "52o_1",  # West approach
            "53o_0", "53o_1", "54o_0", "54o_1"   # East approach
        ]
        
        # Initialize state history for tracking changes
        self.previous_states = {}
        
    def get_lane_queue_length(self, lane_id: str) -> int:
        """Get queue length for a specific lane"""
        try:
            # Get vehicles on the lane
            vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
            
            queue_count = 0
            for vehicle_id in vehicles:
                # Check if vehicle is stopped or moving very slowly
                speed = traci.vehicle.getSpeed(vehicle_id)
                if speed < 0.1:  # Consider stopped if speed < 0.1 m/s
                    queue_count += 1
            
            return queue_count
            
        except Exception as e:
            print(f"Error getting queue length for lane {lane_id}: {e}")
            return 0
    
    def get_lane_cumulative_waiting_time(self, lane_id: str) -> float:
        """Get cumulative waiting time for a specific lane"""
        try:
            # Get vehicles on the lane
            vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
            
            total_waiting_time = 0.0
            for vehicle_id in vehicles:
                # Get waiting time for this vehicle
                waiting_time = traci.vehicle.getWaitingTime(vehicle_id)
                total_waiting_time += waiting_time
            
            return total_waiting_time
            
        except Exception as e:
            print(f"Error getting waiting time for lane {lane_id}: {e}")
            return 0.0
    
    def get_lane_vehicle_count(self, lane_id: str) -> int:
        """Get number of vehicles on a specific lane"""
        try:
            vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
            return len(vehicles)
        except Exception as e:
            print(f"Error getting vehicle count for lane {lane_id}: {e}")
            return 0
    
    def get_lane_avg_speed(self, lane_id: str) -> float:
        """Get average speed for vehicles on a specific lane"""
        try:
            vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
            
            if not vehicles:
                return 0.0
            
            speeds = []
            for vehicle_id in vehicles:
                speed = traci.vehicle.getSpeed(vehicle_id)
                speeds.append(speed)
            
            return np.mean(speeds)
            
        except Exception as e:
            print(f"Error getting average speed for lane {lane_id}: {e}")
            return 0.0
    
    def get_lane_state(self, lane_id: str) -> LaneState:
        """Get complete state for a specific lane"""
        queue_length = self.get_lane_queue_length(lane_id)
        cumulative_waiting_time = self.get_lane_cumulative_waiting_time(lane_id)
        vehicle_count = self.get_lane_vehicle_count(lane_id)
        avg_speed = self.get_lane_avg_speed(lane_id)
        
        return LaneState(
            queue_length=queue_length,
            cumulative_waiting_time=cumulative_waiting_time,
            vehicle_count=vehicle_count,
            avg_speed=avg_speed
        )
    
    def get_24d_state_vector(self) -> np.ndarray:
        """
        Extract 24-dimensional state vector from all lanes
        
        Returns:
            np.ndarray: 24-dimensional vector with format:
            [queue_length_1, waiting_time_1, queue_length_2, waiting_time_2, ...]
            for all 12 lanes (24 values total)
        """
        state_vector = []
        
        # Get state for each lane
        for lane_id in self.lane_ids:
            try:
                lane_state = self.get_lane_state(lane_id)
                state_vector.extend([
                    lane_state.queue_length,
                    lane_state.cumulative_waiting_time
                ])
            except Exception as e:
                print(f"Error processing lane {lane_id}: {e}")
                # Add zeros if lane doesn't exist or error occurs
                state_vector.extend([0, 0])
        
        # Ensure we have exactly 24 values
        if len(state_vector) < 24:
            state_vector.extend([0] * (24 - len(state_vector)))
        elif len(state_vector) > 24:
            state_vector = state_vector[:24]
        
        return np.array(state_vector, dtype=np.float32)
    
    def get_detailed_state_dict(self) -> Dict[str, LaneState]:
        """Get detailed state information for all lanes"""
        detailed_state = {}
        
        for lane_id in self.lane_ids:
            try:
                detailed_state[lane_id] = self.get_lane_state(lane_id)
            except Exception as e:
                print(f"Error getting detailed state for lane {lane_id}: {e}")
                detailed_state[lane_id] = LaneState(0, 0.0, 0, 0.0)
        
        return detailed_state
    
    def get_state_summary(self) -> Dict[str, float]:
        """Get summary statistics across all lanes"""
        detailed_state = self.get_detailed_state_dict()
        
        total_queue_length = sum(state.queue_length for state in detailed_state.values())
        total_waiting_time = sum(state.cumulative_waiting_time for state in detailed_state.values())
        total_vehicles = sum(state.vehicle_count for state in detailed_state.values())
        
        # Calculate average speed across all lanes
        speeds = [state.avg_speed for state in detailed_state.values() if state.avg_speed > 0]
        avg_speed = np.mean(speeds) if speeds else 0.0
        
        return {
            'total_queue_length': total_queue_length,
            'total_waiting_time': total_waiting_time,
            'total_vehicles': total_vehicles,
            'avg_speed': avg_speed
        }
    
    def get_state_change(self, current_state: np.ndarray) -> np.ndarray:
        """Calculate change in state from previous step"""
        if len(self.previous_states) == 0:
            # First step, no change
            self.previous_states['state'] = current_state
            return np.zeros_like(current_state)
        
        previous_state = self.previous_states['state']
        state_change = current_state - previous_state
        
        # Update previous state
        self.previous_states['state'] = current_state.copy()
        
        return state_change

# Global instance for easy access
state_extractor = StateExtractor()

def get_24d_state_vector() -> np.ndarray:
    """Convenience function to get 24D state vector"""
    return state_extractor.get_24d_state_vector()

def get_state_summary() -> Dict[str, float]:
    """Convenience function to get state summary"""
    return state_extractor.get_state_summary()

def get_detailed_state_dict() -> Dict[str, LaneState]:
    """Convenience function to get detailed state"""
    return state_extractor.get_detailed_state_dict()
