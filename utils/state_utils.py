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
class ApproachState:
    """State information for a single approach"""
    queue_length: int
    total_waiting_time: float
    vehicle_count: int

class StateExtractor:
    """Extracts traffic state information from SUMO simulation"""
    
    def __init__(self):
        # Define approach edges for the intersection
        # Each approach has input edges that feed into the intersection
        self.approach_edges = {
            'north': ['3i', '4i'],    # North approach (input edges)
            'east': ['1i', '2i'],     # East approach (input edges)  
            'south': ['3o', '4o'],    # South approach (input edges)
            'west': ['1o', '2o']      # West approach (input edges)
        }
        
        # Initialize state history for tracking changes
        self.previous_states = {}
        
    def get_approach_queue_length(self, approach_name: str) -> int:
        """Get total queue length for a specific approach (sum of all input edges)"""
        try:
            total_queue = 0
            for edge_id in self.approach_edges[approach_name]:
                # Get all lanes in this edge
                for lane_idx in range(traci.edge.getLaneNumber(edge_id)):
                    lane_id = f"{edge_id}_{lane_idx}"
                    # Get vehicles on the lane
                    vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                    
                    for vehicle_id in vehicles:
                        # Check if vehicle is stopped or moving very slowly
                        speed = traci.vehicle.getSpeed(vehicle_id)
                        if speed < 0.1:  # Consider stopped if speed < 0.1 m/s
                            total_queue += 1
            
            return total_queue
            
        except Exception as e:
            print(f"Error getting queue length for {approach_name} approach: {e}")
            return 0
    
    def get_approach_waiting_time(self, approach_name: str) -> float:
        """Get total waiting time for a specific approach (sum of all input edges)"""
        try:
            total_waiting_time = 0.0
            for edge_id in self.approach_edges[approach_name]:
                # Get all lanes in this edge
                for lane_idx in range(traci.edge.getLaneNumber(edge_id)):
                    lane_id = f"{edge_id}_{lane_idx}"
                    # Get vehicles on the lane
                    vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                    
                    for vehicle_id in vehicles:
                        # Get waiting time for this vehicle
                        waiting_time = traci.vehicle.getWaitingTime(vehicle_id)
                        total_waiting_time += waiting_time
            
            return total_waiting_time
            
        except Exception as e:
            print(f"Error getting waiting time for {approach_name} approach: {e}")
            return 0.0
    
    def get_approach_vehicle_count(self, approach_name: str) -> int:
        """Get total vehicle count for a specific approach (sum of all input edges)"""
        try:
            total_vehicles = 0
            for edge_id in self.approach_edges[approach_name]:
                # Get all lanes in this edge
                for lane_idx in range(traci.edge.getLaneNumber(edge_id)):
                    lane_id = f"{edge_id}_{lane_idx}"
                    vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                    total_vehicles += len(vehicles)
            return total_vehicles
        except Exception as e:
            print(f"Error getting vehicle count for {approach_name} approach: {e}")
            return 0
    
    def get_approach_state(self, approach_name: str) -> ApproachState:
        """Get complete state information for a specific approach"""
        queue_length = self.get_approach_queue_length(approach_name)
        total_waiting_time = self.get_approach_waiting_time(approach_name)
        vehicle_count = self.get_approach_vehicle_count(approach_name)
        
        return ApproachState(
            queue_length=queue_length,
            total_waiting_time=total_waiting_time,
            vehicle_count=vehicle_count
        )
    
    def get_current_phase(self) -> int:
        """Get current traffic light phase"""
        try:
            # Get the traffic light ID (usually "0" for single intersection)
            traffic_light_id = "0"
            current_phase = traci.trafficlight.getPhase(traffic_light_id)
            return current_phase
        except Exception as e:
            print(f"Error getting current phase: {e}")
            return 0
    
    def get_8d_state_vector(self) -> np.ndarray:
        """
        Extract 8-dimensional state vector: 4 queue lengths + 4 one-hot phase encoding
        
        Returns:
            np.ndarray: 8-dimensional vector with format:
            [north_queue, east_queue, south_queue, west_queue, phase_0, phase_1, phase_2, phase_3]
        """
        try:
            # Get queue lengths for each approach
            north_queue = self.get_approach_queue_length('north')
            east_queue = self.get_approach_queue_length('east')
            south_queue = self.get_approach_queue_length('south')
            west_queue = self.get_approach_queue_length('west')
            
            # Get current phase and create one-hot encoding
            current_phase = self.get_current_phase()
            phase_encoding = [0, 0, 0, 0]  # Assuming 4 phases
            if 0 <= current_phase < 4:
                phase_encoding[current_phase] = 1
            
            # Combine into 8D state vector
            state_vector = [
                north_queue, east_queue, south_queue, west_queue,
                phase_encoding[0], phase_encoding[1], phase_encoding[2], phase_encoding[3]
            ]
            
            return np.array(state_vector, dtype=np.float32)
            
        except Exception as e:
            print(f"Error creating state vector: {e}")
            # Return zero vector on error
            return np.zeros(8, dtype=np.float32)
    
    def get_detailed_state_dict(self) -> Dict[str, ApproachState]:
        """Get detailed state information for all approaches"""
        detailed_state = {}
        
        for approach_name in self.approach_edges.keys():
            try:
                detailed_state[approach_name] = self.get_approach_state(approach_name)
            except Exception as e:
                print(f"Error getting detailed state for {approach_name} approach: {e}")
                detailed_state[approach_name] = ApproachState(0, 0.0, 0)
        
        return detailed_state
    
    def get_state_summary(self) -> Dict[str, float]:
        """Get summary statistics across all approaches"""
        detailed_state = self.get_detailed_state_dict()
        
        total_queue_length = sum(state.queue_length for state in detailed_state.values())
        total_waiting_time = sum(state.total_waiting_time for state in detailed_state.values())
        total_vehicles = sum(state.vehicle_count for state in detailed_state.values())
        
        return {
            'total_queue_length': total_queue_length,
            'total_waiting_time': total_waiting_time,
            'total_vehicles': total_vehicles,
            'current_phase': self.get_current_phase()
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

def get_8d_state_vector() -> np.ndarray:
    """Convenience function to get 8D state vector"""
    return state_extractor.get_8d_state_vector()

def get_state_summary() -> Dict[str, float]:
    """Convenience function to get state summary"""
    return state_extractor.get_state_summary()

def get_detailed_state_dict() -> Dict[str, ApproachState]:
    """Convenience function to get detailed state"""
    return state_extractor.get_detailed_state_dict()
