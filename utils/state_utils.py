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
        # Define approach edges for the 4-lane symmetrical intersection
        # Each approach has dedicated lanes for different movements
        self.approach_edges = {
            'north': {
                'straight_left': ['N_in'],    # Lanes 0, 1, 2 (left-turn + straight lanes)
                'right': ['N_in']             # Lane 3 (right-turn lane)
            },
            'east': {
                'straight_left': ['E_in'],    # Lanes 0, 1, 2 (left-turn + straight lanes)
                'right': ['E_in']             # Lane 3 (right-turn lane)
            },
            'south': {
                'straight_left': ['S_in'],    # Lanes 0, 1, 2 (left-turn + straight lanes)
                'right': ['S_in']             # Lane 3 (right-turn lane)
            },
            'west': {
                'straight_left': ['W_in'],    # Lanes 0, 1, 2 (left-turn + straight lanes)
                'right': ['W_in']             # Lane 3 (right-turn lane)
            }
        }
        
        # Initialize state history for tracking changes
        self.previous_states = {}
        
    def get_approach_queue_length(self, approach_name: str, movement_type: str) -> int:
        """Get queue length for a specific approach and movement type"""
        try:
            total_queue = 0
            edges = self.approach_edges[approach_name][movement_type]
            
            for edge_id in edges:
                # Get all lanes in this edge
                for lane_idx in range(traci.edge.getLaneNumber(edge_id)):
                    # For straight_left: use lanes 0, 1, 2 (left-turn + straight lanes)
                    # For right: use lane 3 (right-turn lane)
                    if movement_type == 'straight_left' and 0 <= lane_idx <= 2:
                        lane_id = f"{edge_id}_{lane_idx}"
                        # Get vehicles on the lane
                        vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                        
                        for vehicle_id in vehicles:
                            # Check if vehicle is stopped or moving very slowly
                            speed = traci.vehicle.getSpeed(vehicle_id)
                            if speed < 0.1:  # Consider stopped if speed < 0.1 m/s
                                total_queue += 1
                    elif movement_type == 'right' and lane_idx == 3:
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
            print(f"Error getting queue length for {approach_name} {movement_type}: {e}")
            return 0
    
    def get_approach_waiting_time(self, approach_name: str, movement_type: str) -> float:
        """Get total waiting time for a specific approach and movement type"""
        try:
            total_waiting_time = 0.0
            edges = self.approach_edges[approach_name][movement_type]
            
            for edge_id in edges:
                # Get all lanes in this edge
                for lane_idx in range(traci.edge.getLaneNumber(edge_id)):
                    # For straight_left: use lanes 0, 1, 2 (left-turn + straight lanes)
                    # For right: use lane 3 (right-turn lane)
                    if movement_type == 'straight_left' and 0 <= lane_idx <= 2:
                        lane_id = f"{edge_id}_{lane_idx}"
                        # Get vehicles on the lane
                        vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                        
                        for vehicle_id in vehicles:
                            # Get waiting time for this vehicle
                            waiting_time = traci.vehicle.getWaitingTime(vehicle_id)
                            total_waiting_time += waiting_time
                    elif movement_type == 'right' and lane_idx == 3:
                        lane_id = f"{edge_id}_{lane_idx}"
                        # Get vehicles on the lane
                        vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                        
                        for vehicle_id in vehicles:
                            # Get waiting time for this vehicle
                            waiting_time = traci.vehicle.getWaitingTime(vehicle_id)
                            total_waiting_time += waiting_time
            
            return total_waiting_time
            
        except Exception as e:
            print(f"Error getting waiting time for {approach_name} {movement_type}: {e}")
            return 0.0
    
    def get_approach_vehicle_count(self, approach_name: str, movement_type: str) -> int:
        """Get total vehicle count for a specific approach and movement type"""
        try:
            total_vehicles = 0
            edges = self.approach_edges[approach_name][movement_type]
            
            for edge_id in edges:
                # Get all lanes in this edge
                for lane_idx in range(traci.edge.getLaneNumber(edge_id)):
                    # For straight_left: use lanes 0, 1, 2 (left-turn + straight lanes)
                    # For right: use lane 3 (right-turn lane)
                    if movement_type == 'straight_left' and 0 <= lane_idx <= 2:
                        lane_id = f"{edge_id}_{lane_idx}"
                        vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                        total_vehicles += len(vehicles)
                    elif movement_type == 'right' and lane_idx == 3:
                        lane_id = f"{edge_id}_{lane_idx}"
                        vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                        total_vehicles += len(vehicles)
            return total_vehicles
        except Exception as e:
            print(f"Error getting vehicle count for {approach_name} {movement_type}: {e}")
            return 0
    
    def get_approach_state(self, approach_name: str) -> ApproachState:
        """Get complete state information for a specific approach"""
        queue_length = self.get_approach_queue_length(approach_name, 'straight_left') + self.get_approach_queue_length(approach_name, 'right')
        total_waiting_time = self.get_approach_waiting_time(approach_name, 'straight_left') + self.get_approach_waiting_time(approach_name, 'right')
        vehicle_count = self.get_approach_vehicle_count(approach_name, 'straight_left') + self.get_approach_vehicle_count(approach_name, 'right')
        
        return ApproachState(
            queue_length=queue_length,
            total_waiting_time=total_waiting_time,
            vehicle_count=vehicle_count
        )
    
    def get_current_phase(self) -> int:
        """Get current traffic light phase"""
        try:
            # Get the traffic light ID (use "C" for the gpt_newint intersection)
            traffic_light_id = "C"
            current_phase = traci.trafficlight.getPhase(traffic_light_id)
            return current_phase
        except Exception as e:
            print(f"Error getting current phase: {e}")
            return 0
    
    def get_12d_state_vector(self) -> np.ndarray:
        """
        Extract 12-dimensional state vector: 8 queue lengths + 4 one-hot phase encoding
        
        Returns:
            np.ndarray: 12-dimensional vector with format:
            [N_straight_left_q, N_right_q, S_straight_left_q, S_right_q,
             E_straight_left_q, E_right_q, W_straight_left_q, W_right_q,
             current_phase_one_hot_0, current_phase_one_hot_1, current_phase_one_hot_2, current_phase_one_hot_3]
        """
        try:
            # Get queue lengths for each approach and movement type
            N_straight_left_q = self.get_approach_queue_length('north', 'straight_left')
            N_right_q = self.get_approach_queue_length('north', 'right')
            S_straight_left_q = self.get_approach_queue_length('south', 'straight_left')
            S_right_q = self.get_approach_queue_length('south', 'right')
            E_straight_left_q = self.get_approach_queue_length('east', 'straight_left')
            E_right_q = self.get_approach_queue_length('east', 'right')
            W_straight_left_q = self.get_approach_queue_length('west', 'straight_left')
            W_right_q = self.get_approach_queue_length('west', 'right')
            
            # Get current phase and create one-hot encoding
            current_phase = self.get_current_phase()
            phase_encoding = [0, 0, 0, 0]  # 4 phases (0-3)
            if 0 <= current_phase < 4:
                phase_encoding[current_phase] = 1
            
            # Combine into 12D state vector (8 queue lengths + 4 phase encoding)
            state_vector = [
                N_straight_left_q, N_right_q, S_straight_left_q, S_right_q,
                E_straight_left_q, E_right_q, W_straight_left_q, W_right_q,
                phase_encoding[0], phase_encoding[1], phase_encoding[2], phase_encoding[3]
            ]
            
            return np.array(state_vector, dtype=np.float32)
            
        except Exception as e:
            print(f"Error creating state vector: {e}")
            # Return zero vector on error
            return np.zeros(12, dtype=np.float32)
    
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

def get_12d_state_vector() -> np.ndarray:
    """Convenience function to get 12D state vector"""
    return state_extractor.get_12d_state_vector()

def get_state_summary() -> Dict[str, float]:
    """Convenience function to get state summary"""
    return state_extractor.get_state_summary()

def get_detailed_state_dict() -> Dict[str, ApproachState]:
    """Convenience function to get detailed state"""
    return state_extractor.get_detailed_state_dict()
