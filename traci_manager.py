#!/usr/bin/env python3
"""
Traci Manager - Handles SUMO simulation connection and basic operations
Provides a clean interface for connecting to SUMO and managing simulation state
"""

import traci
import time
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class SimulationState(Enum):
    """Simulation states"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"

@dataclass
class TrafficState:
    """Current traffic state data"""
    timestamp: float
    vehicle_count: int
    waiting_vehicles: int
    avg_waiting_time: float
    avg_speed: float
    queue_length: int
    current_phase: int
    phase_duration: float

class TraciManager:
    """Manages SUMO simulation connection and basic operations"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file or "Sumo_env/Single intersection lhd/uniform_simulation.sumocfg"
        self.simulation_state = SimulationState.STOPPED
        self.traffic_light_id = "0"  # Default traffic light ID
        self.edge_ids = []  # Will be populated when simulation starts
        self.connection = None
        
    def start_simulation(self, config_file: str = None) -> bool:
        """Start SUMO simulation with specified config"""
        try:
            if config_file:
                self.config_file = config_file
                
            # Check if config file exists
            if not os.path.exists(self.config_file):
                print(f"Error: Config file not found: {self.config_file}")
                return False
            
            # Start SUMO with traci
            sumo_cmd = ["sumo", "-c", self.config_file]
            traci.start(sumo_cmd)
            
            # Initialize edge IDs for traffic monitoring
            self._initialize_edge_ids()
            
            self.simulation_state = SimulationState.RUNNING
            print(f"Simulation started with config: {self.config_file}")
            return True
            
        except Exception as e:
            print(f"Error starting simulation: {e}")
            self.simulation_state = SimulationState.ERROR
            return False
    
    def _initialize_edge_ids(self):
        """Initialize edge IDs for traffic monitoring"""
        try:
            # Get all edges in the network
            all_edges = traci.edge.getIDList()
            
            # Filter for edges that are likely to have traffic
            # This is a simple heuristic - you might need to adjust based on your network
            self.edge_ids = [edge for edge in all_edges if not edge.startswith(":")]
            
            print(f"Initialized {len(self.edge_ids)} edges for monitoring")
            
        except Exception as e:
            print(f"Error initializing edge IDs: {e}")
            self.edge_ids = []
    
    def step_simulation(self, steps: int = 1) -> bool:
        """Step the simulation forward"""
        try:
            if self.simulation_state != SimulationState.RUNNING:
                return False
                
            for _ in range(steps):
                traci.simulationStep()
                
            return True
            
        except Exception as e:
            print(f"Error stepping simulation: {e}")
            self.simulation_state = SimulationState.ERROR
            return False
    
    def get_traffic_state(self) -> TrafficState:
        """Get current traffic state"""
        try:
            # Get current simulation time
            timestamp = traci.simulation.getTime()
            
            # Get vehicle statistics
            total_vehicles = traci.vehicle.getIDCount()
            waiting_vehicles = 0
            total_waiting_time = 0
            total_speed = 0
            total_queue_length = 0
            
            # Calculate metrics across all edges
            for edge_id in self.edge_ids:
                waiting_vehicles += traci.edge.getLastStepHaltingNumber(edge_id)
                total_waiting_time += traci.edge.getWaitingTime(edge_id)
                total_queue_length += traci.edge.getLastStepHaltingNumber(edge_id)
            
            # Get vehicle speeds
            vehicle_ids = traci.vehicle.getIDList()
            for vehicle_id in vehicle_ids:
                speed = traci.vehicle.getSpeed(vehicle_id)
                total_speed += speed
            
            # Calculate averages
            avg_waiting_time = total_waiting_time / max(len(self.edge_ids), 1)
            avg_speed = total_speed / max(len(vehicle_ids), 1)
            
            # Get current traffic light state
            current_phase = traci.trafficlight.getPhase(self.traffic_light_id)
            phase_duration = traci.trafficlight.getPhaseDuration(self.traffic_light_id)
            
            return TrafficState(
                timestamp=timestamp,
                vehicle_count=total_vehicles,
                waiting_vehicles=waiting_vehicles,
                avg_waiting_time=avg_waiting_time,
                avg_speed=avg_speed,
                queue_length=total_queue_length,
                current_phase=current_phase,
                phase_duration=phase_duration
            )
            
        except Exception as e:
            print(f"Error getting traffic state: {e}")
            return None
    
    def change_signal_phase(self, phase_id: int, duration: float = None) -> bool:
        """Change traffic signal phase"""
        try:
            if self.simulation_state != SimulationState.RUNNING:
                return False
            
            # Set the phase
            traci.trafficlight.setPhase(self.traffic_light_id, phase_id)
            
            # Set duration if provided
            if duration is not None:
                traci.trafficlight.setPhaseDuration(self.traffic_light_id, duration)
            
            return True
            
        except Exception as e:
            print(f"Error changing signal phase: {e}")
            return False
    
    def get_signal_info(self) -> Dict[str, Any]:
        """Get current traffic signal information"""
        try:
            return {
                "current_phase": traci.trafficlight.getPhase(self.traffic_light_id),
                "phase_duration": traci.trafficlight.getPhaseDuration(self.traffic_light_id),
                "phase_name": traci.trafficlight.getPhaseName(self.traffic_light_id),
                "program_id": traci.trafficlight.getProgram(self.traffic_light_id),
                "state": traci.trafficlight.getRedYellowGreenState(self.traffic_light_id)
            }
        except Exception as e:
            print(f"Error getting signal info: {e}")
            return {}
    
    def get_edge_metrics(self, edge_id: str) -> Dict[str, Any]:
        """Get detailed metrics for a specific edge"""
        try:
            return {
                "vehicle_count": traci.edge.getLastStepVehicleNumber(edge_id),
                "waiting_time": traci.edge.getWaitingTime(edge_id),
                "queue_length": traci.edge.getLastStepHaltingNumber(edge_id),
                "mean_speed": traci.edge.getLastStepMeanSpeed(edge_id),
                "occupancy": traci.edge.getLastStepOccupancy(edge_id)
            }
        except Exception as e:
            print(f"Error getting edge metrics: {e}")
            return {}
    
    def is_simulation_running(self) -> bool:
        """Check if simulation is still running"""
        try:
            return traci.simulation.getMinExpectedNumber() > 0
        except:
            return False
    
    def pause_simulation(self):
        """Pause the simulation"""
        self.simulation_state = SimulationState.PAUSED
        print("Simulation paused")
    
    def resume_simulation(self):
        """Resume the simulation"""
        self.simulation_state = SimulationState.RUNNING
        print("Simulation resumed")
    
    def stop_simulation(self):
        """Stop and close the simulation"""
        try:
            traci.close()
            self.simulation_state = SimulationState.STOPPED
            print("Simulation stopped")
        except Exception as e:
            print(f"Error stopping simulation: {e}")
    
    def run_simulation_for_duration(self, duration: int, step_size: int = 1):
        """Run simulation for a specified duration"""
        if not self.start_simulation():
            return False
        
        try:
            current_time = 0
            while current_time < duration and self.is_simulation_running():
                self.step_simulation(step_size)
                current_time += step_size
                
                # Print progress every 100 steps
                if current_time % 100 == 0:
                    state = self.get_traffic_state()
                    if state:
                        print(f"Time: {current_time}, Vehicles: {state.vehicle_count}, "
                              f"Waiting: {state.waiting_vehicles}, Phase: {state.current_phase}")
            
            return True
            
        except Exception as e:
            print(f"Error running simulation: {e}")
            return False
        finally:
            self.stop_simulation()

# Example usage
if __name__ == "__main__":
    # Create traci manager
    manager = TraciManager()
    
    # Run a simple simulation
    print("Starting simulation...")
    success = manager.run_simulation_for_duration(1000)
    
    if success:
        print("Simulation completed successfully")
    else:
        print("Simulation failed")
