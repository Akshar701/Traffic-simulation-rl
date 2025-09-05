#!/usr/bin/env python3
"""
Traci Manager - Handles SUMO simulation connection and basic operations
Provides a clean interface for connecting to SUMO and managing simulation state
"""

import traci
import time
import os
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Import our state and reward utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.state_utils import get_12d_state_vector, get_state_summary
from utils.reward_utils import calculate_reward, simple_reward, reset_reward_calculator

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
    per_lane_queues: Dict[str, int] = None
    per_lane_waiting_times: Dict[str, float] = None
    per_lane_vehicle_counts: Dict[str, int] = None
    directional_flow: Dict[str, float] = None
    signal_phase_timing: Dict[str, float] = None
    congestion_per_lane: Dict[str, str] = None
    emergency_vehicles: List[str] = None
    pedestrian_waiting: int = None

class TraciManager:
    """Manages SUMO simulation connection and basic operations"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file or "Sumo_env/gpt_newint/intersection.sumocfg"
        self.simulation_state = SimulationState.STOPPED
        self.traffic_light_id = "C"  # Traffic light ID for our intersection
        self.edge_ids = []  # Will be populated when simulation starts
        self.connection = None
        
        # Phase definitions matching our SUMO configuration
        self.phases = {
            0: {"name": "NS_Left_Straight", "duration": 30, "description": "North-South left-turn + straight lanes green"},
            1: {"name": "NS_Yellow", "duration": 3, "description": "North-South yellow transition"},
            2: {"name": "EW_Left_Straight", "duration": 30, "description": "East-West left-turn + straight lanes green"},
            3: {"name": "EW_Yellow", "duration": 3, "description": "East-West yellow transition"}
        }
        
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
    
    def get_traffic_state(self) -> Optional[TrafficState]:
        """Get enhanced traffic state with per-lane metrics"""
        try:
            if self.simulation_state != SimulationState.RUNNING:
                return None
            
            # Basic metrics
            vehicle_count = len(traci.vehicle.getIDList())
            waiting_vehicles = len([v for v in traci.vehicle.getIDList() 
                                  if traci.vehicle.getWaitingTime(v) > 0])
            
            # Calculate averages
            if vehicle_count > 0:
                waiting_times = [traci.vehicle.getWaitingTime(v) for v in traci.vehicle.getIDList()]
                avg_waiting_time = sum(waiting_times) / vehicle_count
                speeds = [traci.vehicle.getSpeed(v) for v in traci.vehicle.getIDList()]
                avg_speed = sum(speeds) / vehicle_count
            else:
                avg_waiting_time = 0.0
                avg_speed = 0.0
            
            # Get signal information
            signal_info = self.get_signal_info()
            current_phase = signal_info.get("current_phase", 0)
            phase_duration = signal_info.get("phase_duration", 30.0)
            
            # Enhanced per-lane metrics
            per_lane_queues = self._get_per_lane_queues()
            per_lane_waiting_times = self._get_per_lane_waiting_times()
            per_lane_vehicle_counts = self._get_per_lane_vehicle_counts()
            directional_flow = self._get_directional_flow()
            signal_phase_timing = self._get_signal_phase_timing()
            congestion_per_lane = self._get_congestion_per_lane()
            emergency_vehicles = self._get_emergency_vehicles()
            pedestrian_waiting = self._get_pedestrian_waiting()
            
            # Calculate total queue length
            queue_length = sum(per_lane_queues.values())
            
            return TrafficState(
                timestamp=time.time(),
                vehicle_count=vehicle_count,
                waiting_vehicles=waiting_vehicles,
                avg_waiting_time=avg_waiting_time,
                avg_speed=avg_speed,
                queue_length=queue_length,
                current_phase=current_phase,
                phase_duration=phase_duration,
                per_lane_queues=per_lane_queues,
                per_lane_waiting_times=per_lane_waiting_times,
                per_lane_vehicle_counts=per_lane_vehicle_counts,
                directional_flow=directional_flow,
                signal_phase_timing=signal_phase_timing,
                congestion_per_lane=congestion_per_lane,
                emergency_vehicles=emergency_vehicles,
                pedestrian_waiting=pedestrian_waiting
            )
            
        except Exception as e:
            print(f"Error getting traffic state: {e}")
            return None
    
    def _get_per_lane_queues(self) -> Dict[str, int]:
        """Get queue length for each lane"""
        queues = {}
        try:
            for edge_id in self.edge_ids:
                for lane_idx in range(traci.edge.getLaneNumber(edge_id)):
                    lane_id = f"{edge_id}_{lane_idx}"
                    queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
                    queues[lane_id] = queue_length
        except Exception as e:
            print(f"Error getting per-lane queues: {e}")
        return queues
    
    def _get_per_lane_waiting_times(self) -> Dict[str, float]:
        """Get average waiting time for each lane"""
        waiting_times = {}
        try:
            for edge_id in self.edge_ids:
                for lane_idx in range(traci.edge.getLaneNumber(edge_id)):
                    lane_id = f"{edge_id}_{lane_idx}"
                    vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                    if vehicles:
                        lane_waiting_times = [traci.vehicle.getWaitingTime(v) for v in vehicles]
                        waiting_times[lane_id] = sum(lane_waiting_times) / len(lane_waiting_times)
                    else:
                        waiting_times[lane_id] = 0.0
        except Exception as e:
            print(f"Error getting per-lane waiting times: {e}")
        return waiting_times
    
    def _get_per_lane_vehicle_counts(self) -> Dict[str, int]:
        """Get vehicle count for each lane"""
        vehicle_counts = {}
        try:
            for edge_id in self.edge_ids:
                for lane_idx in range(traci.edge.getLaneNumber(edge_id)):
                    lane_id = f"{edge_id}_{lane_idx}"
                    vehicle_count = len(traci.lane.getLastStepVehicleIDs(lane_id))
                    vehicle_counts[lane_id] = vehicle_count
        except Exception as e:
            print(f"Error getting per-lane vehicle counts: {e}")
        return vehicle_counts
    
    def _get_directional_flow(self) -> Dict[str, float]:
        """Get traffic flow by direction"""
        flow = {"NS": 0.0, "EW": 0.0, "NE": 0.0, "NW": 0.0, "SE": 0.0, "SW": 0.0}
        try:
            # This is a simplified implementation - you might need to enhance based on your network
            for vehicle_id in traci.vehicle.getIDList():
                route = traci.vehicle.getRoute(vehicle_id)
                if len(route) >= 2:
                    # Determine direction based on route
                    # This is a placeholder - implement based on your specific network
                    flow["NS"] += 1.0  # Simplified
        except Exception as e:
            print(f"Error getting directional flow: {e}")
        return flow
    
    def _get_signal_phase_timing(self) -> Dict[str, float]:
        """Get signal phase timing information"""
        timing = {}
        try:
            signal_info = self.get_signal_info()
            timing["current_phase"] = float(signal_info.get("current_phase", 0))
            timing["phase_duration"] = signal_info.get("phase_duration", 30.0)
            timing["time_in_phase"] = signal_info.get("time_in_phase", 0.0)
            timing["time_to_next_phase"] = timing["phase_duration"] - timing["time_in_phase"]
        except Exception as e:
            print(f"Error getting signal phase timing: {e}")
        return timing
    
    def _get_congestion_per_lane(self) -> Dict[str, str]:
        """Get congestion level for each lane"""
        congestion = {}
        try:
            for edge_id in self.edge_ids:
                for lane_idx in range(traci.edge.getLaneNumber(edge_id)):
                    lane_id = f"{edge_id}_{lane_idx}"
                    queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
                    
                    if queue_length == 0:
                        congestion[lane_id] = "low"
                    elif queue_length <= 3:
                        congestion[lane_id] = "moderate"
                    elif queue_length <= 8:
                        congestion[lane_id] = "high"
                    else:
                        congestion[lane_id] = "severe"
        except Exception as e:
            print(f"Error getting congestion per lane: {e}")
        return congestion
    
    def _get_emergency_vehicles(self) -> List[str]:
        """Get list of emergency vehicles"""
        emergency_vehicles = []
        try:
            for vehicle_id in traci.vehicle.getIDList():
                vehicle_type = traci.vehicle.getTypeID(vehicle_id)
                if "emergency" in vehicle_type.lower() or "ambulance" in vehicle_type.lower():
                    emergency_vehicles.append(vehicle_id)
        except Exception as e:
            print(f"Error getting emergency vehicles: {e}")
        return emergency_vehicles
    
    def _get_pedestrian_waiting(self) -> int:
        """Get number of waiting pedestrians"""
        try:
            # This is a placeholder - implement based on your network
            return 0
        except Exception as e:
            print(f"Error getting pedestrian waiting: {e}")
            return 0
    
    def change_signal_phase(self, phase_id: int, duration: float = None) -> bool:
        """Change traffic signal phase"""
        try:
            if self.simulation_state != SimulationState.RUNNING:
                return False
            
            # Validate phase ID
            if phase_id not in self.phases:
                print(f"Error: Invalid phase ID {phase_id}. Valid phases: {list(self.phases.keys())}")
                return False
            
            # Set the phase
            traci.trafficlight.setPhase(self.traffic_light_id, phase_id)
            
            # Set duration - use default from phase definition if not provided
            if duration is None:
                duration = self.phases[phase_id]["duration"]
            
            traci.trafficlight.setPhaseDuration(self.traffic_light_id, duration)
            
            phase_name = self.get_action_name(phase_id)
            print(f"Changed to phase {phase_id} ({phase_name}) with duration {duration}s")
            
            return True
            
        except Exception as e:
            print(f"Error changing signal phase: {e}")
            return False
    
    def execute_action(self, action: int) -> bool:
        """Execute an action (0-3) corresponding to our phase system"""
        try:
            if self.simulation_state != SimulationState.RUNNING:
                return False
            
            # Validate action
            if action not in self.phases:
                print(f"Error: Invalid action {action}. Valid actions: {list(self.phases.keys())}")
                return False
            
            # Execute the action by setting the corresponding phase
            return self.change_signal_phase(action)
            
        except Exception as e:
            print(f"Error executing action: {e}")
            return False
    
    def get_signal_info(self) -> Dict[str, Any]:
        """Get current traffic signal information"""
        try:
            current_phase = traci.trafficlight.getPhase(self.traffic_light_id)
            return {
                "current_phase": current_phase,
                "phase_duration": traci.trafficlight.getPhaseDuration(self.traffic_light_id),
                "phase_name": traci.trafficlight.getPhaseName(self.traffic_light_id),
                "program_id": traci.trafficlight.getProgram(self.traffic_light_id),
                "state": traci.trafficlight.getRedYellowGreenState(self.traffic_light_id),
                "phase_info": self.phases.get(current_phase, {})
            }
        except Exception as e:
            print(f"Error getting signal info: {e}")
            return {}
    
    def get_12d_state_vector(self) -> np.ndarray:
        """Get 12-dimensional state vector for RL agent"""
        try:
            if self.simulation_state != SimulationState.RUNNING:
                return np.zeros(12, dtype=np.float32)
            
            # Use our state utils to get the 12D state vector
            state_vector = get_12d_state_vector()
            return state_vector
            
        except Exception as e:
            print(f"Error getting 12D state vector: {e}")
            return np.zeros(12, dtype=np.float32)
    
    def get_simple_reward(self, prev_waiting_time: float = None, prev_queue_length: int = None) -> float:
        """Calculate simple reward using our reward function"""
        try:
            if self.simulation_state != SimulationState.RUNNING:
                return 0.0
            
            # Get current state summary
            state_summary = get_state_summary()
            current_waiting_time = state_summary.get('total_waiting_time', 0.0)
            current_queue_length = state_summary.get('total_queue_length', 0)
            
            if prev_waiting_time is not None and prev_queue_length is not None:
                # Use direct calculation
                return simple_reward(prev_waiting_time, current_waiting_time, current_queue_length)
            else:
                # Use the integrated reward calculator
                return calculate_reward(
                    current_waiting_time,
                    current_queue_length,
                    state_summary.get('avg_speed', 0.0),
                    state_summary.get('total_vehicles', 0)
                )
                
        except Exception as e:
            print(f"Error calculating reward: {e}")
            return 0.0
    
    def get_action_name(self, action: int) -> str:
        """Get human-readable name for an action"""
        return self.phases.get(action, {}).get('name', f'Action_{action}')
    
    def get_phase_info(self, phase_id: int) -> Dict[str, Any]:
        """Get information about a specific phase"""
        return self.phases.get(phase_id, {})
    
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
    
    def reset_reward_calculator(self):
        """Reset the reward calculator for a new episode"""
        try:
            reset_reward_calculator()
            print("Reward calculator reset")
        except Exception as e:
            print(f"Error resetting reward calculator: {e}")
    
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
    
    print("🚀 TraciManager Demo with Updated Features")
    print("=" * 50)
    
    # Show phase definitions
    print("\n📋 Phase Definitions:")
    for phase_id, phase_info in manager.phases.items():
        print(f"   Phase {phase_id}: {phase_info['name']} ({phase_info['description']})")
    
    # Show action names
    print(f"\n🎯 Action Names:")
    for action in range(4):
        action_name = manager.get_action_name(action)
        print(f"   Action {action}: {action_name}")
    
    # Run a simple simulation
    print(f"\n🚦 Starting simulation...")
    success = manager.run_simulation_for_duration(100, step_size=10)
    
    if success:
        print("✅ Simulation completed successfully")
        
        # Demonstrate state vector and reward calculation
        print(f"\n📊 Final State:")
        state_vector = manager.get_12d_state_vector()
        print(f"   State vector shape: {state_vector.shape}")
        print(f"   State vector: {state_vector}")
        
        reward = manager.get_simple_reward()
        print(f"   Final reward: {reward:.3f}")
        
    else:
        print("❌ Simulation failed")
