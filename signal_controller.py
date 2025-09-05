#!/usr/bin/env python3
"""
Signal Controller - Dynamic traffic light management
Provides intelligent signal control based on real-time traffic conditions
"""

import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from traci_manager import TraciManager, TrafficState
import numpy as np

# Import our state and reward utilities
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.state_utils import get_12d_state_vector, get_state_summary
from utils.reward_utils import calculate_reward, simple_reward, reset_reward_calculator

class SignalPhase(Enum):
    """Traffic signal phases - simplified 4-phase system"""
    NS_LEFT_STRAIGHT = 0  # North-South left-turn + straight lanes green
    NS_YELLOW = 1         # North-South yellow transition
    EW_LEFT_STRAIGHT = 2  # East-West left-turn + straight lanes green
    EW_YELLOW = 3         # East-West yellow transition

@dataclass
class SignalDecision:
    """Signal control decision"""
    phase_id: int
    duration: float
    reason: str
    priority: int  # Higher number = higher priority

class SignalController:
    """Intelligent traffic signal controller"""
    
    def __init__(self, traci_manager: TraciManager):
        self.traci_manager = traci_manager
        self.current_phase = 0
        self.phase_duration = 30.0
        self.min_phase_duration = 10.0
        self.max_phase_duration = 60.0
        self.yellow_duration = 3.0
        
        # Control parameters
        self.queue_threshold = 5
        self.waiting_time_threshold = 30.0
        self.emergency_vehicle_detected = False
        
        # Phase information matching tls.tll.xml (8 phases total)
        self.phase_info = {
            0: {"name": "NS_Straight_Left", "description": "North-South straight + left lanes green", "default_duration": 30},
            1: {"name": "NS_Yellow", "description": "North-South yellow transition", "default_duration": 4},
            2: {"name": "EW_Straight_Left", "description": "East-West straight + left lanes green", "default_duration": 30},
            3: {"name": "EW_Yellow", "description": "East-West yellow transition", "default_duration": 4},
            4: {"name": "NS_Protected_Right", "description": "North-South protected right turn", "default_duration": 15},
            5: {"name": "NS_Right_Yellow", "description": "North-South right turn yellow", "default_duration": 4},
            6: {"name": "EW_Protected_Right", "description": "East-West protected right turn", "default_duration": 15},
            7: {"name": "EW_Right_Yellow", "description": "East-West right turn yellow", "default_duration": 4}
        }
        
        # Action mapping: actions 0,1,2,3 correspond to phases 0,2,4,6 (skipping yellow phases)
        self.action_to_phase = {
            0: 0,  # Action 0 -> Phase 0 (NS_Straight_Left)
            1: 2,  # Action 1 -> Phase 2 (EW_Straight_Left)
            2: 4,  # Action 2 -> Phase 4 (NS_Protected_Right)
            3: 6   # Action 3 -> Phase 6 (EW_Protected_Right)
        }
        
        # Performance tracking
        self.phase_performance = {}
        self.decision_history = []
        
        # RL integration
        self.previous_waiting_time = 0.0
        self.previous_queue_length = 0
        
    def get_current_signal_info(self) -> Dict[str, Any]:
        """Get current signal information"""
        signal_info = self.traci_manager.get_signal_info()
        self.current_phase = signal_info.get("current_phase", 0)
        self.phase_duration = signal_info.get("phase_duration", 30.0)
        
        return {
            "current_phase": self.current_phase,
            "phase_name": self.phase_info.get(self.current_phase, {}).get("name", "Unknown"),
            "phase_duration": self.phase_duration,
            "phase_description": self.phase_info.get(self.current_phase, {}).get("description", ""),
            "signal_state": signal_info.get("state", "")
        }
    
    def make_decision(self, traffic_state: TrafficState) -> SignalDecision:
        """Make intelligent signal control decision with improved logic"""
        # Get current signal info
        signal_info = self.get_current_signal_info()
        current_phase = signal_info["current_phase"]
        
        # Check for emergency conditions first
        emergency_decision = self._check_emergency_conditions(traffic_state)
        if emergency_decision:
            return emergency_decision
        
        # Check if current phase should be extended (with improved logic)
        extend_decision = self._check_phase_extension_improved(traffic_state, current_phase)
        if extend_decision:
            return extend_decision
        
        # Check if phase should be skipped (with improved logic)
        skip_decision = self._check_phase_skip_improved(traffic_state, current_phase)
        if skip_decision:
            return skip_decision
        
        # Default: proceed to next phase with adaptive duration
        next_phase = self._get_next_phase(current_phase)
        duration = self._calculate_adaptive_duration(traffic_state, next_phase)
        
        return SignalDecision(
            phase_id=next_phase,
            duration=duration,
            reason="Normal progression",
            priority=1
        )
    
    def _check_emergency_conditions(self, traffic_state: TrafficState) -> Optional[SignalDecision]:
        """Check for emergency conditions that require immediate action"""
        # Check for severe congestion
        if traffic_state.queue_length > 20:
            return SignalDecision(
                phase_id=self._get_optimal_phase_for_congestion(traffic_state),
                duration=45.0,
                reason="Severe congestion detected",
                priority=10
            )
        
        # Check for very long waiting times
        if traffic_state.avg_waiting_time > 60:
            return SignalDecision(
                phase_id=self._get_optimal_phase_for_waiting(traffic_state),
                duration=40.0,
                reason="Long waiting times detected",
                priority=9
            )
        
        return None
    
    def _check_phase_extension_improved(self, traffic_state: TrafficState, current_phase: int) -> Optional[SignalDecision]:
        """Improved phase extension logic to prevent infinite loops"""
        # Don't extend if we've already been in this phase too long
        if self.phase_duration > self.max_phase_duration:
            return None
        
        # Check if current phase is serving heavy traffic
        if current_phase == 0:  # NS_Straight_Left phase
            # Check North-South traffic
            ns_queues = self._get_ns_queue_length(traffic_state)
            ns_waiting = self._get_ns_waiting_time(traffic_state)
            
            if ns_queues > self.queue_threshold and ns_waiting < self.waiting_time_threshold:
                # Extend but with a reasonable limit
                extension = min(15.0, self.max_phase_duration - self.phase_duration)
                if extension > 5.0:
                    return SignalDecision(
                        phase_id=current_phase,
                        duration=self.phase_duration + extension,
                        reason=f"NS traffic heavy (queues: {ns_queues})",
                        priority=3
                    )
        
        elif current_phase == 2:  # EW_Straight_Left phase
            # Check East-West traffic
            ew_queues = self._get_ew_queue_length(traffic_state)
            ew_waiting = self._get_ew_waiting_time(traffic_state)
            
            if ew_queues > self.queue_threshold and ew_waiting < self.waiting_time_threshold:
                # Extend but with a reasonable limit
                extension = min(15.0, self.max_phase_duration - self.phase_duration)
                if extension > 5.0:
                    return SignalDecision(
                        phase_id=current_phase,
                        duration=self.phase_duration + extension,
                        reason=f"EW traffic heavy (queues: {ew_queues})",
                        priority=3
                    )
        
        elif current_phase == 4:  # NS_Protected_Right phase
            # Check North-South right turn traffic
            ns_queues = self._get_ns_queue_length(traffic_state)
            if ns_queues > self.queue_threshold:
                extension = min(10.0, self.max_phase_duration - self.phase_duration)
                if extension > 3.0:
                    return SignalDecision(
                        phase_id=current_phase,
                        duration=self.phase_duration + extension,
                        reason=f"NS right turn traffic heavy (queues: {ns_queues})",
                        priority=3
                    )
        
        elif current_phase == 6:  # EW_Protected_Right phase
            # Check East-West right turn traffic
            ew_queues = self._get_ew_queue_length(traffic_state)
            if ew_queues > self.queue_threshold:
                extension = min(10.0, self.max_phase_duration - self.phase_duration)
                if extension > 3.0:
                    return SignalDecision(
                        phase_id=current_phase,
                        duration=self.phase_duration + extension,
                        reason=f"EW right turn traffic heavy (queues: {ew_queues})",
                        priority=3
                    )
        
        return None
    
    def _check_phase_skip_improved(self, traffic_state: TrafficState, current_phase: int) -> Optional[SignalDecision]:
        """Improved phase skip logic to prevent infinite loops"""
        # Don't skip if we haven't been in this phase long enough
        if self.phase_duration < self.min_phase_duration:
            return None
        
        # Check if current phase is serving very light traffic
        if current_phase == 0:  # NS_Straight_Left phase
            ns_queues = self._get_ns_queue_length(traffic_state)
            if ns_queues <= 1:  # Very light traffic
                next_phase = self._get_next_phase(current_phase)
                return SignalDecision(
                    phase_id=next_phase,
                    duration=30.0,
                    reason="NS traffic light, skipping",
                    priority=2
                )
        
        elif current_phase == 2:  # EW_Straight_Left phase
            ew_queues = self._get_ew_queue_length(traffic_state)
            if ew_queues <= 1:  # Very light traffic
                next_phase = self._get_next_phase(current_phase)
                return SignalDecision(
                    phase_id=next_phase,
                    duration=30.0,
                    reason="EW traffic light, skipping",
                    priority=2
                )
        
        elif current_phase == 4:  # NS_Protected_Right phase
            ns_queues = self._get_ns_queue_length(traffic_state)
            if ns_queues <= 1:  # Very light traffic
                next_phase = self._get_next_phase(current_phase)
                return SignalDecision(
                    phase_id=next_phase,
                    duration=15.0,
                    reason="NS right turn traffic light, skipping",
                    priority=2
                )
        
        elif current_phase == 6:  # EW_Protected_Right phase
            ew_queues = self._get_ew_queue_length(traffic_state)
            if ew_queues <= 1:  # Very light traffic
                next_phase = self._get_next_phase(current_phase)
                return SignalDecision(
                    phase_id=next_phase,
                    duration=15.0,
                    reason="EW right turn traffic light, skipping",
                    priority=2
                )
        
        return None
    
    def _get_next_phase_decision(self, current_phase: int) -> SignalDecision:
        """Get decision for next phase in sequence"""
        next_phase = self._get_next_phase(current_phase)
        default_duration = self.phase_info[next_phase]["default_duration"]
        
        return SignalDecision(
            phase_id=next_phase,
            duration=default_duration,
            reason="Normal phase progression",
            priority=1
        )
    
    def _has_waiting_vehicles_in_current_direction(self, phase: int, traffic_state: TrafficState) -> bool:
        """Check if there are waiting vehicles in the current signal direction"""
        # This is a simplified check - in a real system you'd check specific lanes
        # For now, we'll use overall queue length as a proxy
        return traffic_state.queue_length > self.queue_threshold
    
    def _has_waiting_vehicles_in_other_directions(self, current_phase: int, traffic_state: TrafficState) -> bool:
        """Check if there are waiting vehicles in other directions"""
        # Simplified check - in reality you'd check specific lanes for each direction
        return traffic_state.waiting_vehicles > 0
    
    def _get_optimal_phase_for_congestion(self, traffic_state: TrafficState) -> int:
        """Get optimal phase for reducing congestion"""
        # Get traffic metrics
        ns_queues = self._get_ns_queue_length(traffic_state)
        ew_queues = self._get_ew_queue_length(traffic_state)
        
        # Simple heuristic: choose phase with highest congestion
        if ns_queues > ew_queues + 2:  # NS has significantly more traffic
            return 0  # NS_Straight_Left
        elif ew_queues > ns_queues + 2:  # EW has significantly more traffic
            return 2  # EW_Straight_Left
        elif ns_queues > 5:  # High NS traffic, consider right turn phase
            return 4  # NS_Protected_Right
        elif ew_queues > 5:  # High EW traffic, consider right turn phase
            return 6  # EW_Protected_Right
        else:
            # Default: alternate between main phases
            current_phase = self.get_current_signal_info()["current_phase"]
            if current_phase in [0, 1]:  # Currently NS
                return 2  # Switch to EW_Straight_Left
            else:
                return 0  # Switch to NS_Straight_Left
    
    def _get_optimal_phase_for_waiting(self, traffic_state: TrafficState) -> int:
        """Get optimal phase for reducing waiting times"""
        # Similar to congestion optimization
        return self._get_optimal_phase_for_congestion(traffic_state)
    
    def _get_next_phase(self, current_phase: int) -> int:
        """Get next phase in the sequence - 8-phase system"""
        phase_sequence = [0, 1, 2, 3, 4, 5, 6, 7, 0]  # Circular sequence through all 8 phases
        current_index = phase_sequence.index(current_phase)
        return phase_sequence[current_index + 1]
    
    def _get_next_green_phase(self, current_phase: int) -> int:
        """Get next green phase - 8-phase system (phases 0, 2, 4, 6 are green)"""
        green_phases = [0, 2, 4, 6]  # NS_Straight_Left, EW_Straight_Left, NS_Protected_Right, EW_Protected_Right
        if current_phase in green_phases:
            current_index = green_phases.index(current_phase)
            return green_phases[(current_index + 1) % len(green_phases)]
        else:
            # If currently in yellow, go to the next green phase
            if current_phase in [1]:  # NS_Yellow
                return 2  # EW_Straight_Left
            elif current_phase in [3]:  # EW_Yellow
                return 4  # NS_Protected_Right
            elif current_phase in [5]:  # NS_Right_Yellow
                return 6  # EW_Protected_Right
            elif current_phase in [7]:  # EW_Right_Yellow
                return 0  # NS_Straight_Left
            else:
                return 0  # Default
    
    def execute_decision(self, decision: SignalDecision) -> bool:
        """Execute a signal control decision"""
        try:
            # Log the decision
            self.decision_history.append({
                "timestamp": time.time(),
                "phase_id": decision.phase_id,
                "duration": decision.duration,
                "reason": decision.reason,
                "priority": decision.priority
            })
            
            # Execute the decision
            success = self.traci_manager.change_signal_phase(
                decision.phase_id, 
                decision.duration
            )
            
            if success:
                print(f"Signal changed to phase {decision.phase_id} "
                      f"for {decision.duration}s - {decision.reason}")
            
            return success
            
        except Exception as e:
            print(f"Error executing signal decision: {e}")
            return False
    
    def run_adaptive_control(self, duration: int = 1000, step_size: int = 1):
        """Run adaptive signal control for a specified duration"""
        if not self.traci_manager.start_simulation():
            print("Failed to start simulation")
            return False
        
        # Reset reward calculator for new episode
        self.reset_reward_calculator()
        
        try:
            current_time = 0
            while current_time < duration and self.traci_manager.is_simulation_running():
                # Step simulation
                self.traci_manager.step_simulation(step_size)
                current_time += step_size
                
                # Get current traffic state
                traffic_state = self.traci_manager.get_traffic_state()
                if traffic_state:
                    # Update reward state
                    self.update_reward_state(traffic_state)
                    
                    # Make decision
                    decision = self.make_decision(traffic_state)
                    
                    # Execute decision if it's different from current state
                    signal_info = self.get_current_signal_info()
                    if decision.phase_id != signal_info["current_phase"]:
                        self.execute_decision(decision)
                
                # Small delay to make it observable
                time.sleep(0.1)
            
            return True
            
        except Exception as e:
            print(f"Error in adaptive control: {e}")
            return False
        finally:
            self.traci_manager.stop_simulation()
    
    def get_12d_state_vector(self) -> np.ndarray:
        """Get 12-dimensional state vector for RL agent"""
        try:
            return self.traci_manager.get_12d_state_vector()
        except Exception as e:
            print(f"Error getting 12D state vector: {e}")
            return np.zeros(12, dtype=np.float32)
    
    def get_simple_reward(self) -> float:
        """Calculate simple reward using our reward function"""
        try:
            return self.traci_manager.get_simple_reward(
                self.previous_waiting_time, 
                self.previous_queue_length
            )
        except Exception as e:
            print(f"Error calculating reward: {e}")
            return 0.0
    
    def update_reward_state(self, traffic_state: TrafficState):
        """Update previous state for reward calculation"""
        self.previous_waiting_time = traffic_state.avg_waiting_time
        self.previous_queue_length = traffic_state.queue_length
    
    def reset_reward_calculator(self):
        """Reset reward calculator for new episode"""
        try:
            self.traci_manager.reset_reward_calculator()
            self.previous_waiting_time = 0.0
            self.previous_queue_length = 0
            print("Signal controller reward state reset")
        except Exception as e:
            print(f"Error resetting reward calculator: {e}")
    
    def get_control_summary(self) -> Dict[str, Any]:
        """Get summary of signal control performance"""
        return {
            "total_decisions": len(self.decision_history),
            "current_phase": self.get_current_signal_info(),
            "recent_decisions": self.decision_history[-10:] if self.decision_history else [],
            "phase_performance": self.phase_performance,
            "rl_state": {
                "previous_waiting_time": self.previous_waiting_time,
                "previous_queue_length": self.previous_queue_length
            }
        }

    def _get_ns_queue_length(self, traffic_state: TrafficState) -> int:
        """Get North-South queue length"""
        if traffic_state.per_lane_queues:
            ns_queues = 0
            for lane_id, queue in traffic_state.per_lane_queues.items():
                if any(direction in lane_id for direction in ['3i', '4i']):  # North-South lanes
                    ns_queues += queue
            return ns_queues
        return 0
    
    def _get_ew_queue_length(self, traffic_state: TrafficState) -> int:
        """Get East-West queue length"""
        if traffic_state.per_lane_queues:
            ew_queues = 0
            for lane_id, queue in traffic_state.per_lane_queues.items():
                if any(direction in lane_id for direction in ['1i', '2i']):  # East-West lanes
                    ew_queues += queue
            return ew_queues
        return 0
    
    def _get_ns_waiting_time(self, traffic_state: TrafficState) -> float:
        """Get North-South average waiting time"""
        if traffic_state.per_lane_waiting_times:
            ns_times = []
            for lane_id, waiting_time in traffic_state.per_lane_waiting_times.items():
                if any(direction in lane_id for direction in ['3i', '4i']):  # North-South lanes
                    ns_times.append(waiting_time)
            return sum(ns_times) / len(ns_times) if ns_times else 0.0
        return 0.0
    
    def _get_ew_waiting_time(self, traffic_state: TrafficState) -> float:
        """Get East-West average waiting time"""
        if traffic_state.per_lane_waiting_times:
            ew_times = []
            for lane_id, waiting_time in traffic_state.per_lane_waiting_times.items():
                if any(direction in lane_id for direction in ['1i', '2i']):  # East-West lanes
                    ew_times.append(waiting_time)
            return sum(ew_times) / len(ew_times) if ew_times else 0.0
        return 0.0
    
    def _calculate_adaptive_duration(self, traffic_state: TrafficState, phase_id: int) -> float:
        """Calculate adaptive phase duration based on traffic conditions"""
        base_duration = 30.0
        
        # Adjust based on overall congestion
        if traffic_state.queue_length > 20:
            base_duration = 45.0
        elif traffic_state.queue_length > 10:
            base_duration = 35.0
        elif traffic_state.queue_length < 5:
            base_duration = 25.0
        
        # Adjust based on waiting time
        if traffic_state.avg_waiting_time > 20.0:
            base_duration += 10.0
        elif traffic_state.avg_waiting_time < 5.0:
            base_duration -= 5.0
        
        return max(self.min_phase_duration, min(self.max_phase_duration, base_duration))

# Example usage
if __name__ == "__main__":
    # Create traci manager and signal controller
    traci_manager = TraciManager()
    signal_controller = SignalController(traci_manager)
    
    print("🚦 SignalController Demo with Updated Features")
    print("=" * 60)
    
    # Show phase definitions
    print("\n📋 Phase Definitions:")
    for phase_id, phase_info in signal_controller.phase_info.items():
        print(f"   Phase {phase_id}: {phase_info['name']} ({phase_info['description']})")
    
    # Show signal phase enum
    print(f"\n🎯 Signal Phase Enum:")
    for phase in SignalPhase:
        print(f"   {phase.name} = {phase.value}")
    
    # Run adaptive control
    print(f"\n🚦 Starting adaptive signal control...")
    success = signal_controller.run_adaptive_control(duration=100, step_size=10)
    
    if success:
        print("✅ Adaptive control completed successfully")
        summary = signal_controller.get_control_summary()
        print(f"   Total decisions made: {summary['total_decisions']}")
        print(f"   Current phase: {summary['current_phase']}")
        
        # Demonstrate RL integration
        print(f"\n📊 RL Integration Demo:")
        state_vector = signal_controller.get_12d_state_vector()
        print(f"   State vector shape: {state_vector.shape}")
        print(f"   State vector: {state_vector}")
        
        reward = signal_controller.get_simple_reward()
        print(f"   Simple reward: {reward:.3f}")
        
    else:
        print("❌ Adaptive control failed")
