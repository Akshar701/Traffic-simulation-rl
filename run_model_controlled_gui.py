#!/usr/bin/env python3
"""
Run SUMO GUI with DQN Model Control
===================================

This script loads your trained DQN model and uses it to control traffic lights
in real-time while showing the SUMO GUI visualization.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import traci
import subprocess
import signal
from typing import Dict, List, Optional

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.dqn_agent import DQNAgent
from utils.state_utils import get_12d_state_vector, get_state_summary
from utils.reward_utils import calculate_reward, reset_reward_calculator

class ModelControlledGUI:
    """GUI runner with DQN model controlling traffic lights"""
    
    def __init__(self, model_path: str, config_file: str):
        self.model_path = model_path
        self.config_file = config_file
        
        # Initialize agent
        self.agent = DQNAgent(
            state_size=12,
            action_size=4,
            hidden_size=256,
            device='auto'
        )
        
        # Load the trained model
        self.load_model()
        
        # Set epsilon to 0 for deterministic behavior
        self.agent.epsilon = 0.0
        
        # Simulation state
        self.simulation_running = False
        self.current_step = 0
        self.total_reward = 0.0
        
        print(f"✅ Trained model loaded from: {model_path}")
        print(f"🚦 SUMO config: {self.config_file}")
        print(f"🎯 Agent epsilon set to 0.0 for deterministic behavior")
    
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.agent.load(self.model_path)
        print(f"📊 Model loaded successfully")
        print(f"   • Epsilon: {self.agent.epsilon:.3f}")
        print(f"   • Step count: {self.agent.step_count}")
    
    def start_gui_simulation(self):
        """Start SUMO GUI simulation with TraCI"""
        try:
            # Check if config file exists
            if not os.path.exists(self.config_file):
                raise FileNotFoundError(f"Config file not found: {self.config_file}")
            
            # Start SUMO GUI with traci
            sumo_cmd = ["sumo-gui", "-c", self.config_file, "--start"]
            traci.start(sumo_cmd)
            
            self.simulation_running = True
            print(f"🎬 SUMO GUI started with config: {self.config_file}")
            
        except Exception as e:
            print(f"❌ Error starting SUMO GUI: {e}")
            raise
    
    def get_current_state(self) -> np.ndarray:
        """Get current state from SUMO"""
        try:
            # Use the global state extractor
            state = get_12d_state_vector()
            return state
        except Exception as e:
            print(f"Warning: Error getting state: {e}")
            return np.zeros(12, dtype=np.float32)
    
    def execute_model_action(self, action: int):
        """Execute the action chosen by the model"""
        if not self.simulation_running:
            return
        
        try:
            # Action mapping (same as in traffic_env.py)
            action_to_phase = {
                0: 0,  # Action 0 -> Phase 0 (NS_Left_Straight)
                1: 2,  # Action 1 -> Phase 2 (EW_Left_Straight)
                2: 4,  # Action 2 -> Phase 4 (NS_Right)
                3: 6   # Action 3 -> Phase 6 (EW_Right)
            }
            
            tl_ids = traci.trafficlight.getIDList()
            if tl_ids:
                tl_id = tl_ids[0]
                phase = action_to_phase.get(action, 0)
                traci.trafficlight.setPhase(tl_id, phase)
                
                # Print action info
                action_names = {
                    0: "North-South Green",
                    1: "East-West Green", 
                    2: "North-South Right",
                    3: "East-West Right"
                }
                print(f"🤖 Model Action: {action} ({action_names.get(action, 'Unknown')}) -> Phase {phase}")
                
        except Exception as e:
            print(f"Warning: Error executing action {action}: {e}")
    
    def calculate_step_reward(self) -> float:
        """Calculate reward for current step"""
        try:
            # Get state summary for reward calculation
            state_summary = get_state_summary()
            
            # Extract relevant metrics
            waiting_time = state_summary.get('total_waiting_time', 0.0)
            queue_length = int(state_summary.get('total_queue_length', 0))
            avg_speed = state_summary.get('avg_speed', 0.0)
            vehicle_count = int(state_summary.get('total_vehicles', 0))
            
            # Calculate reward
            reward = calculate_reward(waiting_time, queue_length, avg_speed, vehicle_count)
            return reward
            
        except Exception as e:
            print(f"Warning: Error calculating reward: {e}")
            return 0.0
    
    def run_model_controlled_simulation(self, max_steps: int = 1000, step_delay: float = 0.5):
        """Run simulation with model controlling traffic lights"""
        print(f"\n🎬 Starting Model-Controlled SUMO GUI...")
        print(f"   • Max steps: {max_steps}")
        print(f"   • Step delay: {step_delay}s")
        print(f"   • Press Ctrl+C to stop the simulation")
        print("-" * 60)
        
        try:
            # Start GUI simulation
            self.start_gui_simulation()
            
            # Reset reward calculator
            reset_reward_calculator()
            
            # Get initial state
            state = self.get_current_state()
            
            print("🚀 Starting model-controlled simulation...")
            print("   • Watch the traffic lights change based on AI decisions")
            print("   • The model will make decisions every few seconds")
            print("   • You can see the action being taken in the terminal")
            
            while self.current_step < max_steps and self.simulation_running:
                # Get action from trained model
                action = self.agent.act(state)
                
                # Execute the action (change traffic light)
                self.execute_model_action(action)
                
                # Advance simulation by one step
                traci.simulationStep()
                
                # Calculate reward
                reward = self.calculate_step_reward()
                self.total_reward += reward
                
                # Get new state
                state = self.get_current_state()
                
                self.current_step += 1
                
                # Print progress every 10 steps
                if self.current_step % 10 == 0:
                    print(f"Step {self.current_step:3d}: Reward={reward:6.2f}, Total={self.total_reward:7.2f}")
                
                # Check if simulation is done
                if traci.simulation.getMinExpectedNumber() == 0:
                    print(f"✅ Simulation completed at step {self.current_step}")
                    break
                
                # Delay for better visualization
                time.sleep(step_delay)
        
        except KeyboardInterrupt:
            print(f"\n⏹️  Simulation stopped by user")
            print(f"   • Total steps: {self.current_step}")
            print(f"   • Total reward: {self.total_reward:.2f}")
        
        except Exception as e:
            print(f"❌ Error during simulation: {e}")
        
        finally:
            # Close SUMO
            self.close()
    
    def close(self):
        """Close the simulation"""
        if self.simulation_running:
            try:
                traci.close()
                self.simulation_running = False
                print("🔒 SUMO simulation closed")
            except:
                pass

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run SUMO GUI with DQN model control')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to trained model (.pth file)')
    parser.add_argument('--config', type=str, 
                       default='Sumo_env/gpt_newint/intersection.sumocfg',
                       help='SUMO configuration file')
    parser.add_argument('--max-steps', type=int, default=200, 
                       help='Maximum steps per episode')
    parser.add_argument('--step-delay', type=float, default=1.0, 
                       help='Delay between steps in seconds')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        print(f"Available models in trained_models/:")
        models_dir = "trained_models"
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith('.pth'):
                    print(f"   • {os.path.join(models_dir, file)}")
        sys.exit(1)
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"❌ Error: Config file not found: {args.config}")
        print(f"Available configs in Sumo_env/gpt_newint/:")
        config_dir = "Sumo_env/gpt_newint"
        if os.path.exists(config_dir):
            for file in os.listdir(config_dir):
                if file.endswith('.sumocfg'):
                    print(f"   • {os.path.join(config_dir, file)}")
        sys.exit(1)
    
    try:
        # Create runner
        runner = ModelControlledGUI(args.model, args.config)
        
        # Run with model control
        runner.run_model_controlled_simulation(args.max_steps, args.step_delay)
    
    except Exception as e:
        print(f"❌ Error running simulation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
