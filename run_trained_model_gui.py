#!/usr/bin/env python3
"""
Run SUMO GUI with Trained DQN Model
===================================

This script loads a trained DQN model and runs it with SUMO GUI visualization.
It properly starts sumo-gui instead of sumo for GUI visualization.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import traci
import subprocess
from typing import Dict, List, Optional
import json
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.dqn_agent import DQNAgent
from utils.state_utils import get_12d_state_vector, get_state_summary
from utils.reward_utils import calculate_reward, reset_reward_calculator

class TrainedModelGUIRunner:
    """Runner for trained DQN models with SUMO GUI"""
    
    def __init__(self, model_path: str, config_file: str = None):
        self.model_path = model_path
        self.config_file = config_file or "Sumo_env/gpt_newint/intersection.sumocfg"
        
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
        """Start SUMO GUI simulation"""
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
    
    def get_state(self) -> np.ndarray:
        """Get current state from SUMO"""
        try:
            # Get traffic light information
            tl_ids = traci.trafficlight.getIDList()
            if not tl_ids:
                return np.zeros(12, dtype=np.float32)
            
            tl_id = tl_ids[0]  # Use first traffic light
            
            # Get state using the same function as the environment
            state = get_12d_state_vector(tl_id)
            return state
            
        except Exception as e:
            print(f"Warning: Error getting state: {e}")
            return np.zeros(12, dtype=np.float32)
    
    def execute_action(self, action: int):
        """Execute the given action"""
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
                
        except Exception as e:
            print(f"Warning: Error executing action {action}: {e}")
    
    def get_reward(self) -> float:
        """Calculate reward for current state"""
        try:
            tl_ids = traci.trafficlight.getIDList()
            if not tl_ids:
                return 0.0
            
            tl_id = tl_ids[0]
            reward = calculate_reward(tl_id)
            return reward
            
        except Exception as e:
            print(f"Warning: Error calculating reward: {e}")
            return 0.0
    
    def step(self, action: int) -> tuple:
        """Execute one step in the simulation"""
        if not self.simulation_running:
            return np.zeros(12), 0.0, True, {}
        
        # Execute action
        self.execute_action(action)
        
        # Advance simulation
        traci.simulationStep()
        
        # Get new state and reward
        next_state = self.get_state()
        reward = self.get_reward()
        
        # Check if simulation is done
        done = traci.simulation.getMinExpectedNumber() == 0
        
        self.current_step += 1
        
        return next_state, reward, done, {}
    
    def run_with_gui(self, max_steps: int = 1000):
        """Run the trained model with SUMO GUI"""
        print(f"\n🎬 Starting SUMO GUI with trained model...")
        print(f"   • Max steps: {max_steps}")
        print(f"   • Press Ctrl+C to stop the simulation")
        print("-" * 60)
        
        try:
            # Start GUI simulation
            self.start_gui_simulation()
            
            # Reset reward calculator
            reset_reward_calculator()
            
            # Get initial state
            state = self.get_state()
            total_reward = 0
            step = 0
            
            print("🚀 Starting simulation with trained model...")
            print("   • Watch the traffic lights change based on AI decisions")
            print("   • Green = AI chose this direction")
            print("   • Red = AI chose a different direction")
            
            while step < max_steps and self.simulation_running:
                # Get action from trained model
                action = self.agent.act(state)
                
                # Take action
                next_state, reward, done, info = self.step(action)
                
                total_reward += reward
                step += 1
                
                # Print progress every 50 steps
                if step % 50 == 0:
                    print(f"Step {step:3d}: Action={action}, Reward={reward:6.2f}, Total={total_reward:7.2f}")
                
                state = next_state
                
                if done:
                    print(f"✅ Episode completed at step {step}")
                    break
                
                # Small delay for better visualization
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print(f"\n⏹️  Simulation stopped by user")
            print(f"   • Total steps: {step}")
            print(f"   • Total reward: {total_reward:.2f}")
        
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
    parser = argparse.ArgumentParser(description='Run SUMO GUI with trained DQN model')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to trained model (.pth file)')
    parser.add_argument('--config', type=str, 
                       default='Sumo_env/gpt_newint/intersection.sumocfg',
                       help='SUMO configuration file')
    parser.add_argument('--max-steps', type=int, default=1000, 
                       help='Maximum steps per episode')
    
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
        runner = TrainedModelGUIRunner(args.model, args.config)
        
        # Run with GUI
        runner.run_with_gui(args.max_steps)
    
    except Exception as e:
        print(f"❌ Error running simulation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
