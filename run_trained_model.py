#!/usr/bin/env python3
"""
Run SUMO with Trained DQN Model
===============================

This script loads a trained DQN model and runs it with SUMO traffic simulation.
It provides options for different simulation scenarios and visualization.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
from typing import Dict, List, Optional
import json
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.dqn_agent import DQNAgent
from envs.traffic_env import TrafficEnv

class TrainedModelRunner:
    """Runner for trained DQN models with SUMO"""
    
    def __init__(self, model_path: str, config_file: str = None):
        self.model_path = model_path
        self.config_file = config_file or "Sumo_env/gpt_newint/intersection.sumocfg"
        
        # Initialize environment
        self.env = TrafficEnv(
            config_file=self.config_file,
            max_steps=1000
        )
        
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
    
    def run_single_episode(self, max_steps: int = 1000, verbose: bool = True) -> Dict:
        """Run a single episode with the trained model"""
        if verbose:
            print(f"\n🚀 Starting episode with trained model...")
            print(f"   • Max steps: {max_steps}")
            print(f"   • Config: {self.config_file}")
            print("-" * 60)
        
        # Reset environment
        state = self.env.reset()
        total_reward = 0
        step_rewards = []
        actions_taken = []
        step_info = []
        
        for step in range(max_steps):
            # Get action from trained model
            action = self.agent.act(state)
            
            # Take action in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Store data
            actions_taken.append(action)
            step_rewards.append(reward)
            total_reward += reward
            
            # Store step information
            step_info.append({
                'step': step + 1,
                'action': action,
                'reward': reward,
                'total_reward': total_reward,
                'state': state.tolist(),
                'info': info
            })
            
            if verbose and (step + 1) % 50 == 0:
                print(f"Step {step + 1:3d}: Action={action}, Reward={reward:6.2f}, Total={total_reward:7.2f}")
            
            state = next_state
            
            if done:
                if verbose:
                    print(f"Episode completed at step {step + 1}")
                break
        
        # Calculate action distribution
        action_counts = {}
        for action in actions_taken:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        episode_results = {
            'total_reward': total_reward,
            'total_steps': len(actions_taken),
            'mean_reward': np.mean(step_rewards),
            'action_counts': action_counts,
            'action_distribution': {
                f'action_{i}': (action_counts.get(i, 0) / len(actions_taken)) * 100 
                for i in range(4)
            },
            'step_rewards': step_rewards,
            'step_info': step_info
        }
        
        if verbose:
            self.print_episode_summary(episode_results)
        
        return episode_results
    
    def print_episode_summary(self, results: Dict):
        """Print episode summary"""
        print(f"\n📊 EPISODE SUMMARY:")
        print(f"   🎯 Total Reward: {results['total_reward']:.2f}")
        print(f"   ⏱️  Total Steps: {results['total_steps']}")
        print(f"   📈 Mean Reward: {results['mean_reward']:.2f}")
        
        print(f"\n🚦 Action Distribution:")
        for i in range(4):
            percentage = results['action_distribution'][f'action_{i}']
            print(f"   Action {i}: {percentage:.1f}%")
        
        # Performance assessment
        if results['total_reward'] > -20:
            performance = "🟢 EXCELLENT"
        elif results['total_reward'] > -50:
            performance = "🟡 GOOD"
        elif results['total_reward'] > -100:
            performance = "🟠 FAIR"
        else:
            performance = "🔴 POOR"
        
        print(f"\n🎯 Performance: {performance}")
        print("=" * 60)
    
    def run_multiple_episodes(self, num_episodes: int = 5, max_steps: int = 1000) -> List[Dict]:
        """Run multiple episodes and return results"""
        print(f"\n🚀 Running {num_episodes} episodes with trained model...")
        print("=" * 60)
        
        all_results = []
        total_rewards = []
        
        for episode in range(num_episodes):
            print(f"\n📺 Episode {episode + 1}/{num_episodes}")
            results = self.run_single_episode(max_steps, verbose=False)
            all_results.append(results)
            total_rewards.append(results['total_reward'])
            
            # Print episode summary
            print(f"   🎯 Reward: {results['total_reward']:7.2f}")
            print(f"   ⏱️  Steps: {results['total_steps']:3d}")
            print(f"   📈 Mean: {results['mean_reward']:7.2f}")
        
        # Print overall summary
        print(f"\n📊 OVERALL SUMMARY ({num_episodes} episodes):")
        print(f"   🎯 Mean Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"   📈 Best Episode: {max(total_rewards):.2f}")
        print(f"   📉 Worst Episode: {min(total_rewards):.2f}")
        print(f"   ⏱️  Mean Steps: {np.mean([r['total_steps'] for r in all_results]):.1f}")
        
        # Overall performance assessment
        mean_reward = np.mean(total_rewards)
        if mean_reward > -20:
            performance = "🟢 EXCELLENT"
        elif mean_reward > -50:
            performance = "🟡 GOOD"
        elif mean_reward > -100:
            performance = "🟠 FAIR"
        else:
            performance = "🔴 POOR"
        
        print(f"   🎯 Overall Performance: {performance}")
        print("=" * 60)
        
        return all_results
    
    def save_results(self, results: List[Dict], filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"trained_model_results_{timestamp}.json"
        
        # Create results directory if it doesn't exist
        results_dir = "training_results"
        os.makedirs(results_dir, exist_ok=True)
        
        filepath = os.path.join(results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump({
                'model_path': self.model_path,
                'config_file': self.config_file,
                'timestamp': datetime.now().isoformat(),
                'results': results
            }, f, indent=2)
        
        print(f"💾 Results saved to: {filepath}")
    
    def run_with_visualization(self, max_steps: int = 1000):
        """Run with SUMO GUI visualization"""
        print(f"\n🎬 Running with SUMO GUI visualization...")
        print(f"   • This will open the SUMO GUI")
        print(f"   • Press Ctrl+C to stop the simulation")
        print("-" * 60)
        
        try:
            # Reset environment with GUI
            state = self.env.reset()
            total_reward = 0
            step = 0
            
            print("🚀 Starting visualization...")
            
            while step < max_steps:
                # Get action from trained model
                action = self.agent.act(state)
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                total_reward += reward
                step += 1
                
                if step % 50 == 0:
                    print(f"Step {step}: Action={action}, Reward={reward:.2f}, Total={total_reward:.2f}")
                
                state = next_state
                
                if done:
                    print(f"Episode completed at step {step}")
                    break
                
                # Small delay for visualization
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print(f"\n⏹️  Simulation stopped by user")
            print(f"   • Total steps: {step}")
            print(f"   • Total reward: {total_reward:.2f}")
        
        finally:
            # Close SUMO
            self.env.close()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run SUMO with trained DQN model')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to trained model (.pth file)')
    parser.add_argument('--config', type=str, 
                       default='Sumo_env/gpt_newint/intersection.sumocfg',
                       help='SUMO configuration file')
    parser.add_argument('--episodes', type=int, default=1, 
                       help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=1000, 
                       help='Maximum steps per episode')
    parser.add_argument('--visualize', action='store_true', 
                       help='Run with SUMO GUI visualization')
    parser.add_argument('--save-results', action='store_true', 
                       help='Save results to JSON file')
    parser.add_argument('--verbose', action='store_true', 
                       help='Verbose output')
    
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
        runner = TrainedModelRunner(args.model, args.config)
        
        if args.visualize:
            # Run with visualization
            runner.run_with_visualization(args.max_steps)
        else:
            # Run episodes
            if args.episodes == 1:
                results = [runner.run_single_episode(args.max_steps, args.verbose)]
            else:
                results = runner.run_multiple_episodes(args.episodes, args.max_steps)
            
            # Save results if requested
            if args.save_results:
                runner.save_results(results)
    
    except Exception as e:
        print(f"❌ Error running simulation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
