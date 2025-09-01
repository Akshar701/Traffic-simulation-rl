#!/usr/bin/env python3
"""
DQN Training Script
==================

Training script for the Enhanced DQN agent for traffic signal control.
Includes monitoring, logging, and evaluation capabilities.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from typing import Dict, List, Optional

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.dqn_agent import DQNAgent
from envs.traffic_env import TrafficEnv

class DQNTrainer:
    """DQN Training Manager"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.results_dir = config.get('results_dir', 'training_results')
        self.models_dir = config.get('models_dir', 'trained_models')
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Print GPU information
        self.print_gpu_info()
        
        # Initialize environment and agent
        self.env = TrafficEnv(
            config_file=config.get('config_file', 'Sumo_env/Single intersection lhd/uniform_simulation.sumocfg'),
            max_steps=config.get('max_steps', 1000)
        )
        
        self.agent = DQNAgent(
            state_size=config.get('state_size', 24),
            action_size=config.get('action_size', 4),
            hidden_size=config.get('hidden_size', 256),
            learning_rate=config.get('learning_rate', 1e-4),
            gamma=config.get('gamma', 0.99),
            epsilon=config.get('epsilon', 1.0),
            epsilon_min=config.get('epsilon_min', 0.01),
            epsilon_decay=config.get('epsilon_decay', 0.995),
            memory_size=config.get('memory_size', 10000),
            batch_size=config.get('batch_size', 32),
            target_update_freq=config.get('target_update_freq', 1000),
            device=config.get('device', 'auto'),
            mixed_precision=config.get('mixed_precision', True)
        )
        
        # Training tracking
        self.training_history = []
        self.evaluation_history = []
        self.start_time = None
        
    def print_gpu_info(self):
        """Print GPU information and optimization settings"""
        print("🚀 GPU Configuration:")
        print("=" * 40)
        
        if torch.cuda.is_available():
            print(f"✅ CUDA Available: {torch.version.cuda}")
            print(f"🎯 GPU Device: {torch.cuda.get_device_name()}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"🔧 CUDA Capability: {torch.cuda.get_device_capability()}")
            print(f"📊 GPU Count: {torch.cuda.device_count()}")
            
            # Memory info
            memory_allocated = torch.cuda.memory_allocated() / 1e6
            memory_reserved = torch.cuda.memory_reserved() / 1e6
            print(f"💾 Memory Allocated: {memory_allocated:.1f} MB")
            print(f"💾 Memory Reserved: {memory_reserved:.1f} MB")
            
            # Optimization settings
            print(f"⚡ cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
            print(f"⚡ cuDNN Deterministic: {torch.backends.cudnn.deterministic}")
            
        else:
            print("⚠️ CUDA not available - using CPU")
            print("💡 For optimal performance, install CUDA-enabled PyTorch")
        
        print("=" * 40)
        
    def print_training_header(self, episodes: int, eval_freq: int, save_freq: int):
        """Print training header with configuration"""
        print("🚦" + "="*60)
        print("🚀 DQN TRAINING FOR TRAFFIC SIGNAL CONTROL")
        print("="*60)
        print(f"📊 Training Configuration:")
        print(f"   • Episodes: {episodes}")
        print(f"   • Evaluation Frequency: Every {eval_freq} episodes")
        print(f"   • Model Save Frequency: Every {save_freq} episodes")
        print(f"   • Max Steps per Episode: {self.config.get('max_steps', 1000)}")
        print(f"   • Learning Rate: {self.config.get('learning_rate', 1e-4)}")
        print(f"   • Initial Epsilon: {self.config.get('epsilon', 1.0)}")
        print(f"   • Target Network Update: Every {self.config.get('target_update_freq', 1000)} steps")
        print("="*60)
        
    def print_episode_progress(self, episode: int, total_reward: float, steps: int, 
                              epsilon: float, recent_rewards: List[float], 
                              training_loss: Optional[float] = None):
        """Print detailed episode progress"""
        # Calculate performance metrics
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        reward_trend = "↗️" if len(recent_rewards) >= 2 and recent_rewards[-1] > recent_rewards[-2] else "↘️"
        
        # Determine performance level
        if total_reward > -20:
            performance = "🟢 EXCELLENT"
        elif total_reward > -50:
            performance = "🟡 GOOD"
        elif total_reward > -100:
            performance = "🟠 FAIR"
        else:
            performance = "🔴 POOR"
        
        # Get GPU memory info
        gpu_memory = ""
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1e6
            gpu_memory = f" | 🚀 GPU: {memory_mb:.1f}MB"
        
        # Print episode summary
        print(f"Episode {episode:4d} | {performance}")
        print(f"   📈 Reward: {total_reward:7.2f} {reward_trend}")
        print(f"   ⏱️  Steps: {steps:3d} | 🎯 Epsilon: {epsilon:.3f}{gpu_memory}")
        print(f"   📊 Avg (10): {avg_reward:7.2f}")
        
        if training_loss is not None:
            print(f"   🧠 Loss: {training_loss:.6f}")
        
        # Print action distribution if available
        if hasattr(self.agent, 'recent_actions') and self.agent.recent_actions:
            action_counts = {}
            for action in self.agent.recent_actions[-steps:]:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            action_dist = []
            for i in range(4):
                count = action_counts.get(i, 0)
                percentage = (count / steps) * 100 if steps > 0 else 0
                action_dist.append(f"{self.agent.get_action_name(i)}: {percentage:.1f}%")
            
            print(f"   🚦 Actions: {' | '.join(action_dist)}")
        
        print("-" * 60)
        
    def print_evaluation_results(self, episode: int, eval_results: Dict):
        """Print evaluation results in a user-friendly format"""
        mean_reward = eval_results['mean_reward']
        std_reward = eval_results['std_reward']
        mean_steps = eval_results['mean_steps']
        
        # Determine evaluation performance
        if mean_reward > -20:
            eval_performance = "🟢 EXCELLENT"
        elif mean_reward > -50:
            eval_performance = "🟡 GOOD"
        elif mean_reward > -100:
            eval_performance = "🟠 FAIR"
        else:
            eval_performance = "🔴 POOR"
        
        print(f"📊 EVALUATION at Episode {episode} | {eval_performance}")
        print(f"   🎯 Mean Reward: {mean_reward:7.2f} ± {std_reward:.2f}")
        print(f"   ⏱️  Mean Steps: {mean_steps:.1f}")
        print(f"   📈 Individual Episodes: {eval_results['episode_rewards']}")
        print("="*60)
        
    def print_training_summary(self, training_time: float, total_episodes: int):
        """Print training summary"""
        print("\n🎉 TRAINING COMPLETED!")
        print("="*60)
        print(f"⏱️  Total Time: {training_time:.2f} seconds")
        print(f"📊 Episodes Trained: {total_episodes}")
        print(f"🧠 Final Epsilon: {self.agent.epsilon:.3f}")
        print(f"💾 Models Saved: {len([f for f in os.listdir(self.models_dir) if f.endswith('.pth')])}")
        
        # Performance analysis
        if self.training_history:
            recent_rewards = [h['total_reward'] for h in self.training_history[-50:]]
            initial_avg = np.mean([h['total_reward'] for h in self.training_history[:50]])
            final_avg = np.mean(recent_rewards)
            improvement = final_avg - initial_avg
            
            print(f"\n📈 Performance Analysis:")
            print(f"   🎯 Initial Average Reward: {initial_avg:.2f}")
            print(f"   🎯 Final Average Reward: {final_avg:.2f}")
            print(f"   📈 Improvement: {improvement:+.2f}")
            
            if improvement > 0:
                print(f"   ✅ Agent is learning and improving!")
            else:
                print(f"   ⚠️  Agent may need more training or hyperparameter tuning")
        
        print("="*60)
        
    def train(self, episodes: int, eval_freq: int = 50, save_freq: int = 100):
        """
        Train the DQN agent with enhanced user feedback
        
        Args:
            episodes: Number of episodes to train
            eval_freq: Frequency of evaluation
            save_freq: Frequency of model saving
        """
        self.print_training_header(episodes, eval_freq, save_freq)
        
        self.start_time = time.time()
        recent_rewards = []
        
        print("🚀 Starting training...")
        print("💡 The agent will start with random exploration (epsilon = 1.0)")
        print("💡 As training progresses, it will learn and reduce exploration")
        print("💡 Watch for improving rewards and more consistent performance")
        print("-" * 60)
        
        for episode in range(episodes):
            # Train one episode
            total_reward, steps, step_rewards = self.agent.train_episode(
                self.env, max_steps=self.config.get('max_steps', 1000)
            )
            
            # Track recent rewards for trend analysis
            recent_rewards.append(total_reward)
            if len(recent_rewards) > 10:
                recent_rewards.pop(0)
            
            # Record training data
            training_data = {
                'episode': episode + 1,
                'total_reward': total_reward,
                'steps': steps,
                'epsilon': self.agent.epsilon,
                'mean_step_reward': np.mean(step_rewards) if step_rewards else 0,
                'timestamp': datetime.now().isoformat()
            }
            self.training_history.append(training_data)
            
            # Print progress every episode for better visibility
            self.print_episode_progress(
                episode + 1, total_reward, steps, self.agent.epsilon, 
                recent_rewards[-10:], None  # We'll add loss tracking later
            )
            
            # Evaluate periodically
            if (episode + 1) % eval_freq == 0:
                print(f"🔍 Running evaluation...")
                eval_results = self.evaluate(episodes=5)
                self.evaluation_history.append({
                    'episode': episode + 1,
                    'eval_results': eval_results,
                    'timestamp': datetime.now().isoformat()
                })
                
                self.print_evaluation_results(episode + 1, eval_results)
            
            # Save model periodically
            if (episode + 1) % save_freq == 0:
                model_path = os.path.join(self.models_dir, f"dqn_episode_{episode + 1}.pth")
                self.agent.save(model_path)
                print(f"💾 Model checkpoint saved: {model_path}")
                print("-" * 60)
        
        # Final save
        final_model_path = os.path.join(self.models_dir, "dqn_final.pth")
        self.agent.save(final_model_path)
        
        training_time = time.time() - self.start_time
        self.print_training_summary(training_time, episodes)
        
        # Save training results
        self.save_training_results()
        
        return self.training_history
    
    def evaluate(self, episodes: int = 5) -> Dict:
        """Evaluate the current agent"""
        print(f"🔍 Evaluating agent over {episodes} episodes...")
        eval_results = self.agent.evaluate(self.env, episodes=episodes)
        return eval_results
    
    def save_training_results(self):
        """Save training results and plots"""
        # Save training history
        history_path = os.path.join(self.results_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save evaluation history
        eval_path = os.path.join(self.results_dir, 'evaluation_history.json')
        with open(eval_path, 'w') as f:
            json.dump(self.evaluation_history, f, indent=2)
        
        # Create plots
        self.create_training_plots()
        
        print(f"📊 Training results saved to: {self.results_dir}")
    
    def create_training_plots(self):
        """Create training visualization plots"""
        if not self.training_history:
            return
        
        # Extract data
        episodes = [h['episode'] for h in self.training_history]
        rewards = [h['total_reward'] for h in self.training_history]
        epsilons = [h['epsilon'] for h in self.training_history]
        steps = [h['steps'] for h in self.training_history]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        ax1.plot(episodes, rewards, 'b-', alpha=0.6)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True, alpha=0.3)
        
        # Moving average of rewards
        window = min(50, len(rewards) // 10)
        if window > 1:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
            ax1.legend()
        
        # Epsilon decay
        ax2.plot(episodes, epsilons, 'g-')
        ax2.set_title('Epsilon Decay')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Epsilon')
        ax2.grid(True, alpha=0.3)
        
        # Episode steps
        ax3.plot(episodes, steps, 'm-', alpha=0.6)
        ax3.set_title('Episode Steps')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Steps')
        ax3.grid(True, alpha=0.3)
        
        # Evaluation results
        if self.evaluation_history:
            eval_episodes = [h['episode'] for h in self.evaluation_history]
            eval_rewards = [h['eval_results']['mean_reward'] for h in self.evaluation_history]
            eval_stds = [h['eval_results']['std_reward'] for h in self.evaluation_history]
            
            ax4.errorbar(eval_episodes, eval_rewards, yerr=eval_stds, 
                        fmt='o-', capsize=5, capthick=2)
            ax4.set_title('Evaluation Results')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Mean Reward')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.results_dir, 'training_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Training plots saved: {plot_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        self.agent.load(model_path)
        print(f"✅ Model loaded: {model_path}")
    
    def test_agent(self, episodes: int = 5):
        """Test the trained agent with detailed output"""
        print(f"🧪 Testing trained agent over {episodes} episodes...")
        print("="*60)
        
        test_results = []
        total_rewards = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            actions_taken = []
            
            while steps < self.config.get('max_steps', 1000):
                action = self.agent.act(state, training=False)
                next_state, reward, done, info = self.env.step(action)
                
                actions_taken.append(action)
                total_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            # Calculate action distribution
            action_counts = {}
            for action in actions_taken:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            test_results.append({
                'episode': episode + 1,
                'total_reward': total_reward,
                'steps': steps,
                'actions_taken': actions_taken,
                'action_counts': action_counts
            })
            
            total_rewards.append(total_reward)
            
            # Print episode results
            print(f"Episode {episode + 1}:")
            print(f"   📈 Reward: {total_reward:.2f}")
            print(f"   ⏱️  Steps: {steps}")
            
            # Print action distribution
            action_dist = []
            for i in range(4):
                count = action_counts.get(i, 0)
                percentage = (count / steps) * 100 if steps > 0 else 0
                action_dist.append(f"{self.agent.get_action_name(i)}: {percentage:.1f}%")
            print(f"   🚦 Actions: {' | '.join(action_dist)}")
            print("-" * 40)
        
        # Print summary
        avg_reward = np.mean(total_rewards)
        avg_steps = np.mean([r['steps'] for r in test_results])
        
        print(f"\n📊 TEST SUMMARY:")
        print(f"   🎯 Average Reward: {avg_reward:.2f}")
        print(f"   ⏱️  Average Steps: {avg_steps:.1f}")
        print(f"   📈 Best Episode: {max(total_rewards):.2f}")
        print(f"   📉 Worst Episode: {min(total_rewards):.2f}")
        
        # Performance assessment
        if avg_reward > -20:
            print(f"   🟢 Performance: EXCELLENT")
        elif avg_reward > -50:
            print(f"   🟡 Performance: GOOD")
        elif avg_reward > -100:
            print(f"   🟠 Performance: FAIR")
        else:
            print(f"   🔴 Performance: POOR")
        
        print("="*60)
        
        return test_results

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train DQN agent for traffic signal control')
    parser.add_argument('--episodes', type=int, default=500, help='Number of training episodes')
    parser.add_argument('--eval-freq', type=int, default=50, help='Evaluation frequency')
    parser.add_argument('--save-freq', type=int, default=100, help='Model save frequency')
    parser.add_argument('--config', type=str, default='uniform_simulation.sumocfg', 
                       help='SUMO config file')
    parser.add_argument('--load-model', type=str, help='Path to load existing model')
    parser.add_argument('--test-only', action='store_true', help='Only test the agent')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cuda', 'cpu'], help='Device to use for training')
    parser.add_argument('--no-mixed-precision', action='store_true', 
                       help='Disable mixed precision training (FP16)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--memory-size', type=int, default=10000, help='Experience replay buffer size')
    
    args = parser.parse_args()
    
        # Training configuration
    config = {
        'config_file': f'Sumo_env/Single intersection lhd/{args.config}',
        'max_steps': 1000,
        'state_size': 24,
        'action_size': 4,
        'hidden_size': 256,
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'epsilon': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'memory_size': args.memory_size,
        'batch_size': args.batch_size,
        'target_update_freq': 1000,
        'device': args.device,
        'mixed_precision': not args.no_mixed_precision,  # Enable mixed precision for GPU
        'results_dir': 'training_results',
        'models_dir': 'trained_models'
    }
    
    # Create trainer
    trainer = DQNTrainer(config)
    
    if args.load_model:
        trainer.load_model(args.load_model)
    
    if args.test_only:
        # Test only
        test_results = trainer.test_agent(episodes=10)
    else:
        # Train the agent
        training_history = trainer.train(
            episodes=args.episodes,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq
        )
        
        # Test the trained agent
        print("\n🧪 Testing trained agent...")
        test_results = trainer.test_agent(episodes=5)

if __name__ == "__main__":
    main()
