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
import csv
import logging
from datetime import datetime
from typing import Dict, List, Optional
from torch.utils.tensorboard import SummaryWriter

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.dqn_agent import DQNAgent
from envs.traffic_env import TrafficEnv

class DQNTrainer:
    """DQN Training Manager with TensorBoard Integration and Formal Logging"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.results_dir = config.get('results_dir', 'training_results')
        self.models_dir = config.get('models_dir', 'trained_models')
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Setup TensorBoard
        self._setup_tensorboard()
        
        # Log GPU information
        self._log_gpu_info()
        
        # Initialize environment and agent
        self.env = TrafficEnv(
            config_file=config.get('config_file', 'Sumo_env/gpt_newint/intersection.sumocfg'),
            max_steps=config.get('max_steps', 1000)
        )
        
        # Use DQNAgent
        self.agent = DQNAgent(
            state_size=config.get('state_size', 12),
            action_size=config.get('action_size', 4),
            hidden_size=config.get('hidden_size', 256),
            learning_rate=config.get('learning_rate', 1e-4),
            gamma=config.get('gamma', 0.99),
            epsilon=config.get('epsilon', 1.0),
            epsilon_min=config.get('epsilon_min', 0.01),
            epsilon_decay=config.get('epsilon_decay', 0.995),
            memory_size=config.get('memory_size', 10000),
            batch_size=config.get('batch_size', 64),
            target_update_freq=config.get('target_update_freq', 1000),
            device=config.get('device', 'auto'),
            mixed_precision=config.get('mixed_precision', True)
        )
        
        # Training tracking
        self.training_history = []
        self.evaluation_history = []
        self.start_time = None
        self.episode_count = 0
        self.step_count = 0
    
    def _setup_logging(self):
        """Setup formal logging framework"""
        # Create logs directory
        logs_dir = os.path.join(self.results_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(logs_dir, f'training_{timestamp}.log')
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🚀 Training session started - Log file: {log_file}")
        self.logger.info("=" * 60)
    
    def _setup_tensorboard(self):
        """Setup TensorBoard for experiment tracking"""
        # Create TensorBoard logs directory
        tb_logs_dir = os.path.join(self.results_dir, 'tensorboard_logs')
        os.makedirs(tb_logs_dir, exist_ok=True)
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(tb_logs_dir, f'run_{timestamp}')
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=run_dir)
        
        # Log hyperparameters
        self._log_hyperparameters()
        
        self.logger.info(f"📊 TensorBoard logging enabled - Run directory: {run_dir}")
        self.logger.info(f"📊 To view TensorBoard: tensorboard --logdir {tb_logs_dir}")
    
    def _log_hyperparameters(self):
        """Log hyperparameters to TensorBoard"""
        hparams = {
            'learning_rate': self.config.get('learning_rate', 1e-4),
            'gamma': self.config.get('gamma', 0.99),
            'epsilon': self.config.get('epsilon', 1.0),
            'epsilon_min': self.config.get('epsilon_min', 0.01),
            'epsilon_decay': self.config.get('epsilon_decay', 0.995),
            'memory_size': self.config.get('memory_size', 10000),
            'batch_size': self.config.get('batch_size', 64),
            'target_update_freq': self.config.get('target_update_freq', 1000),
            'hidden_size': self.config.get('hidden_size', 256),
            'max_steps': self.config.get('max_steps', 1000),
            'mixed_precision': self.config.get('mixed_precision', True)
        }
        
        # Log hyperparameters as text
        hparams_text = "\n".join([f"{k}: {v}" for k, v in hparams.items()])
        self.writer.add_text('Hyperparameters', hparams_text, 0)
        
        self.logger.info("📋 Hyperparameters logged to TensorBoard")
    
    def _log_gpu_info(self):
        """Log GPU information and optimization settings"""
        self.logger.info("🚀 GPU Configuration:")
        self.logger.info("=" * 40)
        
        if torch.cuda.is_available():
            self.logger.info(f"✅ CUDA Available: {torch.version.cuda}")
            self.logger.info(f"🎯 GPU Device: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            self.logger.info(f"🔧 CUDA Capability: {torch.cuda.get_device_capability()}")
            self.logger.info(f"📊 GPU Count: {torch.cuda.device_count()}")
            
            # Memory info
            memory_allocated = torch.cuda.memory_allocated() / 1e6
            memory_reserved = torch.cuda.memory_reserved() / 1e6
            self.logger.info(f"💾 Memory Allocated: {memory_allocated:.1f} MB")
            self.logger.info(f"💾 Memory Reserved: {memory_reserved:.1f} MB")
            
            # Optimization settings
            self.logger.info(f"⚡ cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
            self.logger.info(f"⚡ cuDNN Deterministic: {torch.backends.cudnn.deterministic}")
            
            # Log GPU info to TensorBoard
            self.writer.add_text('System/GPU_Device', torch.cuda.get_device_name(0), 0)
            self.writer.add_text('System/CUDA_Version', torch.version.cuda, 0)
            self.writer.add_scalar('System/GPU_Memory_GB', torch.cuda.get_device_properties(0).total_memory / 1e9, 0)
            
        else:
            self.logger.warning("⚠️ CUDA not available - using CPU")
            self.logger.info("💡 For optimal performance, install CUDA-enabled PyTorch")
            self.writer.add_text('System/Device', 'CPU', 0)
        
        self.logger.info("=" * 40)
        
    def _log_training_header(self, episodes: int, eval_freq: int, save_freq: int):
        """Log training header with configuration"""
        self.logger.info("🚦" + "="*60)
        self.logger.info("🚀 ADAPTIVE DQN TRAINING FOR TRAFFIC SIGNAL CONTROL")
        self.logger.info("="*60)
        self.logger.info(f"📊 Training Configuration:")
        self.logger.info(f"   • Episodes: {episodes}")
        self.logger.info(f"   • Evaluation Frequency: Every {eval_freq} episodes")
        self.logger.info(f"   • Model Save Frequency: Every {save_freq} episodes")
        self.logger.info(f"   • Max Steps per Episode: {self.config.get('max_steps', 1000)}")
        self.logger.info(f"   • Learning Rate: {self.config.get('learning_rate', 1e-4)}")
        self.logger.info(f"   • Initial Epsilon: {self.config.get('epsilon', 1.0)}")
        self.logger.info(f"   • Target Network Update: Every {self.config.get('target_update_freq', 1000)} steps")
        self.logger.info(f"   • Mixed Precision: {self.config.get('mixed_precision', True)}")
        self.logger.info(f"   • Performance Window: {self.config.get('performance_window', 50)} episodes")
        self.logger.info("="*60)
        
    def _log_episode_progress(self, episode: int, total_reward: float, steps: int, 
                              epsilon: float, recent_rewards: List[float], 
                              training_loss: Optional[float] = None):
        """Log detailed episode progress with TensorBoard integration"""
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
        
        # Log episode summary
        self.logger.info(f"Episode {episode:4d} | {performance}")
        self.logger.info(f"   📈 Reward: {total_reward:7.2f} {reward_trend}")
        self.logger.info(f"   ⏱️  Steps: {steps:3d} | 🎯 Epsilon: {epsilon:.3f}{gpu_memory}")
        self.logger.info(f"   📊 Avg (10): {avg_reward:7.2f}")
        
        # Log to TensorBoard
        self.writer.add_scalar('Training/Episode_Reward', total_reward, episode)
        self.writer.add_scalar('Training/Average_Reward_10', avg_reward, episode)
        self.writer.add_scalar('Training/Episode_Steps', steps, episode)
        self.writer.add_scalar('Training/Epsilon', epsilon, episode)
        
        if torch.cuda.is_available():
            self.writer.add_scalar('System/GPU_Memory_MB', torch.cuda.memory_allocated() / 1e6, episode)
        
        if training_loss is not None:
            self.logger.info(f"   🧠 Loss: {training_loss:.6f}")
            self.writer.add_scalar('Training/Loss', training_loss, episode)
        
        # Log action distribution if available
        if hasattr(self.agent, 'recent_actions') and self.agent.recent_actions:
            action_counts = {}
            for action in self.agent.recent_actions[-steps:]:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            action_dist = []
            for i in range(4):
                count = action_counts.get(i, 0)
                percentage = (count / steps) * 100 if steps > 0 else 0
                action_dist.append(f"Action {i}: {percentage:.1f}%")
                # Log individual action percentages to TensorBoard
                self.writer.add_scalar(f'Training/Action_{i}_Percentage', percentage, episode)
            
            self.logger.info(f"   🚦 Actions: {' | '.join(action_dist)}")
        
        self.logger.info("-" * 60)
        
    def _log_evaluation_results(self, episode: int, eval_results: Dict):
        """Log evaluation results with TensorBoard integration"""
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
        
        self.logger.info(f"📊 EVALUATION at Episode {episode} | {eval_performance}")
        self.logger.info(f"   🎯 Mean Reward: {mean_reward:7.2f} ± {std_reward:.2f}")
        self.logger.info(f"   ⏱️  Mean Steps: {mean_steps:.1f}")
        self.logger.info(f"   📈 Individual Episodes: {eval_results['episode_rewards']}")
        
        # Log to TensorBoard
        self.writer.add_scalar('Evaluation/Mean_Reward', mean_reward, episode)
        self.writer.add_scalar('Evaluation/Std_Reward', std_reward, episode)
        self.writer.add_scalar('Evaluation/Mean_Steps', mean_steps, episode)
        
        # Log additional metrics if available
        if 'mean_waiting_time' in eval_results:
            self.writer.add_scalar('Evaluation/Mean_Waiting_Time', eval_results['mean_waiting_time'], episode)
        if 'mean_queue_length' in eval_results:
            self.writer.add_scalar('Evaluation/Mean_Queue_Length', eval_results['mean_queue_length'], episode)
        if 'mean_speed' in eval_results:
            self.writer.add_scalar('Evaluation/Mean_Speed', eval_results['mean_speed'], episode)
        
        # Print adaptive hyperparameter info
        if hasattr(self.agent, 'get_adaptive_stats'):
            try:
                adaptive_stats = self.agent.get_adaptive_stats()
                print(f"   🎯 Performance Level: {adaptive_stats.get('recent_performance_level', 'unknown').upper()}")
                print(f"   📚 Current Learning Rate: {adaptive_stats.get('current_lr', 0):.2e}")
                print(f"   🎲 Current Epsilon: {self.agent.epsilon:.3f}")
            except:
                pass
        
        print("="*60)
        
    def print_training_summary(self, training_time: float, total_episodes: int):
        """Print training summary with adaptive features"""
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
        
        # Print basic training summary
        print(f"\n🎯 Training Summary:")
        print(f"   📚 Final Learning Rate: {self.agent.optimizer.param_groups[0]['lr']:.2e}")
        
        print("="*60)
        
    def train(self, episodes: int, eval_freq: int = 50, save_freq: int = 100):
        """
        Train the DQN agent with enhanced user feedback and adaptive features
        """
        self._log_training_header(episodes, eval_freq, save_freq)
        
        self.start_time = time.time()
        recent_rewards = []
        
        self.logger.info("🚀 Starting training...")
        self.logger.info("💡 The agent will start with random exploration (epsilon = 1.0)")
        self.logger.info("💡 As training progresses, it will learn and reduce exploration")
        self.logger.info("💡 Watch for improving rewards and more consistent performance")
        self.logger.info("🎯 Adaptive hyperparameters will automatically adjust based on performance")
        self.logger.info("-" * 60)
        
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
            
            # Add basic training data
            training_data.update({
                'current_lr': self.agent.optimizer.param_groups[0]['lr']
            })
            
            self.training_history.append(training_data)
            
            # Get recent training loss
            recent_loss = None
            if self.agent.training_losses:
                recent_loss = self.agent.training_losses[-1] if self.agent.training_losses else None
            
            # Log progress every episode for better visibility
            self._log_episode_progress(
                episode + 1, total_reward, steps, self.agent.epsilon, 
                recent_rewards[-10:], recent_loss
            )
            
            # Evaluate periodically
            if (episode + 1) % eval_freq == 0:
                self.logger.info(f"🔍 Running evaluation...")
                eval_results = self.evaluate(episodes=5)
                self.evaluation_history.append({
                    'episode': episode + 1,
                    'eval_results': eval_results,
                    'timestamp': datetime.now().isoformat()
                })
                
                self._log_evaluation_results(episode + 1, eval_results)
            
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
        
        # Close TensorBoard writer
        self.writer.close()
        self.logger.info("📊 TensorBoard writer closed")
        
        return self.training_history
    
    def evaluate(self, episodes: int = 5) -> Dict:
        """Evaluate the current agent"""
        self.logger.info(f"🔍 Evaluating agent over {episodes} episodes...")
        eval_results = self.agent.evaluate(self.env, episodes=episodes)
        return eval_results
    
    def save_training_results(self):
        """Save training results and plots with adaptive features"""
        self.logger.info("💾 Saving training results...")
        
        # Save training history as JSON
        history_path = os.path.join(self.results_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        self.logger.info(f"📄 Training history saved to: {history_path}")
        
        # Save training history as CSV for easy analysis
        csv_path = os.path.join(self.results_dir, 'training_history.csv')
        with open(csv_path, 'w', newline='') as f:
            if self.training_history:
                writer = csv.DictWriter(f, fieldnames=self.training_history[0].keys())
                writer.writeheader()
                writer.writerows(self.training_history)
        self.logger.info(f"📊 Training history CSV saved to: {csv_path}")
        
        # Save evaluation history as JSON
        eval_path = os.path.join(self.results_dir, 'evaluation_history.json')
        with open(eval_path, 'w') as f:
            json.dump(self.evaluation_history, f, indent=2)
        self.logger.info(f"📄 Evaluation history saved to: {eval_path}")
        
        # Save evaluation history as CSV
        eval_csv_path = os.path.join(self.results_dir, 'evaluation_history.csv')
        with open(eval_csv_path, 'w', newline='') as f:
            if self.evaluation_history:
                # Flatten evaluation data for CSV
                csv_data = []
                for eval_entry in self.evaluation_history:
                    row = {
                        'episode': eval_entry['episode'],
                        'timestamp': eval_entry['timestamp'],
                        'mean_reward': eval_entry['eval_results']['mean_reward'],
                        'std_reward': eval_entry['eval_results']['std_reward'],
                        'mean_steps': eval_entry['eval_results']['mean_steps']
                    }
                    csv_data.append(row)
                
                if csv_data:
                    writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                    writer.writeheader()
                    writer.writerows(csv_data)
        
        # Save training losses as CSV
        if self.agent.training_losses:
            losses_path = os.path.join(self.results_dir, 'training_losses.csv')
            with open(losses_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'loss'])
                for i, loss in enumerate(self.agent.training_losses):
                    writer.writerow([i, loss])
        
        # Save adaptive hyperparameter history if available
        if hasattr(self.agent, 'adaptation_history') and self.agent.adaptation_history:
            adaptive_path = os.path.join(self.results_dir, 'adaptive_history.json')
            with open(adaptive_path, 'w') as f:
                json.dump(self.agent.adaptation_history, f, indent=2)
            
            # Also save as CSV
            adaptive_csv_path = os.path.join(self.results_dir, 'adaptive_history.csv')
            with open(adaptive_csv_path, 'w', newline='') as f:
                if self.agent.adaptation_history:
                    # Extract key fields for CSV
                    csv_data = []
                    for adaptation in self.agent.adaptation_history:
                        row = {
                            'episode': adaptation['episode'],
                            'timestamp': adaptation['timestamp'],
                            'performance_level': adaptation['performance_level'],
                            'improvement_rate': adaptation['improvement_rate'],
                            'adaptations': '; '.join(adaptation['adaptations'])
                        }
                        csv_data.append(row)
                    
                    if csv_data:
                        writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                        writer.writeheader()
                        writer.writerows(csv_data)
        
        # Create plots
        self.create_training_plots()
        
        print(f"📊 Training results saved to: {self.results_dir}")
        print(f"   📄 JSON files: training_history.json, evaluation_history.json")
        print(f"   📊 CSV files: training_history.csv, evaluation_history.csv, training_losses.csv")
        print(f"   🎯 Training files saved successfully")
    
    def create_training_plots(self):
        """Create training visualization plots with adaptive features"""
        if not self.training_history:
            return
        
        # Extract data
        episodes = [h['episode'] for h in self.training_history]
        rewards = [h['total_reward'] for h in self.training_history]
        epsilons = [h['epsilon'] for h in self.training_history]
        steps = [h['steps'] for h in self.training_history]
        
        # Extract learning rate data
        learning_rates = [h.get('current_lr', 0) for h in self.training_history]
        
        # Create subplots - 2x2 for basic training plots
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
        
        # Training loss
        if self.agent.training_losses:
            loss_steps = list(range(len(self.agent.training_losses)))
            ax4.plot(loss_steps, self.agent.training_losses, 'r-', alpha=0.6)
            ax4.set_title('Training Loss (Huber)')
            ax4.set_xlabel('Training Step')
            ax4.set_ylabel('Loss')
            ax4.grid(True, alpha=0.3)
            
            # Moving average of loss
            if len(self.agent.training_losses) > 100:
                loss_window = 100
                loss_moving_avg = np.convolve(self.agent.training_losses, 
                                            np.ones(loss_window)/loss_window, mode='valid')
                ax4.plot(loss_steps[loss_window-1:], loss_moving_avg, 'b-', 
                        linewidth=2, label=f'Moving Avg ({loss_window})')
                ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No training loss data', ha='center', va='center', 
                    transform=ax4.transAxes)
            ax4.set_title('Training Loss')
        
        # Add title and layout
        plt.suptitle('DQN Training Results', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.results_dir, 'training_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
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
                action = self.agent.act(state)
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
                action_dist.append(f"Action {i}: {percentage:.1f}%")
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
        
        # Print basic training summary
        print(f"\n🎯 Training Summary:")
        print(f"   📚 Learning Rate: {self.agent.optimizer.param_groups[0]['lr']:.2e}")
        print(f"   🎲 Epsilon: {self.agent.epsilon:.3f}")
        
        print("="*60)
        
        return test_results

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Adaptive DQN agent for traffic signal control')
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
                       help='Disable mixed precision training (default: enabled on GPU)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--memory-size', type=int, default=10000, help='Experience replay buffer size')
    parser.add_argument('--performance-window', type=int, default=50,
                       help='Window size for performance evaluation')
    
    args = parser.parse_args()
    
    # Training configuration
    config = {
        'config_file': f'Sumo_env/gpt_newint/{args.config}',
        'max_steps': 1000,
        'state_size': 12,
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
        'mixed_precision': not args.no_mixed_precision,
        'performance_window': args.performance_window,
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
