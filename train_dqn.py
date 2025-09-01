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
            device=config.get('device', 'auto')
        )
        
        # Training tracking
        self.training_history = []
        self.evaluation_history = []
        
    def train(self, episodes: int, eval_freq: int = 50, save_freq: int = 100):
        """
        Train the DQN agent
        
        Args:
            episodes: Number of episodes to train
            eval_freq: Frequency of evaluation
            save_freq: Frequency of model saving
        """
        print(f"🚀 Starting DQN Training")
        print(f"📊 Episodes: {episodes}")
        print(f"🔍 Evaluation frequency: {eval_freq}")
        print(f"💾 Save frequency: {save_freq}")
        print("=" * 50)
        
        start_time = time.time()
        
        for episode in range(episodes):
            # Train one episode
            total_reward, steps, step_rewards = self.agent.train_episode(
                self.env, max_steps=self.config.get('max_steps', 1000)
            )
            
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
            
            # Print progress
            if (episode + 1) % 10 == 0:
                recent_rewards = [h['total_reward'] for h in self.training_history[-10:]]
                avg_reward = np.mean(recent_rewards)
                print(f"Episode {episode + 1:4d} | "
                      f"Reward: {total_reward:6.2f} | "
                      f"Steps: {steps:3d} | "
                      f"Epsilon: {self.agent.epsilon:.3f} | "
                      f"Avg (10): {avg_reward:.2f}")
            
            # Evaluate periodically
            if (episode + 1) % eval_freq == 0:
                eval_results = self.evaluate(episodes=5)
                self.evaluation_history.append({
                    'episode': episode + 1,
                    'eval_results': eval_results,
                    'timestamp': datetime.now().isoformat()
                })
                
                print(f"📈 Evaluation at episode {episode + 1}: "
                      f"Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
            
            # Save model periodically
            if (episode + 1) % save_freq == 0:
                model_path = os.path.join(self.models_dir, f"dqn_episode_{episode + 1}.pth")
                self.agent.save(model_path)
                print(f"💾 Model saved: {model_path}")
        
        # Final save
        final_model_path = os.path.join(self.models_dir, "dqn_final.pth")
        self.agent.save(final_model_path)
        
        training_time = time.time() - start_time
        print(f"\n🎉 Training completed in {training_time:.2f} seconds")
        print(f"📁 Final model saved: {final_model_path}")
        
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
        """Test the trained agent"""
        print(f"🧪 Testing agent over {episodes} episodes...")
        
        test_results = []
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
            
            test_results.append({
                'episode': episode + 1,
                'total_reward': total_reward,
                'steps': steps,
                'actions_taken': actions_taken,
                'action_counts': {action: actions_taken.count(action) for action in range(4)}
            })
            
            print(f"Episode {episode + 1}: Reward {total_reward:.2f}, Steps {steps}")
        
        # Print summary
        avg_reward = np.mean([r['total_reward'] for r in test_results])
        avg_steps = np.mean([r['steps'] for r in test_results])
        
        print(f"\n📊 Test Summary:")
        print(f"   Average Reward: {avg_reward:.2f}")
        print(f"   Average Steps: {avg_steps:.1f}")
        
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
        'memory_size': 10000,
        'batch_size': 32,
        'target_update_freq': 1000,
        'device': 'auto',
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
