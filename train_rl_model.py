#!/usr/bin/env python3
"""
RL Model Training Script
========================

Trains reinforcement learning models for traffic signal control using stable-baselines3.
Supports multiple algorithms and provides comprehensive training monitoring.
"""

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import argparse
from datetime import datetime

# RL libraries
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Local imports
from rl_environment import TrafficSignalEnv

class RLTrainer:
    """Reinforcement Learning Trainer for Traffic Signal Control"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models_dir = config.get('models_dir', 'trained_models')
        self.logs_dir = config.get('logs_dir', 'training_logs')
        self.results_dir = config.get('results_dir', 'training_results')
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Training parameters
        self.algorithm = config.get('algorithm', 'PPO')
        self.total_timesteps = config.get('total_timesteps', 100000)
        self.eval_freq = config.get('eval_freq', 10000)
        self.save_freq = config.get('save_freq', 50000)
        self.n_eval_episodes = config.get('n_eval_episodes', 5)
        
        # Environment parameters
        self.config_file = config.get('config_file', 'Sumo_env/Single intersection lhd/uniform_simulation.sumocfg')
        self.max_steps = config.get('max_steps', 1000)
        
        # Training history
        self.training_history = []
        self.eval_history = []
        
    def create_env(self, is_eval: bool = False) -> TrafficSignalEnv:
        """Create environment instance"""
        env = TrafficSignalEnv(
            config_file=self.config_file,
            max_steps=self.max_steps
        )
        
        # Wrap with Monitor for logging
        env = Monitor(env)
        
        return env
    
    def create_vec_env(self, n_envs: int = 1, is_eval: bool = False):
        """Create vectorized environment"""
        env_fns = [lambda: self.create_env(is_eval) for _ in range(n_envs)]
        vec_env = DummyVecEnv(env_fns)
        
        # Normalize observations and rewards
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0
        )
        
        return vec_env
    
    def create_model(self, env):
        """Create RL model based on algorithm"""
        if self.algorithm == 'PPO':
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                clip_range_vf=None,
                normalize_advantage=True,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                use_sde=True,
                sde_sample_freq=4,
                target_kl=None,
                tensorboard_log=self.logs_dir,
                verbose=1
            )
        elif self.algorithm == 'A2C':
            model = A2C(
                "MlpPolicy",
                env,
                learning_rate=7e-4,
                n_steps=5,
                gamma=0.99,
                gae_lambda=1.0,
                ent_coef=0.01,
                vf_coef=0.25,
                max_grad_norm=0.5,
                rms_prop_eps=1e-5,
                use_sde=True,
                sde_sample_freq=4,
                tensorboard_log=self.logs_dir,
                verbose=1
            )
        elif self.algorithm == 'DQN':
            model = DQN(
                "MlpPolicy",
                env,
                learning_rate=1e-4,
                buffer_size=1000000,
                learning_starts=1000,
                batch_size=32,
                tau=1.0,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=1000,
                exploration_fraction=0.1,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                max_grad_norm=10,
                tensorboard_log=self.logs_dir,
                verbose=1
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        return model
    
    def create_callbacks(self, eval_env):
        """Create training callbacks"""
        callbacks = []
        
        # Evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{self.models_dir}/best_{self.algorithm}",
            log_path=f"{self.logs_dir}/eval_{self.algorithm}",
            eval_freq=self.eval_freq,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.save_freq,
            save_path=f"{self.models_dir}/checkpoints_{self.algorithm}",
            name_prefix=f"{self.algorithm}_model"
        )
        callbacks.append(checkpoint_callback)
        
        return callbacks
    
    def train(self):
        """Train the RL model"""
        print(f"🚀 Starting {self.algorithm} training...")
        print(f"📁 Models will be saved to: {self.models_dir}")
        print(f"📊 Logs will be saved to: {self.logs_dir}")
        
        # Create environments
        train_env = self.create_vec_env(n_envs=4)
        eval_env = self.create_vec_env(n_envs=1, is_eval=True)
        
        # Create model
        model = self.create_model(train_env)
        
        # Create callbacks
        callbacks = self.create_callbacks(eval_env)
        
        # Train model
        start_time = time.time()
        
        try:
            model.learn(
                total_timesteps=self.total_timesteps,
                callback=callbacks,
                progress_bar=True
            )
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise
        
        training_time = time.time() - start_time
        
        # Save final model
        final_model_path = f"{self.models_dir}/{self.algorithm}_final"
        model.save(final_model_path)
        print(f"✅ Final model saved to: {final_model_path}")
        
        # Save training configuration
        self.save_training_config(training_time)
        
        # Clean up
        train_env.close()
        eval_env.close()
        
        print(f"🎉 Training completed in {training_time:.2f} seconds")
        
        return model
    
    def save_training_config(self, training_time: float):
        """Save training configuration and results"""
        config_data = {
            'algorithm': self.algorithm,
            'total_timesteps': self.total_timesteps,
            'training_time_seconds': training_time,
            'config_file': self.config_file,
            'max_steps': self.max_steps,
            'eval_freq': self.eval_freq,
            'save_freq': self.save_freq,
            'n_eval_episodes': self.n_eval_episodes,
            'timestamp': datetime.now().isoformat(),
            'config': self.config
        }
        
        config_path = f"{self.results_dir}/{self.algorithm}_training_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"📋 Training config saved to: {config_path}")
    
    def evaluate_model(self, model_path: str, n_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate a trained model"""
        print(f"🔍 Evaluating model: {model_path}")
        
        # Load model
        if self.algorithm == 'PPO':
            model = PPO.load(model_path)
        elif self.algorithm == 'A2C':
            model = A2C.load(model_path)
        elif self.algorithm == 'DQN':
            model = DQN.load(model_path)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        # Create evaluation environment
        eval_env = self.create_vec_env(n_envs=1, is_eval=True)
        
        # Run evaluation
        episode_rewards = []
        episode_lengths = []
        episode_metrics = []
        
        for episode in range(n_episodes):
            obs = eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            episode_metric = []
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward[0]
                episode_length += 1
                
                if 'metrics' in info[0]:
                    episode_metric.append(info[0]['metrics'])
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_metrics.append(episode_metric)
            
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
        
        # Calculate evaluation statistics
        eval_results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_metrics': episode_metrics
        }
        
        # Save evaluation results
        eval_path = f"{self.results_dir}/{self.algorithm}_evaluation.json"
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"📊 Evaluation results saved to: {eval_path}")
        print(f"📈 Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"📏 Mean Length: {eval_results['mean_length']:.2f} ± {eval_results['std_length']:.2f}")
        
        eval_env.close()
        
        return eval_results

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train RL model for traffic signal control')
    parser.add_argument('--algorithm', type=str, default='PPO', 
                       choices=['PPO', 'A2C', 'DQN'], help='RL algorithm to use')
    parser.add_argument('--timesteps', type=int, default=100000, 
                       help='Total training timesteps')
    parser.add_argument('--config', type=str, 
                       default='Sumo_env/Single intersection lhd/uniform_simulation.sumocfg',
                       help='SUMO config file')
    parser.add_argument('--max-steps', type=int, default=1000, 
                       help='Maximum steps per episode')
    parser.add_argument('--eval-episodes', type=int, default=5, 
                       help='Number of evaluation episodes')
    parser.add_argument('--eval-only', type=str, 
                       help='Path to model for evaluation only')
    
    args = parser.parse_args()
    
    # Training configuration
    config = {
        'algorithm': args.algorithm,
        'total_timesteps': args.timesteps,
        'config_file': args.config,
        'max_steps': args.max_steps,
        'n_eval_episodes': args.eval_episodes,
        'eval_freq': max(1000, args.timesteps // 20),
        'save_freq': max(5000, args.timesteps // 10),
        'models_dir': 'trained_models',
        'logs_dir': 'training_logs',
        'results_dir': 'training_results'
    }
    
    # Create trainer
    trainer = RLTrainer(config)
    
    if args.eval_only:
        # Evaluation only
        trainer.evaluate_model(args.eval_only, n_episodes=10)
    else:
        # Training
        model = trainer.train()
        
        # Evaluate best model
        best_model_path = f"{config['models_dir']}/best_{args.algorithm}/best_model"
        if os.path.exists(best_model_path + ".zip"):
            trainer.evaluate_model(best_model_path, n_episodes=10)

if __name__ == "__main__":
    main()
