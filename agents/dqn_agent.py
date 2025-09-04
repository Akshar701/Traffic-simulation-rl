#!/usr/bin/env python3
"""
Enhanced Deep Q-Network (DQN) Agent - GPU Optimized
==================================================

A DQN agent enhanced with Experience Replay and Target Network for stable learning.
Specifically designed for traffic signal control at single intersections.
Optimized for NVIDIA GPU training with CUDA acceleration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import List, Tuple, Optional
import os
import json
from datetime import datetime
import gc

class DQNNetwork(nn.Module):
    """
    Neural Network Architecture for DQN - GPU Optimized
    
    Input Layer: 8 neurons (state vector size)
    Hidden Layer 1: 256 neurons with ReLU
    Hidden Layer 2: 256 neurons with ReLU  
    Output Layer: 4 neurons (action space size)
    """
    
    def __init__(self, input_size: int = 8, hidden_size: int = 256, output_size: int = 4):
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Initialize weights for better training
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Linear activation for Q-values
        return x

class ExperienceReplay:
    """Experience Replay Buffer for DQN - GPU Optimized"""
    
    def __init__(self, capacity: int = 10000, device: torch.device = None):
        self.buffer = deque(maxlen=capacity)
        self.device = device
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                              torch.Tensor, torch.Tensor]:
        """Sample a batch of experiences and convert to GPU tensors"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors and move to GPU
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(np.array(dones)).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)

class DQNAgent:
    """
    Enhanced Deep Q-Network Agent for Traffic Signal Control - GPU Optimized
    
    Features:
    - Experience Replay for stable learning
    - Target Network for stable Q-value estimation
    - Epsilon-greedy exploration strategy
    - State: 8D vector (4 queue lengths + 4 one-hot phase encoding)
    - Actions: 4 discrete actions (NS Green, EW Green, Extend, Skip)
    - Reward: Change in cumulative waiting time
    - GPU acceleration with CUDA optimization
    """
    
    def __init__(self, 
                 state_size: int = 8,
                 action_size: int = 4,
                 hidden_size: int = 256,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 1000,
                 device: str = 'auto',
                 mixed_precision: bool = True):
        """
        Initialize DQN Agent with GPU optimization
        
        Args:
            state_size: Size of state vector (8 for 4 approaches + 4 phases)
            action_size: Number of possible actions (4)
            hidden_size: Number of neurons in hidden layers (256)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            memory_size: Size of experience replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            device: Device to use for training ('cpu', 'cuda', or 'auto')
            mixed_precision: Whether to use mixed precision training (FP16)
        """
        
        # Set device with GPU optimization
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                # Set CUDA optimization flags
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name()}")
                print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                self.device = torch.device('cpu')
                print("⚠️ CUDA not available, using CPU")
        else:
            self.device = torch.device(device)
        
        # Mixed precision training
        self.mixed_precision = mixed_precision and self.device.type == 'cuda'
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            print("🎯 Mixed precision training enabled (FP16)")
        
        # Hyperparameters
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Networks - Move to GPU
        self.q_network = DQNNetwork(state_size, hidden_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, hidden_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer with GPU optimization
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay with GPU support
        self.memory = ExperienceReplay(memory_size, self.device)
        
        # Training tracking
        self.step_count = 0
        self.episode_count = 0
        self.training_losses = []
        self.episode_rewards = []
        self.recent_actions = []  # Track recent actions for analysis
        
        # Action mapping
        self.action_names = {
            0: "NS Green",
            1: "EW Green", 
            2: "Extend Phase",
            3: "Skip Phase"
        }
        
        # Memory management
        self._memory_cleanup_threshold = 0.9  # Cleanup when 90% full
        self._last_memory_cleanup = 0
        
        # Initialize GPU memory monitoring
        self._init_gpu_memory_monitoring()
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy - GPU optimized
        
        Args:
            state: Current state vector
            training: Whether in training mode (affects exploration)
            
        Returns:
            int: Chosen action
        """
        if training and random.random() < self.epsilon:
            # Random action (exploration)
            action = random.randrange(self.action_size)
        else:
            # Greedy action (exploitation) - GPU optimized
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        q_values = self.q_network(state_tensor)
                else:
                    q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
        
        # Track action for analysis
        self.recent_actions.append(action)
        if len(self.recent_actions) > 1000:  # Keep last 1000 actions
            self.recent_actions.pop(0)
        
        return action
    
    def _init_gpu_memory_monitoring(self):
        """Initialize GPU memory monitoring without clearing cache"""
        if self.device.type == 'cuda':
            # Get initial GPU memory info
            self._initial_gpu_memory = torch.cuda.memory_allocated()
            print(f"🧹 Initial GPU memory allocated: {self._initial_gpu_memory / 1e6:.1f} MB")
            
            # Set memory management strategy
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                # Reserve some GPU memory for other processes
                torch.cuda.set_per_process_memory_fraction(0.9)
                print("📊 GPU memory fraction set to 90%")
    
    def _monitor_gpu_memory(self, operation: str = "general"):
        """Monitor GPU memory usage without clearing cache"""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            
            # Calculate memory efficiency
            efficiency = allocated / reserved if reserved > 0 else 1.0
            
            # Log memory usage for debugging (only when significant)
            if allocated > self._initial_gpu_memory * 2:  # If memory doubled
                print(f"📊 GPU Memory ({operation}): "
                      f"Allocated: {allocated / 1e6:.1f}MB, "
                      f"Reserved: {reserved / 1e6:.1f}MB, "
                      f"Efficiency: {efficiency:.2%}")
            
            # Return memory info for external monitoring
            return {
                'allocated_mb': allocated / 1e6,
                'reserved_mb': reserved / 1e6,
                'total_mb': total / 1e6,
                'efficiency': efficiency
            }
        return {}
    
    def _smart_memory_cleanup(self):
        """Smart memory cleanup only when necessary"""
        if self.device.type == 'cuda':
            # Only cleanup if memory usage is high and we haven't cleaned recently
            current_memory = torch.cuda.memory_allocated()
            memory_ratio = current_memory / torch.cuda.get_device_properties(0).total_memory
            
            if (memory_ratio > self._memory_cleanup_threshold and 
                self.step_count - self._last_memory_cleanup > 1000):
                
                # Use garbage collection instead of cache clearing
                gc.collect()
                
                # Only clear cache if memory is critically high
                if memory_ratio > 0.95:
                    torch.cuda.empty_cache()
                    print(f"🧹 Critical memory cleanup performed at step {self.step_count}")
                else:
                    print(f"🧹 Memory cleanup performed at step {self.step_count}")
                
                self._last_memory_cleanup = self.step_count
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
        
        # Smart memory cleanup
        self._smart_memory_cleanup()
    
    def replay(self) -> Optional[float]:
        """
        Train the network using experience replay - GPU optimized
        
        Returns:
            float: Training loss (None if not enough samples)
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from memory (already on GPU)
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Training with mixed precision
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                # Current Q-values
                current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
                
                # Next Q-values (from target network)
                with torch.no_grad():
                    next_q_values = self.target_network(next_states).max(1)[0]
                    target_q_values = rewards + (self.gamma * next_q_values * ~dones)
                
                # Compute Huber loss for training stability
                loss = F.huber_loss(current_q_values.squeeze(), target_q_values, delta=1.0)
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard training
            # Current Q-values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Next Q-values (from target network)
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
            # Compute Huber loss for training stability
            loss = F.huber_loss(current_q_values.squeeze(), target_q_values, delta=1.0)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Store training loss for monitoring
        loss_value = loss.item()
        self.training_losses.append(loss_value)
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Monitor GPU memory after training
        self._monitor_gpu_memory("training")
        
        return loss.item()
    
    def train_episode(self, env, max_steps: int = 1000) -> Tuple[float, int, List[float]]:
        """
        Train for one episode - GPU optimized
        
        Args:
            env: Environment instance
            max_steps: Maximum steps per episode
            
        Returns:
            Tuple of (total_reward, steps_taken, step_rewards)
        """
        state = env.reset()
        total_reward = 0
        step_rewards = []
        
        for step in range(max_steps):
            # Choose action
            action = self.act(state, training=True)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            self.remember(state, action, reward, next_state, done)
            
            # Train network
            loss = self.replay()
            
            # Update state and tracking
            state = next_state
            total_reward += reward
            step_rewards.append(reward)
            
            if done:
                break
        
        self.episode_count += 1
        self.episode_rewards.append(total_reward)
        
        # Smart memory cleanup after episode (only when necessary)
        if self.episode_count % 5 == 0:  # Every 5 episodes
            self._smart_memory_cleanup()
        
        return total_reward, len(step_rewards), step_rewards
    
    def evaluate(self, env, episodes: int = 5, max_steps: int = 1000) -> dict:
        """
        Evaluate the agent without exploration - GPU optimized
        
        Args:
            env: Environment instance
            episodes: Number of episodes to evaluate
            max_steps: Maximum steps per episode
            
        Returns:
            dict: Evaluation results
        """
        eval_rewards = []
        eval_steps = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            for step in range(max_steps):
                # Choose action (no exploration)
                action = self.act(state, training=False)
                
                # Take action
                state, reward, done, info = env.step(action)
                
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            eval_rewards.append(total_reward)
            eval_steps.append(steps)
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_steps': np.mean(eval_steps),
            'episode_rewards': eval_rewards,
            'episode_steps': eval_steps
        }
    
    def save(self, filepath: str):
        """Save the agent model and training state"""
        save_data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'training_losses': self.training_losses,
            'episode_rewards': self.episode_rewards,
            'hyperparameters': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'hidden_size': self.hidden_size,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq
            }
        }
        
        # Save to CPU before saving to file
        torch.save(save_data, filepath, _use_new_zipfile_serialization=False)
        print(f"💾 Agent saved to: {filepath}")
    
    def load(self, filepath: str):
        """Load the agent model and training state"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        save_data = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(save_data['q_network_state_dict'])
        self.target_network.load_state_dict(save_data['target_network_state_dict'])
        self.optimizer.load_state_dict(save_data['optimizer_state_dict'])
        
        self.epsilon = save_data['epsilon']
        self.step_count = save_data['step_count']
        self.episode_count = save_data['episode_count']
        self.training_losses = save_data['training_losses']
        self.episode_rewards = save_data['episode_rewards']
        
        print(f"✅ Agent loaded from: {filepath}")
        print(f"📊 Loaded {self.episode_count} episodes, epsilon: {self.epsilon:.3f}")
    
    def get_action_name(self, action: int) -> str:
        """Get human-readable action name"""
        return self.action_names.get(action, f"Unknown Action {action}")
    
    def get_training_stats(self) -> dict:
        """Get current training statistics with memory info"""
        # Get GPU memory info
        gpu_memory_info = self._monitor_gpu_memory("stats")
        
        return {
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'mean_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'recent_losses': self.training_losses[-100:] if self.training_losses else [],
            'gpu_memory_mb': gpu_memory_info.get('allocated_mb', 0),
            'gpu_memory_efficiency': gpu_memory_info.get('efficiency', 0),
            'device': str(self.device)
        }
    
    def optimize_for_inference(self):
        """Optimize the model for inference (faster predictions)"""
        if self.device.type == 'cuda':
            # Enable TensorRT-like optimizations
            self.q_network.eval()
            with torch.no_grad():
                # Warm up the model
                dummy_input = torch.randn(1, self.state_size).to(self.device)
                for _ in range(10):
                    _ = self.q_network(dummy_input)
            
            print("🚀 Model optimized for inference")
    
    def clear_gpu_cache(self):
        """Clear GPU memory cache - use sparingly"""
        if self.device.type == 'cuda':
            # Only clear cache when absolutely necessary
            current_memory = torch.cuda.memory_allocated()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            
            if current_memory / total_memory > 0.95:  # Only if 95%+ full
                torch.cuda.empty_cache()
                gc.collect()
                print(f"🧹 GPU cache cleared due to high memory usage")
                print(f"   Memory before: {current_memory / 1e6:.1f} MB")
                print(f"   Memory after: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
            else:
                print(f"⚠️ GPU cache clearing skipped - memory usage is acceptable")
                print(f"   Current usage: {current_memory / total_memory:.1%}")
    
    def get_memory_efficiency_report(self) -> dict:
        """Get detailed memory efficiency report"""
        gpu_info = self._monitor_gpu_memory("report")
        
        return {
            'gpu_memory': gpu_info,
            'training_memory': {
                'losses_stored': len(self.training_losses),
                'rewards_stored': len(self.episode_rewards),
                'actions_tracked': len(self.recent_actions)
            },
            'recommendations': self._get_memory_recommendations(gpu_info)
        }
    
    def _get_memory_recommendations(self, gpu_info: dict) -> List[str]:
        """Get memory optimization recommendations"""
        recommendations = []
        
        # GPU memory recommendations
        if gpu_info.get('efficiency', 1.0) < 0.7:
            recommendations.append("GPU memory efficiency is low - consider reducing batch size")
        
        if gpu_info.get('allocated_mb', 0) > 8000:  # 8GB
            recommendations.append("GPU memory usage is high - consider gradient accumulation")
        
        if not recommendations:
            recommendations.append("Memory usage is optimal")
        
        return recommendations

class AdaptiveDQNAgent(DQNAgent):
    """
    Enhanced DQN Agent with Adaptive Hyperparameters
    
    Features:
    - Adaptive learning rate scheduling based on performance
    - Performance-based epsilon decay
    - Dynamic batch size adjustment
    - Adaptive target network update frequency
    - Performance monitoring and automatic tuning
    """
    
    def __init__(self, 
                 state_size: int = 8,
                 action_size: int = 4,
                 hidden_size: int = 256,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 1000,
                 device: str = 'auto',
                 mixed_precision: bool = True,
                 adaptive_lr: bool = True,
                 adaptive_epsilon: bool = True,
                 performance_window: int = 50):
        """
        Initialize Adaptive DQN Agent
        
        Args:
            ... (inherited from DQNAgent)
            adaptive_lr: Whether to use adaptive learning rate
            adaptive_epsilon: Whether to use adaptive epsilon decay
            performance_window: Window size for performance evaluation
        """
        super().__init__(
            state_size, action_size, hidden_size, learning_rate, gamma,
            epsilon, epsilon_min, epsilon_decay, memory_size, batch_size,
            target_update_freq, device, mixed_precision
        )
        
        # Adaptive hyperparameter settings
        self.adaptive_lr = adaptive_lr
        self.adaptive_epsilon = adaptive_epsilon
        self.performance_window = performance_window
        
        # Performance tracking for adaptation
        self.performance_history = []
        self.reward_trends = []
        self.loss_trends = []
        
        # Learning rate scheduling
        if self.adaptive_lr:
            self._init_learning_rate_scheduler()
        
        # Adaptive parameters
        self.initial_lr = learning_rate
        self.initial_epsilon = epsilon
        self.initial_batch_size = batch_size
        self.initial_target_update_freq = target_update_freq
        
        # Performance thresholds
        self.performance_thresholds = {
            'excellent': -20,    # Reward > -20
            'good': -50,         # Reward > -50
            'fair': -100,        # Reward > -100
            'poor': -1000        # Reward > -1000
        }
        
        # Adaptation history
        self.adaptation_history = []
        
        print("🎯 Adaptive hyperparameters enabled")
        if self.adaptive_lr:
            print(f"   📚 Learning rate scheduling: {type(self.scheduler).__name__}")
        if self.adaptive_epsilon:
            print(f"   🎲 Adaptive epsilon decay enabled")
    
    def _init_learning_rate_scheduler(self):
        """Initialize learning rate scheduler"""
        # Use ReduceLROnPlateau for automatic learning rate reduction
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max',           # Monitor reward (higher is better)
            factor=0.5,           # Reduce LR by half
            patience=25,          # Wait 25 episodes before reducing
            min_lr=1e-6          # Minimum learning rate
        )
        
        # Alternative: Cosine annealing scheduler for more gradual decay
        # self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     self.optimizer, T_0=100, T_mult=2, eta_min=1e-6
        # )
    
    def _calculate_performance_metrics(self) -> dict:
        """Calculate performance metrics for adaptation"""
        if len(self.episode_rewards) < self.performance_window:
            return {}
        
        # Get recent performance window
        recent_rewards = self.episode_rewards[-self.performance_window:]
        recent_losses = self.training_losses[-self.performance_window:] if self.training_losses else []
        
        # Calculate metrics
        avg_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)
        reward_trend = self._calculate_trend(recent_rewards)
        
        # Loss metrics
        avg_loss = np.mean(recent_losses) if recent_losses else 0
        loss_trend = self._calculate_trend(recent_losses) if recent_losses else 0
        
        # Performance classification
        performance_level = self._classify_performance(avg_reward)
        
        # Calculate improvement rate
        if len(self.episode_rewards) >= self.performance_window * 2:
            previous_window = self.episode_rewards[-2*self.performance_window:-self.performance_window]
            improvement_rate = (avg_reward - np.mean(previous_window)) / abs(np.mean(previous_window)) if np.mean(previous_window) != 0 else 0
        else:
            improvement_rate = 0
        
        return {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'reward_trend': reward_trend,
            'avg_loss': avg_loss,
            'loss_trend': loss_trend,
            'performance_level': performance_level,
            'improvement_rate': improvement_rate,
            'episodes_analyzed': len(recent_rewards)
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend of values (positive = improving, negative = declining)"""
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression slope
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize by the range of values
        if len(values) > 0:
            value_range = max(values) - min(values)
            if value_range > 0:
                slope = slope / value_range
        
        return slope
    
    def _classify_performance(self, avg_reward: float) -> str:
        """Classify performance based on average reward"""
        if avg_reward > self.performance_thresholds['excellent']:
            return 'excellent'
        elif avg_reward > self.performance_thresholds['good']:
            return 'good'
        elif avg_reward > self.performance_thresholds['fair']:
            return 'fair'
        elif avg_reward > self.performance_thresholds['poor']:
            return 'poor'
        else:
            return 'critical'
    
    def _adapt_hyperparameters(self, performance_metrics: dict):
        """Adapt hyperparameters based on performance"""
        if not performance_metrics:
            return
        
        adaptations = []
        performance_level = performance_metrics.get('performance_level', 'unknown')
        improvement_rate = performance_metrics.get('improvement_rate', 0)
        
        # Learning rate adaptation
        if self.adaptive_lr and self.scheduler:
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Step the scheduler based on performance
            if performance_level in ['excellent', 'good']:
                # Good performance - let scheduler handle it
                pass
            elif performance_level in ['fair', 'poor']:
                # Poor performance - reduce learning rate
                if current_lr > 1e-6:
                    new_lr = current_lr * 0.8
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    adaptations.append(f"LR reduced: {current_lr:.2e} → {new_lr:.2e}")
            
            # Update scheduler with current performance
            self.scheduler.step(performance_metrics['avg_reward'])
        
        # Epsilon adaptation
        if self.adaptive_epsilon:
            current_epsilon = self.epsilon
            
            if performance_level == 'excellent':
                # Excellent performance - reduce exploration faster
                if current_epsilon > self.epsilon_min:
                    self.epsilon = max(self.epsilon_min, current_epsilon * 0.99)
                    adaptations.append(f"Epsilon reduced: {current_epsilon:.3f} → {self.epsilon:.3f}")
            
            elif performance_level in ['poor', 'critical']:
                # Poor performance - maintain exploration
                if current_epsilon < 0.5:
                    self.epsilon = min(0.8, current_epsilon * 1.1)
                    adaptations.append(f"Epsilon increased: {current_epsilon:.3f} → {self.epsilon:.3f}")
            
            elif performance_level == 'fair':
                # Fair performance - moderate exploration
                if current_epsilon > self.epsilon_min:
                    self.epsilon = max(self.epsilon_min, current_epsilon * 0.998)
                    adaptations.append(f"Epsilon moderately reduced: {current_epsilon:.3f} → {self.epsilon:.3f}")
        
        # Batch size adaptation
        if performance_level == 'excellent' and improvement_rate > 0.1:
            # Excellent performance with good improvement - increase batch size
            if self.batch_size < 128:
                old_batch_size = self.batch_size
                self.batch_size = min(128, self.batch_size * 2)
                adaptations.append(f"Batch size increased: {old_batch_size} → {self.batch_size}")
        
        elif performance_level in ['poor', 'critical']:
            # Poor performance - reduce batch size for more frequent updates
            if self.batch_size > 16:
                old_batch_size = self.batch_size
                self.batch_size = max(16, self.batch_size // 2)
                adaptations.append(f"Batch size decreased: {old_batch_size} → {self.batch_size}")
        
        # Target network update frequency adaptation
        if performance_level == 'excellent' and improvement_rate > 0.05:
            # Good performance - update target network more frequently
            if self.target_update_freq > 500:
                old_freq = self.target_update_freq
                self.target_update_freq = max(500, self.target_update_freq // 2)
                adaptations.append(f"Target update frequency increased: {old_freq} → {self.target_update_freq}")
        
        elif performance_level in ['poor', 'critical']:
            # Poor performance - update target network less frequently for stability
            if self.target_update_freq < 2000:
                old_freq = self.target_update_freq
                self.target_update_freq = min(2000, self.target_update_freq * 2)
                adaptations.append(f"Target update frequency decreased: {old_freq} → {self.target_update_freq}")
        
        # Record adaptations
        if adaptations:
            adaptation_record = {
                'episode': self.episode_count,
                'performance_level': performance_level,
                'improvement_rate': improvement_rate,
                'adaptations': adaptations,
                'timestamp': datetime.now().isoformat()
            }
            self.adaptation_history.append(adaptation_record)
            
            print(f"🎯 Hyperparameter adaptation at episode {self.episode_count}:")
            for adaptation in adaptations:
                print(f"   • {adaptation}")
    
    def _adaptive_epsilon_decay(self, performance_metrics: dict):
        """Adaptive epsilon decay based on performance"""
        if not self.adaptive_epsilon:
            return
        
        performance_level = performance_metrics.get('performance_level', 'unknown')
        improvement_rate = performance_metrics.get('improvement_rate', 0)
        
        # Base epsilon decay
        if self.epsilon > self.epsilon_min:
            # Performance-based decay rate
            if performance_level == 'excellent':
                decay_rate = 0.99  # Fast decay for excellent performance
            elif performance_level == 'good':
                decay_rate = 0.995  # Moderate decay for good performance
            elif performance_level == 'fair':
                decay_rate = 0.998  # Slow decay for fair performance
            else:
                decay_rate = 0.999  # Very slow decay for poor performance
            
            # Adjust based on improvement rate
            if improvement_rate > 0.1:
                decay_rate *= 0.99  # Faster decay if improving
            elif improvement_rate < -0.1:
                decay_rate *= 1.01  # Slower decay if declining
            
            self.epsilon = max(self.epsilon_min, self.epsilon * decay_rate)
    
    def train_episode(self, env, max_steps: int = 1000) -> Tuple[float, int, List[float]]:
        """
        Train for one episode with adaptive hyperparameters
        """
        # Train episode using parent method
        total_reward, steps, step_rewards = super().train_episode(env, max_steps)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics()
        
        # Adapt hyperparameters if we have enough data
        if performance_metrics:
            self._adapt_hyperparameters(performance_metrics)
            self._adaptive_epsilon_decay(performance_metrics)
        
        # Store performance metrics
        if performance_metrics:
            self.performance_history.append(performance_metrics)
        
        return total_reward, steps, step_rewards
    
    def get_adaptive_stats(self) -> dict:
        """Get adaptive hyperparameter statistics"""
        base_stats = self.get_training_stats()
        
        # Add adaptive-specific stats
        adaptive_stats = {
            'adaptive_lr': self.adaptive_lr,
            'adaptive_epsilon': self.adaptive_epsilon,
            'current_lr': self.optimizer.param_groups[0]['lr'],
            'initial_lr': self.initial_lr,
            'lr_change_factor': self.optimizer.param_groups[0]['lr'] / self.initial_lr,
            'epsilon_change_factor': self.epsilon / self.initial_epsilon,
            'batch_size_change_factor': self.batch_size / self.initial_batch_size,
            'target_update_freq_change_factor': self.target_update_freq / self.initial_target_update_freq,
            'adaptation_count': len(self.adaptation_history),
            'performance_history_length': len(self.performance_history)
        }
        
        # Add recent performance metrics
        if self.performance_history:
            recent_performance = self.performance_history[-1]
            adaptive_stats.update({
                'recent_performance_level': recent_performance.get('performance_level', 'unknown'),
                'recent_improvement_rate': recent_performance.get('improvement_rate', 0),
                'recent_avg_reward': recent_performance.get('avg_reward', 0)
            })
        
        # Add adaptation recommendations
        adaptive_stats['recommendations'] = self._get_adaptation_recommendations()
        
        return {**base_stats, **adaptive_stats}
    
    def _get_adaptation_recommendations(self) -> List[str]:
        """Get recommendations for hyperparameter adaptation"""
        recommendations = []
        
        if not self.performance_history:
            return ["Collect more performance data for recommendations"]
        
        recent_performance = self.performance_history[-1]
        performance_level = recent_performance.get('performance_level', 'unknown')
        improvement_rate = recent_performance.get('improvement_rate', 0)
        
        # Learning rate recommendations
        if self.adaptive_lr:
            current_lr = self.optimizer.param_groups[0]['lr']
            if performance_level in ['poor', 'critical'] and current_lr > 1e-5:
                recommendations.append("Consider reducing learning rate for stability")
            elif performance_level == 'excellent' and improvement_rate < 0.05:
                recommendations.append("Consider increasing learning rate for faster convergence")
        
        # Epsilon recommendations
        if self.adaptive_epsilon:
            if performance_level in ['poor', 'critical'] and self.epsilon < 0.3:
                recommendations.append("Increase exploration (epsilon) for better policy discovery")
            elif performance_level == 'excellent' and self.epsilon > 0.1:
                recommendations.append("Reduce exploration (epsilon) to exploit learned policy")
        
        # Batch size recommendations
        if performance_level == 'excellent' and improvement_rate > 0.1:
            recommendations.append("Consider increasing batch size for more stable gradients")
        elif performance_level in ['poor', 'critical']:
            recommendations.append("Consider decreasing batch size for more frequent updates")
        
        # Target network recommendations
        if performance_level in ['poor', 'critical']:
            recommendations.append("Consider increasing target network update frequency for stability")
        
        if not recommendations:
            recommendations.append("Current hyperparameters appear optimal")
        
        return recommendations
    
    def reset_hyperparameters(self):
        """Reset hyperparameters to initial values"""
        print("🔄 Resetting hyperparameters to initial values...")
        
        # Reset learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr
        
        # Reset epsilon
        self.epsilon = self.initial_epsilon
        
        # Reset batch size
        self.batch_size = self.initial_batch_size
        
        # Reset target update frequency
        self.target_update_freq = self.initial_target_update_freq
        
        # Reset scheduler if available
        if hasattr(self, 'scheduler') and self.scheduler:
            if hasattr(self.scheduler, 'reset'):
                self.scheduler.reset()
        
        print(f"✅ Hyperparameters reset:")
        print(f"   Learning Rate: {self.initial_lr}")
        print(f"   Epsilon: {self.initial_epsilon}")
        print(f"   Batch Size: {self.initial_batch_size}")
        print(f"   Target Update Freq: {self.initial_target_update_freq}")
    
    def save(self, filepath: str):
        """Save the agent with adaptive hyperparameter state"""
        save_data = super().save(filepath)
        
        # Add adaptive-specific data
        adaptive_data = {
            'adaptive_lr': self.adaptive_lr,
            'adaptive_epsilon': self.adaptive_epsilon,
            'performance_window': self.performance_window,
            'performance_history': self.performance_history,
            'adaptation_history': self.adaptation_history,
            'initial_lr': self.initial_lr,
            'initial_epsilon': self.initial_epsilon,
            'initial_batch_size': self.initial_batch_size,
            'initial_target_update_freq': self.initial_target_update_freq
        }
        
        # Merge with base save data
        if isinstance(save_data, dict):
            save_data.update(adaptive_data)
        
        return save_data
    
    def load(self, filepath: str):
        """Load the agent with adaptive hyperparameter state"""
        super().load(filepath)
        
        # Load adaptive-specific data if available
        try:
            save_data = torch.load(filepath, map_location=self.device)
            
            if 'adaptive_lr' in save_data:
                self.adaptive_lr = save_data['adaptive_lr']
                self.adaptive_epsilon = save_data['adaptive_epsilon']
                self.performance_window = save_data.get('performance_window', 50)
                self.performance_history = save_data.get('performance_history', [])
                self.adaptation_history = save_data.get('adaptation_history', [])
                self.initial_lr = save_data.get('initial_lr', self.learning_rate)
                self.initial_epsilon = save_data.get('initial_epsilon', 1.0)
                self.initial_batch_size = save_data.get('initial_batch_size', self.batch_size)
                self.initial_target_update_freq = save_data.get('initial_target_update_freq', self.target_update_freq)
                
                print("✅ Adaptive hyperparameters loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load adaptive hyperparameters: {e}")
