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
    
    Input Layer: 24 neurons (state vector size)
    Hidden Layer 1: 256 neurons with ReLU
    Hidden Layer 2: 256 neurons with ReLU  
    Output Layer: 4 neurons (action space size)
    """
    
    def __init__(self, input_size: int = 24, hidden_size: int = 256, output_size: int = 4):
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
    - State: 24D vector (queue length + waiting time per lane)
    - Actions: 4 discrete actions (NS Green, EW Green, Extend, Skip)
    - Reward: Change in cumulative waiting time
    - GPU acceleration with CUDA optimization
    """
    
    def __init__(self, 
                 state_size: int = 24,
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
            state_size: Size of state vector (24 for 12 lanes * 2 metrics)
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
        
        # GPU memory optimization
        if self.device.type == 'cuda':
            # Clear GPU cache
            torch.cuda.empty_cache()
            print(f"🧹 GPU memory cleared. Available: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    
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
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
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
                
                # Compute loss
                loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
            
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
            
            # Compute loss
            loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
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
        
        # GPU memory cleanup
        if self.device.type == 'cuda' and self.episode_count % 10 == 0:
            torch.cuda.empty_cache()
        
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
        """Get current training statistics"""
        gpu_memory = 0
        if self.device.type == 'cuda':
            gpu_memory = torch.cuda.memory_allocated() / 1e6  # MB
        
        return {
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'mean_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'recent_losses': self.training_losses[-100:] if self.training_losses else [],
            'gpu_memory_mb': gpu_memory,
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
        """Clear GPU memory cache"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
            print(f"🧹 GPU cache cleared. Memory: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
