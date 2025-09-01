#!/usr/bin/env python3
"""
Enhanced Deep Q-Network (DQN) Agent
===================================

A DQN agent enhanced with Experience Replay and Target Network for stable learning.
Specifically designed for traffic signal control at single intersections.
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

class DQNNetwork(nn.Module):
    """
    Neural Network Architecture for DQN
    
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
    """Experience Replay Buffer for DQN"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                              np.ndarray, np.ndarray]:
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self) -> int:
        return len(self.buffer)

class DQNAgent:
    """
    Enhanced Deep Q-Network Agent for Traffic Signal Control
    
    Features:
    - Experience Replay for stable learning
    - Target Network for stable Q-value estimation
    - Epsilon-greedy exploration strategy
    - State: 24D vector (queue length + waiting time per lane)
    - Actions: 4 discrete actions (NS Green, EW Green, Extend, Skip)
    - Reward: Change in cumulative waiting time
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
                 device: str = 'auto'):
        """
        Initialize DQN Agent
        
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
        """
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
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
        
        # Networks
        self.q_network = DQNNetwork(state_size, hidden_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, hidden_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = ExperienceReplay(memory_size)
        
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
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy
        
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
            # Greedy action (exploitation)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
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
        Train the network using experience replay
        
        Returns:
            float: Training loss (None if not enough samples)
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
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
        Train for one episode
        
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
        
        return total_reward, len(step_rewards), step_rewards
    
    def evaluate(self, env, episodes: int = 5, max_steps: int = 1000) -> dict:
        """
        Evaluate the agent without exploration
        
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
        
        torch.save(save_data, filepath)
        print(f"Agent saved to: {filepath}")
    
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
        
        print(f"Agent loaded from: {filepath}")
        print(f"Loaded {self.episode_count} episodes, epsilon: {self.epsilon:.3f}")
    
    def get_action_name(self, action: int) -> str:
        """Get human-readable action name"""
        return self.action_names.get(action, f"Unknown Action {action}")
    
    def get_training_stats(self) -> dict:
        """Get current training statistics"""
        return {
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'mean_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'recent_losses': self.training_losses[-100:] if self.training_losses else []
        }
