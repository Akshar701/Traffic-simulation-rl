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
from typing import List, Tuple, Optional, Dict, Any
import os

class DQNNetwork(nn.Module):
    """
    Neural Network Architecture for DQN
    
    Input Layer: 12 neurons (state vector size)
    Hidden Layer 1: 256 neurons with ReLU
    Hidden Layer 2: 256 neurons with ReLU  
    Output Layer: 4 neurons (action space size)
    """
    def __init__(self, input_size: int = 12, hidden_size: int = 256, output_size: int = 4):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ExperienceReplay:
    """Experience Replay Buffer for DQN"""
    def __init__(self, capacity: int = 10000, device: torch.device = None):
        self.buffer = deque(maxlen=capacity)
        self.device = device
    
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        
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
    Enhanced Deep Q-Network Agent for Traffic Signal Control
    """
    def __init__(self, 
                 state_size: int = 12,
                 action_size: int = 4,
                 hidden_size: int = 256,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 64,
                 target_update_freq: int = 1000,
                 device: str = 'auto',
                 mixed_precision: bool = True):
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        if self.device.type == 'cuda':
            print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ Using CPU")
        
        self.mixed_precision = mixed_precision and self.device.type == 'cuda'
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            print("🎯 Mixed precision training enabled (FP16)")
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.q_network = DQNNetwork(state_size, hidden_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, hidden_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = ExperienceReplay(memory_size, self.device)
        
        self.step_count = 0
    
    def act(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    q_values = self.q_network(state_tensor)
            else:
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
            
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
            loss = F.huber_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()

    def save(self, filepath: str):
        """Save the agent model and training state."""
        save_data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }
        torch.save(save_data, filepath)
        print(f"💾 Agent saved to: {filepath}")

    def load(self, filepath: str):
        """Load the agent model and training state."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        save_data = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(save_data['q_network_state_dict'])
        self.target_network.load_state_dict(save_data['target_network_state_dict'])
        self.optimizer.load_state_dict(save_data['optimizer_state_dict'])
        self.epsilon = save_data['epsilon']
        self.step_count = save_data['step_count']
        
        print(f"✅ Agent loaded from: {filepath}")
        print(f"📊 Resuming from step {self.step_count}, epsilon: {self.epsilon:.3f}")