# 🚦 Traffic Signal Control with Reinforcement Learning

A modular traffic signal control system using SUMO simulation and reinforcement learning for intelligent traffic management.

## 📁 **Project Structure**

```
├── agents/
│   ├── __init__.py          # Agents package
│   └── dqn_agent.py         # Enhanced DQN agent
├── envs/
│   ├── __init__.py          # Environment package
│   └── traffic_env.py       # Gym-compatible traffic environment
├── utils/
│   ├── __init__.py          # Utilities package
│   ├── state_utils.py       # State extraction utilities
│   └── reward_utils.py      # Reward calculation utilities
├── Sumo_env/
│   └── gpt_newint/  # SUMO simulation files
├── traci_manager.py         # SUMO TraCI interface
├── signal_controller.py     # Traffic signal control logic
├── live_dashboard.py        # Real-time monitoring dashboard
├── live_metrics.py          # Live metrics collection
├── generate_traffic.py      # Traffic generation utilities
├── train_dqn.py             # DQN training script
├── test_dqn_agent.py        # DQN agent testing
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 **Quick Start**

### **Installation**
```bash
pip install -r requirements.txt
```

### **Run Dashboard**
```bash
python3 live_dashboard.py
```

### **Use Environment for RL**
```python
from envs.traffic_env import TrafficEnv

# Create environment
env = TrafficEnv()

# Reset for new episode
state = env.reset()

# Take action
action = 0  # NS Green
state, reward, done, info = env.step(action)

# Close environment
env.close()
```

### **Train DQN Agent (GPU Optimized)**
```python
from agents.dqn_agent import DQNAgent
from envs.traffic_env import TrafficEnv

# Create environment and agent with GPU
env = TrafficEnv()
agent = DQNAgent(state_size=12, action_size=4, device='cuda', mixed_precision=True)

# Train for one episode
state = env.reset()
total_reward = 0
for step in range(1000):
    action = agent.act(state)
    next_state, reward, done, info = env.step(action)
    agent.remember(state, action, reward, next_state, done)
    
    if len(agent.memory) >= agent.batch_size:
        loss = agent.replay()
    
    state = next_state
    total_reward += reward
    
    if done:
        break

# Save trained agent
agent.save("trained_model.pth")
```

### **GPU Training Commands**
```bash
# Check GPU setup
python3 check_gpu.py

# Train with GPU acceleration
python3 train_dqn.py --episodes 500 --device cuda --batch-size 64

# Train with custom settings
python3 train_dqn.py --episodes 1000 --device cuda --batch-size 128 --memory-size 20000
```

## 🎯 **Key Features**

### **Enhanced DQN Agent - GPU Optimized**
- **Neural Network**: 24→256→256→4 MLP with ReLU activation
- **Experience Replay**: 10,000 experience buffer for stable learning
- **Target Network**: Separate network for stable Q-value estimation
- **Epsilon-Greedy**: Exploration strategy with decay (1.0 → 0.01)
- **GPU Acceleration**: CUDA support with mixed precision training (FP16)
- **Memory Optimization**: Automatic GPU memory management and cleanup

### **Gym-Compatible Environment**
- **State Space**: 12-dimensional vector (8 queue lengths + 4 one-hot phase encoding)
- **Action Space**: 4 discrete actions (NS Green, EW Green, Extend, Skip)
- **Reward Function**: Multi-component based on efficiency, throughput, waiting time

### **Modular Architecture**
- **State Utils**: Extract traffic state information
- **Reward Utils**: Calculate rewards with CSV logging
- **Environment**: Clean gym interface for RL algorithms

### **Real-time Monitoring**
- Live dashboard with metrics visualization
- Performance charts and signal control
- Multiple traffic scenarios (uniform, tidal, asymmetric, congested)

## 📊 **Environment Details**

### **State Representation**
12-dimensional vector containing:
- Queue lengths for 8 lane groups (N_straight_left_q, N_right_q, S_straight_left_q, S_right_q, E_straight_left_q, E_right_q, W_straight_left_q, W_right_q)
- 4-dimensional one-hot encoding of active green phase (0-3)

### **Actions**
- **0**: NS_Left_Straight (North-South left-turn + straight lanes green, 30s)
- **1**: NS_Yellow (North-South yellow transition, 3s)
- **2**: EW_Left_Straight (East-West left-turn + straight lanes green, 30s)
- **3**: EW_Yellow (East-West yellow transition, 3s)

### **Reward Function**
Simple reward function: `R = (prev_waiting_time - curr_waiting_time) - 0.1 * total_queue_length`
- **Positive reward** when waiting time decreases
- **Small penalty (0.1)** for large queues to prevent ignoring fairness
- **Clean implementation** with no fairness or throughput terms

## 🔧 **Development**

### **Adding New Features**
- **State**: Modify `utils/state_utils.py`
- **Reward**: Modify `utils/reward_utils.py`
- **Environment**: Modify `envs/traffic_env.py`

### **Testing & GPU Setup**
```python
# Check GPU setup and compatibility
python3 check_gpu.py

# Test DQN agent
python3 test_dqn_agent.py

# Test environment
from envs.traffic_env import TrafficEnv
env = TrafficEnv()
state = env.reset()
# Test your changes here
env.close()
```

## 📈 **Performance Metrics**

The system tracks:
- **Efficiency Score**: Overall traffic flow performance
- **Average Waiting Time**: Vehicle waiting times
- **Queue Length**: Number of stopped vehicles
- **Throughput**: Average vehicle speed and count

## 🤝 **Contributing**

See `CONTRIBUTING.md` for development guidelines and team collaboration information.

## 📄 **License**

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

**Status**: ✅ **DQN Agent GPU Optimized & Ready**

The repository now includes a complete Enhanced DQN agent with GPU acceleration and training infrastructure.
