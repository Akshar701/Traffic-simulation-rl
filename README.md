# 🚦 Traffic Signal Control with Reinforcement Learning

A modular traffic signal control system using SUMO simulation and reinforcement learning for intelligent traffic management. This repository includes both a Tkinter-based live dashboard and a comprehensive Streamlit web dashboard for real-time traffic monitoring and control.

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

### **Run Dashboards**

#### **Option 1: Tkinter Live Dashboard (Local)**
```bash
python3 dashboard/live_dashboard.py
```

#### **Option 2: Streamlit Web Dashboard (Recommended)**
```bash
# Navigate to the Streamlit dashboard directory
cd "C:\Users\adity\OneDrive\Desktop\AI TRAFFIC\AI-Powered-Traffic-Management-System-"

# Run the Streamlit dashboard
streamlit run app.py --server.port 8501
```

**Access the dashboard at:** http://localhost:8501

#### **Option 3: Using Provided Scripts**
```bash
# Windows Batch Script
.\run_streamlit_dashboard.bat

# Windows PowerShell Script  
.\run_streamlit_dashboard.ps1
```

**Note:** The scripts point to the separate Streamlit dashboard project located at `C:\Users\adity\OneDrive\Desktop\AI TRAFFIC\AI-Powered-Traffic-Management-System-`

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

### **Dual Dashboard System**
- **Tkinter Dashboard**: Local desktop application for real-time monitoring
- **Streamlit Dashboard**: Modern web-based dashboard with advanced analytics
- **Real-time Monitoring**: Live metrics visualization and performance charts
- **Multiple Traffic Scenarios**: uniform, tidal, asymmetric, congested

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

## 🌐 **Streamlit Dashboard**

### **Overview**
The Streamlit dashboard is a modern, web-based interface for traffic management that provides:
- **Real-time Analytics**: Live traffic metrics and performance indicators
- **Interactive Visualizations**: Dynamic charts and graphs using Plotly
- **AI Integration**: Reinforcement learning model integration and comparison
- **Professional UI**: Dark theme with responsive design
- **Modular Architecture**: Organized into separate component files

### **Dashboard Features**
- **KPI Monitoring**: Travel time, wait time, vehicle density tracking
- **Intersection Control**: Real-time traffic light management
- **Performance Analytics**: AI vs traditional control comparison
- **Video Monitoring**: Camera feed integration and object detection
- **Data Export**: CSV/JSON data export capabilities

### **Dashboard Structure**
```
dashboard/
├── dashboard.py                   # Main application entry point
├── config.py                      # Configuration settings
├── styles.py                      # CSS styles and theming
├── layout_components.py           # UI layout and navigation
├── kpi_components.py             # Key Performance Indicators
├── intersection_components.py     # Traffic intersection controls
├── analytics_components.py       # Performance analytics & charts
└── video_components.py           # Camera feeds & video monitoring
```

### **Streamlit Requirements**
```bash
# Core Streamlit dependencies
streamlit==1.38.0
pandas==2.2.2
numpy==1.26.4
plotly==5.24.1

# Optional RL integration
gymnasium==0.29.1
torch==2.3.1
```

### **Dashboard Configuration**
- **Port**: 8501 (default)
- **Layout**: Wide mode for better visualization
- **Theme**: Dark professional theme
- **Auto-refresh**: Configurable refresh rates
- **Data Source**: JSON files in `data/` directory

## 🛠️ **Available Commands & Scripts**

### **Training & Development**
```bash
# Check GPU setup and compatibility
python3 check_gpu.py

# Train DQN agent with GPU acceleration
python3 train_dqn.py --episodes 500 --device cuda --batch-size 64

# Test DQN agent functionality
python3 test_dqn_agent.py

# Test training integration
python3 test_training_integration.py

# Test different configurations
python3 test_different_configs.py

# Demo RL training
python3 demo_rl_training.py
```

### **Simulation & Analysis**
```bash
# Generate traffic scenarios
python3 generate_traffic.py

# Run benchmarks
python3 run_benchmarks.py

# Analyze benchmark results
python3 analyze_benchmarks.py

# Compare RL vs fixed lights
python3 compare_rl_vs_fixed.py

# Benchmark fixed lights
python3 benchmark_fixed_lights.py
```

### **Model Execution**
```bash
# Run trained model
python3 run_trained_model.py

# Run trained model with GUI
python3 run_trained_model_gui.py

# Run model controlled GUI
python3 run_model_controlled_gui.py

# Simple GUI runner
python3 simple_gui_runner.py
```

### **Monitoring & Visualization**
```bash
# Start TensorBoard for training visualization
python3 start_tensorboard.py

# Run Tkinter live dashboard
python3 dashboard/live_dashboard.py

# Run Streamlit dashboard (from correct directory)
cd "C:\Users\adity\OneDrive\Desktop\AI TRAFFIC\AI-Powered-Traffic-Management-System-"
streamlit run app.py --server.port 8501
```

### **API & Integration**
```bash
# Run RL API server
python3 api_rl.py
```

### **Windows Scripts**
```bash
# Run Streamlit dashboard (Batch)
.\run_streamlit_dashboard.bat

# Run Streamlit dashboard (PowerShell)
.\run_streamlit_dashboard.ps1
```

**Note**: The Windows scripts point to the separate Streamlit dashboard project directory.

## 📦 **Requirements & Dependencies**

### **Core Dependencies (Traffic Simulation RL)**
```bash
# Core data processing
numpy==1.26.4
pandas==2.2.2
matplotlib==3.8.4
seaborn==0.13.2

# Web framework for dashboard
Flask==3.0.3
Flask-CORS==4.0.1

# Data processing and analysis
scipy==1.13.1
scikit-learn==1.5.1

# Reinforcement Learning
gym==0.26.2
stable-baselines3==2.3.2
torch==2.3.1
tensorboard==2.17.0

# Traffic simulation
traci==1.19.0
sumolib==1.19.0

# API and utilities
requests==2.32.3
python-dotenv==1.0.1

# Testing
pytest==8.3.2
pytest-cov==5.0.0

# Development tools
black==24.4.2
flake8==7.1.1
mypy==1.10.1

# Visualization
plotly==5.22.0
dash==2.17.1
dash-bootstrap-components==1.6.0

# Configuration
pyyaml==6.0.2
configparser==7.1.0

# Logging and monitoring
loguru==0.7.2
psutil==6.0.0
```

### **Streamlit Dashboard Dependencies**
```bash
# Core Streamlit dependencies
streamlit==1.38.0
pandas==2.2.2
numpy==1.26.4
plotly==5.24.1

# Optional RL integration
gymnasium==0.29.1
torch==2.3.1
```

### **Installation Commands**
```bash
# Install main project dependencies
pip install -r requirements.txt

# Install Streamlit dashboard dependencies (if running separately)
cd "C:\Users\adity\OneDrive\Desktop\AI TRAFFIC\AI-Powered-Traffic-Management-System-"
pip install -r requirements.txt
```

### **System Requirements**
- **Python**: 3.8+ (recommended 3.9+)
- **CUDA**: 11.0+ (for GPU acceleration)
- **SUMO**: 1.19.0+ (traffic simulation)
- **Memory**: 8GB+ RAM recommended
- **Storage**: 2GB+ free space

## 🔧 **Development**

### **Adding New Features**
- **State**: Modify `utils/state_utils.py`
- **Reward**: Modify `utils/reward_utils.py`
- **Environment**: Modify `envs/traffic_env.py`
- **Dashboard**: Modify files in `dashboard/` directory

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

### **Dashboard Development**
```bash
# Test Streamlit dashboard locally
cd "C:\Users\adity\OneDrive\Desktop\AI TRAFFIC\AI-Powered-Traffic-Management-System-"
streamlit run app.py --server.port 8501

# Test individual dashboard components
python -c "import dashboard.dashboard; print('Dashboard imports successful')"
```

## 📈 **Performance Metrics**

The system tracks:
- **Efficiency Score**: Overall traffic flow performance
- **Average Waiting Time**: Vehicle waiting times
- **Queue Length**: Number of stopped vehicles
- **Throughput**: Average vehicle speed and count

## 🔧 **Troubleshooting**

### **Common Issues**

#### **Streamlit Dashboard Issues**
```bash
# Issue: "File does not exist: app.py"
# Solution: Navigate to correct directory
cd "C:\Users\adity\OneDrive\Desktop\AI TRAFFIC\AI-Powered-Traffic-Management-System-"
streamlit run app.py --server.port 8501

# Issue: "Connection refused" on localhost:8501
# Solution: Wait for server to start, then refresh browser

# Issue: PowerShell "&&" syntax error
# Solution: Use separate commands or semicolon (;)
cd "path"; streamlit run app.py
```

#### **GPU Training Issues**
```bash
# Check GPU availability
python3 check_gpu.py

# Fallback to CPU training
python3 train_dqn.py --device cpu

# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

#### **SUMO Integration Issues**
```bash
# Check SUMO installation
python -c "import traci; print('SUMO TraCI available')"

# Verify SUMO environment variables
echo $SUMO_HOME  # Linux/Mac
echo %SUMO_HOME%  # Windows
```

#### **Import Errors**
```bash
# Install missing dependencies
pip install -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"

# Test individual modules
python -c "from envs.traffic_env import TrafficEnv; print('Environment OK')"
```

### **Performance Optimization**
- **GPU Memory**: Use smaller batch sizes if running out of memory
- **Dashboard Refresh**: Adjust refresh rates for better performance
- **SUMO Simulation**: Use appropriate time steps for your hardware

## 🤝 **Contributing**

See `CONTRIBUTING.md` for development guidelines and team collaboration information.

## 📄 **License**

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

**Status**: ✅ **Complete Traffic Management System Ready**

The repository now includes:
- ✅ Enhanced DQN agent with GPU acceleration
- ✅ Dual dashboard system (Tkinter + Streamlit)
- ✅ Comprehensive training infrastructure
- ✅ Real-time monitoring and analytics
- ✅ Complete documentation and troubleshooting guide
