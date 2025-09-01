# Traffic Simulation & RL - Hackathon Project

A real-time traffic simulation and reinforcement learning system for intelligent traffic signal control using SUMO.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- SUMO (install with `brew install sumo` on macOS)

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Test the system
python test_traci_integration.py

# Run live dashboard
python live_dashboard.py
```

## 📁 Project Structure

### Core Components
- `traci_manager.py` - SUMO simulation connection and control
- `live_metrics.py` - Real-time traffic data collection
- `signal_controller.py` - Intelligent signal control logic
- `live_dashboard.py` - Real-time monitoring dashboard

### Simulation Files
- `Sumo_env/` - SUMO configuration and network files
- `generate_traffic.py` - Traffic generation utilities
- `demo_rl_training.py` - RL training demonstration

### Documentation
- `TEAM_COLLABORATION.md` - Team workflow guide
- `PROJECT_STATUS.md` - Current project status

## 🔧 Key Features

- **Real-time traffic monitoring** with live metrics
- **Adaptive signal control** based on traffic conditions
- **Interactive dashboard** for traffic authorities
- **Multiple traffic scenarios** (uniform, tidal, asymmetric, congested)
- **RL-ready environment** for AI agent development

## 👥 Team Collaboration

See `TEAM_COLLABORATION.md` for detailed workflow and contribution guidelines.

## 📊 Current Status

See `PROJECT_STATUS.md` for detailed project status and next steps.
