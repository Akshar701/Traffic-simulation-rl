# Dashboard Package

This folder contains all dashboard-related functionality for the traffic simulation RL project.

## Contents

### Core Dashboard Files
- **`live_dashboard.py`** - Main dashboard application with real-time traffic monitoring
- **`live_metrics.py`** - Real-time metrics collection and processing
- **`reward_log.csv`** - Historical reward data for analysis

### Training Results
- **`training_results/`** - Directory containing training visualization and data
  - `training_plots.png` - Training performance plots
  - `evaluation_history.csv/json` - Evaluation metrics
  - `training_history.csv/json` - Training progress data
  - `training_losses.csv` - Detailed loss tracking

## Usage

### Running the Dashboard
```bash
cd dashboard
python3 live_dashboard.py
```

### Importing Dashboard Components
```python
from dashboard import LiveDashboard, MetricsCollector, LiveMetrics
```

## Features

- **Real-time Monitoring**: Live traffic state visualization
- **Metrics Collection**: Automated collection of traffic performance metrics
- **Training Visualization**: Historical training data and plots
- **Interactive Controls**: Start/stop simulation, scenario selection
- **Performance Charts**: Real-time efficiency, waiting time, and queue length charts

## Dependencies

The dashboard requires:
- `traci_manager.py` (from parent directory)
- `signal_controller.py` (from parent directory)
- Standard Python packages: tkinter, matplotlib, numpy
