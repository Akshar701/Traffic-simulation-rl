#!/usr/bin/env python3
"""
Dashboard Configuration and Data Structures
Defines constants, data structures, and configuration for dashboard integration
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

class TrafficScenario(Enum):
    """Available traffic scenarios"""
    UNIFORM = "uniform"
    TIDAL = "tidal"
    ASYMMETRIC = "asymmetric"
    RANDOM = "random"

class SimulationStatus(Enum):
    """Simulation status values"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

class CongestionLevel(Enum):
    """Traffic congestion levels"""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    SEVERE = "SEVERE"

@dataclass
class DashboardMetrics:
    """Standardized metrics structure for dashboard"""
    timestamp: str
    episode: int
    scenario: str
    total_vehicles: int
    avg_speed: float
    avg_waiting_time: float
    avg_queue_length: float
    efficiency_score: float
    congestion_level: str
    throughput: float
    density: float
    status: str

@dataclass
class BaselineData:
    """Baseline performance data structure"""
    scenario: str
    timestamp: str
    metrics: Dict[str, Any]
    efficiency_score: float
    congestion_level: str

@dataclass
class SessionData:
    """Session tracking data structure"""
    session_id: str
    start_time: str
    end_time: str = None
    status: str = "active"
    episodes: List[Dict] = None
    total_episodes: int = 0
    successful_episodes: int = 0

# Dashboard Configuration Constants
DASHBOARD_CONFIG = {
    "update_interval": 2.0,  # seconds
    "max_episodes_per_session": 100,
    "default_simulation_duration": 1500,  # steps
    "default_vehicle_count": 1200,
    "metrics_history_size": 1000,
    "baseline_file_prefix": "baseline_",
    "session_file_prefix": "session_",
    "episode_file_prefix": "episode_"
}

# Traffic Scenario Definitions
TRAFFIC_SCENARIOS = {
    "uniform": {
        "name": "Uniform Traffic",
        "description": "Balanced traffic from all directions",
        "route_weights": [1.0] * 12,
        "color": "#4CAF50",
        "icon": "🟢"
    },
    "tidal": {
        "name": "Tidal Traffic", 
        "description": "Heavy East-West traffic (rush hour pattern)",
        "route_weights": [0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2],
        "color": "#FF9800",
        "icon": "🟠"
    },
    "asymmetric": {
        "name": "Asymmetric Traffic",
        "description": "Heavy North+East traffic",
        "route_weights": [0.2, 0.2, 0.3, 0.3, 0.2, 0.2, 0.3, 0.2, 0.3, 0.3, 0.2, 0.3],
        "color": "#F44336", 
        "icon": "🔴"
    },
    "random": {
        "name": "Random (RL Mode)",
        "description": "Random scenario with variability for RL training",
        "route_weights": None,  # Will be randomly selected
        "color": "#9C27B0",
        "icon": "🟣"
    }
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    "efficiency_score": {
        "excellent": 80,
        "good": 70,
        "moderate": 50,
        "poor": 30
    },
    "waiting_time": {
        "low": 10,
        "moderate": 30,
        "high": 60
    },
    "queue_length": {
        "low": 2,
        "moderate": 5,
        "high": 10
    },
    "speed": {
        "low": 10,
        "moderate": 15,
        "high": 20
    }
}

# Dashboard Widget Configuration
WIDGET_CONFIG = {
    "metrics_cards": {
        "efficiency_score": {
            "title": "Efficiency Score",
            "unit": "/100",
            "color_scheme": "green_red",
            "thresholds": [30, 50, 70, 80]
        },
        "avg_waiting_time": {
            "title": "Average Waiting Time",
            "unit": "seconds",
            "color_scheme": "red_green",
            "thresholds": [60, 30, 10]
        },
        "avg_speed": {
            "title": "Average Speed",
            "unit": "m/s",
            "color_scheme": "green_red",
            "thresholds": [10, 15, 20]
        },
        "total_vehicles": {
            "title": "Total Vehicles",
            "unit": "",
            "color_scheme": "blue",
            "thresholds": []
        }
    },
    "charts": {
        "metrics_over_time": {
            "title": "Performance Over Time",
            "metrics": ["efficiency_score", "avg_waiting_time", "avg_speed"],
            "chart_type": "line"
        },
        "scenario_comparison": {
            "title": "Scenario Performance Comparison",
            "metrics": ["efficiency_score", "avg_waiting_time"],
            "chart_type": "bar"
        },
        "congestion_distribution": {
            "title": "Congestion Level Distribution",
            "metrics": ["congestion_level"],
            "chart_type": "pie"
        }
    }
}

# API Response Templates
API_RESPONSES = {
    "success": {
        "status": "success",
        "message": "Operation completed successfully"
    },
    "error": {
        "status": "error",
        "message": "An error occurred"
    },
    "no_data": {
        "status": "no_data",
        "message": "No data available"
    }
}

# File Paths and Directories
PATHS = {
    "output_dir": "Sumo_env/Single intersection lhd",
    "baseline_dir": "baselines",
    "session_dir": "sessions",
    "episode_dir": "episodes",
    "logs_dir": "logs"
}

# Real-time Collection Settings
REALTIME_CONFIG = {
    "collection_interval": 2.0,  # seconds
    "max_queue_size": 1000,
    "timeout": 300,  # seconds
    "retry_interval": 5.0  # seconds
}

def get_scenario_info(scenario_id: str) -> Dict:
    """Get scenario information by ID"""
    return TRAFFIC_SCENARIOS.get(scenario_id, {
        "name": "Unknown Scenario",
        "description": "Unknown scenario type",
        "route_weights": [1.0] * 12,
        "color": "#808080",
        "icon": "❓"
    })

def get_performance_level(metric: str, value: float) -> str:
    """Get performance level based on metric and value"""
    thresholds = PERFORMANCE_THRESHOLDS.get(metric, {})
    
    if metric == "efficiency_score":
        if value >= thresholds.get("excellent", 80):
            return "excellent"
        elif value >= thresholds.get("good", 70):
            return "good"
        elif value >= thresholds.get("moderate", 50):
            return "moderate"
        else:
            return "poor"
    elif metric == "waiting_time":
        if value <= thresholds.get("low", 10):
            return "low"
        elif value <= thresholds.get("moderate", 30):
            return "moderate"
        else:
            return "high"
    elif metric == "speed":
        if value >= thresholds.get("high", 20):
            return "high"
        elif value >= thresholds.get("moderate", 15):
            return "moderate"
        else:
            return "low"
    
    return "unknown"

def format_metric_value(metric: str, value: float) -> str:
    """Format metric value for display"""
    if metric == "efficiency_score":
        return f"{value:.1f}/100"
    elif metric == "avg_speed":
        return f"{value:.1f} m/s ({value*3.6:.1f} km/h)"
    elif metric == "avg_waiting_time":
        return f"{value:.1f} seconds"
    elif metric == "total_vehicles":
        return f"{value:,}"
    else:
        return f"{value:.2f}"

# Export all configurations
__all__ = [
    'TrafficScenario',
    'SimulationStatus', 
    'CongestionLevel',
    'DashboardMetrics',
    'BaselineData',
    'SessionData',
    'DASHBOARD_CONFIG',
    'TRAFFIC_SCENARIOS',
    'PERFORMANCE_THRESHOLDS',
    'WIDGET_CONFIG',
    'API_RESPONSES',
    'PATHS',
    'REALTIME_CONFIG',
    'get_scenario_info',
    'get_performance_level',
    'format_metric_value'
]
