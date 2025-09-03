#!/usr/bin/env python3
"""
Live Dashboard - Real-time traffic monitoring and control
Integrates with traci-based system for live data visualization
"""

import time
import json
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

from traci_manager import TraciManager, TrafficState
from live_metrics import MetricsCollector, LiveMetrics
from signal_controller import SignalController, SignalDecision

class LiveDashboard:
    """Real-time traffic monitoring dashboard"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Traffic Simulation - Live Dashboard")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.traci_manager = TraciManager()
        self.metrics_collector = MetricsCollector(self.traci_manager)
        self.signal_controller = SignalController(self.traci_manager)
        
        # Dashboard state
        self.is_running = False
        self.update_interval = 1000  # milliseconds
        self.simulation_thread = None
        
        # Data storage for charts
        self.time_data = []
        self.efficiency_data = []
        self.waiting_time_data = []
        self.vehicle_count_data = []
        self.queue_length_data = []
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the dashboard user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control panel
        self.setup_control_panel(main_frame)
        
        # Metrics panel
        self.setup_metrics_panel(main_frame)
        
        # Charts panel
        self.setup_charts_panel(main_frame)
        
        # Status bar
        self.setup_status_bar(main_frame)
        
    def setup_control_panel(self, parent):
        """Setup control panel"""
        control_frame = ttk.LabelFrame(parent, text="Simulation Control", padding="5")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Start/Stop button
        self.start_stop_btn = ttk.Button(control_frame, text="Start Simulation", 
                                        command=self.toggle_simulation)
        self.start_stop_btn.grid(row=0, column=0, padx=5)
        
        # Scenario selection
        ttk.Label(control_frame, text="Scenario:").grid(row=0, column=1, padx=5)
        self.scenario_var = tk.StringVar(value="uniform")
        scenario_combo = ttk.Combobox(control_frame, textvariable=self.scenario_var,
                                     values=["uniform", "tidal", "asymmetric", "congested"])
        scenario_combo.grid(row=0, column=2, padx=5)
        
        # Duration input
        ttk.Label(control_frame, text="Duration (steps):").grid(row=0, column=3, padx=5)
        self.duration_var = tk.StringVar(value="1000")
        duration_entry = ttk.Entry(control_frame, textvariable=self.duration_var, width=10)
        duration_entry.grid(row=0, column=4, padx=5)
        
        # Control mode
        ttk.Label(control_frame, text="Control Mode:").grid(row=0, column=5, padx=5)
        self.control_mode_var = tk.StringVar(value="adaptive")
        control_combo = ttk.Combobox(control_frame, textvariable=self.control_mode_var,
                                    values=["static", "adaptive", "manual"])
        control_combo.grid(row=0, column=6, padx=5)
        
        # Manual control buttons
        self.manual_frame = ttk.Frame(control_frame)
        self.manual_frame.grid(row=1, column=0, columnspan=7, pady=5)
        
        ttk.Button(self.manual_frame, text="NS Green", 
                  command=lambda: self.manual_signal_change(0)).grid(row=0, column=0, padx=2)
        ttk.Button(self.manual_frame, text="EW Green", 
                  command=lambda: self.manual_signal_change(2)).grid(row=0, column=1, padx=2)
        ttk.Button(self.manual_frame, text="Extend Phase", 
                  command=self.extend_current_phase).grid(row=0, column=2, padx=2)
        ttk.Button(self.manual_frame, text="Skip Phase", 
                  command=self.skip_current_phase).grid(row=0, column=3, padx=2)
        
    def setup_metrics_panel(self, parent):
        """Setup metrics display panel"""
        metrics_frame = ttk.LabelFrame(parent, text="Live Metrics", padding="5")
        metrics_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Metrics labels
        self.metric_labels = {}
        metrics = [
            ("efficiency_score", "Efficiency Score"),
            ("avg_waiting_time", "Avg Waiting Time"),
            ("avg_speed", "Avg Speed"),
            ("vehicle_count", "Vehicle Count"),
            ("queue_length", "Queue Length"),
            ("current_phase", "Current Phase"),
            ("congestion_level", "Congestion Level")
        ]
        
        for i, (key, label) in enumerate(metrics):
            ttk.Label(metrics_frame, text=f"{label}:").grid(row=i, column=0, sticky=tk.W, pady=2)
            self.metric_labels[key] = ttk.Label(metrics_frame, text="--", font=("Arial", 10, "bold"))
            self.metric_labels[key].grid(row=i, column=1, sticky=tk.W, padx=10, pady=2)
        
        # Signal info
        signal_frame = ttk.LabelFrame(metrics_frame, text="Signal Information", padding="5")
        signal_frame.grid(row=len(metrics), column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        self.signal_labels = {}
        signal_info = [
            ("phase_name", "Phase"),
            ("phase_duration", "Duration"),
            ("signal_state", "State")
        ]
        
        for i, (key, label) in enumerate(signal_info):
            ttk.Label(signal_frame, text=f"{label}:").grid(row=i, column=0, sticky=tk.W, pady=2)
            self.signal_labels[key] = ttk.Label(signal_frame, text="--", font=("Arial", 9))
            self.signal_labels[key].grid(row=i, column=1, sticky=tk.W, padx=10, pady=2)
        
    def setup_charts_panel(self, parent):
        """Setup charts panel"""
        charts_frame = ttk.LabelFrame(parent, text="Performance Charts", padding="5")
        charts_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(2, 2, 1)  # Efficiency
        self.ax2 = self.fig.add_subplot(2, 2, 2)  # Waiting Time
        self.ax3 = self.fig.add_subplot(2, 2, 3)  # Vehicle Count
        self.ax4 = self.fig.add_subplot(2, 2, 4)  # Queue Length
        
        self.fig.tight_layout()
        
    def setup_status_bar(self, parent):
        """Setup status bar"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT)
        
        self.progress_bar = ttk.Progressbar(status_frame, mode='determinate')
        self.progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        
    def toggle_simulation(self):
        """Start or stop simulation"""
        if not self.is_running:
            self.start_simulation()
        else:
            self.stop_simulation()
    
    def start_simulation(self):
        """Start the simulation"""
        try:
            # Get simulation parameters
            scenario = self.scenario_var.get()
            duration = int(self.duration_var.get())
            control_mode = self.control_mode_var.get()
            
            # Update UI
            self.start_stop_btn.config(text="Stop Simulation")
            self.is_running = True
            self.status_label.config(text="Starting simulation...")
            
            # Start simulation in separate thread
            self.simulation_thread = threading.Thread(
                target=self.run_simulation,
                args=(scenario, duration, control_mode)
            )
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
            
            # Start UI updates
            self.root.after(self.update_interval, self.update_dashboard)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start simulation: {e}")
            self.stop_simulation()
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
        self.start_stop_btn.config(text="Start Simulation")
        self.status_label.config(text="Simulation stopped")
        
        # Stop metrics collection
        self.metrics_collector.stop_collection()
        
        # Stop traci manager
        self.traci_manager.stop_simulation()
    
    def run_simulation(self, scenario: str, duration: int, control_mode: str):
        """Run simulation in separate thread"""
        try:
            # Start simulation with appropriate config
            config_file = f"Sumo_env/Single intersection lhd/{scenario}_simulation.sumocfg"
            
            if not self.traci_manager.start_simulation(config_file):
                raise Exception("Failed to start SUMO simulation")
            
            # Start metrics collection
            self.metrics_collector.start_collection(interval=1.0)
            
            # Run simulation based on control mode
            if control_mode == "adaptive":
                self.run_adaptive_simulation(duration)
            elif control_mode == "manual":
                self.run_manual_simulation(duration)
            else:
                self.run_static_simulation(duration)
                
        except Exception as e:
            print(f"Simulation error: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Simulation failed: {e}"))
            self.root.after(0, self.stop_simulation)
    
    def run_adaptive_simulation(self, duration: int):
        """Run simulation with adaptive control"""
        current_time = 0
        while current_time < duration and self.is_running and self.traci_manager.is_simulation_running():
            # Step simulation
            self.traci_manager.step_simulation(1)
            current_time += 1
            
            # Get traffic state and make decision
            traffic_state = self.traci_manager.get_traffic_state()
            if traffic_state:
                decision = self.signal_controller.make_decision(traffic_state)
                self.signal_controller.execute_decision(decision)
            
            time.sleep(0.1)
    
    def run_manual_simulation(self, duration: int):
        """Run simulation with manual control"""
        current_time = 0
        while current_time < duration and self.is_running and self.traci_manager.is_simulation_running():
            self.traci_manager.step_simulation(1)
            current_time += 1
            time.sleep(0.1)
    
    def run_static_simulation(self, duration: int):
        """Run simulation with static control"""
        current_time = 0
        while current_time < duration and self.is_running and self.traci_manager.is_simulation_running():
            self.traci_manager.step_simulation(1)
            current_time += 1
            time.sleep(0.1)
    
    def update_dashboard(self):
        """Update dashboard display"""
        if not self.is_running:
            return
        
        try:
            # Get current metrics
            current_metrics = self.metrics_collector.get_current_metrics()
            if current_metrics:
                # Update metric labels
                self.update_metric_labels(current_metrics)
                
                # Update signal info
                self.update_signal_labels()
                
                # Update charts
                self.update_charts(current_metrics)
                
                # Update status
                self.status_label.config(text=f"Running - Time: {current_metrics.simulation_time:.1f}")
            
            # Schedule next update
            self.root.after(self.update_interval, self.update_dashboard)
            
        except Exception as e:
            print(f"Dashboard update error: {e}")
    
    def update_metric_labels(self, metrics: LiveMetrics):
        """Update metric display labels"""
        self.metric_labels["efficiency_score"].config(text=f"{metrics.efficiency_score:.1f}/100")
        self.metric_labels["avg_waiting_time"].config(text=f"{metrics.avg_waiting_time:.1f}s")
        self.metric_labels["avg_speed"].config(text=f"{metrics.avg_speed:.1f} m/s")
        self.metric_labels["vehicle_count"].config(text=str(metrics.vehicle_count))
        self.metric_labels["queue_length"].config(text=str(metrics.queue_length))
        self.metric_labels["current_phase"].config(text=str(metrics.current_phase))
        self.metric_labels["congestion_level"].config(text=metrics.congestion_level)
    
    def update_signal_labels(self):
        """Update signal information labels"""
        signal_info = self.signal_controller.get_current_signal_info()
        
        self.signal_labels["phase_name"].config(text=signal_info["phase_name"])
        self.signal_labels["phase_duration"].config(text=f"{signal_info['phase_duration']:.1f}s")
        self.signal_labels["signal_state"].config(text=signal_info["signal_state"])
    
    def update_charts(self, metrics: LiveMetrics):
        """Update performance charts"""
        # Add new data point
        self.time_data.append(metrics.simulation_time)
        self.efficiency_data.append(metrics.efficiency_score)
        self.waiting_time_data.append(metrics.avg_waiting_time)
        self.vehicle_count_data.append(metrics.vehicle_count)
        self.queue_length_data.append(metrics.queue_length)
        
        # Keep only last 100 points
        max_points = 100
        if len(self.time_data) > max_points:
            self.time_data = self.time_data[-max_points:]
            self.efficiency_data = self.efficiency_data[-max_points:]
            self.waiting_time_data = self.waiting_time_data[-max_points:]
            self.vehicle_count_data = self.vehicle_count_data[-max_points:]
            self.queue_length_data = self.queue_length_data[-max_points:]
        
        # Update plots
        self.ax1.clear()
        self.ax1.plot(self.time_data, self.efficiency_data, 'g-')
        self.ax1.set_title('Efficiency Score')
        self.ax1.set_ylabel('Score')
        self.ax1.grid(True)
        
        self.ax2.clear()
        self.ax2.plot(self.time_data, self.waiting_time_data, 'r-')
        self.ax2.set_title('Average Waiting Time')
        self.ax2.set_ylabel('Seconds')
        self.ax2.grid(True)
        
        self.ax3.clear()
        self.ax3.plot(self.time_data, self.vehicle_count_data, 'b-')
        self.ax3.set_title('Vehicle Count')
        self.ax3.set_ylabel('Count')
        self.ax3.grid(True)
        
        self.ax4.clear()
        self.ax4.plot(self.time_data, self.queue_length_data, 'orange')
        self.ax4.set_title('Queue Length')
        self.ax4.set_ylabel('Length')
        self.ax4.grid(True)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def manual_signal_change(self, phase_id: int):
        """Manually change signal phase"""
        if self.is_running:
            self.signal_controller.change_signal_phase(phase_id, 30.0)
    
    def extend_current_phase(self):
        """Extend current signal phase"""
        if self.is_running:
            signal_info = self.signal_controller.get_current_signal_info()
            current_phase = signal_info["current_phase"]
            current_duration = signal_info["phase_duration"]
            self.signal_controller.change_signal_phase(current_phase, current_duration + 10)
    
    def skip_current_phase(self):
        """Skip current signal phase"""
        if self.is_running:
            signal_info = self.signal_controller.get_current_signal_info()
            current_phase = signal_info["current_phase"]
            next_phase = self.signal_controller._get_next_phase(current_phase)
            self.signal_controller.change_signal_phase(next_phase, 30.0)
    
    def run(self):
        """Start the dashboard"""
        self.root.mainloop()

# Example usage
if __name__ == "__main__":
    dashboard = LiveDashboard()
    dashboard.run()
