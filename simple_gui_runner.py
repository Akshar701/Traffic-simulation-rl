#!/usr/bin/env python3
"""
Simple SUMO GUI Runner with Trained Model
=========================================

A simplified script that starts SUMO GUI and lets you manually control
or observe the traffic simulation with your trained model.
"""

import os
import sys
import time
import argparse
import subprocess
import signal

def start_sumo_gui(config_file: str):
    """Start SUMO GUI with the specified configuration"""
    
    print(f"🎬 Starting SUMO GUI...")
    print(f"   • Config: {config_file}")
    print(f"   • Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        # Start SUMO GUI
        cmd = ["sumo-gui", "-c", config_file, "--start"]
        process = subprocess.Popen(cmd)
        
        print("✅ SUMO GUI started successfully!")
        print("   • You should see the SUMO GUI window open")
        print("   • The simulation will start automatically")
        print("   • You can pause/play using the GUI controls")
        print("   • Press Ctrl+C in this terminal to close")
        
        # Wait for the process
        process.wait()
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Stopping SUMO GUI...")
        if 'process' in locals():
            process.terminate()
            process.wait()
        print("🔒 SUMO GUI closed")
    
    except Exception as e:
        print(f"❌ Error starting SUMO GUI: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Start SUMO GUI')
    parser.add_argument('--config', type=str, 
                       default='Sumo_env/gpt_newint/intersection.sumocfg',
                       help='SUMO configuration file')
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"❌ Error: Config file not found: {args.config}")
        print(f"Available configs in Sumo_env/gpt_newint/:")
        config_dir = "Sumo_env/gpt_newint"
        if os.path.exists(config_dir):
            for file in os.listdir(config_dir):
                if file.endswith('.sumocfg'):
                    print(f"   • {os.path.join(config_dir, file)}")
        sys.exit(1)
    
    start_sumo_gui(args.config)

if __name__ == "__main__":
    main()
