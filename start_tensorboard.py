#!/usr/bin/env python3
"""
TensorBoard Launcher Script
===========================

Simple script to launch TensorBoard for viewing training logs.
This script automatically finds the latest TensorBoard logs directory
and starts TensorBoard on the default port (6006).

Usage:
    python start_tensorboard.py [--port 6006] [--logdir path/to/logs]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def find_latest_tensorboard_logs(results_dir="training_results"):
    """Find the latest TensorBoard logs directory"""
    tb_logs_dir = os.path.join(results_dir, "tensorboard_logs")
    
    if not os.path.exists(tb_logs_dir):
        print(f"❌ TensorBoard logs directory not found: {tb_logs_dir}")
        return None
    
    # Find the most recent run directory
    run_dirs = [d for d in os.listdir(tb_logs_dir) if d.startswith('run_')]
    
    if not run_dirs:
        print(f"❌ No run directories found in: {tb_logs_dir}")
        return None
    
    # Sort by creation time (most recent first)
    run_dirs.sort(key=lambda x: os.path.getctime(os.path.join(tb_logs_dir, x)), reverse=True)
    latest_run = run_dirs[0]
    
    return os.path.join(tb_logs_dir, latest_run)

def launch_tensorboard(logdir, port=6006):
    """Launch TensorBoard with the specified log directory"""
    try:
        print(f"🚀 Launching TensorBoard...")
        print(f"📊 Log directory: {logdir}")
        print(f"🌐 Port: {port}")
        print(f"🔗 URL: http://localhost:{port}")
        print("=" * 50)
        
        # Launch TensorBoard
        cmd = ["tensorboard", "--logdir", logdir, "--port", str(port)]
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching TensorBoard: {e}")
        print("💡 Make sure TensorBoard is installed: pip install tensorboard")
        return False
    except KeyboardInterrupt:
        print("\n🛑 TensorBoard stopped by user")
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Launch TensorBoard for training logs')
    parser.add_argument('--port', type=int, default=6006, help='Port for TensorBoard (default: 6006)')
    parser.add_argument('--logdir', type=str, help='Custom log directory path')
    parser.add_argument('--results-dir', type=str, default='training_results', 
                       help='Results directory to search for logs (default: training_results)')
    
    args = parser.parse_args()
    
    # Determine log directory
    if args.logdir:
        logdir = args.logdir
        if not os.path.exists(logdir):
            print(f"❌ Specified log directory not found: {logdir}")
            return 1
    else:
        logdir = find_latest_tensorboard_logs(args.results_dir)
        if not logdir:
            print("💡 Usage examples:")
            print("   python start_tensorboard.py")
            print("   python start_tensorboard.py --logdir path/to/logs")
            print("   python start_tensorboard.py --port 6007")
            return 1
    
    # Launch TensorBoard
    success = launch_tensorboard(logdir, args.port)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
