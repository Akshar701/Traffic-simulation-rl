#!/usr/bin/env python3
"""
Test Training Integration
========================

Quick test script to demonstrate the new TensorBoard and logging integration.
This script runs a short training session to verify everything works correctly.

Usage:
    python test_training_integration.py
"""

import os
import sys
import time
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_dqn import DQNTrainer

def test_training_integration():
    """Test the new training integration features"""
    print("🧪 Testing Training Integration Features")
    print("=" * 50)
    
    # Configuration for quick test
    config = {
        'results_dir': 'test_training_results',
        'models_dir': 'test_trained_models',
        'config_file': 'Sumo_env/gpt_newint/intersection.sumocfg',
        'max_steps': 100,  # Short episodes for testing
        'learning_rate': 1e-3,
        'epsilon': 0.5,  # Start with some exploration
        'epsilon_min': 0.1,
        'epsilon_decay': 0.99,
        'memory_size': 1000,
        'batch_size': 32,
        'target_update_freq': 100,
        'hidden_size': 128,
        'mixed_precision': False  # Disable for testing
    }
    
    try:
        # Initialize trainer
        print("🚀 Initializing DQN Trainer...")
        trainer = DQNTrainer(config)
        
        # Run short training session
        print("🏃 Running short training session (5 episodes)...")
        start_time = time.time()
        
        training_history = trainer.train(
            episodes=5,
            eval_freq=3,  # Evaluate after episode 3
            save_freq=5   # Save after episode 5
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✅ Training completed in {training_time:.2f} seconds")
        print(f"📊 Training episodes: {len(training_history)}")
        
        # Check if files were created
        results_dir = config['results_dir']
        expected_files = [
            'logs/training_*.log',
            'tensorboard_logs/run_*/',
            'training_history.json',
            'training_history.csv',
            'evaluation_history.json',
            'evaluation_history.csv'
        ]
        
        print("\n📁 Checking generated files:")
        for file_pattern in expected_files:
            if '*' in file_pattern:
                # Check if directory exists
                dir_path = os.path.join(results_dir, file_pattern.split('*')[0])
                if os.path.exists(dir_path):
                    files = os.listdir(dir_path)
                    if files:
                        print(f"   ✅ {file_pattern} - Found {len(files)} files")
                    else:
                        print(f"   ⚠️  {file_pattern} - Directory empty")
                else:
                    print(f"   ❌ {file_pattern} - Directory not found")
            else:
                file_path = os.path.join(results_dir, file_pattern)
                if os.path.exists(file_path):
                    print(f"   ✅ {file_pattern}")
                else:
                    print(f"   ❌ {file_pattern}")
        
        # Show TensorBoard instructions
        print(f"\n📊 TensorBoard Integration:")
        print(f"   📁 Logs directory: {results_dir}/tensorboard_logs/")
        print(f"   🚀 To view TensorBoard: python start_tensorboard.py --results-dir {results_dir}")
        print(f"   🌐 Or manually: tensorboard --logdir {results_dir}/tensorboard_logs/")
        
        # Show logging instructions
        print(f"\n📝 Logging Integration:")
        logs_dir = os.path.join(results_dir, 'logs')
        if os.path.exists(logs_dir):
            log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
            if log_files:
                latest_log = sorted(log_files)[-1]
                print(f"   📄 Latest log file: {logs_dir}/{latest_log}")
                print(f"   📖 To view logs: tail -f {logs_dir}/{latest_log}")
        
        print(f"\n🎉 Integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🧪 Training Integration Test")
    print("=" * 50)
    print("This test will:")
    print("  • Initialize DQN trainer with logging and TensorBoard")
    print("  • Run 5 training episodes")
    print("  • Verify file generation")
    print("  • Show usage instructions")
    print("=" * 50)
    
    # Check if SUMO config exists
    config_file = 'Sumo_env/gpt_newint/intersection.sumocfg'
    if not os.path.exists(config_file):
        print(f"❌ SUMO config file not found: {config_file}")
        print("💡 Make sure you're running from the project root directory")
        return 1
    
    # Run test
    success = test_training_integration()
    
    if success:
        print("\n✅ All tests passed! The integration is working correctly.")
        print("\n💡 Next steps:")
        print("   1. Run full training: python train_dqn.py --episodes 100")
        print("   2. View TensorBoard: python start_tensorboard.py")
        print("   3. Check logs: tail -f training_results/logs/training_*.log")
        return 0
    else:
        print("\n❌ Tests failed. Check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
