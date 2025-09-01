#!/usr/bin/env python3
"""
Test Improvements - Verify system is ready for RL training
=========================================================

Tests the enhanced traffic state collection, improved signal controller,
and RL environment setup.
"""

import time
import numpy as np
from traci_manager import TraciManager, TrafficState
from signal_controller import SignalController
from rl_environment import TrafficSignalEnv

def test_enhanced_traffic_state():
    """Test enhanced traffic state collection"""
    print("🧪 Testing Enhanced Traffic State Collection...")
    
    traci_manager = TraciManager()
    
    # Start simulation
    if not traci_manager.start_simulation():
        print("❌ Failed to start simulation")
        return False
    
    # Get traffic state
    traffic_state = traci_manager.get_traffic_state()
    
    if traffic_state:
        print("✅ Traffic state collected successfully")
        print(f"   Vehicle count: {traffic_state.vehicle_count}")
        print(f"   Waiting vehicles: {traffic_state.waiting_vehicles}")
        print(f"   Queue length: {traffic_state.queue_length}")
        print(f"   Per-lane queues: {len(traffic_state.per_lane_queues) if traffic_state.per_lane_queues else 0} lanes")
        print(f"   Per-lane waiting times: {len(traffic_state.per_lane_waiting_times) if traffic_state.per_lane_waiting_times else 0} lanes")
        print(f"   Emergency vehicles: {len(traffic_state.emergency_vehicles) if traffic_state.emergency_vehicles else 0}")
        
        # Test a few simulation steps
        for i in range(10):
            traci_manager.step_simulation(1)
            time.sleep(0.1)
        
        traci_manager.stop_simulation()
        return True
    else:
        print("❌ Failed to collect traffic state")
        traci_manager.stop_simulation()
        return False

def test_improved_signal_controller():
    """Test improved signal controller logic"""
    print("\n🧪 Testing Improved Signal Controller...")
    
    traci_manager = TraciManager()
    signal_controller = SignalController(traci_manager)
    
    # Start simulation
    if not traci_manager.start_simulation():
        print("❌ Failed to start simulation")
        return False
    
    # Test signal controller decisions
    for i in range(20):
        traffic_state = traci_manager.get_traffic_state()
        if traffic_state:
            decision = signal_controller.make_decision(traffic_state)
            print(f"   Step {i+1}: Phase {decision.phase_id}, Duration {decision.duration:.1f}s, Reason: {decision.reason}")
        
        traci_manager.step_simulation(1)
        time.sleep(0.1)
    
    traci_manager.stop_simulation()
    print("✅ Signal controller tested successfully")
    return True

def test_rl_environment():
    """Test RL environment setup"""
    print("\n🧪 Testing RL Environment...")
    
    try:
        # Create environment
        env = TrafficSignalEnv(max_steps=100)
        
        # Test reset
        initial_state = env.reset()
        print(f"✅ Environment reset successful, state shape: {initial_state.shape}")
        
        # Test a few steps
        for i in range(5):
            action = env.action_space.sample()  # Random action
            state, reward, done, info = env.step(action)
            print(f"   Step {i+1}: Action {action}, Reward {reward:.3f}, Done {done}")
            
            if done:
                break
        
        env.close()
        print("✅ RL environment tested successfully")
        return True
        
    except Exception as e:
        print(f"❌ RL environment test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing System Improvements for RL Training")
    print("=" * 50)
    
    tests = [
        test_enhanced_traffic_state,
        test_improved_signal_controller,
        test_rl_environment
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    test_names = [
        "Enhanced Traffic State",
        "Improved Signal Controller", 
        "RL Environment"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name}: {status}")
    
    all_passed = all(results)
    
    if all_passed:
        print("\n🎉 All tests passed! System is ready for RL training.")
        print("\n📋 Next steps:")
        print("   1. Run: python3 train_rl_model.py --algorithm PPO --timesteps 50000")
        print("   2. Monitor training progress in training_logs/")
        print("   3. Check trained models in trained_models/")
        print("   4. Evaluate with: python3 train_rl_model.py --eval-only trained_models/best_PPO/best_model")
    else:
        print("\n⚠️ Some tests failed. Please fix issues before proceeding with RL training.")
    
    return all_passed

if __name__ == "__main__":
    main()
