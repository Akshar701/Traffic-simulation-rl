#!/usr/bin/env python3
"""
Test New Structure - Verify modular components work correctly
============================================================

Tests the new gym-compatible environment and utility functions
to ensure everything works before model implementation.
"""

import sys
import os
import numpy as np
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.traffic_env import TrafficEnv
from utils.state_utils import get_24d_state_vector, get_state_summary
from utils.reward_utils import calculate_reward, reset_reward_calculator

def test_state_utils():
    """Test state extraction utilities"""
    print("🧪 Testing State Utils...")
    
    try:
        # Test state vector extraction (will fail without simulation running)
        state_vector = get_24d_state_vector()
        print(f"✅ State vector shape: {state_vector.shape}")
        print(f"   Expected: (24,), Got: {state_vector.shape}")
        
        # Test state summary
        state_summary = get_state_summary()
        print(f"✅ State summary keys: {list(state_summary.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ State utils test failed: {e}")
        return False

def test_reward_utils():
    """Test reward calculation utilities"""
    print("\n🧪 Testing Reward Utils...")
    
    try:
        # Test reward calculation with dummy values
        reward = calculate_reward(
            current_waiting_time=10.0,
            current_queue_length=5,
            avg_speed=8.0,
            vehicle_count=15
        )
        print(f"✅ Reward calculation: {reward:.3f}")
        
        # Test waiting time change function
        from utils.reward_utils import reward_waiting_time_change
        waiting_change = reward_waiting_time_change(prev_wait=15.0, curr_wait=10.0)
        print(f"✅ Waiting time change reward: {waiting_change:.3f}")
        
        # Test reset function
        reset_reward_calculator()
        print("✅ Reward calculator reset successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Reward utils test failed: {e}")
        return False

def test_traffic_env():
    """Test gym-compatible traffic environment"""
    print("\n🧪 Testing Traffic Environment...")
    
    try:
        # Create environment
        env = TrafficEnv(max_steps=100)
        print("✅ Environment created successfully")
        
        # Test action space
        print(f"✅ Action space: {env.action_space}")
        print(f"   Actions: {env.action_space.n}")
        
        # Test observation space
        print(f"✅ Observation space: {env.observation_space}")
        print(f"   Shape: {env.observation_space.shape}")
        
        # Test reset
        initial_state = env.reset()
        print(f"✅ Environment reset successful")
        print(f"   Initial state shape: {initial_state.shape}")
        print(f"   Expected: (24,), Got: {initial_state.shape}")
        
        # Test a few steps
        print("\n📊 Testing environment steps...")
        for i in range(5):
            action = env.action_space.sample()  # Random action
            state, reward, done, info = env.step(action)
            print(f"   Step {i+1}: Action {action}, Reward {reward:.3f}, Done {done}")
            
            if done:
                break
        
        # Test episode summary
        summary = env.get_episode_summary()
        print(f"✅ Episode summary: {summary}")
        
        # Clean up
        env.close()
        print("✅ Environment closed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Traffic environment test failed: {e}")
        return False

def test_integration():
    """Test integration between components"""
    print("\n🧪 Testing Component Integration...")
    
    try:
        # Create environment
        env = TrafficEnv(max_steps=50)
        
        # Reset environment
        state = env.reset()
        
        # Test full episode
        total_reward = 0
        step_count = 0
        
        while step_count < 10:  # Limit to 10 steps for testing
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            # Verify state shape
            assert state.shape == (24,), f"Expected state shape (24,), got {state.shape}"
            
            # Verify reward is a number
            assert isinstance(reward, (int, float)), f"Expected numeric reward, got {type(reward)}"
            
            if done:
                break
        
        print(f"✅ Integration test successful")
        print(f"   Steps completed: {step_count}")
        print(f"   Total reward: {total_reward:.3f}")
        print(f"   Final state shape: {state.shape}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing New Modular Structure")
    print("=" * 50)
    
    tests = [
        test_state_utils,
        test_reward_utils,
        test_traffic_env,
        test_integration
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
        "State Utils",
        "Reward Utils", 
        "Traffic Environment",
        "Integration"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name}: {status}")
    
    all_passed = all(results)
    
    if all_passed:
        print("\n🎉 All tests passed! New structure is ready for model implementation.")
        print("\n📋 Structure Summary:")
        print("   📁 envs/traffic_env.py - Gym-compatible environment")
        print("   📁 utils/state_utils.py - 24D state vector extraction")
        print("   📁 utils/reward_utils.py - Reward calculation with CSV logging")
        print("   📁 utils/__init__.py - Package initialization")
        print("   📁 envs/__init__.py - Package initialization")
        print("\n🚀 Ready to start building RL models!")
    else:
        print("\n⚠️ Some tests failed. Please fix issues before model implementation.")
    
    return all_passed

if __name__ == "__main__":
    main()
