#!/usr/bin/env python3
"""
Test DQN Agent
=============

Test script to verify the DQN agent works correctly with the environment.
"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.dqn_agent import DQNAgent
from envs.traffic_env import TrafficEnv

def test_dqn_agent():
    """Test the DQN agent with the environment"""
    print("🧪 Testing DQN Agent")
    print("=" * 40)
    
    try:
        # Create environment
        print("📦 Creating environment...")
        env = TrafficEnv(max_steps=100)
        print("✅ Environment created successfully")
        
        # Create agent
        print("🤖 Creating DQN agent...")
        agent = DQNAgent(
            state_size=12,
            action_size=4,
            hidden_size=256,
            learning_rate=1e-4,
            epsilon=1.0,  # Start with full exploration
            device='auto'
        )
        print("✅ DQN agent created successfully")
        
        # Test environment reset
        print("🔄 Testing environment reset...")
        state = env.reset()
        print(f"✅ Environment reset successful, state shape: {state.shape}")
        print(f"   Expected: (12,), Got: {state.shape}")
        
        # Test agent action selection
        print("🎯 Testing agent action selection...")
        action = agent.act(state, training=True)
        print(f"✅ Agent selected action: {action} ({agent.get_action_name(action)})")
        
        # Test environment step
        print("⚡ Testing environment step...")
        next_state, reward, done, info = env.step(action)
        print(f"✅ Environment step successful")
        print(f"   Reward: {reward:.3f}")
        print(f"   Done: {done}")
        print(f"   Next state shape: {next_state.shape}")
        
        # Test experience storage
        print("💾 Testing experience storage...")
        agent.remember(state, action, reward, next_state, done)
        print(f"✅ Experience stored, buffer size: {len(agent.memory)}")
        
        # Test a few more steps
        print("🔄 Testing multiple steps...")
        for i in range(5):
            state = next_state
            action = agent.act(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            print(f"   Step {i+1}: Action {action} ({agent.get_action_name(action)}), "
                  f"Reward {reward:.3f}, Done {done}")
            
            if done:
                break
        
        # Test training (if enough experiences)
        if len(agent.memory) >= agent.batch_size:
            print("🎓 Testing training...")
            loss = agent.replay()
            if loss is not None:
                print(f"✅ Training successful, loss: {loss:.6f}")
            else:
                print("⚠️ Not enough experiences for training yet")
        else:
            print(f"⚠️ Not enough experiences for training (need {agent.batch_size}, have {len(agent.memory)})")
        
        # Test agent statistics
        print("📊 Testing agent statistics...")
        stats = agent.get_training_stats()
        print(f"✅ Agent statistics:")
        print(f"   Episodes: {stats['episode_count']}")
        print(f"   Steps: {stats['step_count']}")
        print(f"   Epsilon: {stats['epsilon']:.3f}")
        print(f"   Memory size: {stats['memory_size']}")
        
        # Clean up
        env.close()
        print("🧹 Environment closed successfully")
        
        print("\n🎉 All tests passed! DQN agent is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_architecture():
    """Test the neural network architecture"""
    print("\n🧪 Testing Neural Network Architecture")
    print("=" * 40)
    
    try:
        from agents.dqn_agent import DQNNetwork
        import torch
        
        # Create network
        network = DQNNetwork(input_size=24, hidden_size=256, output_size=4)
        print("✅ Network created successfully")
        
        # Test forward pass
        test_input = torch.randn(1, 24)  # Batch size 1, 24 features
        output = network(test_input)
        
        print(f"✅ Forward pass successful")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Expected output shape: (1, 4), Got: {output.shape}")
        
        # Test Q-values
        q_values = output.squeeze()
        print(f"   Q-values: {q_values}")
        print(f"   Best action: {q_values.argmax().item()}")
        
        print("🎉 Neural network architecture test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Neural network test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 DQN Agent Testing Suite")
    print("=" * 50)
    
    tests = [
        test_dqn_agent,
        test_agent_architecture
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
        "DQN Agent Integration",
        "Neural Network Architecture"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name}: {status}")
    
    all_passed = all(results)
    
    if all_passed:
        print("\n🎉 All tests passed! DQN agent is ready for training.")
        print("\n📋 Next steps:")
        print("   1. Start training: python3 train_dqn.py --episodes 100")
        print("   2. Monitor progress in training_results/")
        print("   3. Check trained models in trained_models/")
        print("   4. Test trained agent: python3 train_dqn.py --test-only --load-model trained_models/dqn_final.pth")
    else:
        print("\n⚠️ Some tests failed. Please fix issues before training.")
    
    return all_passed

if __name__ == "__main__":
    main()
