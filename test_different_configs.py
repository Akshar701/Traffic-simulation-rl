#!/usr/bin/env python3
"""
Test Trained Model with Different SUMO Configurations
====================================================

This script tests the trained DQN model with different SUMO configurations
to evaluate its performance across various traffic scenarios.
"""

import os
import sys
import subprocess
import json
from datetime import datetime

def test_model_with_configs():
    """Test the trained model with different SUMO configurations"""
    
    # Available configurations
    configs = [
        {
            'name': 'Uniform Traffic',
            'file': 'Sumo_env/gpt_newint/uniform_simulation.sumocfg',
            'description': 'Uniform traffic distribution across all directions'
        },
        {
            'name': 'Congested Traffic',
            'file': 'Sumo_env/gpt_newint/congested_simulation.sumocfg',
            'description': 'High traffic density with congestion'
        },
        {
            'name': 'Tidal Traffic',
            'file': 'Sumo_env/gpt_newint/tidal_simulation.sumocfg',
            'description': 'Tidal traffic patterns (rush hour simulation)'
        },
        {
            'name': 'Random Traffic',
            'file': 'Sumo_env/gpt_newint/random_simulation.sumocfg',
            'description': 'Random traffic patterns'
        }
    ]
    
    # Available models
    models = [
        'trained_models/dqn_episode_100.pth',
        'trained_models/dqn_episode_200.pth',
        'trained_models/dqn_final.pth'
    ]
    
    results = []
    
    print("🚀 Testing Trained Models with Different Configurations")
    print("=" * 70)
    
    for model in models:
        if not os.path.exists(model):
            print(f"⚠️  Model not found: {model}")
            continue
            
        print(f"\n📊 Testing Model: {model}")
        print("-" * 50)
        
        for config in configs:
            if not os.path.exists(config['file']):
                print(f"⚠️  Config not found: {config['file']}")
                continue
            
            print(f"\n🚦 Testing: {config['name']}")
            print(f"   📄 Config: {config['file']}")
            print(f"   📝 Description: {config['description']}")
            
            try:
                # Run the model with this configuration
                cmd = [
                    'python', 'run_trained_model.py',
                    '--model', model,
                    '--config', config['file'],
                    '--episodes', '3',
                    '--save-results'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    # Extract reward from output
                    output_lines = result.stdout.split('\n')
                    reward_line = None
                    for line in output_lines:
                        if 'Mean Reward:' in line:
                            reward_line = line
                            break
                    
                    if reward_line:
                        # Parse the reward value
                        reward_str = reward_line.split('Mean Reward:')[1].split('±')[0].strip()
                        mean_reward = float(reward_str)
                        
                        result_data = {
                            'model': model,
                            'config_name': config['name'],
                            'config_file': config['file'],
                            'mean_reward': mean_reward,
                            'timestamp': datetime.now().isoformat(),
                            'status': 'success'
                        }
                        
                        # Performance assessment
                        if mean_reward > -20:
                            performance = "🟢 EXCELLENT"
                        elif mean_reward > -50:
                            performance = "🟡 GOOD"
                        elif mean_reward > -100:
                            performance = "🟠 FAIR"
                        else:
                            performance = "🔴 POOR"
                        
                        print(f"   🎯 Mean Reward: {mean_reward:.2f}")
                        print(f"   📊 Performance: {performance}")
                        
                    else:
                        result_data = {
                            'model': model,
                            'config_name': config['name'],
                            'config_file': config['file'],
                            'mean_reward': None,
                            'timestamp': datetime.now().isoformat(),
                            'status': 'success_no_reward'
                        }
                        print(f"   ⚠️  Could not extract reward from output")
                
                else:
                    result_data = {
                        'model': model,
                        'config_name': config['name'],
                        'config_file': config['file'],
                        'mean_reward': None,
                        'timestamp': datetime.now().isoformat(),
                        'status': 'error',
                        'error': result.stderr
                    }
                    print(f"   ❌ Error: {result.stderr[:100]}...")
                
                results.append(result_data)
                
            except subprocess.TimeoutExpired:
                result_data = {
                    'model': model,
                    'config_name': config['name'],
                    'config_file': config['file'],
                    'mean_reward': None,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'timeout'
                }
                results.append(result_data)
                print(f"   ⏰ Timeout after 5 minutes")
            
            except Exception as e:
                result_data = {
                    'model': model,
                    'config_name': config['name'],
                    'config_file': config['file'],
                    'mean_reward': None,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'exception',
                    'error': str(e)
                }
                results.append(result_data)
                print(f"   ❌ Exception: {str(e)}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"training_results/config_test_results_{timestamp}.json"
    
    os.makedirs("training_results", exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump({
            'test_timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)
    
    # Print summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 70)
    
    successful_tests = [r for r in results if r['status'] == 'success' and r['mean_reward'] is not None]
    
    if successful_tests:
        print(f"✅ Successful tests: {len(successful_tests)}")
        
        # Group by model
        by_model = {}
        for result in successful_tests:
            model = result['model']
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(result)
        
        for model, model_results in by_model.items():
            print(f"\n📊 {model}:")
            for result in model_results:
                performance = "🟢" if result['mean_reward'] > -20 else "🟡" if result['mean_reward'] > -50 else "🟠" if result['mean_reward'] > -100 else "🔴"
                print(f"   {performance} {result['config_name']}: {result['mean_reward']:.2f}")
        
        # Find best performing model-config combination
        best_result = max(successful_tests, key=lambda x: x['mean_reward'])
        print(f"\n🏆 Best Performance:")
        print(f"   Model: {best_result['model']}")
        print(f"   Config: {best_result['config_name']}")
        print(f"   Reward: {best_result['mean_reward']:.2f}")
    
    else:
        print("❌ No successful tests completed")
    
    print(f"\n💾 Results saved to: {results_file}")
    print("=" * 70)

if __name__ == "__main__":
    test_model_with_configs()
