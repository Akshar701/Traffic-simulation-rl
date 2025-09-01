#!/usr/bin/env python3
"""
Test Traci Integration - Verify that the new traci-based system works
Tests the core components: TraciManager, MetricsCollector, and SignalController
"""

import time
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from traci_manager import TraciManager, TrafficState
from live_metrics import MetricsCollector, LiveMetrics
from signal_controller import SignalController, SignalDecision

def test_traci_manager():
    """Test TraciManager functionality"""
    print("🧪 Testing TraciManager...")
    
    try:
        # Create traci manager
        manager = TraciManager()
        print("✅ TraciManager created successfully")
        
        # Test starting simulation
        success = manager.start_simulation()
        if success:
            print("✅ Simulation started successfully")
            
            # Test getting traffic state
            state = manager.get_traffic_state()
            if state:
                print(f"✅ Traffic state retrieved: {state.vehicle_count} vehicles")
            else:
                print("❌ Failed to get traffic state")
            
            # Test stepping simulation
            for i in range(10):
                manager.step_simulation(1)
                state = manager.get_traffic_state()
                if state:
                    print(f"   Step {i+1}: {state.vehicle_count} vehicles, "
                          f"Phase {state.current_phase}")
            
            # Test signal control
            signal_info = manager.get_signal_info()
            print(f"✅ Signal info: Phase {signal_info.get('current_phase', 'N/A')}")
            
            # Stop simulation
            manager.stop_simulation()
            print("✅ Simulation stopped successfully")
            
        else:
            print("❌ Failed to start simulation")
            return False
            
    except Exception as e:
        print(f"❌ TraciManager test failed: {e}")
        return False
    
    print("✅ TraciManager test completed successfully\n")
    return True

def test_metrics_collector():
    """Test MetricsCollector functionality"""
    print("🧪 Testing MetricsCollector...")
    
    try:
        # Create components
        manager = TraciManager()
        collector = MetricsCollector(manager)
        print("✅ MetricsCollector created successfully")
        
        # Start simulation
        if manager.start_simulation():
            print("✅ Simulation started for metrics test")
            
            # Start collection
            collector.start_collection(interval=0.5)
            print("✅ Metrics collection started")
            
            # Run for a few steps
            for i in range(20):
                manager.step_simulation(1)
                time.sleep(0.1)
                
                # Check metrics
                current_metrics = collector.get_current_metrics()
                if current_metrics:
                    print(f"   Step {i+1}: Efficiency {current_metrics.efficiency_score:.1f}, "
                          f"Congestion {current_metrics.congestion_level}")
            
            # Stop collection
            collector.stop_collection()
            print("✅ Metrics collection stopped")
            
            # Test metrics history
            history = collector.get_metrics_history(5)
            print(f"✅ Retrieved {len(history)} metrics from history")
            
            # Test performance summary
            summary = collector.get_performance_summary()
            print(f"✅ Performance summary generated: {len(summary)} items")
            
            # Stop simulation
            manager.stop_simulation()
            print("✅ Simulation stopped")
            
        else:
            print("❌ Failed to start simulation for metrics test")
            return False
            
    except Exception as e:
        print(f"❌ MetricsCollector test failed: {e}")
        return False
    
    print("✅ MetricsCollector test completed successfully\n")
    return True

def test_signal_controller():
    """Test SignalController functionality"""
    print("🧪 Testing SignalController...")
    
    try:
        # Create components
        manager = TraciManager()
        controller = SignalController(manager)
        print("✅ SignalController created successfully")
        
        # Start simulation
        if manager.start_simulation():
            print("✅ Simulation started for signal control test")
            
            # Test getting signal info
            signal_info = controller.get_current_signal_info()
            print(f"✅ Signal info: {signal_info['phase_name']}")
            
            # Test making decisions
            for i in range(15):
                manager.step_simulation(1)
                time.sleep(0.1)
                
                # Get traffic state
                traffic_state = manager.get_traffic_state()
                if traffic_state:
                    # Make decision
                    decision = controller.make_decision(traffic_state)
                    print(f"   Step {i+1}: Decision - Phase {decision.phase_id}, "
                          f"Duration {decision.duration}s, Reason: {decision.reason}")
                    
                    # Execute decision (only if different from current)
                    current_signal = controller.get_current_signal_info()
                    if decision.phase_id != current_signal["current_phase"]:
                        controller.execute_decision(decision)
                        print(f"      Signal changed to phase {decision.phase_id}")
            
            # Test control summary
            summary = controller.get_control_summary()
            print(f"✅ Control summary: {summary['total_decisions']} decisions made")
            
            # Stop simulation
            manager.stop_simulation()
            print("✅ Simulation stopped")
            
        else:
            print("❌ Failed to start simulation for signal control test")
            return False
            
    except Exception as e:
        print(f"❌ SignalController test failed: {e}")
        return False
    
    print("✅ SignalController test completed successfully\n")
    return True

def test_integration():
    """Test full system integration"""
    print("🧪 Testing Full System Integration...")
    
    try:
        # Create all components
        manager = TraciManager()
        collector = MetricsCollector(manager)
        controller = SignalController(manager)
        print("✅ All components created successfully")
        
        # Start simulation
        if manager.start_simulation():
            print("✅ Simulation started for integration test")
            
            # Start metrics collection
            collector.start_collection(interval=0.5)
            print("✅ Metrics collection started")
            
            # Run integrated simulation
            for i in range(30):
                # Step simulation
                manager.step_simulation(1)
                time.sleep(0.1)
                
                # Get traffic state
                traffic_state = manager.get_traffic_state()
                if traffic_state:
                    # Make signal decision
                    decision = controller.make_decision(traffic_state)
                    
                    # Execute decision if needed
                    current_signal = controller.get_current_signal_info()
                    if decision.phase_id != current_signal["current_phase"]:
                        controller.execute_decision(decision)
                    
                    # Get current metrics
                    current_metrics = collector.get_current_metrics()
                    if current_metrics:
                        print(f"   Step {i+1}: Phase {current_signal['current_phase']}, "
                              f"Efficiency {current_metrics.efficiency_score:.1f}, "
                              f"Vehicles {current_metrics.vehicle_count}")
            
            # Stop everything
            collector.stop_collection()
            manager.stop_simulation()
            print("✅ Integration test completed successfully")
            
        else:
            print("❌ Failed to start simulation for integration test")
            return False
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False
    
    print("✅ Full system integration test completed successfully\n")
    return True

def main():
    """Run all tests"""
    print("🚀 Starting Traci Integration Tests\n")
    
    tests = [
        ("TraciManager", test_traci_manager),
        ("MetricsCollector", test_metrics_collector),
        ("SignalController", test_signal_controller),
        ("Full Integration", test_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"📋 Running {test_name} test...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed\n")
    
    print("📊 Test Results:")
    print(f"   Passed: {passed}/{total}")
    print(f"   Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Your traci integration is working correctly.")
        print("\n🚀 You can now:")
        print("   - Run 'python live_dashboard.py' for the live dashboard")
        print("   - Use the components in your RL training")
        print("   - Build upon this foundation for your AI traffic control system")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
