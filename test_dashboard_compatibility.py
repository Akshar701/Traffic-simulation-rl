#!/usr/bin/env python3
"""
Test Dashboard Compatibility
Verifies that all dashboard-compatible components work together
"""

import sys
import os
import json
from datetime import datetime

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from generate_traffic import TrafficGenerator
        print("✓ TrafficGenerator imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import TrafficGenerator: {e}")
        return False
    
    try:
        from dashboard_metrics import EnhancedTrafficAnalyzer, RealTimeMetricsCollector
        print("✓ EnhancedTrafficAnalyzer and RealTimeMetricsCollector imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import dashboard_metrics: {e}")
        return False
    
    try:
        from traffic_api import TrafficAPI
        print("✓ TrafficAPI imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import TrafficAPI: {e}")
        return False
    
    try:
        from dashboard_config import (
            TrafficScenario, SimulationStatus, CongestionLevel,
            DashboardMetrics, BaselineData, SessionData,
            DASHBOARD_CONFIG, TRAFFIC_SCENARIOS, PERFORMANCE_THRESHOLDS,
            get_scenario_info, get_performance_level, format_metric_value
        )
        print("✓ Dashboard configuration imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import dashboard_config: {e}")
        return False
    
    return True

def test_traffic_generator():
    """Test enhanced traffic generator"""
    print("\nTesting enhanced traffic generator...")
    
    try:
        from generate_traffic import TrafficGenerator
        
        generator = TrafficGenerator(1500, 1200)
        
        # Test dashboard data method
        dashboard_data = generator.get_dashboard_data()
        expected_keys = ['current_episode', 'current_scenario', 'metrics_history', 'total_episodes', 'last_update']
        
        for key in expected_keys:
            if key not in dashboard_data:
                print(f"✗ Missing key in dashboard_data: {key}")
                return False
        
        print("✓ TrafficGenerator dashboard methods work correctly")
        return True
        
    except Exception as e:
        print(f"✗ TrafficGenerator test failed: {e}")
        return False

def test_metrics_analyzer():
    """Test enhanced metrics analyzer"""
    print("\nTesting enhanced metrics analyzer...")
    
    try:
        from dashboard_metrics import EnhancedTrafficAnalyzer
        
        analyzer = EnhancedTrafficAnalyzer()
        
        # Test dashboard data method
        dashboard_data = analyzer.get_dashboard_data("test_scenario")
        expected_keys = ['scenario', 'timestamp', 'metrics', 'efficiency_score', 'congestion_level', 'comparison', 'summary']
        
        for key in expected_keys:
            if key not in dashboard_data:
                print(f"✗ Missing key in dashboard_data: {key}")
                return False
        
        print("✓ EnhancedTrafficAnalyzer dashboard methods work correctly")
        return True
        
    except Exception as e:
        print(f"✗ EnhancedTrafficAnalyzer test failed: {e}")
        return False

def test_traffic_api():
    """Test traffic API"""
    print("\nTesting traffic API...")
    
    try:
        from traffic_api import TrafficAPI
        
        api = TrafficAPI()
        
        # Test system status
        status = api.get_system_status()
        if status['status'] != 'success':
            print(f"✗ System status failed: {status}")
            return False
        
        # Test available scenarios
        scenarios = api.get_available_scenarios()
        if scenarios['status'] != 'success':
            print(f"✗ Available scenarios failed: {scenarios}")
            return False
        
        # Test session management
        session_result = api.start_session("test_session")
        if session_result['status'] != 'success':
            print(f"✗ Session start failed: {session_result}")
            return False
        
        # Test session summary
        summary = api.get_session_summary()
        if summary['status'] != 'success':
            print(f"✗ Session summary failed: {summary}")
            return False
        
        # End session
        end_result = api.end_session()
        if end_result['status'] != 'success':
            print(f"✗ Session end failed: {end_result}")
            return False
        
        print("✓ TrafficAPI methods work correctly")
        return True
        
    except Exception as e:
        print(f"✗ TrafficAPI test failed: {e}")
        return False

def test_dashboard_config():
    """Test dashboard configuration"""
    print("\nTesting dashboard configuration...")
    
    try:
        from dashboard_config import (
            get_scenario_info, get_performance_level, format_metric_value,
            TRAFFIC_SCENARIOS, PERFORMANCE_THRESHOLDS
        )
        
        # Test scenario info
        uniform_info = get_scenario_info("uniform")
        if uniform_info['name'] != "Uniform Traffic":
            print(f"✗ Scenario info failed: {uniform_info}")
            return False
        
        # Test performance level
        level = get_performance_level("efficiency_score", 85.0)
        if level != "excellent":
            print(f"✗ Performance level failed: {level}")
            return False
        
        # Test metric formatting
        formatted = format_metric_value("efficiency_score", 85.5)
        if formatted != "85.5/100":
            print(f"✗ Metric formatting failed: {formatted}")
            return False
        
        print("✓ Dashboard configuration functions work correctly")
        return True
        
    except Exception as e:
        print(f"✗ Dashboard configuration test failed: {e}")
        return False

def test_data_structures():
    """Test data structures"""
    print("\nTesting data structures...")
    
    try:
        from dashboard_config import DashboardMetrics, BaselineData, SessionData
        
        # Test DashboardMetrics
        metrics = DashboardMetrics(
            timestamp="2024-01-01T12:00:00",
            episode=1,
            scenario="uniform",
            total_vehicles=1000,
            avg_speed=15.5,
            avg_waiting_time=20.0,
            avg_queue_length=3.0,
            efficiency_score=75.0,
            congestion_level="MODERATE",
            throughput=800,
            density=0.3,
            status="completed"
        )
        
        if metrics.efficiency_score != 75.0:
            print(f"✗ DashboardMetrics failed: {metrics}")
            return False
        
        # Test BaselineData
        baseline = BaselineData(
            scenario="uniform",
            timestamp="2024-01-01T12:00:00",
            metrics={"total_vehicles": 1000},
            efficiency_score=80.0,
            congestion_level="LOW"
        )
        
        if baseline.efficiency_score != 80.0:
            print(f"✗ BaselineData failed: {baseline}")
            return False
        
        # Test SessionData
        session = SessionData(
            session_id="test_session",
            start_time="2024-01-01T12:00:00",
            episodes=[],
            total_episodes=0,
            successful_episodes=0
        )
        
        if session.session_id != "test_session":
            print(f"✗ SessionData failed: {session}")
            return False
        
        print("✓ Data structures work correctly")
        return True
        
    except Exception as e:
        print(f"✗ Data structures test failed: {e}")
        return False

def test_integration():
    """Test integration between components"""
    print("\nTesting component integration...")
    
    try:
        from traffic_api import TrafficAPI
        from dashboard_config import get_scenario_info
        
        api = TrafficAPI()
        
        # Start session
        api.start_session("integration_test")
        
        # Get scenarios and test integration
        scenarios = api.get_available_scenarios()
        for scenario_data in scenarios['data']['scenarios']:
            scenario_id = scenario_data['id']
            scenario_info = get_scenario_info(scenario_id)
            
            if scenario_info['name'] != scenario_data['name']:
                print(f"✗ Scenario integration failed for {scenario_id}")
                return False
        
        # End session
        api.end_session()
        
        print("✓ Component integration works correctly")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def main():
    """Run all compatibility tests"""
    print("Dashboard Compatibility Test Suite")
    print("=" * 40)
    
    tests = [
        ("Imports", test_imports),
        ("Traffic Generator", test_traffic_generator),
        ("Metrics Analyzer", test_metrics_analyzer),
        ("Traffic API", test_traffic_api),
        ("Dashboard Config", test_dashboard_config),
        ("Data Structures", test_data_structures),
        ("Integration", test_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"✗ {test_name} test failed")
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Dashboard compatibility is ready.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
