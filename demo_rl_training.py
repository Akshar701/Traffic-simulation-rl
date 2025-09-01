#!/usr/bin/env python3
"""
Enhanced RL Training Demo with Congestion Scenarios
===================================================

This demo shows how to use the enhanced traffic generator for RL training
with better congestion scenarios and optimization opportunities.
"""

import os
import random
from generate_traffic import TrafficGenerator
from dashboard_metrics import EnhancedTrafficAnalyzer

def demo_rl_training_features():
    """Demonstrate enhanced RL training features"""
    print("🚀 Enhanced RL Training Demo")
    print("=" * 50)
    
    # Configuration for RL training
    max_steps = 1500
    n_cars = 1500  # Higher vehicle count for better challenges
    out_dir = "Sumo_env/Single intersection lhd"
    
    # Create generator
    generator = TrafficGenerator(max_steps, n_cars, out_dir)
    
    print("\n📊 RL Training Scenarios Available:")
    print("1. Uniform - Balanced traffic (baseline)")
    print("2. Tidal - Heavy East-West traffic (challenging)")
    print("3. Asymmetric - Heavy North+East traffic (challenging)")
    print("4. Congested - High congestion scenario (very challenging)")
    print("5. Random - Random scenario selection (RL training mode)")
    
    print("\n🎯 RL Training Features:")
    print("✅ Balanced scenario sampling")
    print("✅ Traffic variability (noise)")
    print("✅ Higher vehicle counts (1200-1500)")
    print("✅ Enhanced congestion scenarios")
    print("✅ Poor traffic light timing (creates bottlenecks)")
    print("✅ Sensitive congestion thresholds")
    
    # Demo different scenarios
    scenarios = ["uniform", "tidal", "asymmetric", "congested"]
    
    print(f"\n🧪 Running {len(scenarios)} episodes for RL training demo...")
    
    for i, scenario in enumerate(scenarios):
        print(f"\n--- Episode {i+1}: {scenario.upper()} ---")
        
        # Generate traffic
        route_file = generator.generate(seed=i, scenario=scenario, add_noise=True)
        print(f"✅ Generated {os.path.basename(route_file)}")
        
        # Create sumocfg with poor traffic light timing
        sumocfg_file = os.path.join(out_dir, f"{scenario}_rl_training.sumocfg")
        
        with open(sumocfg_file, "w") as cfg:
            cfg.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="cross_2lanes.net.xml"/>
        <route-files value="{os.path.basename(route_file)}"/>
        <additional-files value="essential_detectors.xml,essential_traffic_lights.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{max_steps}"/>
    </time>
    <processing>
        <lateral-resolution value="0.64"/>
    </processing>
    <report>
        <verbose value="true"/>
        <no-step-log value="true"/>
    </report>
    <!-- Use poor traffic light timing for RL challenges -->
    <traci_server>
        <remote-port value="8813"/>
    </traci_server>
</configuration>""")
        
        print(f"✅ Created {os.path.basename(sumocfg_file)} with poor traffic light timing")
        
        # Run simulation
        result = generator.run_simulation(sumocfg_file, use_gui=False)
        
        if result['status'] == 'success':
            metrics = result['metrics']
            print(f"📈 Episode Results:")
            print(f"   Total Vehicles: {metrics.get('total_vehicles', 'N/A')}")
            print(f"   Efficiency Score: {metrics.get('efficiency_score', 'N/A'):.1f}/100")
            print(f"   Congestion Level: {metrics.get('congestion_level', 'N/A')}")
            print(f"   Average Speed: {metrics.get('avg_speed', 'N/A'):.1f} m/s")
            print(f"   Average Waiting Time: {metrics.get('avg_waiting_time', 'N/A'):.1f} seconds")
        else:
            print(f"❌ Episode failed: {result.get('error', 'Unknown error')}")
    
    print(f"\n🎉 RL Training Demo Complete!")
    print(f"📁 Generated files in: {out_dir}")
    print(f"🔧 Ready for RL agent training with TraCI on port 8813")

def demo_rl_optimization_opportunities():
    """Show optimization opportunities for RL agents"""
    print("\n🎯 RL Optimization Opportunities:")
    print("=" * 40)
    
    print("\n1. 🚦 Traffic Light Control:")
    print("   - Current: Poor timing (10s green phases)")
    print("   - RL Goal: Optimize phase durations")
    print("   - Action Space: Extend/shorten green phases")
    
    print("\n2. 📊 Performance Metrics:")
    print("   - Waiting Time: Minimize average wait")
    print("   - Queue Length: Reduce queue buildup")
    print("   - Throughput: Maximize vehicles processed")
    print("   - Efficiency Score: Target 100/100")
    
    print("\n3. 🎲 Scenario Challenges:")
    print("   - Uniform: Balanced optimization")
    print("   - Tidal: Handle directional bias")
    print("   - Asymmetric: Complex traffic patterns")
    print("   - Congested: High-pressure optimization")
    
    print("\n4. 🔄 RL Training Process:")
    print("   - State: Detector readings + current phase")
    print("   - Action: Traffic light control decisions")
    print("   - Reward: Based on performance metrics")
    print("   - Episode: Complete simulation run")

if __name__ == "__main__":
    try:
        demo_rl_training_features()
        demo_rl_optimization_opportunities()
        
        print(f"\n🚀 Next Steps for RL Implementation:")
        print("1. Install TraCI Python library: pip install traci")
        print("2. Connect RL agent to SUMO via TraCI")
        print("3. Define state space from detector data")
        print("4. Implement action space for traffic light control")
        print("5. Design reward function based on performance metrics")
        print("6. Train RL agent on multiple scenarios")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure SUMO is installed and the environment is set up correctly.")
