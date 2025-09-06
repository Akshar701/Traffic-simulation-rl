#!/usr/bin/env python3
"""
Quick script to run traffic light benchmarks

This script provides an easy way to run the comprehensive benchmarking
system with configurable parameters.
"""

import argparse
import sys
import os
from benchmark_fixed_lights import TrafficLightBenchmarker
from analyze_benchmarks import BenchmarkAnalyzer


def main():
    """Main function to run benchmarks"""
    
    parser = argparse.ArgumentParser(description='Run traffic light benchmarks')
    parser.add_argument('--output-dir', default='benchmark_results',
                       help='Output directory for results (default: benchmark_results)')
    parser.add_argument('--num-runs', type=int, default=5,
                       help='Number of runs per scenario (default: 5)')
    parser.add_argument('--duration', type=int, default=3600,
                       help='Simulation duration in seconds (default: 3600 = 1 hour)')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Skip analysis and visualization (just run benchmarks)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with reduced parameters')
    
    args = parser.parse_args()
    
    # Adjust parameters for quick test
    if args.quick_test:
        args.num_runs = 2
        args.duration = 600  # 10 minutes
        print("🚀 QUICK TEST MODE: 2 runs, 10 minutes each")
    
    print("🚦 TRAFFIC LIGHT BENCHMARKING")
    print("=" * 50)
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🔄 Runs per scenario: {args.num_runs}")
    print(f"⏱️  Simulation duration: {args.duration}s ({args.duration/60:.1f} minutes)")
    print(f"📊 Analysis: {'Skipped' if args.skip_analysis else 'Enabled'}")
    print()
    
    # Create benchmarker
    benchmarker = TrafficLightBenchmarker(
        output_dir=args.output_dir,
        num_runs=args.num_runs,
        simulation_duration=args.duration
    )
    
    # Run benchmarks
    print("🏃 Starting benchmark runs...")
    benchmarker.run_benchmark()
    
    # Run analysis if requested
    if not args.skip_analysis:
        print("\n📊 Starting analysis...")
        analyzer = BenchmarkAnalyzer(args.output_dir)
        analyzer.run_analysis()
    
    print(f"\n🎉 BENCHMARKING COMPLETE!")
    print(f"📁 Results saved to: {args.output_dir}/")
    
    if not args.skip_analysis:
        print("📈 Generated files:")
        print("   - benchmark_results_detailed.json")
        print("   - benchmark_summary.csv")
        print("   - benchmark_routes.csv")
        print("   - BENCHMARK_REPORT.md")
        print("   - INSIGHTS_REPORT.md")
        print("   - summary_comparison.png")
        print("   - route_analysis.png")
        print("   - scenario_comparison.png")


if __name__ == "__main__":
    main()
