#!/usr/bin/env python3
"""
Benchmark Analysis and Visualization Script

This script analyzes the benchmark results and creates visualizations
to compare performance across different traffic scenarios.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import argparse


class BenchmarkAnalyzer:
    """Analyze and visualize benchmark results"""
    
    def __init__(self, results_dir="benchmark_results"):
        self.results_dir = results_dir
        self.results_file = os.path.join(results_dir, "benchmark_results_detailed.json")
        self.summary_file = os.path.join(results_dir, "benchmark_summary.csv")
        self.routes_file = os.path.join(results_dir, "benchmark_routes.csv")
        
        # Set up plotting style with fallback
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')
        
        sns.set_palette("husl")
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        
    def load_results(self):
        """Load benchmark results from files"""
        if not os.path.exists(self.results_file):
            raise FileNotFoundError(f"Results file not found: {self.results_file}")
        
        with open(self.results_file, 'r') as f:
            self.results = json.load(f)
        
        # Check if summary file exists and has content
        if os.path.exists(self.summary_file) and os.path.getsize(self.summary_file) > 0:
            self.summary_df = pd.read_csv(self.summary_file)
        else:
            print("⚠️  No summary data available - all runs may have failed")
            self.summary_df = pd.DataFrame()
        
        # Check if routes file exists and has content
        if os.path.exists(self.routes_file) and os.path.getsize(self.routes_file) > 0:
            self.routes_df = pd.read_csv(self.routes_file)
        else:
            print("⚠️  No route data available - all runs may have failed")
            self.routes_df = pd.DataFrame()
        
        print(f"✅ Loaded results for {len(self.results)} scenarios")
        if not self.summary_df.empty:
            print(f"✅ Summary data: {len(self.summary_df)} rows")
        if not self.routes_df.empty:
            print(f"✅ Route data: {len(self.routes_df)} rows")
    
    def create_summary_plots(self):
        """Create summary comparison plots"""
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Traffic Light Performance Comparison (Fixed Duration)', fontsize=16, fontweight='bold')
        
        # 1. Average Travel Time Comparison
        ax1 = axes[0, 0]
        scenarios = self.summary_df['scenario']
        travel_times = self.summary_df['avg_travel_time_mean']
        travel_errors = self.summary_df['avg_travel_time_std']
        
        bars1 = ax1.bar(scenarios, travel_times, yerr=travel_errors, capsize=5, alpha=0.7)
        ax1.set_title('Average Travel Time by Scenario', fontweight='bold')
        ax1.set_ylabel('Travel Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value, error in zip(bars1, travel_times, travel_errors):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + error + 0.5,
                    f'{value:.1f}±{error:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Throughput Comparison
        ax2 = axes[0, 1]
        throughput = self.summary_df['total_throughput_mean']
        throughput_errors = self.summary_df['total_throughput_std']
        
        bars2 = ax2.bar(scenarios, throughput, yerr=throughput_errors, capsize=5, alpha=0.7, color='green')
        ax2.set_title('Total Throughput by Scenario', fontweight='bold')
        ax2.set_ylabel('Vehicles Completed')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, value, error in zip(bars2, throughput, throughput_errors):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + error + 10,
                    f'{value:.0f}±{error:.0f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Average Speed Comparison
        ax3 = axes[1, 0]
        speeds = self.summary_df['avg_speed_mean']
        speed_errors = self.summary_df['avg_speed_std']
        
        bars3 = ax3.bar(scenarios, speeds, yerr=speed_errors, capsize=5, alpha=0.7, color='orange')
        ax3.set_title('Average Speed by Scenario', fontweight='bold')
        ax3.set_ylabel('Speed (m/s)')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, value, error in zip(bars3, speeds, speed_errors):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + error + 0.1,
                    f'{value:.1f}±{error:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Total Waiting Time Comparison
        ax4 = axes[1, 1]
        waiting_times = self.summary_df['total_waiting_time_mean']
        waiting_errors = self.summary_df['total_waiting_time_std']
        
        bars4 = ax4.bar(scenarios, waiting_times, yerr=waiting_errors, capsize=5, alpha=0.7, color='red')
        ax4.set_title('Total Waiting Time by Scenario', fontweight='bold')
        ax4.set_ylabel('Waiting Time (seconds)')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, value, error in zip(bars4, waiting_times, waiting_errors):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + error + 5,
                    f'{value:.0f}±{error:.0f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'summary_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_route_analysis(self):
        """Create detailed route-by-route analysis"""
        
        # Create figure for route analysis
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('Route-Specific Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Travel Time by Route and Scenario
        ax1 = axes[0, 0]
        pivot_travel = self.routes_df.pivot(index='route_name', columns='scenario', values='avg_travel_time_mean')
        sns.heatmap(pivot_travel, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax1, cbar_kws={'label': 'Travel Time (s)'})
        ax1.set_title('Average Travel Time by Route and Scenario', fontweight='bold')
        ax1.set_xlabel('Scenario')
        ax1.set_ylabel('Route')
        
        # 2. Throughput by Route and Scenario
        ax2 = axes[0, 1]
        pivot_throughput = self.routes_df.pivot(index='route_name', columns='scenario', values='throughput_mean')
        sns.heatmap(pivot_throughput, annot=True, fmt='.0f', cmap='Greens', ax=ax2, cbar_kws={'label': 'Throughput'})
        ax2.set_title('Throughput by Route and Scenario', fontweight='bold')
        ax2.set_xlabel('Scenario')
        ax2.set_ylabel('Route')
        
        # 3. Average Speed by Route and Scenario
        ax3 = axes[1, 0]
        pivot_speed = self.routes_df.pivot(index='route_name', columns='scenario', values='avg_speed_mean')
        sns.heatmap(pivot_speed, annot=True, fmt='.1f', cmap='Blues', ax=ax3, cbar_kws={'label': 'Speed (m/s)'})
        ax3.set_title('Average Speed by Route and Scenario', fontweight='bold')
        ax3.set_xlabel('Scenario')
        ax3.set_ylabel('Route')
        
        # 4. Delay by Route and Scenario
        ax4 = axes[1, 1]
        pivot_delay = self.routes_df.pivot(index='route_name', columns='scenario', values='avg_delay_mean')
        sns.heatmap(pivot_delay, annot=True, fmt='.1f', cmap='Reds', ax=ax4, cbar_kws={'label': 'Delay (s)'})
        ax4.set_title('Average Delay by Route and Scenario', fontweight='bold')
        ax4.set_xlabel('Scenario')
        ax4.set_ylabel('Route')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'route_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_scenario_comparison(self):
        """Create detailed scenario comparison plots"""
        
        # Create figure for scenario comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed Scenario Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Travel Time Distribution
        ax1 = axes[0, 0]
        for scenario in self.summary_df['scenario']:
            scenario_data = self.routes_df[self.routes_df['scenario'] == scenario]
            ax1.scatter(scenario_data['route_name'], scenario_data['avg_travel_time_mean'], 
                       label=scenario, alpha=0.7, s=60)
        ax1.set_title('Travel Time Distribution by Route', fontweight='bold')
        ax1.set_ylabel('Travel Time (s)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Throughput vs Travel Time
        ax2 = axes[0, 1]
        for scenario in self.summary_df['scenario']:
            scenario_data = self.routes_df[self.routes_df['scenario'] == scenario]
            ax2.scatter(scenario_data['avg_travel_time_mean'], scenario_data['throughput_mean'], 
                       label=scenario, alpha=0.7, s=60)
        ax2.set_title('Throughput vs Travel Time', fontweight='bold')
        ax2.set_xlabel('Travel Time (s)')
        ax2.set_ylabel('Throughput')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Speed vs Delay
        ax3 = axes[1, 0]
        for scenario in self.summary_df['scenario']:
            scenario_data = self.routes_df[self.routes_df['scenario'] == scenario]
            ax3.scatter(scenario_data['avg_speed_mean'], scenario_data['avg_delay_mean'], 
                       label=scenario, alpha=0.7, s=60)
        ax3.set_title('Speed vs Delay', fontweight='bold')
        ax3.set_xlabel('Average Speed (m/s)')
        ax3.set_ylabel('Average Delay (s)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Efficiency Score (Throughput / Travel Time)
        ax4 = axes[1, 1]
        for scenario in self.summary_df['scenario']:
            scenario_data = self.routes_df[self.routes_df['scenario'] == scenario]
            efficiency = scenario_data['throughput_mean'] / scenario_data['avg_travel_time_mean']
            ax4.scatter(scenario_data['route_name'], efficiency, label=scenario, alpha=0.7, s=60)
        ax4.set_title('Efficiency Score (Throughput/Travel Time)', fontweight='bold')
        ax4.set_ylabel('Efficiency Score')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'scenario_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_insights_report(self):
        """Generate insights and recommendations"""
        
        report_file = os.path.join(self.results_dir, 'INSIGHTS_REPORT.md')
        
        with open(report_file, 'w') as f:
            f.write("# Benchmark Insights and Recommendations\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Find best and worst performing scenarios
            best_travel_time = self.summary_df.loc[self.summary_df['avg_travel_time_mean'].idxmin()]
            worst_travel_time = self.summary_df.loc[self.summary_df['avg_travel_time_mean'].idxmax()]
            
            best_throughput = self.summary_df.loc[self.summary_df['total_throughput_mean'].idxmax()]
            worst_throughput = self.summary_df.loc[self.summary_df['total_throughput_mean'].idxmin()]
            
            f.write("## Key Findings\n\n")
            f.write(f"### Travel Time Performance\n")
            f.write(f"- **Best:** {best_travel_time['scenario']} ({best_travel_time['avg_travel_time_mean']:.1f}s)\n")
            f.write(f"- **Worst:** {worst_travel_time['scenario']} ({worst_travel_time['avg_travel_time_mean']:.1f}s)\n")
            f.write(f"- **Improvement Potential:** {worst_travel_time['avg_travel_time_mean'] - best_travel_time['avg_travel_time_mean']:.1f}s\n\n")
            
            f.write(f"### Throughput Performance\n")
            f.write(f"- **Best:** {best_throughput['scenario']} ({best_throughput['total_throughput_mean']:.0f} vehicles)\n")
            f.write(f"- **Worst:** {worst_throughput['scenario']} ({worst_throughput['total_throughput_mean']:.0f} vehicles)\n")
            f.write(f"- **Capacity Difference:** {best_throughput['total_throughput_mean'] - worst_throughput['total_throughput_mean']:.0f} vehicles\n\n")
            
            # Route-specific insights
            f.write("## Route-Specific Insights\n\n")
            
            # Find most problematic routes
            route_avg = self.routes_df.groupby('route_name').agg({
                'avg_travel_time_mean': 'mean',
                'avg_delay_mean': 'mean',
                'throughput_mean': 'mean'
            }).round(2)
            
            worst_routes = route_avg.nlargest(3, 'avg_travel_time_mean')
            best_routes = route_avg.nsmallest(3, 'avg_travel_time_mean')
            
            f.write("### Most Problematic Routes (Highest Travel Time)\n")
            for route, data in worst_routes.iterrows():
                f.write(f"- **{route}:** {data['avg_travel_time_mean']:.1f}s travel time, {data['avg_delay_mean']:.1f}s delay\n")
            
            f.write("\n### Best Performing Routes (Lowest Travel Time)\n")
            for route, data in best_routes.iterrows():
                f.write(f"- **{route}:** {data['avg_travel_time_mean']:.1f}s travel time, {data['avg_delay_mean']:.1f}s delay\n")
            
            # Recommendations
            f.write("\n## Recommendations for RL Model Training\n\n")
            f.write("### Priority Areas for Improvement\n")
            f.write("1. **Focus on high-delay routes** - These offer the most improvement potential\n")
            f.write("2. **Optimize for congested scenarios** - These show the worst performance\n")
            f.write("3. **Balance throughput vs travel time** - Some scenarios trade one for the other\n\n")
            
            f.write("### Baseline Targets for RL Model\n")
            for _, row in self.summary_df.iterrows():
                f.write(f"- **{row['scenario']}:** Target < {row['avg_travel_time_mean']:.1f}s travel time, > {row['total_throughput_mean']:.0f} throughput\n")
            
            f.write("\n### Success Metrics\n")
            f.write("- **Travel Time Reduction:** > 10% improvement over baseline\n")
            f.write("- **Throughput Increase:** > 5% improvement over baseline\n")
            f.write("- **Delay Reduction:** > 15% improvement over baseline\n")
            f.write("- **Consistency:** Lower variance across multiple runs\n")
    
    def run_analysis(self):
        """Run complete analysis"""
        print("📊 BENCHMARK ANALYSIS")
        print("=" * 40)
        
        # Load results
        self.load_results()
        
        # Check if we have any data to analyze
        if self.summary_df.empty and self.routes_df.empty:
            print("❌ No data available for analysis - all benchmark runs failed")
            print("   Check benchmark.log for details about failures")
            return
        
        # Create visualizations only if we have data
        if not self.summary_df.empty:
            print("📈 Creating summary comparison plots...")
            self.create_summary_plots()
            
            print("📊 Creating scenario comparison...")
            self.create_scenario_comparison()
        
        if not self.routes_df.empty:
            print("🗺️  Creating route analysis...")
            self.create_route_analysis()
        
        print("💡 Generating insights report...")
        self.generate_insights_report()
        
        print("✅ Analysis complete!")
        print(f"📁 Results saved to: {self.results_dir}/")
        if not self.summary_df.empty:
            print("   - summary_comparison.png")
            print("   - scenario_comparison.png")
        if not self.routes_df.empty:
            print("   - route_analysis.png")
        print("   - INSIGHTS_REPORT.md")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Analyze traffic light benchmark results')
    parser.add_argument('--results-dir', default='benchmark_results', 
                       help='Directory containing benchmark results')
    
    args = parser.parse_args()
    
    analyzer = BenchmarkAnalyzer(args.results_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
