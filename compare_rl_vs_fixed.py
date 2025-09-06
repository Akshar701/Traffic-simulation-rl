#!/usr/bin/env python3
"""
RL vs Fixed Traffic Light Performance Comparison

This script compares the performance of RL-managed traffic lights vs fixed-duration
traffic lights to demonstrate the 10% improvement in commute time.

Usage:
    python compare_rl_vs_fixed.py --fixed-results benchmark_results/ --rl-results rl_results/
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class RLvsFixedComparator:
    """Compare RL vs Fixed traffic light performance"""
    
    def __init__(self, fixed_results_dir, rl_results_dir, output_dir="comparison_results"):
        self.fixed_results_dir = fixed_results_dir
        self.rl_results_dir = rl_results_dir
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load results
        self.fixed_results = self._load_results(fixed_results_dir)
        self.rl_results = self._load_results(rl_results_dir)
    
    def _load_results(self, results_dir):
        """Load benchmark results from directory"""
        try:
            # Load summary CSV
            summary_file = os.path.join(results_dir, "benchmark_summary.csv")
            if os.path.exists(summary_file):
                summary_df = pd.read_csv(summary_file)
            else:
                print(f"Warning: {summary_file} not found")
                return None
            
            # Load detailed JSON
            detailed_file = os.path.join(results_dir, "benchmark_results_detailed.json")
            if os.path.exists(detailed_file):
                with open(detailed_file, 'r') as f:
                    detailed_data = json.load(f)
            else:
                print(f"Warning: {detailed_file} not found")
                detailed_data = None
            
            return {
                'summary': summary_df,
                'detailed': detailed_data
            }
        except Exception as e:
            print(f"Error loading results from {results_dir}: {e}")
            return None
    
    def calculate_improvement(self):
        """Calculate the improvement percentage for each metric"""
        if not self.fixed_results or not self.rl_results:
            print("Error: Missing results data")
            return None
        
        fixed_df = self.fixed_results['summary']
        rl_df = self.rl_results['summary']
        
        # Merge on scenario
        comparison = pd.merge(
            fixed_df, rl_df, 
            on='scenario', 
            suffixes=('_fixed', '_rl')
        )
        
        # Calculate improvement percentages
        improvements = {}
        
        # Primary metric: Commute time improvement
        if 'avg_commute_time_mean_fixed' in comparison.columns and 'avg_commute_time_mean_rl' in comparison.columns:
            comparison['commute_time_improvement_pct'] = (
                (comparison['avg_commute_time_mean_fixed'] - comparison['avg_commute_time_mean_rl']) / 
                comparison['avg_commute_time_mean_fixed'] * 100
            )
            improvements['commute_time'] = comparison['commute_time_improvement_pct'].mean()
        
        # Secondary metrics
        metrics = ['avg_travel_time', 'total_waiting_time', 'avg_speed', 'total_throughput']
        for metric in metrics:
            fixed_col = f'{metric}_mean_fixed'
            rl_col = f'{metric}_mean_rl'
            if fixed_col in comparison.columns and rl_col in comparison.columns:
                if metric == 'avg_speed':  # Higher speed is better
                    comparison[f'{metric}_improvement_pct'] = (
                        (comparison[rl_col] - comparison[fixed_col]) / 
                        comparison[fixed_col] * 100
                    )
                else:  # Lower values are better (except throughput)
                    if metric == 'total_throughput':
                        comparison[f'{metric}_improvement_pct'] = (
                            (comparison[rl_col] - comparison[fixed_col]) / 
                            comparison[fixed_col] * 100
                        )
                    else:
                        comparison[f'{metric}_improvement_pct'] = (
                            (comparison[fixed_col] - comparison[rl_col]) / 
                            comparison[fixed_col] * 100
                        )
                
                improvements[metric] = comparison[f'{metric}_improvement_pct'].mean()
        
        return comparison, improvements
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        comparison, improvements = self.calculate_improvement()
        
        if comparison is None:
            return
        
        # Create comparison report
        report_file = os.path.join(self.output_dir, "RL_vs_Fixed_Comparison_Report.md")
        
        with open(report_file, 'w') as f:
            f.write("# RL vs Fixed Traffic Light Performance Comparison\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 🎯 Key Performance Improvements\n\n")
            
            # Primary metric
            if 'commute_time' in improvements:
                f.write(f"### 🚗 **Commute Time Reduction: {improvements['commute_time']:.1f}%**\n\n")
                if improvements['commute_time'] >= 10.0:
                    f.write("✅ **TARGET ACHIEVED**: 10% improvement goal met!\n\n")
                else:
                    f.write(f"⚠️ **TARGET**: Need {10.0 - improvements['commute_time']:.1f}% more improvement to reach 10% goal\n\n")
            
            # Secondary metrics
            f.write("### 📊 Secondary Performance Metrics\n\n")
            f.write("| Metric | Improvement | Status |\n")
            f.write("|--------|-------------|--------|\n")
            
            for metric, improvement in improvements.items():
                if metric != 'commute_time':
                    status = "✅ Good" if improvement > 0 else "❌ Needs work"
                    f.write(f"| {metric.replace('_', ' ').title()} | {improvement:.1f}% | {status} |\n")
            
            f.write("\n## 📈 Detailed Scenario Comparison\n\n")
            f.write("| Scenario | Fixed Commute Time | RL Commute Time | Improvement |\n")
            f.write("|----------|-------------------|-----------------|-------------|\n")
            
            for _, row in comparison.iterrows():
                if 'avg_commute_time_mean_fixed' in row and 'avg_commute_time_mean_rl' in row:
                    f.write(f"| {row['scenario']} | {row['avg_commute_time_mean_fixed']:.1f}s | {row['avg_commute_time_mean_rl']:.1f}s | {row.get('commute_time_improvement_pct', 0):.1f}% |\n")
            
            f.write("\n## 🎯 Conclusion\n\n")
            if 'commute_time' in improvements and improvements['commute_time'] >= 10.0:
                f.write("**SUCCESS**: The RL-based traffic management system successfully achieved the target of 10% reduction in average commute time, demonstrating significant improvement over fixed-duration traffic lights.\n\n")
            else:
                f.write("**PROGRESS**: The RL system shows improvement over fixed traffic lights, but additional optimization may be needed to reach the 10% target.\n\n")
        
        print(f"✅ Comparison report saved to: {report_file}")
        
        # Save comparison data
        comparison.to_csv(os.path.join(self.output_dir, "detailed_comparison.csv"), index=False)
        
        return improvements
    
    def create_visualization(self):
        """Create visualization comparing RL vs Fixed performance"""
        comparison, improvements = self.calculate_improvement()
        
        if comparison is None:
            return
        
        # Set up the plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RL vs Fixed Traffic Light Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Commute Time Comparison (Primary Metric)
        ax1 = axes[0, 0]
        scenarios = comparison['scenario']
        fixed_commute = comparison['avg_commute_time_mean_fixed']
        rl_commute = comparison['avg_commute_time_mean_rl']
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        ax1.bar(x - width/2, fixed_commute, width, label='Fixed Duration', alpha=0.8, color='red')
        ax1.bar(x + width/2, rl_commute, width, label='RL Managed', alpha=0.8, color='green')
        
        ax1.set_xlabel('Traffic Scenario')
        ax1.set_ylabel('Average Commute Time (seconds)')
        ax1.set_title('Commute Time Comparison (Primary Metric)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(scenarios)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add improvement percentages on bars
        for i, (fixed, rl) in enumerate(zip(fixed_commute, rl_commute)):
            improvement = (fixed - rl) / fixed * 100
            ax1.text(i, max(fixed, rl) + 1, f'{improvement:.1f}%', ha='center', fontweight='bold')
        
        # 2. Throughput Comparison
        ax2 = axes[0, 1]
        fixed_throughput = comparison['total_throughput_mean_fixed']
        rl_throughput = comparison['total_throughput_mean_rl']
        
        ax2.bar(x - width/2, fixed_throughput, width, label='Fixed Duration', alpha=0.8, color='red')
        ax2.bar(x + width/2, rl_throughput, width, label='RL Managed', alpha=0.8, color='green')
        
        ax2.set_xlabel('Traffic Scenario')
        ax2.set_ylabel('Total Throughput (vehicles)')
        ax2.set_title('Throughput Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(scenarios)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Overall Improvement Summary
        ax3 = axes[1, 0]
        metrics = list(improvements.keys())
        values = list(improvements.values())
        colors = ['green' if v > 0 else 'red' for v in values]
        
        bars = ax3.bar(metrics, values, color=colors, alpha=0.7)
        ax3.set_ylabel('Improvement (%)')
        ax3.set_title('Overall Performance Improvements')
        ax3.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -0.5),
                    f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
        
        # 4. Target Achievement
        ax4 = axes[1, 1]
        target_achieved = improvements.get('commute_time', 0) >= 10.0
        target_value = improvements.get('commute_time', 0)
        
        ax4.bar(['Target (10%)', 'Achieved'], [10.0, target_value], 
                color=['gray', 'green' if target_achieved else 'orange'], alpha=0.7)
        ax4.set_ylabel('Commute Time Reduction (%)')
        ax4.set_title('10% Improvement Target Achievement')
        ax4.grid(True, alpha=0.3)
        
        # Add target line
        ax4.axhline(y=10.0, color='red', linestyle='--', alpha=0.7, label='Target')
        ax4.legend()
        
        # Add value labels
        ax4.text(0, 10.5, '10.0%', ha='center', fontweight='bold')
        ax4.text(1, target_value + 0.5, f'{target_value:.1f}%', ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.output_dir, "rl_vs_fixed_comparison.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualization saved to: {plot_file}")
    
    def run_comparison(self):
        """Run complete comparison analysis"""
        print("🔍 Running RL vs Fixed Traffic Light Comparison...")
        
        if not self.fixed_results:
            print("❌ Error: Fixed results not found")
            return None
        
        if not self.rl_results:
            print("❌ Error: RL results not found")
            return None
        
        # Generate comparison
        improvements = self.generate_comparison_report()
        self.create_visualization()
        
        # Print summary
        print("\n🎯 COMPARISON SUMMARY")
        print("=" * 50)
        
        if improvements and 'commute_time' in improvements:
            print(f"🚗 Commute Time Reduction: {improvements['commute_time']:.1f}%")
            if improvements['commute_time'] >= 10.0:
                print("✅ TARGET ACHIEVED: 10% improvement goal met!")
            else:
                print(f"⚠️  Need {10.0 - improvements['commute_time']:.1f}% more to reach 10% target")
        
        print(f"\n📁 Results saved to: {self.output_dir}/")
        
        return improvements


def main():
    parser = argparse.ArgumentParser(description='Compare RL vs Fixed traffic light performance')
    parser.add_argument('--fixed-results', required=True, help='Directory containing fixed traffic light benchmark results')
    parser.add_argument('--rl-results', required=True, help='Directory containing RL traffic light benchmark results')
    parser.add_argument('--output-dir', default='comparison_results', help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    # Validate input directories
    if not os.path.exists(args.fixed_results):
        print(f"❌ Error: Fixed results directory not found: {args.fixed_results}")
        return
    
    if not os.path.exists(args.rl_results):
        print(f"❌ Error: RL results directory not found: {args.rl_results}")
        return
    
    # Run comparison
    comparator = RLvsFixedComparator(args.fixed_results, args.rl_results, args.output_dir)
    improvements = comparator.run_comparison()
    
    if improvements:
        print("\n🎉 Comparison completed successfully!")
    else:
        print("\n❌ Comparison failed. Check input directories and data.")


if __name__ == "__main__":
    main()
