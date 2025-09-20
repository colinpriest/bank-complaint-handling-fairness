#!/usr/bin/env python3
"""
Migration Script: Factorial to Stochastic Experiment Creation

This script helps migrate from the old factorial experiment creation approach
to the new stochastic sampling approach. It provides options to:

1. Analyze existing experiments
2. Create new experiments using stochastic approach
3. Compare results between approaches
4. Clean up or archive old experiments if needed

Usage:
    python migrate_to_stochastic_experiments.py [options]
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, List

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stochastic_experiment_creator import StochasticExperimentCreator
from experiment_creation_comparison import ExperimentCreationComparison

class ExperimentMigrationManager:
    """Manages migration from factorial to stochastic experiment creation"""
    
    def __init__(self):
        self.creator = StochasticExperimentCreator()
        self.comparison = ExperimentCreationComparison()
    
    def analyze_existing_experiments(self) -> Dict[str, Any]:
        """Analyze existing experiments in the database"""
        print("Analyzing existing experiments...")
        print("=" * 50)
        
        try:
            stats = self.creator.get_experiment_statistics()
            
            print("Current experiment counts by type:")
            for exp_type, methods in stats.items():
                if isinstance(methods, dict):
                    total = sum(methods.values())
                    print(f"  {exp_type}: {total:,} total")
                    for method, count in methods.items():
                        print(f"    {method}: {count:,}")
                else:
                    print(f"  {exp_type}: {methods:,}")
            
            print(f"\nTotal experiments: {stats.get('total', 0):,}")
            
            # Analyze experiment distribution
            total_experiments = stats.get('total', 0)
            if total_experiments > 0:
                baseline_pct = (stats.get('baseline', {}).get('zero-shot', 0) + 
                              stats.get('baseline', {}).get('n-shot', 0)) / total_experiments * 100
                persona_pct = (stats.get('persona_injected', {}).get('zero-shot', 0) + 
                             stats.get('persona_injected', {}).get('n-shot', 0)) / total_experiments * 100
                mitigation_pct = (stats.get('bias_mitigation', {}).get('zero-shot', 0) + 
                                stats.get('bias_mitigation', {}).get('n-shot', 0)) / total_experiments * 100
                
                print(f"\nExperiment distribution:")
                print(f"  Baseline: {baseline_pct:.1f}%")
                print(f"  Persona-injected: {persona_pct:.1f}%")
                print(f"  Bias-mitigation: {mitigation_pct:.1f}%")
            
            return stats
            
        except Exception as e:
            print(f"Error analyzing experiments: {e}")
            return {}
    
    def create_stochastic_experiments(self, ground_truth_limit: int = None, 
                                    dry_run: bool = False) -> Dict[str, Any]:
        """Create new experiments using stochastic approach"""
        
        if dry_run:
            print("DRY RUN: Would create experiments using stochastic approach")
            print("=" * 60)
            
            # Calculate what would be created
            comparison = self.comparison.compare_approaches(
                ground_truth_limit or 100, 
                self.creator.personas_per_baseline
            )
            
            stochastic = comparison['stochastic_approach']
            print(f"Would create:")
            print(f"  Baseline experiments: {stochastic['baseline_experiments']:,}")
            print(f"  Persona-injected experiments: {stochastic['persona_injected_experiments']:,}")
            print(f"  Bias-mitigation experiments: {stochastic['bias_mitigation_experiments']:,}")
            print(f"  Total: {stochastic['total_experiments']:,}")
            
            return {'dry_run': True, 'would_create': stochastic}
        
        else:
            print("Creating experiments using stochastic approach...")
            print("=" * 60)
            
            results = self.creator.create_experiments_stochastic(ground_truth_limit)
            return results
    
    def compare_approaches(self, ground_truth_count: int = 100):
        """Compare factorial vs stochastic approaches"""
        print("Comparing factorial vs stochastic approaches...")
        print("=" * 60)
        
        comparison = self.comparison.print_comparison(ground_truth_count)
        return comparison
    
    def get_migration_recommendations(self, existing_stats: Dict[str, Any]) -> List[str]:
        """Get recommendations for migration based on existing experiments"""
        recommendations = []
        
        total_existing = existing_stats.get('total', 0)
        
        if total_existing == 0:
            recommendations.append("No existing experiments found. You can start fresh with stochastic approach.")
            recommendations.append("Consider starting with a small batch (e.g., 10-20 ground truth cases) to test the approach.")
        
        elif total_existing < 1000:
            recommendations.append("Small number of existing experiments. You can add stochastic experiments alongside existing ones.")
            recommendations.append("Consider running stochastic approach on remaining ground truth cases.")
        
        elif total_existing < 10000:
            recommendations.append("Moderate number of existing experiments. Consider analyzing coverage gaps.")
            recommendations.append("Stochastic approach can fill in missing combinations efficiently.")
        
        else:
            recommendations.append("Large number of existing experiments. Consider analyzing if current coverage is sufficient.")
            recommendations.append("Stochastic approach may be more efficient for future experiments.")
        
        # Check for specific patterns
        baseline_count = (existing_stats.get('baseline', {}).get('zero-shot', 0) + 
                         existing_stats.get('baseline', {}).get('n-shot', 0))
        persona_count = (existing_stats.get('persona_injected', {}).get('zero-shot', 0) + 
                       existing_stats.get('persona_injected', {}).get('n-shot', 0))
        mitigation_count = (existing_stats.get('bias_mitigation', {}).get('zero-shot', 0) + 
                          existing_stats.get('bias_mitigation', {}).get('n-shot', 0))
        
        if baseline_count == 0:
            recommendations.append("No baseline experiments found. These are essential for comparison.")
        
        if persona_count == 0:
            recommendations.append("No persona-injected experiments found. These are needed for bias analysis.")
        
        if mitigation_count == 0:
            recommendations.append("No bias-mitigation experiments found. These are needed for mitigation analysis.")
        
        return recommendations
    
    def print_migration_plan(self, ground_truth_limit: int = None):
        """Print a comprehensive migration plan"""
        print("EXPERIMENT MIGRATION PLAN")
        print("=" * 60)
        
        # Analyze existing experiments
        existing_stats = self.analyze_existing_experiments()
        
        # Get recommendations
        recommendations = self.get_migration_recommendations(existing_stats)
        
        print(f"\nMIGRATION RECOMMENDATIONS:")
        print("-" * 40)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        # Show comparison
        print(f"\nAPPROACH COMPARISON:")
        print("-" * 40)
        self.compare_approaches(ground_truth_limit or 100)
        
        # Show what would be created
        print(f"\nSTOCHASTIC EXPERIMENT CREATION:")
        print("-" * 40)
        self.create_stochastic_experiments(ground_truth_limit, dry_run=True)
        
        print(f"\nNEXT STEPS:")
        print("-" * 40)
        print("1. Review the comparison above")
        print("2. Decide on ground truth limit for initial batch")
        print("3. Run stochastic experiment creation")
        print("4. Analyze results and iterate")
        
        return {
            'existing_stats': existing_stats,
            'recommendations': recommendations
        }

def main():
    """Main function for migration management"""
    parser = argparse.ArgumentParser(description='Migrate to stochastic experiment creation')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze existing experiments')
    parser.add_argument('--compare', action='store_true',
                       help='Compare factorial vs stochastic approaches')
    parser.add_argument('--create', action='store_true',
                       help='Create new experiments using stochastic approach')
    parser.add_argument('--plan', action='store_true',
                       help='Show comprehensive migration plan')
    parser.add_argument('--ground-truth-limit', type=int, default=None,
                       help='Limit number of ground truth cases to process')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be created without actually creating')
    
    args = parser.parse_args()
    
    # If no specific action, show the plan
    if not any([args.analyze, args.compare, args.create, args.plan]):
        args.plan = True
    
    migration_manager = ExperimentMigrationManager()
    
    try:
        if args.analyze:
            migration_manager.analyze_existing_experiments()
        
        if args.compare:
            migration_manager.compare_approaches(args.ground_truth_limit or 100)
        
        if args.create:
            migration_manager.create_stochastic_experiments(
                args.ground_truth_limit, 
                args.dry_run
            )
        
        if args.plan:
            migration_manager.print_migration_plan(args.ground_truth_limit)
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        migration_manager.creator.close()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
