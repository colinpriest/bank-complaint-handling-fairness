#!/usr/bin/env python3
"""
Experiment Creation Approach Comparison

This script compares the old factorial combination approach with the new 
stochastic sampling approach for experiment creation.

The comparison shows:
1. How many experiments each approach would create
2. The computational efficiency differences
3. The coverage differences
"""

import os
import sys
from typing import Dict, Any

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stochastic_experiment_creator import StochasticExperimentCreator

class ExperimentCreationComparison:
    """Compare factorial vs stochastic experiment creation approaches"""
    
    def __init__(self):
        self.creator = StochasticExperimentCreator()
    
    def calculate_factorial_approach(self, ground_truth_count: int = 100) -> Dict[str, int]:
        """
        Calculate how many experiments the old factorial approach would create
        
        Old approach: All combinations of:
        - Ground truth cases (100)
        - Decision methods (2: zero-shot, n-shot)
        - Personas (24: 4 ethnicities × 2 genders × 3 geographies)
        - Mitigation strategies (7: excluding DPP strategies)
        
        This creates: 100 × 2 × 24 × 7 = 33,600 experiments
        Plus baseline experiments: 100 × 2 = 200
        Total: 33,800 experiments
        """
        
        # Get actual counts from database
        personas = self.creator.get_all_personas()
        mitigation_strategies = self.creator.get_mitigation_strategies()
        
        persona_count = len(personas)
        mitigation_count = len(mitigation_strategies)
        decision_methods = 2  # zero-shot, n-shot
        
        # Baseline experiments (no persona, no mitigation)
        baseline_experiments = ground_truth_count * decision_methods
        
        # Persona-injected experiments (persona, no mitigation)
        persona_injected_experiments = ground_truth_count * decision_methods * persona_count
        
        # Bias-mitigation experiments (persona + mitigation)
        bias_mitigation_experiments = ground_truth_count * decision_methods * persona_count * mitigation_count
        
        total_factorial = baseline_experiments + persona_injected_experiments + bias_mitigation_experiments
        
        return {
            'ground_truth_cases': ground_truth_count,
            'decision_methods': decision_methods,
            'personas': persona_count,
            'mitigation_strategies': mitigation_count,
            'baseline_experiments': baseline_experiments,
            'persona_injected_experiments': persona_injected_experiments,
            'bias_mitigation_experiments': bias_mitigation_experiments,
            'total_experiments': total_factorial
        }
    
    def calculate_stochastic_approach(self, ground_truth_count: int = 100, 
                                    personas_per_baseline: int = 10) -> Dict[str, int]:
        """
        Calculate how many experiments the new stochastic approach would create
        
        New approach:
        1. Baseline experiments: ground_truth_count × 2 (zero-shot, n-shot)
        2. Persona-injected experiments: baseline_count × personas_per_baseline
        3. Bias-mitigation experiments: persona_injected_count × mitigation_strategies
        
        This creates a much smaller, more manageable set of experiments
        """
        
        # Get actual counts from database
        mitigation_strategies = self.creator.get_mitigation_strategies()
        mitigation_count = len(mitigation_strategies)
        decision_methods = 2  # zero-shot, n-shot
        
        # Step 1: Baseline experiments
        baseline_experiments = ground_truth_count * decision_methods
        
        # Step 2: Persona-injected experiments (stochastic sampling)
        persona_injected_experiments = baseline_experiments * personas_per_baseline
        
        # Step 3: Bias-mitigation experiments
        bias_mitigation_experiments = persona_injected_experiments * mitigation_count
        
        total_stochastic = baseline_experiments + persona_injected_experiments + bias_mitigation_experiments
        
        return {
            'ground_truth_cases': ground_truth_count,
            'decision_methods': decision_methods,
            'personas_per_baseline': personas_per_baseline,
            'mitigation_strategies': mitigation_count,
            'baseline_experiments': baseline_experiments,
            'persona_injected_experiments': persona_injected_experiments,
            'bias_mitigation_experiments': bias_mitigation_experiments,
            'total_experiments': total_stochastic
        }
    
    def compare_approaches(self, ground_truth_count: int = 100, 
                          personas_per_baseline: int = 10) -> Dict[str, Any]:
        """Compare factorial vs stochastic approaches"""
        
        factorial = self.calculate_factorial_approach(ground_truth_count)
        stochastic = self.calculate_stochastic_approach(ground_truth_count, personas_per_baseline)
        
        # Calculate efficiency metrics
        reduction_factor = factorial['total_experiments'] / stochastic['total_experiments']
        reduction_percentage = ((factorial['total_experiments'] - stochastic['total_experiments']) / 
                              factorial['total_experiments']) * 100
        
        # Calculate coverage metrics
        total_personas = factorial['personas']
        coverage_percentage = (personas_per_baseline / total_personas) * 100
        
        return {
            'factorial_approach': factorial,
            'stochastic_approach': stochastic,
            'efficiency': {
                'reduction_factor': reduction_factor,
                'reduction_percentage': reduction_percentage,
                'experiments_saved': factorial['total_experiments'] - stochastic['total_experiments']
            },
            'coverage': {
                'total_personas': total_personas,
                'personas_per_baseline': personas_per_baseline,
                'coverage_percentage': coverage_percentage
            }
        }
    
    def print_comparison(self, ground_truth_count: int = 100, 
                        personas_per_baseline: int = 10):
        """Print a detailed comparison of the two approaches"""
        
        comparison = self.compare_approaches(ground_truth_count, personas_per_baseline)
        
        print("EXPERIMENT CREATION APPROACH COMPARISON")
        print("=" * 60)
        print(f"Configuration: {ground_truth_count} ground truth cases, {personas_per_baseline} personas per baseline")
        print()
        
        # Factorial approach
        print("FACTORIAL APPROACH (Old Method)")
        print("-" * 40)
        factorial = comparison['factorial_approach']
        print(f"Ground truth cases: {factorial['ground_truth_cases']:,}")
        print(f"Decision methods: {factorial['decision_methods']}")
        print(f"Personas (all combinations): {factorial['personas']:,}")
        print(f"Mitigation strategies: {factorial['mitigation_strategies']}")
        print()
        print(f"Baseline experiments: {factorial['baseline_experiments']:,}")
        print(f"Persona-injected experiments: {factorial['persona_injected_experiments']:,}")
        print(f"Bias-mitigation experiments: {factorial['bias_mitigation_experiments']:,}")
        print(f"TOTAL EXPERIMENTS: {factorial['total_experiments']:,}")
        print()
        
        # Stochastic approach
        print("STOCHASTIC APPROACH (New Method)")
        print("-" * 40)
        stochastic = comparison['stochastic_approach']
        print(f"Ground truth cases: {stochastic['ground_truth_cases']:,}")
        print(f"Decision methods: {stochastic['decision_methods']}")
        print(f"Personas per baseline (sampled): {stochastic['personas_per_baseline']}")
        print(f"Mitigation strategies: {stochastic['mitigation_strategies']}")
        print()
        print(f"Baseline experiments: {stochastic['baseline_experiments']:,}")
        print(f"Persona-injected experiments: {stochastic['persona_injected_experiments']:,}")
        print(f"Bias-mitigation experiments: {stochastic['bias_mitigation_experiments']:,}")
        print(f"TOTAL EXPERIMENTS: {stochastic['total_experiments']:,}")
        print()
        
        # Efficiency comparison
        print("EFFICIENCY COMPARISON")
        print("-" * 40)
        efficiency = comparison['efficiency']
        print(f"Experiments saved: {efficiency['experiments_saved']:,}")
        print(f"Reduction factor: {efficiency['reduction_factor']:.1f}x")
        print(f"Reduction percentage: {efficiency['reduction_percentage']:.1f}%")
        print()
        
        # Coverage comparison
        print("COVERAGE ANALYSIS")
        print("-" * 40)
        coverage = comparison['coverage']
        print(f"Total personas available: {coverage['total_personas']:,}")
        print(f"Personas sampled per baseline: {coverage['personas_per_baseline']}")
        print(f"Coverage percentage: {coverage['coverage_percentage']:.1f}%")
        print()
        
        # Benefits
        print("BENEFITS OF STOCHASTIC APPROACH")
        print("-" * 40)
        print("✓ Dramatically reduced computational requirements")
        print("✓ Faster experiment execution and analysis")
        print("✓ More manageable dataset size")
        print("✓ Still provides good statistical coverage")
        print("✓ Reproducible results (seeded random sampling)")
        print("✓ Easier to scale and extend")
        print()
        
        # Trade-offs
        print("TRADE-OFFS")
        print("-" * 40)
        print("⚠ Does not test every possible persona combination")
        print("⚠ May miss some edge cases in persona interactions")
        print("⚠ Requires statistical analysis to validate coverage")
        print()
        
        return comparison

def main():
    """Main function to run the comparison"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare factorial vs stochastic experiment creation')
    parser.add_argument('--ground-truth-count', type=int, default=100,
                       help='Number of ground truth cases to analyze (default: 100)')
    parser.add_argument('--personas-per-baseline', type=int, default=10,
                       help='Number of personas to sample per baseline (default: 10)')
    
    args = parser.parse_args()
    
    comparison = ExperimentCreationComparison()
    
    try:
        comparison.print_comparison(args.ground_truth_count, args.personas_per_baseline)
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        comparison.creator.close()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
