#!/usr/bin/env python3
"""
Test accuracy equalization analysis for fairness strategies
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from advanced_fairness_analysis import AdvancedFairnessAnalyzer

def test_equalization_analysis():
    """Test how fairness strategies affect accuracy equalization"""
    
    print("Loading LLM results data...")
    runs_file = Path("advanced_results/enhanced_runs.jsonl")
    
    if not runs_file.exists():
        print("Enhanced runs file not found!")
        return
    
    # Load LLM data
    runs = []
    with open(runs_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < 10000:  # Sample for testing
                runs.append(json.loads(line))
            else:
                break
    
    df = pd.DataFrame(runs)
    print(f"Loaded {len(df)} LLM results")
    
    # Initialize analyzer
    analyzer = AdvancedFairnessAnalyzer()
    
    # Run comprehensive directional fairness analysis
    print("\nRunning comprehensive directional fairness analysis...")
    results = analyzer.analyze_directional_fairness_comprehensive(df, cfpb_ground_truth_mean=2.70)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    # Display results
    print("\n" + "="*80)
    print("FAIRNESS STRATEGY ACCURACY EQUALIZATION ANALYSIS")
    print("="*80)
    
    # Show equalization summary
    if "equalization_summary" in results:
        summary = results["equalization_summary"]
        print("\n📊 EQUALIZATION SUMMARY:")
        print(f"  Strategies analyzed: {summary['strategies_analyzed']}")
        print(f"  Best variance (most equal): {summary['best_variance']:.6f}")
        print(f"  Worst variance (least equal): {summary['worst_variance']:.6f}")
        print(f"  Variance range: {summary['variance_range']:.6f}")
        
        if summary.get("equalizing_strategies"):
            print(f"\n✅ STRATEGIES THAT IMPROVE EQUALIZATION:")
            for strategy in summary["equalizing_strategies"]:
                metrics = results["strategy_equalization_metrics"].get(strategy, {})
                improvement = metrics.get("percent_improvement_vs_baseline", 0)
                print(f"  • {strategy}: {improvement:.1f}% improvement")
        
        if summary.get("disequalizing_strategies"):
            print(f"\n❌ STRATEGIES THAT WORSEN EQUALIZATION:")
            for strategy in summary["disequalizing_strategies"]:
                metrics = results["strategy_equalization_metrics"].get(strategy, {})
                worsening = metrics.get("percent_improvement_vs_baseline", 0)
                print(f"  • {strategy}: {abs(worsening):.1f}% worse")
        
        if summary.get("neutral_strategies"):
            print(f"\n⚪ NEUTRAL STRATEGIES (no significant effect):")
            for strategy in summary["neutral_strategies"][:5]:  # Limit display
                print(f"  • {strategy}")
    
    # Show top rankings
    if results.get("equalization_rankings"):
        print("\n🏆 TOP 5 STRATEGIES FOR ACCURACY EQUALIZATION:")
        print("(Lower variance = better equalization across demographic groups)")
        print("\nRank | Strategy | Accuracy Variance | Range | Gini | Score")
        print("-"*70)
        
        for i, ranking in enumerate(results["equalization_rankings"][:5], 1):
            print(f"{i:4} | {ranking['strategy'][:20]:20} | "
                  f"{ranking['accuracy_variance']:.6f} | "
                  f"{ranking['accuracy_range']:.3f} | "
                  f"{ranking['gini_coefficient']:.3f} | "
                  f"{ranking['equalization_score']:.3f}")
    
    # Show best equalizing strategies in detail
    if results.get("best_equalizing_strategies"):
        print("\n📈 DETAILED ANALYSIS OF BEST EQUALIZING STRATEGIES:")
        
        for strategy in results["best_equalizing_strategies"][:3]:
            metrics = results["strategy_equalization_metrics"].get(strategy)
            if metrics:
                print(f"\n  Strategy: {strategy}")
                print(f"    Accuracy variance: {metrics['accuracy_variance']:.6f}")
                print(f"    Accuracy range: {metrics['accuracy_range']:.3f}")
                print(f"    Gini coefficient: {metrics['gini_coefficient']:.3f}")
                print(f"    Mean absolute error: {metrics['mean_absolute_error']:.3f}")
                
                if metrics.get("percent_improvement_vs_baseline") is not None:
                    print(f"    Improvement vs baseline: {metrics['percent_improvement_vs_baseline']:.1f}%")
                
                # Show per-group accuracy
                if "group_accuracies" in metrics:
                    print(f"    Per-group MAE from ground truth:")
                    sorted_groups = sorted(metrics["group_accuracies"].items(), 
                                         key=lambda x: x[1]["mae_from_truth"])
                    for group, acc in sorted_groups[:5]:  # Top 5 groups
                        mae = acc["mae_from_truth"]
                        n = acc["sample_size"]
                        print(f"      • {group[:30]:30}: {mae:.3f} (n={n})")
    
    # Compare baseline strategy
    baseline_metrics = results["strategy_equalization_metrics"].get("none")
    if baseline_metrics:
        print("\n📊 BASELINE (NO STRATEGY) METRICS:")
        print(f"  Accuracy variance: {baseline_metrics['accuracy_variance']:.6f}")
        print(f"  Accuracy range: {baseline_metrics['accuracy_range']:.3f}")
        print(f"  Mean absolute error: {baseline_metrics['mean_absolute_error']:.3f}")
        
        if baseline_metrics.get("baseline_differential") is not None:
            print(f"  Baseline vs demographic differential: {baseline_metrics['baseline_differential']:.3f}")
    
    # Key insights
    print("\n💡 KEY INSIGHTS:")
    
    # Find strategy with best improvement
    best_improvement = 0
    best_strategy = None
    for strategy, metrics in results["strategy_equalization_metrics"].items():
        if strategy != "none" and "percent_improvement_vs_baseline" in metrics:
            if metrics["percent_improvement_vs_baseline"] > best_improvement:
                best_improvement = metrics["percent_improvement_vs_baseline"]
                best_strategy = strategy
    
    if best_strategy:
        print(f"  • Best strategy for equalization: {best_strategy} "
              f"({best_improvement:.1f}% variance reduction)")
    
    # Check if any strategy achieves good equalization
    min_variance = min(results["accuracy_variance_by_strategy"].values()) if results["accuracy_variance_by_strategy"] else float('inf')
    if min_variance < 0.01:
        print(f"  • Excellent equalization achieved (variance < 0.01)")
    elif min_variance < 0.05:
        print(f"  • Good equalization achieved (variance < 0.05)")
    elif min_variance < 0.1:
        print(f"  • Moderate equalization achieved (variance < 0.1)")
    else:
        print(f"  • Poor equalization - significant accuracy disparities remain")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    test_equalization_analysis()