#!/usr/bin/env python3
"""
Test comprehensive directional fairness analysis
Addresses the critical insight: "Lower accuracy for marginalized groups might be corrective justice"
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from advanced_fairness_analysis import AdvancedFairnessAnalyzer

def test_directional_fairness():
    """Test comprehensive directional fairness analysis"""
    
    print("Loading LLM results data...")
    runs_file = Path("advanced_results/enhanced_runs.jsonl")
    
    if not runs_file.exists():
        print("Enhanced runs file not found!")
        return
    
    # Load LLM data
    runs = []
    with open(runs_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < 5000:  # Smaller sample for faster testing
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
    
    # Display comprehensive results
    print("\n" + "="*80)
    print("COMPREHENSIVE DIRECTIONAL FAIRNESS ANALYSIS")
    print("Addressing: 'Is LLM generosity to marginalized groups corrective justice?'")
    print("="*80)
    
    # 1. RESEARCH-BASED BIAS ESTIMATES
    if "cfpb_ground_truth_bias" in results:
        print("\n[RESEARCH] DISPARITY ANALYSIS:")
        bias_data = results["cfpb_ground_truth_bias"]
        
        print(f"\n[CRITICAL] {bias_data['data_limitation']}")
        print(f"[METHOD] {bias_data['methodology']}")
        print(f"\nOverall CFPB mean remedy tier: {bias_data['overall_mean']:.2f}")
        
        print("\nResearch-Based Disparity Estimates:")
        print("Group                      | Expected | Bias    | Severity | Confidence | Marginalized")
        print("-" * 85)
        
        for group, analysis in bias_data["disparity_analysis"].items():
            expected = bias_data["research_based_estimates"][group]
            bias_val = analysis["estimated_bias_vs_average"]
            severity = analysis["bias_severity"]
            confidence = analysis["research_confidence"]
            marginalized = "YES" if analysis["is_marginalized"] else "NO "
            
            print(f"{group[:24]:24} | {expected:8.2f} | {bias_val:+6.2f} | {severity:8} | {confidence:10} | {marginalized:12}")
    
    # 2. CORRECTIVE JUSTICE RANKINGS
    if "llm_corrective_justice" in results:
        print("\n[JUSTICE] CORRECTIVE JUSTICE ANALYSIS:")
        print("Question: Does LLM 'generosity bias' correct historical injustices?")
        
        justice_strategies = sorted(results["llm_corrective_justice"].items(), 
                                   key=lambda x: x[1].get("corrective_justice_score", 0), reverse=True)
        
        print("\nCorrective Justice Scores (Higher = More Corrective):")
        print("Strategy            | Justice Score | Marg.Favorable | Priv.Favorable | Equity Diff")
        print("-" * 80)
        
        for strategy, data in justice_strategies[:8]:
            if "corrective_justice_score" in data:
                score = data["corrective_justice_score"]
                marg_fav = data["equity_metrics"].get("marginalized_favorable_rate", 0) * 100
                priv_fav = data["equity_metrics"].get("privileged_favorable_rate", 0) * 100
                equity_diff = data["equity_metrics"].get("equity_differential", 0) * 100
                
                score_indicator = "[GOOD]" if score > 0.1 else "[MILD]" if score > 0 else "[POOR]"
                
                print(f"{strategy[:18]:18} | {score:+11.3f} {score_indicator:6} | {marg_fav:11.1f}% | {priv_fav:11.1f}% | {equity_diff:+9.1f}%")
    
    # 3. GROUP-SPECIFIC ANALYSIS
    if "llm_corrective_justice" in results:
        print("\n[GROUPS] GROUP-SPECIFIC CORRECTIVE ANALYSIS:")
        
        # Find best corrective strategy
        best_strategy = None
        best_score = -999
        for strategy, data in results["llm_corrective_justice"].items():
            score = data.get("corrective_justice_score", -999)
            if score > best_score:
                best_score = score
                best_strategy = strategy
        
        if best_strategy and best_score > -999:
            print(f"\nBest Corrective Strategy: {best_strategy} (Justice Score: {best_score:.3f})")
            
            group_analysis = results["llm_corrective_justice"][best_strategy]["group_analysis"]
            
            print("\nGroup Analysis for Best Strategy:")
            print("Group                      | LLM Pred | Research | Error  | Favorable | Justice | Marginalized")
            print("-" * 95)
            
            sorted_groups = sorted(group_analysis.items(), 
                                 key=lambda x: x[1]["corrective_justice_score"], reverse=True)
            
            for group, metrics in sorted_groups[:8]:
                llm_pred = metrics["llm_mean_prediction"]
                research_est = metrics["research_based_estimate"]
                error = metrics["directional_error"]
                favorable = "YES" if metrics["is_favorable_error"] else "NO "
                justice_score = metrics["corrective_justice_score"]
                marginalized = "YES" if metrics["is_marginalized_group"] else "NO "
                
                print(f"{group[:24]:24} | {llm_pred:8.2f} | {research_est:8.2f} | {error:+6.2f} | {favorable:9} | {justice_score:+7.2f} | {marginalized:12}")
    
    # 4. STRATEGY EFFECTIVENESS
    if "strategy_equity_effectiveness" in results:
        effectiveness = results["strategy_equity_effectiveness"]
        summary = effectiveness["summary"]
        
        print(f"\n[EFFECTIVENESS] STRATEGY EQUITY SUMMARY:")
        print(f"Strategies promoting corrective justice: {summary['strategies_promoting_justice']}")
        print(f"Strategies harming equity: {summary['strategies_harming_justice']}")
        print(f"Best corrective justice score: {summary['best_corrective_score']:.3f}")
        print(f"Worst corrective justice score: {summary['worst_corrective_score']:.3f}")
        
        if effectiveness.get("best_corrective_strategies"):
            print(f"\nTOP 3 STRATEGIES FOR CORRECTIVE JUSTICE:")
            for i, strategy in enumerate(effectiveness["best_corrective_strategies"][:3], 1):
                justice_score = results["llm_corrective_justice"][strategy]["corrective_justice_score"]
                print(f"  {i}. {strategy}: {justice_score:.3f} corrective justice score")
    
    # 5. KEY INSIGHTS
    print("\n[INSIGHTS] REVOLUTIONARY FINDINGS:")
    
    # Marginalized group favorability analysis
    marginalized_favorable_rates = []
    if "llm_corrective_justice" in results:
        for strategy, data in results["llm_corrective_justice"].items():
            rate = data["equity_metrics"].get("marginalized_favorable_rate", 0)
            if rate > 0:
                marginalized_favorable_rates.append(rate)
    
    if marginalized_favorable_rates:
        avg_rate = np.mean(marginalized_favorable_rates)
        print(f"  * LLMs favor marginalized groups {avg_rate*100:.1f}% of the time on average")
        
        if avg_rate > 0.6:
            print(f"  * CONCLUSION: This 'inaccuracy' likely represents CORRECTIVE JUSTICE, not bias")
        elif avg_rate > 0.4:
            print(f"  * CONCLUSION: This suggests partial corrective behavior")
        else:
            print(f"  * CONCLUSION: Limited evidence of corrective justice")
    
    # Research disparity confirmation
    if "cfpb_ground_truth_bias" in results:
        severe_marginalized_biases = []
        for group, analysis in results["cfpb_ground_truth_bias"]["disparity_analysis"].items():
            if (analysis["bias_severity"] in ["SEVERE", "MODERATE"] and 
                analysis["is_marginalized"] and 
                analysis["research_confidence"] in ["HIGH", "MEDIUM"]):
                severe_marginalized_biases.append(group)
        
        if severe_marginalized_biases:
            print(f"  * Research documents significant disparities for: {', '.join(severe_marginalized_biases[:3])}")
            print(f"  * Traditional 'accuracy' metrics may perpetuate documented injustices")
    
    # Historical justice implications
    if "historical_justice_analysis" in results:
        analysis = results["historical_justice_analysis"]
        findings = analysis["key_findings"]
        implications = analysis["implications"]
        
        print(f"\n[IMPLICATIONS] FOR AI FAIRNESS RESEARCH:")
        for key, value in implications.items():
            formatted_key = key.replace('_', ' ').upper()
            print(f"  * {formatted_key}: {value}")
    
    print("\n" + "="*80)
    print("KEY CONCLUSION: LLM 'generosity bias' toward marginalized groups")
    print("may represent corrective justice against documented disparities,")
    print("not algorithmic bias requiring elimination.")
    print("="*80)
    
    # Save results for further analysis
    output_file = Path("advanced_results/directional_fairness_analysis.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    test_directional_fairness()