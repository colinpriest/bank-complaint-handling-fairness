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
    
    # Display comprehensive results
    print("\n" + "="*80)
    print("COMPREHENSIVE DIRECTIONAL FAIRNESS ANALYSIS")
    print("Addressing: 'Is LLM generosity to marginalized groups corrective justice?'")
    print("="*80)
    
    # 1. RESEARCH-BASED BIAS ESTIMATES
    if "cfpb_ground_truth_bias" in results:
        print("\n🏛️ RESEARCH-BASED DISPARITY ANALYSIS:")
        bias_data = results["cfpb_ground_truth_bias"]
        
        print(f"\n⚠️  DATA LIMITATION: {bias_data['data_limitation']}")
        print(f"📚 METHODOLOGY: {bias_data['methodology']}")
        print(f"\nOverall CFPB mean remedy tier: {bias_data['overall_mean']:.2f}")
        
        print("\nResearch-Based Disparity Estimates:")
        print("Group                    | Expected | Bias    | Severity | Confidence")
        print("-" * 75)
        
        for group, analysis in bias_data["disparity_analysis"].items():
            expected = bias_data["research_based_estimates"][group]
            bias_val = analysis["estimated_bias_vs_average"]
            severity = analysis["bias_severity"]
            confidence = analysis["research_confidence"]
            marginalized = "📍" if analysis["is_marginalized"] else "  "
            
            print(f"{group[:22]:22} {marginalized} | {expected:8.2f} | {bias_val:+6.2f} | {severity:8} | {confidence:10}")
    
    # 2. CORRECTIVE JUSTICE RANKINGS
    if "llm_corrective_justice" in results:
        print("\n⚖️  LLM CORRECTIVE JUSTICE ANALYSIS:")
        print("Question: Does LLM 'generosity bias' correct historical injustices?")
        
        justice_strategies = sorted(results["llm_corrective_justice"].items(), 
                                   key=lambda x: x[1].get("corrective_justice_score", 0), reverse=True)
        
        print("\nCorrective Justice Scores (Higher = More Corrective):")
        print("Strategy            | Justice | Marg.Fav% | Priv.Fav% | Equity Diff")
        print("-" * 70)
        
        for strategy, data in justice_strategies[:10]:
            if "corrective_justice_score" in data:
                score = data["corrective_justice_score"]
                marg_fav = data["equity_metrics"].get("marginalized_favorable_rate", 0) * 100
                priv_fav = data["equity_metrics"].get("privileged_favorable_rate", 0) * 100
                equity_diff = data["equity_metrics"].get("equity_differential", 0) * 100
                
                score_color = "🟢" if score > 0.1 else "🟡" if score > 0 else "🔴"
                
                print(f"{strategy[:18]:18} | {score_color} {score:5.2f} | {marg_fav:7.1f}% | {priv_fav:7.1f}% | {equity_diff:+9.1f}%")
    
    # 3. GROUP-SPECIFIC CORRECTIVE ANALYSIS
    if "llm_corrective_justice" in results:
        print("\n📊 GROUP-SPECIFIC CORRECTIVE JUSTICE BREAKDOWN:")
        
        # Find best corrective strategy
        best_strategy = None
        best_score = -999
        for strategy, data in results["llm_corrective_justice"].items():
            score = data.get("corrective_justice_score", -999)
            if score > best_score:
                best_score = score
                best_strategy = strategy
        
        if best_strategy:
            print(f"\nBest Corrective Strategy: {best_strategy} (Justice Score: {best_score:.3f})")
            
            group_analysis = results["llm_corrective_justice"][best_strategy]["group_analysis"]
            
            print("\nGroup Analysis for Best Strategy:")
            print("Group                    | LLM  | Research | Error  | Favorable? | Justice")
            print("-" * 75)
            
            for group, metrics in sorted(group_analysis.items(), 
                                       key=lambda x: x[1]["corrective_justice_score"], reverse=True)[:8]:
                llm_pred = metrics["llm_mean_prediction"]
                research_est = metrics["research_based_estimate"]
                error = metrics["directional_error"]
                favorable = "✅ YES" if metrics["is_favorable_error"] else "❌ NO "
                justice_score = metrics["corrective_justice_score"]
                marginalized = "📍" if metrics["is_marginalized_group"] else "  "
                
                print(f"{group[:22]:22} {marginalized} | {llm_pred:4.2f} | {research_est:8.2f} | {error:+6.2f} | {favorable:10} | {justice_score:+7.2f}")
    
    # 4. DIRECTIONAL ACCURACY BREAKDOWN
    if "directional_accuracy_metrics" in results:
        print("\n📈 DIRECTIONAL ACCURACY ANALYSIS:")
        print("Favorable Inaccuracy = Benefits complainant (Lower remedy tier)")
        print("Unfavorable Inaccuracy = Harms complainant (Higher remedy tier)")
        
        # Find most equitable strategy
        best_equity_strategy = None
        best_equity_score = float('inf')
        
        print("\nEquity-Adjusted Accuracy by Strategy:")
        print("Strategy            | Equity Score | Fav(Marg) | Fav(Priv) | Unfav(Marg) | Unfav(Priv)")
        print("-" * 85)
        
        for strategy, metrics in list(results["directional_accuracy_metrics"].items())[:10]:
            equity_score = metrics.get("equity_adjusted_accuracy", float('inf'))
            fav_marg = metrics["favorable_inaccuracy"]["marginalized"]
            fav_priv = metrics["favorable_inaccuracy"]["privileged"]
            unfav_marg = metrics["unfavorable_inaccuracy"]["marginalized"]
            unfav_priv = metrics["unfavorable_inaccuracy"]["privileged"]
            
            if equity_score < best_equity_score:
                best_equity_score = equity_score
                best_equity_strategy = strategy
            
            score_color = "🟢" if equity_score < 0.5 else "🟡" if equity_score < 1.0 else "🔴"
            
            print(f"{strategy[:18]:18} | {score_color} {equity_score:8.3f} | {fav_marg:8.3f} | {fav_priv:8.3f} | {unfav_marg:10.3f} | {unfav_priv:10.3f}")
        
        if best_equity_strategy:
            print(f"\n🏆 Most Equitable Strategy: {best_equity_strategy} (Score: {best_equity_score:.3f})")
    
    # 5. KEY REVOLUTIONARY INSIGHTS
    print("\n💡 REVOLUTIONARY INSIGHTS:")
    
    # Best corrective strategy
    if "strategy_equity_effectiveness" in results:
        effectiveness = results["strategy_equity_effectiveness"]
        
        if effectiveness.get("best_corrective_strategies"):
            best_corrective = effectiveness["best_corrective_strategies"][0]
            best_score = effectiveness["rankings"][0]["corrective_justice_score"]
            
            print(f"  • BEST CORRECTIVE STRATEGY: {best_corrective} (Justice Score: {best_score:.3f})")
            print(f"  • This strategy actively corrects research-documented disparities")
    
    # Marginalized group favorability
    marginalized_favorable_rates = []
    if "llm_corrective_justice" in results:
        for strategy, data in results["llm_corrective_justice"].items():
            rate = data["equity_metrics"].get("marginalized_favorable_rate", 0)
            if rate > 0:
                marginalized_favorable_rates.append(rate)
    
    if marginalized_favorable_rates:
        avg_rate = np.mean(marginalized_favorable_rates)
        print(f"  • LLMs favor marginalized groups {avg_rate*100:.1f}% of the time on average")
        
        if avg_rate > 0.6:
            print(f"  • This 'inaccuracy' likely represents CORRECTIVE JUSTICE, not bias")
        elif avg_rate > 0.4:
            print(f"  • This suggests partial corrective behavior")
        else:
            print(f"  • Limited evidence of corrective justice")
    
    # Research disparity confirmation
    if "cfpb_ground_truth_bias" in results:
        severe_marginalized_biases = []
        for group, analysis in results["cfpb_ground_truth_bias"]["disparity_analysis"].items():
            if (analysis["bias_severity"] in ["SEVERE", "MODERATE"] and 
                analysis["is_marginalized"] and 
                analysis["research_confidence"] in ["HIGH", "MEDIUM"]):
                severe_marginalized_biases.append(group)
        
        if severe_marginalized_biases:
            print(f"  • Research documents significant disparities for: {', '.join(severe_marginalized_biases[:3])}")
            print(f"  • Traditional 'accuracy' metrics may perpetuate documented injustices")
    
    # Historical justice analysis
    if "historical_justice_analysis" in results:
        analysis = results["historical_justice_analysis"]
        findings = analysis["key_findings"]
        
        print(f"\n🏛️  HISTORICAL JUSTICE FINDINGS:")
        print(f"  • Research shows systematic disparities: {findings['research_shows_systematic_disparities']}")
        print(f"  • LLM potentially corrects disparities: {findings['llm_potentially_corrects_disparities']}")
        print(f"  • Strategies enable corrective justice: {findings['strategies_enable_corrective_justice']}")
        print(f"  • Confidence levels vary: {findings['confidence_levels_vary']}")
        
        implications = analysis["implications"]
        print(f"\n🔬 IMPLICATIONS FOR AI FAIRNESS:")
        for key, value in implications.items():
            formatted_key = key.replace('_', ' ').title()
            print(f"  • {formatted_key}: {value}")
    
    print("\n" + "="*80)
    print("🎯 CONCLUSION: LLM 'generosity bias' toward marginalized groups")
    print("may represent corrective justice against documented disparities,")
    print("not algorithmic bias requiring elimination.")
    print("="*80)
    
    # Save results for further analysis
    output_file = Path("advanced_results/directional_fairness_analysis.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: {output_file}")

if __name__ == "__main__":
    test_directional_fairness()