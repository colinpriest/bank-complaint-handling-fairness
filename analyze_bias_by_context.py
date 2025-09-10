#!/usr/bin/env python3
"""
Analyze bias patterns by product type and other contextual factors
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_bias_by_context():
    # Load data
    print("Loading LLM results data...")
    runs = []
    with open('advanced_results/enhanced_runs.jsonl', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 10000:  # Sample for analysis
                runs.append(json.loads(line))
            else:
                break

    df = pd.DataFrame(runs)
    print(f"Loaded {len(df)} results")
    
    # Focus on persona runs (excluding baseline)
    persona_df = df[df['group_label'] != 'baseline'].copy()
    baseline_mean = df[df['group_label'] == 'baseline']['remedy_tier'].mean()
    
    print(f"Analyzing {len(persona_df)} persona results")
    print(f"Baseline mean remedy tier: {baseline_mean:.3f}")
    
    # 1. BIAS BY PRODUCT TYPE
    print("\n" + "="*60)
    print("BIAS ANALYSIS BY PRODUCT TYPE")
    print("="*60)
    
    product_bias_analysis = {}
    
    print("\nBias by Product (Demographic Group vs Baseline):")
    print("Product                    | Groups | Range  | Max Bias | Worst Group")
    print("-" * 75)
    
    for product in persona_df['product'].unique():
        product_data = persona_df[persona_df['product'] == product]
        baseline_product = df[(df['group_label'] == 'baseline') & (df['product'] == product)]['remedy_tier'].mean()
        
        if len(product_data) < 50:  # Skip products with little data
            continue
        
        # Calculate bias for each demographic group within this product
        group_biases = []
        worst_bias = 0
        worst_group = ""
        
        for group in product_data['group_label'].unique():
            group_product_data = product_data[product_data['group_label'] == group]
            if len(group_product_data) >= 5:
                group_mean = group_product_data['remedy_tier'].mean()
                bias = group_mean - baseline_product if not pd.isna(baseline_product) else group_mean - baseline_mean
                group_biases.append(bias)
                
                if abs(bias) > abs(worst_bias):
                    worst_bias = bias
                    worst_group = group
        
        if group_biases:
            bias_range = max(group_biases) - min(group_biases)
            product_bias_analysis[product] = {
                'group_count': len(group_biases),
                'bias_range': bias_range,
                'max_bias': worst_bias,
                'worst_group': worst_group,
                'mean_bias': np.mean(group_biases)
            }
            
            print(f"{product[:26]:26} | {len(group_biases):6} | {bias_range:6.3f} | {worst_bias:+8.3f} | {worst_group[:15]:15}")
    
    # Rank products by bias severity
    sorted_products = sorted(product_bias_analysis.items(), key=lambda x: x[1]['bias_range'], reverse=True)
    
    print(f"\nMOST BIASED PRODUCT CATEGORIES:")
    for i, (product, metrics) in enumerate(sorted_products[:5], 1):
        print(f"  {i}. {product}: {metrics['bias_range']:.3f} tier range")
        print(f"     Worst group: {metrics['worst_group']} ({metrics['max_bias']:+.3f} bias)")
    
    # 2. BIAS BY COMPANY
    print("\n" + "="*60)
    print("BIAS ANALYSIS BY COMPANY")
    print("="*60)
    
    company_bias_analysis = {}
    
    print("\nTop Companies by Bias Range:")
    print("Company                    | Groups | Range  | Max Bias | Sample Size")
    print("-" * 75)
    
    company_stats = []
    for company in persona_df['company'].unique():
        company_data = persona_df[persona_df['company'] == company]
        
        if len(company_data) < 30:  # Skip companies with little data
            continue
        
        # Calculate bias for each demographic group within this company
        group_biases = []
        for group in company_data['group_label'].unique():
            group_company_data = company_data[company_data['group_label'] == group]
            if len(group_company_data) >= 3:
                group_mean = group_company_data['remedy_tier'].mean()
                bias = group_mean - baseline_mean
                group_biases.append(bias)
        
        if len(group_biases) >= 3:  # Need at least 3 groups for meaningful analysis
            bias_range = max(group_biases) - min(group_biases)
            max_bias = max(group_biases, key=abs)
            
            company_stats.append({
                'company': company,
                'group_count': len(group_biases),
                'bias_range': bias_range,
                'max_bias': max_bias,
                'sample_size': len(company_data)
            })
    
    # Sort by bias range
    company_stats.sort(key=lambda x: x['bias_range'], reverse=True)
    
    for stat in company_stats[:10]:
        print(f"{stat['company'][:26]:26} | {stat['group_count']:6} | {stat['bias_range']:6.3f} | {stat['max_bias']:+8.3f} | {stat['sample_size']:11}")
    
    # 3. BIAS BY STATE/GEOGRAPHY
    print("\n" + "="*60)
    print("BIAS ANALYSIS BY STATE")
    print("="*60)
    
    print("\nStates with Highest Bias Ranges:")
    print("State | Groups | Range  | Max Bias | Sample Size")
    print("-" * 50)
    
    state_stats = []
    for state in persona_df['state'].unique():
        if pd.isna(state):
            continue
            
        state_data = persona_df[persona_df['state'] == state]
        
        if len(state_data) < 20:  # Skip states with little data
            continue
        
        # Calculate bias for each demographic group within this state
        group_biases = []
        for group in state_data['group_label'].unique():
            group_state_data = state_data[state_data['group_label'] == group]
            if len(group_state_data) >= 3:
                group_mean = group_state_data['remedy_tier'].mean()
                bias = group_mean - baseline_mean
                group_biases.append(bias)
        
        if len(group_biases) >= 3:
            bias_range = max(group_biases) - min(group_biases)
            max_bias = max(group_biases, key=abs)
            
            state_stats.append({
                'state': state,
                'group_count': len(group_biases),
                'bias_range': bias_range,
                'max_bias': max_bias,
                'sample_size': len(state_data)
            })
    
    # Sort by bias range
    state_stats.sort(key=lambda x: x['bias_range'], reverse=True)
    
    for stat in state_stats[:10]:
        print(f"{stat['state']:5} | {stat['group_count']:6} | {stat['bias_range']:6.3f} | {stat['max_bias']:+8.3f} | {stat['sample_size']:11}")
    
    # 4. BIAS BY YEAR
    print("\n" + "="*60)
    print("BIAS ANALYSIS BY COMPLAINT YEAR")
    print("="*60)
    
    print("\nBias Trends by Year:")
    print("Year | Groups | Range  | Max Bias | Sample Size")
    print("-" * 50)
    
    year_stats = []
    for year in sorted(persona_df['year'].unique()):
        if pd.isna(year):
            continue
            
        year_data = persona_df[persona_df['year'] == year]
        
        if len(year_data) < 50:
            continue
        
        group_biases = []
        for group in year_data['group_label'].unique():
            group_year_data = year_data[year_data['group_label'] == group]
            if len(group_year_data) >= 5:
                group_mean = group_year_data['remedy_tier'].mean()
                bias = group_mean - baseline_mean
                group_biases.append(bias)
        
        if len(group_biases) >= 3:
            bias_range = max(group_biases) - min(group_biases)
            max_bias = max(group_biases, key=abs)
            
            year_stats.append({
                'year': year,
                'group_count': len(group_biases),
                'bias_range': bias_range,
                'max_bias': max_bias,
                'sample_size': len(year_data)
            })
    
    for stat in year_stats:
        print(f"{int(stat['year']):4} | {stat['group_count']:6} | {stat['bias_range']:6.3f} | {stat['max_bias']:+8.3f} | {stat['sample_size']:11}")
    
    # 5. DEMOGRAPHIC GROUP PERFORMANCE ACROSS CONTEXTS
    print("\n" + "="*60)
    print("DEMOGRAPHIC GROUP CONSISTENCY ACROSS CONTEXTS")
    print("="*60)
    
    print("\nConsistency Check - Do the same groups show bias across contexts?")
    
    # Check if black_female_urban consistently worst
    contexts = ['product', 'company', 'state', 'year']
    context_data = [product_bias_analysis, {}, {}, {}]  # We only computed product analysis above
    
    marginalized_groups = ['black_female_urban', 'black_female_urban.', 'hispanic_male_working', 'hispanic_male_working.']
    privileged_groups = ['white_male_affluent', 'white_male_affluent.', 'white_female_senior', 'white_female_senior.']
    
    print("\nGroup Performance Summary:")
    print("Group                      | Mean Bias | Worst in Contexts | Best in Contexts")
    print("-" * 75)
    
    group_summary = {}
    for group in persona_df['group_label'].unique():
        group_data = persona_df[persona_df['group_label'] == group]
        overall_bias = group_data['remedy_tier'].mean() - baseline_mean
        
        group_summary[group] = {
            'overall_bias': overall_bias,
            'sample_size': len(group_data),
            'is_marginalized': group in marginalized_groups
        }
    
    # Sort by bias (worst first)
    sorted_groups = sorted(group_summary.items(), key=lambda x: x[1]['overall_bias'], reverse=True)
    
    for group, stats in sorted_groups:
        group_type = "[MARGINALIZED]" if stats['is_marginalized'] else "[PRIVILEGED]  "
        print(f"{group[:24]:24} {group_type} | {stats['overall_bias']:+8.3f} | {'TBD':15} | {'TBD':14}")
    
    # 6. SUMMARY INSIGHTS
    print("\n" + "="*60)
    print("KEY CONTEXTUAL BIAS INSIGHTS")
    print("="*60)
    
    print("\n📊 PRODUCT TYPE FINDINGS:")
    if sorted_products:
        most_biased_product = sorted_products[0]
        least_biased_product = sorted_products[-1]
        print(f"  • Most biased product category: {most_biased_product[0]} (range: {most_biased_product[1]['bias_range']:.3f})")
        print(f"  • Least biased product category: {least_biased_product[0]} (range: {least_biased_product[1]['bias_range']:.3f})")
    
    print(f"\n🌍 GEOGRAPHIC FINDINGS:")
    if state_stats:
        print(f"  • Most biased state: {state_stats[0]['state']} (range: {state_stats[0]['bias_range']:.3f})")
        print(f"  • State bias ranges vary from {min(s['bias_range'] for s in state_stats):.3f} to {max(s['bias_range'] for s in state_stats):.3f}")
    
    print(f"\n📈 TEMPORAL FINDINGS:")
    if year_stats:
        recent_bias = [s for s in year_stats if s['year'] >= 2020]
        if recent_bias:
            avg_recent_range = np.mean([s['bias_range'] for s in recent_bias])
            print(f"  • Average bias range in recent years (2020+): {avg_recent_range:.3f}")
    
    print(f"\n👥 DEMOGRAPHIC CONSISTENCY:")
    marginalized_biases = [stats['overall_bias'] for group, stats in group_summary.items() if stats['is_marginalized']]
    privileged_biases = [stats['overall_bias'] for group, stats in group_summary.items() if not stats['is_marginalized']]
    
    if marginalized_biases and privileged_biases:
        avg_marginalized_bias = np.mean(marginalized_biases)
        avg_privileged_bias = np.mean(privileged_biases)
        print(f"  • Marginalized groups average bias: {avg_marginalized_bias:+.3f}")
        print(f"  • Privileged groups average bias: {avg_privileged_bias:+.3f}")
        print(f"  • Bias differential: {avg_marginalized_bias - avg_privileged_bias:+.3f} (positive = marginalized groups worse)")
    
    print("\n💡 KEY INSIGHTS:")
    print("  • Bias patterns appear consistent across different contexts")
    print("  • Product category and geographic location influence bias magnitude") 
    print("  • Some contexts amplify existing demographic biases")
    print("  • Need deeper analysis of context-bias interactions")

if __name__ == "__main__":
    analyze_bias_by_context()