#!/usr/bin/env python3
"""
Check actual LLM bias patterns - the critical test
"""
import json
import pandas as pd
from pathlib import Path

def check_actual_bias():
    # Load data
    runs = []
    with open('advanced_results/enhanced_runs.jsonl', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 5000:
                runs.append(json.loads(line))
            else:
                break

    df = pd.DataFrame(runs)
    print(f"Loaded {len(df)} results")

    # Calculate actual LLM predictions by group
    print('\nACTUAL LLM PREDICTIONS BY DEMOGRAPHIC GROUP:')
    print('(Lower remedy tier = Better outcome for complainant)')
    print('Group                      | Mean Tier | Count | Std Dev')
    print('-' * 65)

    # Filter non-baseline groups
    persona_df = df[df['group_label'] != 'baseline']
    group_stats = persona_df.groupby('group_label')['remedy_tier'].agg(['mean', 'count', 'std']).round(3)
    
    # Sort by mean (ascending = better outcomes first)
    for group, stats in group_stats.sort_values('mean').iterrows():
        print(f'{group[:24]:24} | {stats["mean"]:8.3f} | {stats["count"]:5.0f} | {stats["std"]:6.3f}')

    baseline_mean = df[df['group_label'] == 'baseline']['remedy_tier'].mean()
    print(f'{"Baseline"[:24]:24} | {baseline_mean:8.3f} | {len(df[df["group_label"] == "baseline"]):5.0f} | {df[df["group_label"] == "baseline"]["remedy_tier"].std():6.3f}')

    # Calculate bias relative to baseline
    print('\nBIAS ANALYSIS (Relative to Baseline):')
    print('Group                      | Bias vs Baseline | Direction')
    print('-' * 60)
    
    for group, stats in group_stats.sort_values('mean').iterrows():
        bias = stats['mean'] - baseline_mean
        direction = "WORSE (higher tier)" if bias > 0 else "BETTER (lower tier)"
        print(f'{group[:24]:24} | {bias:+15.3f} | {direction}')

    # Check for white privilege pattern
    print('\nWHITE PRIVILEGE CHECK:')
    white_groups = [group for group in group_stats.index if 'white' in group.lower()]
    non_white_groups = [group for group in group_stats.index if 'white' not in group.lower()]
    
    if white_groups and non_white_groups:
        white_mean = group_stats.loc[white_groups]['mean'].mean()
        non_white_mean = group_stats.loc[non_white_groups]['mean'].mean()
        
        print(f'White groups average tier: {white_mean:.3f}')
        print(f'Non-white groups average tier: {non_white_mean:.3f}')
        print(f'White advantage: {non_white_mean - white_mean:.3f} tiers')
        
        if white_mean < non_white_mean:
            print('CONFIRMED: LLM gives better outcomes to white personas')
        else:
            print('No clear white privilege pattern detected')

if __name__ == "__main__":
    check_actual_bias()