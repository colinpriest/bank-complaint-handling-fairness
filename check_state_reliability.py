#!/usr/bin/env python3
"""
Check statistical reliability of state bias variations
"""
import json
import pandas as pd
import numpy as np
from scipy import stats

def check_state_reliability():
    """Check if state bias variations are statistically reliable"""
    
    print("STATISTICAL RELIABILITY OF STATE BIAS VARIATIONS")
    print("="*60)
    
    # Load LLM results
    runs = []
    with open('advanced_results/enhanced_runs.jsonl', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 5000:  # Sample for analysis
                runs.append(json.loads(line))
    
    df = pd.DataFrame(runs)
    
    # Focus on persona runs vs baseline
    baseline_df = df[df['group_label'] == 'baseline']
    persona_df = df[df['group_label'] != 'baseline']
    baseline_mean = baseline_df['remedy_tier'].mean()
    
    print(f"Baseline mean: {baseline_mean:.3f} (n={len(baseline_df)})")
    
    # Calculate state bias with statistical reliability measures
    state_analysis = []
    
    for state in persona_df['state'].unique():
        if pd.isna(state):
            continue
            
        state_data = persona_df[persona_df['state'] == state]
        
        if len(state_data) >= 5:  # Minimum for analysis
            state_mean = state_data['remedy_tier'].mean()
            state_std = state_data['remedy_tier'].std()
            bias = state_mean - baseline_mean
            
            # Standard error and confidence interval
            sem = state_std / np.sqrt(len(state_data))
            ci_95 = 1.96 * sem
            
            # Statistical significance test vs baseline
            baseline_sample = baseline_df['remedy_tier'].values
            state_sample = state_data['remedy_tier'].values
            
            try:
                t_stat, p_value = stats.ttest_ind(state_sample, baseline_sample, equal_var=False)
            except:
                t_stat, p_value = 0, 1.0
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(state_sample) - 1) * state_std**2 + (len(baseline_sample) - 1) * baseline_df['remedy_tier'].std()**2) / (len(state_sample) + len(baseline_sample) - 2))
            cohens_d = bias / pooled_std if pooled_std > 0 else 0
            
            state_analysis.append({
                'state': state,
                'count': len(state_data),
                'mean': state_mean,
                'bias': bias,
                'std': state_std,
                'sem': sem,
                'ci_95': ci_95,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05
            })
    
    # Sort by absolute bias
    state_analysis.sort(key=lambda x: abs(x['bias']), reverse=True)
    
    print(f"\nState | Count | Mean  | Bias  | 95%CI | p-value | Cohen's d | Reliable?")
    print("-" * 80)
    
    reliable_count = 0
    for state_info in state_analysis:
        # Reliability criteria
        large_n = state_info['count'] >= 30
        tight_ci = state_info['ci_95'] < 0.3
        significant = state_info['significant']
        
        reliability = "RELIABLE" if (large_n and tight_ci and significant) else \
                     "MARGINAL" if (state_info['count'] >= 15 and state_info['ci_95'] < 0.5) else \
                     "UNRELIABLE"
        
        if reliability == "RELIABLE":
            reliable_count += 1
        
        sig_marker = "***" if state_info['p_value'] < 0.001 else \
                    "**" if state_info['p_value'] < 0.01 else \
                    "*" if state_info['p_value'] < 0.05 else ""
        
        print(f"{state_info['state']:5} | {state_info['count']:5} | {state_info['mean']:5.3f} | "
              f"{state_info['bias']:+5.3f} | ±{state_info['ci_95']:4.3f} | "
              f"{state_info['p_value']:7.4f} | {state_info['cohens_d']:8.3f} | "
              f"{reliability:10} {sig_marker}")
    
    # Sample size analysis
    print(f"\n" + "="*60)
    print("SAMPLE SIZE ANALYSIS")
    print("="*60)
    
    counts = [s['count'] for s in state_analysis]
    print(f"Total states analyzed: {len(state_analysis)}")
    print(f"Mean sample size: {np.mean(counts):.1f}")
    print(f"Median sample size: {np.median(counts):.1f}")
    print(f"Min sample size: {np.min(counts)}")
    print(f"Max sample size: {np.max(counts)}")
    print(f"States with n >= 30: {len([c for c in counts if c >= 30])}")
    print(f"States with n >= 50: {len([c for c in counts if c >= 50])}")
    print(f"States with n >= 100: {len([c for c in counts if c >= 100])}")
    
    # Effect size analysis
    print(f"\n" + "="*60)
    print("EFFECT SIZE ANALYSIS")
    print("="*60)
    
    effect_sizes = [abs(s['cohens_d']) for s in state_analysis]
    print(f"Mean effect size (|Cohen's d|): {np.mean(effect_sizes):.3f}")
    print(f"Large effects (|d| > 0.8): {len([d for d in effect_sizes if d > 0.8])}")
    print(f"Medium effects (|d| > 0.5): {len([d for d in effect_sizes if d > 0.5])}")
    print(f"Small effects (|d| > 0.2): {len([d for d in effect_sizes if d > 0.2])}")
    
    # Multiple testing correction
    print(f"\n" + "="*60)
    print("MULTIPLE TESTING CORRECTION")
    print("="*60)
    
    alpha_bonferroni = 0.05 / len(state_analysis)
    bonferroni_significant = [s for s in state_analysis if s['p_value'] < alpha_bonferroni]
    
    # False Discovery Rate (Benjamini-Hochberg)
    p_values = [s['p_value'] for s in state_analysis]
    p_sorted_idx = np.argsort(p_values)
    fdr_threshold = 0.05
    
    fdr_significant = []
    for i, idx in enumerate(p_sorted_idx):
        bh_threshold = fdr_threshold * (i + 1) / len(p_values)
        if p_values[idx] <= bh_threshold:
            fdr_significant.append(state_analysis[idx])
    
    print(f"Bonferroni-corrected alpha: {alpha_bonferroni:.6f}")
    print(f"States significant after Bonferroni correction: {len(bonferroni_significant)}")
    print(f"States significant after FDR correction: {len(fdr_significant)}")
    
    if bonferroni_significant:
        print(f"Bonferroni-significant states: {[s['state'] for s in bonferroni_significant]}")
    
    if fdr_significant:
        print(f"FDR-significant states: {[s['state'] for s in fdr_significant]}")
    
    # Overall reliability assessment
    print(f"\n" + "="*60)
    print("OVERALL RELIABILITY ASSESSMENT")
    print("="*60)
    
    print(f"RELIABLE state findings: {reliable_count}/{len(state_analysis)} ({reliable_count/len(state_analysis)*100:.1f}%)")
    
    reliable_states = [s for s in state_analysis if s['count'] >= 30 and s['ci_95'] < 0.3 and s['significant']]
    
    if reliable_states:
        print(f"\nMOST RELIABLE FINDINGS:")
        for state_info in reliable_states[:5]:
            print(f"  {state_info['state']}: {state_info['bias']:+.3f} tier bias "
                  f"(n={state_info['count']}, p={state_info['p_value']:.4f}, "
                  f"CI=±{state_info['ci_95']:.3f})")
    
    # Conservative conclusions
    large_sample_states = [s for s in state_analysis if s['count'] >= 50]
    if large_sample_states:
        print(f"\nCONSERVATIVE ANALYSIS (n >= 50 only):")
        large_sample_states.sort(key=lambda x: abs(x['bias']), reverse=True)
        for state_info in large_sample_states[:5]:
            print(f"  {state_info['state']}: {state_info['bias']:+.3f} tier bias "
                  f"(n={state_info['count']}, CI=±{state_info['ci_95']:.3f})")
    
    print(f"\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    if reliable_count >= 3:
        print("✅ STATE BIAS VARIATIONS ARE STATISTICALLY RELIABLE")
        print(f"   {reliable_count} states show reliable bias patterns")
        print("   Geographic bias effects are real and substantial")
    elif len(bonferroni_significant) >= 2:
        print("⚠️  STATE BIAS VARIATIONS ARE PARTIALLY RELIABLE") 
        print(f"   {len(bonferroni_significant)} states survive multiple testing correction")
        print("   Some geographic effects are real but sample sizes limit precision")
    else:
        print("❌ STATE BIAS VARIATIONS ARE NOT RELIABLY DETECTABLE")
        print("   Sample sizes too small for reliable state-level bias detection")
        print("   Observed variations may be statistical noise")
    
    return {
        'total_states': len(state_analysis),
        'reliable_states': reliable_count,
        'bonferroni_significant': len(bonferroni_significant),
        'fdr_significant': len(fdr_significant),
        'mean_sample_size': np.mean(counts),
        'largest_bias': max(state_analysis, key=lambda x: abs(x['bias']))
    }

if __name__ == "__main__":
    results = check_state_reliability()
    print(f"\nAnalysis complete: {results['reliable_states']}/{results['total_states']} states reliable")