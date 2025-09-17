#!/usr/bin/env python3
"""Test DPP analysis with 3-tier accuracy"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from scipy.stats import ttest_ind

def collapse_tier(tier):
    """Collapse to 3-tier system: No Action, Non-Monetary, Monetary"""
    if tier == 0:
        return "No Action"
    elif tier == 1:
        return "Non-Monetary"
    elif tier in [2, 3, 4]:
        return "Monetary"
    else:
        return "Unknown"

def analyze_dpp_with_accuracy(raw_results: List[Dict]) -> Dict[str, Any]:
    """Test the DPP analysis with 3-tier accuracy"""

    print(f"[DEBUG] Total results: {len(raw_results)}")

    # Get comparison group data
    comparison_results = [r for r in raw_results if r.get('comparison_group') == 'dpp_vs_nn']
    print(f"[DEBUG] Comparison results: {len(comparison_results)}")

    if not comparison_results:
        return {"finding": "INSUFFICIENT DATA"}

    dpp_results = [r for r in comparison_results if r.get('use_dpp', False)]
    nn_results = [r for r in comparison_results if not r.get('use_dpp', False)]
    print(f"[DEBUG] DPP={len(dpp_results)}, NN={len(nn_results)}")

    # Build baseline lookup
    baseline_lookup = {}
    all_baseline = [r for r in raw_results if r.get('group_label') == 'baseline']
    print(f"[DEBUG] Baseline results: {len(all_baseline)}")

    for baseline in all_baseline:
        case_id = baseline.get('case_id', '')
        if case_id:
            baseline_lookup[case_id] = baseline.get('remedy_tier')

    print(f"[DEBUG] Baseline lookup size: {len(baseline_lookup)}")

    # Calculate accuracy for DPP
    dpp_accuracy_data = []
    dpp_matches_found = 0

    for r in dpp_results:
        case_id = r.get('case_id', '')
        print(f"[DEBUG] DPP case_id: {case_id}")
        if case_id in baseline_lookup:
            dpp_matches_found += 1
            baseline_tier = baseline_lookup[case_id]
            baseline_collapsed = collapse_tier(baseline_tier)
            prediction_collapsed = collapse_tier(r['remedy_tier'])
            exact_match = 1 if baseline_collapsed == prediction_collapsed else 0
            dpp_accuracy_data.append(exact_match)
            print(f"[DEBUG] DPP: baseline={baseline_tier}->{baseline_collapsed}, pred={r['remedy_tier']}->{prediction_collapsed}, match={exact_match}")

    # Calculate accuracy for NN
    nn_accuracy_data = []
    nn_matches_found = 0

    for r in nn_results:
        case_id = r.get('case_id', '')
        if case_id in baseline_lookup:
            nn_matches_found += 1
            baseline_tier = baseline_lookup[case_id]
            baseline_collapsed = collapse_tier(baseline_tier)
            prediction_collapsed = collapse_tier(r['remedy_tier'])
            exact_match = 1 if baseline_collapsed == prediction_collapsed else 0
            nn_accuracy_data.append(exact_match)

    print(f"[DEBUG] DPP matches found: {dpp_matches_found}, accuracy data: {len(dpp_accuracy_data)}")
    print(f"[DEBUG] NN matches found: {nn_matches_found}, accuracy data: {len(nn_accuracy_data)}")

    if dpp_accuracy_data and nn_accuracy_data:
        dpp_accuracy = float(np.mean(dpp_accuracy_data))
        nn_accuracy = float(np.mean(nn_accuracy_data))

        print(f"[DEBUG] DPP accuracy: {dpp_accuracy:.1%}")
        print(f"[DEBUG] NN accuracy: {nn_accuracy:.1%}")

        acc_t_stat, acc_p_value = ttest_ind(dpp_accuracy_data, nn_accuracy_data, equal_var=False)

        return {
            "dpp_accuracy": dpp_accuracy,
            "nn_accuracy": nn_accuracy,
            "accuracy_p_value": acc_p_value,
            "accuracy_interpretation": f"DPP accuracy: {dpp_accuracy:.1%}, NN accuracy: {nn_accuracy:.1%} (p={acc_p_value:.4f})"
        }
    else:
        print("[DEBUG] No accuracy data found")
        return {"finding": "NO ACCURACY DATA"}

# Test
results_file = Path("nshot_v2_results/runs.jsonl")
if results_file.exists():
    raw_results = []
    with open(results_file, 'r') as f:
        for line in f:
            raw_results.append(json.loads(line))

    result = analyze_dpp_with_accuracy(raw_results)
    print(f"\n=== RESULT ===")
    for key, value in result.items():
        print(f"{key}: {value}")
else:
    print("Results file not found")