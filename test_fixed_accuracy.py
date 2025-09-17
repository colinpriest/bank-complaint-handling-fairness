#!/usr/bin/env python3
"""Test DPP analysis with fixed case ID matching"""

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

def analyze_dpp_with_fixed_accuracy(raw_results: List[Dict]) -> Dict[str, Any]:
    """Test the DPP analysis with fixed case ID matching"""

    print(f"[DEBUG] Total results: {len(raw_results)}")

    # Get comparison group data
    comparison_results = [r for r in raw_results if r.get('comparison_group') == 'dpp_vs_nn']
    dpp_results = [r for r in comparison_results if r.get('use_dpp', False)]
    nn_results = [r for r in comparison_results if not r.get('use_dpp', False)]
    print(f"[DEBUG] DPP={len(dpp_results)}, NN={len(nn_results)}")

    # Build baseline lookup with fixed matching
    baseline_lookup = {}
    all_baseline = [r for r in raw_results if r.get('group_label') == 'baseline']

    for baseline in all_baseline:
        case_id = baseline.get('case_id', '')
        if case_id.startswith('nshot_'):
            try:
                index = case_id.split('_')[1]
                baseline_lookup[index] = baseline.get('remedy_tier')
                print(f"[DEBUG] Baseline: {case_id} -> index {index}, tier {baseline.get('remedy_tier')}")
            except:
                pass

    print(f"[DEBUG] Baseline lookup size: {len(baseline_lookup)}")

    # Test accuracy calculation
    dpp_accuracy_data = []
    for r in dpp_results:
        case_id = r.get('case_id', '')
        if case_id.startswith('nshot_comparison_'):
            try:
                index = case_id.split('_')[2]
                if index in baseline_lookup:
                    baseline_tier = baseline_lookup[index]
                    baseline_collapsed = collapse_tier(baseline_tier)
                    prediction_collapsed = collapse_tier(r['remedy_tier'])
                    exact_match = 1 if baseline_collapsed == prediction_collapsed else 0
                    dpp_accuracy_data.append(exact_match)
                    print(f"[DEBUG] DPP Match: {case_id} -> index {index}, baseline={baseline_tier}->{baseline_collapsed}, pred={r['remedy_tier']}->{prediction_collapsed}, match={exact_match}")
                else:
                    print(f"[DEBUG] DPP No match: {case_id} -> index {index} not in baseline")
            except Exception as e:
                print(f"[DEBUG] DPP Error: {case_id} -> {e}")

    nn_accuracy_data = []
    for r in nn_results:
        case_id = r.get('case_id', '')
        if case_id.startswith('nshot_comparison_'):
            try:
                index = case_id.split('_')[2]
                if index in baseline_lookup:
                    baseline_tier = baseline_lookup[index]
                    baseline_collapsed = collapse_tier(baseline_tier)
                    prediction_collapsed = collapse_tier(r['remedy_tier'])
                    exact_match = 1 if baseline_collapsed == prediction_collapsed else 0
                    nn_accuracy_data.append(exact_match)
                    print(f"[DEBUG] NN Match: {case_id} -> index {index}, baseline={baseline_tier}->{baseline_collapsed}, pred={r['remedy_tier']}->{prediction_collapsed}, match={exact_match}")
            except Exception as e:
                print(f"[DEBUG] NN Error: {case_id} -> {e}")

    print(f"[DEBUG] DPP accuracy data points: {len(dpp_accuracy_data)}")
    print(f"[DEBUG] NN accuracy data points: {len(nn_accuracy_data)}")

    if dpp_accuracy_data and nn_accuracy_data:
        dpp_accuracy = float(np.mean(dpp_accuracy_data))
        nn_accuracy = float(np.mean(nn_accuracy_data))

        print(f"[DEBUG] DPP accuracy: {dpp_accuracy:.1%}")
        print(f"[DEBUG] NN accuracy: {nn_accuracy:.1%}")

        return {
            "dpp_accuracy": dpp_accuracy,
            "nn_accuracy": nn_accuracy,
            "success": True
        }
    else:
        return {"success": False}

# Test
results_file = Path("nshot_v2_results/runs.jsonl")
if results_file.exists():
    raw_results = []
    with open(results_file, 'r') as f:
        for line in f:
            raw_results.append(json.loads(line))

    result = analyze_dpp_with_fixed_accuracy(raw_results)
    print(f"\n=== FINAL RESULT ===")
    for key, value in result.items():
        print(f"{key}: {value}")
else:
    print("Results file not found")