#!/usr/bin/env python3
"""Test the updated DPP analysis function with accuracy"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from scipy.stats import ttest_ind

def _analyze_dpp_effectiveness(raw_results: List[Dict]) -> Dict[str, Any]:
    """Analyze effectiveness of DPP vs nearest neighbor selection with accuracy"""

    print(f"[DEBUG] Total results: {len(raw_results)}")

    # First look for comparison group data
    comparison_results = [r for r in raw_results if r.get('comparison_group') == 'dpp_vs_nn']
    print(f"[DEBUG] Comparison results: {len(comparison_results)}")

    if comparison_results:
        dpp_results = [r for r in comparison_results if r.get('use_dpp', False)]
        nn_results = [r for r in comparison_results if not r.get('use_dpp', False)]
        print(f"[DEBUG] Using comparison group: DPP={len(dpp_results)}, NN={len(nn_results)}")
    else:
        dpp_results = [r for r in raw_results if r.get('use_dpp', False)]
        nn_results = [r for r in raw_results if not r.get('use_dpp', False)]
        print(f"[DEBUG] Using all results: DPP={len(dpp_results)}, NN={len(nn_results)}")

    if not dpp_results or not nn_results:
        print("[DEBUG] Insufficient data!")
        return {"finding": "INSUFFICIENT DATA"}

    # Compare remedy tiers
    dpp_tiers = [r['remedy_tier'] for r in dpp_results]
    nn_tiers = [r['remedy_tier'] for r in nn_results]
    t_stat, p_value = ttest_ind(dpp_tiers, nn_tiers, equal_var=False)

    print(f"[DEBUG] DPP tier mean: {np.mean(dpp_tiers):.3f}")
    print(f"[DEBUG] NN tier mean: {np.mean(nn_tiers):.3f}")

    # Compare accuracy
    dpp_accuracy_data = []
    nn_accuracy_data = []

    for r in dpp_results:
        if 'baseline_tier' in r and 'remedy_tier' in r:
            exact_match = 1 if r['baseline_tier'] == r['remedy_tier'] else 0
            dpp_accuracy_data.append(exact_match)

    for r in nn_results:
        if 'baseline_tier' in r and 'remedy_tier' in r:
            exact_match = 1 if r['baseline_tier'] == r['remedy_tier'] else 0
            nn_accuracy_data.append(exact_match)

    print(f"[DEBUG] DPP accuracy data points: {len(dpp_accuracy_data)}")
    print(f"[DEBUG] NN accuracy data points: {len(nn_accuracy_data)}")

    accuracy_results = {}
    if dpp_accuracy_data and nn_accuracy_data:
        dpp_accuracy = float(np.mean(dpp_accuracy_data))
        nn_accuracy = float(np.mean(nn_accuracy_data))

        print(f"[DEBUG] DPP accuracy: {dpp_accuracy:.1%}")
        print(f"[DEBUG] NN accuracy: {nn_accuracy:.1%}")

        acc_t_stat, acc_p_value = ttest_ind(dpp_accuracy_data, nn_accuracy_data, equal_var=False)

        accuracy_results = {
            "dpp_accuracy": dpp_accuracy,
            "nn_accuracy": nn_accuracy,
            "accuracy_t_statistic": float(acc_t_stat),
            "accuracy_p_value": float(acc_p_value),
            "accuracy_finding": "DPP MORE ACCURATE" if acc_p_value < 0.05 and dpp_accuracy > nn_accuracy
                               else "NN MORE ACCURATE" if acc_p_value < 0.05 and nn_accuracy > dpp_accuracy
                               else "NO ACCURACY DIFFERENCE",
            "accuracy_interpretation": f"DPP accuracy: {dpp_accuracy:.1%}, NN accuracy: {nn_accuracy:.1%} (p={acc_p_value:.4f})"
        }

    # Determine overall finding
    tier_significant = p_value < 0.05
    accuracy_significant = accuracy_results.get('accuracy_p_value', 1) < 0.05 if accuracy_results else False

    if tier_significant or accuracy_significant:
        if tier_significant and accuracy_significant:
            finding = "DPP AFFECTS BOTH TIERS AND ACCURACY"
        elif tier_significant:
            finding = "DPP AFFECTS TIERS ONLY"
        else:
            finding = "DPP AFFECTS ACCURACY ONLY"
    else:
        finding = "NO DIFFERENCE"

    result = {
        "finding": finding,
        "dpp_mean": float(np.mean(dpp_tiers)),
        "nn_mean": float(np.mean(nn_tiers)),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "interpretation": f"DPP selection {'significantly affects' if p_value < 0.05 else 'does not affect'} remedy tiers (p={p_value:.4f})"
    }

    result.update(accuracy_results)
    return result

# Test with real data
results_file = Path("nshot_v2_results/runs.jsonl")
if results_file.exists():
    raw_results = []
    with open(results_file, 'r') as f:
        for line in f:
            raw_results.append(json.loads(line))

    result = _analyze_dpp_effectiveness(raw_results)
    print(f"\n=== FINAL RESULT ===")
    for key, value in result.items():
        print(f"{key}: {value}")
else:
    print("Results file not found")