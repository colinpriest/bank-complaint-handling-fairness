#!/usr/bin/env python3
"""Test the DPP analysis function directly"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

def _analyze_dpp_effectiveness(raw_results: List[Dict]) -> Dict[str, Any]:
    """Analyze effectiveness of DPP vs nearest neighbor selection"""

    print(f"[DEBUG] Total results: {len(raw_results)}")

    # First look for comparison group data (direct head-to-head comparison)
    comparison_results = [r for r in raw_results if r.get('comparison_group') == 'dpp_vs_nn']
    print(f"[DEBUG] Comparison results: {len(comparison_results)}")

    if comparison_results:
        # Use the comparison group for analysis
        dpp_results = [r for r in comparison_results if r.get('use_dpp', False)]
        nn_results = [r for r in comparison_results if not r.get('use_dpp', False)]
        print(f"[DEBUG] Using comparison group: DPP={len(dpp_results)}, NN={len(nn_results)}")
    else:
        # Fallback to all results if no comparison group
        dpp_results = [r for r in raw_results if r.get('use_dpp', False)]
        nn_results = [r for r in raw_results if not r.get('use_dpp', False)]
        print(f"[DEBUG] Using all results: DPP={len(dpp_results)}, NN={len(nn_results)}")

    if not dpp_results or not nn_results:
        print("[DEBUG] Insufficient data!")
        return {
            "finding": "INSUFFICIENT DATA",
            "interpretation": "Need both DPP and NN results for comparison. Run with sufficient samples to enable comparison."
        }

    # Compare bias metrics
    dpp_tiers = [r['remedy_tier'] for r in dpp_results]
    nn_tiers = [r['remedy_tier'] for r in nn_results]

    print(f"[DEBUG] DPP mean: {np.mean(dpp_tiers):.3f}")
    print(f"[DEBUG] NN mean: {np.mean(nn_tiers):.3f}")

    # Statistical test
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(dpp_tiers, nn_tiers, equal_var=False)

    result = {
        "finding": "DPP EFFECTIVE" if p_value < 0.05 else "NO DIFFERENCE",
        "dpp_mean": float(np.mean(dpp_tiers)),
        "nn_mean": float(np.mean(nn_tiers)),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "interpretation": f"DPP selection {'significantly affects' if p_value < 0.05 else 'does not affect'} outcomes (p={p_value:.4f})"
    }

    print(f"[DEBUG] Result: {result}")
    return result

# Load and test
results_file = Path("nshot_v2_results/runs.jsonl")
if results_file.exists():
    raw_results = []
    with open(results_file, 'r') as f:
        for line in f:
            raw_results.append(json.loads(line))

    result = _analyze_dpp_effectiveness(raw_results)
    print(f"\nFinal result: {result}")
else:
    print("Results file not found")