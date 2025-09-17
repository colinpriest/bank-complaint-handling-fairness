#!/usr/bin/env python3
"""Debug script to check DPP analysis data"""

import json
from pathlib import Path

results_file = Path("nshot_v2_results/runs.jsonl")

if results_file.exists():
    raw_results = []
    with open(results_file, 'r') as f:
        for line in f:
            raw_results.append(json.loads(line))

    print(f"Total results: {len(raw_results)}")

    # Check comparison group data
    comparison_results = [r for r in raw_results if r.get('comparison_group') == 'dpp_vs_nn']
    print(f"Comparison group results: {len(comparison_results)}")

    if comparison_results:
        dpp_results = [r for r in comparison_results if r.get('use_dpp', False)]
        nn_results = [r for r in comparison_results if not r.get('use_dpp', False)]

        print(f"DPP results: {len(dpp_results)}")
        print(f"NN results: {len(nn_results)}")

        if dpp_results and nn_results:
            dpp_tiers = [r['remedy_tier'] for r in dpp_results]
            nn_tiers = [r['remedy_tier'] for r in nn_results]

            print(f"DPP mean tier: {sum(dpp_tiers)/len(dpp_tiers):.3f}")
            print(f"NN mean tier: {sum(nn_tiers)/len(nn_tiers):.3f}")
            print("SUCCESS: Both DPP and NN data found!")
        else:
            print("ISSUE: Missing DPP or NN data in comparison group")
    else:
        print("ISSUE: No comparison group data found")

        # Fallback check
        all_dpp = [r for r in raw_results if r.get('use_dpp', False)]
        all_nn = [r for r in raw_results if not r.get('use_dpp', False)]
        print(f"All DPP results: {len(all_dpp)}")
        print(f"All NN results: {len(all_nn)}")
else:
    print("Results file not found")