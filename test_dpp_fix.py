#!/usr/bin/env python3
"""Quick test to verify DPP vs NN fix works"""

import json
from pathlib import Path

# Check if runs.jsonl exists and contains both DPP and NN results
results_file = Path("nshot_v2_results/runs.jsonl")

if results_file.exists():
    dpp_count = 0
    nn_count = 0

    with open(results_file, 'r') as f:
        for line in f:
            try:
                result = json.loads(line)
                if result.get('use_dpp', False):
                    dpp_count += 1
                else:
                    nn_count += 1
            except:
                pass

    print(f"DPP results (use_dpp=True): {dpp_count}")
    print(f"NN results (use_dpp=False): {nn_count}")

    if dpp_count > 0 and nn_count > 0:
        print("\n✓ SUCCESS: Both DPP and NN results found!")
        print(f"DPP/NN ratio: {dpp_count/nn_count:.2f}")
    elif dpp_count > 0 and nn_count == 0:
        print("\n✗ ISSUE: Only DPP results found, no NN results")
        print("The bug is NOT fixed - experiments should run both methods")
    elif nn_count > 0 and dpp_count == 0:
        print("\n✗ ISSUE: Only NN results found, no DPP results")
    else:
        print("\n✗ No results found")
else:
    print(f"Results file not found at {results_file}")
    print("Run experiment first: python nshot_fairness_analysis_V2.py --experiment-only --samples 5")