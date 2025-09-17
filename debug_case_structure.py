#!/usr/bin/env python3
"""Debug case structure to understand the data organization"""

import json
from pathlib import Path
from collections import defaultdict

# Load the runs data
runs_file = Path('out/runs.jsonl')
data = []
with open(runs_file, 'r') as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

print(f"Total records loaded: {len(data)}")

# First, let's understand the structure - group by case_id
cases = defaultdict(lambda: {'variants': [], 'group_labels': []})
for record in data[:1000]:  # Look at first 1000 records
    case_id = record.get('case_id')
    variant = record.get('variant', 'unknown')
    group_label = record.get('group_label', 'unknown')

    cases[case_id]['variants'].append(variant)
    cases[case_id]['group_labels'].append(group_label)

print(f"\nTotal unique cases in first 1000 records: {len(cases)}")

# Analyze a few cases in detail
print("\nDetailed analysis of first 5 cases:")
for i, (case_id, case_data) in enumerate(list(cases.items())[:5]):
    print(f"\nCase {i+1}: {case_id}")

    # Count variants
    variant_counts = {}
    for v in case_data['variants']:
        variant_counts[v] = variant_counts.get(v, 0) + 1

    print(f"  Total records: {len(case_data['variants'])}")
    print(f"  Variant breakdown:")
    for v, count in sorted(variant_counts.items()):
        print(f"    {v}: {count}")

    # Check for baseline
    has_baseline = 'NC' in case_data['variants'] or 'baseline' in case_data['group_labels']
    print(f"  Has baseline: {has_baseline}")

    # Count unique group labels (excluding baseline)
    unique_groups = set(g for g in case_data['group_labels'] if g != 'baseline')
    print(f"  Unique persona groups: {len(unique_groups)}")

# Overall statistics
print("\n=== OVERALL STATISTICS (first 1000 records) ===")
cases_with_baseline = sum(1 for case_data in cases.values()
                          if 'NC' in case_data['variants'] or 'baseline' in case_data['group_labels'])
cases_with_G = sum(1 for case_data in cases.values() if 'G' in case_data['variants'])
cases_with_both = sum(1 for case_data in cases.values()
                      if ('NC' in case_data['variants'] or 'baseline' in case_data['group_labels'])
                      and 'G' in case_data['variants'])

print(f"Cases with baseline (NC): {cases_with_baseline}")
print(f"Cases with personas (G): {cases_with_G}")
print(f"Cases with BOTH baseline and personas: {cases_with_both}")

# Check the expected structure
print("\n=== EXPECTED STRUCTURE CHECK ===")
print("Each case should have:")
print("  - 1 baseline (NC) record")
print("  - 10 persona (G) records (one per persona)")
print("  - 10 strategy records (one per persona with a random strategy)")
print("  Total: 21 records per case")

# Verify this structure
for case_id, case_data in list(cases.items())[:3]:
    nc_count = case_data['variants'].count('NC')
    g_count = case_data['variants'].count('G')
    other_count = len(case_data['variants']) - nc_count - g_count
    print(f"\nCase {case_id}: NC={nc_count}, G={g_count}, Other={other_count}, Total={len(case_data['variants'])}")