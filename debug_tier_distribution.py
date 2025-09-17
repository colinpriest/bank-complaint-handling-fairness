#!/usr/bin/env python3
"""Debug tier distribution in filtered data"""

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

# Simulate filtering to sample size 100
filtered_data = data[:100]

# Organize by case ID
cases = defaultdict(lambda: {'baseline': None, 'personas': []})
for record in filtered_data:
    case_id = record.get('case_id')
    variant = record.get('variant', '')
    group_label = record.get('group_label', '')

    if variant == 'NC' or 'baseline' in group_label.lower():
        cases[case_id]['baseline'] = record
    else:
        cases[case_id]['personas'].append(record)

# Check baseline tier distribution
print("Baseline tier distribution:")
tier_counts = defaultdict(int)
for case_id, case_data in cases.items():
    baseline = case_data['baseline']
    if baseline:
        tier = baseline.get('remedy_tier')
        tier_counts[tier] += 1
        print(f"  Case {case_id}: Tier {tier}")

print(f"\nTier summary:")
for tier, count in sorted(tier_counts.items()):
    tier_type = "Non-Monetary" if tier in [0, 1] else "Monetary" if tier in [2, 3, 4] else "Unknown"
    print(f"  Tier {tier} ({tier_type}): {count} cases")

# Check demographic distribution by tier
print("\nDemographic distribution by severity type:")
non_monetary_demos = defaultdict(int)
monetary_demos = defaultdict(int)

for case_id, case_data in cases.items():
    baseline = case_data['baseline']
    if not baseline:
        continue

    baseline_tier = baseline.get('remedy_tier')
    if baseline_tier is None:
        continue

    is_non_monetary = baseline_tier in [0, 1]
    is_monetary = baseline_tier in [2, 3, 4]

    for persona in case_data['personas']:
        group_label = persona.get('group_label', '').lower()
        parts = group_label.split('_')

        # Parse gender
        gender = None
        if 'female' in parts:
            gender = 'female'
        elif 'male' in parts:
            gender = 'male'

        # Parse ethnicity
        ethnicity = None
        for eth in ['asian', 'black', 'latino', 'white']:
            if eth in parts:
                ethnicity = eth
                break

        if is_non_monetary:
            if gender:
                non_monetary_demos[f"gender_{gender}"] += 1
            if ethnicity:
                non_monetary_demos[f"ethnicity_{ethnicity}"] += 1
        elif is_monetary:
            if gender:
                monetary_demos[f"gender_{gender}"] += 1
            if ethnicity:
                monetary_demos[f"ethnicity_{ethnicity}"] += 1

print("\nNon-Monetary demographics:")
for demo, count in sorted(non_monetary_demos.items()):
    print(f"  {demo}: {count}")

print("\nMonetary demographics:")
for demo, count in sorted(monetary_demos.items()):
    print(f"  {demo}: {count}")