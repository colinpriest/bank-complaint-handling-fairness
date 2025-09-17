#!/usr/bin/env python3
"""Debug severity bias demographic analysis - check data parsing"""

import json
from pathlib import Path
import numpy as np

# Load the runs data
runs_file = Path('out/runs.jsonl')
data = []
with open(runs_file, 'r') as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

print(f"Total records loaded: {len(data)}")

# Organize results by case ID to match baseline with persona results
complaint_data = {}
for result in data:
    case_id = result.get('case_id')
    if not case_id:
        continue

    if case_id not in complaint_data:
        complaint_data[case_id] = {'baseline': None, 'personas': []}

    # Identify baseline vs persona results
    variant = result.get('variant', '')
    group_label = result.get('group_label', '')

    # Check if this is a baseline result
    if variant == 'NC' or 'baseline' in group_label.lower() or group_label == 'baseline':
        complaint_data[case_id]['baseline'] = result
    else:
        # This is a persona result
        complaint_data[case_id]['personas'].append(result)

print(f"Total complaints: {len(complaint_data)}")

# Group by baseline remediation tier
tier_groups = {}
for case_id, data in complaint_data.items():
    baseline = data.get('baseline')
    personas = data.get('personas', [])

    if not baseline or not personas:
        continue

    baseline_tier = baseline.get('remedy_tier')
    if baseline_tier is None:
        continue

    # Convert to string for consistent grouping
    tier_key = str(baseline_tier)
    if tier_key not in tier_groups:
        tier_groups[tier_key] = []

    # Add persona results for this baseline tier
    for persona in personas:
        persona_tier = persona.get('remedy_tier')
        if persona_tier is not None:
            tier_groups[tier_key].append({
                'baseline_tier': baseline_tier,
                'persona_tier': persona_tier,
                'group': persona.get('group_label', 'unknown'),
                'variant': persona.get('variant', '')
            })

print(f"\nTier groups: {list(tier_groups.keys())}")
for tier, results in tier_groups.items():
    print(f"  Tier {tier}: {len(results)} persona results")

# Now test demographic parsing
gender_non_monetary = {'male': [], 'female': []}
gender_monetary = {'male': [], 'female': []}
ethnicity_keys = ['asian', 'black', 'latino', 'white']
ethnicity_non_monetary = {k: [] for k in ethnicity_keys}
ethnicity_monetary = {k: [] for k in ethnicity_keys}
geography_keys = ['urban_affluent', 'urban_poor', 'rural']
geography_non_monetary = {k: [] for k in geography_keys}
geography_monetary = {k: [] for k in geography_keys}

# Build gender-, ethnicity-, and geography-specific bias lists from original results
for tier, results in tier_groups.items():
    try:
        tier_num = int(tier)
    except Exception:
        continue
    is_non_monetary = tier_num in [0, 1]
    is_monetary = tier_num in [2, 3, 4]

    if not (is_non_monetary or is_monetary):
        continue

    for result in results:
        gl = (result.get('group') or '').lower()

        # Parse demographics from structured group labels
        # Format: ethnicity_gender_geography (e.g., "asian_male_rural")
        parts = gl.split('_')

        # Parse gender
        gender = None
        if 'male' in parts:
            gender = 'male'
        elif 'female' in parts:
            gender = 'female'

        # Parse ethnicity
        ethnicity = None
        for eth in ethnicity_keys:
            if eth in parts:
                ethnicity = eth
                break

        # Parse geography
        geography = None
        if 'affluent' in gl:
            geography = 'urban_affluent'
        elif 'poor' in gl:
            geography = 'urban_poor'
        elif 'rural' in parts:
            geography = 'rural'

        # Calculate bias
        bias = result.get('persona_tier', 0) - result.get('baseline_tier', 0)

        # Add to appropriate lists
        if is_non_monetary:
            if gender is not None:
                gender_non_monetary[gender].append(bias)
            if ethnicity is not None:
                ethnicity_non_monetary[ethnicity].append(bias)
            if geography is not None:
                geography_non_monetary[geography].append(bias)
        else:  # is_monetary
            if gender is not None:
                gender_monetary[gender].append(bias)
            if ethnicity is not None:
                ethnicity_monetary[ethnicity].append(bias)
            if geography is not None:
                geography_monetary[geography].append(bias)

# Print results
print("\n=== GENDER BIAS DATA ===")
for gender in ['male', 'female']:
    nm = gender_non_monetary[gender]
    m = gender_monetary[gender]
    print(f"{gender.capitalize()}:")
    print(f"  Non-Monetary: {len(nm)} samples, mean={np.mean(nm) if nm else 'N/A':.3f}")
    print(f"  Monetary: {len(m)} samples, mean={np.mean(m) if m else 'N/A':.3f}")

print("\n=== ETHNICITY BIAS DATA ===")
for eth in ethnicity_keys:
    nm = ethnicity_non_monetary[eth]
    m = ethnicity_monetary[eth]
    print(f"{eth.capitalize()}:")
    print(f"  Non-Monetary: {len(nm)} samples, mean={np.mean(nm) if nm else 'N/A':.3f}")
    print(f"  Monetary: {len(m)} samples, mean={np.mean(m) if m else 'N/A':.3f}")

print("\n=== GEOGRAPHY BIAS DATA ===")
for geo in geography_keys:
    nm = geography_non_monetary[geo]
    m = geography_monetary[geo]
    geo_display = geo.replace('_', ' ').title()
    print(f"{geo_display}:")
    print(f"  Non-Monetary: {len(nm)} samples, mean={np.mean(nm) if nm else 'N/A':.3f}")
    print(f"  Monetary: {len(m)} samples, mean={np.mean(m) if m else 'N/A':.3f}")