#!/usr/bin/env python3
"""Debug severity bias analysis with 1000 samples"""

import json
from pathlib import Path
import sys
import numpy as np

# Add the fairness_analysis package to the path
sys.path.insert(0, str(Path(__file__).parent))

from fairness_analysis import AdvancedFairnessAnalyzer

# Initialize analyzer
analyzer = AdvancedFairnessAnalyzer(results_dir="gpt_4o_mini_results")
analyzer.sample_size = 1000

# Load existing results
analyzer._load_existing_results()

print(f"Total raw results loaded: {len(analyzer.raw_results)}")

# Manually run the severity bias analysis logic
raw_results = analyzer.raw_results

# Organize results by case ID
complaint_data = {}
for result in raw_results:
    case_id = result.get('case_id')
    if not case_id:
        continue

    if case_id not in complaint_data:
        complaint_data[case_id] = {'baseline': None, 'personas': []}

    variant = result.get('variant', '')
    group_label = result.get('group_label', '')

    if variant == 'NC' or 'baseline' in group_label.lower() or group_label == 'baseline':
        complaint_data[case_id]['baseline'] = result
    else:
        complaint_data[case_id]['personas'].append(result)

print(f"Total unique cases: {len(complaint_data)}")

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

    tier_key = str(baseline_tier)
    if tier_key not in tier_groups:
        tier_groups[tier_key] = []

    for persona in personas:
        persona_tier = persona.get('remedy_tier')
        if persona_tier is not None:
            tier_groups[tier_key].append({
                'baseline_tier': baseline_tier,
                'persona_tier': persona_tier,
                'group': persona.get('group_label', 'unknown')
            })

print(f"Tier groups: {list(tier_groups.keys())}")
for tier, results in tier_groups.items():
    print(f"  Tier {tier}: {len(results)} persona results")

# Test demographic parsing
ethnicity_keys = ['asian', 'black', 'latino', 'white']
gender_non_monetary = {'male': [], 'female': []}
gender_monetary = {'male': [], 'female': []}
ethnicity_non_monetary = {k: [] for k in ethnicity_keys}
ethnicity_monetary = {k: [] for k in ethnicity_keys}
geography_keys = ['urban_affluent', 'urban_poor', 'rural']
geography_non_monetary = {k: [] for k in geography_keys}
geography_monetary = {k: [] for k in geography_keys}

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
        parts = gl.split('_')

        # Parse gender - check for 'female' first
        gender = None
        if 'female' in parts:
            gender = 'female'
        elif 'male' in parts:
            gender = 'male'

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

        bias = result.get('persona_tier', 0) - result.get('baseline_tier', 0)

        if is_non_monetary:
            if gender is not None:
                gender_non_monetary[gender].append(bias)
            if ethnicity is not None:
                ethnicity_non_monetary[ethnicity].append(bias)
            if geography is not None:
                geography_non_monetary[geography].append(bias)
        else:
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
    print(f"  Non-Monetary: {len(nm)} samples")
    print(f"  Monetary: {len(m)} samples")

print("\n=== ETHNICITY BIAS DATA ===")
for eth in ethnicity_keys:
    nm = ethnicity_non_monetary[eth]
    m = ethnicity_monetary[eth]
    print(f"{eth.capitalize()}:")
    print(f"  Non-Monetary: {len(nm)} samples")
    print(f"  Monetary: {len(m)} samples")

print("\n=== GEOGRAPHY BIAS DATA ===")
for geo in geography_keys:
    nm = geography_non_monetary[geo]
    m = geography_monetary[geo]
    geo_display = geo.replace('_', ' ').title()
    print(f"{geo_display}:")
    print(f"  Non-Monetary: {len(nm)} samples")
    print(f"  Monetary: {len(m)} samples")