#!/usr/bin/env python3
"""Test that severity bias analysis properly filters variants"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from fairness_analysis import AdvancedFairnessAnalyzer

# Initialize analyzer
analyzer = AdvancedFairnessAnalyzer(results_dir="gpt_4o_mini_results")
analyzer.sample_size = 1000

# Load existing results
analyzer._load_existing_results()

print(f"Total raw results: {len(analyzer.raw_results)}")

# Check what variants are in the data
variants_count = {}
for result in analyzer.raw_results:
    variant = result.get('variant', 'unknown')
    variants_count[variant] = variants_count.get(variant, 0) + 1

print("\nVariants in data:")
for variant, count in sorted(variants_count.items()):
    print(f"  {variant}: {count}")

# Now simulate the filtering logic from analyze_severity_bias_variation
complaint_data = {}
for result in analyzer.raw_results:
    case_id = result.get('case_id')
    if not case_id:
        continue

    if case_id not in complaint_data:
        complaint_data[case_id] = {'baseline': None, 'personas': []}

    variant = result.get('variant', '')
    group_label = result.get('group_label', '')

    # Check if this is a baseline result
    if variant == 'NC' or 'baseline' in group_label.lower() or group_label == 'baseline':
        complaint_data[case_id]['baseline'] = result
    elif variant == 'G':
        # This is a standard persona result (excluding bias mitigation strategies)
        complaint_data[case_id]['personas'].append(result)
    # Else: skip bias mitigation strategies

# Count personas per case
total_personas = 0
cases_with_personas = 0
for case_id, data in complaint_data.items():
    if data['personas']:
        cases_with_personas += 1
        total_personas += len(data['personas'])

print(f"\nFiltering results:")
print(f"  Total cases: {len(complaint_data)}")
print(f"  Cases with baseline: {sum(1 for d in complaint_data.values() if d['baseline'])}")
print(f"  Cases with personas (G only): {cases_with_personas}")
print(f"  Total persona records (G only): {total_personas}")
print(f"  Average personas per case: {total_personas/cases_with_personas if cases_with_personas > 0 else 0:.1f}")

# Check what group labels are in the filtered personas
group_labels = set()
for data in complaint_data.values():
    for persona in data['personas']:
        group_labels.add(persona.get('group_label', 'unknown'))

print(f"\nFiltered persona group labels (sample):")
for label in sorted(list(group_labels)[:10]):
    print(f"  {label}")