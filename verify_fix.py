#!/usr/bin/env python3
import json

# Load the test data
with open('test_all_personas/pairs.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

print(f'Total records: {len(data)}')

# Count baseline records
baseline = [r for r in data if r['variant'] == 'NC']
print(f'Baseline records: {len(baseline)}')

# Count persona records
personas = [r for r in data if r['variant'] == 'G']
print(f'Persona records: {len(personas)}')

# Check all variants
variants = set(r['variant'] for r in data)
print(f'Variants: {sorted(variants)}')
for v in sorted(variants):
    count = len([r for r in data if r['variant'] == v])
    print(f'  {v}: {count} records')

# Count unique case_ids
unique_case_ids = set(r['case_id'] for r in data)
print(f'Unique case_ids: {len(unique_case_ids)}')

# Calculate personas per complaint
personas_per_complaint = len(personas) / len(baseline)
print(f'Personas per complaint: {personas_per_complaint:.1f}')

# Check group_label distribution
group_labels = set(r['group_label'] for r in data)
print(f'Group labels: {sorted(group_labels)}')
for gl in sorted(group_labels):
    count = len([r for r in data if r['group_label'] == gl])
    print(f'  {gl}: {count} records')

# Check if each complaint has all 3 personas
case_id_personas = {}
for record in data:
    if record['variant'] == 'G':  # Only persona records
        case_id = record['case_id']
        group_label = record['group_label']
        if case_id not in case_id_personas:
            case_id_personas[case_id] = set()
        case_id_personas[case_id].add(group_label)

print(f'\nPersona coverage per complaint:')
for case_id, personas_set in list(case_id_personas.items())[:5]:  # Show first 5
    print(f'  {case_id}: {sorted(personas_set)}')

# Check if all complaints have all 3 personas
all_have_all_personas = all(len(personas_set) == 3 for personas_set in case_id_personas.values())
print(f'\nAll complaints have all 3 personas: {all_have_all_personas}')
