#!/usr/bin/env python3
"""Debug filtered data to see what records are available after sampling"""

import json
from pathlib import Path

# Load the runs data
runs_file = Path('out/runs.jsonl')
data = []
with open(runs_file, 'r') as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

# Simulate filtering to sample size 100 (just like the analyzer does)
sample_size = 100
filtered_data = data[:sample_size]

print(f"Total records: {len(data)}")
print(f"Filtered to: {len(filtered_data)} records")

# Check what we have in the filtered data
unique_case_ids = set()
unique_groups = set()
unique_variants = set()

for record in filtered_data:
    unique_case_ids.add(record.get('case_id'))
    unique_groups.add(record.get('group_label'))
    unique_variants.add(record.get('variant'))

print(f"\nUnique case IDs: {len(unique_case_ids)}")
print(f"Unique groups: {len(unique_groups)}")
print(f"Groups: {sorted(unique_groups)}")
print(f"Unique variants: {sorted(unique_variants)}")

# Check how many records per case
from collections import Counter
case_counter = Counter(r.get('case_id') for r in filtered_data)
print(f"\nRecords per case:")
for case_id, count in list(case_counter.items())[:5]:
    print(f"  {case_id}: {count} records")