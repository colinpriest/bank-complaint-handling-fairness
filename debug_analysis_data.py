#!/usr/bin/env python3
"""Debug why demographic analysis sections have missing data"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from fairness_analysis import AdvancedFairnessAnalyzer

# Initialize analyzer for nshot results
analyzer = AdvancedFairnessAnalyzer(results_dir="nshot_v2_results")
analyzer.sample_size = 100  # Use same as nshot analysis

# Load existing results
analyzer._load_existing_results()

print(f"Raw results loaded: {len(analyzer.raw_results) if analyzer.raw_results else 0}")
print(f"Persona results loaded: {len(analyzer.persona_results) if analyzer.persona_results else 0}")

# Check if raw_results has the right structure
if analyzer.raw_results:
    print("\nSample raw result structure:")
    sample = analyzer.raw_results[0] if analyzer.raw_results else None
    if sample:
        print(f"  Keys: {list(sample.keys())}")
        print(f"  case_id: {sample.get('case_id')}")
        print(f"  variant: {sample.get('variant')}")
        print(f"  group_label: {sample.get('group_label')}")
        print(f"  remedy_tier: {sample.get('remedy_tier')}")

# Try to run the demographic injection analysis
print("\n=== Testing demographic_injection_effect analysis ===")
result = analyzer.statistical_analyzer.analyze_demographic_injection_effect(analyzer.raw_results)
print(f"Finding 1: {result.get('finding_1')}")
print(f"Finding 2: {result.get('finding_2')}")
print(f"Baseline mean: {result.get('baseline_mean')}")
print(f"Personas mean: {result.get('personas_mean')}")

# Try gender analysis
print("\n=== Testing gender_effects analysis ===")
result = analyzer.statistical_analyzer.analyze_gender_effects(analyzer.raw_results)
print(f"Finding: {result.get('finding')}")
print(f"Error: {result.get('error')}")

# Check what files exist in nshot_v2_results
print("\n=== Files in nshot_v2_results ===")
results_dir = Path("nshot_v2_results")
if results_dir.exists():
    for file in results_dir.iterdir():
        if file.is_file():
            print(f"  {file.name}: {file.stat().st_size} bytes")