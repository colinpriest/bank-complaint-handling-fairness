#!/usr/bin/env python3
"""Debug severity bias analysis results"""

import json
from pathlib import Path
import sys

# Add the fairness_analysis package to the path
sys.path.insert(0, str(Path(__file__).parent))

from fairness_analysis import AdvancedFairnessAnalyzer

# Initialize analyzer
analyzer = AdvancedFairnessAnalyzer(results_dir="gpt_4o_mini_results")
analyzer.sample_size = 100

# Load existing results
analyzer._load_existing_results()

# Run just the severity bias variation analysis
severity_analysis = analyzer.statistical_analyzer.analyze_severity_bias_variation(analyzer.raw_results)

# Check what keys are in the result
print("Keys in severity analysis result:")
for key in severity_analysis.keys():
    print(f"  - {key}")

# Check if the demographic test results exist
if 'gender_monetary_bias_test' in severity_analysis:
    print("\nGender monetary bias test found!")
    print(json.dumps(severity_analysis['gender_monetary_bias_test'], indent=2))
else:
    print("\nGender monetary bias test NOT found!")

if 'ethnicity_monetary_bias_test' in severity_analysis:
    print("\nEthnicity monetary bias test found!")
    print(json.dumps(severity_analysis['ethnicity_monetary_bias_test'], indent=2))
else:
    print("\nEthnicity monetary bias test NOT found!")

if 'geography_monetary_bias_test' in severity_analysis:
    print("\nGeography monetary bias test found!")
    print(json.dumps(severity_analysis['geography_monetary_bias_test'], indent=2))
else:
    print("\nGeography monetary bias test NOT found!")

# Print the tier stats
print("\nTier stats available:")
if 'tier_stats' in severity_analysis:
    for tier, stats in severity_analysis['tier_stats'].items():
        print(f"  Tier {tier}: {stats['sample_size']} samples")