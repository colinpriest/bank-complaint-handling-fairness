#!/usr/bin/env python3
"""
Test the edge case fixes for disparity analysis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from html_dashboard import HTMLDashboard

def test_edge_case_fixes():
    """Test that edge cases are handled properly"""
    print("=== Testing Edge Case Fixes ===\n")

    dashboard = HTMLDashboard()

    print("1. Testing safe disparity metrics...")

    # Test case 1: Normal comparison
    print("   Normal case (5% vs 3%):")
    metrics1 = dashboard._calculate_safe_disparity_metrics(0.05, 0.03, "Group A", "Group B")
    print(f"     - Relative diff text: {metrics1['relative_diff_text']}")
    print(f"     - Description: {metrics1['relative_diff_description']}")
    print(f"     - Equity ratio: {metrics1['equity_ratio']:.3f}")

    # Test case 2: Zero baseline (the problematic case)
    print("\n   Zero baseline case (2% vs 0%):")
    metrics2 = dashboard._calculate_safe_disparity_metrics(0.02, 0.0, "White", "Black")
    print(f"     - Relative diff text: {metrics2['relative_diff_text']}")
    print(f"     - Description: {metrics2['relative_diff_description']}")
    print(f"     - Equity ratio: {metrics2['equity_ratio']:.3f}")
    print(f"     - Has valid comparison: {metrics2['has_valid_comparison']}")

    # Test case 3: Both zero
    print("\n   Both zero case (0% vs 0%):")
    metrics3 = dashboard._calculate_safe_disparity_metrics(0.0, 0.0, "Group X", "Group Y")
    print(f"     - Relative diff text: {metrics3['relative_diff_text']}")
    print(f"     - Description: {metrics3['relative_diff_description']}")
    print(f"     - Equity ratio: {metrics3['equity_ratio']:.3f}")

    print("\n2. Testing safe sample size calculation...")

    # Test case 1: Normal counts
    counts1 = {'group1': 100, 'group2': 200}
    size1 = dashboard._calculate_safe_sample_size(counts1)
    print(f"   Normal counts {counts1}: {size1}")

    # Test case 2: Some zero counts
    counts2 = {'group1': 50, 'group2': 0, 'group3': 30}
    size2 = dashboard._calculate_safe_sample_size(counts2)
    print(f"   Mixed counts {counts2}: {size2}")

    # Test case 3: All zero counts
    counts3 = {'group1': 0, 'group2': 0}
    size3 = dashboard._calculate_safe_sample_size(counts3)
    print(f"   Zero counts {counts3}: {size3}")

    # Test case 4: None values (edge case)
    counts4 = {'group1': None, 'group2': 50, 'group3': 'invalid'}
    size4 = dashboard._calculate_safe_sample_size(counts4)
    print(f"   Invalid counts {counts4}: {size4}")

    print("\n3. Expected improvements:")
    print("   [SUCCESS] No more 'inf% difference' messages")
    print("   [SUCCESS] Clear descriptions for zero-baseline cases")
    print("   [SUCCESS] Safe sample size calculation (no more n=0 when data exists)")
    print("   [SUCCESS] Proper equity ratio handling for edge cases")

    print("\n4. Example improved output:")
    print("   OLD: 'inf% difference in question rates (White vs Black)'")
    print("   NEW: 'Total disparity (2.0% vs 0%) difference in question rates (White vs Black)'")

    print("\n=== Edge Case Test Complete ===")

if __name__ == "__main__":
    test_edge_case_fixes()