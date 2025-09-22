#!/usr/bin/env python3
"""
Final verification that all edge case fixes work across gender, ethnicity, and geography
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from html_dashboard import HTMLDashboard

def test_final_edge_case_verification():
    """Test all edge case fixes are working"""
    print("=== Final Edge Case Verification ===\n")

    dashboard = HTMLDashboard()

    print("1. Testing problematic ethnicity case (White vs Black with 0% baseline)...")

    # Simulate the exact case from the user's report
    ethnicity_data = {
        'n_shot_question_rate': {
            'white': {
                'question_rate': 0.02,  # 2%
                'total_cases': 100
            },
            'black': {
                'question_rate': 0.0,   # 0% - the problematic baseline
                'total_cases': 50
            }
        }
    }

    try:
        result_html = dashboard._build_improved_ethnicity_question_rate_disparity_analysis(
            ethnicity_data, 'N-Shot'
        )
        print("   [SUCCESS] Ethnicity analysis completed without error")

        # Check if we get better text than "inf%"
        if "inf%" not in result_html and "Total disparity" in result_html:
            print("   [SUCCESS] No 'inf%' messages found")
            print("   [SUCCESS] Proper disparity description used")
        else:
            print("   [WARNING] May still contain problematic text")

    except Exception as e:
        print(f"   [ERROR] Ethnicity analysis failed: {e}")

    print("\n2. Testing zero sample size handling...")

    # Test with zero/None counts
    zero_counts = {
        'group1': 0,
        'group2': None,
        'group3': 0
    }

    sample_size = dashboard._calculate_safe_sample_size(zero_counts)
    print(f"   Zero counts result: {sample_size} (should be 0)")

    if sample_size == 0:
        print("   [SUCCESS] Zero sample size handled correctly")
    else:
        print("   [ERROR] Sample size calculation incorrect")

    print("\n3. Testing all demographic analyses with edge cases...")

    edge_case_data = {
        'n_shot_question_rate': {
            'group_a': {'question_rate': 0.01, 'total_cases': 200},
            'group_b': {'question_rate': 0.0, 'total_cases': 100}  # Zero baseline
        }
    }

    analyses = [
        ('Gender', dashboard._build_improved_gender_question_rate_disparity_analysis),
        ('Geography', dashboard._build_improved_geographic_question_rate_disparity_analysis),
        ('Ethnicity', dashboard._build_improved_ethnicity_question_rate_disparity_analysis)
    ]

    for name, analysis_func in analyses:
        try:
            result = analysis_func(edge_case_data, 'N-Shot')
            if "inf%" not in result and "Total disparity" in result:
                print(f"   [SUCCESS] {name} analysis handles zero baseline correctly")
            else:
                print(f"   [WARNING] {name} analysis may have issues")
        except Exception as e:
            print(f"   [ERROR] {name} analysis failed: {e}")

    print("\n4. Expected fixes summary:")
    print("   Before: 'inf% difference in question rates (White vs Black)'")
    print("   After:  'Total disparity (2.0% vs 0%) difference in question rates (White vs Black)'")
    print()
    print("   Before: 'n = 0' (when data exists)")
    print("   After:  'n = 300' (actual total sample size)")
    print()
    print("   Before: 'equity_deficit = 1.000' (misleading)")
    print("   After:  'equity_deficit = 1.000' (still 1.0 but better context)")

    print("\n=== Verification Complete ===")
    print("All edge cases should now be handled gracefully with clear, user-friendly messages.")

if __name__ == "__main__":
    test_final_edge_case_verification()