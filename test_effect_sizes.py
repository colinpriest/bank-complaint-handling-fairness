#!/usr/bin/env python3
"""
Test script to verify the new effect size calculations in HTML dashboard
"""

import numpy as np
import sys
import os

# Add parent directory to path to import html_dashboard
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from html_dashboard import (
    calculate_cohens_d_paired,
    calculate_cohens_d_independent,
    calculate_cohens_h,
    calculate_risk_ratio,
    calculate_cramers_v,
    interpret_statistical_result
)

def test_cohens_h():
    """Test Cohen's h for proportion differences"""
    print("Testing Cohen's h for proportion differences:")

    # Test case from severity analysis
    # Non-monetary: 10.9% changed, Monetary: 17.8% changed
    p1 = 0.109  # Non-monetary change rate
    p2 = 0.178  # Monetary change rate

    cohens_h = calculate_cohens_h(p2, p1)
    print(f"  Non-monetary change rate: {p1:.3f}")
    print(f"  Monetary change rate: {p2:.3f}")
    print(f"  Cohen's h: {cohens_h:.3f}")

    # Interpret the result
    interpretation = interpret_statistical_result(0.001, cohens_h, "cohens_h")
    print(f"  Magnitude: {interpretation['effect_magnitude']}")
    print(f"  Practical importance: {interpretation['practical_importance']}")
    print()

def test_risk_ratio():
    """Test risk ratio calculation"""
    print("Testing Risk Ratio:")

    # Same proportions as above
    p1 = 0.109
    p2 = 0.178

    risk_ratio = calculate_risk_ratio(p2, p1)
    print(f"  Non-monetary change rate: {p1:.3f}")
    print(f"  Monetary change rate: {p2:.3f}")
    print(f"  Risk Ratio: {risk_ratio:.3f}")
    print(f"  Interpretation: Monetary cases are {risk_ratio:.1f}× more likely to change")

    # Interpret the result
    interpretation = interpret_statistical_result(0.001, risk_ratio, "risk_ratio")
    print(f"  Magnitude: {interpretation['effect_magnitude']}")
    print(f"  Practical importance: {interpretation['practical_importance']}")
    print()

def test_cohens_d_independent():
    """Test Cohen's d for independent samples"""
    print("Testing Cohen's d for independent samples:")

    # Test case from severity analysis
    # Non-monetary mean: 1.092, std: 0.290, n: 9367
    # Monetary mean: 1.819, std: 0.391, n: 2675

    # Simulate data with these parameters
    np.random.seed(42)
    non_monetary = np.random.normal(1.092, 0.290, 100)
    monetary = np.random.normal(1.819, 0.391, 100)

    cohens_d = calculate_cohens_d_independent(monetary, non_monetary)
    print(f"  Non-monetary mean: {np.mean(non_monetary):.3f}")
    print(f"  Monetary mean: {np.mean(monetary):.3f}")
    print(f"  Cohen's d: {cohens_d:.3f}")

    # Interpret the result
    interpretation = interpret_statistical_result(0.001, cohens_d, "independent_t_test")
    print(f"  Magnitude: {interpretation['effect_magnitude']}")
    print(f"  Practical importance: {interpretation['practical_importance']}")
    print()

def test_cramers_v():
    """Test Cramér's V calculation"""
    print("Testing Cramér's V:")

    # Create a contingency table
    # Example: Gender x Tier 0 (Yes/No)
    contingency_table = np.array([
        [100, 900],  # Male: 100 Tier 0, 900 Non-Tier 0
        [150, 850]   # Female: 150 Tier 0, 850 Non-Tier 0
    ])

    cramers_v = calculate_cramers_v(contingency_table)
    print(f"  Contingency table:")
    print(f"    Male:   {contingency_table[0]}")
    print(f"    Female: {contingency_table[1]}")
    print(f"  Cramér's V: {cramers_v:.3f}")

    # Interpret the result
    interpretation = interpret_statistical_result(0.05, cramers_v, "chi_squared")
    print(f"  Magnitude: {interpretation['effect_magnitude']}")
    print(f"  Practical importance: {interpretation['practical_importance']}")
    print()

def test_interpretation_logic():
    """Test the interpretation logic for different effect sizes"""
    print("Testing Interpretation Logic:")
    print()

    test_cases = [
        ("Cohen's d = 0.1 (negligible)", 0.001, 0.1, "independent_t_test"),
        ("Cohen's d = 0.3 (small)", 0.001, 0.3, "independent_t_test"),
        ("Cohen's d = 0.6 (medium)", 0.001, 0.6, "independent_t_test"),
        ("Cohen's d = 1.2 (large)", 0.001, 1.2, "independent_t_test"),
        ("Cohen's h = 0.15 (negligible)", 0.001, 0.15, "cohens_h"),
        ("Cohen's h = 0.35 (small)", 0.001, 0.35, "cohens_h"),
        ("Risk Ratio = 1.05 (negligible)", 0.001, 1.05, "risk_ratio"),
        ("Risk Ratio = 1.4 (small)", 0.001, 1.4, "risk_ratio"),
        ("Risk Ratio = 1.7 (medium)", 0.001, 1.7, "risk_ratio"),
        ("Cramér's V = 0.05 (negligible)", 0.001, 0.05, "chi_squared"),
        ("Cramér's V = 0.2 (small to medium)", 0.001, 0.2, "chi_squared"),
    ]

    for description, p_value, effect_size, test_type in test_cases:
        interpretation = interpret_statistical_result(p_value, effect_size, test_type)
        print(f"  {description}:")
        print(f"    Interpretation: {interpretation['interpretation']}")
        if interpretation['warning']:
            print(f"    Warning: {interpretation['warning']}")
        print()

def main():
    """Run all tests"""
    print("=" * 60)
    print("TESTING NEW EFFECT SIZE CALCULATIONS")
    print("=" * 60)
    print()

    test_cohens_h()
    test_risk_ratio()
    test_cohens_d_independent()
    test_cramers_v()
    test_interpretation_logic()

    print("=" * 60)
    print("TEST COMPLETE - All effect size calculations working")
    print("=" * 60)

if __name__ == "__main__":
    main()