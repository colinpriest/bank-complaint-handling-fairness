#!/usr/bin/env python3
"""
Test the corrected n-shot vs zero-shot disparity analysis
"""

import numpy as np
from scipy.stats import chi2_contingency

def calculate_cramers_v(contingency_table):
    """Calculate Cramér's V (legacy measure)"""
    chi2 = chi2_contingency(contingency_table)[0]
    n = np.sum(contingency_table)
    min_dim = min(contingency_table.shape) - 1
    if min_dim == 0 or n == 0:
        return 0
    return np.sqrt(chi2 / (n * min_dim))

def test_corrected_analysis():
    """Test the corrected disparity analysis with the example data"""
    print("=== Testing Corrected N-Shot vs Zero-Shot Analysis ===\n")

    # Example data from the user's question
    zero_shot_count = 10000
    zero_shot_questions = 113
    n_shot_count = 10000
    n_shot_questions = 5

    print("Original Data:")
    print(f"Zero-Shot: {zero_shot_questions}/{zero_shot_count} = {(zero_shot_questions/zero_shot_count)*100:.1f}%")
    print(f"N-Shot: {n_shot_questions}/{n_shot_count} = {(n_shot_questions/n_shot_count)*100:.1f}%")

    # Calculate rates
    zero_shot_rate = zero_shot_questions / zero_shot_count
    n_shot_rate = n_shot_questions / n_shot_count

    print(f"\nRate Comparison:")
    print(f"Zero-shot rate: {zero_shot_rate:.4f} ({zero_shot_rate*100:.1f}%)")
    print(f"N-shot rate: {n_shot_rate:.4f} ({n_shot_rate*100:.1f}%)")

    # OLD METHOD: Misleading Cramér's V
    contingency_table = np.array([
        [zero_shot_questions, zero_shot_count - zero_shot_questions],
        [n_shot_questions, n_shot_count - n_shot_questions]
    ])

    cramers_v = calculate_cramers_v(contingency_table)
    print(f"\n=== OLD ANALYSIS (MISLEADING) ===")
    print(f"Cramér's V: {cramers_v:.3f}")
    print(f"Interpretation: 'negligible' effect size")
    print(f"PROBLEM: This completely misses the massive practical difference!")

    # NEW METHOD: Proper Disparity Analysis
    print(f"\n=== CORRECTED ANALYSIS ===")

    # Calculate disparity metrics
    if n_shot_rate > 0:
        disparity_ratio = zero_shot_rate / n_shot_rate
        equity_ratio = n_shot_rate / zero_shot_rate
    else:
        disparity_ratio = float('inf')
        equity_ratio = 0.0

    reduction_percentage = ((zero_shot_rate - n_shot_rate) / zero_shot_rate) * 100

    # Assess severity
    if equity_ratio < 0.50:
        severity = "SEVERE"
        severity_description = "severe disparity (>50% worse than legal discrimination threshold)"
    elif equity_ratio < 0.67:
        severity = "MATERIAL"
        severity_description = "material disparity (two-thirds rule threshold)"
    elif equity_ratio < 0.80:
        severity = "CONCERNING"
        severity_description = "concerning disparity (approaching EEOC 80% rule threshold)"
    else:
        severity = "ACCEPTABLE"
        severity_description = "acceptable disparity (meets EEOC 80% rule standard)"

    print(f"Disparity Ratio: {disparity_ratio:.1f}× (Zero-shot questions {disparity_ratio:.1f}× more often)")
    print(f"Equity Ratio: {equity_ratio:.3f}")
    print(f"Severity: {severity} - {severity_description}")
    print(f"Reduction: {reduction_percentage:.0f}% decrease with n-shot examples")

    # Compare with percentage difference
    percentage_difference = abs(zero_shot_rate - n_shot_rate) * 100
    print(f"Percentage point difference: {percentage_difference:.1f} percentage points")

    # Show the dramatic practical difference
    print(f"\n=== PRACTICAL IMPACT ===")
    print(f"• Zero-shot asks questions in {zero_shot_questions} out of {zero_shot_count} cases")
    print(f"• N-shot asks questions in only {n_shot_questions} out of {n_shot_count} cases")
    print(f"• This means n-shot reduces questioning by {reduction_percentage:.0f}%")
    print(f"• Zero-shot is {disparity_ratio:.0f}× more likely to ask for information")

    if disparity_ratio > 10:
        print(f"• This is a MASSIVE behavioral change, not 'negligible'!")

    print(f"\n=== CORRECTED CONCLUSION ===")
    print(f"Effect Size: Equity ratio = {equity_ratio:.3f} ({severity} disparity)")
    print(f"Practical Significance: N-shot examples DRAMATICALLY reduce questioning behavior")
    print(f"                       ({disparity_ratio:.0f}× reduction, {reduction_percentage:.0f}% decrease)")
    print(f"Implication: N-Shot examples may be over-constraining the model's")
    print(f"            information-gathering behavior, potentially reducing")
    print(f"            appropriate due diligence in complex cases.")

if __name__ == "__main__":
    test_corrected_analysis()