#!/usr/bin/env python3
"""
Test the eta squared interpretation fix
"""

import sys
sys.path.append('.')
from html_dashboard import interpret_statistical_result

def test_eta_squared_interpretation():
    """Test eta squared interpretation with correct thresholds"""

    test_cases = [
        (0.005, "negligible"),  # < 0.01
        (0.017, "small"),       # 0.01 <= x < 0.06
        (0.08, "medium"),       # 0.06 <= x < 0.14
        (0.20, "large")         # >= 0.14
    ]

    print("Testing eta squared interpretation:")
    print("Value | Expected | Actual | Correct?")
    print("------|----------|--------|----------")

    for eta_squared, expected in test_cases:
        # Test with non-significant p-value to focus on effect size
        result = interpret_statistical_result(0.10, eta_squared, "eta_squared")
        actual = result["effect_magnitude"]
        correct = "YES" if actual == expected else "NO"
        print(f"{eta_squared:5.3f} | {expected:8} | {actual:6} | {correct}")

    # Test the specific case from the geography analysis
    print(f"\nSpecific case: eta_squared = 0.017")
    result = interpret_statistical_result(0.3764, 0.017, "eta_squared")
    print(f"Effect magnitude: {result['effect_magnitude']}")
    print(f"Practical importance: {result['practical_importance']}")
    print(f"Interpretation: {result['interpretation']}")

if __name__ == "__main__":
    test_eta_squared_interpretation()