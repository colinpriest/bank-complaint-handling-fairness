#!/usr/bin/env python3
"""
Test the severity impact analysis interpretation logic
"""

import numpy as np

# Simulate the severity analysis scenario
nm_count = 9367
m_count = 2675
nm_unchanged_pct = 0.891
m_unchanged_pct = 0.822

# Calculate change rates
nm_changed_rate = 1 - nm_unchanged_pct  # 0.109
m_changed_rate = 1 - m_unchanged_pct     # 0.178

# Mean tiers
nm_mean = 1.092
m_mean = 1.819
nm_std = 0.290
m_std = 0.391

# Calculate effect sizes
def calculate_cohens_h(p1, p2):
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    return phi1 - phi2

def calculate_risk_ratio(p1, p2):
    if p2 == 0:
        return float('inf') if p1 > 0 else 1.0
    return p1 / p2

# Cohen's h
cohens_h = calculate_cohens_h(m_changed_rate, nm_changed_rate)
print(f"Cohen's h: {cohens_h:.3f}")

# Risk Ratio
risk_ratio = calculate_risk_ratio(m_changed_rate, nm_changed_rate)
print(f"Risk Ratio: {risk_ratio:.2f} ({(risk_ratio - 1) * 100:.0f}% higher)")

# Cohen's d for mean tier difference
pooled_var = ((nm_count - 1) * nm_std**2 + (m_count - 1) * m_std**2) / (nm_count + m_count - 2)
pooled_std = np.sqrt(pooled_var)
cohens_d = (m_mean - nm_mean) / pooled_std
print(f"Cohen's d: {cohens_d:.3f}")

# Interpretation logic from the fixed code
if risk_ratio > 1.5 or risk_ratio < 0.67:
    overall_importance = 'substantial'
    overall_magnitude = 'large'
    print(f"\nInterpretation: Statistically significant and practically {overall_importance}")
    print(f"Reason: Risk ratio {risk_ratio:.2f} shows >50% change")
elif abs(cohens_d) >= 0.5:
    if abs(cohens_d) < 0.8:
        overall_importance = 'moderate'
        overall_magnitude = 'medium'
    else:
        overall_importance = 'substantial'
        overall_magnitude = 'large'
    print(f"\nInterpretation: Statistically significant and practically {overall_importance}")
    print(f"Reason: Cohen's d = {cohens_d:.3f} shows {overall_magnitude} effect")
elif risk_ratio > 1.3 or risk_ratio < 0.77:
    overall_importance = 'moderate'
    overall_magnitude = 'medium'
    print(f"\nInterpretation: Statistically significant and practically {overall_importance}")
    print(f"Reason: Risk ratio {risk_ratio:.2f} shows >30% change")
else:
    print(f"\nInterpretation: Statistically significant but practically trivial")
    print(f"Reason: All effect sizes are small")

print(f"\nConclusion: Monetary cases show {(risk_ratio - 1) * 100:.0f}% higher tier change rate.")
print(f"This indicates demographic factors have {'substantially ' if risk_ratio > 1.5 else ''}greater influence in high-stakes decisions.")