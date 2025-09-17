#!/usr/bin/env python3
"""
Power analysis to calculate required sample size for Gender Effects test
"""

import numpy as np
from scipy import stats
import math

def calculate_power_analysis():
    """Calculate required sample size for Gender Effects to be statistically significant"""
    
    # Current data from the analysis
    male_mean = 1.406
    female_mean = 1.383
    male_std = 1.175
    female_std = 1.198
    male_n = 5005
    female_n = 4995
    
    # Calculate current effect size (Cohen's d)
    mean_diff = male_mean - female_mean
    pooled_std = math.sqrt(((male_n - 1) * male_std**2 + (female_n - 1) * female_std**2) / (male_n + female_n - 2))
    cohens_d = mean_diff / pooled_std
    
    print(f"Current Analysis:")
    print(f"Male mean: {male_mean:.3f}, Female mean: {female_mean:.3f}")
    print(f"Mean difference: {mean_diff:.3f}")
    print(f"Pooled standard deviation: {pooled_std:.3f}")
    print(f"Effect size (Cohen's d): {cohens_d:.4f}")
    print(f"Current sample sizes: Male={male_n}, Female={female_n}")
    print()
    
    # Calculate current t-statistic and p-value
    current_se = pooled_std * math.sqrt(1/male_n + 1/female_n)
    current_t = mean_diff / current_se
    current_df = male_n + female_n - 2
    current_p = 2 * (1 - stats.t.cdf(abs(current_t), current_df))
    
    print(f"Current test results:")
    print(f"Standard error: {current_se:.4f}")
    print(f"t-statistic: {current_t:.3f}")
    print(f"Degrees of freedom: {current_df}")
    print(f"p-value: {current_p:.4f}")
    print()
    
    # Calculate required sample size for statistical significance
    # For two-sample t-test with equal sample sizes
    alpha = 0.05  # significance level
    power = 0.80  # desired power (80%)
    
    # Using power analysis formula for two-sample t-test
    # n = 2 * (z_alpha/2 + z_beta)^2 * sigma^2 / delta^2
    
    z_alpha_2 = stats.norm.ppf(1 - alpha/2)  # 1.96 for alpha=0.05
    z_beta = stats.norm.ppf(power)  # 0.84 for power=0.80
    
    # Required sample size per group
    n_required = 2 * ((z_alpha_2 + z_beta)**2) * (pooled_std**2) / (mean_diff**2)
    n_required = math.ceil(n_required)
    
    print(f"Power Analysis for Statistical Significance:")
    print(f"Alpha (significance level): {alpha}")
    print(f"Desired power: {power}")
    print(f"z_alpha/2: {z_alpha_2:.3f}")
    print(f"z_beta: {z_beta:.3f}")
    print()
    print(f"Required sample size per group: {n_required:,}")
    print(f"Total required sample size: {2 * n_required:,}")
    print()
    
    # Calculate how much to increase from current
    current_total = male_n + female_n
    required_total = 2 * n_required
    increase_factor = required_total / current_total
    
    print(f"Sample Size Comparison:")
    print(f"Current total sample size: {current_total:,}")
    print(f"Required total sample size: {required_total:,}")
    print(f"Increase factor: {increase_factor:.1f}x")
    print(f"Additional samples needed: {required_total - current_total:,}")
    print()
    
    # Alternative: What if we want p < 0.01?
    alpha_strict = 0.01
    z_alpha_2_strict = stats.norm.ppf(1 - alpha_strict/2)
    n_required_strict = 2 * ((z_alpha_2_strict + z_beta)**2) * (pooled_std**2) / (mean_diff**2)
    n_required_strict = math.ceil(n_required_strict)
    
    print(f"For p < 0.01 significance:")
    print(f"Required sample size per group: {n_required_strict:,}")
    print(f"Total required sample size: {2 * n_required_strict:,}")
    print(f"Increase factor: {(2 * n_required_strict) / current_total:.1f}x")
    print()
    
    # Show what p-value we'd get with different sample sizes
    print("P-values at different sample sizes:")
    multipliers = [1.5, 2, 3, 4, 5, 10]
    for mult in multipliers:
        new_n = int(current_total * mult / 2)  # per group
        new_se = pooled_std * math.sqrt(2/new_n)
        new_t = mean_diff / new_se
        new_df = 2 * new_n - 2
        new_p = 2 * (1 - stats.t.cdf(abs(new_t), new_df))
        print(f"  {mult}x current size ({new_n:,} per group): p = {new_p:.4f}")

if __name__ == "__main__":
    calculate_power_analysis()