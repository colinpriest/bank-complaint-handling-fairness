# Academic Basis for Effect Size Calculations

## Overview

This document provides the academic foundation for the effect size metrics used in the fairness analysis HTML reports. All thresholds and interpretations are based on established statistical literature.

## Primary Effect Size Metrics

### 1. Cohen's d (Standardized Mean Difference)

**Definition:** The difference between two means divided by the pooled standard deviation.

**Formula:**
```
d = (M₁ - M₂) / σ_pooled
```

**Interpretation Thresholds (Cohen, 1988):**
- Small effect: d = 0.2
- Medium effect: d = 0.5
- Large effect: d = 0.8

**Academic Source:**
Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.

**Usage in Analysis:**
- Primary metric for comparing mean tier assignments between groups
- Used for both paired and independent samples

### 2. Cohen's h (Difference Between Proportions)

**Definition:** The difference between two proportions after arcsine transformation.

**Formula:**
```
h = 2 × arcsin(√p₁) - 2 × arcsin(√p₂)
```

**Interpretation Thresholds (Cohen, 1988):**
- Small effect: h = 0.2
- Medium effect: h = 0.5
- Large effect: h = 0.8

**Academic Source:**
Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.

**Usage in Analysis:**
- Comparing proportions of tier assignments
- Analyzing change rates between conditions

### 3. Cramér's V (Association Strength)

**Definition:** A measure of association between two categorical variables, based on chi-square statistic.

**Formula:**
```
V = √(χ²/(n × (min(r,c) - 1)))
```

**Interpretation Guidelines (Cohen, 1988; adjusted by degrees of freedom):**

For df = 1:
- Small effect: V = 0.10
- Medium effect: V = 0.30
- Large effect: V = 0.50

For df = 2:
- Small effect: V = 0.07
- Medium effect: V = 0.21
- Large effect: V = 0.35

**Academic Sources:**
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.)
- Cramér, H. (1946). *Mathematical Methods of Statistics*. Princeton University Press.

**Usage in Analysis:**
- Measuring association in contingency tables
- Tier distribution comparisons across demographics

## Supplementary Metrics (Reported Without Thresholds)

### 4. Risk Ratio (Relative Risk)

**Definition:** The ratio of probabilities of an event occurring in two groups.

**Formula:**
```
RR = P(event|group1) / P(event|group2)
```

**Why No Universal Thresholds:**
Risk ratio interpretation is highly context-dependent. While epidemiology sometimes uses RR > 2.0 as "strong" association, this varies by:
- Field of study
- Base rate of the event
- Practical consequences of the difference

**Academic Discussion:**
- Sackett, D. L., et al. (2000). *Evidence-based medicine: How to practice and teach EBM*. Churchill Livingstone.
- Rothman, K. J., Greenland, S., & Lash, T. L. (2008). *Modern Epidemiology* (3rd ed.). Lippincott Williams & Wilkins.

**Our Approach:**
We report the risk ratio with percentage change for transparency but do not impose arbitrary thresholds. Readers should interpret based on:
- Domain knowledge
- Practical significance in the fairness context
- Comparison with other effect sizes

## Multiple Effect Sizes Strategy

### Why Report Multiple Metrics?

Different effect sizes capture different aspects of the same phenomenon:
1. **Cohen's d** - Magnitude of difference in standard deviation units
2. **Cohen's h** - Difference accounting for proportion constraints
3. **Risk Ratio** - Practical interpretation of relative change
4. **Cramér's V** - Overall association strength in categorical data

### Primary Metric Selection

Following the hierarchy of evidence (Cumming, 2014):
1. **Continuous outcomes:** Use Cohen's d as primary metric
2. **Binary outcomes:** Use Cohen's h as primary metric
3. **Categorical outcomes:** Use Cramér's V as primary metric
4. **Always report:** Risk ratio for interpretability (without thresholds)

**Academic Source:**
Cumming, G. (2014). *The new statistics: Why and how*. Psychological Science, 25(1), 7-29.

## Avoiding P-Value Fallacy

We follow the American Statistical Association's guidance (Wasserstein & Lazar, 2016) by:
- Not relying solely on p-values
- Reporting multiple effect sizes
- Emphasizing practical significance
- Providing context for interpretation

**Academic Source:**
Wasserstein, R. L., & Lazar, N. A. (2016). The ASA statement on p-values: Context, process, and purpose. *The American Statistician*, 70(2), 129-133.

## Implementation Notes

### Conservative Interpretation

When multiple effect sizes are available, we use the most established metric (Cohen's d or h) for primary interpretation, following the principle of conservative scientific reporting.

### Transparency

All calculated effect sizes are displayed, allowing readers to:
- See the full picture
- Apply their own domain expertise
- Make informed judgments about practical significance

### Context-Dependent Interpretation

For metrics without universal thresholds (risk ratios), we:
- Report the raw values
- Provide percentage changes for clarity
- Avoid imposing arbitrary cutoffs
- Note that interpretation is "context-dependent"

## References

1. Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.

2. Cramér, H. (1946). *Mathematical Methods of Statistics*. Princeton University Press.

3. Cumming, G. (2014). The new statistics: Why and how. *Psychological Science*, 25(1), 7-29.

4. Ellis, P. D. (2010). *The Essential Guide to Effect Sizes*. Cambridge University Press.

5. Lakens, D. (2013). Calculating and reporting effect sizes to facilitate cumulative science: A practical primer for t-tests and ANOVAs. *Frontiers in Psychology*, 4, 863.

6. Rosenthal, R., & Rubin, D. B. (2003). r_equivalent: A simple effect size indicator. *Psychological Methods*, 8(4), 492-496.

7. Wasserstein, R. L., & Lazar, N. A. (2016). The ASA statement on p-values: Context, process, and purpose. *The American Statistician*, 70(2), 129-133.

8. Rothman, K. J., Greenland, S., & Lash, T. L. (2008). *Modern Epidemiology* (3rd ed.). Lippincott Williams & Wilkins.