# Effect Size Specifications for HTML Reports

## Overview

This document specifies how to add effect size calculations and interpretations to the existing statistical tests in the HTML dashboard to help users understand the practical significance of statistically significant results.

## Current Statistical Tests in HTML Reports

Based on analysis of `html_dashboard.py`, the following statistical tests are currently implemented:

1. **Paired t-test** - Used for comparing baseline vs persona-injected experiments
2. **Chi-squared test of independence** - Used for categorical data comparisons
3. **Chi-squared test on counts** - Used for tier distribution analysis
4. **Unknown/Generic tests** - Various other statistical tests from analysis results

## Effect Size Specifications by Test Type

### 1. Paired t-test (Lines 1681-1729 in html_dashboard.py)

**Current Implementation:**
- Calculates t-statistic and p-value
- Reports degrees of freedom

**Effect Size to Add: Cohen's d**

**Calculation:**
```python
import numpy as np

def calculate_cohens_d_paired(group1, group2):
    """Calculate Cohen's d for paired samples"""
    differences = np.array(group1) - np.array(group2)
    return np.mean(differences) / np.std(differences, ddof=1)
```

**Interpretation Guidelines:**
- Small effect: |d| < 0.2
- Medium effect: 0.2 ≤ |d| < 0.8
- Large effect: |d| ≥ 0.8

**Implementation Location:** Around line 1686 after t-statistic calculation

### 2. Chi-squared Test of Independence (Lines 1795-1836, 1898-1939)

**Current Implementation:**
- Uses `chi2_contingency()` from scipy.stats
- Reports chi-squared statistic and p-value

**Effect Size to Add: Cramér's V**

**Calculation:**
```python
import numpy as np
from scipy.stats import chi2_contingency

def calculate_cramers_v(contingency_table):
    """Calculate Cramér's V for chi-squared test"""
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    n = contingency_table.sum()
    min_dim = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim))
    return cramers_v
```

**Interpretation Guidelines:**
- Small effect: V < 0.1
- Medium effect: 0.1 ≤ V < 0.3
- Large effect: V ≥ 0.3

**Implementation Location:** After `chi2_contingency()` calls around lines 1805 and 1908

### 3. Chi-squared Test on Counts (Lines 2229-2397)

**Current Implementation:**
- Reports chi-squared test results for tier distributions
- Used in tier0, tier2, and general tier analysis

**Effect Size to Add: Cramér's V**

**Calculation:** Same as above

**Implementation Location:** Multiple locations in tier analysis functions

### 4. Generic Statistical Tests (Lines 2902-4535)

**Current Implementation:**
- Various test types reported from analysis results
- May already include some effect sizes in the data

**Effect Size Strategy:**
- Check if effect size already exists in stats dictionary
- If not available, note limitation in the report
- Prioritize adding Cohen's d for t-tests and Cramér's V for chi-squared tests

## Enhanced Interpretation Logic

### Current Logic Pattern
```python
is_significant = p_value < 0.05
conclusion = "rejected" if is_significant else "not rejected"
```

### Enhanced Logic with Effect Size

```python
def interpret_statistical_result(p_value, effect_size, test_type):
    """Enhanced interpretation including effect size materiality"""

    # Statistical significance
    is_significant = p_value < 0.05
    significance_text = "rejected" if is_significant else "not rejected"

    # Effect size interpretation
    if test_type == "paired_t_test":
        if abs(effect_size) < 0.2:
            effect_magnitude = "negligible"
            practical_importance = "trivial"
        elif abs(effect_size) < 0.8:
            effect_magnitude = "small to medium"
            practical_importance = "modest"
        else:
            effect_magnitude = "large"
            practical_importance = "substantial"

    elif test_type == "chi_squared":
        if effect_size < 0.1:
            effect_magnitude = "negligible"
            practical_importance = "trivial"
        elif effect_size < 0.3:
            effect_magnitude = "small to medium"
            practical_importance = "modest"
        else:
            effect_magnitude = "large"
            practical_importance = "substantial"

    # Combined interpretation
    if is_significant and practical_importance == "trivial":
        interpretation = f"statistically significant but practically {practical_importance}"
        warning = " (large sample size may detect trivial differences)"
    elif is_significant and practical_importance != "trivial":
        interpretation = f"statistically significant and practically {practical_importance}"
        warning = ""
    else:
        interpretation = f"not statistically significant (effect size: {effect_magnitude})"
        warning = ""

    return {
        'significance_text': significance_text,
        'effect_magnitude': effect_magnitude,
        'practical_importance': practical_importance,
        'interpretation': interpretation,
        'warning': warning
    }
```

### Enhanced HTML Template

**Replace existing conclusion patterns with:**

```html
<p><strong>Effect Size:</strong> {effect_size:.3f} ({effect_magnitude})</p>
<p><strong>p-value:</strong> {p_value_display}</p>
<p><strong>Conclusion:</strong> The null hypothesis was <strong>{significance_text}</strong> (p {"<" if p_value < 0.05 else "≥"} 0.05).</p>
<p><strong>Practical Significance:</strong> This result is {interpretation}{warning}.</p>
```

## Implementation Priority

### High Priority (Add immediately)
1. **Paired t-tests** - Cohen's d calculation (most common test)
2. **Chi-squared independence tests** - Cramér's V calculation

### Medium Priority
1. **Chi-squared count tests** - Cramér's V for tier distribution analysis
2. **Generic test enhancement** - Check for existing effect sizes

### Low Priority
1. **Additional effect sizes** - Eta-squared for ANOVA, Cliff's delta for non-parametric tests

## Code Locations for Implementation

### 1. Add Effect Size Calculation Functions
**Location:** Top of `html_dashboard.py` after imports

```python
def calculate_cohens_d_paired(group1, group2):
    """Calculate Cohen's d for paired samples"""
    differences = np.array(group1) - np.array(group2)
    return np.mean(differences) / np.std(differences, ddof=1)

def calculate_cramers_v(contingency_table):
    """Calculate Cramér's V for chi-squared test"""
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    n = contingency_table.sum()
    min_dim = min(contingency_table.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))

def interpret_statistical_result(p_value, effect_size, test_type):
    """Enhanced interpretation including effect size materiality"""
    # Implementation as specified above
```

### 2. Modify Existing Test Implementations

**Paired t-test (around line 1686):**
```python
# After calculating t_statistic
if len(baseline_scores) > 0 and len(injected_scores) > 0:
    cohens_d = calculate_cohens_d_paired(injected_scores, baseline_scores)
    interpretation = interpret_statistical_result(p_value, cohens_d, "paired_t_test")
```

**Chi-squared tests (around lines 1805, 1908):**
```python
# After chi2_contingency call
cramers_v = calculate_cramers_v(contingency_table)
interpretation = interpret_statistical_result(p_value, cramers_v, "chi_squared")
```

### 3. Update HTML Templates

**Update all statistical test HTML outputs to include:**
- Effect size value and magnitude
- Enhanced practical significance interpretation
- Warning about trivial effects when appropriate

## Testing and Validation

1. **Unit Tests:** Create tests for effect size calculation functions
2. **Integration Tests:** Verify HTML output includes effect sizes
3. **Manual Validation:** Check interpretations align with statistical best practices
4. **Documentation:** Update user documentation to explain effect size meanings

## Future Enhancements

1. **Additional Effect Sizes:**
   - Eta-squared (η²) for ANOVA
   - Cliff's delta for non-parametric tests
   - Glass's delta for unequal variances

2. **Confidence Intervals for Effect Sizes:**
   - Bootstrap confidence intervals for Cohen's d
   - Analytical confidence intervals for Cramér's V

3. **Effect Size Visualization:**
   - Visual indicators for effect magnitude
   - Effect size vs sample size plots

## References

- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.)
- Lakens, D. (2013). Calculating and reporting effect sizes to facilitate cumulative science
- American Psychological Association (2020). Publication Manual (7th ed.) - Effect size reporting guidelines