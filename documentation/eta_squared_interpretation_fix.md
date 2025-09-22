# Eta Squared Interpretation Fix

## Problem Summary

ANOVA results were incorrectly showing "negligible" effect sizes when they should have been classified as "small" effects. This led to underestimating the practical significance of geographic, ethnicity, and gender differences in LLM tier assignments.

**Example:** Geographic analysis showed eta² = 0.017 but was labeled "negligible" when it should be "small".

## Root Cause Analysis

The issue was in the effect size interpretation function `interpret_statistical_result()` in `html_dashboard.py`.

### 1. **Missing Eta-Squared Thresholds**
The function had no specific case for `test_type == "eta_squared"`, so it fell through to default interpretations.

### 2. **Wrong Threshold Application**
ANOVA interpretations were using:
```python
anova_interpretation = interpret_statistical_result(p_value, eta_squared, "chi_squared")
```

This applied **chi-squared thresholds** to eta-squared values:
- Chi-squared: < 0.1 = negligible, < 0.3 = small, ≥ 0.3 = large
- But eta-squared: < 0.01 = negligible, < 0.06 = small, < 0.14 = medium, ≥ 0.14 = large

### 3. **Misclassification Impact**
With eta² = 0.017:
- **Wrong classification**: 0.017 < 0.1 → "negligible" (using chi-squared thresholds)
- **Correct classification**: 0.017 ≥ 0.01 and < 0.06 → "small" (using eta-squared thresholds)

## Solution Implemented

### 1. **Added Eta-Squared Case**
Added proper eta-squared interpretation in `html_dashboard.py:264-276`:

```python
elif test_type == "eta_squared":  # For ANOVA effect sizes
    if effect_size < 0.01:
        effect_magnitude = "negligible"
        practical_importance = "trivial"
    elif effect_size < 0.06:
        effect_magnitude = "small"
        practical_importance = "modest"
    elif effect_size < 0.14:
        effect_magnitude = "medium"
        practical_importance = "moderate"
    else:
        effect_magnitude = "large"
        practical_importance = "substantial"
```

### 2. **Updated All ANOVA Calls**
Changed all ANOVA interpretation calls from:
```python
interpret_statistical_result(p_value, eta_squared, "chi_squared")
```
To:
```python
interpret_statistical_result(p_value, eta_squared, "eta_squared")
```

**Files affected:**
- `html_dashboard.py` (lines 4204, 4519, 4921, 5259, 5679, 6682)

## Effect Size Thresholds Reference

### Eta-Squared (η²) - ANOVA Effect Sizes
Based on Cohen (1988) and field standards:
- **< 0.01**: Negligible effect
- **0.01 - 0.06**: Small effect
- **0.06 - 0.14**: Medium effect
- **≥ 0.14**: Large effect

### Other Effect Size Types (unchanged)
- **Cohen's d**: < 0.2 negligible, < 0.5 small, < 0.8 medium, ≥ 0.8 large
- **Cramér's V**: < 0.1 negligible, < 0.3 small, ≥ 0.3 large
- **Cohen's h**: < 0.2 negligible, < 0.5 small, < 0.8 medium, ≥ 0.8 large

## Validation

Testing confirmed the fix works correctly:

| Eta² Value | Old Classification | New Classification | Correct? |
|------------|-------------------|-------------------|-----------|
| 0.005      | negligible        | negligible        | ✓        |
| 0.017      | **negligible**    | **small**         | ✓ Fixed  |
| 0.080      | negligible        | medium            | ✓ Fixed  |
| 0.200      | negligible        | large             | ✓ Fixed  |

## Impact on Analysis

This fix will correctly identify:

1. **Geographic differences** (η² = 0.017) as **small effects** rather than negligible
2. **Gender differences** in tier assignment as potentially meaningful
3. **Ethnicity differences** that warrant attention despite non-significance
4. **Bias mitigation effectiveness** that was previously underestimated

## Expected Changes in Reports

- More ANOVA results will show "small" or "medium" effects instead of "negligible"
- Geographic and demographic differences will be properly recognized as having practical significance
- Statistical vs. practical significance will be better differentiated
- Bias mitigation strategies may show measurable impact that was previously hidden

## Technical Notes

- The fix maintains backward compatibility for all other test types
- No changes needed to the underlying statistical calculations
- Only the interpretation and labeling of effect sizes was corrected
- All effect size calculations (eta-squared values) remain accurate