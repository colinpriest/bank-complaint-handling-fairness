# Practical Materiality Framework Implementation

## Overview

Implemented a comprehensive practical materiality framework for assessing demographic disparities in AI/ML systems, replacing purely statistical significance testing with regulatory-aligned materiality thresholds.

## Problem Addressed

Previously, the system focused on statistical significance (p-values) which can detect trivial differences in large samples. A 20.8% disparity in tier assignments was being described as merely "statistically significant" when it clearly represents a severe regulatory and fairness issue.

## Solution Components

### 1. **Materiality Framework (`practical_materiality_framework.py`)**

Created a comprehensive framework with evidence-based thresholds:

#### Disparity Levels
- **CRITICAL** (>30%): Emergency response required
- **SEVERE** (20-30%): Immediate remediation required
- **MATERIAL** (10-20%): Investigation and remediation needed
- **CONCERNING** (5-10%): Enhanced monitoring required
- **MINIMAL** (2-5%): Document and monitor
- **NEGLIGIBLE** (<2%): No action required

#### Key Features
- **80% Rule Compliance**: Based on EEOC guidelines (29 CFR 1607.4(D))
- **Odds Ratio Analysis**: Alternative measurement method
- **Sample Size Guidance**: Confidence assessment based on n
- **Regulatory Citations**: Full documentation of legal basis
- **Action Recommendations**: Specific steps for each severity level

### 2. **Dashboard Integration (`html_dashboard.py`)**

#### Tier Impact Analysis Enhancement
- Modified `_build_tier_impact_table()` to use MaterialityFramework
- Added practical assessment alongside statistical testing
- Color-coded severity levels (red for severe, orange for material, etc.)
- Displays regulatory citations and required actions

#### Headline Results Integration
- Results with SEVERE or MATERIAL disparities automatically added to collector
- New effect type `disparity_rate` with proper interpretation
- Enhanced badges for severity levels with visual indicators
- Pulsing animation for SEVERE disparities to draw attention

#### CSS Styling
Added new badge styles:
```css
.effect-badge.severe-effect {
    background: #d32f2f;
    animation: pulse 2s infinite;
}
.effect-badge.material-effect {
    background: #f57c00;
}
```

### 3. **Statistical Interpretation Updates**

- Added `practical_materiality` test type to `interpret_statistical_result()`
- Proper thresholds for disparity rates
- Clear practical importance descriptions

## Regulatory Basis

The framework is grounded in:

### Primary Legal Authorities
- **EEOC 80% Rule**: 29 CFR 1607.4(D) - Uniform Guidelines
- **ECOA**: 15 USC 1691 - Equal Credit Opportunity Act
- **Regulation B**: 12 CFR 1002 - ECOA Implementation
- **Fair Housing Act**: 42 USC 3601
- **UDAAP**: 12 USC 5531 - Unfair/Deceptive Practices

### Regulatory Guidance
- **OCC Bulletin 2011-12**: Model Risk Management
- **Fed SR 11-7**: Model Risk Management Guidance
- **CFPB Circular 2022-03**: AI/ML Adverse Action Requirements
- **HUD 2020 Rule**: Disparate Impact Standards

### Case Law
- Texas Dept. Housing v. Inclusive Communities (2015)
- Griggs v. Duke Power (1971)
- Ricci v. DeStefano (2009)

## Example Output

For a 20.8% disparity rate:

```
Materiality Level: SEVERE
80% Rule Compliance: FAIL
Regulatory Citation: Exceeds EEOC 80% rule threshold (29 CFR 1607.4(D))
Implication: SEVERE DISPARITY: Immediate remediation required. High risk of regulatory enforcement action.

Required Actions:
• Immediate root cause analysis required
• Develop remediation plan within 30 days
• Consider model adjustment or replacement
```

## Impact on Analysis

### Before
- Focus on p-values (e.g., p < 0.05)
- 20.8% disparity described as "statistically significant"
- No regulatory context or action guidance
- Risk of missing material business/legal issues

### After
- Clear severity classification based on magnitude
- Regulatory compliance assessment (80% rule)
- Specific remediation requirements
- Direct link to enforcement risk

## Files Modified

1. **`practical_materiality_framework.py`** (NEW)
   - Complete framework implementation
   - Regulatory citations database
   - Assessment and reporting functions

2. **`html_dashboard.py`**
   - Lines 2514-2566: Tier impact materiality assessment
   - Lines 290-305: New practical_materiality test type
   - Lines 1993-2005: Disparity rate badge display
   - Lines 1096-1124: CSS for severity badges

## Testing

Confirmed framework correctly identifies:
- 20.8% disparity → SEVERE (correct)
- Fails 80% rule (correct)
- Requires immediate remediation (correct)
- Provides proper regulatory citations

## Future Enhancements

1. **Trend Analysis**: Track disparity changes over time
2. **Mitigation Effectiveness**: Measure impact of bias reduction strategies
3. **Automated Reporting**: Generate compliance documentation
4. **Risk Scoring**: Combine multiple disparity metrics

## Conclusion

The practical materiality framework provides a legally-grounded, business-relevant assessment of demographic disparities that goes beyond statistical significance to identify genuine fairness and compliance risks. This ensures that material disparities like the 20.8% tier difference are properly flagged as severe issues requiring immediate attention, regardless of p-values.