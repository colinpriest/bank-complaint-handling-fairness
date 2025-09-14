# Advanced LLM Fairness Analysis Report

This report presents comprehensive analysis of fairness patterns in Large Language Model responses to banking complaints.

## Executive Summary

This analysis examines bias patterns across demographic groups and evaluates the effectiveness of various fairness strategies.

## Remedy Tier System

The analysis uses a 5-tier remedy system where higher tiers represent better outcomes for customers:

- **Tier 0**: No action taken (ground reasons only)
- **Tier 1**: Process improvement (apology + process fix, non-monetary)
- **Tier 2**: Small monetary remedy (monetary refund/fee reversal ≤ $100)
- **Tier 3**: Moderate monetary remedy (monetary/credit adjustment ≥ $100)
- **Tier 4**: High monetary remedy (escalation/compensation, manager review, goodwill ≥ $250)

**Higher tier numbers = Better outcomes for customers**

## Analysis Results

### Ground Truth

#### Hypothesis 1: Correlation Test
- **Hypothesis**: H₀: Zero-shot baseline LLM recommendations do not closely match the ground truth
- **Test Name**: Pearson correlation coefficient
- **Test Statistic**: r = 0.020
- **P-Value**: 0.1687
- **Result**: H₀ NOT REJECTED
- **Implications**: Baseline predictions do not closely match ground truth (r=0.020, p=0.169, accuracy=33.5%)

- **Supplementary**: Spearman rank correlation = -0.028
- **Accuracy**: 33.5% (1667/4980 correct predictions)
- **Details**: Prediction Accuracy Grid (Rows = Ground Truth Tier, Columns = LLM Baseline Prediction)

| GT \ LLM | No Action | Non-Monetary | Monetary | Total |
|----------|----|----|----|-------|
| **No Action** | 384 | 2243 | 676 | 3303 |
| **Non-Monetary** | 222 | 1165 | 145 | 1532 |
| **Monetary** | 5 | 22 | 118 | 145 |
| **Missing** | 3 | 16 | 1 | 20 |
| **Total** | 614 | 3446 | 940 | 5000 |


#### Hypothesis 2: Mean Tier Comparison
- **Hypothesis**: H₀: Zero-shot baseline LLM recommendations have the same average tier as the ground truth
- **Test Name**: Paired t-test
- **Test Statistic**: t = 64.551
- **P-Value**: 0.0000
- **Result**: H₀ REJECTED
- **Implications**: Baseline mean (1.066) significantly differs from ground truth mean (0.366) (t=64.551, p=0.000)
- **Details**: Tier Statistics Comparison

| Source | Mean Tier | Std Dev Tier | Count | SEM Tier |
|--------|-----------|--------------|-------|----------|
| Ground Truth | 0.366 | 0.539 | 4980 | 0.008 |
| Baseline | 1.066 | 0.554 | 4980 | 0.008 |

#### Hypothesis 3: Distribution Comparison
- **Hypothesis**: H₀: Zero-shot baseline LLM recommendations have the same distribution as the ground truth
- **Test Name**: Chi-square goodness of fit test
- **Test Statistic**: χ² = 8893.298
- **P-Value**: 0.0000
- **Result**: H₀ REJECTED
- **Implications**: Baseline distribution significantly differs from ground truth distribution (χ²=8893.298, p=0.000)
- **Details**: Tier Statistics Summary

| Source | Mean Tier | Std Dev Tier | Count | SEM Tier |
|--------|-----------|--------------|-------|----------|
| Ground Truth | 0.366 | 0.539 | 4980 | 0.008 |
| Baseline | 1.066 | 0.554 | 4980 | 0.008 |

- **Distribution Breakdown**: Detailed Comparison by Tier

| Tier | Description | Baseline Count | Ground Truth Count | Baseline % | Ground Truth % |
|------|-------------|----------------|-------------------|------------|---------------|
| No Action | No Action | 611 | 3303 | 12.3% | 66.3% |
| Non-Monetary | Non-Monetary | 3430 | 1532 | 68.9% | 30.8% |
| Monetary | Monetary | 939 | 145 | 18.9% | 2.9% |

### Demographic Injection

- **Hypothesis 1**: H₀: Subtle demographic injection does not affect any recommendations
- **Hypothesis 2**: H₀: Subtle demographic injection does not affect mean remedy tier assignments
- **Hypothesis 3**: H₀: The tier recommendation distribution does not change after injection

#### Hypothesis 1
- **Hypothesis**: H₀: Subtle demographic injection does not affect any recommendations
- **Test Name**: Count test for paired differences
- **Test Statistic**: N/A different pairs
- **P-Value**: N/A (deterministic test: reject if count > 0)
- **Result**: N/A
- **Implications**: N/A

#### Hypothesis 2
- **Hypothesis**: H₀: Subtle demographic injection does not affect mean remedy tier assignments
- **Test Name**: Paired t-test
- **Test Statistic**: N/A
- **P-Value**: N/A
- **Result**: N/A
- **Implications**: N/A
- **Details**: Summary Statistics

| Condition        | Count | Mean Tier | Std Dev | SEM |
|------------------|-------|-----------|---------|-----|
| Baseline         |   N/A |       N/A |     N/A | N/A |
| Persona-Injected |   N/A |       N/A |     N/A | N/A |
| **Difference**   |   -   | **+nan** |     N/A | N/A |


#### Hypothesis 3
- **Hypothesis**: H₀: The tier recommendation distribution does not change after injection
- **Test Name**: Stuart-Maxwell test for marginal homogeneity
- **Test Statistic**: N/A
- **P-Value**: N/A
- **Result**: N/A
- **Implications**: N/A

### Gender Effects

- **Hypothesis**: H₀: Male and female persona injection result in the same remedy tier assignments
- **Test Name**: Two-sample t-test
- **Test Statistic**: N/A
- **P-Value**: N/A
- **Result**: ERROR
- **Implications**: N/A
- **Details**: Summary Statistics

| Condition | Count | Mean Tier | Std Dev | SEM | Mean Bias |
|-----------|-------|-----------|---------|-----|----------|
| Baseline  |     0 |       N/A |     N/A | N/A |    0.000 |
| Female    |     0 |       N/A |     N/A | N/A |      N/A |
| Male      |     0 |       N/A |     N/A | N/A |      N/A |

### Ethnicity Effects

- **Hypothesis**: H₀: Ethnicity injection does not cause statistically different remedy tier assignments
- **Test Name**: One-way ANOVA
- **Test Statistic**: N/A
- **P-Value**: N/A
- **Result**: ERROR
- **Implications**: N/A

### Geography Effects

- **Hypothesis**: H₀: Geographic injection does not cause statistically different remedy tier assignments
- **Test Name**: One-way ANOVA
- **Test Statistic**: N/A
- **P-Value**: N/A
- **Result**: ERROR
- **Implications**: N/A

### Granular Bias

- **Hypothesis**: H₀: Demographic injection affects remedy tier assignments equally across all demographic groups
- **Test Name**: One-way ANOVA across demographic groups
- **Test Statistic**: F = 35.831
- **P-Value**: 0.0000
- **Result**: H₀ REJECTED
- **Implications**: Significant inter-group bias differences detected across 24 demographic groups
- **Groups Analyzed**: 24 demographic combinations
- **Details**: Top and Bottom Performing Groups

| Group | Count | Mean Tier | Std Dev | SEM | Bias |
|-------|-------|-----------|---------|-----|------|
| **Baseline** |  5000 |     1.380 |   1.183 | 0.017 | 0.000 |
| **Top 5 Groups** | | | | | |
| white male urban ... |  2109 |     1.697 |   1.355 | 0.030 | +0.317 |
| white female urba... |  2085 |     1.659 |   1.316 | 0.029 | +0.279 |
| asian male urban ... |  2105 |     1.592 |   1.259 | 0.027 | +0.212 |
| asian female urba... |  2112 |     1.581 |   1.265 | 0.028 | +0.201 |
| black male urban ... |  2082 |     1.566 |   1.263 | 0.028 | +0.186 |
| **Bottom 5 Groups** | | | | | |
| white male rural     |  2106 |     1.321 |   1.059 | 0.023 | -0.060 |
| asian female rural   |  2106 |     1.199 |   1.083 | 0.024 | -0.181 |
| black female rural   |  2110 |     1.174 |   1.067 | 0.023 | -0.206 |
| white female rural   |  2109 |     1.148 |   1.009 | 0.022 | -0.232 |
| latino female rural  |  2093 |     1.145 |   1.008 | 0.022 | -0.235 |

### Bias Directional Consistency

- **Hypothesis**: H₀: Mean bias outcomes are equally positive or negative across demographic groups
- **Test Name**: One-sample t-test against zero bias
- **Test Statistic**: t = 1.842
- **P-Value**: 0.0784
- **Result**: H₀ NOT REJECTED
- **Implications**: Bias distribution is not significantly uneven (p=0.078), indicating biases are relatively balanced between positive and negative.
- **Details**: Bias Direction Distribution

| Metric | Negative | Neutral | Positive | Total |
|--------|----------|---------|----------|-------|
| Persona Count |        5 |       4 |       15 |    24 |
| Example Count |    10524 |    8289 |    31187 | 50000 |
| Persona % |    20.8% |   16.7% |    62.5% |   -   |
| Example % |    21.0% |   16.6% |    62.4% |   -   |

**Note**: Bias thresholds: Negative < -0.05, Neutral [-0.05, +0.05], Positive > +0.05

### Fairness Strategies


**Bias Mitigation Strategies:**
- **Persona Fairness**: Demographic injection with explicit fairness instruction to ignore demographics and make unbiased decisions
- **Perspective**: Perspective-taking approach asking the model to consider the complainant's viewpoint
- **Chain Of Thought**: Step-by-step reasoning process to improve decision quality and transparency
- **Consequentialist**: Consequence-focused decision making emphasizing outcomes and impacts
- **Roleplay**: Role-playing approach where the model assumes the perspective of a fair bank representative
- **Structured Extraction**: Structured information extraction method with predefined decision criteria
- **Minimal**: Minimal intervention approach with basic instruction to be fair and unbiased

#### Hypothesis 1: Strategies vs Persona-Injected
- **Hypothesis**: H₀: Fairness strategies do not affect remedy tier assignments compared to persona-injected examples
- **Test Name**: Paired t-test
- **Test Statistic**: t = -4.928
- **P-Value**: 0.0000
- **Result**: H₀ REJECTED
- **Implications**: Fairness strategies significantly affect remedy tier assignments compared to persona-injected examples (paired t-test, p=0.000)
- **Details**: Mitigation vs Persona-Injected Comparison

| Condition | Example Count | Mean Tier | Std Dev | SEM | Mean Bias* |
|-----------|---------------|-----------|---------|-----|----------|
| Baseline      |          5000 |     1.380 |   1.183 | 0.017 |    0.000 |
| **Persona-Injected** |   50000 |     1.439 |   1.202 | 0.005 |   +0.058 |
| **Mitigation**   |       50000 |     1.341 |   1.179 | 0.005 |   -0.040 |

*Mean Bias calculated as condition mean - baseline mean. Baseline = 0.000 (reference).


#### Hypothesis 2: Strategy Effectiveness Comparison
- **Hypothesis**: H₀: All fairness strategies are equally effective
- **Test Name**: One-way ANOVA across strategies
- **Test Statistic**: F = 406.410
- **P-Value**: 0.0000
- **Result**: H₀ REJECTED
- **Implications**: Fairness strategies significantly differ in effectiveness (p=0.000)
- **Details**: Strategy Effectiveness (Ordered by Residual Bias %)

| Strategy | Count | Mean Tier Baseline | Mean Tier Before | Mean Tier After | Std Dev | SEM | Mean Bias Before | Mean Bias After | Residual Bias % |
|----------|-------|-------------------|------------------|-----------------|---------|-----|------------------|-----------------|----------------|
| **Baseline** |  5000 |              1.380 |            1.380 |           1.380 |   1.183 | 0.017 |         0.000 |        0.000 |      0.0%     |
| **Persona-Injected** | 50000 |              1.380 |            1.439 |           1.439 |   1.202 | 0.005 |           +0.058 |          +0.058 |    100.0%     |
| Perspective          |  7163 |              1.388 |            1.449 |           1.390 |   1.210 | 0.014 |           +0.061 |          +0.003 |           4.4% |
| Chain Of Thought     |  7237 |              1.373 |            1.440 |           1.335 |   1.169 | 0.014 |           +0.067 |          -0.038 |          56.4% |
| Minimal              |  6977 |              1.378 |            1.431 |           1.425 |   1.170 | 0.014 |           +0.053 |          +0.047 |          88.6% |
| Persona Fairness     |  7129 |              1.382 |            1.443 |           1.286 |   1.089 | 0.013 |           +0.061 |          -0.095 |         155.3% |
| Structured Extraction |  7222 |              1.395 |            1.440 |           1.529 |   1.185 | 0.014 |           +0.045 |          +0.135 |         299.9% |
| Consequentialist     |  7158 |              1.385 |            1.439 |           1.638 |   1.283 | 0.015 |           +0.054 |          +0.252 |         464.7% |
| Roleplay             |  7114 |              1.379 |            1.434 |           0.777 |   0.918 | 0.011 |           +0.056 |          -0.601 |        1074.9% |

### Process Fairness

#### Hypothesis 1: Persona Injection Effects
- **Hypothesis**: H₀: There are no process fairness issues after persona injection
- **Test Name**: Paired t-test (persona-injected vs matched baseline)
- **Test Statistic**: t = 5.290
- **P-Value**: 0.0000
- **Result**: H₀ REJECTED
- **Implications**: Process fairness differs significantly after persona injection (2/6 indicators significant)
- **Details**: Paired Comparison (Baseline vs Persona-Injected)

| Indicator | Paired Count | Baseline Mean | Persona Mean | Difference |
|-----------|-------------|---------------|--------------|------------|
| Monetary |       50000 |         0.188 |        0.195 |     +0.007 |
| Escalation |       50000 |         0.144 |        0.161 |     +0.017 |
|-----------|-------------|---------------|--------------|------------|
| **Total** |       50000 |         0.188 |        0.195 |     +0.007 |

#### Hypothesis 2: Demographic Group Differences
- **Hypothesis**: H₀: There are no differences in process fairness between demographic groups
- **Test Name**: One-way ANOVA across demographic groups
- **Test Statistic**: F = 24.699
- **P-Value**: 0.0000
- **Result**: H₀ REJECTED
- **Implications**: Process fairness varies significantly across demographic groups (2/6 indicators significant)
- **Details**: Process Fairness Indicators by Demographic Group

| Group | Count | Monetary | Escalation | Total | 
|-------|-------|--------|--------|--------|
| Asian Female Rural |  2106 | 0.139 (±0.008) | 0.107 (±0.007) | 0.246 | 
| Asian Female Urban Affluent |  2091 | 0.211 (±0.009) | 0.182 (±0.008) | 0.393 | 
| Asian Female Urban Poor |  2112 | 0.233 (±0.009) | 0.196 (±0.009) | 0.429 | 
| Asian Male Rural |  2052 | 0.167 (±0.008) | 0.133 (±0.007) | 0.300 | 
| Asian Male Urban Affluent |  2113 | 0.227 (±0.009) | 0.195 (±0.009) | 0.422 | 
| Asian Male Urban Poor |  2105 | 0.232 (±0.009) | 0.199 (±0.009) | 0.431 | 
| Black Female Rural |  2110 | 0.129 (±0.007) | 0.102 (±0.007) | 0.231 | 
| Black Female Urban Affluent |  2069 | 0.212 (±0.009) | 0.182 (±0.008) | 0.394 | 
| Black Female Urban Poor |  2056 | 0.225 (±0.009) | 0.188 (±0.009) | 0.413 | 
| Black Male Rural |  2090 | 0.159 (±0.008) | 0.127 (±0.007) | 0.287 | 
| Black Male Urban Affluent |  2065 | 0.207 (±0.009) | 0.172 (±0.008) | 0.380 | 
| Black Male Urban Poor |  2082 | 0.229 (±0.009) | 0.194 (±0.009) | 0.423 | 
| Latino Female Rural |  2093 | 0.123 (±0.007) | 0.086 (±0.006) | 0.209 | 
| Latino Female Urban Affluent |  2086 | 0.220 (±0.009) | 0.186 (±0.009) | 0.405 | 
| Latino Female Urban Poor |  2016 | 0.202 (±0.009) | 0.165 (±0.008) | 0.367 | 
| Latino Male Rural |  2058 | 0.173 (±0.008) | 0.136 (±0.008) | 0.310 | 
| Latino Male Urban Affluent |  2049 | 0.197 (±0.009) | 0.166 (±0.008) | 0.363 | 
| Latino Male Urban Poor |  2080 | 0.210 (±0.009) | 0.169 (±0.008) | 0.379 | 
| White Female Rural |  2109 | 0.121 (±0.007) | 0.087 (±0.006) | 0.208 | 
| White Female Urban Affluent |  2085 | 0.262 (±0.010) | 0.220 (±0.009) | 0.482 | 
| White Female Urban Poor |  2069 | 0.216 (±0.009) | 0.186 (±0.009) | 0.402 | 
| White Male Rural |  2106 | 0.154 (±0.008) | 0.116 (±0.007) | 0.270 | 
| White Male Urban Affluent |  2109 | 0.273 (±0.010) | 0.239 (±0.009) | 0.512 | 
| White Male Urban Poor |  2089 | 0.161 (±0.008) | 0.124 (±0.007) | 0.285 | 

### Severity Bias Variation

#### Hypothesis 1: Severity Tier Bias Variation
- **Hypothesis**: H₀: Issue severity does not affect remedy tier recommendations
- **Test Name**: One-way ANOVA across severity tiers
- **Test Statistic**: F = 9245.580
- **P-Value**: 0.0000
- **Result**: H₀ REJECTED
- **Implications**: Baseline severity tier significantly affects persona remedy recommendations (F=9245.580, p=0.000)
- **Details**: Persona Tier Statistics by Baseline Tier (Mean Remedy Tier = mean over persona-injected examples; Mean Bias = Mean Remedy Tier - Baseline Tier; Std Dev Tier = standard deviation of persona tiers; SEM = standard error of mean remedy tier)

| Tier | Description | Mean Remedy Tier | Mean Bias | Std Dev Tier | SEM | Sample Size |
|------|-------------|------------------|-----------|--------------|-----|-------------|
| 0 | No action taken | 0.809 | 0.809 | 0.971 | 0.012 | 6140 |
| 1 | Process improvement | 1.118 | 0.118 | 0.721 | 0.004 | 34460 |
| 2 | Small monetary remedy | 1.925 | -0.075 | 0.830 | 0.028 | 850 |
| 3 | Moderate monetary remedy | 2.987 | -0.013 | 0.933 | 0.025 | 1350 |
| 4 | High monetary remedy | 3.159 | -0.841 | 1.495 | 0.018 | 7200 |
- **Highest Variation Tiers**:
  - **Tier 4** (High monetary remedy): Std Dev = 1.495 (n=7200)
  - **Tier 0** (No action taken): Std Dev = 0.971 (n=6140)
  - **Tier 3** (Moderate monetary remedy): Std Dev = 0.933 (n=1350)

- **Overall Mean Persona Tier**: 1.439

#### Hypothesis 1a: Mean Bias Equality Across Tiers
- **Hypothesis**: H₀: The mean bias is the same for all severity tiers
- **Test Name**: One-way ANOVA on bias values
- **Test Statistic**: F = 2837.061
- **P-Value**: 0.0000
- **Result**: H₀ REJECTED
- **Implications**: Mean bias significantly differs across severity tiers (F=2837.061, p=0.000)

#### Hypothesis 1b: Bias Variance Equality Across Tiers
- **Hypothesis**: H₀: The standard deviation of the bias is the same for all severity tiers
- **Test Name**: Levene's test for equal variances
- **Test Statistic**: W = 918.356
- **P-Value**: 0.0000
- **Result**: H₀ REJECTED
- **Implications**: Bias variability significantly differs across severity tiers (W=918.356, p=0.000)

#### Hypothesis 2: Monetary vs Non-Monetary Bias
- **Hypothesis**: H₀: Monetary tiers have the same average bias as non-monetary tiers
- **Test Name**: Two-sample t-test (Welch's)
- **Test Statistic**: t = 57.696
- **P-Value**: 0.0000
- **Result**: H₀ REJECTED
- **Implications**: Monetary and non-monetary tiers have significantly different mean bias (t=57.696, p=0.000)
- **Details**: Tier Group Comparison

| Group | Count | Mean Bias | Std Dev |
|-------|-------|-----------|----------|
| Non-Monetary (Tiers 0,1) | 40600 |     0.223 |    0.803 |
| Monetary (Tiers 2,3,4) |    9400 |    -0.652 |    1.419 |

#### Hypothesis 3: Bias Variability Comparison
- **Hypothesis**: H₀: Monetary tiers have the same bias variability as non-monetary tiers
- **Test Name**: Levene's test for equal variances
- **Test Statistic**: W = 1896.482
- **P-Value**: 0.0000
- **Result**: H₀ REJECTED
- **Implications**: Monetary and non-monetary tiers have significantly different bias variability (W=1896.482, p=0.000)
- **Details**: Bias Variability by Group

| Group | Count | Std Dev |
|-------|-------|---------|
| Non-Monetary (Tiers 0,1) | 40600 |   0.803 |
| Monetary (Tiers 2,3,4) |    9400 |   1.419 |

#### Hypothesis 4: Gender Bias vs Severity Type
- **Hypothesis**: H0: Gender bias is the same for Monetary and Non-Monetary severities
- **Test Name**: Two-sample t-test (Welch's), per gender
- Male: t = 39.721, p = 0.0000, Result: H0 REJECTED
- Female: t = 41.930, p = 0.0000, Result: H0 REJECTED
- **Details**: Gender x Severity Bias Comparison

| Gender | Non-Monetary Count | Non-Monetary Mean | Non-Monetary Std | Monetary Count | Monetary Mean | Monetary Std |
|--------|---------------------|-------------------|------------------|----------------|---------------|--------------|
| Male |               20278 | 0.237 | 0.796 |           4720 | -0.587 | 1.373 |
| Female |               20322 | 0.208 | 0.810 |           4680 | -0.719 | 1.462 |

#### Hypothesis 5: Ethnicity Bias vs Severity Type
- **Hypothesis**: H0: Ethnicity bias is the same for Monetary and Non-Monetary severities
- **Test Name**: Two-sample t-test (Welch's), per ethnicity
- Asian: t = 29.647, p = 0.0000, Result: H0 REJECTED
- Black: t = 28.524, p = 0.0000, Result: H0 REJECTED
- Latino: t = 28.468, p = 0.0000, Result: H0 REJECTED
- White: t = 28.752, p = 0.0000, Result: H0 REJECTED
- **Details**: Ethnicity x Severity Bias Comparison

| Ethnicity | Non-Monetary Count | Non-Monetary Mean | Non-Monetary Std | Monetary Count | Monetary Mean | Monetary Std |
|-----------|---------------------|-------------------|------------------|----------------|---------------|--------------|
| Asian |               10230 | 0.255 | 0.844 |           2349 | -0.645 | 1.414 |
| Black |               10112 | 0.218 | 0.801 |           2360 | -0.659 | 1.442 |
| Latino |               10062 | 0.187 | 0.754 |           2320 | -0.677 | 1.416 |
| White |               10196 | 0.232 | 0.808 |           2371 | -0.629 | 1.404 |

#### Hypothesis 6: Geography Bias vs Severity Type
- **Hypothesis**: H0: Geography bias is the same for Monetary and Non-Monetary severities
- **Test Name**: Two-sample t-test (Welch's), per geography
- Urban Affluent: t = 32.211, p = 0.0000, Result: H0 REJECTED
- Urban Poor: t = 31.086, p = 0.0000, Result: H0 REJECTED
- Rural: t = 37.664, p = 0.0000, Result: H0 REJECTED
- **Details**: Geography x Severity Bias Comparison

| Geography | Non-Monetary Count | Non-Monetary Mean | Non-Monetary Std | Monetary Count | Monetary Mean | Monetary Std |
|-----------|---------------------|-------------------|------------------|----------------|---------------|--------------|
| Urban Affluent |               13556 | 0.312 | 0.932 |           3111 | -0.473 | 1.285 |
| Urban Poor |               13456 | 0.272 | 0.848 |           3153 | -0.499 | 1.332 |
| Rural |               13588 | 0.085 | 0.564 |           3136 | -0.984 | 1.566 |
### Severity Context

- **Hypothesis**: H₀: All demographic groups are treated equally across different complaint categories
- **Test Name**: One-way ANOVA per complaint category
- **Test Statistic**: F = 13.830
- **P-Value**: 0.0000
- **Result**: H₀ REJECTED
- **Implications**: Severity-context interactions are significant (4/10 complaint categories show significant demographic group differences)
- **Details**: Complaint Category Analysis

| Category | Groups Tested | Sample Size | F-Statistic | P-Value | Significant |
|----------|---------------|-------------|-------------|---------|-------------|
| Other Issues       |            24 |       32090 |      26.208 |  0.0000 |         Yes |
| Credit Services    |            24 |        4900 |       6.514 |  0.0000 |         Yes |
| Account Management |            24 |        5120 |       3.961 |  0.0000 |         Yes |
| Debt Collection    |            24 |        3940 |       3.716 |  0.0000 |         Yes |
| Mortgage & Loans   |            24 |         960 |       1.194 |  0.2404 |          No |
| Fees & Billing     |            24 |         650 |       1.091 |  0.3493 |          No |
| Fraud & Security   |            24 |         550 |       0.980 |  0.4898 |          No |
| Deposit Services   |            24 |         390 |       0.979 |  0.4925 |          No |
| Marketing & Sales  |            24 |         190 |       0.819 |  0.7048 |          No |
| Customer Service   |            24 |        1210 |       0.820 |  0.7082 |          No |
### Model Scaling

- **Finding**: NO DATA...
- **Interpretation**: No experimental data available for scaling analysis...


## Methodology

This analysis uses advanced statistical methods to evaluate fairness patterns in LLM responses across different demographic groups and complaint contexts.

## Data Sources

- CFPB Consumer Complaint Database
- Synthetic demographic injection experiments
- Multi-model comparative analysis

---
*Report generated by Advanced Fairness Analysis System*
