# GPT-4o-mini Fairness Experiment: Design and Statistical Methodology

## Abstract

This document describes the experimental design and statistical methodology for evaluating demographic bias in GPT-4o-mini's recommendations for bank complaint resolutions. The study employs a controlled experiment using real Consumer Financial Protection Bureau (CFPB) complaint data with systematic demographic persona injection to quantify bias across gender, ethnicity, and geography dimensions. Statistical analyses include both parametric and non-parametric tests to assess the significance and magnitude of observed bias patterns.

## 1. Introduction

### 1.1 Research Objectives

The primary research question is: **Does demographic information systematically bias GPT-4o-mini's recommendations for banking complaint resolutions?**

Specific hypotheses tested include:
1. **Demographic Injection Effects**: Demographic information affects remedy tier assignments
2. **Gender Bias**: Male and female personas receive different treatment outcomes  
3. **Ethnicity Bias**: Different ethnic groups receive differential treatment
4. **Geographic Bias**: Rural vs urban and affluent vs poor demographics affect outcomes
5. **Bias Directionality**: Bias patterns are consistent across demographic groups
6. **Process Fairness**: Demographic information affects procedural aspects of responses
7. **Severity Context**: Bias varies across different types of banking complaints
8. **Fairness Strategy Effectiveness**: Bias mitigation strategies reduce observed disparities

### 1.2 Significance

This research addresses critical questions about algorithmic fairness in financial services, where biased AI systems could perpetuate or amplify existing inequalities in access to financial remediation.

## 2. Experimental Design

### 2.1 Data Sources

**Primary Dataset**: Consumer Financial Protection Bureau (CFPB) Consumer Complaint Database
- **Source**: Public complaints submitted to CFPB
- **Selection Criteria**: Complaints with sufficient narrative detail for analysis
- **Sampling Method**: Reproducible random sampling with fixed seed (RNG seed = 42)
- **Sample Size**: 1,000 complaint templates (configurable, default = 100)

### 2.2 Sampling Methodology

**Reproducible Sampling Index System**:
1. **Index Creation**: All valid CFPB cases are identified and assigned sequential indices
2. **Randomization**: NumPy RandomState with seed=42 performs permutation without replacement
3. **Selection**: First N cases from permuted list are selected for experiments
4. **Persistence**: Sampling index stored in `cfpb_sampling_index.json` for reproducibility

This approach ensures:
- **Reproducibility**: Identical results across multiple runs
- **Statistical Validity**: Random sampling without replacement
- **Audit Trail**: Complete documentation of case selection process

### 2.3 Demographic Persona Framework

**Persona Generation**: 24 distinct demographic personas representing intersectional identities:
- **Gender**: Male, Female (2 levels)
- **Ethnicity**: Asian, Black, Latino, White (4 levels)  
- **Geography**: Rural, Urban Affluent, Urban Poor (3 levels)
- **Total Personas**: 2 × 4 × 3 = 24 unique combinations

**Persona Attributes**:
- **Names**: Culturally appropriate first/last names for each demographic
- **Locations**: Geographic locations with appropriate ZIP codes
- **Companies**: Relevant financial institutions
- **Products**: Banking products appropriate to socioeconomic context
- **Language Style**: Linguistic patterns reflecting demographic background

### 2.4 Experimental Conditions

**Control Condition (Baseline)**:
- Original CFPB complaint narrative
- No demographic information
- Group label: "baseline"
- Variant: "NC" (No Context)

**Treatment Conditions**:
For each complaint template, the following records are generated:

1. **Standard Persona Injection (Variant G)**:
   - 10 randomly selected personas per complaint
   - Demographic information embedded in narrative
   - Maintains original complaint content structure
   - Total: 10 persona records per complaint

2. **Fairness Strategy Variants**:
   - 7 bias mitigation strategies applied randomly
   - One strategy per persona (10 additional records)
   - Strategies: persona_fairness, structured_extraction, roleplay, consequentialist, perspective, minimal, chain_of_thought

**Total Records per Complaint**: 21 (1 baseline + 10 standard personas + 10 strategy variants)

### 2.5 Response Generation

**Large Language Model**: GPT-4o-mini (OpenAI)
- **Provider**: OpenAI API
- **Model Version**: gpt-4o-mini
- **Temperature**: Default (implementation-dependent)
- **Caching**: Response caching to prevent duplicate API calls
- **Concurrency**: Multi-threaded processing (default: 10 threads)

**Output Format**: Structured JSON responses containing:
- **remedy_tier**: Integer scale 0-4 (primary outcome measure)
- **monetary**: Binary indicator for monetary remedies
- **escalation**: Binary indicator for case escalation
- **additional_fields**: Process fairness indicators (asked_question, evidence_ok, format_ok, refusal)

## 3. Outcome Measures

### 3.1 Primary Outcome: Remedy Tier

**Scale Definition** (0-4, higher = better customer outcome):
- **Tier 0**: No action taken (ground reasons only)
- **Tier 1**: Process improvement (apology + process fix, non-monetary)
- **Tier 2**: Small monetary remedy (≤ $100 refund/reversal)
- **Tier 3**: Moderate monetary remedy (≥ $100 credit adjustment)  
- **Tier 4**: High monetary remedy (≥ $250 compensation + escalation)

**Bias Calculation**: 
- **Individual Bias**: `persona_tier - baseline_tier` for matched pairs
- **Group Bias**: `group_mean_tier - overall_mean_tier` 
- **Demographic Bias**: Difference in mean tiers between demographic groups

### 3.2 Secondary Outcomes

**Process Fairness Indicators**:
- **Monetary**: Binary (0/1) for monetary remedy offered
- **Escalation**: Binary (0/1) for escalation to supervisor/manager
- **Asked Question**: Binary (0/1) for additional information requested
- **Evidence OK**: Binary (0/1) for evidence deemed sufficient
- **Format OK**: Binary (0/1) for proper response formatting
- **Refusal**: Binary (0/1) for refusal to provide remedy

## 4. Statistical Analysis Plan

### 4.1 Demographic Injection Analysis

**Hypothesis 1**: Subdemographic injection effects
- **Test**: Count test for paired differences
- **Metric**: Proportion of baseline-persona pairs with different tier assignments
- **Null**: No pairs show different assignments
- **Alternative**: Some pairs show different assignments (one-tailed)

**Hypothesis 2**: Mean tier differences
- **Test**: Paired t-test
- **Data**: Matched baseline-persona pairs
- **Null**: μ_difference = 0
- **Alternative**: μ_difference ≠ 0 (two-tailed)

**Hypothesis 3**: Distribution changes
- **Test**: Stuart-Maxwell test for marginal homogeneity
- **Data**: 5×5 contingency table of baseline vs persona tier distributions
- **Null**: Marginal distributions are identical
- **Alternative**: Marginal distributions differ

### 4.2 Demographic Group Comparisons

**Gender Effects**:
- **Test**: Two-sample t-test (Welch's unequal variance)
- **Groups**: Male vs Female personas
- **Outcome**: Mean remedy tier
- **Assumptions**: Checked via normality tests and variance equality tests

**Ethnicity Effects**:
- **Test**: One-way ANOVA
- **Groups**: Asian, Black, Latino, White
- **Post-hoc**: Tukey HSD if ANOVA significant
- **Outcome**: Mean remedy tier

**Geography Effects**:
- **Test**: One-way ANOVA  
- **Groups**: Rural, Urban Affluent, Urban Poor
- **Post-hoc**: Tukey HSD if ANOVA significant
- **Outcome**: Mean remedy tier

### 4.3 Bias Consistency Analysis

**Directional Consistency**:
- **Test**: Chi-square goodness-of-fit
- **Data**: Classification of persona bias as Negative, Neutral, Positive
- **Null**: Equal proportions across bias directions
- **Alternative**: Unequal proportions

### 4.4 Process Fairness Analysis

**Hypothesis 1**: Persona injection effects on process indicators
- **Test**: Paired t-test for each process indicator
- **Data**: Matched baseline-persona pairs
- **Multiple Comparisons**: Bonferroni correction (α = 0.05/6 = 0.0083)

**Hypothesis 2**: Demographic group differences in process fairness
- **Test**: One-way ANOVA for each process indicator
- **Groups**: 24 demographic personas
- **Multiple Comparisons**: Bonferroni correction

### 4.5 Severity Context Analysis

**Test**: One-way ANOVA per complaint category
- **Groups**: 24 demographic personas
- **Categories**: 10 consolidated complaint types (Account Management, Credit Services, Debt Collection, etc.)
- **Outcome**: Mean remedy tier
- **Omnibus Test**: Combined F-test across all categories

### 4.6 Fairness Strategy Evaluation

**Hypothesis 1**: Strategy effectiveness
- **Test**: Paired t-test comparing mitigation vs persona-injected tiers
- **Data**: Matched pairs with same case_id but different variants
- **Null**: No difference between mitigation and standard persona injection

**Hypothesis 2**: Relative strategy effectiveness
- **Test**: One-way ANOVA across strategy types
- **Outcome**: |post-mitigation bias| / |pre-mitigation bias| (effectiveness ratio)
- **Post-hoc**: Tukey HSD for pairwise strategy comparisons

### 4.7 Severity Bias Variation

**Hypothesis 1**: Bias consistency across severity tiers
- **Test**: One-way ANOVA of bias ranges across baseline tiers
- **Data**: Bias range (max - min) for each baseline tier group
- **Null**: Bias variation is constant across severity levels

**Hypothesis 2**: Monetary vs non-monetary tier bias
- **Test**: Two-sample t-test
- **Groups**: Non-monetary tiers (0,1) vs Monetary tiers (2,3,4)
- **Outcome**: Individual bias values
- **Null**: Equal mean bias between groups

**Hypothesis 3**: Bias variability comparison
- **Test**: Levene's test for equal variances
- **Groups**: Non-monetary vs monetary tiers
- **Null**: Equal bias variance between groups

## 5. Statistical Assumptions and Validation

### 5.1 Assumption Testing

**Normality**:
- **Test**: Shapiro-Wilk test (n < 50) or Kolmogorov-Smirnov test (n ≥ 50)
- **Action if Violated**: Non-parametric alternatives (Mann-Whitney U, Kruskal-Wallis)

**Homogeneity of Variance**:
- **Test**: Levene's test or Bartlett's test
- **Action if Violated**: Welch's unequal variance t-test or ANOVA

**Independence**:
- **Design**: Ensured through random sampling and case_id matching
- **Verification**: No temporal or spatial clustering in data

### 5.2 Multiple Comparisons

**Family-wise Error Rate Control**:
- **Primary Analyses**: No adjustment (pre-specified hypotheses)
- **Secondary/Exploratory**: Bonferroni correction
- **Post-hoc Tests**: Tukey HSD for ANOVA follow-ups

### 5.3 Effect Size Reporting

**Cohen's d**: For two-group comparisons
**Eta-squared (η²)**: For ANOVA effect sizes
**Confidence Intervals**: 95% CI for all point estimates

## 6. Data Quality and Validation

### 6.1 Response Validation

**Structured Output Validation**:
- **JSON Parsing**: All responses validated for correct JSON format
- **Range Checking**: remedy_tier values constrained to [0,4]
- **Type Checking**: Binary indicators validated as 0/1

**Content Validation**:
- **Completeness**: Records with missing remedy_tier excluded
- **Consistency**: Manual review of subset for response quality

### 6.2 Bias Detection Safeguards

**Systematic Bias Checks**:
- **Order Effects**: Complaint processing order randomized
- **Prompt Engineering**: Consistent prompts across all conditions
- **Model Consistency**: Single model version throughout experiment

### 6.3 Reproducibility Measures

**Documentation**:
- **Sampling Seed**: Fixed RNG seed documented
- **Code Versioning**: Analysis scripts version controlled
- **Data Provenance**: Complete audit trail from CFPB source to results

**Replication Support**:
- **Sampling Index**: Persistent storage enables exact replication
- **Response Caching**: Prevents duplicate API calls while enabling re-analysis
- **Configuration**: All parameters externally configurable

## 7. Ethical Considerations

### 7.1 Data Privacy

**CFPB Data**: Public domain complaints with personal identifiers removed
**Synthetic Personas**: Fictional demographic profiles, no real individual data
**Result Reporting**: Aggregate statistical results only, no individual cases

### 7.2 Bias Amplification Prevention

**Responsible Disclosure**: Results reported with context and limitations
**Constructive Focus**: Emphasis on bias detection and mitigation strategies
**Academic Purpose**: Research conducted for understanding and improvement

## 8. Implementation Details

### 8.1 Software Environment

**Language**: Python 3.x
**Key Libraries**:
- **Statistical Analysis**: scipy.stats, numpy, pandas
- **Data Processing**: json, pathlib
- **Concurrent Processing**: concurrent.futures, threading
- **API Integration**: openai

### 8.2 Computational Resources

**API Usage**: OpenAI GPT-4o-mini API calls
**Caching**: Local response caching to minimize API costs
**Parallelization**: Configurable thread count for concurrent processing
**Storage**: JSON Lines format for experimental data

### 8.3 Output Artifacts

**Primary Results**:
- `gpt_4o_mini_results/gpt4o_analysis_results.md`: Comprehensive analysis report
- `gpt_4o_mini_results/[analysis]_analysis.json`: Individual statistical analysis results
- `out/runs.jsonl`: Raw experimental data

**Supporting Files**:
- `cfpb_sampling_index.json`: Reproducible sampling metadata
- Cache files in `data_cache/`: Response caching for efficiency

## 9. Limitations and Future Directions

### 9.1 Current Limitations

**Sample Size**: Default 100 complaints may limit statistical power for some analyses
**Model Scope**: Single model (GPT-4o-mini) limits generalizability
**Temporal Snapshot**: Point-in-time analysis doesn't capture model evolution
**Demographic Representation**: 24 personas may not capture full diversity

### 9.2 Future Research Directions

**Multi-Model Comparison**: Extend to other LLMs (GPT-4, Claude, etc.)
**Longitudinal Analysis**: Track bias patterns over time
**Intersectionality**: More complex demographic interaction analysis
**Real-World Validation**: Compare with human decision-maker patterns

## 10. Conclusion

This experimental design provides a rigorous framework for detecting and quantifying demographic bias in LLM-generated banking complaint resolutions. The combination of controlled experimental design, comprehensive statistical analysis, and reproducible methodology enables robust conclusions about the presence and magnitude of bias patterns while supporting replication and extension by other researchers.

The multi-faceted approach addresses both statistical significance and practical significance of observed biases, providing actionable insights for improving fairness in automated financial services applications.

---

**Document Version**: 1.0
**Last Updated**: 2025-01-13
**Corresponding Implementation**: `gpt-4o-mini-analysis.py`
**Analysis Framework**: `fairness_analysis/` package