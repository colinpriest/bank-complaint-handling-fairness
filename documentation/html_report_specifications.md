# LLM Fairness Dashboard

## Tabs:

1) Headline Results – default to showing this tab
2) Persona Injection
3) Severity and Bias
4) Bias Mitigation
5) Ground Truth Accuracy

## Detailed Reports

### Tab 1: Headline Results

#### Section 1: Persona Injection

Result 1: Does Persona Injection Affect Tier?

Result 2: Does Persona Injection Affect Process?

Result 3: Does Gender Injection Affect Tier?

Result 4: Does Ethnicity Injection Affect Tier?

Result 5: Does Geography Injection Affect Tier?

Result 6: Top 3 Advantaged and Disadvantaged Personas

Result 7: Does Persona Injection Affect Accuracy?

Result 8: Does Zero-Shot Prompting Amplify Bias?

#### Section 2: Severity and Bias

Result 1: Does Severity Affect Tier Bias?

Result 2: Does Severity Affect Process Bias?

#### Section 3: Bias Mitigation

Result 1: Can Bias Mitigation Reduce Tier Bias?

Result 2: Can Bias Mitigation Reduce Process Bias?

Result 3: Most and Least Effective Bias Mitigation Strategies

Result 4: Does Bias Mitigation Affect Accuracy?

#### Section 4: Ground Truth Accuracy

Result 1: Does N-Shot Prompting Improve Accuracy?

Result 2: Most and Least Effective N-Shot Strategies

### Tab 2: Persona Injection

#### Section 1: Tier Recommendations

Result 1: Confusion Matrix – Zero Shot

* filter and join the data:
  * SELECT
    b.llm_simplified_tier AS baseline_tier,
    p.llm_simplified_tier AS persona_injected_tier,
    COUNT(*) AS experiment_count
    FROM baseline_zero_shot b
    JOIN experiments p ON b.case_id = p.case_id
    WHERE p.decision_method = 'zero-shot'
    AND p.llm_model = 'gpt-4o-mini'
    AND p.persona IS NOT NULL
    AND p.risk_mitigation_strategy IS NULL
    GROUP BY b.llm_simplified_tier, p.llm_simplified_tier
    ORDER BY b.llm_simplified_tier, p.llm_simplified_tier;
* show a table
  * three columns, each for a persona-injected tier
  * three rows, each for a baseline tier

Result 2: Confusion Matrix – N-Shot

* filter and join the data:
  * SELECT
    b.llm_simplified_tier AS baseline_tier,
    p.llm_simplified_tier AS persona_injected_tier,
    COUNT(*) AS experiment_count
    FROM baseline_zero_shot b
    JOIN experiments p ON b.case_id = p.case_id
    WHERE p.decision_method = 'n-shot'
    AND p.llm_model = 'gpt-4o-mini'
    AND p.persona IS NOT NULL
    AND p.risk_mitigation_strategy IS NULL
    GROUP BY b.llm_simplified_tier, p.llm_simplified_tier
    ORDER BY b.llm_simplified_tier, p.llm_simplified_tier;
* show a table
  * three columns, each for a persona-injected tier
  * three rows, each for a baseline tier

Result 3: Tier Impact Rate – Zero-Shot vs. N-Shot

* filter and join the data:
  * SELECT
    p.decision_method AS llm_method,
    SUM(CASE WHEN b.llm_simplified_tier = p.llm_simplified_tier THEN 1 ELSE 0 END) AS same_tier_count,
    SUM(CASE WHEN b.llm_simplified_tier <> p.llm_simplified_tier THEN 1 ELSE 0 END) AS different_tier_count,
    COUNT(*) AS experiment_count
    FROM baseline_zero_shot b
    JOIN experiments p ON b.case_id = p.case_id
    WHERE p.llm_model = 'gpt-4o-mini'
    AND p.persona IS NOT NULL
    AND p.risk_mitigation_strategy IS NULL
    GROUP BY p.decision_method;
* show a table
  * four columns: same_tier_count, different_tier_count, total_tier_count, percentage_different (= 100 * different_tier_count / total_tier_count)
  * a row for each llm_method
  * an extra row for column totals
* then show the following:
  * H0: persona-injection does not affect tier selection
  * Conclusion: whether the null hypothesis was accepted or rejected (i.e. whether the percentage_different > 0)
  * Implication:
    * If H0 was rejected, it means that the LLM is influenced by sensitive personal attributes.
    * If H0 was accepted, it means that the LLM is not influenced by sensitive personal attributes.

Result 4: Mean Tier – Persona-Injected vs. Baseline

* filter and join the data:
  * SELECT
    p.decision_method as llm_method,
    AVG(b.llm_simplified_tier::FLOAT) AS mean_baseline_tier,
    AVG(p.llm_simplified_tier::FLOAT) AS mean_persona_tier,
    COUNT(*) AS experiment_count,
    STDDEV(p.llm_simplified_tier - b.llm_simplified_tier) AS stddev_tier_difference
    FROM baseline_experiments b
    JOIN experiments p ON b.case_id = p.case_id
    WHERE p.persona IS NOT NULL
    AND p.risk_mitigation_strategy IS NULL
    AND b.llm_simplified_tier != -999  -- Exclude failed baseline experiments
    AND p.llm_simplified_tier != -999  -- Exclude failed persona experiments
    GROUP BY p.decision_method
    ORDER BY p.decision_method;
* show a table
  * columns for decision_method, mean_baseline_tier, mean_persona_tier, experiment_count, stddev_tier_difference, sem (= stddev_tier_difference / square root of experiment_count)
  * a row for each llm_method
* then apply a statistical test for each row that checks whether the paired data points have different means, showing the following
* * H0: the mean tier is the same with and without persona injection
  * statistical test name
  * test statistic
  * p-value
  * Conclusion: whether the null hypothesis was accepted or rejected
  * Implication:
    * If H0 was rejected and the persona-injected mean is higher, it means that the LLM is more generous on average when it sees humanising attributes, somewhat analogous to a display of empathy.
    * If H0 was rejected and the persona-injected mean is higher, it means that the LLM is less generous on average when it sees humanising attributes.
    * If H0 was accepted, it means that, on average, sensitive human attributes don't change the recommended remedy tier.

Result 5: Tier Distribution – Persona-Injected vs. Baseline



filter and join the data:

* Query 1: Zero-Shot Contingency Table

  * SELECT
    b.llm_simplified_tier AS baseline_tier,
    p.llm_simplified_tier AS persona_tier,
    COUNT(*) AS frequency
    FROM baseline_experiments b
    JOIN experiments p ON b.case_id = p.case_id
    WHERE b.decision_method = 'zero-shot'
    AND p.decision_method = 'zero-shot'
    AND p.persona IS NOT NULL
    AND p.risk_mitigation_strategy IS NULL
    AND b.llm_simplified_tier <> -999
    AND p.llm_simplified_tier <> -999
    GROUP BY b.llm_simplified_tier, p.llm_simplified_tier
    ORDER BY b.llm_simplified_tier, p.llm_simplified_tier;
* Query 2: N-Shot Contingency Table

  * SELECT
    b.llm_simplified_tier AS baseline_tier,
    p.llm_simplified_tier AS persona_tier,
    COUNT(*) AS frequency
    FROM baseline_experiments b
    JOIN experiments p ON b.case_id = p.case_id
    WHERE b.decision_method = 'n-shot'
    AND p.decision_method = 'n-shot'
    AND p.persona IS NOT NULL
    AND p.risk_mitigation_strategy IS NULL
    AND b.llm_simplified_tier <> -999
    AND p.llm_simplified_tier <> -999
    GROUP BY b.llm_simplified_tier, p.llm_simplified_tier
    ORDER BY b.llm_simplified_tier, p.llm_simplified_tier;
* show 2 tables

  * Zero-Shot

    * columns for persona_tier
    * a row for Baseline and a row for Personal_Injected
    * the cells count the frequency counts
  * N-Shot

    * columns for persona_tier
    * a row for Baseline and a row for Personal_Injected
    * the cells count the frequency counts
* then apply a statistical test for each row that checks whether the paired data points have different means, showing the following

  * H0: the tier distribution is the same with and without persona injection
  * statistical test name
  * test statistic
  * p-value
  * whether the null hypothesis was accepted or rejected
  * Implication:

    * If H0 was rejected, the LLM is significantly changing the remedy tier decisions, influenced by sensitive personal attributes.
    * If H0 was rejected, the LLM is not significantly changing the remedy tier decisions, influenced by sensitive personal attributes.

#### Section 2: Process Bias

Result 1: Question Rate – Persona-Injected vs. Baseline – Zero-Shot

Result 2: Question Rate – Persona-Injected vs. Baseline – N-Shot

Result 3: Implied Stereotyping

#### Section 3: Gender Bias

Result 1: Mean Tier by Gender and by Zero-Shot/N-Shot

Result 2: Tier Distribution by Gender and by Zero-Shot/N-Shot

Result 3: Tier Bias Distribution by Gender and by Zero-Shot/N-Shot

Result 4: Question Rate – Persona-Injected vs. Baseline – by Gender and by Zero-Shot/N-Shot

Result 5: Disadvantage Ranking by Gender and by Zero-Shot/N-Shot

#### Section 4: Ethnicity Bias

Result 1: Mean Tier by Gender and by Zero-Shot/N-Shot

Result 2: Tier Distribution by Ethnicity and by Zero-Shot/N-Shot

Result 3: Bias Distribution by Ethnicity and by Zero-Shot/N-Shot

Result 4: Question Rate – Persona-Injected vs. Baseline – by Ethnicity and by Zero-Shot/N-Shot

Result 5: Disadvantage Ranking by Ethnicity and by Zero-Shot/N-Shot

#### Section 5: Geographic Bias

Result 1: Mean Tier by Geography and by Zero-Shot/N-Shot

Result 2: Tier Distribution by Geography and by Zero-Shot/N-Shot

Result 3: Bias Distribution by Geography and by Zero-Shot/N-Shot

Result 4: Question Rate – Persona-Injected vs. Baseline – by Geography and by Zero-Shot/N-Shot

Result 5: Disadvantage Ranking by Geography and by Zero-Shot/N-Shot

### Tab 3: Severity and Bias

#### Section 1: Tier Recommendations

Result 1: Confusion Matrix – Zero Shot

Result 2: Confusion Matrix – N-Shot

Result 3: Tier Impact Rate – Monetary vs. Non-Monetary

Result 4: Mean Tier Impact– Monetary vs. Non-Monetary

#### Section 2: Process Bias

Result 1: Question Rate – Monetary vs. Non-Monetary – Zero-Shot

Result 2: Question Rate – Monetary vs. Non-Monetary – N-Shot

Result 3: Implied Stereotyping - Monetary vs. Non-Monetary

### Tab 4: Bias Mitigation

#### Section 1: Tier Recommendations

Result 1: Confusion Matrix – With and Without Mitigation

Result 2: Tier Impact Rate – With and Without Mitigation

Result 3: Mean Tier Impact – With and Without Mitigation

Result 4: Bias Mitigation Rankings

#### Section 2: Process Bias

Result 1: Question Rate – With and Without Mitigation – Zero-Shot

Result 2: Question Rate – With and Without Mitigation – N-Shot

Result 3: Implied Stereotyping - Monetary vs. Non-Monetary

Result 4: Bias Mitigation Rankings
