"""
Statistical analysis methods for fairness testing
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Suppress scipy precision warnings for nearly identical data
warnings.filterwarnings('ignore', message='Precision loss occurred in moment calculation', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='Each of the input arrays is constant', category=RuntimeWarning)


class StatisticalAnalyzer:
    """Handles all statistical analysis for fairness testing"""
    
    def __init__(self):
        self.alpha = 0.05  # Significance level
        
    def analyze_demographic_injection_effect(self, raw_results: List[Dict]) -> Dict:
        """
        Hypothesis 0: Test whether demographic injection affects LLM decisions
        H₀: Subtle demographic injection does not affect remedy tier assignments
        H₁: Demographic information significantly influences LLM decisions
        """
        if not raw_results:
            return {
                "hypothesis": "H₀: Subtle demographic injection does not affect remedy tier assignments",
                "finding": "NOT TESTED - No raw experimental data available",
                "baseline_mean": float('nan'),
                "personas_mean": float('nan'), 
                "mean_difference": float('nan'),
                "test_statistic": float('nan'),
                "p_value": float('nan'),
                "effect_size": float('nan'),
                "interpretation": "No raw experimental data available. Run experiments first with --run-experiment or --full",
                "note": "This analysis requires raw experimental results"
            }
            
        try:
            # Extract real remedy_tier values for baseline and persona groups
            # Real data uses: variant="NC" for baseline, "G" and "persona_fairness" for personas
            baseline_tiers = [
                item['remedy_tier'] for item in raw_results 
                if item.get('variant') == 'NC' and 'remedy_tier' in item
            ]
            
            # Persona conditions: all non-baseline variants (G and persona_fairness)
            persona_tiers = [
                item['remedy_tier'] for item in raw_results 
                if item.get('variant') in ['G', 'persona_fairness'] and 'remedy_tier' in item
            ]
            
            if len(baseline_tiers) < 10 or len(persona_tiers) < 10:
                return {
                    "hypothesis": "H₀: Subtle demographic injection does not affect remedy tier assignments",
                    "finding": "NOT TESTED - Insufficient sample data",
                    "baseline_mean": float('nan'),
                    "personas_mean": float('nan'), 
                    "mean_difference": float('nan'),
                    "test_statistic": float('nan'),
                    "p_value": float('nan'),
                    "effect_size": float('nan'),
                    "interpretation": "Insufficient sample data for reliable statistical analysis",
                    "note": f"Need at least 10 samples each. Got baseline={len(baseline_tiers)}, persona={len(persona_tiers)}"
                }
            
            # Perform independent t-test
            from scipy.stats import ttest_ind
            t_stat, p_value = ttest_ind(baseline_tiers, persona_tiers)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(baseline_tiers) - 1) * np.var(baseline_tiers, ddof=1) + 
                                (len(persona_tiers) - 1) * np.var(persona_tiers, ddof=1)) / 
                               (len(baseline_tiers) + len(persona_tiers) - 2))
            cohens_d = (np.mean(baseline_tiers) - np.mean(persona_tiers)) / pooled_std
            
            # Determine finding
            finding = "H₀ REJECTED" if p_value < self.alpha else "H₀ NOT REJECTED"
            interpretation = f"Significant effect of demographic injection detected (p={p_value:.3f})" if p_value < self.alpha else f"No significant effect of demographic injection (p={p_value:.3f})"
            
            return {
                "hypothesis": "H₀: Subtle demographic injection does not affect remedy tier assignments",
                "finding": finding,
                "baseline_mean": float(np.mean(baseline_tiers)),
                "personas_mean": float(np.mean(persona_tiers)),
                "mean_difference": float(np.mean(baseline_tiers) - np.mean(persona_tiers)),
                "test_statistic": float(t_stat),
                "p_value": float(p_value),
                "effect_size": float(cohens_d),
                "interpretation": interpretation,
                "sample_sizes": f"baseline={len(baseline_tiers)}, persona={len(persona_tiers)}"
            }
            
        except Exception as e:
            return {
                "hypothesis": "H₀: Subtle demographic injection does not affect remedy tier assignments",
                "finding": "ERROR",
                "baseline_mean": float('nan'),
                "personas_mean": float('nan'),
                "mean_difference": float('nan'),
                "test_statistic": float('nan'),
                "p_value": float('nan'),
                "effect_size": float('nan'),
                "interpretation": f"Analysis failed: {str(e)}",
                "error": str(e)
            }
    
    def analyze_gender_effects(self, raw_results: List[Dict]) -> Dict:
        """Analyze gender injection effects and gender bias patterns"""
        if not raw_results:
            return {"finding": "NOT TESTED", "error": "No raw experimental data available"}
        
        try:
            # Extract data by gender (inferred from group_label)
            male_data = [r for r in raw_results if 'male' in r.get('group_label', '').lower()]
            female_data = [r for r in raw_results if 'female' in r.get('group_label', '').lower()]
            
            if not male_data or not female_data:
                return {"finding": "NOT TESTED", "error": "Insufficient gender data"}
            
            # Extract remedy tiers
            male_tiers = [r.get('remedy_tier') for r in male_data if r.get('remedy_tier') is not None]
            female_tiers = [r.get('remedy_tier') for r in female_data if r.get('remedy_tier') is not None]
            
            if len(male_tiers) < 10 or len(female_tiers) < 10:
                return {"finding": "NOT TESTED", "error": "Insufficient sample data"}
            
            # Perform independent t-test
            from scipy.stats import ttest_ind
            t_stat, p_value = ttest_ind(male_tiers, female_tiers)
            
            # Calculate effect size
            pooled_std = np.sqrt(((len(male_tiers) - 1) * np.var(male_tiers, ddof=1) + 
                                (len(female_tiers) - 1) * np.var(female_tiers, ddof=1)) / 
                               (len(male_tiers) + len(female_tiers) - 2))
            cohens_d = (np.mean(male_tiers) - np.mean(female_tiers)) / pooled_std
            
            finding = "H₀ REJECTED" if p_value < 0.05 else "H₀ NOT REJECTED"
            
            return {
                "finding": finding,
                "male_mean": float(np.mean(male_tiers)),
                "female_mean": float(np.mean(female_tiers)),
                "mean_difference": float(np.mean(male_tiers) - np.mean(female_tiers)),
                "test_statistic": float(t_stat),
                "p_value": float(p_value),
                "effect_size": float(cohens_d),
                "interpretation": f"Gender {'significantly affects' if p_value < 0.05 else 'does not significantly affect'} remedy tier assignments",
                "sample_sizes": {"male": len(male_tiers), "female": len(female_tiers)}
            }
            
        except Exception as e:
            return {"finding": "ERROR", "error": f"Analysis failed: {str(e)}"}
    
    def analyze_ethnicity_effects(self, raw_results: List[Dict]) -> Dict:
        """Analyze ethnicity injection effects and ethnic bias patterns"""
        if not raw_results:
            return {"finding": "NOT TESTED", "error": "No raw experimental data available"}
        
        try:
            # Extract data by ethnicity (inferred from group_label)
            ethnicity_groups = {}
            for record in raw_results:
                group_label = record.get('group_label', '')
                if 'black' in group_label.lower():
                    ethnicity = 'black'
                elif 'hispanic' in group_label.lower():
                    ethnicity = 'hispanic'
                elif 'white' in group_label.lower():
                    ethnicity = 'white'
                elif 'asian' in group_label.lower():
                    ethnicity = 'asian'
                else:
                    continue
                
                if ethnicity not in ethnicity_groups:
                    ethnicity_groups[ethnicity] = []
                if record.get('remedy_tier') is not None:
                    ethnicity_groups[ethnicity].append(record.get('remedy_tier'))
            
            if len(ethnicity_groups) < 2:
                return {"finding": "NOT TESTED", "error": "Insufficient ethnicity data"}
            
            # Perform one-way ANOVA
            from scipy.stats import f_oneway
            groups = list(ethnicity_groups.values())
            f_stat, p_value = f_oneway(*groups)
            
            # Calculate means for each ethnicity
            ethnicity_means = {eth: float(np.mean(tiers)) for eth, tiers in ethnicity_groups.items()}
            
            finding = "H₀ REJECTED" if p_value < 0.05 else "H₀ NOT REJECTED"
            
            return {
                "finding": finding,
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "ethnicity_means": ethnicity_means,
                "interpretation": f"Ethnicity {'significantly affects' if p_value < 0.05 else 'does not significantly affect'} remedy tier assignments",
                "sample_sizes": {eth: len(tiers) for eth, tiers in ethnicity_groups.items()}
            }
            
        except Exception as e:
            return {"finding": "ERROR", "error": f"Analysis failed: {str(e)}"}
    
    def analyze_geography_effects(self, raw_results: List[Dict]) -> Dict:
        """Analyze geography injection effects and geographic bias patterns"""
        if not raw_results:
            return {"finding": "NOT TESTED", "error": "No raw experimental data available"}
        
        try:
            # Extract data by geography (3 categories: urban affluent, urban poor, rural)
            geography_groups = {}
            for record in raw_results:
                group_label = record.get('group_label', '').lower()
                
                # Categorize based on group_label patterns
                if 'urban' in group_label:
                    geography = 'urban_affluent'  # Default urban to affluent
                elif 'affluent' in group_label:
                    geography = 'urban_affluent'
                elif 'working' in group_label:
                    geography = 'urban_poor'  # Working class typically indicates urban poor
                elif 'rural' in group_label:
                    geography = 'rural'
                else:
                    # Try to infer from other patterns
                    if 'affluent' in group_label or 'beverly' in group_label or 'greenwich' in group_label or 'palo alto' in group_label or 'scarsdale' in group_label or 'wellesley' in group_label:
                        geography = 'urban_affluent'
                    elif 'working' in group_label or 'el paso' in group_label or 'miami' in group_label or 'phoenix' in group_label or 'san antonio' in group_label or 'santa ana' in group_label:
                        geography = 'urban_poor'
                    else:
                        continue  # Skip if can't categorize
                
                if geography not in geography_groups:
                    geography_groups[geography] = []
                if record.get('remedy_tier') is not None:
                    geography_groups[geography].append(record.get('remedy_tier'))
            
            if len(geography_groups) < 2:
                return {"finding": "NOT TESTED", "error": "Insufficient geography data"}
            
            # Perform one-way ANOVA for multiple groups
            from scipy.stats import f_oneway
            groups = list(geography_groups.values())
            f_stat, p_value = f_oneway(*groups)
            
            # Calculate means for each geography
            geography_means = {geo: float(np.mean(tiers)) for geo, tiers in geography_groups.items()}
            
            finding = "H₀ REJECTED" if p_value < 0.05 else "H₀ NOT REJECTED"
            
            return {
                "finding": finding,
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "geography_means": geography_means,
                "interpretation": f"Geography {'significantly affects' if p_value < 0.05 else 'does not significantly affect'} remedy tier assignments",
                "sample_sizes": {geo: len(tiers) for geo, tiers in geography_groups.items()}
            }
            
        except Exception as e:
            return {"finding": "ERROR", "error": f"Analysis failed: {str(e)}"}
    
    def analyze_granular_bias(self, raw_results: List[Dict]) -> Dict:
        """
        Analyze granular bias patterns across demographic groups
        
        Hypothesis: H₀: Subtle demographic injection affects remedy tier assignments the same for all groups
        """
        if not raw_results:
            return {
                "finding": "NOT TESTED", 
                "error": "No raw experimental data available"
            }
        
        try:
            # Extract remedy tiers by demographic group (ethnicity × gender × urban/rural)
            group_tiers = {}
            for record in raw_results:
                group_label = record.get('group_label')
                remedy_tier = record.get('remedy_tier')
                
                if group_label and remedy_tier is not None:
                    if group_label not in group_tiers:
                        group_tiers[group_label] = []
                    group_tiers[group_label].append(remedy_tier)
            
            # Filter out baseline and focus on persona groups
            persona_groups = {k: v for k, v in group_tiers.items() if k != 'baseline'}
            
            if len(persona_groups) < 2:
                return {"finding": "NOT TESTED", "error": "Insufficient persona groups for analysis"}
            
            # Calculate bias magnitudes for each persona group
            baseline_tiers = group_tiers.get('baseline', [])
            if not baseline_tiers:
                return {"finding": "NOT TESTED", "error": "No baseline data available"}
            
            baseline_mean = np.mean(baseline_tiers)
            
            # Test for significant differences between persona groups
            group_data = [tiers for tiers in persona_groups.values()]
            f_stat, p_value = stats.f_oneway(*group_data)
            
            return {
                "finding": "H₀ REJECTED" if p_value < self.alpha else "H₀ NOT REJECTED",
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "interpretation": "Significant inter-group bias differences detected" if p_value < self.alpha else "No significant inter-group bias differences",
                "persona_groups_analyzed": len(persona_groups),
                "baseline_mean": float(baseline_mean),
                "bias_magnitudes": {group: float(np.mean(tiers) - baseline_mean) for group, tiers in persona_groups.items()},
                "group_means": {group: float(np.mean(tiers)) for group, tiers in persona_groups.items()}
            }
            
        except Exception as e:
            return {"finding": "ERROR", "error": f"Analysis failed: {str(e)}"}
    
    def analyze_bias_directional_consistency(self, raw_results: List[Dict]) -> Dict:
        """
        Analyze directional consistency of bias patterns using statistical testing
        
        Hypothesis: H₀: Mean bias outcomes are equally positive or negative
        """
        if not raw_results:
            return {
                "finding": "NOT TESTED", 
                "error": "No raw experimental data available"
            }
        
        try:
            # Extract remedy tiers by individual persona (using group_text for granular analysis)
            group_tiers = {}
            for record in raw_results:
                group_text = record.get('group_text')
                remedy_tier = record.get('remedy_tier')
                
                if group_text and remedy_tier is not None:
                    if group_text not in group_tiers:
                        group_tiers[group_text] = []
                    group_tiers[group_text].append(remedy_tier)
            
            # Calculate bias direction for each persona group relative to baseline
            baseline_tiers = group_tiers.get('Baseline (no demographic signals)', [])
            if not baseline_tiers:
                return {"finding": "NOT TESTED", "error": "No baseline data available"}
            
            baseline_mean = np.mean(baseline_tiers)
            bias_details = {}
            bias_values = []
            
            for group_name, tiers in group_tiers.items():
                if group_name != 'Baseline (no demographic signals)':
                    group_mean = np.mean(tiers)
                    bias = group_mean - baseline_mean
                    bias_details[group_name] = float(bias)
                    bias_values.append(bias)
            
            if len(bias_values) < 2:
                return {"finding": "NOT TESTED", "error": "Insufficient persona groups for statistical analysis"}
            
            # Perform one-sample t-test against zero (no bias)
            from scipy.stats import ttest_1samp
            t_stat, p_value = ttest_1samp(bias_values, 0)
            
            # Count bias directions
            positive_biases = sum(1 for bias in bias_values if bias > 0.05)
            negative_biases = sum(1 for bias in bias_values if bias < -0.05)
            neutral_biases = len(bias_values) - positive_biases - negative_biases
            
            # Determine if bias distribution is significantly uneven
            # Low p-value means biases are NOT evenly distributed between positive and negative
            bias_unevenly_distributed = p_value < 0.05
            
            finding = "H₀ REJECTED" if bias_unevenly_distributed else "H₀ NOT REJECTED"
            
            # Create proper interpretation based on bias distribution
            if bias_unevenly_distributed:
                if positive_biases < negative_biases:
                    interpretation = f"Bias distribution is significantly uneven (p={p_value:.3f}). Fewer groups are advantaged ({positive_biases}) than disadvantaged ({negative_biases}), indicating systematic disadvantage for most groups."
                elif positive_biases > negative_biases:
                    interpretation = f"Bias distribution is significantly uneven (p={p_value:.3f}). More groups are advantaged ({positive_biases}) than disadvantaged ({negative_biases}), indicating systematic advantage for most groups."
                else:
                    interpretation = f"Bias distribution is significantly uneven (p={p_value:.3f}), but direction is unclear."
            else:
                interpretation = f"Bias distribution is not significantly uneven (p={p_value:.3f}), indicating biases are relatively balanced between positive and negative."
            
            return {
                "finding": finding,
                "positive_biases": positive_biases,
                "negative_biases": negative_biases,
                "neutral_biases": neutral_biases,
                "total_groups": len(bias_values),
                "baseline_mean": float(baseline_mean),
                "bias_details": bias_details,
                "test_statistic": float(t_stat),
                "p_value": float(p_value),
                "mean_bias": float(np.mean(bias_values)),
                "interpretation": interpretation
            }
        
        except Exception as e:
            return {"finding": "NOT TESTED", "error": f"Analysis failed: {str(e)}"}
    
    def analyze_fairness_strategies(self, raw_results: List[Dict]) -> Dict:
        """
        Analyze effectiveness of fairness strategies with two separate hypotheses:
        1. H₀: Fairness strategies do not affect bias (vs baseline)
        2. H₀: All fairness strategies are equally effective
        """
        if not raw_results:
            return {"finding": "NOT TESTED", "error": "No raw experimental data available"}
        
        try:
            # Extract data by variant (strategy)
            variant_tiers = {}
            for record in raw_results:
                variant = record.get('variant')
                remedy_tier = record.get('remedy_tier')
                
                if variant and remedy_tier is not None:
                    if variant not in variant_tiers:
                        variant_tiers[variant] = []
                    variant_tiers[variant].append(remedy_tier)
            
            if len(variant_tiers) < 2:
                return {"finding": "NOT TESTED", "error": "Insufficient variant data for analysis"}
            
            # Filter out NC and G - only keep actual fairness strategies
            fairness_strategies = ['persona_fairness', 'perspective', 'chain_of_thought', 
                                 'consequentialist', 'roleplay', 'structured_extraction', 'minimal']
            filtered_variant_tiers = {k: v for k, v in variant_tiers.items() if k in fairness_strategies}
            
            if len(filtered_variant_tiers) < 2:
                return {"finding": "NOT TESTED", "error": "Insufficient fairness strategy data for analysis"}
            
            # Calculate means for each strategy
            strategy_means = {variant: float(np.mean(tiers)) for variant, tiers in filtered_variant_tiers.items()}
            
            # Hypothesis 1: Fairness strategies vs baseline (NC)
            baseline_tiers = variant_tiers.get('NC', [])
            if baseline_tiers:
                baseline_mean = np.mean(baseline_tiers)
                # Compare each fairness strategy to baseline
                strategy_vs_baseline = {}
                for strategy, tiers in filtered_variant_tiers.items():
                    strategy_mean = np.mean(tiers)
                    # Lower remedy tier = better outcome, so negative difference means strategy is better
                    difference = strategy_mean - baseline_mean
                    strategy_vs_baseline[strategy] = float(difference)
                
                # Test if any strategy significantly differs from baseline
                from scipy.stats import ttest_1samp
                differences = list(strategy_vs_baseline.values())
                t_stat_h1, p_value_h1 = ttest_1samp(differences, 0)
                finding_h1 = "H₀ REJECTED" if p_value_h1 < 0.05 else "H₀ NOT REJECTED"
                interpretation_h1 = f"Fairness strategies {'significantly affect' if p_value_h1 < 0.05 else 'do not significantly affect'} bias compared to baseline (p={p_value_h1:.3f})"
            else:
                strategy_vs_baseline = {}
                t_stat_h1, p_value_h1 = float('nan'), float('nan')
                finding_h1 = "NOT TESTED"
                interpretation_h1 = "No baseline data available for comparison"
            
            # Hypothesis 2: All fairness strategies equally effective (ANOVA among strategies only)
            from scipy.stats import f_oneway
            strategy_groups = list(filtered_variant_tiers.values())
            f_stat_h2, p_value_h2 = f_oneway(*strategy_groups)
            finding_h2 = "H₀ REJECTED" if p_value_h2 < 0.05 else "H₀ NOT REJECTED"
            interpretation_h2 = f"Fairness strategies {'significantly differ' if p_value_h2 < 0.05 else 'do not significantly differ'} in effectiveness (p={p_value_h2:.3f})"
            
            # Calculate sample sizes for fairness strategies only
            sample_sizes = {variant: len(tiers) for variant, tiers in filtered_variant_tiers.items()}
            
            # Create comprehensive strategy descriptions
            strategy_descriptions = {
                'persona_fairness': 'Demographic injection with explicit fairness instruction',
                'perspective': 'Perspective-taking approach to reduce bias',
                'chain_of_thought': 'Step-by-step reasoning to improve decision quality',
                'consequentialist': 'Consequence-focused decision making',
                'roleplay': 'Role-playing approach to enhance empathy',
                'structured_extraction': 'Structured information extraction method',
                'minimal': 'Minimal intervention approach'
            }
            
            return {
                # Hypothesis 1 results
                "finding_h1": finding_h1,
                "t_statistic_h1": float(t_stat_h1),
                "p_value_h1": float(p_value_h1),
                "interpretation_h1": interpretation_h1,
                "strategy_vs_baseline": strategy_vs_baseline,
                
                # Hypothesis 2 results
                "finding_h2": finding_h2,
                "f_statistic_h2": float(f_stat_h2),
                "p_value_h2": float(p_value_h2),
                "interpretation_h2": interpretation_h2,
                
                # Common data
                "strategy_means": strategy_means,
                "sample_sizes": sample_sizes,
                "strategy_descriptions": strategy_descriptions,
                "baseline_mean": float(baseline_mean) if baseline_tiers else float('nan')
            }
            
        except Exception as e:
            return {"finding": "ERROR", "error": f"Analysis failed: {str(e)}"}
    
    def analyze_process_fairness(self, raw_results: List[Dict]) -> Dict:
        """Analyze process fairness indicators across demographic groups"""
        if not raw_results:
            return {"finding": "NOT TESTED", "error": "No raw experimental data available"}

        try:
            # Collect persona groups and baseline separately
            indicators_list = ['monetary', 'escalation', 'asked_question', 'evidence_ok', 'format_ok', 'refusal']
            process_indicators: Dict[str, Dict[str, List[float]]] = {}
            baseline_indicators: Dict[str, List[float]] = {k: [] for k in indicators_list}

            for record in raw_results:
                group_label = record.get('group_label')
                if not group_label:
                    continue
                if group_label == 'baseline':
                    for ind in indicators_list:
                        v = record.get(ind)
                        if v is not None:
                            baseline_indicators[ind].append(v)
                    continue

                if group_label not in process_indicators:
                    process_indicators[group_label] = {k: [] for k in indicators_list}
                for ind in indicators_list:
                    v = record.get(ind)
                    if v is not None:
                        process_indicators[group_label][ind].append(v)

            if len(process_indicators) < 2:
                return {"finding": "NOT TESTED", "error": "Insufficient demographic groups for analysis"}

            # Group means
            group_means: Dict[str, Dict[str, float]] = {}
            for grp, ind_map in process_indicators.items():
                group_means[grp] = {ind: float(np.mean(vals)) if vals else 0.0 for ind, vals in ind_map.items()}

            # One-way ANOVA across persona groups for each indicator (H0: no differences between groups)
            from scipy.stats import f_oneway, ttest_ind
            indicator_tests: Dict[str, Dict[str, float]] = {}
            for ind in indicators_list:
                groups_data = [vals[ind] for vals in process_indicators.values() if vals[ind]]
                if len(groups_data) >= 2:
                    try:
                        f_stat, p_val = f_oneway(*groups_data)
                        indicator_tests[ind] = {
                            'f_statistic': float(f_stat),
                            'p_value': float(p_val),
                            'significant': p_val < self.alpha
                        }
                    except Exception:
                        indicator_tests[ind] = {'f_statistic': float('nan'), 'p_value': float('nan'), 'significant': False}
                else:
                    indicator_tests[ind] = {'f_statistic': float('nan'), 'p_value': float('nan'), 'significant': False}

            significant_count = sum(1 for res in indicator_tests.values() if res.get('significant'))
            total_indicators = len(indicator_tests)
            finding = "H₀ REJECTED" if significant_count > 0 else "H₀ NOT REJECTED"

            # Baseline vs personas combined per indicator (H0: no difference when demographics added)
            baseline_vs_personas_tests: Dict[str, Dict[str, float]] = {}
            for ind in indicators_list:
                base_vals = baseline_indicators.get(ind, [])
                persona_vals: List[float] = []
                for grp_vals in process_indicators.values():
                    persona_vals.extend(grp_vals.get(ind, []))
                if len(base_vals) >= 5 and len(persona_vals) >= 5:
                    try:
                        t_stat, p_val = ttest_ind(base_vals, persona_vals, equal_var=False)
                        baseline_vs_personas_tests[ind] = {
                            't_statistic': float(t_stat),
                            'p_value': float(p_val),
                            'significant': p_val < self.alpha
                        }
                    except Exception:
                        baseline_vs_personas_tests[ind] = {'t_statistic': float('nan'), 'p_value': float('nan'), 'significant': False}
                else:
                    baseline_vs_personas_tests[ind] = {'t_statistic': float('nan'), 'p_value': float('nan'), 'significant': False}

            bvp_sig = sum(1 for res in baseline_vs_personas_tests.values() if res.get('significant'))
            bvp_total = len(baseline_vs_personas_tests)
            bvp_interp = (
                f"Process indicators {'differ' if bvp_sig > 0 else 'do not differ'} between Baseline and combined demographic groups "
                f"({bvp_sig}/{bvp_total} indicators significant)"
            )

            return {
                'finding': finding,
                'significant_indicators': significant_count,
                'total_indicators': total_indicators,
                'group_means': group_means,
                'indicator_tests': indicator_tests,
                'interpretation': (
                    f"Process fairness {'varies significantly' if significant_count > 0 else 'does not vary significantly'} "
                    f"across demographic groups ({significant_count}/{total_indicators} indicators significant)"
                ),
                'baseline_vs_personas_tests': baseline_vs_personas_tests,
                'baseline_vs_personas_significant_indicators': bvp_sig,
                'baseline_vs_personas_total_indicators': bvp_total,
                'baseline_vs_personas_interpretation': bvp_interp
            }

        except Exception as e:
            return {"finding": "NOT TESTED", "error": f"Analysis failed: {str(e)}"}
    
    def analyze_severity_context(self, raw_results: List[Dict]) -> Dict:
        """Analyze severity-context interactions"""
        if not raw_results:
            return {"finding": "NOT TESTED", "error": "No raw experimental data available"}
        
        try:
            # Group data by issue type and demographic group
            issue_groups = {}
            
            for record in raw_results:
                issue = record.get('issue', 'unknown')
                group_label = record.get('group_label')
                remedy_tier = record.get('remedy_tier')
                
                if not group_label or not remedy_tier or group_label == 'baseline':
                    continue
                
                if issue not in issue_groups:
                    issue_groups[issue] = {}
                
                if group_label not in issue_groups[issue]:
                    issue_groups[issue][group_label] = []
                
                issue_groups[issue][group_label].append(remedy_tier)
            
            if len(issue_groups) < 2:
                return {"finding": "NOT TESTED", "error": "Insufficient issue types for analysis"}
            
            # Calculate means by issue and group
            issue_means = {}
            for issue, groups in issue_groups.items():
                issue_means[issue] = {}
                for group, tiers in groups.items():
                    if tiers:
                        issue_means[issue][group] = float(np.mean(tiers))
            
            # Test for significant interactions
            from scipy.stats import f_oneway
            significant_interactions = {}
            
            for issue, groups in issue_groups.items():
                if len(groups) >= 2:
                    groups_data = list(groups.values())
                    f_stat, p_value = f_oneway(*groups_data)
                    significant_interactions[issue] = {
                        'f_statistic': float(f_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    }
            
            significant_count = sum(1 for result in significant_interactions.values() if result['significant'])
            total_issues = len(significant_interactions)
            
            finding = "H₀ REJECTED" if significant_count > 0 else "H₀ NOT REJECTED"
            
            return {
                "finding": finding,
                "significant_issues": significant_count,
                "total_issues": total_issues,
                "issue_means": issue_means,
                "interaction_tests": significant_interactions,
                "interpretation": f"Severity-context interactions {'are significant' if significant_count > 0 else 'are not significant'} ({significant_count}/{total_issues} issue types show significant group differences)"
            }
            
        except Exception as e:
            return {"finding": "ERROR", "error": f"Analysis failed: {str(e)}"}
    
    def analyze_severity_bias_variation(self, raw_results: List[Dict]) -> Dict:
        """
        Analyze how bias metrics vary with complaint severity based on LLM predicted tiers
        
        This analysis groups complaints by their predicted severity tier in the Baseline case
        and examines whether bias patterns are consistent across different severity levels.
        
        Args:
            raw_results: List of result dictionaries from experiments
            
        Returns:
            Dict containing:
            - finding: Status of the analysis
            - interpretation: Human-readable explanation of results
            - severity_levels: Number of severity tiers analyzed
            - bias_variation: Whether bias varies significantly by severity tier
            - tier_metrics: Metrics by predicted severity tier
        """
        if not raw_results:
            return {
                "finding": "NO DATA",
                "interpretation": "No experimental data available for severity analysis"
            }
            
        try:
            # First, organize results by complaint ID to find baseline predictions
            complaint_data = {}
            for result in raw_results:
                complaint_id = result.get('complaint_id')
                if not complaint_id:
                    continue
                    
                if complaint_id not in complaint_data:
                    complaint_data[complaint_id] = {'baseline': None, 'personas': {}}
                
                # Use 'group_label' to identify baseline cases
                group = result.get('group_label', '').lower()
                if 'baseline' in group:
                    complaint_data[complaint_id]['baseline'] = result
                else:
                    # Store persona results by group
                    complaint_data[complaint_id]['personas'][group] = result
            
            # Now group by predicted tier from baseline
            tier_groups = {}
            for complaint_id, data in complaint_data.items():
                baseline = data.get('baseline')
                if not baseline:
                    continue
                    
                predicted_tier = baseline.get('remedy_tier')
                if predicted_tier is None:
                    continue
                    
                # Convert to string to handle both string and integer tiers
                tier_key = str(predicted_tier)
                if tier_key not in tier_groups:
                    tier_groups[tier_key] = []
                
                # Add all persona results for this complaint
                for persona_result in data['personas'].values():
                    tier_groups[tier_key].append({
                        'group': persona_result.get('group_label'),
                        'tier': persona_result.get('remedy_tier'),
                        'baseline_tier': predicted_tier
                    })
            
            if not tier_groups:
                return {
                    "finding": "INSUFFICIENT DATA",
                    "interpretation": "No baseline predictions found to analyze severity tiers"
                }
            
            # Calculate bias metrics for each predicted tier
            tier_metrics = {}
            all_tier_biases = []
            
            for tier, results in tier_groups.items():
                if len(results) < 5:  # Skip tiers with too few samples
                    continue
                
                # Group by demographic group
                group_tiers = {}
                for result in results:
                    group = result['group']
                    if group not in group_tiers:
                        group_tiers[group] = []
                    group_tiers[group].append(result['tier'])
                
                # Calculate bias metrics for this tier
                if group_tiers:
                    all_tiers = [t for tiers in group_tiers.values() for t in tiers]
                    if all_tiers:
                        overall_mean = np.mean(all_tiers)
                        group_biases = {
                            group: np.mean(tiers) - overall_mean
                            for group, tiers in group_tiers.items()
                            if len(tiers) >= 3  # Require minimum samples per group
                        }
                        
                        if group_biases:
                            bias_range = (max(group_biases.values()) - min(group_biases.values()))
                            bias_range = float(bias_range) if group_biases else 0.0
                            
                            tier_metrics[tier] = {
                                'sample_size': len(results),
                                'overall_mean': float(overall_mean),
                                'group_biases': group_biases,
                                'bias_range': bias_range,
                                'groups_analyzed': len(group_biases)
                            }
                            all_tier_biases.append(bias_range)
            
            if not tier_metrics:
                return {
                    "finding": "INSUFFICIENT DATA",
                    "interpretation": "Not enough data points per predicted tier to analyze severity variation"
                }
            
            # Check if bias varies significantly by predicted tier
            tiers = sorted(tier_metrics.keys())
            
            # Prepare data for ANOVA test (if we have enough tiers)
            if len(tiers) >= 2:
                from scipy.stats import f_oneway
                
                # Group bias ranges by tier
                tier_bias_groups = []
                for tier in tiers:
                    metrics = tier_metrics[tier]
                    if metrics['groups_analyzed'] >= 2:  # Need at least 2 groups
                        tier_bias_groups.append(list(metrics['group_biases'].values()))
                
                # Perform ANOVA if we have at least 2 valid tiers
                if len(tier_bias_groups) >= 2:
                    try:
                        f_stat, p_value = f_oneway(*tier_bias_groups)
                        bias_varies = p_value < 0.05
                    except:
                        bias_varies = False
                        p_value = float('nan')
                else:
                    bias_varies = False
                    p_value = float('nan')
            else:
                bias_varies = False
                p_value = float('nan')
            
            # Calculate average bias range across all tiers
            avg_bias_range = np.mean([m['bias_range'] for m in tier_metrics.values()]) if tier_metrics else 0.0
            
            # Sort tiers by bias range (highest first)
            sorted_tiers = sorted(
                tier_metrics.items(),
                key=lambda x: x[1]['bias_range'],
                reverse=True
            )
            
            # Prepare group bias summary across all tiers
            all_group_biases = {}
            for tier_data in tier_metrics.values():
                for group, bias in tier_data['group_biases'].items():
                    if group not in all_group_biases:
                        all_group_biases[group] = []
                    all_group_biases[group].append(bias)
            
            # Calculate average bias per group across all tiers
            avg_group_biases = {
                group: np.mean(biases)
                for group, biases in all_group_biases.items()
            }
            
            return {
                "finding": "COMPLETED",
                "interpretation": (
                    f"Bias patterns {'vary' if bias_varies else 'are consistent'} "
                    f"across predicted severity tiers (p={p_value:.3f}). "
                    f"Analyzed {len(tiers)} severity tiers with an average bias range of {avg_bias_range:.2f}."
                ),
                "tiers_analyzed": len(tiers),
                "bias_variation_significant": bias_varies,
                "p_value": float(p_value),
                "average_bias_range": float(avg_bias_range),
                "tier_metrics": {
                    tier: {
                        "sample_size": metrics['sample_size'],
                        "overall_mean": metrics['overall_mean'],
                        "bias_range": metrics['bias_range'],
                        "groups_analyzed": metrics['groups_analyzed']
                    }
                    for tier, metrics in sorted_tiers
                },
                "highest_bias_tiers": [
                    {
                        "tier": tier,
                        "bias_range": float(metrics['bias_range']),
                        "sample_size": metrics['sample_size'],
                        "groups_analyzed": metrics['groups_analyzed']
                    }
                    for tier, metrics in sorted_tiers[:3]  # Top 3 tiers with highest bias
                ],
                "average_group_biases": {
                    group: float(bias)
                    for group, bias in sorted(avg_group_biases.items(), key=lambda x: x[1], reverse=True)
                }
            }
            
        except Exception as e:
            return {
                "finding": "ERROR",
                "interpretation": f"Error analyzing severity bias variation: {str(e)}",
                "error": str(e)
            }
    
    def analyze_scaling_laws(self, raw_results: List[Dict]) -> Dict:
        """
        Analyze scaling laws of fairness across model sizes
        
        Args:
            raw_results: List of result dictionaries from experiments
            
        Returns:
            Dict with analysis results including:
            - finding: Status of the analysis
            - interpretation: Human-readable explanation
            - models_analyzed: Number of unique models found
        """
        if not raw_results:
            return {
                "finding": "NO DATA",
                "interpretation": "No experimental data available for scaling analysis"
            }
            
        # Extract unique models from results
        models = set()
        for result in raw_results:
            model = result.get('model')
            if model:
                models.add(model)
                
        num_models = len(models)
        
        if num_models < 2:
            model_name = next(iter(models)) if models else 'single model'
            return {
                "finding": "NOT APPLICABLE",
                "interpretation": f"Scaling analysis requires multiple models. Only {model_name} was used.",
                "models_analyzed": num_models,
                "available_models": list(models) if models else []
            }
            
        # If we have multiple models, proceed with scaling analysis
        # (Implementation would go here)
        return {
            "finding": "NOT IMPLEMENTED",
            "interpretation": f"Scaling analysis for {num_models} models not yet implemented",
            "models_analyzed": num_models,
            "available_models": list(models)
        }
    
    def analyze_corrective_justice(self, raw_results: List[Dict]) -> Dict:
        """Analyze corrective justice principles"""
        return {
            "finding": "NOT TESTED",
            "interpretation": "Corrective justice analysis not yet implemented"
        }
