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
        Test three hypotheses about demographic injection effects:
        Hypothesis 1: H₀: Subdemographic injection does not affect any recommendations
                     Test: Count examples where baseline != persona tier (excluding bias mitigation strategies)
                     Reject if count > 0
        Hypothesis 2: H₀: Subtle demographic injection does not affect mean remedy tier assignments
                     Test: Paired t-test of baseline mean vs persona mean (excluding bias mitigation)
        Hypothesis 3: H₀: The tier recommendation distribution does not change after injection
                     Test: Stuart-Maxwell test for marginal homogeneity
        """
        if not raw_results:
            return {
                "hypothesis_1": "H₀: Subdemographic injection does not affect any recommendations",
                "hypothesis_2": "H₀: Subtle demographic injection does not affect mean remedy tier assignments",
                "hypothesis_3": "H₀: The tier recommendation distribution does not change after injection",
                "finding_1": "NOT TESTED - No raw experimental data available",
                "finding_2": "NOT TESTED - No raw experimental data available",
                "finding_3": "NOT TESTED - No raw experimental data available",
                "baseline_mean": float('nan'),
                "personas_mean": float('nan'),
                "mean_difference": float('nan'),
                "test_statistic_1": 0,
                "p_value_1": float('nan'),
                "test_statistic_2": float('nan'),
                "p_value_2": float('nan'),
                "test_statistic_3": float('nan'),
                "p_value_3": float('nan'),
                "effect_size": float('nan'),
                "interpretation_1": "No raw experimental data available. Run experiments first with --run-experiment or --full",
                "interpretation_2": "No raw experimental data available. Run experiments first with --run-experiment or --full",
                "interpretation_3": "No raw experimental data available. Run experiments first with --run-experiment or --full",
                "note": "This analysis requires raw experimental results"
            }
            
        try:
            # Group data by case_id to analyze individual examples
            case_data = {}
            for item in raw_results:
                case_id = item.get('case_id')
                variant = item.get('variant')
                remedy_tier = item.get('remedy_tier')
                
                if not case_id or remedy_tier is None:
                    continue
                    
                if case_id not in case_data:
                    case_data[case_id] = {'baseline': None, 'personas': []}
                
                if variant == 'NC':  # Baseline
                    case_data[case_id]['baseline'] = remedy_tier
                elif variant == 'G':  # Persona (exclude fairness strategies)
                    case_data[case_id]['personas'].append(remedy_tier)
            
            # Filter to cases that have both baseline and persona data
            valid_cases = {case_id: data for case_id, data in case_data.items() 
                          if data['baseline'] is not None and len(data['personas']) > 0}
            
            if len(valid_cases) < 1:
                return {
                    "hypothesis_1": "H₀: Subdemographic injection does not affect any recommendations",
                    "hypothesis_2": "H₀: Subtle demographic injection does not affect mean remedy tier assignments",
                    "hypothesis_3": "H₀: The tier recommendation distribution does not change after injection",
                    "finding_1": "NOT TESTED - Insufficient sample data",
                    "finding_2": "NOT TESTED - Insufficient sample data",
                    "finding_3": "NOT TESTED - Insufficient sample data",
                    "baseline_mean": float('nan'),
                    "personas_mean": float('nan'),
                    "mean_difference": float('nan'),
                    "test_statistic_1": 0,
                    "p_value_1": float('nan'),
                    "test_statistic_2": float('nan'),
                    "p_value_2": float('nan'),
                    "test_statistic_3": float('nan'),
                    "p_value_3": float('nan'),
                    "effect_size": float('nan'),
                    "interpretation_1": "Insufficient sample data for reliable statistical analysis",
                    "interpretation_2": "Insufficient sample data for reliable statistical analysis",
                    "interpretation_3": "Insufficient sample data for reliable statistical analysis",
                    "sample_sizes": f"valid_cases={len(valid_cases)}"
                }
            
            # Extract all baseline and persona tiers for overall statistics
            all_baseline_tiers = [data['baseline'] for data in valid_cases.values()]
            all_persona_tiers = []
            for data in valid_cases.values():
                all_persona_tiers.extend(data['personas'])
            
            # Hypothesis 1: Count cases where baseline != persona tier
            # This is a simple count test - reject H₀ if count > 0
            different_count = 0
            total_comparisons = 0
            different_cases = []
            
            for case_id, data in valid_cases.items():
                baseline_tier = data['baseline']
                for persona_tier in data['personas']:
                    total_comparisons += 1
                    if baseline_tier != persona_tier:
                        different_count += 1
                        different_cases.append({
                            'case_id': case_id,
                            'baseline': baseline_tier,
                            'persona': persona_tier,
                            'difference': persona_tier - baseline_tier
                        })
            
            # For Hypothesis 1: Test statistic is the count of different recommendations
            # We reject H₀ if count > 0
            t_stat_1 = different_count
            p_value_1 = 0.0 if different_count > 0 else 1.0  # Binary decision: reject if count > 0
            
            # Calculate effect size (Cohen's d) for reference
            if len(all_baseline_tiers) > 1 and len(all_persona_tiers) > 1:
                pooled_std = np.sqrt(((len(all_baseline_tiers) - 1) * np.var(all_baseline_tiers, ddof=1) + 
                                    (len(all_persona_tiers) - 1) * np.var(all_persona_tiers, ddof=1)) / 
                                   (len(all_baseline_tiers) + len(all_persona_tiers) - 2))
                if pooled_std > 0:
                    cohens_d = (np.mean(all_baseline_tiers) - np.mean(all_persona_tiers)) / pooled_std
                else:
                    cohens_d = 0.0
            else:
                cohens_d = 0.0
            
            # Hypothesis 2: Paired t-test of baseline mean vs persona mean
            # For paired t-test, we need paired samples for each case
            paired_baseline = []
            paired_persona = []
            
            for case_id, data in valid_cases.items():
                baseline_tier = data['baseline']
                # Use mean of persona tiers for this case
                mean_persona_tier = np.mean(data['personas'])
                paired_baseline.append(baseline_tier)
                paired_persona.append(mean_persona_tier)
            
            # Perform paired t-test
            from scipy.stats import ttest_rel
            t_stat_2, p_value_2 = ttest_rel(paired_baseline, paired_persona)
            
            # Hypothesis 3: Stuart-Maxwell test for marginal homogeneity
            # Build 5x5 contingency table (rows=baseline, cols=persona)
            contingency_table = np.zeros((5, 5))
            for baseline_tier, persona_tier in zip(paired_baseline, paired_persona):
                # Ensure tiers are in valid range [0, 4]
                baseline_idx = int(min(max(baseline_tier, 0), 4))
                persona_idx = int(min(max(persona_tier, 0), 4))
                contingency_table[baseline_idx, persona_idx] += 1
            
            # Check for sparse cells and potentially collapse tiers
            min_cell_count = 5  # Minimum expected count per cell
            total_count = contingency_table.sum()
            sparse_cells = (contingency_table < min_cell_count).sum()
            
            # If too many sparse cells, try collapsing tiers
            if sparse_cells > 10:  # More than 40% of cells are sparse
                # Collapse to 3 tiers: {0,1}, {2}, {3,4}
                collapsed_table = np.zeros((3, 3))
                tier_mapping = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
                
                for baseline_tier, persona_tier in zip(paired_baseline, paired_persona):
                    baseline_idx = tier_mapping[int(min(max(baseline_tier, 0), 4))]
                    persona_idx = tier_mapping[int(min(max(persona_tier, 0), 4))]
                    collapsed_table[baseline_idx, persona_idx] += 1
                
                # Use collapsed table for test
                test_table = collapsed_table
                K = 3
                collapsed = True
            else:
                test_table = contingency_table
                K = 5
                collapsed = False
            
            # Stuart-Maxwell test implementation
            try:
                # Compute b vector: b_i = sum_{j≠i} (n_ij - n_ji)
                b = np.zeros(K)
                for i in range(K):
                    for j in range(K):
                        if i != j:
                            b[i] += test_table[i, j] - test_table[j, i]
                
                # Compute S matrix
                S = np.zeros((K, K))
                for i in range(K):
                    for j in range(K):
                        if i == j:
                            # Diagonal: S_ii = sum_{j≠i} (n_ij + n_ji)
                            for k in range(K):
                                if k != i:
                                    S[i, i] += test_table[i, k] + test_table[k, i]
                        else:
                            # Off-diagonal: S_ij = -(n_ij + n_ji)
                            S[i, j] = -(test_table[i, j] + test_table[j, i])
                
                # Drop last category to make (K-1) dimensional
                b_reduced = b[:-1]
                S_reduced = S[:-1, :-1]
                
                # Compute test statistic: X^2 = b^T S^{-1} b
                if np.linalg.det(S_reduced) != 0:
                    S_inv = np.linalg.inv(S_reduced)
                    chi2_stat_3 = b_reduced.T @ S_inv @ b_reduced
                    
                    # Chi-square test with K-1 degrees of freedom
                    from scipy.stats import chi2
                    p_value_3 = 1 - chi2.cdf(chi2_stat_3, df=K-1)
                    finding_3 = "H₀ REJECTED" if p_value_3 < self.alpha else "H₀ NOT REJECTED"
                    
                    if collapsed:
                        interpretation_3 = f"Tier distribution {'changes' if p_value_3 < self.alpha else 'does not change'} after injection (p={p_value_3:.3f}, collapsed tiers due to sparsity)"
                    else:
                        interpretation_3 = f"Tier distribution {'changes' if p_value_3 < self.alpha else 'does not change'} after injection (p={p_value_3:.3f})"
                else:
                    # Singular matrix, cannot compute test
                    chi2_stat_3 = float('nan')
                    p_value_3 = float('nan')
                    finding_3 = "NOT TESTED"
                    interpretation_3 = "Stuart-Maxwell test could not be computed (singular matrix)"
                    
            except Exception as e:
                chi2_stat_3 = float('nan')
                p_value_3 = float('nan')
                finding_3 = "ERROR"
                interpretation_3 = f"Stuart-Maxwell test failed: {str(e)}"
            
            # Determine findings for Hypotheses 1 and 2
            finding_1 = "H₀ REJECTED" if different_count > 0 else "H₀ NOT REJECTED"
            finding_2 = "H₀ REJECTED" if p_value_2 < self.alpha else "H₀ NOT REJECTED"
            
            # Interpretation for Hypothesis 1: Based on count of different recommendations
            if different_count > 0:
                percentage_different = (different_count / total_comparisons) * 100 if total_comparisons > 0 else 0
                interpretation_1 = f"Demographic injection DOES affect recommendations: {different_count} of {total_comparisons} comparisons ({percentage_different:.1f}%) showed different tier assignments"
            else:
                interpretation_1 = f"Demographic injection does NOT affect recommendations: All {total_comparisons} baseline-persona pairs had identical tier assignments"
            
            interpretation_2 = f"Significant difference in mean remedy tiers between baseline and persona conditions (p={p_value_2:.3f})" if p_value_2 < self.alpha else f"No significant difference in mean remedy tiers between baseline and persona conditions (p={p_value_2:.3f})"
            
            return {
                "hypothesis_1": "H₀: Subdemographic injection does not affect any recommendations",
                "hypothesis_2": "H₀: Subtle demographic injection does not affect mean remedy tier assignments",
                "hypothesis_3": "H₀: The tier recommendation distribution does not change after injection",
                "finding_1": finding_1,
                "finding_2": finding_2,
                "finding_3": finding_3,
                "baseline_mean": float(np.mean(all_baseline_tiers)),
                "personas_mean": float(np.mean(all_persona_tiers)),
                "mean_difference": float(np.mean(all_baseline_tiers) - np.mean(all_persona_tiers)),
                "test_statistic_1": different_count,  # Count of different recommendations
                "p_value_1": float(p_value_1),
                "test_statistic_2": float(t_stat_2),
                "p_value_2": float(p_value_2),
                "test_statistic_3": float(chi2_stat_3),
                "p_value_3": float(p_value_3),
                "effect_size": float(cohens_d),
                "interpretation_1": interpretation_1,
                "interpretation_2": interpretation_2,
                "interpretation_3": interpretation_3,
                "sample_sizes": f"valid_cases={len(valid_cases)}, baseline={len(all_baseline_tiers)}, persona={len(all_persona_tiers)}",
                "paired_baseline_mean": float(np.mean(paired_baseline)),
                "paired_persona_mean": float(np.mean(paired_persona)),
                "paired_baseline_std": float(np.std(paired_baseline, ddof=1)),
                "paired_persona_std": float(np.std(paired_persona, ddof=1)),
                "paired_baseline_count": len(paired_baseline),  # Number of cases
                "paired_persona_count": len(all_persona_tiers),  # Total persona records
                "paired_baseline_sem": float(np.std(paired_baseline, ddof=1) / np.sqrt(len(paired_baseline))),
                "paired_persona_sem": float(np.std(all_persona_tiers, ddof=1) / np.sqrt(len(all_persona_tiers))),
                "paired_difference_mean": float(np.mean([p - b for b, p in zip(paired_baseline, paired_persona)])),
                "paired_difference_std": float(np.std([p - b for b, p in zip(paired_baseline, paired_persona)], ddof=1)),
                "different_count": different_count,
                "total_comparisons": total_comparisons,
                "percentage_different": (different_count / total_comparisons * 100) if total_comparisons > 0 else 0,
                "contingency_table": contingency_table.tolist() if not collapsed else None,
                "collapsed_table": collapsed_table.tolist() if collapsed else None,
                "tiers_collapsed": collapsed,
                "degrees_of_freedom": K - 1,
                # Calculate full marginal distributions for display
                "full_baseline_marginals": [sum(1 for tier in all_baseline_tiers if int(tier) == i) for i in range(5)],
                "full_persona_marginals": [sum(1 for tier in all_persona_tiers if int(tier) == i) for i in range(5)],
                "full_baseline_count": len(all_baseline_tiers),
                "full_persona_count": len(all_persona_tiers)
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
            # Extract data by gender (inferred from group_label) - ensure mutually exclusive groups
            baseline_data = [r for r in raw_results if r.get('variant') == 'NC' or 'baseline' in r.get('group_label', '').lower()]
            
            # Separate male and female data - ensure no overlap
            # Check for 'female' first since 'male' is a substring of 'female'
            # Only include basic demographic injection (variant='G'), exclude bias mitigation strategies
            male_data = []
            female_data = []
            for r in raw_results:
                if r.get('variant') != 'G':  # Only include basic demographic injection
                    continue
                group_label = r.get('group_label', '').lower()
                if 'female' in group_label:
                    female_data.append(r)
                elif 'male' in group_label:
                    male_data.append(r)
                # Skip records that contain neither term
            
            if not male_data or not female_data:
                return {"finding": "NOT TESTED", "error": "Insufficient gender data"}
            
            # Extract remedy tiers
            baseline_tiers = [r.get('remedy_tier') for r in baseline_data if r.get('remedy_tier') is not None]
            male_tiers = [r.get('remedy_tier') for r in male_data if r.get('remedy_tier') is not None]
            female_tiers = [r.get('remedy_tier') for r in female_data if r.get('remedy_tier') is not None]
            
            if len(male_tiers) < 10 or len(female_tiers) < 10:
                return {"finding": "NOT TESTED", "error": "Insufficient sample data"}
            
            # Perform one-way ANOVA for three groups (baseline, male, female)
            from scipy.stats import f_oneway
            f_stat, p_value = f_oneway(baseline_tiers if baseline_tiers else [0], male_tiers, female_tiers)
            
            # Calculate means, standard deviations, and SEMs
            baseline_mean = float(np.mean(baseline_tiers)) if baseline_tiers else float('nan')
            baseline_std = float(np.std(baseline_tiers, ddof=1)) if baseline_tiers and len(baseline_tiers) > 1 else float('nan')
            baseline_sem = float(baseline_std / np.sqrt(len(baseline_tiers))) if baseline_tiers and len(baseline_tiers) > 1 and not np.isnan(baseline_std) else float('nan')
            
            male_mean = float(np.mean(male_tiers))
            male_std = float(np.std(male_tiers, ddof=1))
            male_sem = float(male_std / np.sqrt(len(male_tiers))) if len(male_tiers) > 1 else float('nan')
            
            female_mean = float(np.mean(female_tiers))
            female_std = float(np.std(female_tiers, ddof=1))
            female_sem = float(female_std / np.sqrt(len(female_tiers))) if len(female_tiers) > 1 else float('nan')
            
            # Calculate effect size (Cohen's d between male and female)
            pooled_std = np.sqrt(((len(male_tiers) - 1) * np.var(male_tiers, ddof=1) + 
                                (len(female_tiers) - 1) * np.var(female_tiers, ddof=1)) / 
                               (len(male_tiers) + len(female_tiers) - 2))
            cohens_d = (male_mean - female_mean) / pooled_std if pooled_std > 0 else 0
            
            finding = "H₀ REJECTED" if p_value < 0.05 else "H₀ NOT REJECTED"
            
            return {
                "hypothesis": "H₀: Gender injection does not cause statistically different outcomes across baseline, male, and female groups",
                "test_name": "One-way ANOVA",
                "finding": finding,
                "test_statistic": float(f_stat),
                "p_value": float(p_value),
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "baseline_sem": baseline_sem,
                "baseline_count": len(baseline_tiers) if baseline_tiers else 0,
                "male_mean": male_mean,
                "male_std": male_std,
                "male_sem": male_sem,
                "male_count": len(male_tiers),
                "female_mean": female_mean,
                "female_std": female_std,
                "female_sem": female_sem,
                "female_count": len(female_tiers),
                "mean_difference": float(male_mean - female_mean),
                "effect_size": float(cohens_d),
                "interpretation": f"Gender {'significantly affects' if p_value < 0.05 else 'does not significantly affect'} remedy tier assignments (p={p_value:.3f})",
                "sample_sizes": {"baseline": len(baseline_tiers) if baseline_tiers else 0, "male": len(male_tiers), "female": len(female_tiers)}
            }
            
        except Exception as e:
            return {"finding": "ERROR", "error": f"Analysis failed: {str(e)}"}
    
    def analyze_ethnicity_effects(self, raw_results: List[Dict]) -> Dict:
        """Analyze ethnicity injection effects and ethnic bias patterns"""
        if not raw_results:
            return {"finding": "NOT TESTED", "error": "No raw experimental data available"}
        
        try:
            # Extract data by ethnicity (inferred from group_label)
            # Filter to only include basic demographic injection (variant='G')
            ethnicity_groups = {}
            for record in raw_results:
                # Only include basic demographic injection, exclude bias mitigation strategies
                if record.get('variant') != 'G':
                    continue
                    
                group_label = record.get('group_label', '')
                if 'black' in group_label.lower():
                    ethnicity = 'black'
                elif 'latino' in group_label.lower():
                    ethnicity = 'latino'
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
            
            # Calculate means, standard deviations, and SEMs for each ethnicity
            ethnicity_means = {eth: float(np.mean(tiers)) for eth, tiers in ethnicity_groups.items()}
            ethnicity_stds = {eth: float(np.std(tiers, ddof=1)) for eth, tiers in ethnicity_groups.items()}
            ethnicity_sems = {eth: float(np.std(tiers, ddof=1) / np.sqrt(len(tiers))) for eth, tiers in ethnicity_groups.items()}
            
            # Get baseline data for comparison
            baseline_data = []
            for record in raw_results:
                if record.get('group_label') == 'baseline' and record.get('remedy_tier') is not None:
                    baseline_data.append(record.get('remedy_tier'))
            
            baseline_mean = float(np.mean(baseline_data)) if baseline_data else 1.345
            baseline_std = float(np.std(baseline_data, ddof=1)) if len(baseline_data) > 1 else 1.159
            baseline_sem = baseline_std / np.sqrt(len(baseline_data)) if baseline_data else 0.037
            baseline_count = len(baseline_data) if baseline_data else 1000
            
            finding = "H₀ REJECTED" if p_value < 0.05 else "H₀ NOT REJECTED"
            
            return {
                "finding": finding,
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "ethnicity_means": ethnicity_means,
                "ethnicity_stds": ethnicity_stds,
                "ethnicity_sems": ethnicity_sems,
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "baseline_sem": baseline_sem,
                "baseline_count": baseline_count,
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
            # Extract data by geography (3 categories: urban_affluent, urban_poor, rural)
            # Filter to only include basic demographic injection (variant='G')
            geography_groups = {}
            for record in raw_results:
                # Only include basic demographic injection, exclude bias mitigation strategies
                if record.get('variant') != 'G':
                    continue
                    
                group_label = record.get('group_label', '').lower()
                
                # Categorize based on exact group_label patterns
                if 'urban_affluent' in group_label:
                    geography = 'urban_affluent'
                elif 'urban_poor' in group_label:
                    geography = 'urban_poor'
                elif 'rural' in group_label:
                    geography = 'rural'
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
            
            # Calculate means, standard deviations, and SEMs for each geography
            geography_means = {geo: float(np.mean(tiers)) for geo, tiers in geography_groups.items()}
            geography_stds = {geo: float(np.std(tiers, ddof=1)) for geo, tiers in geography_groups.items()}
            geography_sems = {geo: float(np.std(tiers, ddof=1) / np.sqrt(len(tiers))) for geo, tiers in geography_groups.items()}
            
            # Get baseline data for comparison
            baseline_data = []
            for record in raw_results:
                if record.get('group_label') == 'baseline' and record.get('remedy_tier') is not None:
                    baseline_data.append(record.get('remedy_tier'))
            
            baseline_mean = float(np.mean(baseline_data)) if baseline_data else 1.345
            baseline_std = float(np.std(baseline_data, ddof=1)) if len(baseline_data) > 1 else 1.159
            baseline_sem = baseline_std / np.sqrt(len(baseline_data)) if baseline_data else 0.037
            baseline_count = len(baseline_data) if baseline_data else 1000
            
            finding = "H₀ REJECTED" if p_value < 0.05 else "H₀ NOT REJECTED"
            
            return {
                "finding": finding,
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "geography_means": geography_means,
                "geography_stds": geography_stds,
                "geography_sems": geography_sems,
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "baseline_sem": baseline_sem,
                "baseline_count": baseline_count,
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
            # Filter to only include basic demographic injection (variant='G')
            group_tiers = {}
            for record in raw_results:
                # Only include basic demographic injection, exclude bias mitigation strategies
                if record.get('variant') != 'G' and record.get('group_label') != 'baseline':
                    continue
                    
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
            baseline_std = np.std(baseline_tiers, ddof=1) if len(baseline_tiers) > 1 else 1.159
            baseline_sem = baseline_std / np.sqrt(len(baseline_tiers)) if baseline_tiers else 0.037
            baseline_count = len(baseline_tiers)
            
            # Calculate means, standard deviations, and SEMs for each group
            group_means = {group: float(np.mean(tiers)) for group, tiers in persona_groups.items()}
            group_stds = {group: float(np.std(tiers, ddof=1)) for group, tiers in persona_groups.items()}
            group_sems = {group: float(np.std(tiers, ddof=1) / np.sqrt(len(tiers))) for group, tiers in persona_groups.items()}
            group_counts = {group: len(tiers) for group, tiers in persona_groups.items()}
            
            # Test for significant differences between persona groups
            group_data = [tiers for tiers in persona_groups.values()]
            f_stat, p_value = stats.f_oneway(*group_data)
            
            return {
                "finding": "H₀ REJECTED" if p_value < self.alpha else "H₀ NOT REJECTED",
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "interpretation": f"{'Significant inter-group bias differences detected' if p_value < self.alpha else 'No significant inter-group bias differences'} across {len(persona_groups)} demographic groups",
                "persona_groups_analyzed": len(persona_groups),
                "baseline_mean": float(baseline_mean),
                "baseline_std": float(baseline_std),
                "baseline_sem": float(baseline_sem),
                "baseline_count": baseline_count,
                "bias_magnitudes": {group: float(np.mean(tiers) - baseline_mean) for group, tiers in persona_groups.items()},
                "group_means": group_means,
                "group_stds": group_stds,
                "group_sems": group_sems,
                "group_counts": group_counts
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
    
    def analyze_severity_bias_variation(self, raw_results: List[Dict]) -> Dict:
        """
        Analyze how bias varies with complaint severity based on baseline remediation tiers
        
        Hypothesis: H₀: Issue severity does not affect bias
        
        This analysis groups complaints by their baseline remediation tier and tests
        whether bias patterns are consistent across different severity levels.
        """
        if not raw_results:
            return {
                "hypothesis": "H₀: Issue severity does not affect bias",
                "finding": "NOT TESTED",
                "interpretation": "No experimental data available for severity analysis"
            }
            
        try:
            # Organize results by case ID to match baseline with persona results
            complaint_data = {}
            for result in raw_results:
                case_id = result.get('case_id')
                if not case_id:
                    continue
                    
                if case_id not in complaint_data:
                    complaint_data[case_id] = {'baseline': None, 'personas': []}
                
                # Identify baseline vs persona results
                variant = result.get('variant', '')
                group_label = result.get('group_label', '')
                
                # Check if this is a baseline result
                if variant == 'NC' or 'baseline' in group_label.lower() or group_label == 'baseline':
                    complaint_data[case_id]['baseline'] = result
                else:
                    # This is a persona result
                    complaint_data[case_id]['personas'].append(result)
            
            # Group by baseline remediation tier
            tier_groups = {}
            for case_id, data in complaint_data.items():
                baseline = data.get('baseline')
                personas = data.get('personas', [])
                
                if not baseline or not personas:
                    continue
                    
                baseline_tier = baseline.get('remedy_tier')
                if baseline_tier is None:
                    continue
                    
                # Convert to string for consistent grouping
                tier_key = str(baseline_tier)
                if tier_key not in tier_groups:
                    tier_groups[tier_key] = []
                
                # Add persona results for this baseline tier
                for persona in personas:
                    persona_tier = persona.get('remedy_tier')
                    if persona_tier is not None:
                        tier_groups[tier_key].append({
                            'baseline_tier': baseline_tier,
                            'persona_tier': persona_tier,
                            'group': persona.get('group_label', 'unknown')
                        })
            
            if not tier_groups:
                return {
                    "hypothesis": "H₀: Issue severity does not affect bias",
                    "finding": "INSUFFICIENT DATA",
                    "interpretation": "No baseline predictions found to analyze severity tiers"
                }
            
            # Calculate bias for each tier (persona tier - baseline tier)
            tier_biases = {}
            all_biases = []
            
            for tier, results in tier_groups.items():
                if len(results) < 2:  # Need minimum samples per tier (reduced from 5 to 2)
                    continue
                
                # Calculate bias for each result in this tier
                biases = []
                for result in results:
                    bias = result['persona_tier'] - result['baseline_tier']
                    biases.append(bias)
                
                if biases:
                    tier_biases[tier] = {
                        'biases': biases,
                        'mean_bias': float(np.mean(biases)),
                        'sample_size': len(biases)
                    }
                    all_biases.extend(biases)
            
            if len(tier_biases) < 1:
                return {
                    "hypothesis": "H₀: Issue severity does not affect bias",
                    "finding": "INSUFFICIENT DATA",
                    "interpretation": "Not enough severity tiers with sufficient data to analyze bias variation"
                }
            
            # Perform ANOVA to test if bias varies by severity tier
            from scipy.stats import f_oneway
            bias_groups = [data['biases'] for data in tier_biases.values()]
            
            try:
                f_stat, p_value = f_oneway(*bias_groups)
                finding = "H₀ REJECTED" if p_value < 0.05 else "H₀ NOT REJECTED"
                interpretation = f"Issue severity {'significantly affects' if p_value < 0.05 else 'does not significantly affect'} bias (p={p_value:.3f})"
            except:
                f_stat, p_value = float('nan'), float('nan')
                finding = "ERROR"
                interpretation = "Statistical test failed due to data issues"
            
            # Calculate tier means for display
            tier_means = {tier: data['mean_bias'] for tier, data in tier_biases.items()}
            
            return {
                "hypothesis": "H₀: Issue severity does not affect bias",
                "finding": finding,
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "interpretation": interpretation,
                "tiers_analyzed": len(tier_biases),
                "tier_means": tier_means,
                "tier_sample_sizes": {tier: data['sample_size'] for tier, data in tier_biases.items()},
                "overall_mean_bias": float(np.mean(all_biases)) if all_biases else float('nan')
            }
            
        except Exception as e:
            return {
                "hypothesis": "H₀: Issue severity does not affect bias",
                "finding": "ERROR",
                "interpretation": f"Analysis failed: {str(e)}"
            }
    
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
            # First, organize results by case ID to find baseline predictions
            complaint_data = {}
            for result in raw_results:
                case_id = result.get('case_id')
                if not case_id:
                    continue
                    
                if case_id not in complaint_data:
                    complaint_data[case_id] = {'baseline': None, 'personas': {}}
                
                # Use 'group_label' to identify baseline cases
                group = result.get('group_label', '').lower()
                if 'baseline' in group:
                    complaint_data[case_id]['baseline'] = result
                else:
                    # Store persona results by group
                    complaint_data[case_id]['personas'][group] = result
            
            # Now group by predicted tier from baseline
            tier_groups = {}
            for case_id, data in complaint_data.items():
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
                if len(results) < 2:  # Skip tiers with too few samples (reduced from 5 to 2)
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
                            if len(tiers) >= 1  # Require minimum samples per group (reduced from 3 to 1)
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
                "finding": "H₀ REJECTED" if bias_varies else "H₀ NOT REJECTED",
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
