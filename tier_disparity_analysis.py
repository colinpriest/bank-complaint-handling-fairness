#!/usr/bin/env python3
"""
Tier Disparity Analysis Module

This module provides improved metrics for analyzing demographic disparities
in discrete tier assignments (0, 1, 2) that overcome the limitations of
traditional ANOVA/eta-squared approaches.

Author: Bank Complaint Fairness Analysis Team
Date: December 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy.stats import chi2_contingency, fisher_exact
import json


class TierDisparityAnalyzer:
    """
    Analyzer for demographic disparities in tier assignments using
    appropriate metrics for discrete outcomes.
    """

    def __init__(self):
        """Initialize the analyzer"""
        self.tier_names = {
            0: "No Action",
            1: "Non-Monetary Action",
            2: "Monetary Action"
        }

    def analyze_ethnicity_disparities(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive analysis of ethnicity disparities in tier assignments.

        Args:
            data: DataFrame with columns 'ethnicity', 'tier', 'method' (zero_shot/n_shot)

        Returns:
            Dictionary containing all disparity metrics
        """
        results = {
            'tier_distributions': {},
            'disparity_ratios': {},
            'odds_ratios': {},
            'eighty_percent_rule': {},
            'statistical_tests': {},
            'practical_significance': {},
            'summary_metrics': {}
        }

        ethnicities = data['ethnicity'].unique()
        methods = data['method'].unique() if 'method' in data.columns else ['combined']

        for method in methods:
            if method == 'combined':
                method_data = data
            else:
                method_data = data[data['method'] == method]

            # 1. Tier distributions
            results['tier_distributions'][method] = self._calculate_tier_distributions(
                method_data, 'ethnicity', ethnicities
            )

            # 2. Disparity ratios
            results['disparity_ratios'][method] = self._calculate_disparity_ratios(
                results['tier_distributions'][method], ethnicities
            )

            # 3. Odds ratios
            results['odds_ratios'][method] = self._calculate_odds_ratios(
                results['tier_distributions'][method], ethnicities
            )

            # 4. 80% rule compliance
            results['eighty_percent_rule'][method] = self._check_eighty_percent_rule(
                results['tier_distributions'][method], ethnicities
            )

            # 5. Statistical tests
            results['statistical_tests'][method] = self._perform_statistical_tests(
                method_data, 'ethnicity', ethnicities
            )

            # 6. Practical significance assessment
            results['practical_significance'][method] = self._assess_practical_significance(
                results['disparity_ratios'][method],
                results['eighty_percent_rule'][method],
                results['statistical_tests'][method]
            )

        # 7. Summary metrics
        results['summary_metrics'] = self._generate_summary_metrics(results)

        return results

    def _calculate_tier_distributions(self, data: pd.DataFrame,
                                    group_col: str, groups: List[str]) -> Dict[str, Dict]:
        """Calculate tier distribution percentages for each group"""
        distributions = {}

        for group in groups:
            group_data = data[data[group_col] == group]
            total = len(group_data)

            distributions[group] = {
                'total_count': total,
                'tier_0_count': sum(group_data['tier'] == 0),
                'tier_1_count': sum(group_data['tier'] == 1),
                'tier_2_count': sum(group_data['tier'] == 2),
                'tier_0_rate': sum(group_data['tier'] == 0) / total if total > 0 else 0,
                'tier_1_rate': sum(group_data['tier'] == 1) / total if total > 0 else 0,
                'tier_2_rate': sum(group_data['tier'] == 2) / total if total > 0 else 0,
                'mean_tier': group_data['tier'].mean() if total > 0 else 0,
                'std_tier': group_data['tier'].std() if total > 0 else 0
            }

        return distributions

    def _calculate_disparity_ratios(self, distributions: Dict, groups: List[str]) -> Dict[str, Dict]:
        """Calculate disparity ratios between groups for each tier"""
        ratios = {}

        # Find the group with highest rate for each tier (reference group)
        reference_groups = {}
        for tier in [0, 1, 2]:
            tier_key = f'tier_{tier}_rate'
            max_rate = 0
            ref_group = None
            for group in groups:
                if distributions[group][tier_key] > max_rate:
                    max_rate = distributions[group][tier_key]
                    ref_group = group
            reference_groups[tier] = ref_group

        # Calculate ratios relative to reference group
        for tier in [0, 1, 2]:
            tier_key = f'tier_{tier}_rate'
            ref_group = reference_groups[tier]
            ref_rate = distributions[ref_group][tier_key]

            ratios[f'tier_{tier}'] = {
                'reference_group': ref_group,
                'reference_rate': ref_rate,
                'ratios': {}
            }

            for group in groups:
                if group != ref_group:
                    group_rate = distributions[group][tier_key]
                    ratio = group_rate / ref_rate if ref_rate > 0 else 0
                    ratios[f'tier_{tier}']['ratios'][group] = {
                        'rate': group_rate,
                        'ratio': ratio,
                        'passes_80_percent': ratio >= 0.80,
                        'percentage_diff': (group_rate - ref_rate) * 100
                    }

        return ratios

    def _calculate_odds_ratios(self, distributions: Dict, groups: List[str]) -> Dict[str, Dict]:
        """Calculate odds ratios for tier assignments between groups"""
        odds_ratios = {}

        # Calculate for each pair of groups
        for i, group1 in enumerate(groups):
            for group2 in groups[i+1:]:
                pair_key = f'{group1}_vs_{group2}'
                odds_ratios[pair_key] = {}

                for tier in [0, 1, 2]:
                    # Create 2x2 contingency table for this tier
                    # Row 1: Group 1 (tier X, not tier X)
                    # Row 2: Group 2 (tier X, not tier X)
                    tier_count_1 = distributions[group1][f'tier_{tier}_count']
                    other_count_1 = distributions[group1]['total_count'] - tier_count_1
                    tier_count_2 = distributions[group2][f'tier_{tier}_count']
                    other_count_2 = distributions[group2]['total_count'] - tier_count_2

                    # Calculate odds ratio
                    if tier_count_1 > 0 and tier_count_2 > 0 and other_count_1 > 0 and other_count_2 > 0:
                        odds_1 = tier_count_1 / other_count_1
                        odds_2 = tier_count_2 / other_count_2
                        odds_ratio = odds_1 / odds_2

                        # Fisher's exact test for significance
                        contingency = [[tier_count_1, other_count_1],
                                     [tier_count_2, other_count_2]]
                        try:
                            _, p_value = fisher_exact(contingency)
                        except:
                            p_value = 1.0

                        odds_ratios[pair_key][f'tier_{tier}'] = {
                            'odds_ratio': odds_ratio,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'contingency_table': contingency,
                            'interpretation': self._interpret_odds_ratio(odds_ratio)
                        }
                    else:
                        odds_ratios[pair_key][f'tier_{tier}'] = {
                            'odds_ratio': None,
                            'p_value': 1.0,
                            'significant': False,
                            'interpretation': 'Cannot calculate (zero counts)'
                        }

        return odds_ratios

    def _interpret_odds_ratio(self, odds_ratio: float) -> str:
        """Interpret odds ratio magnitude"""
        if odds_ratio is None:
            return "Not calculable"
        elif odds_ratio < 0.5:
            return "Strong negative association"
        elif odds_ratio < 0.8:
            return "Moderate negative association"
        elif odds_ratio <= 1.25:
            return "No meaningful association"
        elif odds_ratio <= 2.0:
            return "Moderate positive association"
        else:
            return "Strong positive association"

    def _check_eighty_percent_rule(self, distributions: Dict, groups: List[str]) -> Dict[str, Any]:
        """Check 80% rule compliance for each tier"""
        compliance = {
            'overall_compliance': True,
            'violations': [],
            'tier_compliance': {}
        }

        for tier in [0, 1, 2]:
            tier_key = f'tier_{tier}_rate'

            # Find highest rate for this tier
            rates = [(group, distributions[group][tier_key]) for group in groups]
            rates.sort(key=lambda x: x[1], reverse=True)
            highest_group, highest_rate = rates[0]

            tier_compliance = {
                'highest_group': highest_group,
                'highest_rate': highest_rate,
                'violations': []
            }

            # Check each other group against 80% rule
            for group, rate in rates[1:]:
                ratio = rate / highest_rate if highest_rate > 0 else 1.0
                passes = ratio >= 0.80

                if not passes:
                    violation = {
                        'group': group,
                        'rate': rate,
                        'ratio': ratio,
                        'deficit': (0.80 - ratio) * 100
                    }
                    tier_compliance['violations'].append(violation)
                    compliance['violations'].append({
                        'tier': tier,
                        'tier_name': self.tier_names[tier],
                        **violation
                    })
                    compliance['overall_compliance'] = False

            compliance['tier_compliance'][tier] = tier_compliance

        return compliance

    def _perform_statistical_tests(self, data: pd.DataFrame,
                                 group_col: str, groups: List[str]) -> Dict[str, Any]:
        """Perform appropriate statistical tests for discrete tier outcomes"""
        tests = {}

        # 1. Chi-squared test for independence (overall association)
        contingency_table = []
        for group in groups:
            group_data = data[data[group_col] == group]
            tier_counts = [
                sum(group_data['tier'] == 0),
                sum(group_data['tier'] == 1),
                sum(group_data['tier'] == 2)
            ]
            contingency_table.append(tier_counts)

        try:
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

            # Calculate Cramér's V
            n = sum(sum(row) for row in contingency_table)
            cramers_v = np.sqrt(chi2_stat / (n * min(len(contingency_table)-1, len(contingency_table[0])-1)))

            tests['chi_squared'] = {
                'statistic': chi2_stat,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'cramers_v': cramers_v,
                'significant': p_value < 0.05,
                'contingency_table': contingency_table,
                'expected_frequencies': expected.tolist()
            }
        except Exception as e:
            tests['chi_squared'] = {'error': str(e)}

        # 2. Pairwise tests for each tier
        tests['pairwise_tests'] = {}
        for tier in [0, 1, 2]:
            tier_tests = {}
            for i, group1 in enumerate(groups):
                for group2 in groups[i+1:]:
                    pair_key = f'{group1}_vs_{group2}'

                    # Create 2x2 table for this specific tier
                    group1_data = data[data[group_col] == group1]
                    group2_data = data[data[group_col] == group2]

                    g1_tier = sum(group1_data['tier'] == tier)
                    g1_other = len(group1_data) - g1_tier
                    g2_tier = sum(group2_data['tier'] == tier)
                    g2_other = len(group2_data) - g2_tier

                    table = [[g1_tier, g1_other], [g2_tier, g2_other]]

                    try:
                        _, p_value = fisher_exact(table)
                        tier_tests[pair_key] = {
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'contingency_table': table
                        }
                    except Exception as e:
                        tier_tests[pair_key] = {'error': str(e)}

            tests['pairwise_tests'][f'tier_{tier}'] = tier_tests

        return tests

    def _assess_practical_significance(self, disparity_ratios: Dict,
                                     eighty_percent_rule: Dict,
                                     statistical_tests: Dict) -> Dict[str, Any]:
        """Assess practical significance using appropriate thresholds for discrete outcomes"""

        assessment = {
            'severity_level': 'minimal',
            'violations_count': len(eighty_percent_rule['violations']),
            'material_disparities': [],
            'concerning_patterns': [],
            'recommendations': []
        }

        # Count 80% rule violations by severity
        severe_violations = [v for v in eighty_percent_rule['violations'] if v['ratio'] < 0.70]
        material_violations = [v for v in eighty_percent_rule['violations'] if 0.70 <= v['ratio'] < 0.80]

        # Assess severity
        if severe_violations:
            assessment['severity_level'] = 'severe'
            assessment['recommendations'].extend([
                "Immediate investigation required - severe disparities detected",
                "Consider model adjustment or replacement",
                "Implement enhanced bias testing"
            ])
        elif material_violations:
            assessment['severity_level'] = 'material'
            assessment['recommendations'].extend([
                "Investigation needed - material disparities detected",
                "Enhanced monitoring recommended",
                "Review decision-making process"
            ])
        elif assessment['violations_count'] > 0:
            assessment['severity_level'] = 'concerning'
            assessment['recommendations'].extend([
                "Monitor closely - some disparities detected",
                "Document findings and maintain oversight"
            ])

        # Identify material disparities in tier distributions
        for tier, tier_data in disparity_ratios.items():
            tier_num = int(tier.split('_')[1])
            tier_name = self.tier_names[tier_num]

            for group, ratio_data in tier_data['ratios'].items():
                if ratio_data['ratio'] < 0.80:
                    assessment['material_disparities'].append({
                        'tier': tier_num,
                        'tier_name': tier_name,
                        'group': group,
                        'ratio': ratio_data['ratio'],
                        'rate': ratio_data['rate'],
                        'reference_rate': tier_data['reference_rate'],
                        'percentage_diff': ratio_data['percentage_diff']
                    })

        # Add statistical significance context
        if 'chi_squared' in statistical_tests and statistical_tests['chi_squared'].get('significant', False):
            assessment['concerning_patterns'].append(
                f"Overall association between demographics and tier assignment is statistically significant "
                f"(p = {statistical_tests['chi_squared']['p_value']:.4f})"
            )

        return assessment

    def _generate_summary_metrics(self, results: Dict) -> Dict[str, Any]:
        """Generate high-level summary metrics"""
        summary = {
            'worst_disparities': [],
            'overall_assessment': {},
            'key_findings': []
        }

        # Find worst disparities across all methods
        for method, practical_sig in results['practical_significance'].items():
            for disparity in practical_sig['material_disparities']:
                disparity['method'] = method
                summary['worst_disparities'].append(disparity)

        # Sort by ratio (worst first)
        summary['worst_disparities'].sort(key=lambda x: x['ratio'])

        # Overall assessment
        max_severity = 'minimal'
        for method, practical_sig in results['practical_significance'].items():
            severity = practical_sig['severity_level']
            if severity == 'severe' or max_severity == 'minimal':
                max_severity = severity
            elif severity == 'material' and max_severity not in ['severe']:
                max_severity = severity

        summary['overall_assessment'] = {
            'severity_level': max_severity,
            'requires_action': max_severity in ['severe', 'material'],
            'total_violations': sum(len(results['eighty_percent_rule'][method]['violations'])
                                  for method in results['eighty_percent_rule'].keys())
        }

        return summary

    def generate_html_report_section(self, results: Dict, title: str = "Tier Disparity Analysis") -> str:
        """Generate HTML section for tier disparity analysis"""

        html = f"""
        <div class="tier-disparity-analysis">
            <h3>{title}</h3>
        """

        # Summary section
        summary = results['summary_metrics']
        severity = summary['overall_assessment']['severity_level']
        severity_color = {
            'severe': '#d32f2f',
            'material': '#f57c00',
            'concerning': '#fbc02d',
            'minimal': '#388e3c'
        }.get(severity, '#666')

        html += f"""
            <div class="summary-alert" style="border-left: 4px solid {severity_color}; padding: 12px; margin: 16px 0; background: #f9f9f9;">
                <h4 style="margin: 0 0 8px 0; color: {severity_color};">
                    Overall Assessment: {severity.upper()}
                </h4>
                <p><strong>Total 80% Rule Violations:</strong> {summary['overall_assessment']['total_violations']}</p>
                <p><strong>Action Required:</strong> {'Yes' if summary['overall_assessment']['requires_action'] else 'No'}</p>
            </div>
        """

        # Method-specific results
        for method in results['tier_distributions'].keys():
            html += self._generate_method_html(results, method)

        html += "</div>"
        return html

    def _generate_method_html(self, results: Dict, method: str) -> str:
        """Generate HTML for a specific method (zero-shot/n-shot)"""

        html = f"""
        <div class="method-section">
            <h4>{method.replace('_', '-').title()} Results</h4>
        """

        # Tier distributions table
        distributions = results['tier_distributions'][method]
        html += """
            <div class="tier-distributions">
                <h5>Tier Distribution by Ethnicity</h5>
                <table class="results-table">
                    <thead>
                        <tr>
                            <th>Ethnicity</th>
                            <th>Total</th>
                            <th>Tier 0 (%)</th>
                            <th>Tier 1 (%)</th>
                            <th>Tier 2 (%)</th>
                            <th>Mean Tier</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        for ethnicity, data in distributions.items():
            html += f"""
                        <tr>
                            <td><strong>{ethnicity.title()}</strong></td>
                            <td>{data['total_count']:,}</td>
                            <td>{data['tier_0_rate']:.1%} ({data['tier_0_count']})</td>
                            <td>{data['tier_1_rate']:.1%} ({data['tier_1_count']})</td>
                            <td>{data['tier_2_rate']:.1%} ({data['tier_2_count']})</td>
                            <td>{data['mean_tier']:.3f}</td>
                        </tr>
            """

        html += """
                    </tbody>
                </table>
            </div>
        """

        # 80% Rule compliance
        eighty_percent = results['eighty_percent_rule'][method]
        if eighty_percent['violations']:
            html += """
                <div class="eighty-percent-violations">
                    <h5 style="color: #d32f2f;">80% Rule Violations</h5>
                    <table class="results-table">
                        <thead>
                            <tr>
                                <th>Tier</th>
                                <th>Group</th>
                                <th>Rate</th>
                                <th>Ratio</th>
                                <th>Deficit</th>
                            </tr>
                        </thead>
                        <tbody>
            """

            for violation in eighty_percent['violations']:
                html += f"""
                            <tr>
                                <td>{violation['tier']} - {violation['tier_name']}</td>
                                <td>{violation['group'].title()}</td>
                                <td>{violation['rate']:.1%}</td>
                                <td style="color: #d32f2f;"><strong>{violation['ratio']:.1%}</strong></td>
                                <td>{violation['deficit']:.1f}pp</td>
                            </tr>
                """

            html += """
                        </tbody>
                    </table>
                </div>
            """
        else:
            html += '<p style="color: #388e3c;"><strong>✓ No 80% Rule violations detected</strong></p>'

        # Statistical tests summary
        stats = results['statistical_tests'][method]
        if 'chi_squared' in stats and 'error' not in stats['chi_squared']:
            chi2_data = stats['chi_squared']
            html += f"""
                <div class="statistical-summary">
                    <h5>Statistical Tests</h5>
                    <p><strong>Chi-squared test:</strong> χ²({chi2_data['degrees_of_freedom']}) = {chi2_data['statistic']:.3f},
                       p = {chi2_data['p_value']:.4f} {'(significant)' if chi2_data['significant'] else '(not significant)'}</p>
                    <p><strong>Cramér's V:</strong> {chi2_data['cramers_v']:.3f}</p>
                </div>
            """

        # Practical significance
        practical = results['practical_significance'][method]
        if practical['recommendations']:
            html += """
                <div class="recommendations">
                    <h5>Recommendations</h5>
                    <ul>
            """
            for rec in practical['recommendations']:
                html += f"<li>{rec}</li>"
            html += """
                    </ul>
                </div>
            """

        html += "</div>"
        return html


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)

    # Simulate ethnicity tier data with realistic disparities
    ethnicities = ['asian', 'black', 'latino', 'white']
    n_per_group = 2500

    sample_data = []

    # Different tier probabilities by ethnicity (reflecting real disparities)
    tier_probs = {
        'asian': [0.35, 0.45, 0.20],    # Higher tier 2 rate
        'black': [0.52, 0.42, 0.06],    # Lower tier 2 rate
        'latino': [0.41, 0.46, 0.13],   # Medium tier 2 rate
        'white': [0.47, 0.44, 0.09]     # Lower tier 2 rate
    }

    for ethnicity in ethnicities:
        probs = tier_probs[ethnicity]
        tiers = np.random.choice([0, 1, 2], size=n_per_group, p=probs)

        for tier in tiers:
            sample_data.append({
                'ethnicity': ethnicity,
                'tier': tier,
                'method': 'zero_shot'
            })

    df = pd.DataFrame(sample_data)

    # Test the analyzer
    analyzer = TierDisparityAnalyzer()
    results = analyzer.analyze_ethnicity_disparities(df)

    print("=== TIER DISPARITY ANALYSIS RESULTS ===")
    print(json.dumps(results['summary_metrics'], indent=2, default=str))

    # Generate HTML report
    html_report = analyzer.generate_html_report_section(results)
    print("\n=== HTML REPORT GENERATED ===")
    print("HTML report generated successfully")