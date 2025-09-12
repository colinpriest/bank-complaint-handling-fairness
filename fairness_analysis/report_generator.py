"""
Report Generator for Advanced Fairness Analysis

This module provides functionality to generate comprehensive reports
from fairness analysis results.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


class ReportGenerator:
    """Generates comprehensive reports from fairness analysis results"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results_dir.mkdir(exist_ok=True)
        
    def generate_comprehensive_report(self, analyses: Dict[str, Any], 
                                    filename: str = "advanced_research_summary.md") -> str:
        """Generate a comprehensive analysis report"""
        report_path = self.results_dir / filename
        
        # Create basic report structure
        report_content = self._create_report_header()
        report_content += self._create_analysis_sections(analyses)
        report_content += self._create_report_footer()
        
        # Write report to file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        return str(report_path)
        
    def generate_directional_fairness_report(self, analyses: Dict[str, Any]) -> str:
        """Generate directional fairness specific report"""
        report_path = self.results_dir / "directional_fairness_report.md"
        
        report_content = "# Directional Fairness Analysis Report\n\n"
        report_content += "This report focuses on directional fairness patterns in LLM responses.\n\n"
        
        if "bias_directional_consistency" in analyses:
            report_content += "## Directional Bias Analysis\n\n"
            report_content += "Analysis of consistent directional biases across different demographic groups.\n\n"
        
        # Write report to file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        return str(report_path)
        
    def _create_report_header(self) -> str:
        """Create the header section of the report"""
        return """# Advanced LLM Fairness Analysis Report

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

"""
        
    def _create_analysis_sections(self, analyses: Dict[str, Any]) -> str:
        """Create analysis sections based on available data"""
        content = "## Analysis Results\n\n"
        
        for analysis_name, analysis_data in analyses.items():
            content += f"### {analysis_name.replace('_', ' ').title()}\n\n"
            
            if isinstance(analysis_data, dict):
                content += self._format_analysis_data(analysis_data, analysis_name, all_analyses=analyses)
            else:
                content += f"Analysis completed: {type(analysis_data).__name__}\n\n"
                
        return content
        
    def _format_analysis_data(self, data: Dict[str, Any], analysis_name: str = "", all_analyses: Optional[Dict[str, Any]] = None) -> str:
        """Format analysis data for report display"""
        content = ""
        
        # Add hypothesis for gender effects analysis
        if analysis_name == "gender_effects":
            content += "- **Hypothesis**: H₀: Subtle gender injection does not affect remedy tier assignments\n"
        
        # Add hypothesis for ethnicity effects analysis
        if analysis_name == "ethnicity_effects":
            content += "- **Hypothesis**: H₀: Subtle ethnicity injection does not affect remedy tier assignments\n"
        
        # Add hypothesis for geography effects analysis
        if analysis_name == "geography_effects":
            content += "- **Hypothesis**: H₀: Subtle geographic and socio-economic injection does not affect remedy tier assignments\n"

        # Add hypothesis for granular bias analysis
        if analysis_name == "granular_bias":
            content += "- **Hypothesis**: H₀: Subtle demographic injection affects remedy tier assignments the same for all groups\n"

        # Add hypothesis for bias directional consistency analysis
        if analysis_name == "bias_directional_consistency":
            content += "- **Hypothesis**: H₀: Mean bias outcomes are equally positive or negative\n"

        # Add hypotheses for fairness strategies analysis
        if analysis_name == "fairness_strategies":
            content += "- **Hypothesis 1**: H₀: Fairness strategies do not affect bias\n"
            content += "- **Hypothesis 2**: H₀: All fairness strategies are equally effective\n"

        # Add hypotheses for process fairness analysis
        if analysis_name == "process_fairness":
            content += "- **Hypothesis (Group ANOVA)**: H₀: There are no differences in process fairness between demographic groups.\n"
            content += "- **Hypothesis (Baseline vs Personas)**: H₀: There are no differences in process when demographic data is added (Baseline vs all personas combined).\n"
        
        # Define remedy tier labels
        tier_labels = {
            0: "No action taken",
            1: "Process improvement", 
            2: "Small monetary remedy",
            3: "Moderate monetary remedy",
            4: "High monetary remedy"
        }
        
        # Special handling for Process Fairness findings to avoid ambiguity
        if analysis_name == "process_fairness":
            group_sig = data.get('significant_indicators')
            group_total = data.get('total_indicators')
            bvp_sig = data.get('baseline_vs_personas_significant_indicators')
            bvp_total = data.get('baseline_vs_personas_total_indicators')

            # Group ANOVA finding with H0 status
            if isinstance(group_sig, int) and isinstance(group_total, int):
                h0_status = "H₀ REJECTED" if group_sig > 0 else "H₀ NOT REJECTED"
                interp = (
                    "Differences detected between demographic groups"
                    if group_sig > 0 else "No significant differences between demographic groups"
                )
                content += f"- **Finding (Group ANOVA)**: {h0_status} — {interp} ({group_sig}/{group_total} indicators significant).\n"
            else:
                content += f"- **Finding (Group ANOVA)**: NOT TESTED\n"

            # Baseline vs Personas finding with H0 status
            if isinstance(bvp_sig, int) and isinstance(bvp_total, int):
                bvp_h0 = "H₀ REJECTED" if bvp_sig > 0 else "H₀ NOT REJECTED"
                bvp_interp = (
                    "Process indicators differ when demographic data is added"
                    if bvp_sig > 0 else "No significant change in process indicators when demographic data is added"
                )
                content += f"- **Finding (Baseline vs Personas)**: {bvp_h0} — {bvp_interp} ({bvp_sig}/{bvp_total} indicators significant).\n"
            else:
                content += f"- **Finding (Baseline vs Personas)**: NOT TESTED\n"

        # Special handling for Fairness Strategies findings
        if analysis_name == "fairness_strategies":
            # Hypothesis 1 results
            finding_h1 = data.get('finding_h1', 'NOT TESTED')
            t_stat_h1 = data.get('t_statistic_h1', float('nan'))
            p_value_h1 = data.get('p_value_h1', float('nan'))
            interpretation_h1 = data.get('interpretation_h1', 'No interpretation available')
            
            content += f"- **Finding 1**: {finding_h1}\n"
            if not str(t_stat_h1).lower() in ['nan', 'inf', '-inf']:
                content += f"- **T-Statistic 1**: {t_stat_h1:.3f}\n"
            if not str(p_value_h1).lower() in ['nan', 'inf', '-inf']:
                content += f"- **P-Value 1**: {p_value_h1:.3f}\n"
            content += f"- **Interpretation 1**: {interpretation_h1}\n"
            
            # Hypothesis 2 results
            finding_h2 = data.get('finding_h2', 'NOT TESTED')
            f_stat_h2 = data.get('f_statistic_h2', float('nan'))
            p_value_h2 = data.get('p_value_h2', float('nan'))
            interpretation_h2 = data.get('interpretation_h2', 'No interpretation available')
            
            content += f"- **Finding 2**: {finding_h2}\n"
            if not str(f_stat_h2).lower() in ['nan', 'inf', '-inf']:
                content += f"- **F-Statistic 2**: {f_stat_h2:.3f}\n"
            if not str(p_value_h2).lower() in ['nan', 'inf', '-inf']:
                content += f"- **P-Value 2**: {p_value_h2:.3f}\n"
            content += f"- **Interpretation 2**: {interpretation_h2}\n"

        for key, value in data.items():
            # Skip generic finding for process_fairness (we already printed two findings above)
            if analysis_name == "process_fairness" and key.lower() == 'finding':
                continue
            # Skip individual hypothesis findings for fairness_strategies (we already printed them above)
            if analysis_name == "fairness_strategies" and key in ['finding_h1', 'finding_h2', 't_statistic_h1', 'p_value_h1', 'interpretation_h1', 'f_statistic_h2', 'p_value_h2', 'interpretation_h2']:
                continue
            if isinstance(value, (int, float)):
                # Format floating point numbers to 3 decimal places
                if isinstance(value, float):
                    # Handle special cases like NaN and infinity
                    if str(value).lower() in ['nan', 'inf', '-inf']:
                        formatted_value = str(value)
                    else:
                        formatted_value = f"{value:.3f}"
                else:
                    formatted_value = str(value)
                
                # Add tier interpretation for mean values (but not differences)
                if ('mean' in key.lower() and 'difference' not in key.lower() and 
                    isinstance(value, (int, float)) and not str(value).lower() in ['nan', 'inf', '-inf']):
                    tier_num = round(value)
                    if tier_num in tier_labels:
                        formatted_value += f" (Tier {tier_num}: {tier_labels[tier_num]})"
                
                content += f"- **{key.replace('_', ' ').title()}**: {formatted_value}\n"
            elif isinstance(value, dict):
                # Check if this is a means dictionary that needs tier interpretation
                if 'means' in key.lower():
                    # Special table formatting for Process Fairness group means
                    if analysis_name == "process_fairness" and all(isinstance(v, dict) for v in value.values()):
                        content += f"- **{key.replace('_', ' ').title()}**:\n\n"
                        # Determine columns from nested metric keys
                        columns = set()
                        for metrics in value.values():
                            if isinstance(metrics, dict):
                                columns.update(metrics.keys())
                        # Preferred ordering for known process fairness indicators
                        preferred_order = ['monetary', 'escalation', 'asked_question', 'evidence_ok', 'format_ok', 'refusal']
                        ordered_cols = [c for c in preferred_order if c in columns]
                        ordered_cols += [c for c in sorted(columns) if c not in ordered_cols]
                        # Header (no single p-value column)
                        header = "| Group | " + " | ".join(col.replace('_', ' ').title() for col in ordered_cols) + " |\n"
                        separator = "|" + " --- |" * (len(ordered_cols) + 1) + "\n"
                        content += header
                        content += separator
                        # Rows
                        # Prepare per-indicator p-values from indicator_tests if available (used below for details table)
                        indicator_tests = all_analyses.get('process_fairness', {}).get('indicator_tests', {}) if all_analyses else {}
                        for group_name, metrics in value.items():
                            row_cells = [group_name]
                            for col in ordered_cols:
                                cell_val = metrics.get(col, "")
                                if isinstance(cell_val, (int, float)):
                                    try:
                                        row_cells.append(f"{float(cell_val):.3f}")
                                    except Exception:
                                        row_cells.append(str(cell_val))
                                else:
                                    row_cells.append(str(cell_val))
                            content += "| " + " | ".join(row_cells) + " |\n"
                        content += "\n"

                        # Add Indicator Tests details as a table (f)
                        if indicator_tests:
                            content += "- **Indicator Tests**:\n\n"
                            content += "| Indicator | F-Statistic | p-value | Significant |\n"
                            content += "| --- | --- | --- | --- |\n"
                            for ind_name in ordered_cols:
                                if ind_name in indicator_tests:
                                    test = indicator_tests[ind_name]
                                    fstat = test.get('f_statistic', "")
                                    pval = test.get('p_value', "")
                                    sig = "Yes" if test.get('significant') else "No"
                                    fcell = f"{float(fstat):.3f}" if isinstance(fstat, (int, float)) else str(fstat)
                                    pcell = f"{float(pval):.3f}" if isinstance(pval, (int, float)) else str(pval)
                                    content += f"| {ind_name} | {fcell} | {pcell} | {sig} |\n"
                            content += "\n"

                        # Baseline vs Personas combined tests (b)
                        bvp_tests = all_analyses.get('process_fairness', {}).get('baseline_vs_personas_tests', {}) if all_analyses else {}
                        bvp_sig = all_analyses.get('process_fairness', {}).get('baseline_vs_personas_significant_indicators') if all_analyses else None
                        bvp_total = all_analyses.get('process_fairness', {}).get('baseline_vs_personas_total_indicators') if all_analyses else None
                        bvp_interp = all_analyses.get('process_fairness', {}).get('baseline_vs_personas_interpretation') if all_analyses else None
                        if bvp_tests:
                            if bvp_sig is not None and bvp_total is not None:
                                content += f"- **Baseline vs Personas (summary)**: {bvp_sig} of {bvp_total} indicators significant.\n"
                            if bvp_interp:
                                content += f"- **Interpretation (Baseline vs Personas)**: {bvp_interp}\n\n"
                            content += "- **Baseline vs Personas Tests**:\n\n"
                            content += "| Indicator | t-Statistic | p-value | Significant |\n"
                            content += "| --- | --- | --- | --- |\n"
                            # Use the same indicator ordering as ordered_cols where possible
                            for ind in ordered_cols:
                                if ind in bvp_tests:
                                    test = bvp_tests[ind]
                                    tstat = test.get('t_statistic', "")
                                    pval = test.get('p_value', "")
                                    sig = "Yes" if test.get('significant') else "No"
                                    tcell = f"{float(tstat):.3f}" if isinstance(tstat, (int, float)) else str(tstat)
                                    pcell = f"{float(pval):.3f}" if isinstance(pval, (int, float)) else str(pval)
                                    content += f"| {ind} | {tcell} | {pcell} | {sig} |\n"
                            # Include any remaining indicators not in ordered_cols
                            for ind, test in bvp_tests.items():
                                if ind in ordered_cols:
                                    continue
                                tstat = test.get('t_statistic', "")
                                pval = test.get('p_value', "")
                                sig = "Yes" if test.get('significant') else "No"
                                tcell = f"{float(tstat):.3f}" if isinstance(tstat, (int, float)) else str(tstat)
                                pcell = f"{float(pval):.3f}" if isinstance(pval, (int, float)) else str(pval)
                                content += f"| {ind} | {tcell} | {pcell} | {sig} |\n"
                            content += "\n"

                        # Add explanatory details for significant/total indicators (e)
                        sig_count = all_analyses.get('process_fairness', {}).get('significant_indicators') if all_analyses else None
                        tot_count = all_analyses.get('process_fairness', {}).get('total_indicators') if all_analyses else None
                        if sig_count is not None and tot_count is not None:
                            sig_list = [ind for ind, res in indicator_tests.items() if isinstance(res, dict) and res.get('significant')]
                            nonsig_list = [ind for ind, res in indicator_tests.items() if isinstance(res, dict) and not res.get('significant')]
                            content += "- **Significant Indicators**: " + \
                                       f"{sig_count} of {tot_count}. Significant: " + \
                                       (", ".join(sig_list) if sig_list else "None") + "\n"
                            content += "- **Non-significant Indicators**: " + \
                                       (", ".join(nonsig_list) if nonsig_list else "None") + "\n\n"

                        # Derived aggregation tables by Gender, Ethnicity, Geography (b, c, d)
                        def _agg_from_group_means(group_means_dict, classifier):
                            buckets = {}
                            for g, metrics in group_means_dict.items():
                                cat = classifier(g)
                                if not cat:
                                    continue
                                if cat not in buckets:
                                    buckets[cat] = {k: [] for k in ordered_cols}
                                for k in ordered_cols:
                                    val = metrics.get(k)
                                    if isinstance(val, (int, float)):
                                        buckets[cat][k].append(val)
                            # compute simple averages
                            agg = {}
                            for cat, cols in buckets.items():
                                agg[cat] = {k: (float(np.mean(v)) if v else 0.0) for k, v in cols.items()}  # type: ignore[name-defined]
                            return agg

                        # Gender table
                        try:
                            import numpy as np  # type: ignore
                            # Important: check 'female' before 'male' because 'female' contains the substring 'male'
                            gender_agg = _agg_from_group_means(
                                value,
                                lambda g: ('female' if 'female' in g.lower() else ('male' if 'male' in g.lower() else None))
                            )
                            if gender_agg:
                                gp_pval = None
                                if all_analyses and isinstance(all_analyses.get('gender_effects'), dict):
                                    gp_pval = all_analyses['gender_effects'].get('p_value')
                                content += "- **Grouped by Gender**:\n\n"
                                header = "| Gender | " + " | ".join(col.replace('_', ' ').title() for col in ordered_cols) + " | p-value |\n"
                                separator = "|" + " --- |" * (len(ordered_cols) + 2) + "\n"
                                content += header + separator
                                for cat in ['female', 'male']:
                                    if cat in gender_agg:
                                        metrics = gender_agg[cat]
                                        row = [cat]
                                        for col in ordered_cols:
                                            row.append(f"{metrics.get(col, 0.0):.3f}")
                                        row.append(f"{gp_pval:.3f}" if isinstance(gp_pval, (int, float)) else (str(gp_pval) if gp_pval is not None else ""))
                                        content += "| " + " | ".join(row) + " |\n"
                                content += "\n"
                        except Exception:
                            pass

                        # Ethnicity table
                        def _eth(gname: str):
                            gl = gname.lower()
                            if 'black' in gl:
                                return 'black'
                            if 'hispanic' in gl or 'latino' in gl:
                                return 'hispanic'
                            if 'white' in gl:
                                return 'white'
                            if 'asian' in gl:
                                return 'asian'
                            return None
                        eth_agg = _agg_from_group_means(value, _eth)
                        if eth_agg:
                            epv = None
                            if all_analyses and isinstance(all_analyses.get('ethnicity_effects'), dict):
                                epv = all_analyses['ethnicity_effects'].get('p_value')
                            content += "- **Grouped by Ethnicity**:\n\n"
                            header = "| Ethnicity | " + " | ".join(col.replace('_', ' ').title() for col in ordered_cols) + " | p-value |\n"
                            separator = "|" + " --- |" * (len(ordered_cols) + 2) + "\n"
                            content += header + separator
                            for cat, metrics in eth_agg.items():
                                row = [cat]
                                for col in ordered_cols:
                                    row.append(f"{metrics.get(col, 0.0):.3f}")
                                row.append(f"{epv:.3f}" if isinstance(epv, (int, float)) else (str(epv) if epv is not None else ""))
                                content += "| " + " | ".join(row) + " |\n"
                            content += "\n"

                        # Geography table
                        def _geo(gname: str):
                            gl = gname.lower()
                            if 'rural' in gl:
                                return 'rural'
                            if 'urban' in gl and 'poor' in gl:
                                return 'urban_poor'
                            if 'working' in gl:
                                return 'urban_poor'
                            if 'affluent' in gl or 'urban' in gl:
                                return 'urban_affluent'
                            return None
                        geo_agg = _agg_from_group_means(value, _geo)
                        if geo_agg:
                            gpv = None
                            if all_analyses and isinstance(all_analyses.get('geography_effects'), dict):
                                gpv = all_analyses['geography_effects'].get('p_value')
                            content += "- **Grouped by Geography**:\n\n"
                            header = "| Geography | " + " | ".join(col.replace('_', ' ').title() for col in ordered_cols) + " | p-value |\n"
                            separator = "|" + " --- |" * (len(ordered_cols) + 2) + "\n"
                            content += header + separator
                            # Keep a stable row order
                            for cat in ['urban_affluent', 'urban_poor', 'rural']:
                                if cat in geo_agg:
                                    metrics = geo_agg[cat]
                                    row = [cat]
                                    for col in ordered_cols:
                                        row.append(f"{metrics.get(col, 0.0):.3f}")
                                    row.append(f"{gpv:.3f}" if isinstance(gpv, (int, float)) else (str(gpv) if gpv is not None else ""))
                                    content += "| " + " | ".join(row) + " |\n"
                            content += "\n"
                    # Special handling for granular bias group means
                    elif analysis_name == "granular_bias" and key == "group_means":
                        content += f"- **{key.replace('_', ' ').title()}**:\n"
                        for group, mean_value in value.items():
                            if isinstance(mean_value, (int, float)):
                                tier_num = round(mean_value)
                                tier_label = tier_labels.get(tier_num, f"Tier {tier_num}")
                                content += f"  - {group}: {mean_value:.3f} ({tier_label})\n"
                            else:
                                content += f"  - {group}: {mean_value}\n"
                    # Special handling for granular bias bias magnitudes
                    elif analysis_name == "granular_bias" and key == "bias_magnitudes":
                        content += f"- **{key.replace('_', ' ').title()}**:\n"
                        for group, bias_value in value.items():
                            content += f"  - {group}: {bias_value:.3f}\n"
                    # Special handling for fairness strategies descriptions
                    elif analysis_name == "fairness_strategies" and key == "strategy_descriptions":
                        content += f"- **{key.replace('_', ' ').title()}**:\n"
                        for strategy, description in value.items():
                            content += f"  - {strategy}: {description}\n"
                    else:
                        content += f"- **{key.replace('_', ' ').title()}**:\n"
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (int, float)):
                                tier_num = round(sub_value)
                                tier_label = tier_labels.get(tier_num, f"Tier {tier_num}")
                                content += f"  - {sub_key}: {sub_value:.3f} ({tier_label})\n"
                            else:
                                content += f"  - {sub_key}: {sub_value}\n"
                else:
                    # Special handling for sample_sizes dictionary
                    if key == 'sample_sizes':
                        content += f"- **{key.replace('_', ' ').title()}**:\n"
                        for sub_key, sub_value in value.items():
                            content += f"  - {sub_key}: {sub_value}\n"
                    else:
                        content += f"- **{key.replace('_', ' ').title()}**: {len(value)} items\n"
            elif isinstance(value, list):
                content += f"- **{key.replace('_', ' ').title()}**: {len(value)} entries\n"
            else:
                content += f"- **{key.replace('_', ' ').title()}**: {str(value)[:100]}...\n"
                
        content += "\n"
        return content
        
    def _create_report_footer(self) -> str:
        """Create the footer section of the report"""
        return """
## Methodology

This analysis uses advanced statistical methods to evaluate fairness patterns in LLM responses across different demographic groups and complaint contexts.

## Data Sources

- CFPB Consumer Complaint Database
- Synthetic demographic injection experiments
- Multi-model comparative analysis

---
*Report generated by Advanced Fairness Analysis System*
"""
