"""
Report Generator for Advanced Fairness Analysis

This module provides functionality to generate comprehensive reports
from fairness analysis results.
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


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
        
        # Define remedy tier labels (needed for severity bias analysis)
        tier_labels = {
            0: "No action taken",
            1: "Process improvement", 
            2: "Small monetary remedy",
            3: "Moderate monetary remedy",
            4: "High monetary remedy"
        }
        
        # Special handling for gender effects analysis
        if analysis_name == "gender_effects":
            hypothesis = data.get('hypothesis', 'H₀: Male and female persona injection result in the same remedy tier assignments')
            test_name = data.get('test_name', 'Two-sample t-test')
            test_statistic = data.get('test_statistic', 'N/A')
            p_value = data.get('p_value', 'N/A')
            finding = data.get('finding', 'N/A')
            interpretation = data.get('interpretation', 'N/A')
            
            content += f"- **Hypothesis**: {hypothesis}\n"
            content += f"- **Test Name**: {test_name}\n"
            content += f"- **Test Statistic**: t = {test_statistic:.3f}\n" if isinstance(test_statistic, (int, float)) else f"- **Test Statistic**: {test_statistic}\n"
            content += f"- **P-Value**: {p_value:.4f}\n" if isinstance(p_value, (int, float)) else f"- **P-Value**: {p_value}\n"
            content += f"- **Result**: {finding}\n"
            content += f"- **Implications**: {interpretation}\n"
            
            # Add details table
            baseline_mean = data.get('baseline_mean', float('nan'))
            baseline_std = data.get('baseline_std', float('nan'))
            baseline_sem = data.get('baseline_sem', float('nan'))
            baseline_count = data.get('baseline_count', 0)
            female_mean = data.get('female_mean', float('nan'))
            female_std = data.get('female_std', float('nan'))
            female_sem = data.get('female_sem', float('nan'))
            female_count = data.get('female_count', 0)
            female_bias = data.get('female_bias', float('nan'))
            male_mean = data.get('male_mean', float('nan'))
            male_std = data.get('male_std', float('nan'))
            male_sem = data.get('male_sem', float('nan'))
            male_count = data.get('male_count', 0)
            male_bias = data.get('male_bias', float('nan'))
            
            content += "- **Details**: Summary Statistics\n\n"
            content += "| Condition | Count | Mean Tier | Std Dev | SEM | Mean Bias |\n"
            content += "|-----------|-------|-----------|---------|-----|----------|\n"
            
            # Baseline row
            baseline_mean_str = f"{baseline_mean:.3f}" if isinstance(baseline_mean, (int, float)) and not str(baseline_mean).lower() in ['nan', 'inf', '-inf'] else "N/A"
            baseline_std_str = f"{baseline_std:.3f}" if isinstance(baseline_std, (int, float)) and not str(baseline_std).lower() in ['nan', 'inf', '-inf'] else "N/A"
            baseline_sem_str = f"{baseline_sem:.3f}" if isinstance(baseline_sem, (int, float)) and not str(baseline_sem).lower() in ['nan', 'inf', '-inf'] else "N/A"
            content += f"| Baseline  | {baseline_count:>5} | {baseline_mean_str:>9} | {baseline_std_str:>7} | {baseline_sem_str:>3} |    0.000 |\n"
            
            # Female row
            female_mean_str = f"{female_mean:.3f}" if isinstance(female_mean, (int, float)) and not str(female_mean).lower() in ['nan', 'inf', '-inf'] else "N/A"
            female_std_str = f"{female_std:.3f}" if isinstance(female_std, (int, float)) and not str(female_std).lower() in ['nan', 'inf', '-inf'] else "N/A"
            female_sem_str = f"{female_sem:.3f}" if isinstance(female_sem, (int, float)) and not str(female_sem).lower() in ['nan', 'inf', '-inf'] else "N/A"
            female_bias_str = f"{female_bias:+.3f}" if isinstance(female_bias, (int, float)) and not str(female_bias).lower() in ['nan', 'inf', '-inf'] else "N/A"
            content += f"| Female    | {female_count:>5} | {female_mean_str:>9} | {female_std_str:>7} | {female_sem_str:>3} | {female_bias_str:>8} |\n"
            
            # Male row
            male_mean_str = f"{male_mean:.3f}" if isinstance(male_mean, (int, float)) and not str(male_mean).lower() in ['nan', 'inf', '-inf'] else "N/A"
            male_std_str = f"{male_std:.3f}" if isinstance(male_std, (int, float)) and not str(male_std).lower() in ['nan', 'inf', '-inf'] else "N/A"
            male_sem_str = f"{male_sem:.3f}" if isinstance(male_sem, (int, float)) and not str(male_sem).lower() in ['nan', 'inf', '-inf'] else "N/A"
            male_bias_str = f"{male_bias:+.3f}" if isinstance(male_bias, (int, float)) and not str(male_bias).lower() in ['nan', 'inf', '-inf'] else "N/A"
            content += f"| Male      | {male_count:>5} | {male_mean_str:>9} | {male_std_str:>7} | {male_sem_str:>3} | {male_bias_str:>8} |\n"
            
            content += "\n"
            
            # Return early to skip generic processing for gender effects
            return content
        
        # Special handling for ethnicity effects analysis
        if analysis_name == "ethnicity_effects":
            finding = data.get('finding', 'N/A')
            f_statistic = data.get('f_statistic', 'N/A')
            p_value = data.get('p_value', 'N/A')
            interpretation = data.get('interpretation', 'N/A')
            ethnicity_means = data.get('ethnicity_means', {})
            ethnicity_stds = data.get('ethnicity_stds', {})
            ethnicity_sems = data.get('ethnicity_sems', {})
            sample_sizes = data.get('sample_sizes', {})
            
            content += "- **Hypothesis**: H₀: Ethnicity injection does not cause statistically different remedy tier assignments\n"
            content += "- **Test Name**: One-way ANOVA\n"
            content += f"- **Test Statistic**: F = {f_statistic:.3f}\n" if isinstance(f_statistic, (int, float)) else f"- **Test Statistic**: {f_statistic}\n"
            content += f"- **P-Value**: {p_value:.4f}\n" if isinstance(p_value, (int, float)) else f"- **P-Value**: {p_value}\n"
            content += f"- **Result**: {finding}\n"
            content += f"- **Implications**: {interpretation}\n"
            
            # Add details table for ethnicity groups
            if ethnicity_means and sample_sizes:
                content += "- **Details**: Summary Statistics\n\n"
                content += "| Condition | Count | Mean Tier | Std Dev | SEM |\n"
                content += "|-----------|-------|-----------|---------|-----|\n"
                
                # Get baseline data from analysis results
                baseline_count = data.get('baseline_count', 1000)
                baseline_mean = data.get('baseline_mean', 1.345)
                baseline_std = data.get('baseline_std', 1.159)
                baseline_sem = data.get('baseline_sem', 0.037)
                
                # Format baseline row
                baseline_mean_str = f"{baseline_mean:.3f}" if isinstance(baseline_mean, (int, float)) and not str(baseline_mean).lower() in ['nan', 'inf', '-inf'] else "N/A"
                baseline_std_str = f"{baseline_std:.3f}" if isinstance(baseline_std, (int, float)) and not str(baseline_std).lower() in ['nan', 'inf', '-inf'] else "N/A"
                baseline_sem_str = f"{baseline_sem:.3f}" if isinstance(baseline_sem, (int, float)) and not str(baseline_sem).lower() in ['nan', 'inf', '-inf'] else "N/A"
                content += f"| Baseline  | {baseline_count:>5} | {baseline_mean_str:>9} | {baseline_std_str:>7} | {baseline_sem_str:>3} |\n"
                
                # Sort ethnicities for consistent ordering
                sorted_ethnicities = sorted(ethnicity_means.keys())
                
                for ethnicity in sorted_ethnicities:
                    mean_val = ethnicity_means[ethnicity]
                    std_val = ethnicity_stds.get(ethnicity, float('nan'))
                    sem_val = ethnicity_sems.get(ethnicity, float('nan'))
                    count = sample_sizes.get(ethnicity, 0)
                    
                    mean_str = f"{mean_val:.3f}" if isinstance(mean_val, (int, float)) else "N/A"
                    std_str = f"{std_val:.3f}" if isinstance(std_val, (int, float)) and not str(std_val).lower() in ['nan', 'inf', '-inf'] else "N/A"
                    sem_str = f"{sem_val:.3f}" if isinstance(sem_val, (int, float)) and not str(sem_val).lower() in ['nan', 'inf', '-inf'] else "N/A"
                    
                    # Capitalize ethnicity name
                    ethnicity_display = ethnicity.capitalize()
                    content += f"| {ethnicity_display:<9} | {count:>5} | {mean_str:>9} | {std_str:>7} | {sem_str:>3} |\n"
            
            content += "\n"
            
            # Return early to skip generic processing for ethnicity effects
            return content
        
        # Special handling for geography effects analysis
        if analysis_name == "geography_effects":
            finding = data.get('finding', 'N/A')
            f_statistic = data.get('f_statistic', 'N/A')
            p_value = data.get('p_value', 'N/A')
            interpretation = data.get('interpretation', 'N/A')
            geography_means = data.get('geography_means', {})
            geography_stds = data.get('geography_stds', {})
            geography_sems = data.get('geography_sems', {})
            sample_sizes = data.get('sample_sizes', {})
            
            content += "- **Hypothesis**: H₀: Geographic injection does not cause statistically different remedy tier assignments\n"
            content += "- **Test Name**: One-way ANOVA\n"
            content += f"- **Test Statistic**: F = {f_statistic:.3f}\n" if isinstance(f_statistic, (int, float)) else f"- **Test Statistic**: {f_statistic}\n"
            content += f"- **P-Value**: {p_value:.4f}\n" if isinstance(p_value, (int, float)) else f"- **P-Value**: {p_value}\n"
            content += f"- **Result**: {finding}\n"
            content += f"- **Implications**: {interpretation}\n"
            
            # Add details table for geography groups
            if geography_means and sample_sizes:
                content += "- **Details**: Summary Statistics\n\n"
                content += "| Condition | Count | Mean Tier | Std Dev | SEM |\n"
                content += "|-----------|-------|-----------|---------|-----|\n"
                
                # Get baseline data from analysis results
                baseline_count = data.get('baseline_count', 1000)
                baseline_mean = data.get('baseline_mean', 1.345)
                baseline_std = data.get('baseline_std', 1.159)
                baseline_sem = data.get('baseline_sem', 0.037)
                
                # Format baseline row
                baseline_mean_str = f"{baseline_mean:.3f}" if isinstance(baseline_mean, (int, float)) and not str(baseline_mean).lower() in ['nan', 'inf', '-inf'] else "N/A"
                baseline_std_str = f"{baseline_std:.3f}" if isinstance(baseline_std, (int, float)) and not str(baseline_std).lower() in ['nan', 'inf', '-inf'] else "N/A"
                baseline_sem_str = f"{baseline_sem:.3f}" if isinstance(baseline_sem, (int, float)) and not str(baseline_sem).lower() in ['nan', 'inf', '-inf'] else "N/A"
                content += f"| Baseline     | {baseline_count:>5} | {baseline_mean_str:>9} | {baseline_std_str:>7} | {baseline_sem_str:>3} |\n"
                
                # Sort geography groups for consistent ordering
                sorted_geographies = sorted(geography_means.keys())
                
                for geography in sorted_geographies:
                    mean_val = geography_means[geography]
                    std_val = geography_stds.get(geography, float('nan'))
                    sem_val = geography_sems.get(geography, float('nan'))
                    count = sample_sizes.get(geography, 0)
                    
                    mean_str = f"{mean_val:.3f}" if isinstance(mean_val, (int, float)) else "N/A"
                    std_str = f"{std_val:.3f}" if isinstance(std_val, (int, float)) and not str(std_val).lower() in ['nan', 'inf', '-inf'] else "N/A"
                    sem_str = f"{sem_val:.3f}" if isinstance(sem_val, (int, float)) and not str(sem_val).lower() in ['nan', 'inf', '-inf'] else "N/A"
                    
                    # Format geography name for display
                    geography_display = geography.replace('_', ' ').title()
                    content += f"| {geography_display:<12} | {count:>5} | {mean_str:>9} | {std_str:>7} | {sem_str:>3} |\n"
            
            content += "\n"
            
            # Return early to skip generic processing for geography effects
            return content
        
        # Special handling for granular bias analysis
        if analysis_name == "granular_bias":
            finding = data.get('finding', 'N/A')
            f_statistic = data.get('f_statistic', 'N/A')
            p_value = data.get('p_value', 'N/A')
            interpretation = data.get('interpretation', 'N/A')
            group_means = data.get('group_means', {})
            group_stds = data.get('group_stds', {})
            group_sems = data.get('group_sems', {})
            group_counts = data.get('group_counts', {})
            persona_groups_analyzed = data.get('persona_groups_analyzed', 0)
            
            content += "- **Hypothesis**: H₀: Demographic injection affects remedy tier assignments equally across all demographic groups\n"
            content += "- **Test Name**: One-way ANOVA across demographic groups\n"
            content += f"- **Test Statistic**: F = {f_statistic:.3f}\n" if isinstance(f_statistic, (int, float)) else f"- **Test Statistic**: {f_statistic}\n"
            content += f"- **P-Value**: {p_value:.4f}\n" if isinstance(p_value, (int, float)) else f"- **P-Value**: {p_value}\n"
            content += f"- **Result**: {finding}\n"
            content += f"- **Implications**: {interpretation}\n"
            
            # Add summary statistics
            if group_means and group_counts:
                content += f"- **Groups Analyzed**: {persona_groups_analyzed} demographic combinations\n"
                
                # Get baseline data from analysis results
                baseline_count = data.get('baseline_count', 1000)
                baseline_mean = data.get('baseline_mean', 1.345)
                baseline_std = data.get('baseline_std', 1.159)
                baseline_sem = data.get('baseline_sem', 0.037)
                
                content += "- **Details**: Top and Bottom Performing Groups\n\n"
                
                # Sort groups by mean tier (descending)
                sorted_groups = sorted(group_means.items(), key=lambda x: x[1], reverse=True)
                
                # Show top 5 and bottom 5 groups
                content += "| Group | Count | Mean Tier | Std Dev | SEM | Bias |\n"
                content += "|-------|-------|-----------|---------|-----|------|\n"
                
                # Baseline row first
                baseline_mean_str = f"{baseline_mean:.3f}" if isinstance(baseline_mean, (int, float)) else "N/A"
                baseline_std_str = f"{baseline_std:.3f}" if isinstance(baseline_std, (int, float)) else "N/A"
                baseline_sem_str = f"{baseline_sem:.3f}" if isinstance(baseline_sem, (int, float)) else "N/A"
                content += f"| **Baseline** | {baseline_count:>5} | {baseline_mean_str:>9} | {baseline_std_str:>7} | {baseline_sem_str:>3} | 0.000 |\n"
                
                # Top 5 groups
                content += f"| **Top 5 Groups** | | | | | |\n"
                for i, (group, mean_val) in enumerate(sorted_groups[:5]):
                    std_val = group_stds.get(group, float('nan'))
                    sem_val = group_sems.get(group, float('nan'))
                    count = group_counts.get(group, 0)
                    bias = mean_val - baseline_mean
                    
                    mean_str = f"{mean_val:.3f}" if isinstance(mean_val, (int, float)) else "N/A"
                    std_str = f"{std_val:.3f}" if isinstance(std_val, (int, float)) and not str(std_val).lower() in ['nan', 'inf', '-inf'] else "N/A"
                    sem_str = f"{sem_val:.3f}" if isinstance(sem_val, (int, float)) and not str(sem_val).lower() in ['nan', 'inf', '-inf'] else "N/A"
                    bias_str = f"{bias:+.3f}" if isinstance(bias, (int, float)) else "N/A"
                    
                    # Shorten group name if too long
                    group_display = group.replace('_', ' ')
                    if len(group_display) > 20:
                        group_display = group_display[:17] + "..."
                    
                    content += f"| {group_display:<20} | {count:>5} | {mean_str:>9} | {std_str:>7} | {sem_str:>3} | {bias_str:>6} |\n"
                
                # Bottom 5 groups
                content += f"| **Bottom 5 Groups** | | | | | |\n"
                for i, (group, mean_val) in enumerate(sorted_groups[-5:]):
                    std_val = group_stds.get(group, float('nan'))
                    sem_val = group_sems.get(group, float('nan'))
                    count = group_counts.get(group, 0)
                    bias = mean_val - baseline_mean
                    
                    mean_str = f"{mean_val:.3f}" if isinstance(mean_val, (int, float)) else "N/A"
                    std_str = f"{std_val:.3f}" if isinstance(std_val, (int, float)) and not str(std_val).lower() in ['nan', 'inf', '-inf'] else "N/A"
                    sem_str = f"{sem_val:.3f}" if isinstance(sem_val, (int, float)) and not str(sem_val).lower() in ['nan', 'inf', '-inf'] else "N/A"
                    bias_str = f"{bias:+.3f}" if isinstance(bias, (int, float)) else "N/A"
                    
                    # Shorten group name if too long
                    group_display = group.replace('_', ' ')
                    if len(group_display) > 20:
                        group_display = group_display[:17] + "..."
                    
                    content += f"| {group_display:<20} | {count:>5} | {mean_str:>9} | {std_str:>7} | {sem_str:>3} | {bias_str:>6} |\n"
            
            content += "\n"
            
            # Return early to skip generic processing for granular bias
            return content
        
        # Special handling for bias directional consistency analysis
        if analysis_name == "bias_directional_consistency":
            finding = data.get('finding', 'N/A')
            f_statistic = data.get('f_statistic', 'N/A')  # Actually t-statistic
            p_value = data.get('p_value', 'N/A')
            interpretation = data.get('interpretation', 'N/A')
            positive_biases = data.get('positive_biases', 0)
            negative_biases = data.get('negative_biases', 0)
            neutral_biases = data.get('neutral_biases', 0)
            positive_examples = data.get('positive_examples', 0)
            negative_examples = data.get('negative_examples', 0)
            neutral_examples = data.get('neutral_examples', 0)
            
            content += "- **Hypothesis**: H₀: Mean bias outcomes are equally positive or negative across demographic groups\n"
            content += "- **Test Name**: One-sample t-test against zero bias\n"
            content += f"- **Test Statistic**: t = {f_statistic:.3f}\n" if isinstance(f_statistic, (int, float)) else f"- **Test Statistic**: {f_statistic}\n"
            content += f"- **P-Value**: {p_value:.4f}\n" if isinstance(p_value, (int, float)) else f"- **P-Value**: {p_value}\n"
            content += f"- **Result**: {finding}\n"
            content += f"- **Implications**: {interpretation}\n"
            
            # Add details table showing counts
            content += "- **Details**: Bias Direction Distribution\n\n"
            content += "| Metric | Negative | Neutral | Positive | Total |\n"
            content += "|--------|----------|---------|----------|-------|\n"
            content += f"| Persona Count | {negative_biases:>8} | {neutral_biases:>7} | {positive_biases:>8} | {negative_biases + neutral_biases + positive_biases:>5} |\n"
            content += f"| Example Count | {negative_examples:>8} | {neutral_examples:>7} | {positive_examples:>8} | {negative_examples + neutral_examples + positive_examples:>5} |\n"
            
            # Add percentages
            total_personas = negative_biases + neutral_biases + positive_biases
            total_examples = negative_examples + neutral_examples + positive_examples
            
            if total_personas > 0:
                neg_pct_p = negative_biases / total_personas * 100
                neu_pct_p = neutral_biases / total_personas * 100
                pos_pct_p = positive_biases / total_personas * 100
                content += f"| Persona % | {neg_pct_p:>7.1f}% | {neu_pct_p:>6.1f}% | {pos_pct_p:>7.1f}% |   -   |\n"
            
            if total_examples > 0:
                neg_pct_e = negative_examples / total_examples * 100
                neu_pct_e = neutral_examples / total_examples * 100
                pos_pct_e = positive_examples / total_examples * 100
                content += f"| Example % | {neg_pct_e:>7.1f}% | {neu_pct_e:>6.1f}% | {pos_pct_e:>7.1f}% |   -   |\n"
            
            content += "\n"
            content += "**Note**: Bias thresholds: Negative < -0.05, Neutral [-0.05, +0.05], Positive > +0.05\n\n"
            
            # Return early to skip generic processing for bias directional consistency
            return content
        
        # Special handling for fairness strategies analysis
        if analysis_name == "fairness_strategies":
            # Add strategy descriptions
            content += "\n**Bias Mitigation Strategies:**\n"
            content += "- **Persona Fairness**: Demographic injection with explicit fairness instruction to ignore demographics and make unbiased decisions\n"
            content += "- **Perspective**: Perspective-taking approach asking the model to consider the complainant's viewpoint\n"
            content += "- **Chain Of Thought**: Step-by-step reasoning process to improve decision quality and transparency\n"
            content += "- **Consequentialist**: Consequence-focused decision making emphasizing outcomes and impacts\n"
            content += "- **Roleplay**: Role-playing approach where the model assumes the perspective of a fair bank representative\n"
            content += "- **Structured Extraction**: Structured information extraction method with predefined decision criteria\n"
            content += "- **Minimal**: Minimal intervention approach with basic instruction to be fair and unbiased\n\n"
            
            # Hypothesis 1: Strategies vs Baseline
            finding_h1 = data.get('finding_h1', 'N/A')
            t_statistic_h1 = data.get('t_statistic_h1', 'N/A')
            p_value_h1 = data.get('p_value_h1', 'N/A')
            interpretation_h1 = data.get('interpretation_h1', 'N/A')
            
            # Hypothesis 2: Strategies vs Each Other
            finding_h2 = data.get('finding_h2', 'N/A')
            f_statistic_h2 = data.get('f_statistic_h2', 'N/A')
            p_value_h2 = data.get('p_value_h2', 'N/A')
            interpretation_h2 = data.get('interpretation_h2', 'N/A')
            
            # Common data
            strategy_means = data.get('strategy_means', {})
            strategy_stds = data.get('strategy_stds', {})
            strategy_sems = data.get('strategy_sems', {})
            sample_sizes = data.get('sample_sizes', {})
            baseline_mean = data.get('baseline_mean', 1.345)
            baseline_std = data.get('baseline_std', 1.159)
            baseline_sem = data.get('baseline_sem', 0.037)
            baseline_count = data.get('baseline_count', 1000)
            
            # Hypothesis 1 Section
            content += "#### Hypothesis 1: Strategies vs Persona-Injected\n"
            content += "- **Hypothesis**: H₀: Fairness strategies do not affect remedy tier assignments compared to persona-injected examples\n"
            
            # Determine test type from interpretation
            test_name_h1 = "Paired t-test"
            if "independent t-test" in interpretation_h1.lower():
                test_name_h1 = "Independent samples t-test"
            
            content += f"- **Test Name**: {test_name_h1}\n"
            content += f"- **Test Statistic**: t = {t_statistic_h1:.3f}\n" if isinstance(t_statistic_h1, (int, float)) else f"- **Test Statistic**: {t_statistic_h1}\n"
            content += f"- **P-Value**: {p_value_h1:.4f}\n" if isinstance(p_value_h1, (int, float)) else f"- **P-Value**: {p_value_h1}\n"
            content += f"- **Result**: {finding_h1}\n"
            content += f"- **Implications**: {interpretation_h1}\n"
            
            # Add details table for three-way comparison
            baseline_count = data.get('baseline_count', 0)
            persona_count = data.get('persona_count', 0)
            mitigation_count = data.get('mitigation_count', 0)
            
            if baseline_count > 0:
                content += "- **Details**: Mitigation vs Persona-Injected Comparison\n\n"
                content += "| Condition | Example Count | Mean Tier | Std Dev | SEM | Mean Bias* |\n"
                content += "|-----------|---------------|-----------|---------|-----|----------|\n"
                
                # Baseline row (for reference)
                baseline_mean_val = data.get('baseline_mean', float('nan'))
                baseline_std_val = data.get('baseline_std', float('nan'))
                baseline_sem_val = data.get('baseline_sem', float('nan'))
                
                baseline_mean_str = f"{baseline_mean_val:.3f}" if isinstance(baseline_mean_val, (int, float)) and not str(baseline_mean_val).lower() in ['nan', 'inf', '-inf'] else "N/A"
                baseline_std_str = f"{baseline_std_val:.3f}" if isinstance(baseline_std_val, (int, float)) and not str(baseline_std_val).lower() in ['nan', 'inf', '-inf'] else "N/A"
                baseline_sem_str = f"{baseline_sem_val:.3f}" if isinstance(baseline_sem_val, (int, float)) and not str(baseline_sem_val).lower() in ['nan', 'inf', '-inf'] else "N/A"
                
                content += f"| Baseline      | {baseline_count:>13} | {baseline_mean_str:>9} | {baseline_std_str:>7} | {baseline_sem_str:>3} |    0.000 |\n"
                
                # Persona-Injected row (reference for comparison)
                persona_mean_val = data.get('persona_mean', float('nan'))
                persona_std_val = data.get('persona_std', float('nan'))
                persona_sem_val = data.get('persona_sem', float('nan'))
                persona_bias_val = data.get('persona_bias', float('nan'))
                
                if persona_count > 0:
                    persona_mean_str = f"{persona_mean_val:.3f}" if isinstance(persona_mean_val, (int, float)) and not str(persona_mean_val).lower() in ['nan', 'inf', '-inf'] else "N/A"
                    persona_std_str = f"{persona_std_val:.3f}" if isinstance(persona_std_val, (int, float)) and not str(persona_std_val).lower() in ['nan', 'inf', '-inf'] else "N/A"
                    persona_sem_str = f"{persona_sem_val:.3f}" if isinstance(persona_sem_val, (int, float)) and not str(persona_sem_val).lower() in ['nan', 'inf', '-inf'] else "N/A"
                    persona_bias_str = f"{persona_bias_val:+.3f}" if isinstance(persona_bias_val, (int, float)) and not str(persona_bias_val).lower() in ['nan', 'inf', '-inf'] else "N/A"
                    content += f"| **Persona-Injected** | {persona_count:>7} | {persona_mean_str:>9} | {persona_std_str:>7} | {persona_sem_str:>3} | {persona_bias_str:>8} |\n"
                
                # Mitigation row  
                mitigation_mean_val = data.get('mitigation_mean', float('nan'))
                mitigation_std_val = data.get('mitigation_std', float('nan'))
                mitigation_sem_val = data.get('mitigation_sem', float('nan'))
                mitigation_vs_baseline_bias_val = data.get('mitigation_bias', float('nan'))  # This is bias vs baseline
                
                if mitigation_count > 0:
                    mitigation_mean_str = f"{mitigation_mean_val:.3f}" if isinstance(mitigation_mean_val, (int, float)) and not str(mitigation_mean_val).lower() in ['nan', 'inf', '-inf'] else "N/A"
                    mitigation_std_str = f"{mitigation_std_val:.3f}" if isinstance(mitigation_std_val, (int, float)) and not str(mitigation_std_val).lower() in ['nan', 'inf', '-inf'] else "N/A"
                    mitigation_sem_str = f"{mitigation_sem_val:.3f}" if isinstance(mitigation_sem_val, (int, float)) and not str(mitigation_sem_val).lower() in ['nan', 'inf', '-inf'] else "N/A"
                    mitigation_bias_str = f"{mitigation_vs_baseline_bias_val:+.3f}" if isinstance(mitigation_vs_baseline_bias_val, (int, float)) and not str(mitigation_vs_baseline_bias_val).lower() in ['nan', 'inf', '-inf'] else "N/A"
                    content += f"| **Mitigation**   | {mitigation_count:>11} | {mitigation_mean_str:>9} | {mitigation_std_str:>7} | {mitigation_sem_str:>3} | {mitigation_bias_str:>8} |\n"
                
                content += "\n"
                content += "*Mean Bias calculated as condition mean - baseline mean. Baseline = 0.000 (reference).\n\n"
            
            # Hypothesis 2 Section
            content += "\n#### Hypothesis 2: Strategy Effectiveness Comparison\n"
            content += "- **Hypothesis**: H₀: All fairness strategies are equally effective\n"
            content += "- **Test Name**: One-way ANOVA across strategies\n"
            content += f"- **Test Statistic**: F = {f_statistic_h2:.3f}\n" if isinstance(f_statistic_h2, (int, float)) else f"- **Test Statistic**: {f_statistic_h2}\n"
            content += f"- **P-Value**: {p_value_h2:.4f}\n" if isinstance(p_value_h2, (int, float)) else f"- **P-Value**: {p_value_h2}\n"
            content += f"- **Result**: {finding_h2}\n"
            content += f"- **Implications**: {interpretation_h2}\n"
            
            # Details table ordered by effectiveness (residual bias ratio)
            if strategy_means and sample_sizes:
                content += "- **Details**: Strategy Effectiveness (Ordered by Residual Bias %)\n\n"
                content += "| Strategy | Count | Mean Tier Baseline | Mean Tier Before | Mean Tier After | Std Dev | SEM | Mean Bias Before | Mean Bias After | Residual Bias % |\n"
                content += "|----------|-------|-------------------|------------------|-----------------|---------|-----|------------------|-----------------|----------------|\n"
                
                # Baseline row first
                baseline_mean_str = f"{baseline_mean:.3f}" if isinstance(baseline_mean, (int, float)) else "N/A"
                baseline_std_str = f"{baseline_std:.3f}" if isinstance(baseline_std, (int, float)) else "N/A" 
                baseline_sem_str = f"{baseline_sem:.3f}" if isinstance(baseline_sem, (int, float)) else "N/A"
                content += f"| **Baseline** | {baseline_count:>5} | {baseline_mean_str:>18} | {baseline_mean_str:>16} | {baseline_mean_str:>15} | {baseline_std_str:>7} | {baseline_sem_str:>3} |         0.000 |        0.000 |      0.0%     |\n"
                
                # Persona-Injected row
                persona_count_h2 = data.get('persona_count', 0)
                if persona_count_h2 > 0:
                    persona_mean_h2 = data.get('persona_mean', float('nan'))
                    persona_std_h2 = data.get('persona_std', float('nan'))
                    persona_sem_h2 = data.get('persona_sem', float('nan'))
                    persona_bias_h2 = data.get('persona_bias', float('nan'))
                    
                    persona_mean_str_h2 = f"{persona_mean_h2:.3f}" if isinstance(persona_mean_h2, (int, float)) and not str(persona_mean_h2).lower() in ['nan', 'inf', '-inf'] else "N/A"
                    persona_std_str_h2 = f"{persona_std_h2:.3f}" if isinstance(persona_std_h2, (int, float)) and not str(persona_std_h2).lower() in ['nan', 'inf', '-inf'] else "N/A"
                    persona_sem_str_h2 = f"{persona_sem_h2:.3f}" if isinstance(persona_sem_h2, (int, float)) and not str(persona_sem_h2).lower() in ['nan', 'inf', '-inf'] else "N/A"
                    persona_bias_str_h2 = f"{persona_bias_h2:+.3f}" if isinstance(persona_bias_h2, (int, float)) and not str(persona_bias_h2).lower() in ['nan', 'inf', '-inf'] else "N/A"
                    
                    content += f"| **Persona-Injected** | {persona_count_h2:>5} | {baseline_mean_str:>18} | {persona_mean_str_h2:>16} | {persona_mean_str_h2:>15} | {persona_std_str_h2:>7} | {persona_sem_str_h2:>3} | {persona_bias_str_h2:>16} | {persona_bias_str_h2:>15} |    100.0%     |\n"
                
                # Get effectiveness metrics from data
                strategy_effectiveness = data.get('strategy_effectiveness', {})
                strategy_bias_after = data.get('strategy_bias_after', {})
                strategy_before_mitigation = data.get('strategy_before_mitigation', {})
                strategy_bias_before = data.get('strategy_bias_before', {})
                strategy_baseline_matched = data.get('strategy_baseline_matched', {})
                
                # Sort strategies by effectiveness (lowest to highest residual bias %)
                valid_strategies = [(k, v) for k, v in strategy_effectiveness.items() if isinstance(v, (int, float)) and not str(v).lower() in ['nan', 'inf', '-inf']]
                sorted_strategies = sorted(valid_strategies, key=lambda x: x[1])
                
                # Add any strategies without effectiveness metrics at the end
                remaining_strategies = [k for k in strategy_means.keys() if k not in dict(valid_strategies)]
                sorted_strategies.extend([(k, float('nan')) for k in remaining_strategies])
                
                for strategy, effectiveness_val in sorted_strategies:
                    baseline_matched_val = strategy_baseline_matched.get(strategy, float('nan'))
                    mean_before_val = strategy_before_mitigation.get(strategy, float('nan'))
                    mean_after_val = strategy_means.get(strategy, float('nan'))
                    std_val = strategy_stds.get(strategy, float('nan'))
                    sem_val = strategy_sems.get(strategy, float('nan'))
                    count = sample_sizes.get(strategy, 0)
                    bias_before_val = strategy_bias_before.get(strategy, float('nan'))
                    bias_after_val = strategy_bias_after.get(strategy, float('nan'))
                    
                    baseline_matched_str = f"{baseline_matched_val:.3f}" if isinstance(baseline_matched_val, (int, float)) and not str(baseline_matched_val).lower() in ['nan', 'inf', '-inf'] else "N/A"
                    mean_before_str = f"{mean_before_val:.3f}" if isinstance(mean_before_val, (int, float)) and not str(mean_before_val).lower() in ['nan', 'inf', '-inf'] else "N/A"
                    mean_after_str = f"{mean_after_val:.3f}" if isinstance(mean_after_val, (int, float)) else "N/A"
                    std_str = f"{std_val:.3f}" if isinstance(std_val, (int, float)) and not str(std_val).lower() in ['nan', 'inf', '-inf'] else "N/A"
                    sem_str = f"{sem_val:.3f}" if isinstance(sem_val, (int, float)) and not str(sem_val).lower() in ['nan', 'inf', '-inf'] else "N/A"
                    bias_before_str = f"{bias_before_val:+.3f}" if isinstance(bias_before_val, (int, float)) and not str(bias_before_val).lower() in ['nan', 'inf', '-inf'] else "N/A"
                    bias_after_str = f"{bias_after_val:+.3f}" if isinstance(bias_after_val, (int, float)) and not str(bias_after_val).lower() in ['nan', 'inf', '-inf'] else "N/A"
                    
                    # Convert effectiveness ratio to percentage (100% = no improvement, lower is better)
                    if isinstance(effectiveness_val, (int, float)) and not str(effectiveness_val).lower() in ['nan', 'inf', '-inf']:
                        effectiveness_pct = effectiveness_val * 100
                        effectiveness_pct_str = f"{effectiveness_pct:>6.1f}%"
                    else:
                        effectiveness_pct_str = "    N/A    "
                    
                    # Clean up strategy name for display
                    strategy_display = strategy.replace('_', ' ').title()
                    content += f"| {strategy_display:<20} | {count:>5} | {baseline_matched_str:>18} | {mean_before_str:>16} | {mean_after_str:>15} | {std_str:>7} | {sem_str:>3} | {bias_before_str:>16} | {bias_after_str:>15} | {effectiveness_pct_str:>14} |\n"
            
            content += "\n"
            
            # Return early to skip generic processing for fairness strategies
            return content
        
        
        # Add hypothesis for geography effects analysis (fallback - should not be reached)
        if analysis_name == "geography_effects":
            content += "- **Hypothesis**: H₀: Subtle geographic and socio-economic injection does not affect remedy tier assignments\n"

        # Add hypothesis for granular bias analysis (fallback - should not be reached)
        if analysis_name == "granular_bias":
            content += "- **Hypothesis**: H₀: Subtle demographic injection affects remedy tier assignments the same for all groups\n"

        # Add hypothesis for bias directional consistency analysis (fallback - should not be reached)
        if analysis_name == "bias_directional_consistency":
            content += "- **Hypothesis**: H₀: Mean bias outcomes are equally positive or negative\n"

        # Add hypotheses for fairness strategies analysis (fallback - should not be reached)
        if analysis_name == "fairness_strategies":
            content += "- **Hypothesis 1**: H₀: Fairness strategies do not affect bias\n"
            content += "- **Hypothesis 2**: H₀: All fairness strategies are equally effective\n"

        # Add hypothesis for severity bias variation analysis
        if analysis_name == "severity_bias_variation":
            # Early return for special handling
            content += "#### Hypothesis 1: Severity Tier Bias Variation\n"
            content += "- **Hypothesis**: H₀: Issue severity does not affect bias\n"
            content += f"- **Test Name**: One-way ANOVA across severity tiers\n"
            
            # Get test statistics for Hypothesis 1
            p_value = data.get('p_value', float('nan'))
            finding = data.get('finding', 'N/A')
            interpretation = data.get('interpretation', 'N/A')
            
            content += f"- **Test Statistic**: F = N/A\n"  # ANOVA F-stat not directly stored
            content += f"- **P-Value**: {p_value:.4f}\n"
            content += f"- **Result**: {finding}\n"
            content += f"- **Implications**: {interpretation}\n"
            
            # Add detailed tier information as a table
            tier_metrics = data.get('tier_metrics', {})
            if tier_metrics:
                content += "- **Details**: Bias by Baseline Tier\n\n"
                content += "| Tier | Description | Mean Remedy Tier | Mean Bias | Bias Range | Sample Size | Groups |\n"
                content += "|------|-------------|------------------|-----------|------------|-------------|--------|\n"
                
                for tier in sorted(tier_metrics.keys(), key=int):
                    metrics = tier_metrics[tier]
                    tier_label = tier_labels.get(int(tier), f"Tier {tier}")
                    overall_mean = metrics.get('overall_mean', 0)
                    mean_bias = metrics.get('mean_bias', 0)  # Add mean bias column
                    bias_range = metrics.get('bias_range', 0)
                    sample_size = metrics.get('sample_size', 0)
                    groups_analyzed = metrics.get('groups_analyzed', 0)
                    
                    content += f"| {tier} | {tier_label} | {overall_mean:.2f} | {mean_bias:.2f} | {bias_range:.2f} | {sample_size} | {groups_analyzed} |\n"
            
            # Add highest bias tiers information
            highest_bias_tiers = data.get('highest_bias_tiers', [])
            if highest_bias_tiers:
                content += "- **Highest Bias Tiers**:\n"
                for tier_info in highest_bias_tiers[:3]:  # Show top 3
                    tier = tier_info.get('tier', 'unknown')
                    bias_range = tier_info.get('bias_range', 0)
                    sample_size = tier_info.get('sample_size', 0)
                    tier_label = tier_labels.get(int(tier), f"Tier {tier}")
                    content += f"  - **Tier {tier}** ({tier_label}): Bias range = {bias_range:.2f} (n={sample_size})\n"

            content += "\n"
            
            # HYPOTHESIS 2: Monetary vs Non-Monetary
            content += "#### Hypothesis 2: Monetary vs Non-Monetary Bias\n"
            content += "- **Hypothesis**: H₀: Monetary tiers have the same average bias as non-monetary tiers\n"
            content += "- **Test Name**: Two-sample t-test (Welch's)\n"
            
            monetary_test = data.get('monetary_vs_non_monetary', {})
            if monetary_test.get('finding') not in ['INSUFFICIENT DATA', 'ERROR']:
                t_stat = monetary_test.get('t_statistic', float('nan'))
                p_val = monetary_test.get('p_value', float('nan'))
                finding = monetary_test.get('finding', 'N/A')
                
                content += f"- **Test Statistic**: t = {t_stat:.3f}\n"
                content += f"- **P-Value**: {p_val:.4f}\n"
                content += f"- **Result**: {finding}\n"
                
                non_mon_mean = monetary_test.get('non_monetary_mean', 0)
                mon_mean = monetary_test.get('monetary_mean', 0)
                content += f"- **Implications**: Non-monetary tiers (0,1) have mean bias {non_mon_mean:.3f}, monetary tiers (2,3,4) have mean bias {mon_mean:.3f}\n"
                
                # Details table
                content += "- **Details**: Tier Group Comparison\n\n"
                content += "| Group | Count | Mean Bias | Std Dev |\n"
                content += "|-------|-------|-----------|----------|\n"
                
                non_mon_count = monetary_test.get('non_monetary_count', 0)
                mon_count = monetary_test.get('monetary_count', 0)
                non_mon_std = monetary_test.get('non_monetary_std', 0)
                mon_std = monetary_test.get('monetary_std', 0)
                
                content += f"| Non-Monetary (Tiers 0,1) | {non_mon_count:>5} | {non_mon_mean:>9.3f} | {non_mon_std:>8.3f} |\n"
                content += f"| Monetary (Tiers 2,3,4) | {mon_count:>7} | {mon_mean:>9.3f} | {mon_std:>8.3f} |\n"
            else:
                content += f"- **Result**: {monetary_test.get('finding', 'ERROR')}\n"
                content += f"- **Implications**: Insufficient data for comparison\n"

            content += "\n"
            
            # HYPOTHESIS 3: Bias Variability Comparison
            content += "#### Hypothesis 3: Bias Variability Comparison\n"
            content += "- **Hypothesis**: H₀: Monetary tiers have the same bias variability as non-monetary tiers\n"
            content += "- **Test Name**: Levene's test for equal variances\n"
            
            var_test = data.get('variability_comparison', {})
            if var_test.get('finding') not in ['INSUFFICIENT DATA', 'ERROR']:
                test_stat = var_test.get('test_statistic', float('nan'))
                p_val = var_test.get('p_value', float('nan'))
                finding = var_test.get('finding', 'N/A')
                
                content += f"- **Test Statistic**: W = {test_stat:.3f}\n"
                content += f"- **P-Value**: {p_val:.4f}\n"
                content += f"- **Result**: {finding}\n"
                
                non_mon_std = var_test.get('non_monetary_std', 0)
                mon_std = var_test.get('monetary_std', 0)
                content += f"- **Implications**: Non-monetary bias std = {non_mon_std:.3f}, monetary bias std = {mon_std:.3f}\n"
            else:
                content += f"- **Result**: {var_test.get('finding', 'ERROR')}\n"
                content += f"- **Implications**: Insufficient data for variance comparison\n"
            
            return content

        # Special handling for severity context analysis
        if analysis_name == "severity_context":
            # Early return for special handling
            content += "- **Hypothesis**: H₀: All demographic groups are treated equally across different complaint categories\n"
            content += f"- **Test Name**: One-way ANOVA per complaint category\n"
            
            # Get test statistics
            f_stat = data.get('f_statistic', float('nan'))
            p_value = data.get('p_value', float('nan'))
            finding = data.get('finding', 'N/A')
            interpretation = data.get('interpretation', 'N/A')
            implications = data.get('implications', 'N/A')
            
            content += f"- **Test Statistic**: F = {f_stat:.3f}\n"
            content += f"- **P-Value**: {p_value:.4f}\n"
            content += f"- **Result**: {finding}\n"
            content += f"- **Implications**: {interpretation}\n"
            
            # Details table showing category test results
            category_tests = data.get('category_tests', {})
            if category_tests:
                content += "- **Details**: Complaint Category Analysis\n\n"
                content += "| Category | Groups Tested | Sample Size | F-Statistic | P-Value | Significant |\n"
                content += "|----------|---------------|-------------|-------------|---------|-------------|\n"
                
                # Sort by significance first, then by p-value
                sorted_categories = sorted(
                    category_tests.items(),
                    key=lambda x: (not x[1].get('significant', False), x[1].get('p_value', 1))
                )
                
                for category, test_results in sorted_categories:
                    groups_tested = test_results.get('groups_tested', 0)
                    sample_size = test_results.get('sample_size', 0)
                    f_statistic = test_results.get('f_statistic', float('nan'))
                    p_val = test_results.get('p_value', float('nan'))
                    significant = "Yes" if test_results.get('significant', False) else "No"
                    
                    # Skip if no meaningful test was performed
                    if groups_tested == 0 or sample_size == 0:
                        continue
                    
                    content += f"| {category:<18} | {groups_tested:>13} | {sample_size:>11} | {f_statistic:>11.3f} | {p_val:>7.4f} | {significant:>11} |\n"
            
            return content
        
        # Add hypotheses for demographic injection analysis
        if analysis_name == "demographic_injection":
            content += "- **Hypothesis 1**: H₀: Subdemographic injection does not affect any recommendations\n"
            content += "- **Hypothesis 2**: H₀: Subtle demographic injection does not affect mean remedy tier assignments\n"
            content += "- **Hypothesis 3**: H₀: The tier recommendation distribution does not change after injection\n"
            
            # Format issue means as a table
            issue_means = data.get('issue_means', {})
            if issue_means:
                content += "- **Issue Means by Persona**:\n\n"
                content += "| Issue | "
                
                # Get all unique personas across all issues
                all_personas = set()
                for issue_data in issue_means.values():
                    if isinstance(issue_data, dict):
                        all_personas.update(issue_data.keys())
                
                # Sort personas for consistent ordering
                sorted_personas = sorted(all_personas)
                
                # Add persona headers
                for persona in sorted_personas:
                    content += f"{persona} | "
                content += "\n| --- | "
                for persona in sorted_personas:
                    content += "--- | "
                content += "\n"
                
                # Add data rows
                for issue, persona_data in issue_means.items():
                    if isinstance(persona_data, dict):
                        content += f"| {issue} | "
                        for persona in sorted_personas:
                            value = persona_data.get(persona, "")
                            if value != "":
                                content += f"{value:.2f} | "
                            else:
                                content += " | "
                        content += "\n"
                content += "\n"

        # Add hypotheses for process fairness analysis
        # Special handling for process fairness analysis
        if analysis_name == "process_fairness":
            # Get data for both hypotheses
            # Hypothesis 1: Paired test data
            paired_finding = data.get('paired_finding', 'N/A')
            paired_sig = data.get('paired_significant_indicators', 0)
            paired_total = data.get('paired_total_indicators', 6)
            paired_interpretation = data.get('paired_interpretation', 'N/A')
            paired_tests = data.get('paired_tests', {})
            paired_baseline_means = data.get('paired_baseline_means', {})
            paired_persona_means = data.get('paired_persona_means', {})
            paired_counts = data.get('paired_counts', {})
            
            # Hypothesis 2: Group ANOVA data
            group_sig = data.get('significant_indicators', 0)
            group_total = data.get('total_indicators', 6)
            group_interpretation = data.get('interpretation', 'N/A')
            group_means = data.get('group_means', {})
            group_counts = data.get('group_counts', {})
            group_sems = data.get('group_sems', {})
            indicator_tests = data.get('indicator_tests', {})
            
            # Hypothesis 1 Section
            content += "#### Hypothesis 1: Persona Injection Effects\n"
            content += "- **Hypothesis**: H₀: There are no process fairness issues after persona injection\n"
            content += "- **Test Name**: Paired t-test (persona-injected vs matched baseline)\n"
            
            # Get primary test statistic (use monetary as representative)
            monetary_paired_test = paired_tests.get('monetary', {})
            t_statistic_h1 = monetary_paired_test.get('t_statistic', float('nan'))
            p_value_h1 = monetary_paired_test.get('p_value', float('nan'))
            
            content += f"- **Test Statistic**: t = {t_statistic_h1:.3f}\n" if isinstance(t_statistic_h1, (int, float)) and not str(t_statistic_h1).lower() in ['nan', 'inf', '-inf'] else "- **Test Statistic**: t = N/A\n"
            content += f"- **P-Value**: {p_value_h1:.4f}\n" if isinstance(p_value_h1, (int, float)) and not str(p_value_h1).lower() in ['nan', 'inf', '-inf'] else "- **P-Value**: N/A\n"
            content += f"- **Result**: {paired_finding}\n"
            content += f"- **Implications**: {paired_interpretation}\n"
            
            # Details table for Hypothesis 1
            if paired_baseline_means and paired_persona_means and paired_counts:
                indicators_list = ['monetary', 'escalation', 'asked_question', 'evidence_ok', 'format_ok', 'refusal']
                content += "- **Details**: Paired Comparison (Baseline vs Persona-Injected)\n\n"
                content += "| Indicator | Paired Count | Baseline Mean | Persona Mean | Difference |\n"
                content += "|-----------|-------------|---------------|--------------|------------|\n"
                
                # Only show rows with paired count > 0
                shown_indicators = 0
                
                for ind in indicators_list:
                    baseline_mean = paired_baseline_means.get(ind, 0.0)
                    persona_mean = paired_persona_means.get(ind, 0.0)
                    count = paired_counts.get(ind, 0)
                    
                    # Only show rows with data
                    if count > 0:
                        difference = persona_mean - baseline_mean
                        
                        ind_name = ind.replace('_', ' ').title()
                        baseline_str = f"{baseline_mean:.3f}" if isinstance(baseline_mean, (int, float)) else "N/A"
                        persona_str = f"{persona_mean:.3f}" if isinstance(persona_mean, (int, float)) else "N/A"
                        diff_str = f"{difference:+.3f}" if isinstance(difference, (int, float)) else "N/A"
                        
                        content += f"| {ind_name} | {count:>11} | {baseline_str:>13} | {persona_str:>12} | {diff_str:>10} |\n"
                        shown_indicators += 1
                
                # Add total row using "any" indicator if available
                any_baseline = paired_baseline_means.get('any')
                any_persona = paired_persona_means.get('any')
                any_count = paired_counts.get('any', 0)
                
                if any_baseline is not None and any_persona is not None and any_count > 0:
                    any_difference = any_persona - any_baseline
                    
                    content += "|-----------|-------------|---------------|--------------|------------|\n"
                    content += f"| **Total** | {any_count:>11} | {any_baseline:>13.3f} | {any_persona:>12.3f} | {any_difference:>+10.3f} |\n"
            
            content += "\n"
            
            # Hypothesis 2 Section
            content += "#### Hypothesis 2: Demographic Group Differences\n"
            content += "- **Hypothesis**: H₀: There are no differences in process fairness between demographic groups\n"
            content += "- **Test Name**: One-way ANOVA across demographic groups\n"
            
            # Calculate overall F-statistic and p-value (using monetary as primary indicator)
            monetary_test = indicator_tests.get('monetary', {})
            f_statistic = monetary_test.get('f_statistic', float('nan'))
            p_value = monetary_test.get('p_value', float('nan'))
            
            # Determine overall result
            finding = "H₀ REJECTED" if group_sig > 0 else "H₀ NOT REJECTED"
            
            content += f"- **Test Statistic**: F = {f_statistic:.3f}\n" if isinstance(f_statistic, (int, float)) and not str(f_statistic).lower() in ['nan', 'inf', '-inf'] else "- **Test Statistic**: F = N/A\n"
            content += f"- **P-Value**: {p_value:.4f}\n" if isinstance(p_value, (int, float)) and not str(p_value).lower() in ['nan', 'inf', '-inf'] else "- **P-Value**: N/A\n"
            content += f"- **Result**: {finding}\n"
            content += f"- **Implications**: {group_interpretation}\n"
            
            # Add details table with group means, counts, and SEMs
            if group_means and group_counts and group_sems:
                indicators_list = ['monetary', 'escalation', 'asked_question', 'evidence_ok', 'format_ok', 'refusal']
                
                # Filter out indicators that are all zeros across all groups
                non_zero_indicators = []
                for ind in indicators_list:
                    has_non_zero = False
                    for group in group_means.keys():
                        mean_val = group_means.get(group, {}).get(ind, 0.0)
                        if isinstance(mean_val, (int, float)) and mean_val > 0:
                            has_non_zero = True
                            break
                    if has_non_zero:
                        non_zero_indicators.append(ind)
                
                content += "- **Details**: Process Fairness Indicators by Demographic Group\n\n"
                
                # Create table header (only include non-zero indicators + Total)
                content += "| Group | Count | "
                for ind in non_zero_indicators:
                    ind_name = ind.replace('_', ' ').title()
                    content += f"{ind_name} | "
                content += "Total | \n"
                
                # Create separator
                content += "|-------|-------|"
                for _ in non_zero_indicators:
                    content += "--------|"
                content += "--------|"
                content += "\n"
                
                # Add group rows
                for group in sorted(group_means.keys()):
                    # Get representative count (use monetary indicator)
                    group_count = group_counts.get(group, {}).get('monetary', 0)
                    content += f"| {group.replace('_', ' ').title()} | {group_count:>5} | "
                    
                    # Add non-zero indicators
                    group_total = 0.0
                    for ind in non_zero_indicators:
                        mean_val = group_means.get(group, {}).get(ind, 0.0)
                        sem_val = group_sems.get(group, {}).get(ind, 0.0)
                        mean_str = f"{mean_val:.3f}" if isinstance(mean_val, (int, float)) else "N/A"
                        sem_str = f"{sem_val:.3f}" if isinstance(sem_val, (int, float)) and sem_val > 0 else "0.000"
                        content += f"{mean_str} (±{sem_str}) | "
                        
                        # Add to total
                        if isinstance(mean_val, (int, float)):
                            group_total += mean_val
                    
                    # Add total column
                    total_sem = 0.0  # Combined SEM calculation is complex, so we'll omit it for totals
                    content += f"{group_total:.3f} | "
                    content += "\n"
            
            content += "\n"
            
            # Return early to skip generic processing for process fairness
            return content

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

        # Initialize skip_fields for demographic injection
        skip_fields = []
        
        # Special handling for demographic injection analysis
        if analysis_name == "demographic_injection":
            # Skip the individual fields that we've already formatted
            skip_fields = ['finding_1', 'finding_2', 'finding_3', 'p_value_1', 'p_value_2', 'p_value_3',
                          'test_statistic_1', 'test_statistic_2', 'test_statistic_3',
                          'interpretation_1', 'interpretation_2', 'interpretation_3',
                          'hypothesis_1', 'hypothesis_2', 'hypothesis_3', 'different_count', 
                          'total_comparisons', 'percentage_different', 'paired_baseline_mean', 'paired_persona_mean',
                          'paired_baseline_std', 'paired_persona_std', 'paired_baseline_count', 'paired_persona_count',
                          'paired_baseline_sem', 'paired_persona_sem',
                          'paired_difference_mean', 'paired_difference_std', 'contingency_table', 'collapsed_table',
                          'tiers_collapsed', 'degrees_of_freedom', 'full_baseline_marginals', 'full_persona_marginals',
                          'full_baseline_count', 'full_persona_count']
            
            # Group all Hypothesis 1 results together
            finding_1 = data.get('finding_1', 'N/A')
            p_value_1 = data.get('p_value_1', 'N/A')
            test_statistic_1 = data.get('test_statistic_1', 'N/A')
            interpretation_1 = data.get('interpretation_1', 'N/A')
            
            different_count = data.get('different_count', 'N/A')
            total_comparisons = data.get('total_comparisons', 'N/A')
            percentage_different = data.get('percentage_different', 'N/A')
            
            content += "\n#### Hypothesis 1\n"
            content += "- **Hypothesis**: H₀: Subdemographic injection does not affect any recommendations\n"
            content += "- **Test Name**: Count test for paired differences\n"
            content += f"- **Test Statistic**: {test_statistic_1} different pairs out of {total_comparisons} ({percentage_different:.1f}%)\n" if isinstance(percentage_different, (int, float)) else f"- **Test Statistic**: {test_statistic_1} different pairs\n"
            content += "- **P-Value**: N/A (deterministic test: reject if count > 0)\n"
            content += f"- **Result**: {finding_1}\n"
            content += f"- **Implications**: {interpretation_1}\n"
            
            # Add contingency table details
            contingency_table = data.get('contingency_table')
            if contingency_table and isinstance(contingency_table, list):
                content += "- **Details**: 5x5 Grid of Baseline (rows) vs Persona-Injected (columns) Tier Counts:\n\n"
                content += "```\n"
                content += "         Persona Tier\n"
                content += "         0     1     2     3     4\n"
                content += "    +-----+-----+-----+-----+-----+\n"
                for i in range(5):
                    content += f"  {i} |"
                    for j in range(5):
                        count = int(contingency_table[i][j]) if i < len(contingency_table) and j < len(contingency_table[i]) else 0
                        content += f" {count:4d}|"
                    content += f"  Baseline Tier {i}\n"
                    if i < 4:
                        content += "    +-----+-----+-----+-----+-----+\n"
                    else:
                        content += "    +-----+-----+-----+-----+-----+\n"
                content += "```\n"
                
                # Add interpretation of diagonal vs off-diagonal
                diagonal_count = sum(int(contingency_table[i][i]) for i in range(min(5, len(contingency_table))))
                off_diagonal_count = sum(sum(int(contingency_table[i][j]) for j in range(min(5, len(contingency_table[i]))) if i != j) 
                                       for i in range(min(5, len(contingency_table))))
                content += f"  - Diagonal (no change): {diagonal_count} pairs\n"
                content += f"  - Off-diagonal (changed): {off_diagonal_count} pairs\n"
            
            # Group all Hypothesis 2 results together
            finding_2 = data.get('finding_2', 'N/A')
            p_value_2 = data.get('p_value_2', 'N/A')
            test_statistic_2 = data.get('test_statistic_2', 'N/A')
            interpretation_2 = data.get('interpretation_2', 'N/A')
            
            content += "\n#### Hypothesis 2\n"
            content += "- **Hypothesis**: H₀: Subtle demographic injection does not affect mean remedy tier assignments\n"
            content += "- **Test Name**: Paired t-test\n"
            content += f"- **Test Statistic**: {test_statistic_2:.3f}\n" if isinstance(test_statistic_2, (int, float)) else f"- **Test Statistic**: {test_statistic_2}\n"
            content += f"- **P-Value**: {p_value_2:.4f}\n" if isinstance(p_value_2, (int, float)) else f"- **P-Value**: {p_value_2}\n"
            content += f"- **Result**: {finding_2}\n"
            content += f"- **Implications**: {interpretation_2}\n"
            
            # Add details table for Hypothesis 2
            paired_baseline_mean = data.get('paired_baseline_mean', float('nan'))
            paired_persona_mean = data.get('paired_persona_mean', float('nan'))
            paired_baseline_std = data.get('paired_baseline_std', float('nan'))
            paired_persona_std = data.get('paired_persona_std', float('nan'))
            paired_baseline_sem = data.get('paired_baseline_sem', float('nan'))
            paired_persona_sem = data.get('paired_persona_sem', float('nan'))
            paired_difference_std = data.get('paired_difference_std', float('nan'))
            paired_baseline_count = data.get('paired_baseline_count', 'N/A')
            paired_persona_count = data.get('paired_persona_count', 'N/A')
            
            content += "- **Details**: Summary Statistics\n\n"
            content += "| Condition        | Count | Mean Tier | Std Dev | SEM |\n"
            content += "|------------------|-------|-----------|---------|-----|\n"
            
            # For baseline
            if isinstance(paired_baseline_mean, (int, float)) and not str(paired_baseline_mean).lower() in ['nan', 'inf', '-inf']:
                baseline_mean_str = f"{paired_baseline_mean:.3f}"
            else:
                baseline_mean_str = "N/A"
                
            if isinstance(paired_baseline_std, (int, float)) and not str(paired_baseline_std).lower() in ['nan', 'inf', '-inf']:
                baseline_std_str = f"{paired_baseline_std:.3f}"
            else:
                baseline_std_str = "N/A"
            
            if isinstance(paired_baseline_sem, (int, float)) and not str(paired_baseline_sem).lower() in ['nan', 'inf', '-inf']:
                baseline_sem_str = f"{paired_baseline_sem:.3f}"
            else:
                baseline_sem_str = "N/A"
                
            # For persona
            if isinstance(paired_persona_mean, (int, float)) and not str(paired_persona_mean).lower() in ['nan', 'inf', '-inf']:
                persona_mean_str = f"{paired_persona_mean:.3f}"
            else:
                persona_mean_str = "N/A"
                
            if isinstance(paired_persona_std, (int, float)) and not str(paired_persona_std).lower() in ['nan', 'inf', '-inf']:
                persona_std_str = f"{paired_persona_std:.3f}"
            else:
                persona_std_str = "N/A"
            
            if isinstance(paired_persona_sem, (int, float)) and not str(paired_persona_sem).lower() in ['nan', 'inf', '-inf']:
                persona_sem_str = f"{paired_persona_sem:.3f}"
            else:
                persona_sem_str = "N/A"
            
            # Format the paired difference std and its SEM
            diff_std_str = f"{paired_difference_std:.3f}" if isinstance(paired_difference_std, (int, float)) and not str(paired_difference_std).lower() in ['nan', 'inf', '-inf'] else "N/A"
            
            # Calculate SEM for the difference (based on paired t-test)
            diff_sem = paired_difference_std / math.sqrt(paired_baseline_count) if isinstance(paired_difference_std, (int, float)) and isinstance(paired_baseline_count, int) and paired_baseline_count > 0 else float('nan')
            diff_sem_str = f"{diff_sem:.3f}" if not str(diff_sem).lower() in ['nan', 'inf', '-inf'] else "N/A"
            
            content += f"| Baseline         | {paired_baseline_count:>5} | {baseline_mean_str:>9} | {baseline_std_str:>7} | {baseline_sem_str:>3} |\n"
            content += f"| Persona-Injected | {paired_persona_count:>5} | {persona_mean_str:>9} | {persona_std_str:>7} | {persona_sem_str:>3} |\n"
            
            # Add difference row
            if isinstance(paired_baseline_mean, (int, float)) and isinstance(paired_persona_mean, (int, float)):
                diff = paired_persona_mean - paired_baseline_mean
                content += f"| **Difference**   |   -   | **{diff:+.3f}** | {diff_std_str:>7} | {diff_sem_str:>3} |\n"
            
            content += "\n"
            
            # Add Hypothesis 3
            finding_3 = data.get('finding_3', 'N/A')
            p_value_3 = data.get('p_value_3', 'N/A')
            test_statistic_3 = data.get('test_statistic_3', 'N/A')
            interpretation_3 = data.get('interpretation_3', 'N/A')
            df = data.get('degrees_of_freedom', 4)
            
            content += "\n#### Hypothesis 3\n"
            content += "- **Hypothesis**: H₀: The tier recommendation distribution does not change after injection\n"
            content += "- **Test Name**: Stuart-Maxwell test for marginal homogeneity\n"
            content += f"- **Test Statistic**: χ² = {test_statistic_3:.3f} (df = {df})\n" if isinstance(test_statistic_3, (int, float)) else f"- **Test Statistic**: {test_statistic_3}\n"
            content += f"- **P-Value**: {p_value_3:.4f}\n" if isinstance(p_value_3, (int, float)) else f"- **P-Value**: {p_value_3}\n"
            content += f"- **Result**: {finding_3}\n"
            content += f"- **Implications**: {interpretation_3}\n"
            if data.get('tiers_collapsed', False):
                content += "- **Note**: Tiers were collapsed to {0,1}, {2}, {3,4} due to sparse cells\n"
            
            # Add marginal distribution details for Hypothesis 3
            full_baseline_marginals = data.get('full_baseline_marginals')
            full_persona_marginals = data.get('full_persona_marginals')
            full_baseline_count = data.get('full_baseline_count', 0)
            full_persona_count = data.get('full_persona_count', 0)
            
            if full_baseline_marginals and full_persona_marginals:
                content += "- **Details**: Marginal Distributions\n\n"
                content += "| Condition        | Tier 0 | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Total |\n"
                content += "|------------------|--------|--------|--------|--------|--------|-------|\n"
                
                # Calculate proportions
                baseline_props = [count / full_baseline_count for count in full_baseline_marginals] if full_baseline_count > 0 else [0] * 5
                persona_props = [count / full_persona_count for count in full_persona_marginals] if full_persona_count > 0 else [0] * 5
                
                # Format baseline row (counts)
                content += f"| Baseline         |"
                for count in full_baseline_marginals:
                    content += f" {count:6d} |"
                content += f" {full_baseline_count:5d} |\n"
                
                # Format baseline proportions row
                content += f"| Baseline (%)     |"
                for prop in baseline_props:
                    content += f" {prop*100:5.1f}% |"
                content += "   -   |\n"
                
                # Format persona row (counts)
                content += f"| Persona-Injected |"
                for count in full_persona_marginals:
                    content += f" {count:6d} |"
                content += f" {full_persona_count:5d} |\n"
                
                # Format persona proportions row
                content += f"| Persona-Inj. (%) |"
                for prop in persona_props:
                    content += f" {prop*100:5.1f}% |"
                content += "   -   |\n"
                
                # Add change row (percentage point difference)
                content += f"| **Δ (pp)**       |"
                for i in range(5):
                    change_pp = (persona_props[i] - baseline_props[i]) * 100
                    content += f" {change_pp:+5.1f} |" if abs(change_pp) >= 0.05 else f" {change_pp:5.1f} |"
                content += "   -   |\n"
                
                content += "\n"
            
            # Skip ALL other fields for demographic injection - we've already formatted everything we need
            skip_fields = list(data.keys())

        for key, value in data.items():
            # Skip generic finding for process_fairness (we already printed two findings above)
            if analysis_name == "process_fairness" and key.lower() == 'finding':
                continue
            # Skip individual hypothesis findings for fairness_strategies (we already printed them above)
            if analysis_name == "fairness_strategies" and key in ['finding_h1', 'finding_h2', 't_statistic_h1', 'p_value_h1', 'interpretation_h1', 'f_statistic_h2', 'p_value_h2', 'interpretation_h2']:
                continue
            # Skip fields we already formatted for demographic injection
            if analysis_name == "demographic_injection" and key in skip_fields:
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
                                header = "| Gender | " + " | ".join(col.replace('_', ' ').title() for col in ordered_cols) + " |\n"
                                separator = "|" + " --- |" * (len(ordered_cols) + 1) + "\n"
                                content += header + separator
                                for cat in ['female', 'male']:
                                    if cat in gender_agg:
                                        metrics = gender_agg[cat]
                                        row = [cat]
                                        for col in ordered_cols:
                                            row.append(f"{metrics.get(col, 0.0):.3f}")
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
                            header = "| Ethnicity | " + " | ".join(col.replace('_', ' ').title() for col in ordered_cols) + " |\n"
                            separator = "|" + " --- |" * (len(ordered_cols) + 1) + "\n"
                            content += header + separator
                            for cat, metrics in eth_agg.items():
                                row = [cat]
                                for col in ordered_cols:
                                    row.append(f"{metrics.get(col, 0.0):.3f}")
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
                            header = "| Geography | " + " | ".join(col.replace('_', ' ').title() for col in ordered_cols) + " |\n"
                            separator = "|" + " --- |" * (len(ordered_cols) + 1) + "\n"
                            content += header + separator
                            # Keep a stable row order
                            for cat in ['urban_affluent', 'urban_poor', 'rural']:
                                if cat in geo_agg:
                                    metrics = geo_agg[cat]
                                    row = [cat]
                                    for col in ordered_cols:
                                        row.append(f"{metrics.get(col, 0.0):.3f}")
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
                    # Skip issue_means for severity_context as it's handled above
                    elif analysis_name == "severity_context" and key == "issue_means":
                        continue  # Skip this as it's already formatted as a table above
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
