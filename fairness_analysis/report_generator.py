"""
Report Generator for Advanced Fairness Analysis

This module provides functionality to generate comprehensive reports
from fairness analysis results.
"""

import json
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
                content += self._format_analysis_data(analysis_data, analysis_name)
            else:
                content += f"Analysis completed: {type(analysis_data).__name__}\n\n"
                
        return content
        
    def _format_analysis_data(self, data: Dict[str, Any], analysis_name: str = "") -> str:
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
        
        # Define remedy tier labels
        tier_labels = {
            0: "No action taken",
            1: "Process improvement", 
            2: "Small monetary remedy",
            3: "Moderate monetary remedy",
            4: "High monetary remedy"
        }
        
        for key, value in data.items():
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
                    content += f"- **{key.replace('_', ' ').title()}**:\n"
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            tier_num = round(sub_value)
                            tier_label = tier_labels.get(tier_num, f"Tier {tier_num}")
                            content += f"  - {sub_key}: {sub_value:.3f} ({tier_label})\n"
                        else:
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
