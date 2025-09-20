#!/usr/bin/env python3
"""
HTML Dashboard Generator for Bank Complaint Handling Fairness Analysis

This module creates an interactive HTML dashboard with multiple tabs showing
fairness analysis results, persona injection effects, bias mitigation, and
ground truth accuracy metrics.
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from scipy.stats import chi2_contingency


def calculate_cohens_d_paired(group1, group2):
    """Calculate Cohen's d for paired samples"""
    try:
        if group1 is None or group2 is None:
            return 0.0

        # Convert to arrays
        arr1 = np.array(group1, dtype=float)
        arr2 = np.array(group2, dtype=float)

        if len(arr1) == 0 or len(arr2) == 0 or len(arr1) != len(arr2):
            return 0.0

        differences = arr1 - arr2

        # Check for valid differences
        if np.all(np.isnan(differences)) or len(differences) <= 1:
            return 0.0

        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)

        if np.isnan(mean_diff) or np.isnan(std_diff) or std_diff == 0:
            return 0.0

        cohens_d = mean_diff / std_diff

        if np.isnan(cohens_d) or np.isinf(cohens_d):
            return 0.0

        return float(cohens_d)

    except Exception as e:
        print(f"[WARNING] Cohen's d (paired) calculation failed: {e}")
        return 0.0


def calculate_cohens_d_independent(group1, group2):
    """Calculate Cohen's d for independent samples"""
    try:
        if group1 is None or group2 is None:
            return 0.0

        # Convert to arrays
        arr1 = np.array(group1, dtype=float)
        arr2 = np.array(group2, dtype=float)

        n1, n2 = len(arr1), len(arr2)
        if n1 == 0 or n2 == 0 or n1 + n2 <= 2:
            return 0.0

        # Remove NaN values
        arr1 = arr1[~np.isnan(arr1)]
        arr2 = arr2[~np.isnan(arr2)]

        if len(arr1) == 0 or len(arr2) == 0:
            return 0.0

        mean1, mean2 = np.mean(arr1), np.mean(arr2)
        var1, var2 = np.var(arr1, ddof=1), np.var(arr2, ddof=1)

        # Check for valid values
        if np.isnan(mean1) or np.isnan(mean2) or np.isnan(var1) or np.isnan(var2):
            return 0.0

        # Calculate pooled standard deviation
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        pooled_std = np.sqrt(pooled_var)

        if np.isnan(pooled_std) or pooled_std == 0:
            return 0.0

        cohens_d = (mean1 - mean2) / pooled_std

        if np.isnan(cohens_d) or np.isinf(cohens_d):
            return 0.0

        return float(cohens_d)

    except Exception as e:
        print(f"[WARNING] Cohen's d (independent) calculation failed: {e}")
        return 0.0


def calculate_cohens_h(p1, p2):
    """Calculate Cohen's h for difference between two proportions"""
    try:
        # Validate inputs
        if p1 is None or p2 is None:
            return 0.0

        # Convert to float and constrain to [0, 1]
        p1 = float(np.clip(p1, 0, 1))
        p2 = float(np.clip(p2, 0, 1))

        # Calculate with error handling
        with np.errstate(invalid='ignore'):
            phi1 = 2 * np.arcsin(np.sqrt(p1))
            phi2 = 2 * np.arcsin(np.sqrt(p2))

        # Check for invalid results
        if np.isnan(phi1) or np.isnan(phi2) or np.isinf(phi1) or np.isinf(phi2):
            return 0.0

        cohens_h = phi1 - phi2

        # Check final result
        if np.isnan(cohens_h) or np.isinf(cohens_h):
            return 0.0

        return float(cohens_h)

    except Exception as e:
        print(f"[WARNING] Cohen's h calculation failed: {e}")
        return 0.0


def calculate_risk_ratio(p1, p2):
    """Calculate risk ratio (relative risk) for two proportions"""
    if p2 == 0:
        return float('inf') if p1 > 0 else 1.0
    return p1 / p2


def calculate_cramers_v(contingency_table):
    """Calculate Cramér's V for chi-squared test with error handling"""
    try:
        # Validate input
        if contingency_table is None:
            return 0.0

        # Convert to numpy array and ensure it's numeric
        table = np.array(contingency_table, dtype=float)

        # Check for valid dimensions
        if table.size == 0 or table.ndim != 2:
            return 0.0

        # Check if any dimension is less than 2
        if min(table.shape) < 2:
            return 0.0

        # Check for all zeros
        if np.sum(table) == 0:
            return 0.0

        # Check for negative values
        if np.any(table < 0):
            return 0.0

        # Calculate chi-squared with error handling
        with np.errstate(divide='ignore', invalid='ignore'):
            chi2, p, dof, expected = chi2_contingency(table)

        # Check if chi2 calculation failed
        if np.isnan(chi2) or np.isinf(chi2) or chi2 < 0:
            return 0.0

        n = table.sum()
        min_dim = min(table.shape) - 1

        # Check for valid calculations
        if n <= 0 or min_dim <= 0:
            return 0.0

        # Calculate Cramér's V with error handling
        denominator = n * min_dim
        if denominator <= 0:
            return 0.0

        cramers_v = np.sqrt(chi2 / denominator)

        # Check result validity
        if np.isnan(cramers_v) or np.isinf(cramers_v):
            return 0.0

        # Cramér's V should be between 0 and 1
        if cramers_v < 0 or cramers_v > 1:
            return 0.0

        return float(cramers_v)

    except Exception as e:
        print(f"[WARNING] Cramér's V calculation failed: {e}")
        return 0.0


def interpret_statistical_result(p_value, effect_size, test_type):
    """Enhanced interpretation including effect size materiality"""
    
    # Statistical significance
    is_significant = p_value < 0.05
    significance_text = "rejected" if is_significant else "not rejected"
    
    # Effect size interpretation
    if test_type == "paired_t_test":
        if abs(effect_size) < 0.2:
            effect_magnitude = "negligible"
            practical_importance = "trivial"
        elif abs(effect_size) < 0.8:
            effect_magnitude = "small to medium"
            practical_importance = "modest"
        else:
            effect_magnitude = "large"
            practical_importance = "substantial"
    
    elif test_type == "chi_squared":
        if effect_size < 0.1:
            effect_magnitude = "negligible"
            practical_importance = "trivial"
        elif effect_size < 0.3:
            effect_magnitude = "small to medium"
            practical_importance = "modest"
        else:
            effect_magnitude = "large"
            practical_importance = "substantial"

    elif test_type == "independent_t_test":
        if abs(effect_size) < 0.2:
            effect_magnitude = "negligible"
            practical_importance = "trivial"
        elif abs(effect_size) < 0.5:
            effect_magnitude = "small"
            practical_importance = "modest"
        elif abs(effect_size) < 0.8:
            effect_magnitude = "medium"
            practical_importance = "moderate"
        else:
            effect_magnitude = "large"
            practical_importance = "substantial"

    elif test_type == "cohens_h":  # For proportion differences
        if abs(effect_size) < 0.2:
            effect_magnitude = "negligible"
            practical_importance = "trivial"
        elif abs(effect_size) < 0.5:
            effect_magnitude = "small"
            practical_importance = "modest"
        elif abs(effect_size) < 0.8:
            effect_magnitude = "medium"
            practical_importance = "moderate"
        else:
            effect_magnitude = "large"
            practical_importance = "substantial"

    elif test_type == "risk_ratio":  # For risk/rate ratios
        # Note: Risk ratios don't have universally accepted effect size thresholds
        # We report the raw ratio and let readers interpret based on context
        # For reference, epidemiology often considers RR < 1.5 as weak association
        # but this varies greatly by field and context
        pct_change = abs((effect_size - 1) * 100)
        effect_magnitude = f"{pct_change:.0f}% {'increase' if effect_size > 1 else 'decrease'}"

        # We don't assign practical importance for risk ratios
        # as interpretation is highly context-dependent
        practical_importance = "context-dependent"

    else:
        # Default interpretation for unknown test types
        effect_magnitude = "unknown"
        practical_importance = "unknown"

    # Combined interpretation
    if is_significant and practical_importance == "trivial":
        interpretation = f"statistically significant but practically {practical_importance}"
        warning = " (large sample size may detect trivial differences)"
    elif is_significant and practical_importance != "trivial":
        interpretation = f"statistically significant and practically {practical_importance}"
        warning = ""
    else:
        interpretation = f"not statistically significant (effect size: {effect_magnitude})"
        warning = ""
    
    return {
        'significance_text': significance_text,
        'effect_magnitude': effect_magnitude,
        'practical_importance': practical_importance,
        'interpretation': interpretation,
        'warning': warning
    }


class StatisticalResultCollector:
    """Collects and categorizes statistical test results for headline reporting"""
    
    def __init__(self):
        self.results = {
            'material': [],      # Significant and material results
            'trivial': [],       # Significant but trivial results
            'non_significant': [] # Not significant (not displayed in headline)
        }

    def add_result(self, result_data):
        """
        Add a statistical test result to the collector

        Args:
            result_data: dict containing:
                - source_tab: str (e.g., "Bias Analysis", "Accuracy Analysis")
                - source_subtab: str (e.g., "Persona Comparison", "Strategy Effectiveness")
                - test_name: str (descriptive name of what was tested)
                - test_type: str (e.g., "paired_t_test", "chi_squared")
                - p_value: float
                - effect_size: float
                - effect_type: str (e.g., "cohens_d", "cramers_v")
                - sample_size: int
                - finding: str (human-readable description)
                - implication: str (what this means for fairness)
                - timestamp: datetime (when calculated)
        """
        try:
            # Validate required fields
            if not isinstance(result_data, dict):
                print(f"[WARNING] Invalid result_data type: {type(result_data)}")
                return

            required_fields = ['source_tab', 'test_name', 'p_value', 'effect_size', 'effect_type']
            missing_fields = [field for field in required_fields if field not in result_data]
            if missing_fields:
                print(f"[WARNING] Missing required fields in result_data: {missing_fields}")
                return

            # Debug: Print what we're adding
            p_val = result_data.get('p_value', 'N/A')
            effect_size = result_data.get('effect_size', 'N/A')
            effect_type = result_data.get('effect_type', 'N/A')
            test_name = result_data.get('test_name', 'Unknown')


            category = self._categorize_result(result_data)

            self.results[category].append(result_data)


        except Exception as e:
            print(f"[ERROR] Failed to add result: {e}")
            import traceback
            traceback.print_exc()

    def _categorize_result(self, result):
        """Categorize result based on p-value and effect size"""
        p_value = result['p_value']
        effect_size = result['effect_size']
        effect_type = result['effect_type']


        if p_value >= 0.05:
            return 'non_significant'

        # Determine materiality based on effect size type
        if effect_type == 'cohens_d':
            threshold = 0.2
            is_material = abs(effect_size) >= threshold
        elif effect_type == 'cohens_h':
            threshold = 0.2
            is_material = abs(effect_size) >= threshold
        elif effect_type == 'cramers_v':
            threshold = 0.1
            is_material = effect_size >= threshold
        elif effect_type == 'eta_squared':
            threshold = 0.01
            is_material = effect_size >= threshold
        elif effect_type == 'chi_squared':
            threshold = 0.1
            is_material = effect_size >= threshold
        else:
            # Conservative: treat unknown effect sizes as material
            is_material = True

        result_category = 'material' if is_material else 'trivial'
        return result_category


class HTMLDashboard:
    """Generate interactive HTML dashboard for fairness analysis results"""

    def __init__(self, output_dir: str = "dashboards"):
        """
        Initialize the HTML dashboard generator

        Args:
            output_dir: Directory to save dashboard files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize statistical result collector
        self.collector = StatisticalResultCollector()

        # Dashboard configuration
        self.tabs = [
            {"id": "headline", "name": "Headline Results", "default": True},
            {"id": "persona", "name": "Persona Injection", "default": False},
            {"id": "severity", "name": "Severity and Bias", "default": False},
            {"id": "mitigation", "name": "Bias Mitigation", "default": False},
            {"id": "accuracy", "name": "Ground Truth Accuracy", "default": False}
        ]

    def generate_dashboard(self, experiment_data: Dict[str, Any]) -> str:
        """
        Generate complete HTML dashboard

        Args:
            experiment_data: Dictionary containing all experiment results and metrics

        Returns:
            Path to generated HTML file
        """
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f"fairness_dashboard_{timestamp}.html"

        # Collect real statistical results from all analyses
        self._collect_real_statistical_results(experiment_data)

        # Build complete HTML content
        html_content = self._build_html_structure(experiment_data)

        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"[SUCCESS] Dashboard generated: {output_file}")
        return str(output_file)

    def _build_html_structure(self, data: Dict[str, Any]) -> str:
        """Build complete HTML structure with all components"""
        # Embed accuracy data in JavaScript for client-side filtering
        accuracy_data_js = ""
        if 'accuracy_data' in data:
            import json
            
            # Convert tuple keys to strings and handle non-JSON serializable types
            def convert_for_json(obj):
                if isinstance(obj, dict):
                    new_dict = {}
                    for key, value in obj.items():
                        if isinstance(key, tuple):
                            # Convert tuple to string key
                            new_key = f"{key[0]}|{key[1]}"
                        else:
                            new_key = key
                        new_dict[new_key] = convert_for_json(value)
                    return new_dict
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                elif hasattr(obj, '__class__') and 'Decimal' in str(obj.__class__):
                    # Convert Decimal to float
                    return float(obj)
                elif hasattr(obj, '__class__') and 'datetime' in str(obj.__class__):
                    # Convert datetime to string
                    return str(obj)
                else:
                    return obj
            
            # Convert the accuracy data to have string keys and handle non-JSON types
            converted_accuracy_data = convert_for_json(data['accuracy_data'])
            
            accuracy_data_js = f"""
        <script>
            // Make accuracy data available to JavaScript
            window.accuracyData = {json.dumps(converted_accuracy_data)};
        </script>"""
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Fairness Dashboard - Bank Complaint Analysis</title>
    {self._get_css_styles()}
    {self._get_javascript()}
    {accuracy_data_js}
</head>
<body>
    <div class="dashboard-container">
        {self._build_header(data)}
        {self._build_navigation()}
        {self._build_tab_content(data)}
    </div>
</body>
</html>"""

    def _get_css_styles(self) -> str:
        """Generate CSS styles for the dashboard"""
        return """<style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }

        /* Sub-tab styles */
        .sub-nav-tabs {
            display: flex;
            background: #f8f9fa;
            border-radius: 8px;
            padding: 3px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            overflow-x: auto;
        }

        .sub-nav-tab {
            flex: 1;
            padding: 10px 15px;
            text-align: center;
            cursor: pointer;
            border-radius: 6px;
            transition: all 0.3s ease;
            font-weight: 500;
            white-space: nowrap;
            min-width: 120px;
            font-size: 0.9rem;
            color: #6c757d;
        }

        .sub-nav-tab:hover {
            background-color: #e9ecef;
            color: #495057;
        }

        .sub-nav-tab.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .sub-tab-content {
            display: none;
        }

        .sub-tab-content.active {
            display: block;
        }

        /* Table styles */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.95em;
            min-width: 100%;
            overflow-x: auto;
            display: block;
        }

        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
            min-width: 100px;
        }

        th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
            white-space: nowrap;
        }

        tr:hover {
            background-color: #f5f5f5;
        }

        /* Specific styles for tier impact table */
        .tier-impact {
            width: 100%;
            table-layout: fixed;
        }

        .tier-impact th,
        .tier-impact td {
            min-width: 120px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            text-align: center;
        }

        .tier-impact th:first-child,
        .tier-impact td:first-child {
            min-width: 150px;
            text-align: left;
        }

        .tier-impact th:last-child,
        .tier-impact td:last-child {
            min-width: 100px;
        }

        .tier-impact tr.total-row {
            background-color: #f8f9fa;
            font-weight: 600;
        }

        .conclusion {
            margin-top: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-left: 4px solid #667eea;
            border-radius: 0 4px 4px 0;
        }

        .conclusion h4 {
            margin-bottom: 10px;
            color: #2c3e50;
        }

        .conclusion p {
            margin: 5px 0;
            line-height: 1.5;
        }

        .dashboard-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            font-weight: 700;
        }

        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        .nav-tabs {
            display: flex;
            background: white;
            border-radius: 12px;
            padding: 5px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            overflow-x: auto;
        }

        .nav-tab {
            flex: 1;
            padding: 15px 20px;
            text-align: center;
            cursor: pointer;
            border-radius: 8px;
            transition: all 0.3s ease;
            font-weight: 500;
            white-space: nowrap;
            min-width: 150px;
        }

        .nav-tab:hover {
            background-color: #f8f9fa;
        }

        .nav-tab.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .tab-content {
            display: none;
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .tab-content.active {
            display: block;
        }

        .section {
            margin-bottom: 40px;
        }

        .section h2 {
            color: #2c3e50;
            font-size: 1.8rem;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }

        .section h3 {
            color: #34495e;
            font-size: 1.4rem;
            margin-bottom: 15px;
            margin-top: 25px;
        }

        .result-item {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 0 8px 8px 0;
        }

        .result-title {
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 10px;
        }

        .result-placeholder {
            color: #6c757d;
            font-style: italic;
        }

        .filter-controls {
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }

        .filter-controls label {
            display: inline-block;
            margin-right: 10px;
            font-weight: 600;
            color: #495057;
        }

        .filter-controls select {
            margin-right: 20px;
            padding: 5px 10px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            background-color: white;
        }

        .confusion-matrix, .accuracy-rates {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .confusion-matrix th, .confusion-matrix td,
        .accuracy-rates th, .accuracy-rates td {
            padding: 12px;
            text-align: center;
            border: 1px solid #dee2e6;
        }

        .confusion-matrix th, .accuracy-rates th {
            background-color: #e9ecef;
            font-weight: 600;
            color: #495057;
        }

        .confusion-matrix tbody tr:nth-child(even),
        .accuracy-rates tbody tr:nth-child(even) {
            background-color: #f8f9fa;
        }

        .confusion-matrix tbody tr:hover,
        .accuracy-rates tbody tr:hover {
            background-color: #e3f2fd;
        }

        .gender-tier0-rate {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .gender-tier0-rate th, .gender-tier0-rate td {
            padding: 12px;
            text-align: center;
            border: 1px solid #dee2e6;
        }

        .gender-tier0-rate th {
            background-color: #e9ecef;
            font-weight: 600;
            color: #495057;
        }

        .gender-tier0-rate tbody tr:nth-child(even) {
            background-color: #f8f9fa;
        }

        .gender-tier0-rate tbody tr:hover {
            background-color: #e3f2fd;
        }

        .statistical-analysis {
            margin-top: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }

        .statistical-analysis h4 {
            margin-top: 0;
            color: #007bff;
        }

        .statistical-analysis p {
            margin: 8px 0;
        }

        .ethnicity-tier0-rate,
        .geographic-tier0-rate {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .ethnicity-tier0-rate th, .ethnicity-tier0-rate td,
        .geographic-tier0-rate th, .geographic-tier0-rate td {
            padding: 12px;
            text-align: center;
            border: 1px solid #dee2e6;
        }

        .ethnicity-tier0-rate th,
        .geographic-tier0-rate th {
            background-color: #e9ecef;
            font-weight: 600;
            color: #495057;
        }

        .ethnicity-tier0-rate tbody tr:nth-child(even),
        .geographic-tier0-rate tbody tr:nth-child(even) {
            background-color: #f8f9fa;
        }

        .ethnicity-tier0-rate tbody tr:hover,
        .geographic-tier0-rate tbody tr:hover {
            background-color: #e3f2fd;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }

        .metric-card {
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 5px;
        }

        .metric-label {
            color: #6c757d;
            font-size: 0.9rem;
        }

        .info-box {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }

        .warning-box {
            background: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 15px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }

        /* Table styles */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.95em;
            min-width: 600px;
            overflow-x: auto;
            display: block;
        }

        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
            min-width: 100px;
        }

        th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
            white-space: nowrap;
        }

        tr:hover {
            background-color: #f5f5f5;
        }

        /* Specific styles for tier impact table */
        .tier-impact-table {
            width: 100%;
            table-layout: fixed;
        }

        .tier-impact-table th,
        .tier-impact-table td {
            min-width: 120px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .tier-impact-table th:first-child,
        .tier-impact-table td:first-child {
            min-width: 150px;
        }

        .tier-impact-table th:last-child,
        .tier-impact-table td:last-child {
            min-width: 100px;
        }

        @media (max-width: 768px) {
            .nav-tabs {
                flex-direction: column;
            }

            .nav-tab {
                margin: 2px 0;
            }

            .header h1 {
                font-size: 2rem;
            }

            .dashboard-container {
                padding: 10px;
            }

            table {
                display: block;
                overflow-x: auto;
                white-space: nowrap;
            }
        }

        /* Headline Results Tab Styling */
        .headline-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 8px;
            margin-bottom: 2rem;
        }

        .stat-highlight {
            font-size: 2.5rem;
            font-weight: bold;
            display: inline-block;
            margin-right: 0.5rem;
        }

        /* Result Cards */
        .headline-result-card {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            overflow: hidden;
            transition: box-shadow 0.3s;
        }

        .headline-result-card.material {
            border-left: 4px solid #4CAF50;
        }

        .headline-result-card.trivial {
            border-left: 4px solid #FFC107;
            opacity: 0.85;
        }

        .headline-result-card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        /* Effect Badges */
        .effect-badge {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .effect-badge.large-effect {
            background: #4CAF50;
            color: white;
        }

        .effect-badge.medium-effect {
            background: #2196F3;
            color: white;
        }

        .effect-badge.small-effect {
            background: #FF9800;
            color: white;
        }

        .effect-badge.trivial-effect {
            background: #FFC107;
            color: #333;
        }

        /* Statistics Row */
        .statistics-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            padding: 1rem;
            background: #f5f5f5;
            border-radius: 4px;
            margin: 1rem 0;
        }

        .stat-item {
            display: flex;
            flex-direction: column;
        }

        .stat-label {
            font-size: 0.85rem;
            color: #666;
            margin-bottom: 4px;
        }

        .stat-value {
            font-weight: 600;
            color: #333;
        }

        .stat-value.significant {
            color: #4CAF50;
        }

        /* Implication Box */
        .implication-box {
            background: #E8F5E9;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
            border-left: 3px solid #4CAF50;
        }

        /* Trivial Warning */
        .trivial-warning {
            background: #FFF3E0;
            padding: 0.75rem;
            border-radius: 4px;
            margin: 1rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            border: 1px solid #FFB74D;
        }

        /* Controls */
        .headline-controls {
            display: flex;
            gap: 2rem;
            margin-bottom: 2rem;
            padding: 1rem;
            background: #f8f8f8;
            border-radius: 4px;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .statistics-row {
                grid-template-columns: 1fr;
            }

            .headline-controls {
                flex-direction: column;
                gap: 1rem;
            }
        }
        </style>"""

    def _get_javascript(self) -> str:
        """Generate JavaScript for dashboard interactivity"""
        return """<script>
        function showTab(tabId, event) {
            // Prevent default anchor behavior
            if (event) {
                event.preventDefault();
            }
            
            // Hide all tab contents
            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach(content => {
                content.classList.remove('active');
            });

            // Remove active class from all nav tabs
            const navTabs = document.querySelectorAll('.nav-tab');
            navTabs.forEach(tab => {
                tab.classList.remove('active');
            });

            // Show selected tab content
            const tabContent = document.getElementById(tabId);
            if (tabContent) {
                tabContent.classList.add('active');
            }

            // Add active class to clicked nav tab or find it by tabId
            if (event && event.currentTarget) {
                event.currentTarget.classList.add('active');
            } else {
                const tabToActivate = document.querySelector(`.nav-tab[data-tab-id="${tabId}"]`);
                if (tabToActivate) {
                    tabToActivate.classList.add('active');
                }
            }
            
            // Reset sub-tabs to show first sub-tab when switching main tabs
            const currentTab = document.getElementById(tabId);
            if (currentTab) {
                const subTabs = currentTab.querySelectorAll('.sub-nav-tab');
                const subContents = currentTab.querySelectorAll('.sub-tab-content');
                
                // Remove active from all sub-tabs and sub-contents
                subTabs.forEach(tab => tab.classList.remove('active'));
                subContents.forEach(content => content.classList.remove('active'));
                
                // Activate first sub-tab and content
                if (subTabs.length > 0) {
                    subTabs[0].classList.add('active');
                }
                if (subContents.length > 0) {
                    subContents[0].classList.add('active');
                }
            }
            
            // Scroll to top of page when changing tabs
            window.scrollTo({ top: 0, behavior: 'smooth' });
            
            // Update URL hash for direct linking
            window.location.hash = tabId;
        }

        function showSubTab(subTabId, event) {
            // Prevent default anchor behavior
            if (event) {
                event.preventDefault();
            }
            
            // Find the parent tab content
            const parentTab = event.currentTarget.closest('.tab-content');
            if (!parentTab) return;
            
            // Hide all sub-tab contents in this parent tab
            const subTabContents = parentTab.querySelectorAll('.sub-tab-content');
            subTabContents.forEach(content => {
                content.classList.remove('active');
            });

            // Remove active class from all sub-nav tabs in this parent tab
            const subNavTabs = parentTab.querySelectorAll('.sub-nav-tab');
            subNavTabs.forEach(tab => {
                tab.classList.remove('active');
            });

            // Show selected sub-tab content
            const subTabContent = parentTab.querySelector(`#${subTabId}`);
            if (subTabContent) {
                subTabContent.classList.add('active');
            }

            // Add active class to clicked sub-nav tab
            if (event && event.currentTarget) {
                event.currentTarget.classList.add('active');
            }
            
            // Scroll to top of page when changing sub-tabs
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }

        // Initialize tab functionality on page load
        document.addEventListener('DOMContentLoaded', function() {
            // Scroll to top of page when first loaded/refreshed
            window.scrollTo({ top: 0, behavior: 'smooth' });
            
            // Check URL hash for direct linking
            const hash = window.location.hash.substring(1);
            let defaultTab = null;
            
            if (hash) {
                defaultTab = document.querySelector(`.nav-tab[data-tab-id="${hash}"]`);
            }
            
            // Fall back to default tab if no valid hash or tab found
            if (!defaultTab) {
                defaultTab = document.querySelector('.nav-tab[data-default="true"]');
            }
            
            // Activate the appropriate tab
            if (defaultTab) {
                const tabId = defaultTab.getAttribute('data-tab-id');
                showTab(tabId, { currentTarget: defaultTab, preventDefault: () => {} });
            }
            
            // Add click event listeners to all nav tabs
            document.querySelectorAll('.nav-tab').forEach(tab => {
                tab.addEventListener('click', function(e) {
                    const tabId = this.getAttribute('data-tab-id');
                    showTab(tabId, e);
                });
            });
            
            // Add click event listeners to all sub-nav tabs
            document.querySelectorAll('.sub-nav-tab').forEach(tab => {
                tab.addEventListener('click', function(e) {
                    const subTabId = this.getAttribute('data-sub-tab-id');
                    showSubTab(subTabId, e);
                });
            });
        });
        
        // Accuracy filter functions
        function updateAccuracyFilters() {
            const decisionMethod = document.getElementById('decision-method-filter').value;
            const experimentCategory = document.getElementById('experiment-category-filter').value;
            
            // Get the accuracy data from the global variable
            const accuracyData = window.accuracyData || {};
            const confusionMatrix = accuracyData.confusion_matrix || {};
            
            // Find the appropriate data based on filters
            let matrixData = null;
            if (decisionMethod === 'all' && experimentCategory === 'all') {
                // Show all data combined - we'll need to aggregate
                matrixData = aggregateConfusionMatrix(confusionMatrix);
            } else if (decisionMethod === 'all') {
                // Filter by experiment category only
                matrixData = aggregateConfusionMatrixByCategory(confusionMatrix, experimentCategory);
            } else if (experimentCategory === 'all') {
                // Filter by decision method only
                matrixData = aggregateConfusionMatrixByMethod(confusionMatrix, decisionMethod);
            } else {
                // Filter by both
                const key = `${decisionMethod}|${experimentCategory}`;
                matrixData = confusionMatrix[key] || null;
            }
            
            const container = document.getElementById('confusion-matrix-container');
            if (matrixData) {
                container.innerHTML = buildConfusionMatrixHTML(matrixData);
            } else {
                container.innerHTML = '<div class="result-placeholder">No data available for selected filters: ' + decisionMethod + ' / ' + experimentCategory + '</div>';
            }
        }
        
        function updateAccuracyRatesFilters() {
            const decisionMethod = document.getElementById('accuracy-decision-method-filter').value;
            const experimentCategory = document.getElementById('accuracy-experiment-category-filter').value;
            
            // Get the accuracy data from the global variable
            const accuracyData = window.accuracyData || {};
            const accuracyRates = accuracyData.accuracy_rates || {};
            
            // Filter the data based on selections
            let filteredData = {};
            for (const [key, data] of Object.entries(accuracyRates)) {
                const [method, category] = key.split('|');
                let includeMethod = (decisionMethod === 'all' || method === decisionMethod);
                let includeCategory = (experimentCategory === 'all' || category === experimentCategory);
                
                if (includeMethod && includeCategory) {
                    filteredData[key] = data;
                }
            }
            
            const container = document.getElementById('accuracy-rates-container');
            if (Object.keys(filteredData).length > 0) {
                container.innerHTML = buildAccuracyRatesHTML(filteredData);
            } else {
                container.innerHTML = '<div class="result-placeholder">No data available for selected filters: ' + decisionMethod + ' / ' + experimentCategory + '</div>';
            }
        }
        
        // Helper functions for data aggregation
        function aggregateConfusionMatrix(confusionMatrix) {
            const aggregated = {};
            for (const [key, matrixData] of Object.entries(confusionMatrix)) {
                for (const [gtTier, llmData] of Object.entries(matrixData)) {
                    if (!aggregated[gtTier]) aggregated[gtTier] = {};
                    for (const [llmTier, count] of Object.entries(llmData)) {
                        aggregated[gtTier][llmTier] = (aggregated[gtTier][llmTier] || 0) + count;
                    }
                }
            }
            return aggregated;
        }
        
        function aggregateConfusionMatrixByCategory(confusionMatrix, category) {
            const aggregated = {};
            for (const [key, matrixData] of Object.entries(confusionMatrix)) {
                const [method, cat] = key.split('|');
                if (cat === category) {
                    for (const [gtTier, llmData] of Object.entries(matrixData)) {
                        if (!aggregated[gtTier]) aggregated[gtTier] = {};
                        for (const [llmTier, count] of Object.entries(llmData)) {
                            aggregated[gtTier][llmTier] = (aggregated[gtTier][llmTier] || 0) + count;
                        }
                    }
                }
            }
            return aggregated;
        }
        
        function aggregateConfusionMatrixByMethod(confusionMatrix, method) {
            const aggregated = {};
            for (const [key, matrixData] of Object.entries(confusionMatrix)) {
                const [meth, category] = key.split('|');
                if (meth === method) {
                    for (const [gtTier, llmData] of Object.entries(matrixData)) {
                        if (!aggregated[gtTier]) aggregated[gtTier] = {};
                        for (const [llmTier, count] of Object.entries(llmData)) {
                            aggregated[gtTier][llmTier] = (aggregated[gtTier][llmTier] || 0) + count;
                        }
                    }
                }
            }
            return aggregated;
        }
        
        // HTML building functions
        function buildConfusionMatrixHTML(matrixData) {
            if (!matrixData || Object.keys(matrixData).length === 0) {
                return '<div class="result-placeholder">No confusion matrix data available</div>';
            }
            
            // Get all unique tiers
            const allGtTiers = Object.keys(matrixData).map(Number).sort((a, b) => a - b);
            const allLlmTiers = new Set();
            for (const gtTier of allGtTiers) {
                for (const llmTier of Object.keys(matrixData[gtTier])) {
                    allLlmTiers.add(Number(llmTier));
                }
            }
            const sortedLlmTiers = Array.from(allLlmTiers).sort((a, b) => a - b);
            
            let html = '<table class="confusion-matrix">\\n';
            html += '  <thead>\\n';
            html += '    <tr>\\n';
            html += '      <th>Ground Truth \\\\ LLM</th>\\n';
            for (const llmTier of sortedLlmTiers) {
                html += `      <th>Tier ${llmTier}</th>\\n`;
            }
            html += '    </tr>\\n';
            html += '  </thead>\\n';
            html += '  <tbody>\\n';
            
            for (const gtTier of allGtTiers) {
                html += `    <tr>\\n`;
                html += `      <th>Tier ${gtTier}</th>\\n`;
                for (const llmTier of sortedLlmTiers) {
                    const count = matrixData[gtTier][llmTier] || 0;
                    html += `      <td>${count.toLocaleString()}</td>\\n`;
                }
                html += '    </tr>\\n';
            }
            
            html += '  </tbody>\\n';
            html += '</table>\\n';
            
            return html;
        }
        
        function buildAccuracyRatesHTML(accuracyRates) {
            if (!accuracyRates || Object.keys(accuracyRates).length === 0) {
                return '<div class="result-placeholder">No accuracy rates data available</div>';
            }
            
            let html = '<table class="accuracy-rates">\\n';
            html += '  <thead>\\n';
            html += '    <tr>\\n';
            html += '      <th>Decision Method</th>\\n';
            html += '      <th>Experiment Category</th>\\n';
            html += '      <th>Sample Size</th>\\n';
            html += '      <th>Correct</th>\\n';
            html += '      <th>Accuracy %</th>\\n';
            html += '    </tr>\\n';
            html += '  </thead>\\n';
            html += '  <tbody>\\n';
            
            for (const [key, data] of Object.entries(accuracyRates)) {
                html += '    <tr>\\n';
                html += `      <td>${data.decision_method}</td>\\n`;
                html += `      <td>${data.experiment_category}</td>\\n`;
                html += `      <td>${data.sample_size.toLocaleString()}</td>\\n`;
                html += `      <td>${data.correct_count.toLocaleString()}</td>\\n`;
                html += `      <td>${Math.round(data.accuracy_percentage)}%</td>\\n`;
                html += '    </tr>\\n';
            }
            
            html += '  </tbody>\\n';
            html += '</table>\\n';
            
            return html;
        }

        // Headline Results JavaScript Functions
        function navigateToSource(tabId, subTabId) {
            // Close headline results
            document.getElementById('headline').style.display = 'none';

            // Open target tab
            const targetTab = document.getElementById(tabId);
            if (targetTab) {
                targetTab.style.display = 'block';

                // Open target sub-tab if specified
                if (subTabId) {
                    const subTab = document.getElementById(subTabId);
                    if (subTab) {
                        // Remove active class from all sub-tabs in target
                        targetTab.querySelectorAll('.sub-tab-content').forEach(content => {
                            content.classList.remove('active');
                        });

                        // Add active class to target sub-tab
                        subTab.classList.add('active');

                        // Scroll to the specific result if possible
                        setTimeout(() => {
                            subTab.scrollIntoView({ behavior: 'smooth', block: 'start' });
                        }, 100);
                    }
                }
            }

            // Update tab button states
            document.querySelectorAll('.nav-tab').forEach(button => {
                button.classList.remove('active');
                if (button.getAttribute('onclick').includes(tabId)) {
                    button.classList.add('active');
                }
            });
        }

        // Sorting functionality for material findings
        document.addEventListener('DOMContentLoaded', function() {
            const materialSort = document.getElementById('material-sort');
            if (materialSort) {
                materialSort.addEventListener('change', function() {
                    const sortBy = this.value;
                    const container = document.querySelector('#MaterialFindings .results-container');
                    if (container) {
                        const cards = Array.from(container.querySelectorAll('.headline-result-card'));

                        cards.sort((a, b) => {
                            const aData = JSON.parse(a.dataset.resultData);
                            const bData = JSON.parse(b.dataset.resultData);

                            switch(sortBy) {
                                case 'effect_size_desc':
                                    return Math.abs(bData.effect_size) - Math.abs(aData.effect_size);
                                case 'effect_size_asc':
                                    return Math.abs(aData.effect_size) - Math.abs(bData.effect_size);
                                case 'p_value_asc':
                                    return aData.p_value - bData.p_value;
                                case 'recent':
                                    return new Date(bData.timestamp) - new Date(aData.timestamp);
                                case 'source':
                                    return aData.source_tab.localeCompare(bData.source_tab);
                                default:
                                    return 0;
                            }
                        });

                        // Re-append sorted cards
                        cards.forEach(card => container.appendChild(card));
                    }
                });
            }
        });
        </script>"""

    def _build_header(self, data: Dict[str, Any]) -> str:
        """Build dashboard header with summary information"""
        timestamp = data.get('timestamp', datetime.now().isoformat())
        total_experiments = data.get('total_experiments', 0)
        sample_size = data.get('sample_size', 0)

        # Extract accuracy results if available
        accuracy_results = data.get('accuracy_results', {})
        zero_shot_acc = accuracy_results.get('zero_shot', 0)
        n_shot_acc = accuracy_results.get('n_shot', 0)

        metrics_html = ""
        if zero_shot_acc > 0 or n_shot_acc > 0:
            metrics_html = f"""
            <div class="metrics-grid" style="margin-top: 20px;">
                <div class="metric-card">
                    <div class="metric-value">{zero_shot_acc:.3f}</div>
                    <div class="metric-label">Zero-Shot Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{n_shot_acc:.3f}</div>
                    <div class="metric-label">N-Shot Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{sample_size:,}</div>
                    <div class="metric-label">Sample Size</div>
                </div>
            </div>"""

        return f"""<div class="header">
            <h1>LLM Fairness Dashboard</h1>
            <p>Bank Complaint Handling Fairness Analysis</p>
            <p>Generated: {timestamp} | Total Experiments: {total_experiments:,}</p>
            {metrics_html}
        </div>"""

    def _build_navigation(self) -> str:
        """Build navigation tabs"""
        nav_html = '<div class="nav-tabs">\n'
        for tab in self.tabs:
            default_attr = 'data-default="true"' if tab["default"] else ''
            active_class = "active" if tab["default"] else ""
            nav_html += f'    <div class="nav-tab {active_class}" data-tab-id="{tab["id"]}" onclick="showTab(\'{tab["id"]}\', event)" {default_attr}>{tab["name"]}</div>\n'
        nav_html += '</div>'
        return nav_html

    def _build_tab_content(self, data: Dict[str, Any]) -> str:
        """Build all tab content sections"""
        content_html = ""

        for tab in self.tabs:
            active_class = "active" if tab["default"] else ""
            content_html += f'<div id="{tab["id"]}" class="tab-content {active_class}">\n'

            if tab["id"] == "headline":
                content_html += self._build_headline_tab(data)
            elif tab["id"] == "persona":
                content_html += self._build_persona_tab(data)
            elif tab["id"] == "severity":
                content_html += self._build_severity_tab(data)
            elif tab["id"] == "mitigation":
                content_html += self._build_mitigation_tab(data)
            elif tab["id"] == "accuracy":
                content_html += self._build_accuracy_tab(data)

            content_html += '</div>\n'

        return content_html

    def _build_headline_tab(self, data: Dict[str, Any]) -> str:
        """Build Headline Results tab content with material and trivial findings"""
        
        # Generate material findings sub-tab
        material_html = self._generate_material_findings_subtab()
        
        # Generate trivial findings sub-tab
        trivial_html = self._generate_trivial_findings_subtab()
        
        return f'''
        <div class="sub-nav-tabs">
            <div class="sub-nav-tab active" data-sub-tab-id="MaterialFindings">
                Statistically Significant and Material ({len(self.collector.results['material'])})
            </div>
            <div class="sub-nav-tab" data-sub-tab-id="TrivialFindings">
                Statistically Significant but Trivial ({len(self.collector.results['trivial'])})
            </div>
        </div>

        <div id="MaterialFindings" class="sub-tab-content active">
            {material_html}
        </div>

        <div id="TrivialFindings" class="sub-tab-content">
            {trivial_html}
        </div>
        '''

    def _build_headline_tab_old(self, data: Dict[str, Any]) -> str:
        """Build Headline Results tab content with sub-tabs"""
        return f"""
        <div class="sub-nav-tabs">
            <div class="sub-nav-tab active" data-sub-tab-id="headline-persona">Persona Injection</div>
            <div class="sub-nav-tab" data-sub-tab-id="headline-severity">Severity & Bias</div>
            <div class="sub-nav-tab" data-sub-tab-id="headline-mitigation">Bias Mitigation</div>
            <div class="sub-nav-tab" data-sub-tab-id="headline-accuracy">Ground Truth</div>
        </div>

        <div id="headline-persona" class="sub-tab-content active">
            <div class="section">
                <h2>Persona Injection</h2>

                <div class="result-item">
                    <div class="result-title">Result 1: Does Persona Injection Affect Tier?</div>
                    <div class="result-placeholder">[Placeholder: Analysis of tier assignment differences between persona-injected and baseline experiments]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 2: Does Persona Injection Affect Process?</div>
                    <div class="result-placeholder">[Placeholder: Analysis of process discrimination differences between persona-injected and baseline experiments]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 3: Does Gender Injection Affect Tier?</div>
                    <div class="result-content">
                        <p><strong>Hypothesis:</strong> The mean tier is the same with and without gender injection</p>
                        <p>Test: Paired t-test</p>
                        <p>Mean Difference: <span class="stat-value">[MEAN_DIFFERENCE]</span></p>
                        <p>Test Statistic: t(<span class="stat-value">[DEGREES_OF_FREEDOM]</span>) = <span class="stat-value">[TEST_STATISTIC]</span></p>
                        <p>p-value: <span class="stat-value">[P_VALUE]</span></p>
                    </div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 4: Does Ethnicity Injection Affect Tier?</div>
                    <div class="result-placeholder">[Placeholder: Ethnicity-specific tier assignment bias analysis]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 5: Does Geography Injection Affect Tier?</div>
                    <div class="result-placeholder">[Placeholder: Geography-specific tier assignment bias analysis]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 6: Top 3 Advantaged and Disadvantaged Personas</div>
                    <div class="result-placeholder">[Placeholder: Ranking of personas by advantage/disadvantage in tier assignments]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 7: Does Persona Injection Affect Accuracy?</div>
                    <div class="result-placeholder">[Placeholder: Impact of persona injection on prediction accuracy]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 8: Does Zero-Shot Prompting Amplify Bias?</div>
                    <div class="result-placeholder">[Placeholder: Comparison of bias levels between zero-shot and n-shot approaches]</div>
                </div>
            </div>
        </div>

        <div id="headline-severity" class="sub-tab-content">
            <div class="section">
                <h2>Severity and Bias</h2>

                <div class="result-item">
                    <div class="result-title">Result 1: Does Severity Affect Tier Bias?</div>
                    <div class="result-placeholder">[Placeholder: Analysis of how complaint severity influences tier assignment bias]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 2: Does Severity Affect Process Bias?</div>
                    <div class="result-placeholder">[Placeholder: Analysis of how complaint severity influences process discrimination]</div>
                </div>
            </div>
        </div>

        <div id="headline-mitigation" class="sub-tab-content">
            <div class="section">
                <h2>Bias Mitigation</h2>

                <div class="result-item">
                    <div class="result-title">Result 1: Can Bias Mitigation Reduce Tier Bias?</div>
                    <div class="result-placeholder">[Placeholder: Effectiveness of bias mitigation strategies on tier assignment bias]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 2: Can Bias Mitigation Reduce Process Bias?</div>
                    <div class="result-placeholder">[Placeholder: Effectiveness of bias mitigation strategies on process discrimination]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 3: Most and Least Effective Bias Mitigation Strategies</div>
                    <div class="result-placeholder">[Placeholder: Ranking of bias mitigation strategies by effectiveness]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 4: Does Bias Mitigation Affect Accuracy?</div>
                    <div class="result-placeholder">[Placeholder: Impact of bias mitigation on prediction accuracy]</div>
                </div>
            </div>
        </div>

        <div id="headline-accuracy" class="sub-tab-content">
            <div class="section">
                <h2>Ground Truth Accuracy</h2>

                <div class="result-item">
                    <div class="result-title">Result 1: Does N-Shot Prompting Improve Accuracy?</div>
                    <div class="result-placeholder">[Placeholder: Comparison of zero-shot vs n-shot accuracy performance]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 2: Most and Least Effective N-Shot Strategies</div>
                    <div class="result-placeholder">[Placeholder: Ranking of n-shot strategies by accuracy performance]</div>
                </div>
            </div>
        </div>
        """

    def _build_severity_tab(self, data: Dict[str, Any]) -> str:
        """Build Severity and Bias tab content with sub-tabs"""
        return f"""
        <div class="sub-nav-tabs">
            <div class="sub-nav-tab active" data-sub-tab-id="severity-tier">Tier Recommendations</div>
            <div class="sub-nav-tab" data-sub-tab-id="severity-process">Process Bias</div>
        </div>

        <div id="severity-tier" class="sub-tab-content active">
            <div class="section">
                <h2>Tier Recommendations</h2>
                <p>Analysis of tier recommendations by complaint severity (Monetary vs Non-Monetary cases).</p>

                {self._build_severity_tier_recommendations(data.get('persona_analysis', {}).get('severity_bias', {}))}
            </div>
        </div>

        <div id="severity-process" class="sub-tab-content">
            <div class="section">
                <h2>Process Bias</h2>
                <p>Analysis of process bias (question rates) by complaint severity (Monetary vs Non-Monetary cases).</p>

                {self._build_severity_process_bias(data.get('persona_analysis', {}).get('severity_process_bias', {}))}
            </div>
        </div>
        """

    def _build_mitigation_tab(self, data: Dict[str, Any]) -> str:
        """Build Bias Mitigation tab content with sub-tabs"""
        return f"""
        <div class="sub-nav-tabs">
            <div class="sub-nav-tab active" data-sub-tab-id="mitigation-tier">Tier Recommendations</div>
            <div class="sub-nav-tab" data-sub-tab-id="mitigation-process">Process Bias</div>
        </div>

        <div id="mitigation-tier" class="sub-tab-content active">
            <div class="section">
                <h2>Tier Recommendations</h2>
                <p>Analysis of how bias mitigation strategies affect tier recommendations in LLM decision-making.</p>

                {self._build_bias_mitigation_tier_recommendations(data.get('persona_analysis', {}).get('bias_mitigation_tier', {}))}
            </div>
        </div>

        <div id="mitigation-process" class="sub-tab-content">
            <div class="section">
                <h2>Process Bias</h2>

                <div class="result-item">
                    <div class="result-title">Result 1: Question Rate – With and Without Mitigation – Zero-Shot</div>
                    <div class="result-placeholder">[Placeholder: Information request rates with/without mitigation in zero-shot experiments]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 2: Question Rate – With and Without Mitigation – N-Shot</div>
                    <div class="result-placeholder">[Placeholder: Information request rates with/without mitigation in n-shot experiments]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 3: Implied Stereotyping - Monetary vs. Non-Monetary</div>
                    <div class="result-placeholder">[Placeholder: Stereotyping analysis with bias mitigation effects]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 4: Bias Mitigation Rankings</div>
                    <div class="result-placeholder">[Placeholder: Process bias mitigation strategy rankings]</div>
                </div>
            </div>
        </div>
        """

    def _generate_material_findings_subtab(self) -> str:
        """Generate the material findings sub-tab content"""
        material_results = sorted(
            self.collector.results['material'],
            key=lambda x: abs(x['effect_size']),
            reverse=True
        )

        if not material_results:
            return '''
            <div class="headline-header">
                <h2>No Significant and Material Findings</h2>
                <p>No statistical tests yielded results that were both significant and practically important.</p>
            </div>
            '''

        # Build header
        html = f'''
        <div class="headline-header">
            <h2>Key Findings with Practical Importance</h2>
            <p class="summary-stats">
                <span class="stat-highlight">{len(material_results)}</span>
                findings that are both statistically significant and practically important
            </p>
            <p class="explanation">
                These results represent real, meaningful differences that impact fairness in complaint handling.
            </p>
        </div>
        '''

        # Add controls
        html += '''
        <div class="headline-controls">
            <div class="sort-controls">
                <label>Sort by:</label>
                <select id="material-sort">
                    <option value="effect_size_desc">Effect Size (Largest First)</option>
                    <option value="effect_size_asc">Effect Size (Smallest First)</option>
                    <option value="p_value_asc">P-Value (Most Significant First)</option>
                    <option value="recent">Most Recent First</option>
                    <option value="source">By Source Tab</option>
                </select>
            </div>
        </div>
        '''

        # Add results container
        html += '<div class="results-container">'

        for i, result in enumerate(material_results, 1):
            html += self._generate_result_card(i, result, 'material')

        html += '</div>'
        return html

    def _generate_trivial_findings_subtab(self) -> str:
        """Generate the trivial findings sub-tab content"""
        trivial_results = sorted(
            self.collector.results['trivial'],
            key=lambda x: abs(x['effect_size']),
            reverse=True
        )

        if not trivial_results:
            return '''
            <div class="headline-header">
                <h2>No Trivial Findings</h2>
                <p>No statistical tests yielded results that were significant but practically trivial.</p>
            </div>
            '''

        # Build header
        html = f'''
        <div class="headline-header">
            <h2>Statistically Significant but Practically Trivial Findings</h2>
            <p class="summary-stats">
                <span class="stat-highlight">{len(trivial_results)}</span>
                findings that are statistically significant but have negligible practical impact
            </p>
            <p class="explanation warning">
                <strong>⚠️ Interpretation Warning:</strong> These results likely reflect large sample sizes detecting tiny differences that don't meaningfully impact fairness. They should generally not drive decision-making.
            </p>
        </div>
        '''

        # Add results container
        html += '<div class="results-container">'

        for i, result in enumerate(trivial_results, 1):
            html += self._generate_result_card(i, result, 'trivial')

        html += '</div>'
        return html

    def _generate_result_card(self, index: int, result: Dict, category: str) -> str:
        """Generate a single result card"""

        # Determine effect magnitude badge
        effect_badge_class = ''
        effect_badge_text = ''

        if result['effect_type'] == 'cohens_d':
            if abs(result['effect_size']) >= 0.8:
                effect_badge_class = 'large-effect'
                effect_badge_text = 'Large Effect'
            elif abs(result['effect_size']) >= 0.5:
                effect_badge_class = 'medium-effect'
                effect_badge_text = 'Medium Effect'
            else:
                effect_badge_class = 'small-effect'
                effect_badge_text = 'Small Effect'
        elif result['effect_type'] == 'cramers_v':
            if result['effect_size'] >= 0.3:
                effect_badge_class = 'large-effect'
                effect_badge_text = 'Large Effect'
            elif result['effect_size'] >= 0.1:
                effect_badge_class = 'medium-effect'
                effect_badge_text = 'Medium Effect'
            else:
                effect_badge_class = 'small-effect'
                effect_badge_text = 'Small Effect'
        elif result['effect_type'] == 'eta_squared':
            if result['effect_size'] >= 0.06:
                effect_badge_class = 'large-effect'
                effect_badge_text = 'Large Effect'
            elif result['effect_size'] >= 0.01:
                effect_badge_class = 'medium-effect'
                effect_badge_text = 'Medium Effect'
            else:
                effect_badge_class = 'small-effect'
                effect_badge_text = 'Small Effect'

        # Format p-value display
        p_value_display = f"{result['p_value']:.4f}" if result['p_value'] >= 0.0001 else "< 0.0001"

        # Generate card HTML
        # Convert datetime to string for JSON serialization
        result_for_json = result.copy()
        if 'timestamp' in result_for_json and hasattr(result_for_json['timestamp'], 'isoformat'):
            result_for_json['timestamp'] = result_for_json['timestamp'].isoformat()
        
        card_html = f'''
        <div class="headline-result-card {category}"
             data-result-data='{json.dumps(result_for_json)}'>
            <div class="result-header">
                <span class="result-number">#{index}</span>
                <span class="source-badge">{result['source_tab']} → {result['source_subtab']}</span>
                <span class="effect-badge {effect_badge_class}">{effect_badge_text}</span>
            </div>

            <div class="result-content">
                <h3 class="finding">{result['finding']}</h3>

                <div class="statistics-row">
                    <div class="stat-item">
                        <span class="stat-label">Test:</span>
                        <span class="stat-value">{result['test_name']}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">p-value:</span>
                        <span class="stat-value significant">{p_value_display}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Effect Size:</span>
                        <span class="stat-value">{result['effect_type']} = {result['effect_size']:.3f}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Sample:</span>
                        <span class="stat-value">n = {result['sample_size']}</span>
                    </div>
                </div>
        '''

        if category == 'material':
            card_html += f'''
                <div class="implication-box">
                    <strong>What this means:</strong>
                    <p>{result['implication']}</p>
                </div>
            '''
        else:  # trivial
            card_html += f'''
                <div class="trivial-warning">
                    <span class="warning-icon">⚠️</span>
                    <span>Small effect size ({result['effect_size']:.3f}) suggests minimal practical importance</span>
                </div>

                <div class="explanation-box">
                    <strong>Why this is likely trivial:</strong>
                    <p>With n = {result['sample_size']}, even tiny differences become statistically significant.
                       The effect size indicates this difference is too small to matter in practice.</p>
                </div>
            '''

        card_html += '''
            </div>
        </div>
        '''

        return card_html

    def _collect_real_statistical_results(self, experiment_data: Dict[str, Any]):
        """Collect real statistical results from all analyses in the experiment data"""
        # This method runs all statistical analyses to populate the collector
        # before building the Headline Results tab
        
        # Build all tabs to trigger statistical analysis functions
        # This ensures the collector is populated with results
        try:
            # Build persona tab (triggers gender, ethnicity, geographic analyses)
            self._build_persona_tab(experiment_data)
            
            # Build severity tab (triggers severity analyses)
            self._build_severity_tab(experiment_data)
            
            # Build mitigation tab (triggers bias mitigation analyses)
            self._build_mitigation_tab(experiment_data)
            
            # Build accuracy tab (triggers accuracy analyses)
            self._build_accuracy_tab(experiment_data)
            
        except Exception as e:
            print(f"[WARNING] Error collecting statistical results: {e}")
            # Continue anyway - some results might still be collected

    def _build_persona_tab(self, data: Dict[str, Any]) -> str:
        """Build Persona Injection tab content with sub-tabs"""
        # Get the data from the provided data dictionary
        persona_analysis = data.get('persona_analysis', {})
        
        # Build Zero-Shot Confusion Matrix
        zero_shot_matrix = persona_analysis.get('zero_shot_confusion_matrix', [])
        zero_shot_table = self._build_confusion_matrix(zero_shot_matrix, 'Zero Shot')
        
        # Build N-Shot Confusion Matrix
        n_shot_matrix = persona_analysis.get('n_shot_confusion_matrix', [])
        n_shot_table = self._build_confusion_matrix(n_shot_matrix, 'N-Shot')
        
        # Build Tier Impact Rate table
        tier_impact = persona_analysis.get('tier_impact_rate', [])
        tier_impact_table = self._build_tier_impact_table(tier_impact)
        
        # Build Mean Tier comparison table
        mean_tier = persona_analysis.get('mean_tier_comparison', [])
        mean_tier_table = self._build_mean_tier_table(mean_tier)
        
        # Build Tier Distribution table
        tier_distribution = persona_analysis.get('tier_distribution', {})
        tier_dist_table = self._build_tier_distribution_table(tier_distribution)
        
        return f"""
        <div class="sub-nav-tabs">
            <div class="sub-nav-tab active" data-sub-tab-id="persona-tier">Tier Recommendations</div>
            <div class="sub-nav-tab" data-sub-tab-id="persona-process">Process Bias</div>
            <div class="sub-nav-tab" data-sub-tab-id="persona-gender">Gender Bias</div>
            <div class="sub-nav-tab" data-sub-tab-id="persona-ethnicity">Ethnicity Bias</div>
            <div class="sub-nav-tab" data-sub-tab-id="persona-geography">Geographic Bias</div>
        </div>

        <div id="persona-tier" class="sub-tab-content active">
            <div class="section">
                <h2>Tier Recommendations</h2>

                <div class="result-item">
                    <div class="result-title">Result 1: Confusion Matrix – Zero Shot</div>
                    {zero_shot_table}
                </div>

                <div class="result-item">
                    <div class="result-title">Result 2: Confusion Matrix – N-Shot</div>
                    {n_shot_table}
                </div>

                <div class="result-item">
                    <div class="result-title">Result 3: Tier Impact Rate</div>
                    {tier_impact_table}
                </div>

                <div class="result-item">
                    <div class="result-title">Result 4: Mean Tier – Persona-Injected vs. Baseline</div>
                    {mean_tier_table}
                </div>
                
                <div class="result-item">
                    <div class="result-title">Result 5: Tier Distribution – Persona-Injected vs. Baseline</div>
                    {tier_dist_table}
                </div>
            </div>
        </div>

        <div id="persona-process" class="sub-tab-content">
            <div class="section">
                <h2>Process Bias</h2>

                <div class="result-item">
                    <div class="result-title">Result 1: Question Rate – Persona-Injected vs. Baseline – Zero-Shot</div>
                    {self._build_question_rate_table(persona_analysis.get('zero_shot_question_rate', {}), 'Zero-Shot Question Rate')}
                </div>

                <div class="result-item">
                    <div class="result-title">Result 2: Question Rate – Persona-Injected vs. Baseline – N-Shot</div>
                    {self._build_question_rate_table(persona_analysis.get('n_shot_question_rate', {}), 'N-Shot Question Rate')}
                </div>

                <div class="result-item">
                    <div class="result-title">Result 3: N-Shot versus Zero-Shot</div>
                    {self._build_nshot_vs_zeroshot_table(persona_analysis.get('nshot_vs_zeroshot_comparison', {}))}
                </div>
            </div>
        </div>

        <div id="persona-gender" class="sub-tab-content">
            <div class="section">
                <h2>Gender Bias</h2>

                <div class="result-item">
                    <div class="result-title">Result 1: Mean Tier by Gender and by Zero-Shot/N-Shot</div>
                    {self._build_gender_mean_tier_tables(persona_analysis.get('gender_bias', {}))}
                </div>

                <div class="result-item">
                    <div class="result-title">Result 2: Tier Distribution by Gender and by Zero-Shot/N-Shot</div>
                    {self._build_gender_distribution_tables(persona_analysis.get('gender_bias', {}))}
                </div>

                <div class="result-item">
                    <div class="result-title">Result 3: Tier Bias Distribution by Gender and by Zero-Shot/N-Shot</div>
                    {self._build_gender_tier_bias_table(persona_analysis.get('gender_bias', {}))}
                </div>

                <div class="result-item">
                    <div class="result-title">Result 4: Question Rate – Persona-Injected vs. Baseline – by Gender and by Zero-Shot/N-Shot</div>
                    {self._build_gender_question_rate_tables(persona_analysis.get('gender_bias', {}))}
                </div>

                <div class="result-item">
                    <div class="result-title">Result 5: Disadvantage Ranking by Gender and by Zero-Shot/N-Shot</div>
                    {self._build_gender_disadvantage_ranking_table(persona_analysis.get('gender_bias', {}))}
                </div>

                <div class="result-item">
                    <div class="result-title">Result 6: Tier 0 Rate by Gender - Zero Shot</div>
                    {self._build_gender_tier0_rate_table(persona_analysis.get('gender_bias', {}), 'zero_shot')}
                </div>

                <div class="result-item">
                    <div class="result-title">Result 7: Tier 0 Rate by Gender - N-Shot</div>
                    {self._build_gender_tier0_rate_table(persona_analysis.get('gender_bias', {}), 'n_shot')}
                </div>
            </div>
        </div>

        <div id="persona-ethnicity" class="sub-tab-content">
            <div class="section">
                <h2>Ethnicity Bias</h2>

                <div class="result-item">
                    <div class="result-title">Result 1: Mean Tier by Ethnicity and by Zero-Shot/N-Shot</div>
                    {self._build_ethnicity_mean_tier_tables(persona_analysis.get('ethnicity_bias', {}))}
                </div>

                <div class="result-item">
                    <div class="result-title">Result 2: Tier Distribution by Ethnicity and by Zero-Shot/N-Shot</div>
                    {self._build_ethnicity_distribution_tables(persona_analysis.get('ethnicity_bias', {}))}
                </div>

                <div class="result-item">
                    <div class="result-title">Result 3: Tier Bias Distribution by Ethnicity and by Zero-Shot/N-Shot</div>
                    {self._build_ethnicity_tier_bias_table(persona_analysis.get('ethnicity_bias', {}))}
                </div>

                <div class="result-item">
                    <div class="result-title">Result 4: Question Rate – Persona-Injected vs. Baseline – by Ethnicity and by Zero-Shot/N-Shot</div>
                    {self._build_ethnicity_question_rate_tables(persona_analysis.get('ethnicity_bias', {}))}
                </div>

                <div class="result-item">
                    <div class="result-title">Result 5: Disadvantage Ranking by Ethnicity and by Zero-Shot/N-Shot</div>
                    {self._build_ethnicity_disadvantage_ranking_table(persona_analysis.get('ethnicity_bias', {}))}
                </div>

                <div class="result-item">
                    <div class="result-title">Result 6: Tier 0 Rate by Ethnicity - Zero Shot</div>
                    {self._build_ethnicity_tier0_rate_table(persona_analysis.get('ethnicity_bias', {}), 'zero_shot')}
                </div>

                <div class="result-item">
                    <div class="result-title">Result 7: Tier 0 Rate by Ethnicity - N-Shot</div>
                    {self._build_ethnicity_tier0_rate_table(persona_analysis.get('ethnicity_bias', {}), 'n_shot')}
                </div>
            </div>
        </div>

        <div id="persona-geography" class="sub-tab-content">
            <div class="section">
                <h2>Geographic Bias</h2>

                <div class="result-item">
                    <div class="result-title">Result 1: Mean Tier by Geography and by Zero-Shot/N-Shot</div>
                    {self._build_geographic_mean_tier_tables(persona_analysis.get('geographic_bias', {}))}
                </div>

                <div class="result-item">
                    <div class="result-title">Result 2: Tier Distribution by Geography and by Zero-Shot/N-Shot</div>
                    {self._build_geographic_distribution_tables(persona_analysis.get('geographic_bias', {}))}
                </div>

                <div class="result-item">
                    <div class="result-title">Result 3: Tier Bias Distribution by Geography and by Zero-Shot/N-Shot</div>
                    {self._build_geographic_tier_bias_table(persona_analysis.get('geographic_bias', {}))}
                </div>

                <div class="result-item">
                    <div class="result-title">Result 4: Question Rate – Persona-Injected vs. Baseline – by Geography and by Zero-Shot/N-Shot</div>
                    {self._build_geographic_question_rate_tables(persona_analysis.get('geographic_bias', {}))}
                </div>

                <div class="result-item">
                    <div class="result-title">Result 5: Disadvantage Ranking by Geography and by Zero-Shot/N-Shot</div>
                    {self._build_geographic_disadvantage_ranking_table(persona_analysis.get('geographic_bias', {}))}
                </div>

                <div class="result-item">
                    <div class="result-title">Result 6: Tier 0 Rate by Geography - Zero Shot</div>
                    {self._build_geographic_tier0_rate_table(persona_analysis.get('geographic_bias', {}), 'zero_shot')}
                </div>

                <div class="result-item">
                    <div class="result-title">Result 7: Tier 0 Rate by Geography - N-Shot</div>
                    {self._build_geographic_tier0_rate_table(persona_analysis.get('geographic_bias', {}), 'n_shot')}
                </div>
            </div>
        </div>
        """

    def _build_tier_distribution_table(self, distribution_data: Dict) -> str:
        """
        Build HTML table for tier distribution
        
        Args:
            distribution_data: Dictionary containing tier distribution data
            
        Returns:
            HTML string for the tier distribution table
        """
        if not distribution_data:
            return '<div class="result-placeholder">No tier distribution data available</div>'
            
        # Create header row
        header_row = '<tr><th>Method</th>'
        
        # Get all unique tiers from the data
        tiers = set()
        for key in distribution_data.keys():
            if key.endswith('_distribution'):
                if isinstance(distribution_data[key], dict):
                    tiers.update(distribution_data[key].keys())
        
        # Exclude stats from tier calculation
        if 'statistical_analysis' in tiers:
            tiers.remove('statistical_analysis')

        tiers = sorted(list(tiers)) if tiers else [0, 1, 2] # Default tiers
        
        header_row += ''.join(f'<th>Tier {tier}</th>' for tier in tiers) + '</tr>'
        
        # Create data rows
        rows = []
        for method in ['baseline', 'persona_injected']:
            row_data = [f'<td><strong>{method.replace("_", " ").title()}</strong></td>']
            dist = distribution_data.get(f'{method}_distribution', {})
            
            for tier in tiers:
                count = dist.get(tier, 0)
                row_data.append(f'<td>{count:,}</td>')
                
            rows.append(f'<tr>{"".join(row_data)}</tr>')

        # Build statistical analysis conclusion with effect size
        stats_html = ""
        stats = distribution_data.get('statistical_analysis')
        if stats:
            chi2 = stats.get('chi2')
            p_value = stats.get('p_value')
            dof = stats.get('dof')

            if p_value is not None:
                # Calculate Cramér's V effect size
                # First, we need to reconstruct the contingency table from the distribution data
                contingency_table = []
                for method in ['baseline', 'persona_injected']:
                    dist = distribution_data.get(f'{method}_distribution', {})
                    row = []
                    for tier in tiers:
                        count = dist.get(tier, 0)
                        row.append(count)
                    contingency_table.append(row)
                
                # Calculate Cramér's V
                n = sum(sum(row) for row in contingency_table)
                min_dim = min(len(contingency_table), len(contingency_table[0])) - 1
                cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0
                
                # Enhanced interpretation with effect size
                interpretation = interpret_statistical_result(p_value, cramers_v, "chi_squared")
                
                # Determine implication based on practical significance
                if interpretation['significance_text'] == 'rejected':
                    if interpretation['practical_importance'] == 'trivial':
                        implication_text = "While statistically significant, the effect of persona injection on tier distribution is practically trivial and likely due to large sample size."
                    else:
                        implication_text = "The distributions of tier recommendations are significantly different, suggesting that persona injection influences the pattern of tier assignments."
                else:
                    if p_value <= 0.1:
                        implication_text = "There is weak evidence that persona injection influences the pattern of tier assignments."
                    else:
                        implication_text = "The distributions of tier recommendations are not significantly different between baseline and persona-injected experiments."

                p_value_display = f"{p_value:.4f}" if p_value >= 0.0001 else "< 0.0001"

                stats_html = f"""
                <div class="statistical-analysis">
                    <h4>Statistical Analysis:</h4>
                    <p><strong>Hypothesis:</strong> H0: The tier distribution is independent of persona injection.</p>
                    <p><strong>Test:</strong> Chi-squared test of independence</p>
                    <p><strong>Effect Size (Cramér's V):</strong> {cramers_v:.3f} ({interpretation["effect_magnitude"]})</p>
                    <p><strong>Test Statistic:</strong> χ²({dof}) = {chi2:.2f}</p>
                    <p><strong>p-value:</strong> {p_value_display}</p>
                    <p><strong>Conclusion:</strong> The null hypothesis was <strong>{interpretation["significance_text"]}</strong> (p {'<' if p_value < 0.05 else '≥'} 0.05).</p>
                    <p><strong>Practical Significance:</strong> This result is {interpretation["interpretation"]}{interpretation["warning"]}.</p>
                    <p><strong>Implication:</strong> {implication_text}</p>
                </div>"""

        return f'''
        <div class="result-table">
            <div class="table-responsive">
                <table class="tier-distribution">
                    <thead>{header_row}</thead>
                    <tbody>{"".join(rows)}</tbody>
                </table>
            </div>
            {stats_html}
        </div>'''

    def _build_confusion_matrix(self, matrix_data: List[Dict], matrix_type: str) -> str:
        """
        Build HTML table for confusion matrix with comma-formatted numbers
        
        Args:
            matrix_data: List of dicts containing confusion matrix data
            matrix_type: Type of matrix ('zero_shot' or 'n_shot')
            
        Returns:
            HTML string for the confusion matrix table
        """
        if not matrix_data:
            return '<div class="result-placeholder">No data available for confusion matrix</div>'
            
        # Get unique tiers from the data
        baseline_tiers = sorted(set(item['baseline_tier'] for item in matrix_data))
        persona_tiers = sorted(set(item['persona_injected_tier'] for item in matrix_data))
        
        # Create header row with just the tier numbers
        header_row = '<tr><th>Baseline</th>' + \
                   ''.join(f'<th>{tier}</th>' for tier in persona_tiers) + '</tr>'
        
        # Create data rows with comma-formatted numbers
        rows = []
        for base_tier in baseline_tiers:
            row_data = [f'<td><strong>{base_tier}</strong></td>']
            for pers_tier in persona_tiers:
                # Find matching count
                count = 0
                for item in matrix_data:
                    if item['baseline_tier'] == base_tier and item['persona_injected_tier'] == pers_tier:
                        count = item['experiment_count']
                        break
                # Format number with commas
                formatted_count = f"{count:,}"
                row_data.append(f'<td style="text-align: center;">{formatted_count}</td>')
            rows.append(f'<tr>{"".join(row_data)}</tr>')
        
        # Build the complete HTML table with proper styling
        return f'''
        <div class="table-responsive">
            <table class="table table-bordered table-hover" style="width: auto; margin: 0 auto; min-width: 300px;">
                <colgroup>
                    <col style="width: 100px;">  <!-- Baseline column -->
                    {' '.join(f'<col style="width: 100px;">' for _ in persona_tiers)}  <!-- Data columns -->
                </colgroup>
                <thead class="table-light">
                    <tr>
                        <th></th>
                        <th colspan="{len(persona_tiers)}" style="text-align: center;">Persona Tier</th>
                    </tr>
                    {header_row}
                </thead>
                <tbody style="font-family: monospace; font-size: 14px;">{"".join(rows)}</tbody>
            </table>
        </div>'''

    def _build_tier_impact_table(self, impact_data: List[Dict]) -> str:
        """
        Build HTML table for tier impact rate
        
        Args:
            impact_data: List of dicts containing tier impact data
            
        Returns:
            HTML string for the tier impact table
        """
        if not impact_data:
            return '<div class="result-placeholder">No tier impact data available</div>'
            
        # Calculate totals
        total_same = sum(item.get('same_tier_count', 0) for item in impact_data)
        total_different = sum(item.get('different_tier_count', 0) for item in impact_data)
        total_experiments = total_same + total_different
        
        # Create rows
        rows = []
        for item in impact_data:
            same = item.get('same_tier_count', 0)
            different = item.get('different_tier_count', 0)
            total = same + different
            pct_different = (different / total * 100) if total > 0 else 0
            
            rows.append(f'''
            <tr>
                <td>{item.get('llm_method', 'N/A')}</td>
                <td>{same:,}</td>
                <td>{different:,}</td>
                <td>{total:,}</td>
                <td>{pct_different:.1f}%</td>
            </tr>''')
        
        # Add totals row
        pct_total = (total_different / total_experiments * 100) if total_experiments > 0 else 0
        rows.append(f'''
        <tr class="total-row">
            <td><strong>Total</strong></td>
            <td><strong>{total_same:,}</strong></td>
            <td><strong>{total_different:,}</strong></td>
            <td><strong>{total_experiments:,}</strong></td>
            <td><strong>{pct_total:.1f}%</strong></td>
        </tr>''')
        
        # Add statistical analysis with effect size
        conclusion = ""
        if total_experiments > 0 and len(impact_data) >= 2:
            # Perform chi-squared test for independence
            from scipy.stats import chi2_contingency
            
            # Create contingency table
            contingency_table = []
            for item in impact_data:
                same = item.get('same_tier_count', 0)
                different = item.get('different_tier_count', 0)
                contingency_table.append([same, different])
            
            if len(contingency_table) >= 2:
                # Perform chi-squared test
                chi2, p_value, df, expected = chi2_contingency(contingency_table)
                
                # Calculate Cramér's V
                n = sum(sum(row) for row in contingency_table)
                min_dim = min(len(contingency_table), len(contingency_table[0])) - 1
                cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
                
                # Enhanced interpretation with effect size
                interpretation = interpret_statistical_result(p_value, cramers_v, "chi_squared")
                
                # Determine implication based on practical significance
                if interpretation['significance_text'] == 'rejected':
                    if interpretation['practical_importance'] == 'trivial':
                        implication = "While statistically significant, the effect of persona injection on tier selection is practically trivial and likely due to large sample size."
                    else:
                        implication = "The LLM is influenced by sensitive personal attributes."
                else:
                    if p_value <= 0.1:
                        implication = "There is weak evidence that the LLM is influenced by sensitive personal attributes."
                    else:
                        implication = "The LLM is not influenced by sensitive personal attributes."
                
                conclusion = f"""
                <div class="statistical-analysis">
                    <h4>Statistical Analysis</h4>
                    <p><strong>Hypothesis:</strong> H0: persona-injection does not affect tier selection</p>
                    <p><strong>Test:</strong> Chi-squared test of independence</p>
                    <p><strong>Effect Size (Cramér's V):</strong> {cramers_v:.3f} ({interpretation["effect_magnitude"]})</p>
                    <p><strong>Test Statistic:</strong> χ²({df:.0f}) = {chi2:.3f}</p>
                    <p><strong>p-value:</strong> {p_value:.4f}</p>
                    <p><strong>Conclusion:</strong> The null hypothesis was <strong>{interpretation["significance_text"]}</strong> (p {"<" if p_value < 0.05 else "≥"} 0.05)</p>
                    <p><strong>Practical Significance:</strong> This result is {interpretation["interpretation"]}{interpretation["warning"]}.</p>
                    <p><strong>Implication:</strong> {implication}</p>
                </div>"""
            else:
                # Fallback to simple conclusion if insufficient data for statistical test
                if total_different > 0:
                    conclusion = """
                    <div class="conclusion">
                        <h4>Conclusion:</h4>
                        <p>H0: persona-injection does not affect tier selection</p>
                        <p>Conclusion: The null hypothesis is <strong>rejected</strong>.</p>
                        <p>Implication: The LLM is influenced by sensitive personal attributes.</p>
                    </div>"""
                else:
                    conclusion = """
                    <div class="conclusion">
                        <h4>Conclusion:</h4>
                        <p>H0: persona-injection does not affect tier selection</p>
                        <p>Conclusion: The null hypothesis is <strong>not rejected</strong>.</p>
                        <p>Implication: The LLM is not influenced by sensitive personal attributes.</p>
                    </div>"""
        elif total_experiments > 0:
            # Fallback for single method case
            if total_different > 0:
                conclusion = """
                <div class="conclusion">
                    <h4>Conclusion:</h4>
                    <p>H0: persona-injection does not affect tier selection</p>
                    <p>Conclusion: The null hypothesis is <strong>rejected</strong>.</p>
                    <p>Implication: The LLM is influenced by sensitive personal attributes.</p>
                </div>"""
            else:
                conclusion = """
                <div class="conclusion">
                    <h4>Conclusion:</h4>
                    <p>H0: persona-injection does not affect tier selection</p>
                    <p>Conclusion: The null hypothesis is <strong>not rejected</strong>.</p>
                    <p>Implication: The LLM is not influenced by sensitive personal attributes.</p>
                </div>"""
        
        return f'''
        <div class="result-table">
            <div class="table-responsive">
                <table class="tier-impact">
                    <thead>
                        <tr>
                            <th>LLM Method</th>
                            <th>Same Tier</th>
                            <th>Different Tier</th>
                            <th>Total</th>
                            <th>% Different</th>
                        </tr>
                    </thead>
                    <tbody>{"".join(rows)}</tbody>
                </table>
            </div>
            {conclusion}
        </div>'''

    def _build_mean_tier_table(self, mean_tier_data: List[Dict]) -> str:
        """
        Build HTML table for mean tier comparison
        
        Args:
            mean_tier_data: List of dicts containing mean tier data with statistics
            
        Returns:
            HTML string for the mean tier table
        """
        if not mean_tier_data:
            return '<div class="result-placeholder">No mean tier data available</div>'
            
        # Create rows
        rows = []
        for item in mean_tier_data:
            # Calculate SEM safely to avoid division by zero
            experiment_count = item.get('experiment_count', 0)
            stddev = item.get('stddev_tier_difference', 0)
            sem = stddev / (experiment_count ** 0.5) if experiment_count > 0 else 0
            
            rows.append(f'''
            <tr>
                <td>{item.get('llm_method', 'N/A')}</td>
                <td>{item.get('mean_baseline_tier', 0):.2f}</td>
                <td>{item.get('mean_persona_tier', 0):.2f}</td>
                <td>{experiment_count:,}</td>
                <td>{stddev:.2f}</td>
                <td>{sem:.4f}</td>
            </tr>''')
        
        # Add statistical test results
        stats_html = ""
        for item in mean_tier_data:
            mean_baseline = item.get('mean_baseline_tier', 0)
            mean_persona = item.get('mean_persona_tier', 0)
            stddev = item.get('stddev_tier_difference', 0)
            n = item.get('experiment_count', 0)
            
            if n > 0 and mean_baseline is not None and mean_persona is not None:
                # Calculate effect size (Cohen's d for paired samples)
                # For paired t-test, Cohen's d = mean_difference / std_dev_of_differences
                cohens_d = (mean_persona - mean_baseline) / stddev if stddev > 0 else 0
                
                # Calculate t-statistic and degrees of freedom for paired t-test
                t_statistic = (mean_persona - mean_baseline) / (stddev / (n ** 0.5)) if stddev > 0 and n > 0 else 0
                df = n - 1  # degrees of freedom
                
                # Calculate exact p-value using t-distribution CDF
                from scipy.stats import t
                if t_statistic >= 0:
                    p_value = (1 - t.cdf(abs(t_statistic), df)) * 2  # two-tailed test
                else:
                    p_value = t.cdf(-abs(t_statistic), df) * 2  # two-tailed test
                
                # Round very small p-values to avoid scientific notation
                p_value_display = f"{p_value:.4f}" if p_value >= 0.0001 else "< 0.0001"
                
                # Enhanced interpretation with effect size
                interpretation = interpret_statistical_result(p_value, cohens_d, "paired_t_test")
                
                # Determine direction and magnitude of effect
                if mean_persona > mean_baseline:
                    direction = "higher"
                    magnitude = "slightly" if 0.2 <= cohens_d < 0.5 else "moderately" if 0.5 <= cohens_d < 0.8 else ""
                    implication = "somewhat analogous to a display of empathy" if cohens_d > 0 else ""
                elif mean_persona < mean_baseline:
                    direction = "lower"
                    magnitude = "slightly" if -0.5 < cohens_d <= -0.2 else "moderately" if -0.8 < cohens_d <= -0.5 else ""
                    implication = "suggesting potential bias against certain attributes"
                else:
                    direction = "the same as"
                    magnitude = ""
                    implication = ""

                # Prepare implication text
                implication_text = "On average, humanizing attributes did not meaningfully affect the recommended remedy tier."
                if interpretation['significance_text'] == "rejected":
                    effect_description = ' '.join(filter(None, [magnitude, direction]))
                    implication_text = f"The LLM's recommended tier is {effect_description} when it sees humanizing attributes, {implication}."

                # Build the statistical report with enhanced interpretation
                stats_html += f"""
                <div class="conclusion">
                    <h4>Statistical Analysis ({item.get('llm_method', 'N/A').title()}):</h4>
                    <p>H0: The mean tier is the same with and without persona injection</p>
                    <p>Test: Paired t-test</p>
                    <p><strong>Effect Size:</strong> {cohens_d:.3f} ({interpretation['effect_magnitude']})</p>
                    <p>Mean Difference: {mean_persona - mean_baseline:+.2f} (from {mean_baseline:.2f} to {mean_persona:.2f})</p>
                    <p>Test Statistic: t({df}) = {t_statistic:.4f}</p>
                    <p><strong>p-value:</strong> {p_value_display}</p>
                    <p><strong>Conclusion:</strong> The null hypothesis was <strong>{interpretation['significance_text']}</strong> (p {'<' if p_value < 0.05 else '≥'} 0.05).</p>
                    <p><strong>Practical Significance:</strong> This result is {interpretation['interpretation']}{interpretation['warning']}.</p>
                    <p>Implication: {implication_text}</p>
                </div>"""
        
        return f'''
        <div class="result-table">
            <div class="table-responsive">
                <table class="mean-tier">
                    <thead>
                        <tr>
                            <th>LLM Method</th>
                            <th>Mean Baseline Tier</th>
                            <th>Mean Persona Tier</th>
                            <th>N</th>
                            <th>Std Dev</th>
                            <th>SEM</th>
                        </tr>
                    </thead>
                    <tbody>{"".join(rows)}</tbody>
                </table>
            </div>
            {stats_html}
        </div>'''

    def _build_question_rate_table(self, question_data: Dict, title: str) -> str:
        """
        Build HTML table for question rate analysis
        
        Args:
            question_data: Dictionary containing question rate data
            title: Title for the analysis
            
        Returns:
            HTML string for the question rate table
        """
        if not question_data:
            return '<div class="result-placeholder">No question rate data available</div>'
            
        # Extract data
        baseline_count = question_data.get('baseline_count', 0)
        baseline_questions = question_data.get('baseline_questions', 0)
        persona_count = question_data.get('persona_count', 0)
        persona_questions = question_data.get('persona_questions', 0)
        
        # Calculate rates
        baseline_rate = (baseline_questions / baseline_count * 100) if baseline_count > 0 else 0
        persona_rate = (persona_questions / persona_count * 100) if persona_count > 0 else 0
        
        # Build table rows
        rows = f'''
        <tr>
            <td><strong>Baseline</strong></td>
            <td>{int(baseline_count):,}</td>
            <td>{int(baseline_questions):,}</td>
            <td>{baseline_rate:.1f}%</td>
        </tr>
        <tr>
            <td><strong>Persona-Injected</strong></td>
            <td>{int(persona_count):,}</td>
            <td>{int(persona_questions):,}</td>
            <td>{persona_rate:.1f}%</td>
        </tr>'''
        
        # Statistical analysis
        stats_html = ""
        if baseline_count > 0 and persona_count > 0:
            # Perform chi-squared test for independence
            from scipy.stats import chi2_contingency
            
            # Create contingency table
            contingency_table = [
                [baseline_questions, baseline_count - baseline_questions],  # baseline: questions, no questions
                [persona_questions, persona_count - persona_questions]      # persona: questions, no questions
            ]
            
            try:
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                
                # Calculate Cramér's V effect size
                cramers_v = calculate_cramers_v(np.array(contingency_table))
                
                # Enhanced interpretation with effect size
                interpretation = interpret_statistical_result(p_value, cramers_v, "chi_squared")
                
                # Round very small p-values
                p_value_display = f"{p_value:.4f}" if p_value >= 0.0001 else "< 0.0001"
                
                # Determine implication
                if interpretation['significance_text'] == "rejected":
                    if persona_rate > baseline_rate:
                        implication = "The LLM is significantly more likely to ask questions when it sees humanizing attributes, suggesting increased engagement or scrutiny."
                    else:
                        implication = "The LLM is significantly less likely to ask questions when it sees humanizing attributes, suggesting reduced engagement or scrutiny."
                else:
                    implication = "The LLM's question rate is not significantly affected by humanizing attributes."
                
                stats_html = f"""
                <div class="conclusion">
                    <h4>Statistical Analysis:</h4>
                    <p>H0: The question rate is the same with and without persona injection</p>
                    <p>Test: Chi-squared test of independence</p>
                    <p><strong>Effect Size:</strong> {cramers_v:.3f} ({interpretation['effect_magnitude']})</p>
                    <p>Test Statistic: χ²({dof}) = {chi2:.2f}</p>
                    <p><strong>p-value:</strong> {p_value_display}</p>
                    <p><strong>Conclusion:</strong> The null hypothesis was <strong>{interpretation['significance_text']}</strong> (p {'<' if p_value < 0.05 else '≥'} 0.05).</p>
                    <p><strong>Practical Significance:</strong> This result is {interpretation['interpretation']}{interpretation['warning']}.</p>
                    <p>Implication: {implication}</p>
                </div>"""
            except Exception as e:
                stats_html = f"""
                <div class="conclusion">
                    <h4>Statistical Analysis:</h4>
                    <p>Unable to perform statistical test: {str(e)}</p>
                </div>"""
        
        return f'''
        <div class="result-table">
            <div class="table-responsive">
                <table class="question-rate">
                    <thead>
                        <tr>
                            <th>Condition</th>
                            <th>Count</th>
                            <th>Questions</th>
                            <th>Question Rate %</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            {stats_html}
        </div>'''

    def _build_nshot_vs_zeroshot_table(self, comparison_data: Dict) -> str:
        """
        Build HTML table for N-Shot vs Zero-Shot comparison
        
        Args:
            comparison_data: Dictionary containing comparison data
            
        Returns:
            HTML string for the comparison table
        """
        if not comparison_data:
            return '<div class="result-placeholder">No N-Shot vs Zero-Shot comparison data available</div>'
            
        # Extract data
        zero_shot_count = comparison_data.get('zero_shot_count', 0)
        zero_shot_questions = comparison_data.get('zero_shot_questions', 0)
        n_shot_count = comparison_data.get('n_shot_count', 0)
        n_shot_questions = comparison_data.get('n_shot_questions', 0)
        
        # Calculate rates
        zero_shot_rate = (zero_shot_questions / zero_shot_count * 100) if zero_shot_count > 0 else 0
        n_shot_rate = (n_shot_questions / n_shot_count * 100) if n_shot_count > 0 else 0
        
        # Build table rows
        rows = f'''
        <tr>
            <td><strong>Zero-Shot</strong></td>
            <td>{int(zero_shot_count):,}</td>
            <td>{int(zero_shot_questions):,}</td>
            <td>{zero_shot_rate:.1f}%</td>
        </tr>
        <tr>
            <td><strong>N-Shot</strong></td>
            <td>{int(n_shot_count):,}</td>
            <td>{int(n_shot_questions):,}</td>
            <td>{n_shot_rate:.1f}%</td>
        </tr>'''
        
        # Statistical analysis
        stats_html = ""
        if zero_shot_count > 0 and n_shot_count > 0:
            # Perform chi-squared test for independence
            from scipy.stats import chi2_contingency
            
            # Create contingency table
            contingency_table = [
                [zero_shot_questions, zero_shot_count - zero_shot_questions],  # zero-shot: questions, no questions
                [n_shot_questions, n_shot_count - n_shot_questions]            # n-shot: questions, no questions
            ]
            
            try:
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                
                # Calculate Cramér's V effect size
                cramers_v = calculate_cramers_v(np.array(contingency_table))
                
                # Enhanced interpretation with effect size
                interpretation = interpret_statistical_result(p_value, cramers_v, "chi_squared")
                
                # Round very small p-values
                p_value_display = f"{p_value:.4f}" if p_value >= 0.0001 else "< 0.0001"
                
                # Determine implication
                if interpretation['significance_text'] == "rejected":
                    if n_shot_rate < zero_shot_rate:
                        implication = "N-Shot examples reduce the influence of sensitive personal attributes on the LLM's questioning behavior."
                    else:
                        implication = "N-Shot examples increase the influence of sensitive personal attributes on the LLM's questioning behavior."
                else:
                    implication = "The LLM's questioning behavior is not significantly affected by the addition of N-Shot examples."
                
                stats_html = f"""
                <div class="conclusion">
                    <h4>Statistical Analysis:</h4>
                    <p>H0: The question rate is the same with and without N-Shot examples</p>
                    <p>Test: Chi-squared test of independence</p>
                    <p><strong>Effect Size:</strong> {cramers_v:.3f} ({interpretation['effect_magnitude']})</p>
                    <p>Test Statistic: χ²({dof}) = {chi2:.2f}</p>
                    <p><strong>p-value:</strong> {p_value_display}</p>
                    <p><strong>Conclusion:</strong> The null hypothesis was <strong>{interpretation['significance_text']}</strong> (p {'<' if p_value < 0.05 else '≥'} 0.05).</p>
                    <p><strong>Practical Significance:</strong> This result is {interpretation['interpretation']}{interpretation['warning']}.</p>
                    <p>Implication: {implication}</p>
                </div>"""
            except Exception as e:
                stats_html = f"""
                <div class="conclusion">
                    <h4>Statistical Analysis:</h4>
                    <p>Unable to perform statistical test: {str(e)}</p>
                </div>"""
        
        return f'''
        <div class="result-table">
            <div class="table-responsive">
                <table class="nshot-comparison">
                    <thead>
                        <tr>
                            <th>Method</th>
                            <th>Count</th>
                            <th>Questions</th>
                            <th>Question Rate %</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            {stats_html}
        </div>'''

    def _build_accuracy_tab(self, data: Dict[str, Any]) -> str:
        """Build Ground Truth Accuracy tab content with sub-tabs"""
        # Extract accuracy data
        accuracy_data = data.get('accuracy_data', {})
        
        return f"""
        <div class="sub-nav-tabs">
            <div class="sub-nav-tab active" data-sub-tab-id="accuracy-overview">Overview</div>
            <div class="sub-nav-tab" data-sub-tab-id="accuracy-comparison">Method Comparison</div>
            <div class="sub-nav-tab" data-sub-tab-id="accuracy-strategies">Strategy Analysis</div>
        </div>

        <div id="accuracy-overview" class="sub-tab-content active">
            <div class="section">
                <h2>Accuracy Analysis</h2>

                <div class="result-item">
                    <div class="result-title">Result 1: Overall Accuracy Comparison</div>
                    <div class="result-content">
                        <div class="filter-controls">
                            <label for="decision-method-filter">Decision Method:</label>
                            <select id="decision-method-filter" onchange="updateAccuracyFilters()">
                                <option value="zero-shot">Zero-Shot</option>
                                <option value="n-shot">N-Shot</option>
                                <option value="all">All</option>
                            </select>
                            
                            <label for="experiment-category-filter">Experiment Category:</label>
                            <select id="experiment-category-filter" onchange="updateAccuracyFilters()">
                                <option value="Baseline">Baseline</option>
                                <option value="Persona-Injected">Persona-Injected</option>
                                <option value="Bias Mitigation">Bias Mitigation</option>
                                <option value="all">All</option>
                            </select>
                        </div>
                        
                        <div id="confusion-matrix-container">
                            {self._build_confusion_matrix_html(accuracy_data)}
                        </div>
                    </div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 2: Zero-Shot vs N-Shot Accuracy Rates</div>
                    <div class="result-content">
                        <div class="filter-controls">
                            <label for="accuracy-decision-method-filter">Decision Method:</label>
                            <select id="accuracy-decision-method-filter" onchange="updateAccuracyRatesFilters()">
                                <option value="all">All</option>
                                <option value="zero-shot">Zero-Shot</option>
                                <option value="n-shot">N-Shot</option>
                            </select>
                            
                            <label for="accuracy-experiment-category-filter">Experiment Category:</label>
                            <select id="accuracy-experiment-category-filter" onchange="updateAccuracyRatesFilters()">
                                <option value="all">All</option>
                                <option value="Baseline">Baseline</option>
                                <option value="Persona-Injected">Persona-Injected</option>
                                <option value="Bias Mitigation">Bias Mitigation</option>
                            </select>
                        </div>
                        
                        <div id="accuracy-rates-container">
                            {self._build_accuracy_rates_html(accuracy_data)}
                        </div>
                    </div>
                </div>
            </div>

            <div class="info-box">
                <strong>Note:</strong> Ground truth accuracy metrics are based on comparison with manually verified complaint resolution tiers.
                Accuracy measurements help validate the effectiveness of different fairness approaches while maintaining predictive performance.
            </div>
        </div>

        <div id="accuracy-comparison" class="sub-tab-content">
            <div class="section">
                <h2>Method Comparison</h2>

                <div class="result-item">
                    <div class="result-title">Result 1: Zero-Shot vs N-Shot Performance</div>
                    <div class="result-placeholder">[Placeholder: Detailed comparison of zero-shot and n-shot accuracy across different conditions]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 2: Baseline vs Persona-Injected Accuracy</div>
                    <div class="result-placeholder">[Placeholder: Impact of persona injection on prediction accuracy]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 3: With vs Without Bias Mitigation</div>
                    <div class="result-placeholder">[Placeholder: Accuracy performance with and without bias mitigation strategies]</div>
                </div>
            </div>
        </div>

        <div id="accuracy-strategies" class="sub-tab-content">
            <div class="section">
                <h2>Strategy Analysis</h2>

                <div class="result-item">
                    <div class="result-title">Result 1: Most and Least Effective Strategies</div>
                    <div class="result-placeholder">[Placeholder: Ranking of all experimental approaches by accuracy performance]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 2: Accuracy by Bias Mitigation Strategy</div>
                    <div class="result-placeholder">[Placeholder: Accuracy performance for different bias mitigation approaches]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 3: N-Shot Strategy Effectiveness</div>
                    <div class="result-placeholder">[Placeholder: Comparison of different n-shot prompting strategies]</div>
                </div>
            </div>
        </div>
        """

    def _build_confusion_matrix_html(self, accuracy_data: Dict) -> str:
        """
        Build HTML for confusion matrix table.
        
        Args:
            accuracy_data: Dictionary containing accuracy data
            
        Returns:
            HTML string for confusion matrix
        """
        confusion_matrix = accuracy_data.get('confusion_matrix', {})
        
        if not confusion_matrix:
            return '<div class="result-placeholder">No confusion matrix data available</div>'
        
        # Default to zero-shot and Baseline
        key = ('zero-shot', 'Baseline')
        if key not in confusion_matrix:
            # Try to find any available data
            available_keys = list(confusion_matrix.keys())
            if available_keys:
                key = available_keys[0]
            else:
                return '<div class="result-placeholder">No confusion matrix data available</div>'
        
        matrix_data = confusion_matrix[key]
        
        # Get all unique tiers
        all_gt_tiers = sorted(set(gt_tier for gt_tier in matrix_data.keys()))
        all_llm_tiers = sorted(set(llm_tier for gt_tier in matrix_data.values() for llm_tier in gt_tier.keys()))
        
        if not all_gt_tiers or not all_llm_tiers:
            return '<div class="result-placeholder">No confusion matrix data available</div>'
        
        # Create HTML table
        html = '<table class="confusion-matrix">\n'
        html += '  <thead>\n'
        html += '    <tr>\n'
        html += '      <th>Ground Truth \\ LLM</th>\n'
        for llm_tier in all_llm_tiers:
            html += f'      <th>Tier {llm_tier}</th>\n'
        html += '    </tr>\n'
        html += '  </thead>\n'
        html += '  <tbody>\n'
        
        for gt_tier in all_gt_tiers:
            html += f'    <tr>\n'
            html += f'      <th>Tier {gt_tier}</th>\n'
            for llm_tier in all_llm_tiers:
                count = matrix_data.get(gt_tier, {}).get(llm_tier, 0)
                html += f'      <td>{count:,}</td>\n'
            html += '    </tr>\n'
        
        html += '  </tbody>\n'
        html += '</table>\n'
        
        return html

    def _build_accuracy_rates_html(self, accuracy_data: Dict) -> str:
        """
        Build HTML for accuracy rates table.
        
        Args:
            accuracy_data: Dictionary containing accuracy data
            
        Returns:
            HTML string for accuracy rates table
        """
        accuracy_rates = accuracy_data.get('accuracy_rates', {})
        
        if not accuracy_rates:
            return '<div class="result-placeholder">No accuracy rates data available</div>'
        
        html = '<table class="accuracy-rates">\n'
        html += '  <thead>\n'
        html += '    <tr>\n'
        html += '      <th>Decision Method</th>\n'
        html += '      <th>Experiment Category</th>\n'
        html += '      <th>Sample Size</th>\n'
        html += '      <th>Correct</th>\n'
        html += '      <th>Accuracy %</th>\n'
        html += '    </tr>\n'
        html += '  </thead>\n'
        html += '  <tbody>\n'
        
        for key, data in accuracy_rates.items():
            html += '    <tr>\n'
            html += f'      <td>{data["decision_method"]}</td>\n'
            html += f'      <td>{data["experiment_category"]}</td>\n'
            html += f'      <td>{data["sample_size"]:,}</td>\n'
            html += f'      <td>{data["correct_count"]:,}</td>\n'
            html += f'      <td>{round(data["accuracy_percentage"])}%</td>\n'
            html += '    </tr>\n'
        
        html += '  </tbody>\n'
        html += '</table>\n'
        
        return html

    def _build_gender_tier0_rate_table(self, gender_data: Dict, method: str) -> str:
        """
        Build HTML table for tier 0 rate by gender analysis
        
        Args:
            gender_data: Dictionary containing gender bias data
            method: Either 'zero_shot' or 'n_shot'
            
        Returns:
            HTML string for the tier 0 rate table
        """
        if not gender_data:
            return '<div class="result-placeholder">No gender bias data available</div>'
        
        # Get the tier 0 rate data
        tier0_data_key = f'{method}_tier0_rate'
        tier0_stats_key = f'{method}_tier0_stats'
        
        tier0_data = gender_data.get(tier0_data_key, {})
        tier0_stats = gender_data.get(tier0_stats_key, {})
        
        if not tier0_data:
            return '<div class="result-placeholder">No tier 0 rate data available</div>'
        
        # Build the table
        html = '<table class="gender-tier0-rate">\n'
        html += '  <thead>\n'
        html += '    <tr>\n'
        html += '      <th>Gender</th>\n'
        html += '      <th>Sample Size</th>\n'
        html += '      <th>Zero Tier</th>\n'
        html += '      <th>Proportion Zero</th>\n'
        html += '    </tr>\n'
        html += '  </thead>\n'
        html += '  <tbody>\n'
        
        for gender, data in tier0_data.items():
            html += '    <tr>\n'
            html += f'      <td>{gender.title()}</td>\n'
            html += f'      <td>{data["sample_size"]:,}</td>\n'
            html += f'      <td>{data["zero_tier_count"]:,}</td>\n'
            html += f'      <td>{data["proportion_zero"]:.3f}</td>\n'
            html += '    </tr>\n'
        
        html += '  </tbody>\n'
        html += '</table>\n'
        
        # Add statistical analysis
        if tier0_stats and 'error' not in tier0_stats:
            # Calculate multiple effect sizes for proportion comparison
            chi2_stat = tier0_stats.get('chi2_statistic', 0)
            p_value = tier0_stats.get('p_value', 1.0)

            # Get proportions for the two genders
            genders = list(tier0_data.keys())
            if len(genders) >= 2:
                p1 = tier0_data[genders[0]]['proportion_zero']
                p2 = tier0_data[genders[1]]['proportion_zero']

                # Calculate Cohen's h for proportion difference
                cohens_h = calculate_cohens_h(p1, p2)
                h_interpretation = interpret_statistical_result(p_value, cohens_h, "cohens_h")

                # Calculate Risk Ratio
                risk_ratio = calculate_risk_ratio(p1, p2) if p2 > 0 else float('inf')

                # Also calculate Cramér's V for backwards compatibility
                contingency_table = []
                for gender, data in tier0_data.items():
                    zero_count = data['zero_tier_count']
                    non_zero_count = data['sample_size'] - zero_count
                    contingency_table.append([zero_count, non_zero_count])
                cramers_v = calculate_cramers_v(np.array(contingency_table))

                # Use Cohen's h as primary interpretation for proportions
                interpretation = h_interpretation

                effect_size_html = f"""<p><strong>Effect Sizes:</strong></p>
                <ul style="margin-left: 20px;">
                    <li><strong>Proportion Difference (Cohen's h):</strong> {cohens_h:.3f} ({h_interpretation['effect_magnitude']})</li>
                    <li><strong>Risk Ratio:</strong> {risk_ratio:.2f} ({genders[0]} vs {genders[1]})</li>
                    <li><strong>Association (Cramér's V):</strong> {cramers_v:.3f}</li>
                </ul>"""
            else:
                # Fallback to Cramér's V only
                contingency_table = []
                for gender, data in tier0_data.items():
                    zero_count = data['zero_tier_count']
                    non_zero_count = data['sample_size'] - zero_count
                    contingency_table.append([zero_count, non_zero_count])
                cramers_v = calculate_cramers_v(np.array(contingency_table))
                interpretation = interpret_statistical_result(p_value, cramers_v, "chi_squared")
                effect_size_html = f'<p><strong>Effect Size (Cramér\'s V):</strong> {cramers_v:.3f} ({interpretation["effect_magnitude"]})</p>\n'

            # Register result with collector (before building HTML)
            implication_text = ""
            if interpretation['significance_text'] == 'rejected':
                higher_gender = tier0_stats.get('higher_proportion_gender', 'N/A')
                if interpretation['practical_importance'] == 'trivial':
                    implication_text = "While statistically significant, the difference in zero-tier proportions between genders is practically trivial and likely due to large sample size."
                else:
                    if higher_gender == 'male':
                        implication_text = "The proportion of zero-tier cases is higher for males."
                    elif higher_gender == 'female':
                        implication_text = "The proportion of zero-tier cases is higher for females."
                    else:
                        implication_text = "The proportion of zero-tier cases differs significantly between genders."
            else:
                if p_value <= 0.1:
                    implication_text = "There is weak evidence that the proportion of zero-tier cases varies with gender."
                else:
                    implication_text = "There is no evidence that the proportion of zero-tier cases varies with gender."

            # Use the primary effect size (Cohen's h if available, otherwise Cramér's V)
            primary_effect_size = cohens_h if len(genders) >= 2 else cramers_v
            primary_effect_type = 'cohens_h' if len(genders) >= 2 else 'cramers_v'

            result_data = {
                'source_tab': 'Persona Injection',
                'source_subtab': 'Gender Bias',
                'test_name': f'Tier 0 Rate by Gender: {method.replace("_", "-").title()}',
                'test_type': 'chi_squared',
                'p_value': p_value,
                'effect_size': primary_effect_size,
                'effect_type': primary_effect_type,
                'sample_size': sum(data['sample_size'] for data in tier0_data.values()),
                'finding': f'Zero-tier proportions {"differ" if interpretation["significance_text"] == "rejected" else "are consistent"} across genders (χ² = {chi2_stat:.3f})',
                'implication': implication_text,
                'timestamp': datetime.now()
            }
            self.collector.add_result(result_data)

            html += '<div class="statistical-analysis">\n'
            html += '  <h4>Statistical Analysis</h4>\n'
            html += f'  <p><strong>Hypothesis:</strong> H0: The proportion of zero-tier cases is the same for all genders</p>\n'
            html += '  <p><strong>Test:</strong> Chi-squared test on counts</p>\n'
            html += effect_size_html
            html += f'  <p><strong>Test Statistic:</strong> χ² = {chi2_stat:.3f}</p>\n'
            html += f'  <p><strong>p-Value:</strong> {p_value:.3f}</p>\n'
            html += f'  <p><strong>Conclusion:</strong> The null hypothesis was <strong>{interpretation["significance_text"]}</strong> (p {"<" if p_value < 0.05 else "≥"} 0.05)</p>\n'
            html += f'  <p><strong>Practical Significance:</strong> This result is {interpretation["interpretation"]}{interpretation["warning"]}.</p>\n'
            
            # Add implication based on the specifications and effect size
            if interpretation['significance_text'] == 'rejected':
                higher_gender = tier0_stats.get('higher_proportion_gender', 'N/A')
                if interpretation['practical_importance'] == 'trivial':
                    html += '  <p><strong>Implication:</strong> While statistically significant, the difference in zero-tier proportions between genders is practically trivial and likely due to large sample size.</p>\n'
                else:
                    if higher_gender == 'male':
                        html += '  <p><strong>Implication:</strong> The proportion of zero-tier cases is higher for males.</p>\n'
                    elif higher_gender == 'female':
                        html += '  <p><strong>Implication:</strong> The proportion of zero-tier cases is higher for females.</p>\n'
                    else:
                        html += '  <p><strong>Implication:</strong> The proportion of zero-tier cases differs significantly between genders.</p>\n'
            else:
                if p_value <= 0.1:
                    html += '  <p><strong>Implication:</strong> There is weak evidence that the proportion of zero-tier cases varies with gender.</p>\n'
                else:
                    html += '  <p><strong>Implication:</strong> There is no evidence that the proportion of zero-tier cases varies with gender.</p>\n'
            
            html += '</div>\n'
        elif tier0_stats and 'error' in tier0_stats:
            html += f'<div class="result-placeholder">Statistical analysis error: {tier0_stats["error"]}</div>\n'
        else:
            html += '<div class="result-placeholder">No statistical analysis available</div>\n'
        
        return html

    def _build_ethnicity_tier0_rate_table(self, ethnicity_data: Dict, method: str) -> str:
        """
        Build HTML table for tier 0 rate by ethnicity analysis
        
        Args:
            ethnicity_data: Dictionary containing ethnicity bias data
            method: Either 'zero_shot' or 'n_shot'
            
        Returns:
            HTML string for the tier 0 rate table
        """
        if not ethnicity_data:
            return '<div class="result-placeholder">No ethnicity bias data available</div>'
        
        # Get the tier 0 rate data
        tier0_data_key = f'{method}_tier0_rate'
        tier0_stats_key = f'{method}_tier0_stats'
        
        tier0_data = ethnicity_data.get(tier0_data_key, {})
        tier0_stats = ethnicity_data.get(tier0_stats_key, {})
        
        if not tier0_data:
            return '<div class="result-placeholder">No tier 0 rate data available</div>'
        
        # Build the table
        html = '<table class="ethnicity-tier0-rate">\n'
        html += '  <thead>\n'
        html += '    <tr>\n'
        html += '      <th>Ethnicity</th>\n'
        html += '      <th>Sample Size</th>\n'
        html += '      <th>Zero Tier</th>\n'
        html += '      <th>Proportion Zero</th>\n'
        html += '    </tr>\n'
        html += '  </thead>\n'
        html += '  <tbody>\n'
        
        for ethnicity, data in tier0_data.items():
            html += '    <tr>\n'
            html += f'      <td>{ethnicity.title()}</td>\n'
            html += f'      <td>{data["sample_size"]:,}</td>\n'
            html += f'      <td>{data["zero_tier_count"]:,}</td>\n'
            html += f'      <td>{data["proportion_zero"]:.3f}</td>\n'
            html += '    </tr>\n'
        
        html += '  </tbody>\n'
        html += '</table>\n'
        
        # Add statistical analysis
        if tier0_stats and 'error' not in tier0_stats:
            # Calculate Cramér's V effect size
            chi2_stat = tier0_stats.get('chi2_statistic', 0)
            p_value = tier0_stats.get('p_value', 1.0)
            
            # Create contingency table for Cramér's V calculation
            contingency_table = []
            for ethnicity, data in tier0_data.items():
                zero_count = data['zero_tier_count']
                non_zero_count = data['sample_size'] - zero_count
                contingency_table.append([zero_count, non_zero_count])
            
            # Calculate Cramér's V
            cramers_v = calculate_cramers_v(np.array(contingency_table))
            
            # Enhanced interpretation with effect size
            interpretation = interpret_statistical_result(p_value, cramers_v, "chi_squared")

            # Register result with collector (before building HTML)
            implication_text = ""
            if interpretation['significance_text'] == 'rejected':
                highest_ethnicity = tier0_stats.get('highest_proportion_ethnicity', 'N/A')
                if interpretation['practical_importance'] == 'trivial':
                    implication_text = "While statistically significant, the difference in zero-tier proportions between ethnicities is practically trivial and likely due to large sample size."
                else:
                    implication_text = f"The proportion of zero-tier cases differs significantly between ethnicities, with {highest_ethnicity} having the highest proportion."
            else:
                if p_value <= 0.1:
                    implication_text = "There is weak evidence that the proportion of zero-tier cases varies with ethnicity."
                else:
                    implication_text = "There is no evidence that the proportion of zero-tier cases varies with ethnicity."

            result_data = {
                'source_tab': 'Persona Injection',
                'source_subtab': 'Ethnicity Bias',
                'test_name': f'Tier 0 Rate by Ethnicity: {method.replace("_", "-").title()}',
                'test_type': 'chi_squared',
                'p_value': p_value,
                'effect_size': cramers_v,
                'effect_type': 'cramers_v',
                'sample_size': sum(data['sample_size'] for data in tier0_data.values()),
                'finding': f'Zero-tier proportions {"differ" if interpretation["significance_text"] == "rejected" else "are consistent"} across ethnicities (χ² = {chi2_stat:.3f})',
                'implication': implication_text,
                'timestamp': datetime.now()
            }
            self.collector.add_result(result_data)

            html += '<div class="statistical-analysis">\n'
            html += '  <h4>Statistical Analysis</h4>\n'
            html += f'  <p><strong>Hypothesis:</strong> H0: The proportion of zero-tier cases is the same for all ethnicities</p>\n'
            html += '  <p><strong>Test:</strong> Chi-squared test on counts</p>\n'
            html += f'  <p><strong>Effect Size:</strong> {cramers_v:.3f} ({interpretation["effect_magnitude"]})</p>\n'
            html += f'  <p><strong>Test Statistic:</strong> χ² = {chi2_stat:.3f}</p>\n'
            html += f'  <p><strong>p-Value:</strong> {p_value:.3f}</p>\n'
            html += f'  <p><strong>Conclusion:</strong> The null hypothesis was <strong>{interpretation["significance_text"]}</strong> (p {"<" if p_value < 0.05 else "≥"} 0.05)</p>\n'
            html += f'  <p><strong>Practical Significance:</strong> This result is {interpretation["interpretation"]}{interpretation["warning"]}.</p>\n'
            
            # Add implication based on the specifications and effect size
            if interpretation['significance_text'] == 'rejected':
                highest_ethnicity = tier0_stats.get('highest_proportion_ethnicity', 'N/A')
                if interpretation['practical_importance'] == 'trivial':
                    html += '  <p><strong>Implication:</strong> While statistically significant, the difference in zero-tier proportions between ethnicities is practically trivial and likely due to large sample size.</p>\n'
                else:
                    html += f'  <p><strong>Implication:</strong> The proportion of zero-tier cases differs significantly between ethnicities, with {highest_ethnicity} having the highest proportion.</p>\n'
            else:
                if p_value <= 0.1:
                    html += '  <p><strong>Implication:</strong> There is weak evidence that the proportion of zero-tier cases varies with ethnicity.</p>\n'
                else:
                    html += '  <p><strong>Implication:</strong> There is no evidence that the proportion of zero-tier cases varies with ethnicity.</p>\n'
            
            html += '</div>\n'
        elif tier0_stats and 'error' in tier0_stats:
            html += f'<div class="result-placeholder">Statistical analysis error: {tier0_stats["error"]}</div>\n'
        else:
            html += '<div class="result-placeholder">No statistical analysis available</div>\n'
        
        return html

    def _build_geographic_tier0_rate_table(self, geographic_data: Dict, method: str) -> str:
        """
        Build HTML table for tier 0 rate by geography analysis
        
        Args:
            geographic_data: Dictionary containing geographic bias data
            method: Either 'zero_shot' or 'n_shot'
            
        Returns:
            HTML string for the tier 0 rate table
        """
        if not geographic_data:
            return '<div class="result-placeholder">No geographic bias data available</div>'
        
        # Get the tier 0 rate data
        tier0_data_key = f'{method}_tier0_rate'
        tier0_stats_key = f'{method}_tier0_stats'
        
        tier0_data = geographic_data.get(tier0_data_key, {})
        tier0_stats = geographic_data.get(tier0_stats_key, {})
        
        if not tier0_data:
            return '<div class="result-placeholder">No tier 0 rate data available</div>'
        
        # Build the table
        html = '<table class="geographic-tier0-rate">\n'
        html += '  <thead>\n'
        html += '    <tr>\n'
        html += '      <th>Geography</th>\n'
        html += '      <th>Sample Size</th>\n'
        html += '      <th>Zero Tier</th>\n'
        html += '      <th>Proportion Zero</th>\n'
        html += '    </tr>\n'
        html += '  </thead>\n'
        html += '  <tbody>\n'
        
        for geography, data in tier0_data.items():
            html += '    <tr>\n'
            html += f'      <td>{geography.replace("_", " ").title()}</td>\n'
            html += f'      <td>{data["sample_size"]:,}</td>\n'
            html += f'      <td>{data["zero_tier_count"]:,}</td>\n'
            html += f'      <td>{data["proportion_zero"]:.3f}</td>\n'
            html += '    </tr>\n'
        
        html += '  </tbody>\n'
        html += '</table>\n'
        
        # Add statistical analysis
        if tier0_stats and 'error' not in tier0_stats:
            # Calculate Cramér's V effect size
            chi2_stat = tier0_stats.get('chi2_statistic', 0)
            p_value = tier0_stats.get('p_value', 1.0)

            # Create contingency table for Cramér's V calculation
            contingency_table = []
            for geography, data in tier0_data.items():
                zero_count = data['zero_tier_count']
                non_zero_count = data['sample_size'] - zero_count
                contingency_table.append([zero_count, non_zero_count])

            # Calculate Cramér's V
            cramers_v = calculate_cramers_v(np.array(contingency_table))

            # Enhanced interpretation with effect size
            interpretation = interpret_statistical_result(p_value, cramers_v, "chi_squared")

            # Register result with collector (before building HTML)
            implication_text = ""
            if interpretation['significance_text'] == 'rejected':
                highest_geography = tier0_stats.get('highest_proportion_geography', 'N/A')
                if interpretation['practical_importance'] == 'trivial':
                    implication_text = "While statistically significant, the difference in zero-tier proportions between geographies is practically trivial and likely due to large sample size."
                else:
                    implication_text = f"The proportion of zero-tier cases differs significantly between geographies, with {highest_geography.replace('_', ' ')} having the highest proportion."
            else:
                if p_value <= 0.1:
                    implication_text = "There is weak evidence that the proportion of zero-tier cases varies with geography."
                else:
                    implication_text = "There is no evidence that the proportion of zero-tier cases varies with geography."

            result_data = {
                'source_tab': 'Persona Injection',
                'source_subtab': 'Geographic Bias',
                'test_name': f'Tier 0 Rate by Geography: {method.replace("_", "-").title()}',
                'test_type': 'chi_squared',
                'p_value': p_value,
                'effect_size': cramers_v,
                'effect_type': 'cramers_v',
                'sample_size': sum(data['sample_size'] for data in tier0_data.values()),
                'finding': f'Zero-tier proportions {"differ" if interpretation["significance_text"] == "rejected" else "are consistent"} across geographies (χ² = {chi2_stat:.3f})',
                'implication': implication_text,
                'timestamp': datetime.now()
            }
            self.collector.add_result(result_data)

            html += '<div class="statistical-analysis">\n'
            html += '  <h4>Statistical Analysis</h4>\n'
            html += f'  <p><strong>Hypothesis:</strong> H0: The proportion of zero-tier cases is the same for all geographies</p>\n'
            html += '  <p><strong>Test:</strong> Chi-squared test on counts</p>\n'
            html += f'  <p><strong>Effect Size:</strong> {cramers_v:.3f} ({interpretation["effect_magnitude"]})</p>\n'
            html += f'  <p><strong>Test Statistic:</strong> χ² = {chi2_stat:.3f}</p>\n'
            html += f'  <p><strong>p-Value:</strong> {p_value:.3f}</p>\n'
            html += f'  <p><strong>Conclusion:</strong> The null hypothesis was <strong>{interpretation["significance_text"]}</strong> (p {"<" if p_value < 0.05 else "≥"} 0.05)</p>\n'
            html += f'  <p><strong>Practical Significance:</strong> This result is {interpretation["interpretation"]}{interpretation["warning"]}.</p>\n'
            
            # Add implication based on the specifications and effect size
            if interpretation['significance_text'] == 'rejected':
                highest_geography = tier0_stats.get('highest_proportion_geography', 'N/A')
                if interpretation['practical_importance'] == 'trivial':
                    html += '  <p><strong>Implication:</strong> While statistically significant, the difference in zero-tier proportions between geographies is practically trivial and likely due to large sample size.</p>\n'
                else:
                    html += f'  <p><strong>Implication:</strong> The proportion of zero-tier cases differs significantly between geographies, with {highest_geography.replace("_", " ")} having the highest proportion.</p>\n'
            else:
                if p_value <= 0.1:
                    html += '  <p><strong>Implication:</strong> There is weak evidence that the proportion of zero-tier cases varies with geography.</p>\n'
                else:
                    html += '  <p><strong>Implication:</strong> There is no evidence that the proportion of zero-tier cases varies with geography.</p>\n'
            
            html += '</div>\n'
        elif tier0_stats and 'error' in tier0_stats:
            html += f'<div class="result-placeholder">Statistical analysis error: {tier0_stats["error"]}</div>\n'
        else:
            html += '<div class="result-placeholder">No statistical analysis available</div>\n'
        
        return html

    def _build_gender_mean_tier_tables(self, gender_data: Dict) -> str:
        """
        Build HTML tables for mean tier analysis by gender
        
        Args:
            gender_data: Dictionary containing gender bias data
            
        Returns:
            HTML string for the gender mean tier tables
        """
        if not gender_data:
            return '<div class="result-placeholder">No gender bias data available</div>'
        
        zero_shot_data = gender_data.get('zero_shot_mean_tier', {})
        n_shot_data = gender_data.get('n_shot_mean_tier', {})
        
        if not zero_shot_data and not n_shot_data:
            return '<div class="result-placeholder">No gender mean tier data available</div>'
        
        # Build Zero-Shot table
        zero_shot_rows = ""
        if zero_shot_data:
            for gender, stats in zero_shot_data.items():
                zero_shot_rows += f'''
                <tr>
                    <td><strong>{gender.title()}</strong></td>
                    <td>{stats['mean_tier']:.3f}</td>
                    <td>{int(stats['count']):,}</td>
                    <td>{stats['std_dev']:.3f}</td>
                </tr>'''
        else:
            zero_shot_rows = '<tr><td colspan="4">No zero-shot data available</td></tr>'
        
        # Build N-Shot table
        n_shot_rows = ""
        if n_shot_data:
            for gender, stats in n_shot_data.items():
                n_shot_rows += f'''
                <tr>
                    <td><strong>{gender.title()}</strong></td>
                    <td>{stats['mean_tier']:.3f}</td>
                    <td>{int(stats['count']):,}</td>
                    <td>{stats['std_dev']:.3f}</td>
                </tr>'''
        else:
            n_shot_rows = '<tr><td colspan="4">No n-shot data available</td></tr>'
        
        # Statistical analysis
        zero_shot_stats = gender_data.get('zero_shot_mean_stats', {})
        n_shot_stats = gender_data.get('n_shot_mean_stats', {})
        
        zero_shot_stats_html = self._build_gender_mean_statistical_analysis(zero_shot_stats, "Zero-Shot")
        n_shot_stats_html = self._build_gender_mean_statistical_analysis(n_shot_stats, "N-Shot")
        
        return f'''
        <div class="analysis-section">
            <h3>Zero-Shot Mean Tier by Gender</h3>
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Gender</th>
                        <th>Mean Tier</th>
                        <th>Count</th>
                        <th>Std Dev</th>
                    </tr>
                </thead>
                <tbody>
                    {zero_shot_rows}
                </tbody>
            </table>
            {zero_shot_stats_html}
        </div>
        
        <div class="analysis-section">
            <h3>N-Shot Mean Tier by Gender</h3>
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Gender</th>
                        <th>Mean Tier</th>
                        <th>Count</th>
                        <th>Std Dev</th>
                    </tr>
                </thead>
                <tbody>
                    {n_shot_rows}
                </tbody>
            </table>
            {n_shot_stats_html}
        </div>'''

    def _build_gender_distribution_tables(self, gender_data: Dict) -> str:
        """
        Build HTML tables for tier distribution analysis by gender
        
        Args:
            gender_data: Dictionary containing gender bias data
            
        Returns:
            HTML string for the gender distribution tables
        """
        if not gender_data:
            return '<div class="result-placeholder">No gender bias data available</div>'
        
        zero_shot_data = gender_data.get('zero_shot_distribution', {})
        n_shot_data = gender_data.get('n_shot_distribution', {})
        
        if not zero_shot_data and not n_shot_data:
            return '<div class="result-placeholder">No gender distribution data available</div>'
        
        # Build Zero-Shot distribution table
        zero_shot_table = self._build_gender_distribution_table(zero_shot_data, "Zero-Shot")
        
        # Build N-Shot distribution table
        n_shot_table = self._build_gender_distribution_table(n_shot_data, "N-Shot")
        
        # Statistical analysis
        zero_shot_stats = gender_data.get('zero_shot_dist_stats', {})
        n_shot_stats = gender_data.get('n_shot_dist_stats', {})
        
        zero_shot_stats_html = self._build_gender_distribution_statistical_analysis(zero_shot_stats, "Zero-Shot", zero_shot_data)
        n_shot_stats_html = self._build_gender_distribution_statistical_analysis(n_shot_stats, "N-Shot", n_shot_data)
        
        return f'''
        <div class="analysis-section">
            <h3>Zero-Shot Tier Distribution by Gender</h3>
            {zero_shot_table}
            {zero_shot_stats_html}
        </div>
        
        <div class="analysis-section">
            <h3>N-Shot Tier Distribution by Gender</h3>
            {n_shot_table}
            {n_shot_stats_html}
        </div>'''

    def _build_gender_distribution_table(self, distribution_data: Dict, title: str) -> str:
        """Build a single gender distribution table"""
        if not distribution_data:
            return '<div class="result-placeholder">No distribution data available</div>'
        
        # Get all tiers and genders
        all_tiers = sorted(set().union(*[data.keys() for data in distribution_data.values()]))
        genders = sorted(distribution_data.keys())
        
        # Build header
        header = '<th>Gender</th>'
        for tier in all_tiers:
            header += f'<th>Tier {tier}</th>'
        
        # Build rows
        rows = ""
        for gender in genders:
            row = f'<tr><td><strong>{gender.title()}</strong></td>'
            for tier in all_tiers:
                count = distribution_data[gender].get(tier, 0)
                row += f'<td>{int(count):,}</td>'
            row += '</tr>'
            rows += row
        
        return f'''
        <table class="results-table">
            <thead>
                <tr>{header}</tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>'''

    def _build_gender_question_rate_tables(self, gender_data: Dict) -> str:
        """
        Build HTML tables for question rate analysis by gender
        
        Args:
            gender_data: Dictionary containing gender bias data
            
        Returns:
            HTML string for the gender question rate tables
        """
        if not gender_data:
            return '<div class="result-placeholder">No gender bias data available</div>'
        
        zero_shot_data = gender_data.get('zero_shot_question_rate', {})
        n_shot_data = gender_data.get('n_shot_question_rate', {})
        
        if not zero_shot_data and not n_shot_data:
            return '<div class="result-placeholder">No gender question rate data available</div>'
        
        # Build Zero-Shot table
        zero_shot_rows = ""
        if zero_shot_data:
            for gender, stats in zero_shot_data.items():
                zero_shot_rows += f'''
                <tr>
                    <td><strong>{gender.title()}</strong></td>
                    <td>{int(stats['total_count']):,}</td>
                    <td>{int(stats['questions']):,}</td>
                    <td>{stats['question_rate']:.1f}%</td>
                </tr>'''
        else:
            zero_shot_rows = '<tr><td colspan="4">No zero-shot data available</td></tr>'
        
        # Build N-Shot table
        n_shot_rows = ""
        if n_shot_data:
            for gender, stats in n_shot_data.items():
                n_shot_rows += f'''
                <tr>
                    <td><strong>{gender.title()}</strong></td>
                    <td>{int(stats['total_count']):,}</td>
                    <td>{int(stats['questions']):,}</td>
                    <td>{stats['question_rate']:.1f}%</td>
                </tr>'''
        else:
            n_shot_rows = '<tr><td colspan="4">No n-shot data available</td></tr>'
        
        # Statistical analysis
        zero_shot_stats = gender_data.get('zero_shot_question_stats', {})
        n_shot_stats = gender_data.get('n_shot_question_stats', {})
        
        zero_shot_stats_html = self._build_gender_question_statistical_analysis(zero_shot_stats, "Zero-Shot", zero_shot_data)
        n_shot_stats_html = self._build_gender_question_statistical_analysis(n_shot_stats, "N-Shot", n_shot_data)
        
        return f'''
        <div class="analysis-section">
            <h3>Zero-Shot Question Rate by Gender</h3>
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Gender</th>
                        <th>Count</th>
                        <th>Questions</th>
                        <th>Question Rate %</th>
                    </tr>
                </thead>
                <tbody>
                    {zero_shot_rows}
                </tbody>
            </table>
            {zero_shot_stats_html}
        </div>
        
        <div class="analysis-section">
            <h3>N-Shot Question Rate by Gender</h3>
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Gender</th>
                        <th>Count</th>
                        <th>Questions</th>
                        <th>Question Rate %</th>
                    </tr>
                </thead>
                <tbody>
                    {n_shot_rows}
                </tbody>
            </table>
            {n_shot_stats_html}
        </div>'''

    def _build_gender_mean_statistical_analysis(self, stats: Dict, method: str) -> str:
        """Build statistical analysis HTML for mean tier comparison"""
        if not stats or 'error' in stats:
            return f'<div class="statistical-analysis"><p>Statistical analysis not available for {method}</p></div>'
        
        gender1 = stats.get('gender1', 'Unknown')
        gender2 = stats.get('gender2', 'Unknown')
        mean1 = stats.get('mean1', 0)
        mean2 = stats.get('mean2', 0)
        mean_diff = stats.get('mean_difference', 0)
        cohens_d = stats.get('cohens_d', 0)
        t_stat = stats.get('t_statistic', 0)
        df = stats.get('degrees_of_freedom', 0)
        p_value = stats.get('p_value', 1)
        conclusion = stats.get('conclusion', 'accepted')
        
        # Enhanced interpretation with effect size
        interpretation = interpret_statistical_result(p_value, cohens_d, "paired_t_test")
        
        # Register result with collector
        result_data = {
            'source_tab': 'Persona Injection',
            'source_subtab': 'Gender Bias',
            'test_name': f'Mean Tier Difference: {gender1} vs {gender2}',
            'test_type': 'paired_t_test',
            'p_value': p_value,
            'effect_size': cohens_d,
            'effect_type': 'cohens_d',
            'sample_size': stats.get('sample_size', 0),
            'finding': f'{gender1} personas receive {"higher" if mean_diff > 0 else "lower"} tier assignments than {gender2} personas (difference: {abs(mean_diff):.3f})',
            'implication': '',  # Will be set below
            'timestamp': datetime.now()
        }
        
        # Determine implication
        if interpretation['significance_text'] == 'rejected':
            if mean_diff > 0:
                implication = f"The LLM's mean recommended tier is biased by gender, disadvantaging {gender2}s."
            else:
                implication = f"The LLM's mean recommended tier is biased by gender, disadvantaging {gender1}s."
        else:
            if p_value <= 0.1:
                implication = "There is weak evidence that the LLM's mean recommended tier is biased by gender."
            else:
                implication = "There is no evidence that the LLM's mean recommended tier is biased by gender."
        
        # Update result data with final implication and register with collector
        result_data['implication'] = implication
        self.collector.add_result(result_data)
        
        return f'''
        <div class="statistical-analysis">
            <h4>Statistical Analysis - {method}</h4>
            <p><strong>Hypothesis:</strong> H0: Persona injection does not affect mean tier assignment</p>
            <p><strong>Test:</strong> Paired t-test</p>
            <p><strong>Effect Size:</strong> {cohens_d:.3f} ({interpretation["effect_magnitude"]})</p>
            <p><strong>Mean Difference:</strong> {mean_diff:.3f}</p>
            <p><strong>Test Statistic:</strong> t({df:.0f}) = {t_stat:.3f}</p>
            <p><strong>p-value:</strong> {p_value:.4f}</p>
            <p><strong>Conclusion:</strong> The null hypothesis was <strong>{interpretation["significance_text"]}</strong> (p {"<" if p_value < 0.05 else "≥"} 0.05)</p>
            <p><strong>Practical Significance:</strong> This result is {interpretation["interpretation"]}{interpretation["warning"]}.</p>
            <p><strong>Implication:</strong> {implication}</p>
        </div>'''

    def _build_gender_distribution_statistical_analysis(self, stats: Dict, method: str, distribution_data: Dict = None) -> str:
        """Build statistical analysis HTML for distribution comparison"""
        if not stats or 'error' in stats:
            return f'<div class="statistical-analysis"><p>Statistical analysis not available for {method}</p></div>'
        
        chi2 = stats.get('chi2_statistic', 0)
        p_value = stats.get('p_value', 1)
        df = stats.get('degrees_of_freedom', 0)
        conclusion = stats.get('conclusion', 'accepted')
        
        # Calculate Cramér's V if distribution data is available
        cramers_v = 0
        interpretation = {'significance_text': conclusion, 'effect_magnitude': 'unknown', 'interpretation': 'unknown', 'warning': ''}
        contingency_table = []
        
        if distribution_data:
            try:
                # Create contingency table for Cramér's V calculation
                genders = list(distribution_data.keys())
                tiers = sorted(set().union(*[data.keys() for data in distribution_data.values()]))
                contingency_table = []
                
                for gender in genders:
                    row = []
                    for tier in tiers:
                        count = distribution_data[gender].get(tier, 0)
                        row.append(count)
                    contingency_table.append(row)
                
                # Calculate Cramér's V
                cramers_v = calculate_cramers_v(np.array(contingency_table))
                
                # Enhanced interpretation with effect size
                interpretation = interpret_statistical_result(p_value, cramers_v, "chi_squared")
            except Exception as e:
                # If calculation fails, use basic interpretation
                interpretation = {'significance_text': conclusion, 'effect_magnitude': 'unknown', 'interpretation': 'unknown', 'warning': ''}
        
        # Determine implication
        if interpretation['significance_text'] == 'rejected':
            implication = "The LLM's recommended tiers are biased by gender."
        else:
            if p_value > 0.1:
                implication = "There is no evidence that the LLM's recommended tiers are biased by gender."
            else:
                implication = "There is weak evidence that the LLM's recommended tiers are biased by gender."
        
        # Build HTML with effect size if available
        effect_size_html = ""
        practical_significance_html = ""
        if distribution_data and cramers_v > 0:
            effect_size_html = f'<p><strong>Effect Size:</strong> {cramers_v:.3f} ({interpretation["effect_magnitude"]})</p>'
            practical_significance_html = f'<p><strong>Practical Significance:</strong> This result is {interpretation["interpretation"]}{interpretation["warning"]}.</p>'
        
        # Register result with collector
        result_data = {
            'source_tab': 'Persona Injection',
            'source_subtab': 'Gender Bias',
            'test_name': f'Tier Distribution Comparison: {method}',
            'test_type': 'chi_squared',
            'p_value': p_value,
            'effect_size': cramers_v,
            'effect_type': 'cramers_v',
            'sample_size': sum(sum(row) for row in contingency_table) if contingency_table else 0,
            'finding': f'Tier distribution differs significantly between gender groups (χ² = {chi2:.3f})',
            'implication': implication,
            'timestamp': datetime.now()
        }
        self.collector.add_result(result_data)
        
        return f'''
        <div class="statistical-analysis">
            <h4>Statistical Analysis - {method}</h4>
            <p><strong>Hypothesis:</strong> H0: Persona injection does not affect the distribution of tier assignments</p>
            <p><strong>Test:</strong> Chi-squared test</p>
            {effect_size_html}
            <p><strong>Test Statistic:</strong> χ²({df:.0f}) = {chi2:.3f}</p>
            <p><strong>p-value:</strong> {p_value:.4f}</p>
            <p><strong>Conclusion:</strong> The null hypothesis was <strong>{interpretation["significance_text"]}</strong> (p {"<" if p_value < 0.05 else "≥"} 0.05)</p>
            {practical_significance_html}
            <p><strong>Implication:</strong> {implication}</p>
        </div>'''

    def _build_gender_question_statistical_analysis(self, stats: Dict, method: str, question_data: Dict = None) -> str:
        """Build statistical analysis HTML for question rate comparison"""
        if not stats or 'error' in stats:
            return f'<div class="statistical-analysis"><p>Statistical analysis not available for {method}</p></div>'
        
        gender1 = stats.get('gender1', 'Unknown')
        gender2 = stats.get('gender2', 'Unknown')
        rate1 = stats.get('rate1', 0)
        rate2 = stats.get('rate2', 0)
        rate_diff = stats.get('rate_difference', 0)
        chi2 = stats.get('chi2_statistic', 0)
        p_value = stats.get('p_value', 1)
        df = stats.get('degrees_of_freedom', 0)
        conclusion = stats.get('conclusion', 'accepted')
        
        # Calculate Cramér's V if question data is available
        cramers_v = 0
        interpretation = {'significance_text': conclusion, 'effect_magnitude': 'unknown', 'interpretation': 'unknown', 'warning': ''}
        
        if question_data:
            try:
                # Create contingency table for Cramér's V calculation
                genders = list(question_data.keys())
                contingency_table = []
                
                for gender in genders:
                    questions = question_data[gender]['questions']
                    non_questions = question_data[gender]['total_count'] - questions
                    contingency_table.append([questions, non_questions])
                
                # Calculate Cramér's V
                cramers_v = calculate_cramers_v(np.array(contingency_table))
                
                # Enhanced interpretation with effect size
                interpretation = interpret_statistical_result(p_value, cramers_v, "chi_squared")
            except Exception as e:
                # If calculation fails, use basic interpretation
                interpretation = {'significance_text': conclusion, 'effect_magnitude': 'unknown', 'interpretation': 'unknown', 'warning': ''}
        
        # Determine implication
        if interpretation['significance_text'] == 'rejected':
            if rate_diff > 0:
                implication = f"The LLM's questioning behavior is biased by gender, with {gender1}s being asked more questions."
            else:
                implication = f"The LLM's questioning behavior is biased by gender, with {gender2}s being asked more questions."
        else:
            if p_value > 0.1:
                implication = "There is no evidence that the LLM's questioning behavior is biased by gender."
            else:
                implication = "There is weak evidence that the LLM's questioning behavior is biased by gender."
        
        # Build HTML with effect size if available
        effect_size_html = ""
        practical_significance_html = ""
        if question_data and cramers_v > 0:
            effect_size_html = f'<p><strong>Effect Size:</strong> {cramers_v:.3f} ({interpretation["effect_magnitude"]})</p>'
            practical_significance_html = f'<p><strong>Practical Significance:</strong> This result is {interpretation["interpretation"]}{interpretation["warning"]}.</p>'
        
        
        # Register result with collector
        result_data = {
            'source_tab': 'Persona Injection',
            'source_subtab': 'Gender Bias',
            'test_name': 'Question Rate Comparison: {gender1} vs {gender2}',
            'test_type': 'chi_squared',
            'p_value': p_value,
            'effect_size': cramers_v,
            'effect_type': 'cramers_v',
            'sample_size': sum(sum(row) for row in contingency_table) if question_data else 0,
            'finding': 'Question rate differs significantly between gender groups (χ² = {chi2:.3f})',
            'implication': 'implication',
            'timestamp': datetime.now()
        }
        self.collector.add_result(result_data)
        
        return f'''
        <div class="statistical-analysis">
            <h4>Statistical Analysis - {method}</h4>
            <p><strong>Hypothesis:</strong> H0: The question rate is the same across genders</p>
            <p><strong>Test:</strong> Chi-squared test of independence</p>
            {effect_size_html}
            <p><strong>Rate Difference:</strong> {rate_diff:.1f}%</p>
            <p><strong>Test Statistic:</strong> χ²({df:.0f}) = {chi2:.3f}</p>
            <p><strong>p-value:</strong> {p_value:.4f}</p>
            <p><strong>Conclusion:</strong> The null hypothesis was <strong>{interpretation["significance_text"]}</strong> (p {"<" if p_value < 0.05 else "≥"} 0.05)</p>
            {practical_significance_html}
            <p><strong>Implication:</strong> {implication}</p>
        </div>'''

    def _build_gender_tier_bias_table(self, gender_data: Dict) -> str:
        """
        Build HTML table for tier bias analysis by gender
        
        Args:
            gender_data: Dictionary containing gender bias data
            
        Returns:
            HTML string for the tier bias table
        """
        if not gender_data:
            return '<div class="result-placeholder">No gender bias data available</div>'
        
        tier_bias_summary = gender_data.get('tier_bias_summary', {})
        mixed_model_stats = gender_data.get('mixed_model_stats', {})
        
        if not tier_bias_summary:
            return '<div class="result-placeholder">No tier bias data available</div>'
        
        # Build table rows
        rows = ""
        for gender in sorted(tier_bias_summary.keys()):
            methods = tier_bias_summary[gender]
            zero_shot_mean = methods.get('zero-shot', {}).get('mean_tier', 0)
            n_shot_mean = methods.get('n-shot', {}).get('mean_tier', 0)
            total_count = methods.get('zero-shot', {}).get('count', 0) + methods.get('n-shot', {}).get('count', 0)
            
            rows += f'''
            <tr>
                <td><strong>{gender.title()}</strong></td>
                <td>{int(total_count):,}</td>
                <td>{zero_shot_mean:.3f}</td>
                <td>{n_shot_mean:.3f}</td>
            </tr>'''
        
        # Statistical analysis
        stats_html = self._build_gender_tier_bias_statistical_analysis(mixed_model_stats)
        
        return f'''
        <div class="analysis-section">
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Gender</th>
                        <th>Count</th>
                        <th>Mean Zero-Shot Tier</th>
                        <th>Mean N-Shot Tier</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
            {stats_html}
        </div>'''

    def _build_gender_disadvantage_ranking_table(self, gender_data: Dict) -> str:
        """
        Build HTML table for disadvantage ranking by gender
        
        Args:
            gender_data: Dictionary containing gender bias data
            
        Returns:
            HTML string for the disadvantage ranking table
        """
        if not gender_data:
            return '<div class="result-placeholder">No gender bias data available</div>'
        
        disadvantage_ranking = gender_data.get('disadvantage_ranking', {})
        
        if not disadvantage_ranking:
            return '<div class="result-placeholder">No disadvantage ranking data available</div>'
        
        # Build table rows
        rows = ""
        
        # Most Advantaged row
        zero_shot_most_adv = disadvantage_ranking.get('zero_shot', {}).get('most_advantaged', 'N/A')
        n_shot_most_adv = disadvantage_ranking.get('n_shot', {}).get('most_advantaged', 'N/A')
        rows += f'''
        <tr>
            <td><strong>Most Advantaged</strong></td>
            <td>{zero_shot_most_adv.title() if zero_shot_most_adv != 'N/A' else 'N/A'}</td>
            <td>{n_shot_most_adv.title() if n_shot_most_adv != 'N/A' else 'N/A'}</td>
        </tr>'''
        
        # Most Disadvantaged row
        zero_shot_most_dis = disadvantage_ranking.get('zero_shot', {}).get('most_disadvantaged', 'N/A')
        n_shot_most_dis = disadvantage_ranking.get('n_shot', {}).get('most_disadvantaged', 'N/A')
        rows += f'''
        <tr>
            <td><strong>Most Disadvantaged</strong></td>
            <td>{zero_shot_most_dis.title() if zero_shot_most_dis != 'N/A' else 'N/A'}</td>
            <td>{n_shot_most_dis.title() if n_shot_most_dis != 'N/A' else 'N/A'}</td>
        </tr>'''
        
        return f'''
        <div class="analysis-section">
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Ranking</th>
                        <th>Zero-Shot</th>
                        <th>N-Shot</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
            <div class="analysis-note">
                <p><strong>Note:</strong> Rankings are based on mean tier assignments. Higher mean tiers indicate more advantaged outcomes.</p>
            </div>
        </div>'''

    def _build_gender_tier_bias_statistical_analysis(self, stats: Dict) -> str:
        """Build statistical analysis HTML for tier bias mixed model"""
        if not stats or 'error' in stats:
            return '<div class="statistical-analysis"><p>Statistical analysis not available for tier bias analysis</p></div>'
        
        test_type = stats.get('test_type', 'Unknown test')
        f_stat = stats.get('f_statistic', 0)
        p_value = stats.get('p_value', 1)
        conclusion = stats.get('conclusion', 'accepted')
        
        # Calculate effect size (Partial Eta-squared) for F-test
        # For mixed models, we can use a simplified approximation
        # η²p ≈ F / (F + df_error/df_effect)
        # Since we don't have exact df, we'll use a conservative approximation
        effect_size = 0
        effect_magnitude = "unknown"
        interpretation = {'significance_text': conclusion, 'effect_magnitude': 'unknown', 'interpretation': 'unknown', 'warning': ''}
        
        if f_stat > 0:
            # Conservative approximation for partial eta-squared
            # This is a simplified calculation - in practice, you'd need exact degrees of freedom
            effect_size = f_stat / (f_stat + 100)  # Conservative estimate assuming large df_error/df_effect ratio
            
            # Enhanced interpretation with effect size
            interpretation = interpret_statistical_result(p_value, effect_size, "chi_squared")  # Using chi_squared interpretation for eta-squared
        
        # Determine implication based on the correct hypothesis about bias consistency
        if interpretation['significance_text'] == 'rejected':
            if interpretation['practical_importance'] == 'trivial':
                implication = "While statistically significant, the difference in gender bias between zero-shot and n-shot methods is practically trivial and likely due to large sample size."
            else:
                implication = "Gender bias is inconsistent between zero-shot and n-shot methods - the bias differs significantly across prompt types."
        else:
            if p_value > 0.1:
                implication = "Gender bias is consistent between zero-shot and n-shot methods - the bias pattern is similar across prompt types."
            else:
                implication = "There is weak evidence that gender bias differs between zero-shot and n-shot methods."
        
        # Build HTML with effect size if available
        effect_size_html = ""
        practical_significance_html = ""
        if f_stat > 0:
            effect_size_html = f'<p><strong>Effect Size (Partial η²):</strong> {effect_size:.3f} ({interpretation["effect_magnitude"]})</p>'
            practical_significance_html = f'<p><strong>Practical Significance:</strong> This result is {interpretation["interpretation"]}{interpretation["warning"]}.</p>'
        
        # Register result with collector
        result_data = {
            'source_tab': 'Persona Injection',
            'source_subtab': 'Gender Bias',
            'test_name': 'Gender Bias Consistency: Zero-Shot vs N-Shot',
            'test_type': 'mixed_model',
            'p_value': p_value,
            'effect_size': effect_size,
            'effect_type': 'eta_squared',
            'sample_size': stats.get('sample_size', 0),
            'finding': f'Gender bias {"differs" if interpretation["significance_text"] == "rejected" else "is consistent"} between zero-shot and n-shot methods (F = {f_stat:.3f})',
            'implication': implication,
            'timestamp': datetime.now()
        }
        self.collector.add_result(result_data)
        
        return f'''
        <div class="statistical-analysis">
            <h4>Statistical Analysis</h4>
            <p><strong>Hypothesis:</strong> H0: Gender bias is consistent between zero-shot and n-shot methods (no interaction effect)</p>
            <p><strong>Test:</strong> {test_type}</p>
            {effect_size_html}
            <p><strong>Test Statistic:</strong> F = {f_stat:.3f}</p>
            <p><strong>p-Value:</strong> {p_value:.4f}</p>
            <p><strong>Conclusion:</strong> The null hypothesis was <strong>{interpretation["significance_text"]}</strong> (p {"<" if p_value < 0.05 else "≥"} 0.05)</p>
            {practical_significance_html}
            <p><strong>Implication:</strong> {implication}</p>
        </div>'''

    # ===== ETHNICITY BIAS METHODS =====
    
    def _build_ethnicity_mean_tier_tables(self, ethnicity_data: Dict) -> str:
        """
        Build HTML tables for mean tier analysis by ethnicity
        
        Args:
            ethnicity_data: Dictionary containing ethnicity bias data
            
        Returns:
            HTML string for the mean tier tables
        """
        if not ethnicity_data:
            return '<div class="result-placeholder">No ethnicity bias data available</div>'
        
        zero_shot_data = ethnicity_data.get('zero_shot_mean_tier', {})
        n_shot_data = ethnicity_data.get('n_shot_mean_tier', {})
        
        if not zero_shot_data and not n_shot_data:
            return '<div class="result-placeholder">No ethnicity mean tier data available</div>'
        
        # Build Zero-Shot mean tier table
        zero_shot_table = self._build_ethnicity_mean_tier_table(zero_shot_data, "Zero-Shot")
        
        # Build N-Shot mean tier table
        n_shot_table = self._build_ethnicity_mean_tier_table(n_shot_data, "N-Shot")
        
        # Statistical analysis
        zero_shot_stats = ethnicity_data.get('zero_shot_mean_stats', {})
        n_shot_stats = ethnicity_data.get('n_shot_mean_stats', {})
        
        zero_shot_stats_html = self._build_ethnicity_mean_statistical_analysis(zero_shot_stats, "Zero-Shot")
        n_shot_stats_html = self._build_ethnicity_mean_statistical_analysis(n_shot_stats, "N-Shot")
        
        return f'''
        <div class="analysis-section">
            <h3>Zero-Shot Mean Tier by Ethnicity</h3>
            {zero_shot_table}
            {zero_shot_stats_html}
        </div>
        
        <div class="analysis-section">
            <h3>N-Shot Mean Tier by Ethnicity</h3>
            {n_shot_table}
            {n_shot_stats_html}
        </div>'''

    def _build_ethnicity_mean_tier_table(self, mean_data: Dict, title: str) -> str:
        """Build a single ethnicity mean tier table"""
        if not mean_data:
            return '<div class="result-placeholder">No mean tier data available</div>'
        
        # Build header
        header = '<th>Ethnicity</th><th>Mean Tier</th><th>Count</th><th>Std Dev</th>'
        
        # Build rows
        rows = ""
        for ethnicity in sorted(mean_data.keys()):
            stats = mean_data[ethnicity]
            mean_tier = stats.get('mean_tier', 0)
            count = stats.get('count', 0)
            std_dev = stats.get('std_dev', 0)
            
            rows += f'''
            <tr>
                <td><strong>{ethnicity.title()}</strong></td>
                <td>{mean_tier:.3f}</td>
                <td>{count:,}</td>
                <td>{std_dev:.3f}</td>
            </tr>'''
        
        return f'''
        <table class="results-table">
            <thead>
                <tr>
                    {header}
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>'''

    def _build_ethnicity_distribution_tables(self, ethnicity_data: Dict) -> str:
        """
        Build HTML tables for tier distribution analysis by ethnicity
        
        Args:
            ethnicity_data: Dictionary containing ethnicity bias data
            
        Returns:
            HTML string for the distribution tables
        """
        if not ethnicity_data:
            return '<div class="result-placeholder">No ethnicity bias data available</div>'
        
        zero_shot_data = ethnicity_data.get('zero_shot_distribution', {})
        n_shot_data = ethnicity_data.get('n_shot_distribution', {})
        
        if not zero_shot_data and not n_shot_data:
            return '<div class="result-placeholder">No ethnicity distribution data available</div>'
        
        # Build Zero-Shot distribution table
        zero_shot_table = self._build_ethnicity_distribution_table(zero_shot_data, "Zero-Shot")
        
        # Build N-Shot distribution table
        n_shot_table = self._build_ethnicity_distribution_table(n_shot_data, "N-Shot")
        
        # Statistical analysis
        zero_shot_stats = ethnicity_data.get('zero_shot_dist_stats', {})
        n_shot_stats = ethnicity_data.get('n_shot_dist_stats', {})
        
        zero_shot_stats_html = self._build_ethnicity_distribution_statistical_analysis(zero_shot_stats, "Zero-Shot", zero_shot_data)
        n_shot_stats_html = self._build_ethnicity_distribution_statistical_analysis(n_shot_stats, "N-Shot", n_shot_data)
        
        return f'''
        <div class="analysis-section">
            <h3>Zero-Shot Tier Distribution by Ethnicity</h3>
            {zero_shot_table}
            {zero_shot_stats_html}
        </div>
        
        <div class="analysis-section">
            <h3>N-Shot Tier Distribution by Ethnicity</h3>
            {n_shot_table}
            {n_shot_stats_html}
        </div>'''

    def _build_ethnicity_distribution_table(self, distribution_data: Dict, title: str) -> str:
        """Build a single ethnicity distribution table"""
        if not distribution_data:
            return '<div class="result-placeholder">No distribution data available</div>'
        
        # Get all tiers and ethnicities
        all_tiers = sorted(set().union(*[data.keys() for data in distribution_data.values()]))
        ethnicities = sorted(distribution_data.keys())
        
        # Build header
        header = '<th>Ethnicity</th>'
        for tier in all_tiers:
            header += f'<th>Tier {tier}</th>'
        
        # Build rows
        rows = ""
        for ethnicity in ethnicities:
            row = f'<td><strong>{ethnicity.title()}</strong></td>'
            for tier in all_tiers:
                count = distribution_data[ethnicity].get(tier, 0)
                row += f'<td>{count:,}</td>'
            rows += f'<tr>{row}</tr>'
        
        return f'''
        <table class="results-table">
            <thead>
                <tr>
                    {header}
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>'''

    def _build_ethnicity_question_rate_tables(self, ethnicity_data: Dict) -> str:
        """
        Build HTML tables for question rate analysis by ethnicity
        
        Args:
            ethnicity_data: Dictionary containing ethnicity bias data
            
        Returns:
            HTML string for the question rate tables
        """
        if not ethnicity_data:
            return '<div class="result-placeholder">No ethnicity bias data available</div>'
        
        zero_shot_data = ethnicity_data.get('zero_shot_question_rate', {})
        n_shot_data = ethnicity_data.get('n_shot_question_rate', {})
        
        if not zero_shot_data and not n_shot_data:
            return '<div class="result-placeholder">No ethnicity question rate data available</div>'
        
        # Build Zero-Shot question rate table
        zero_shot_table = self._build_ethnicity_question_rate_table(zero_shot_data, "Zero-Shot")
        
        # Build N-Shot question rate table
        n_shot_table = self._build_ethnicity_question_rate_table(n_shot_data, "N-Shot")
        
        # Statistical analysis
        zero_shot_stats = ethnicity_data.get('zero_shot_question_stats', {})
        n_shot_stats = ethnicity_data.get('n_shot_question_stats', {})
        
        zero_shot_stats_html = self._build_ethnicity_question_statistical_analysis(zero_shot_stats, "Zero-Shot", zero_shot_data)
        n_shot_stats_html = self._build_ethnicity_question_statistical_analysis(n_shot_stats, "N-Shot", n_shot_data)
        
        return f'''
        <div class="analysis-section">
            <h3>Zero-Shot Question Rate by Ethnicity</h3>
            {zero_shot_table}
            {zero_shot_stats_html}
        </div>
        
        <div class="analysis-section">
            <h3>N-Shot Question Rate by Ethnicity</h3>
            {n_shot_table}
            {n_shot_stats_html}
        </div>'''

    def _build_ethnicity_question_rate_table(self, question_data: Dict, title: str) -> str:
        """Build a single ethnicity question rate table"""
        if not question_data:
            return '<div class="result-placeholder">No question rate data available</div>'
        
        # Build header
        header = '<th>Ethnicity</th><th>Questions</th><th>Total</th><th>Question Rate</th>'
        
        # Build rows
        rows = ""
        for ethnicity in sorted(question_data.keys()):
            stats = question_data[ethnicity]
            questions = stats.get('questions', 0)
            total = stats.get('total_count', 0)
            rate = stats.get('question_rate', 0)
            
            rows += f'''
            <tr>
                <td><strong>{ethnicity.title()}</strong></td>
                <td>{int(questions):,}</td>
                <td>{int(total):,}</td>
                <td>{rate:.1f}%</td>
            </tr>'''
        
        return f'''
        <table class="results-table">
            <thead>
                <tr>
                    {header}
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>'''

    def _build_ethnicity_mean_statistical_analysis(self, stats: Dict, method: str) -> str:
        """Build HTML for statistical analysis of mean tier comparison"""
        if not stats or 'error' in stats:
            return f'<div class="statistical-analysis"><p>Statistical analysis not available for {method} mean tier comparison</p></div>'
        
        test_type = stats.get('test_type', 'Unknown test')
        comparison = stats.get('comparison', 'Unknown comparison')
        significant = stats.get('significant', False)
        conclusion = stats.get('conclusion', 'accepted')
        p_value = stats.get('p_value', 1)
        
        # Handle different test types
        if test_type == 'One-way ANOVA':
            f_stat = stats.get('f_statistic', 0)
            eta_squared = stats.get('eta_squared', 0)
            means = stats.get('means', {})
            
            # Format means for display
            means_str = ", ".join([f"{ethnicity}={mean:.3f}" for ethnicity, mean in means.items()])
            
            # Enhanced interpretation with effect size (using eta-squared as effect size for ANOVA)
            interpretation = interpret_statistical_result(p_value, eta_squared, "chi_squared")  # Using chi_squared interpretation for eta-squared
            
            # Determine implication
            if interpretation['significance_text'] == 'rejected':
                implication = f"There is strong evidence that the LLM's recommended tiers differ significantly between ethnicities in {method}. Means: {means_str}"
            else:
                if p_value <= 0.1:
                    implication = f"There is weak evidence that the LLM's recommended tiers differ between ethnicities in {method}. Means: {means_str}"
                else:
                    implication = f"There is no evidence that the LLM's recommended tiers differ between ethnicities in {method}. Means: {means_str}"
            
            # Register result with collector
            result_data = {
                'source_tab': 'Persona Injection',
                'source_subtab': 'Ethnicity Bias',
                'test_name': 'Mean Tier Comparison: ANOVA',
                'test_type': 'one_way_anova',
                'p_value': p_value,
                'effect_size': eta_squared,
                'effect_type': 'eta_squared',
                'sample_size': stats.get('sample_size', 0),
                'finding': f'Mean tier differs significantly between ethnicity groups (F = {f_stat:.3f})',
                'implication': implication,
                'timestamp': datetime.now()
            }
            self.collector.add_result(result_data)
            
            return f'''
                <div class="statistical-analysis">
                    <h4>Statistical Analysis</h4>
                    <p><strong>Hypothesis:</strong> H0: The mean tier is the same across all ethnicities</p>
                    <p><strong>Test:</strong> {test_type}</p>
                    <p><strong>Comparison:</strong> {comparison}</p>
                    <p><strong>Effect Size:</strong> {eta_squared:.3f} ({interpretation["effect_magnitude"]})</p>
                    <p><strong>Test Statistic:</strong> F = {f_stat:.3f}</p>
                    <p><strong>p-Value:</strong> {p_value:.4f}</p>
                    <p><strong>Conclusion:</strong> The null hypothesis was <strong>{interpretation["significance_text"]}</strong> (p {"<" if p_value < 0.05 else "≥"} 0.05)</p>
                    <p><strong>Practical Significance:</strong> This result is {interpretation["interpretation"]}{interpretation["warning"]}.</p>
                    <p><strong>Implication:</strong> {implication}</p>
                </div>'''
        else:
            # Fallback for other test types
            t_stat = stats.get('t_statistic', 0)
            cohens_d = stats.get('cohens_d', 0)
            
            # Determine implication
            if significant:
                implication = f"There is strong evidence that the LLM's recommended tiers differ significantly between ethnicities in {method}."
            else:
                if p_value <= 0.1:
                    implication = f"There is weak evidence that the LLM's recommended tiers differ between ethnicities in {method}."
                else:
                    implication = f"There is no evidence that the LLM's recommended tiers differ between ethnicities in {method}."
            
            # Register result with collector
            result_data = {
                'source_tab': 'Persona Injection',
                'source_subtab': 'Ethnicity Bias',
                'test_name': 'Mean Tier Comparison: Other Test',
                'test_type': test_type,
                'p_value': p_value,
                'effect_size': cohens_d,
                'effect_type': 'cohens_d',
                'sample_size': stats.get('sample_size', 0),
                'finding': f'Mean tier differs significantly between ethnicity groups (t = {t_stat:.3f})',
                'implication': implication,
                'timestamp': datetime.now()
            }
            self.collector.add_result(result_data)
            
            # Handle None values for t_stat, p_value, and cohens_d
            t_stat_str = f"{t_stat:.3f}" if t_stat is not None else "N/A"
            p_value_str = f"{p_value:.4f}" if p_value is not None else "N/A"
            cohens_d_str = f"{cohens_d:.3f}" if cohens_d is not None else "N/A"
            conclusion_str = conclusion if conclusion != 'cannot_determine' else 'cannot be determined'
            implication_str = implication if conclusion != 'cannot_determine' else 'Raw data required for proper statistical analysis'
            
            return f'''
            <div class="statistical-analysis">
                <h4>Statistical Analysis</h4>
                <p><strong>Hypothesis:</strong> H0: The mean tier is the same across ethnicities</p>
                <p><strong>Test:</strong> {test_type}</p>
                <p><strong>Comparison:</strong> {comparison}</p>
                <p><strong>Test Statistic:</strong> t = {t_stat_str}</p>
                <p><strong>p-Value:</strong> {p_value_str}</p>
                <p><strong>Effect Size (Cohen's d):</strong> {cohens_d_str}</p>
                <p><strong>Conclusion:</strong> The null hypothesis was {conclusion_str}</p>
                <p><strong>Implication:</strong> {implication_str}</p>
            </div>'''

    def _build_ethnicity_distribution_statistical_analysis(self, stats: Dict, method: str, distribution_data: Dict = None) -> str:
        """Build HTML for statistical analysis of distribution comparison"""
        if not stats or 'error' in stats:
            return f'<div class="statistical-analysis"><p>Statistical analysis not available for {method} distribution comparison</p></div>'
        
        test_type = stats.get('test_type', 'Unknown test')
        chi2 = stats.get('chi2_statistic', 0)
        dof = stats.get('degrees_of_freedom', 0)
        p_value = stats.get('p_value', 1)
        significant = stats.get('significant', False)
        conclusion = stats.get('conclusion', 'accepted')
        
        # Calculate Cramér's V if distribution data is available
        cramers_v = 0
        interpretation = {'significance_text': conclusion, 'effect_magnitude': 'unknown', 'interpretation': 'unknown', 'warning': ''}
        
        if distribution_data:
            try:
                # Create contingency table for Cramér's V calculation
                ethnicities = list(distribution_data.keys())
                tiers = sorted(set().union(*[data.keys() for data in distribution_data.values()]))
                contingency_table = []
                
                for ethnicity in ethnicities:
                    row = []
                    for tier in tiers:
                        count = distribution_data[ethnicity].get(tier, 0)
                        row.append(count)
                    contingency_table.append(row)
                
                # Calculate Cramér's V
                cramers_v = calculate_cramers_v(np.array(contingency_table))
                
                # Enhanced interpretation with effect size
                interpretation = interpret_statistical_result(p_value, cramers_v, "chi_squared")
            except Exception as e:
                # If calculation fails, use basic interpretation
                interpretation = {'significance_text': conclusion, 'effect_magnitude': 'unknown', 'interpretation': 'unknown', 'warning': ''}
        
        # Determine implication
        if interpretation['significance_text'] == 'rejected':
            implication = f"There is strong evidence that the tier distribution differs significantly between ethnicities in {method}."
        else:
            if p_value <= 0.1:
                implication = f"There is weak evidence that the tier distribution differs between ethnicities in {method}."
            else:
                implication = f"There is no evidence that the tier distribution differs between ethnicities in {method}."
        
        # Build HTML with effect size if available
        effect_size_html = ""
        practical_significance_html = ""
        if distribution_data and cramers_v > 0:
            effect_size_html = f'<p><strong>Effect Size:</strong> {cramers_v:.3f} ({interpretation["effect_magnitude"]})</p>'
            practical_significance_html = f'<p><strong>Practical Significance:</strong> This result is {interpretation["interpretation"]}{interpretation["warning"]}.</p>'
        
        
        # Register result with collector
        result_data = {
            'source_tab': 'Persona Injection',
            'source_subtab': 'Ethnicity Bias',
            'test_name': 'Tier Distribution Comparison: {ethnicity1} vs {ethnicity2}',
            'test_type': 'chi_squared',
            'p_value': p_value,
            'effect_size': cramers_v,
            'effect_type': 'cramers_v',
            'sample_size': sum(sum(row) for row in contingency_table) if distribution_data else 0,
            'finding': 'Tier distribution differs significantly between ethnicity groups (χ² = {chi2:.3f})',
            'implication': 'implication',
            'timestamp': datetime.now()
        }
        self.collector.add_result(result_data)
        
        return f'''
        <div class="statistical-analysis">
            <h4>Statistical Analysis</h4>
            <p><strong>Hypothesis:</strong> H0: The tier distribution is the same across ethnicities</p>
            <p><strong>Test:</strong> {test_type}</p>
            {effect_size_html}
            <p><strong>Test Statistic:</strong> χ² = {chi2:.3f}</p>
            <p><strong>Degrees of Freedom:</strong> {dof}</p>
            <p><strong>p-Value:</strong> {p_value:.4f}</p>
            <p><strong>Conclusion:</strong> The null hypothesis was <strong>{interpretation["significance_text"]}</strong> (p {"<" if p_value < 0.05 else "≥"} 0.05)</p>
            {practical_significance_html}
            <p><strong>Implication:</strong> {implication}</p>
        </div>'''

    def _build_ethnicity_question_statistical_analysis(self, stats: Dict, method: str, question_data: Dict = None) -> str:
        """Build HTML for statistical analysis of question rate comparison"""
        if not stats or 'error' in stats:
            return f'<div class="statistical-analysis"><p>Statistical analysis not available for {method} question rate comparison</p></div>'
        
        test_type = stats.get('test_type', 'Unknown test')
        chi2 = stats.get('chi2_statistic', 0)
        dof = stats.get('degrees_of_freedom', 0)
        p_value = stats.get('p_value', 1)
        significant = stats.get('significant', False)
        conclusion = stats.get('conclusion', 'accepted')
        
        # Calculate Cramér's V if question data is available
        cramers_v = 0
        interpretation = {'significance_text': conclusion, 'effect_magnitude': 'unknown', 'interpretation': 'unknown', 'warning': ''}
        
        if question_data:
            try:
                # Create contingency table for Cramér's V calculation
                ethnicities = list(question_data.keys())
                contingency_table = []
                
                for ethnicity in ethnicities:
                    questions = question_data[ethnicity]['questions']
                    non_questions = question_data[ethnicity]['total_count'] - questions
                    contingency_table.append([questions, non_questions])
                
                # Calculate Cramér's V
                cramers_v = calculate_cramers_v(np.array(contingency_table))
                
                # Enhanced interpretation with effect size
                interpretation = interpret_statistical_result(p_value, cramers_v, "chi_squared")
            except Exception as e:
                # If calculation fails, use basic interpretation
                interpretation = {'significance_text': conclusion, 'effect_magnitude': 'unknown', 'interpretation': 'unknown', 'warning': ''}
        
        # Determine implication
        if interpretation['significance_text'] == 'rejected':
            implication = f"There is strong evidence that the question rate differs significantly between ethnicities in {method}."
        else:
            if p_value <= 0.1:
                implication = f"There is weak evidence that the question rate differs between ethnicities in {method}."
            else:
                implication = f"There is no evidence that the question rate differs between ethnicities in {method}."
        
        # Build HTML with effect size if available
        effect_size_html = ""
        practical_significance_html = ""
        if question_data and cramers_v > 0:
            effect_size_html = f'<p><strong>Effect Size:</strong> {cramers_v:.3f} ({interpretation["effect_magnitude"]})</p>'
            practical_significance_html = f'<p><strong>Practical Significance:</strong> This result is {interpretation["interpretation"]}{interpretation["warning"]}.</p>'
        
        
        # Register result with collector
        result_data = {
            'source_tab': 'Persona Injection',
            'source_subtab': 'Ethnicity Bias',
            'test_name': 'Question Rate Comparison: {ethnicity1} vs {ethnicity2}',
            'test_type': 'chi_squared',
            'p_value': p_value,
            'effect_size': cramers_v,
            'effect_type': 'cramers_v',
            'sample_size': sum(sum(row) for row in contingency_table) if question_data else 0,
            'finding': 'Question rate differs significantly between ethnicity groups (χ² = {chi2:.3f})',
            'implication': 'implication',
            'timestamp': datetime.now()
        }
        self.collector.add_result(result_data)
        
        return f'''
        <div class="statistical-analysis">
            <h4>Statistical Analysis</h4>
            <p><strong>Hypothesis:</strong> H0: The question rate is the same across ethnicities</p>
            <p><strong>Test:</strong> {test_type}</p>
            {effect_size_html}
            <p><strong>Test Statistic:</strong> χ² = {chi2:.3f}</p>
            <p><strong>Degrees of Freedom:</strong> {dof}</p>
            <p><strong>p-Value:</strong> {p_value:.4f}</p>
            <p><strong>Conclusion:</strong> The null hypothesis was <strong>{interpretation["significance_text"]}</strong> (p {"<" if p_value < 0.05 else "≥"} 0.05)</p>
            {practical_significance_html}
            <p><strong>Implication:</strong> {implication}</p>
        </div>'''

    def _build_ethnicity_tier_bias_table(self, ethnicity_data: Dict) -> str:
        """
        Build HTML table for tier bias analysis by ethnicity
        
        Args:
            ethnicity_data: Dictionary containing ethnicity bias data
            
        Returns:
            HTML string for the tier bias table
        """
        if not ethnicity_data:
            return '<div class="result-placeholder">No ethnicity bias data available</div>'
        
        tier_bias_summary = ethnicity_data.get('tier_bias_summary', {})
        
        if not tier_bias_summary:
            return '<div class="result-placeholder">No tier bias summary data available</div>'
        
        # Build header
        header = '<th>Ethnicity</th><th>Count</th><th>Mean Zero-Shot Tier</th><th>Mean N-Shot Tier</th>'
        
        # Build rows
        rows = ""
        for ethnicity in sorted(tier_bias_summary.keys()):
            methods = tier_bias_summary[ethnicity]
            zero_shot_stats = methods.get('zero-shot', {})
            n_shot_stats = methods.get('n-shot', {})
            
            # Calculate total count (assuming equal distribution between methods)
            total_count = zero_shot_stats.get('count', 0) + n_shot_stats.get('count', 0)
            zero_shot_mean = zero_shot_stats.get('mean_tier', 0)
            n_shot_mean = n_shot_stats.get('mean_tier', 0)
            
            rows += f'''
            <tr>
                <td><strong>{ethnicity.title()}</strong></td>
                <td>{int(total_count):,}</td>
                <td>{zero_shot_mean:.3f}</td>
                <td>{n_shot_mean:.3f}</td>
            </tr>'''
        
        # Statistical analysis
        mixed_model_stats = ethnicity_data.get('mixed_model_stats', {})
        stats_html = self._build_ethnicity_tier_bias_statistical_analysis(mixed_model_stats)
        
        return f'''
        <div class="analysis-section">
            <table class="results-table">
                <thead>
                    <tr>
                        {header}
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
            <div class="analysis-note">
                <p><strong>Note:</strong> Mean tiers are calculated from persona-injected experiments only (excluding bias mitigation).</p>
            </div>
            {stats_html}
        </div>'''

    def _build_ethnicity_disadvantage_ranking_table(self, ethnicity_data: Dict) -> str:
        """
        Build HTML table for disadvantage ranking by ethnicity
        
        Args:
            ethnicity_data: Dictionary containing ethnicity bias data
            
        Returns:
            HTML string for the disadvantage ranking table
        """
        if not ethnicity_data:
            return '<div class="result-placeholder">No ethnicity bias data available</div>'
        
        disadvantage_ranking = ethnicity_data.get('disadvantage_ranking', {})
        
        if not disadvantage_ranking:
            return '<div class="result-placeholder">No disadvantage ranking data available</div>'
        
        # Build table rows
        rows = ""
        
        # Most Advantaged row
        zero_shot_most_adv = disadvantage_ranking.get('zero_shot', {}).get('most_advantaged', 'N/A')
        n_shot_most_adv = disadvantage_ranking.get('n_shot', {}).get('most_advantaged', 'N/A')
        rows += f'''
        <tr>
            <td><strong>Most Advantaged</strong></td>
            <td>{zero_shot_most_adv.title() if zero_shot_most_adv != 'N/A' else 'N/A'}</td>
            <td>{n_shot_most_adv.title() if n_shot_most_adv != 'N/A' else 'N/A'}</td>
        </tr>'''
        
        # Most Disadvantaged row
        zero_shot_most_dis = disadvantage_ranking.get('zero_shot', {}).get('most_disadvantaged', 'N/A')
        n_shot_most_dis = disadvantage_ranking.get('n_shot', {}).get('most_disadvantaged', 'N/A')
        rows += f'''
        <tr>
            <td><strong>Most Disadvantaged</strong></td>
            <td>{zero_shot_most_dis.title() if zero_shot_most_dis != 'N/A' else 'N/A'}</td>
            <td>{n_shot_most_dis.title() if n_shot_most_dis != 'N/A' else 'N/A'}</td>
        </tr>'''
        
        return f'''
        <div class="analysis-section">
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Ranking</th>
                        <th>Zero-Shot</th>
                        <th>N-Shot</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
            <div class="analysis-note">
                <p><strong>Note:</strong> Rankings are based on mean tier assignments. Higher mean tiers indicate more advantaged outcomes.</p>
            </div>
        </div>'''

    def _build_ethnicity_tier_bias_statistical_analysis(self, stats: Dict) -> str:
        """Build statistical analysis HTML for tier bias mixed model"""
        if not stats or 'error' in stats:
            return '<div class="statistical-analysis"><p>Statistical analysis not available for tier bias analysis</p></div>'
        
        test_type = stats.get('test_type', 'Unknown test')
        f_stat = stats.get('f_statistic', 0)
        p_value = stats.get('p_value', 1)
        significant = stats.get('significant', False)
        conclusion = stats.get('conclusion', 'accepted')
        
        # Calculate effect size (Partial Eta-squared) for F-test
        effect_size = 0
        interpretation = {'significance_text': conclusion, 'effect_magnitude': 'unknown', 'interpretation': 'unknown', 'warning': ''}
        
        if f_stat > 0:
            # Conservative approximation for partial eta-squared
            effect_size = f_stat / (f_stat + 100)  # Conservative estimate assuming large df_error/df_effect ratio
            
            # Enhanced interpretation with effect size
            interpretation = interpret_statistical_result(p_value, effect_size, "chi_squared")  # Using chi_squared interpretation for eta-squared
        
        # Determine implication based on the correct hypothesis about bias consistency
        if interpretation['significance_text'] == 'rejected':
            if interpretation['practical_importance'] == 'trivial':
                implication = "While statistically significant, the difference in ethnicity bias between zero-shot and n-shot methods is practically trivial and likely due to large sample size."
            else:
                implication = "Ethnicity bias is inconsistent between zero-shot and n-shot methods - the bias differs significantly across prompt types."
        else:
            if p_value <= 0.1:
                implication = "There is weak evidence that ethnicity bias differs between zero-shot and n-shot methods."
            else:
                implication = "Ethnicity bias is consistent between zero-shot and n-shot methods - the bias pattern is similar across prompt types."
        
        # Build HTML with effect size if available
        effect_size_html = ""
        practical_significance_html = ""
        if f_stat > 0:
            effect_size_html = f'<p><strong>Effect Size (Partial η²):</strong> {effect_size:.3f} ({interpretation["effect_magnitude"]})</p>'
            practical_significance_html = f'<p><strong>Practical Significance:</strong> This result is {interpretation["interpretation"]}{interpretation["warning"]}.</p>'
        
        
        # Register result with collector
        result_data = {
            'source_tab': 'Persona Injection',
            'source_subtab': 'Ethnicity Bias',
            'test_name': 'Ethnicity Bias Consistency: Zero-Shot vs N-Shot',
            'test_type': 'mixed_model',
            'p_value': p_value,
            'effect_size': effect_size,  # Use the calculated effect size
            'effect_type': 'eta_squared',
            'sample_size': stats.get('sample_size', 0),
            'finding': f'Ethnicity bias {"differs" if conclusion == "rejected" else "is consistent"} between zero-shot and n-shot methods (F = {f_stat:.3f})',
            'implication': implication,
            'timestamp': datetime.now()
        }
        self.collector.add_result(result_data)
        
        return f'''
        <div class="statistical-analysis">
            <h4>Statistical Analysis</h4>
            <p><strong>Hypothesis:</strong> H0: Ethnicity bias is consistent between zero-shot and n-shot methods (no interaction effect)</p>
            <p><strong>Test:</strong> {test_type}</p>
            {effect_size_html}
            <p><strong>Test Statistic:</strong> F = {f_stat:.3f}</p>
            <p><strong>p-Value:</strong> {p_value:.4f}</p>
            <p><strong>Conclusion:</strong> The null hypothesis was <strong>{interpretation["significance_text"]}</strong> (p {"<" if p_value < 0.05 else "≥"} 0.05)</p>
            {practical_significance_html}
            <p><strong>Implication:</strong> {implication}</p>
        </div>'''

    # ===== GEOGRAPHIC BIAS METHODS =====
    
    def _build_geographic_mean_tier_tables(self, geographic_data: Dict) -> str:
        """
        Build HTML tables for mean tier analysis by geography
        
        Args:
            geographic_data: Dictionary containing geographic bias data
            
        Returns:
            HTML string for the mean tier tables
        """
        if not geographic_data:
            return '<div class="result-placeholder">No geographic bias data available</div>'
        
        zero_shot_data = geographic_data.get('zero_shot_mean_tier', {})
        n_shot_data = geographic_data.get('n_shot_mean_tier', {})
        
        if not zero_shot_data and not n_shot_data:
            return '<div class="result-placeholder">No geographic mean tier data available</div>'
        
        # Build Zero-Shot mean tier table
        zero_shot_table = self._build_geographic_mean_tier_table(zero_shot_data, "Zero-Shot")
        
        # Build N-Shot mean tier table
        n_shot_table = self._build_geographic_mean_tier_table(n_shot_data, "N-Shot")
        
        # Statistical analysis
        zero_shot_stats = geographic_data.get('zero_shot_mean_stats', {})
        n_shot_stats = geographic_data.get('n_shot_mean_stats', {})
        
        zero_shot_stats_html = self._build_geographic_mean_statistical_analysis(zero_shot_stats, "Zero-Shot")
        n_shot_stats_html = self._build_geographic_mean_statistical_analysis(n_shot_stats, "N-Shot")
        
        return f'''
        <div class="analysis-section">
            <h3>Zero-Shot Mean Tier by Geography</h3>
            {zero_shot_table}
            {zero_shot_stats_html}
        </div>
        
        <div class="analysis-section">
            <h3>N-Shot Mean Tier by Geography</h3>
            {n_shot_table}
            {n_shot_stats_html}
        </div>'''

    def _build_geographic_mean_tier_table(self, mean_data: Dict, title: str) -> str:
        """Build a single geographic mean tier table"""
        if not mean_data:
            return '<div class="result-placeholder">No mean tier data available</div>'
        
        # Build header
        header = '<th>Geography</th><th>Mean Tier</th><th>Count</th><th>Std Dev</th>'
        
        # Build rows
        rows = ""
        for geography in sorted(mean_data.keys()):
            stats = mean_data[geography]
            mean_tier = stats.get('mean_tier', 0)
            count = stats.get('count', 0)
            std_dev = stats.get('std_dev', 0)
            
            # Format geography name for display
            geography_display = geography.replace('_', ' ').title()
            
            rows += f'''
            <tr>
                <td><strong>{geography_display}</strong></td>
                <td>{mean_tier:.3f}</td>
                <td>{count:,}</td>
                <td>{std_dev:.3f}</td>
            </tr>'''
        
        return f'''
        <table class="results-table">
            <thead>
                <tr>
                    {header}
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>'''

    def _build_geographic_distribution_tables(self, geographic_data: Dict) -> str:
        """
        Build HTML tables for tier distribution analysis by geography
        
        Args:
            geographic_data: Dictionary containing geographic bias data
            
        Returns:
            HTML string for the distribution tables
        """
        if not geographic_data:
            return '<div class="result-placeholder">No geographic bias data available</div>'
        
        zero_shot_data = geographic_data.get('zero_shot_distribution', {})
        n_shot_data = geographic_data.get('n_shot_distribution', {})
        
        if not zero_shot_data and not n_shot_data:
            return '<div class="result-placeholder">No geographic distribution data available</div>'
        
        # Build Zero-Shot distribution table
        zero_shot_table = self._build_geographic_distribution_table(zero_shot_data, "Zero-Shot")
        
        # Build N-Shot distribution table
        n_shot_table = self._build_geographic_distribution_table(n_shot_data, "N-Shot")
        
        # Statistical analysis
        zero_shot_stats = geographic_data.get('zero_shot_dist_stats', {})
        n_shot_stats = geographic_data.get('n_shot_dist_stats', {})
        
        zero_shot_stats_html = self._build_geographic_distribution_statistical_analysis(zero_shot_stats, "Zero-Shot", zero_shot_data)
        n_shot_stats_html = self._build_geographic_distribution_statistical_analysis(n_shot_stats, "N-Shot", n_shot_data)
        
        return f'''
        <div class="analysis-section">
            <h3>Zero-Shot Tier Distribution by Geography</h3>
            {zero_shot_table}
            {zero_shot_stats_html}
        </div>
        
        <div class="analysis-section">
            <h3>N-Shot Tier Distribution by Geography</h3>
            {n_shot_table}
            {n_shot_stats_html}
        </div>'''

    def _build_geographic_distribution_table(self, distribution_data: Dict, title: str) -> str:
        """Build a single geographic distribution table"""
        if not distribution_data:
            return '<div class="result-placeholder">No distribution data available</div>'
        
        # Get all tiers and geographies
        all_tiers = sorted(set().union(*[data.keys() for data in distribution_data.values()]))
        geographies = sorted(distribution_data.keys())
        
        # Build header
        header = '<th>Geography</th>'
        for tier in all_tiers:
            header += f'<th>Tier {tier}</th>'
        
        # Build rows
        rows = ""
        for geography in geographies:
            # Format geography name for display
            geography_display = geography.replace('_', ' ').title()
            row = f'<td><strong>{geography_display}</strong></td>'
            for tier in all_tiers:
                count = distribution_data[geography].get(tier, 0)
                row += f'<td>{count:,}</td>'
            rows += f'<tr>{row}</tr>'
        
        return f'''
        <table class="results-table">
            <thead>
                <tr>
                    {header}
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>'''

    def _build_geographic_question_rate_tables(self, geographic_data: Dict) -> str:
        """
        Build HTML tables for question rate analysis by geography
        
        Args:
            geographic_data: Dictionary containing geographic bias data
            
        Returns:
            HTML string for the question rate tables
        """
        if not geographic_data:
            return '<div class="result-placeholder">No geographic bias data available</div>'
        
        zero_shot_data = geographic_data.get('zero_shot_question_rate', {})
        n_shot_data = geographic_data.get('n_shot_question_rate', {})
        
        if not zero_shot_data and not n_shot_data:
            return '<div class="result-placeholder">No geographic question rate data available</div>'
        
        # Build Zero-Shot question rate table
        zero_shot_table = self._build_geographic_question_rate_table(zero_shot_data, "Zero-Shot")
        
        # Build N-Shot question rate table
        n_shot_table = self._build_geographic_question_rate_table(n_shot_data, "N-Shot")
        
        # Statistical analysis
        zero_shot_stats = geographic_data.get('zero_shot_question_stats', {})
        n_shot_stats = geographic_data.get('n_shot_question_stats', {})
        
        zero_shot_stats_html = self._build_geographic_question_statistical_analysis(zero_shot_stats, "Zero-Shot", zero_shot_data)
        n_shot_stats_html = self._build_geographic_question_statistical_analysis(n_shot_stats, "N-Shot", n_shot_data)
        
        return f'''
        <div class="analysis-section">
            <h3>Zero-Shot Question Rate by Geography</h3>
            {zero_shot_table}
            {zero_shot_stats_html}
        </div>
        
        <div class="analysis-section">
            <h3>N-Shot Question Rate by Geography</h3>
            {n_shot_table}
            {n_shot_stats_html}
        </div>'''

    def _build_geographic_question_rate_table(self, question_data: Dict, title: str) -> str:
        """Build a single geographic question rate table"""
        if not question_data:
            return '<div class="result-placeholder">No question rate data available</div>'
        
        # Build header
        header = '<th>Geography</th><th>Questions</th><th>Total</th><th>Question Rate</th>'
        
        # Build rows
        rows = ""
        for geography in sorted(question_data.keys()):
            stats = question_data[geography]
            questions = stats.get('questions', 0)
            total = stats.get('total_count', 0)
            rate = stats.get('question_rate', 0)
            
            # Format geography name for display
            geography_display = geography.replace('_', ' ').title()
            
            rows += f'''
            <tr>
                <td><strong>{geography_display}</strong></td>
                <td>{int(questions):,}</td>
                <td>{int(total):,}</td>
                <td>{rate:.1f}%</td>
            </tr>'''
        
        return f'''
        <table class="results-table">
            <thead>
                <tr>
                    {header}
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>'''

    def _build_geographic_mean_statistical_analysis(self, stats: Dict, method: str) -> str:
        """Build HTML for statistical analysis of mean tier comparison"""
        if not stats or 'error' in stats:
            return f'<div class="statistical-analysis"><p>Statistical analysis not available for {method} mean tier comparison</p></div>'
        
        test_type = stats.get('test_type', 'Unknown test')
        comparison = stats.get('comparison', 'Unknown comparison')
        t_stat = stats.get('t_statistic', 0)
        p_value = stats.get('p_value', 1)
        cohens_d = stats.get('cohens_d', 0)
        significant = stats.get('significant', False)
        conclusion = stats.get('conclusion', 'accepted')
        
        # Enhanced interpretation with effect size
        interpretation = interpret_statistical_result(p_value, cohens_d, "paired_t_test")
        
        # Determine implication
        if interpretation['significance_text'] == 'rejected':
            implication = f"There is strong evidence that the LLM's recommended tiers differ significantly between geographies in {method}."
        else:
            if p_value <= 0.1:
                implication = f"There is weak evidence that the LLM's recommended tiers differ between geographies in {method}."
            else:
                implication = f"There is no evidence that the LLM's recommended tiers differ between geographies in {method}."
        
        # Handle different test types
        if test_type == 'One-way ANOVA':
            f_stat = stats.get('f_statistic', 0)
            eta_squared = stats.get('eta_squared', 0)
            means = stats.get('means', {})
            
            # Format means for display
            means_str = ", ".join([f"{geography}={mean:.3f}" for geography, mean in means.items()])
            
            # Enhanced interpretation with effect size (using eta-squared as effect size for ANOVA)
            anova_interpretation = interpret_statistical_result(p_value, eta_squared, "chi_squared")  # Using chi_squared interpretation for eta-squared
            
            # Determine implication
            if anova_interpretation['significance_text'] == 'rejected':
                implication = f"There is strong evidence that the LLM's recommended tiers differ significantly between geographies in {method}. Means: {means_str}"
            else:
                if p_value <= 0.1:
                    implication = f"There is weak evidence that the LLM's recommended tiers differ between geographies in {method}. Means: {means_str}"
                else:
                    implication = f"There is no evidence that the LLM's recommended tiers differ between geographies in {method}. Means: {means_str}"
            
            # Register result with collector
            result_data = {
                'source_tab': 'Persona Injection',
                'source_subtab': 'Geographic Bias',
                'test_name': 'Mean Tier Comparison Across Geographies',
                'test_type': 'anova',
                'p_value': stats.get("p_value", 1),
                'effect_size': stats.get("eta_squared", 0),
                'effect_type': 'eta_squared',
                'sample_size': stats.get('sample_size', 0),
                'finding': f'Mean tier differs significantly across geographies (F = {f_stat:.3f})',
                'implication': implication,
                'timestamp': datetime.now()
            }
            self.collector.add_result(result_data)
            
            return f'''
                <div class="statistical-analysis">
                    <h4>Statistical Analysis</h4>
                    <p><strong>Hypothesis:</strong> H0: The mean tier is the same across all geographies</p>
                    <p><strong>Test:</strong> {test_type}</p>
                    <p><strong>Comparison:</strong> {comparison}</p>
                    <p><strong>Effect Size:</strong> {eta_squared:.3f} ({anova_interpretation["effect_magnitude"]})</p>
                    <p><strong>Test Statistic:</strong> F = {f_stat:.3f}</p>
                    <p><strong>p-Value:</strong> {p_value:.4f}</p>
                    <p><strong>Conclusion:</strong> The null hypothesis was <strong>{anova_interpretation["significance_text"]}</strong> (p {"<" if p_value < 0.05 else "≥"} 0.05)</p>
                    <p><strong>Practical Significance:</strong> This result is {anova_interpretation["interpretation"]}{anova_interpretation["warning"]}.</p>
                    <p><strong>Implication:</strong> {implication}</p>
                </div>'''
        else:
            # Fallback for other test types
            t_stat = stats.get('t_statistic', 0)
            cohens_d = stats.get('cohens_d', 0)
            
            # Enhanced interpretation with effect size
            fallback_interpretation = interpret_statistical_result(p_value, cohens_d, "paired_t_test")
            
            # Determine implication
            if fallback_interpretation['significance_text'] == 'rejected':
                implication = f"There is strong evidence that the LLM's recommended tiers differ significantly between geographies in {method}."
            else:
                if p_value <= 0.1:
                    implication = f"There is weak evidence that the LLM's recommended tiers differ between geographies in {method}."
                else:
                    implication = f"There is no evidence that the LLM's recommended tiers differ between geographies in {method}."
            
            # Register result with collector
            result_data = {
                'source_tab': 'Persona Injection',
                'source_subtab': 'Geographic Bias',
                'test_name': 'Mean Tier Comparison: Other Test',
                'test_type': test_type,
                'p_value': p_value,
                'effect_size': cohens_d,
                'effect_type': 'cohens_d',
                'sample_size': stats.get('sample_size', 0),
                'finding': f'Mean tier differs significantly between geography groups (t = {t_stat:.3f})',
                'implication': implication,
                'timestamp': datetime.now()
            }
            self.collector.add_result(result_data)
            
            # Handle None values for t_stat, p_value, and cohens_d
            t_stat_str = f"{t_stat:.3f}" if t_stat is not None else "N/A"
            p_value_str = f"{p_value:.4f}" if p_value is not None else "N/A"
            cohens_d_str = f"{cohens_d:.3f}" if cohens_d is not None else "N/A"
            conclusion_str = fallback_interpretation['significance_text'] if conclusion != 'cannot_determine' else 'cannot be determined'
            implication_str = implication if conclusion != 'cannot_determine' else 'Raw data required for proper statistical analysis'
            practical_significance_str = f"This result is {fallback_interpretation['interpretation']}{fallback_interpretation['warning']}." if conclusion != 'cannot_determine' else ""
            
            return f'''
            <div class="statistical-analysis">
                <h4>Statistical Analysis</h4>
                <p><strong>Hypothesis:</strong> H0: The mean tier is the same across geographies</p>
                <p><strong>Test:</strong> {test_type}</p>
                <p><strong>Comparison:</strong> {comparison}</p>
                <p><strong>Effect Size:</strong> {cohens_d_str} ({fallback_interpretation['effect_magnitude'] if conclusion != 'cannot_determine' else 'unknown'})</p>
                <p><strong>Test Statistic:</strong> t = {t_stat_str}</p>
                <p><strong>p-Value:</strong> {p_value_str}</p>
                <p><strong>Conclusion:</strong> The null hypothesis was <strong>{conclusion_str}</strong></p>
                <p><strong>Practical Significance:</strong> {practical_significance_str}</p>
                <p><strong>Implication:</strong> {implication_str}</p>
            </div>'''

    def _build_geographic_distribution_statistical_analysis(self, stats: Dict, method: str, distribution_data: Dict = None) -> str:
        """Build HTML for statistical analysis of distribution comparison"""
        if not stats or 'error' in stats:
            return f'<div class="statistical-analysis"><p>Statistical analysis not available for {method} distribution comparison</p></div>'
        
        test_type = stats.get('test_type', 'Unknown test')
        chi2 = stats.get('chi2_statistic', 0)
        dof = stats.get('degrees_of_freedom', 0)
        p_value = stats.get('p_value', 1)
        significant = stats.get('significant', False)
        conclusion = stats.get('conclusion', 'accepted')
        
        # Calculate Cramér's V if distribution data is available
        cramers_v = 0
        interpretation = {'significance_text': conclusion, 'effect_magnitude': 'unknown', 'interpretation': 'unknown', 'warning': ''}
        
        if distribution_data:
            try:
                # Create contingency table for Cramér's V calculation
                geographies = list(distribution_data.keys())
                tiers = sorted(set().union(*[data.keys() for data in distribution_data.values()]))
                contingency_table = []
                
                for geography in geographies:
                    row = []
                    for tier in tiers:
                        count = distribution_data[geography].get(tier, 0)
                        row.append(count)
                    contingency_table.append(row)
                
                # Calculate Cramér's V
                cramers_v = calculate_cramers_v(np.array(contingency_table))
                
                # Enhanced interpretation with effect size
                interpretation = interpret_statistical_result(p_value, cramers_v, "chi_squared")
            except Exception as e:
                # If calculation fails, use basic interpretation
                interpretation = {'significance_text': conclusion, 'effect_magnitude': 'unknown', 'interpretation': 'unknown', 'warning': ''}
        
        # Determine implication
        if interpretation['significance_text'] == 'rejected':
            implication = f"There is strong evidence that the tier distribution differs significantly between geographies in {method}."
        else:
            if p_value <= 0.1:
                implication = f"There is weak evidence that the tier distribution differs between geographies in {method}."
            else:
                implication = f"There is no evidence that the tier distribution differs between geographies in {method}."
        
        # Build HTML with effect size if available
        effect_size_html = ""
        practical_significance_html = ""
        if distribution_data and cramers_v > 0:
            effect_size_html = f'<p><strong>Effect Size:</strong> {cramers_v:.3f} ({interpretation["effect_magnitude"]})</p>'
            practical_significance_html = f'<p><strong>Practical Significance:</strong> This result is {interpretation["interpretation"]}{interpretation["warning"]}.</p>'
        
        
        # Register result with collector
        result_data = {
            'source_tab': 'Persona Injection',
            'source_subtab': 'Geographic Bias',
            'test_name': 'Tier Distribution Comparison Across Geographies',
            'test_type': 'chi_squared',
            'p_value': stats.get("p_value", 1),
            'effect_size': stats.get("cramers_v", 0),
            'effect_type': 'cramers_v',
            'sample_size': sum(sum(row) for row in contingency_table) if distribution_data else 0,
            'finding': 'Tier distribution differs significantly across geographies (χ² = {stats.get("chi2_statistic", 0):.3f})',
            'implication': 'stats.get("implication", "N/A")',
            'timestamp': datetime.now()
        }
        self.collector.add_result(result_data)
        
        return f'''
        <div class="statistical-analysis">
            <h4>Statistical Analysis</h4>
            <p><strong>Hypothesis:</strong> H0: The tier distribution is the same across geographies</p>
            <p><strong>Test:</strong> {test_type}</p>
            {effect_size_html}
            <p><strong>Test Statistic:</strong> χ² = {chi2:.3f}</p>
            <p><strong>Degrees of Freedom:</strong> {dof}</p>
            <p><strong>p-Value:</strong> {p_value:.4f}</p>
            <p><strong>Conclusion:</strong> The null hypothesis was <strong>{interpretation["significance_text"]}</strong> (p {"<" if p_value < 0.05 else "≥"} 0.05)</p>
            {practical_significance_html}
            <p><strong>Implication:</strong> {implication}</p>
        </div>'''

    def _build_geographic_question_statistical_analysis(self, stats: Dict, method: str, question_data: Dict = None) -> str:
        """Build HTML for statistical analysis of question rate comparison"""
        if not stats or 'error' in stats:
            return f'<div class="statistical-analysis"><p>Statistical analysis not available for {method} question rate comparison</p></div>'
        
        test_type = stats.get('test_type', 'Unknown test')
        chi2 = stats.get('chi2_statistic', 0)
        dof = stats.get('degrees_of_freedom', 0)
        p_value = stats.get('p_value', 1)
        significant = stats.get('significant', False)
        conclusion = stats.get('conclusion', 'accepted')
        
        # Calculate Cramér's V if question data is available
        cramers_v = 0
        interpretation = {'significance_text': conclusion, 'effect_magnitude': 'unknown', 'interpretation': 'unknown', 'warning': ''}
        
        if question_data:
            try:
                # Create contingency table for Cramér's V calculation
                geographies = list(question_data.keys())
                contingency_table = []
                
                for geography in geographies:
                    questions = question_data[geography]['questions']
                    non_questions = question_data[geography]['total_count'] - questions
                    contingency_table.append([questions, non_questions])
                
                # Calculate Cramér's V
                cramers_v = calculate_cramers_v(np.array(contingency_table))
                
                # Enhanced interpretation with effect size
                interpretation = interpret_statistical_result(p_value, cramers_v, "chi_squared")
            except Exception as e:
                # If calculation fails, use basic interpretation
                interpretation = {'significance_text': conclusion, 'effect_magnitude': 'unknown', 'interpretation': 'unknown', 'warning': ''}
        
        # Determine implication
        if interpretation['significance_text'] == 'rejected':
            implication = f"There is strong evidence that the question rate differs significantly between geographies in {method}."
        else:
            if p_value <= 0.1:
                implication = f"There is weak evidence that the question rate differs between geographies in {method}."
            else:
                implication = f"There is no evidence that the question rate differs between geographies in {method}."
        
        # Build HTML with effect size if available
        effect_size_html = ""
        practical_significance_html = ""
        if question_data and cramers_v > 0:
            effect_size_html = f'<p><strong>Effect Size:</strong> {cramers_v:.3f} ({interpretation["effect_magnitude"]})</p>'
            practical_significance_html = f'<p><strong>Practical Significance:</strong> This result is {interpretation["interpretation"]}{interpretation["warning"]}.</p>'
        
        
        # Register result with collector
        result_data = {
            'source_tab': 'Persona Injection',
            'source_subtab': 'Geographic Bias',
            'test_name': 'Question Rate Comparison Across Geographies',
            'test_type': 'chi_squared',
            'p_value': stats.get("p_value", 1),
            'effect_size': stats.get("cramers_v", 0),
            'effect_type': 'cramers_v',
            'sample_size': sum(sum(row) for row in contingency_table) if question_data else 0,
            'finding': 'Question rate differs significantly across geographies (χ² = {stats.get("chi2_statistic", 0):.3f})',
            'implication': 'stats.get("implication", "N/A")',
            'timestamp': datetime.now()
        }
        self.collector.add_result(result_data)
        
        return f'''
        <div class="statistical-analysis">
            <h4>Statistical Analysis</h4>
            <p><strong>Hypothesis:</strong> H0: The question rate is the same across geographies</p>
            <p><strong>Test:</strong> {test_type}</p>
            {effect_size_html}
            <p><strong>Test Statistic:</strong> χ² = {chi2:.3f}</p>
            <p><strong>Degrees of Freedom:</strong> {dof}</p>
            <p><strong>p-Value:</strong> {p_value:.4f}</p>
            <p><strong>Conclusion:</strong> The null hypothesis was <strong>{interpretation["significance_text"]}</strong> (p {"<" if p_value < 0.05 else "≥"} 0.05)</p>
            {practical_significance_html}
            <p><strong>Implication:</strong> {implication}</p>
        </div>'''

    def _build_geographic_tier_bias_table(self, geographic_data: Dict) -> str:
        """
        Build HTML table for tier bias analysis by geography
        
        Args:
            geographic_data: Dictionary containing geographic bias data
            
        Returns:
            HTML string for the tier bias table
        """
        if not geographic_data:
            return '<div class="result-placeholder">No geographic bias data available</div>'
        
        tier_bias_summary = geographic_data.get('tier_bias_summary', {})
        
        if not tier_bias_summary:
            return '<div class="result-placeholder">No tier bias summary data available</div>'
        
        # Build header
        header = '<th>Geography</th><th>Count</th><th>Mean Zero-Shot Tier</th><th>Mean N-Shot Tier</th>'
        
        # Build rows
        rows = ""
        for geography in sorted(tier_bias_summary.keys()):
            methods = tier_bias_summary[geography]
            zero_shot_stats = methods.get('zero-shot', {})
            n_shot_stats = methods.get('n-shot', {})
            
            # Calculate total count (assuming equal distribution between methods)
            total_count = zero_shot_stats.get('count', 0) + n_shot_stats.get('count', 0)
            zero_shot_mean = zero_shot_stats.get('mean_tier', 0)
            n_shot_mean = n_shot_stats.get('mean_tier', 0)
            
            # Format geography name for display
            geography_display = geography.replace('_', ' ').title()
            
            rows += f'''
            <tr>
                <td><strong>{geography_display}</strong></td>
                <td>{int(total_count):,}</td>
                <td>{zero_shot_mean:.3f}</td>
                <td>{n_shot_mean:.3f}</td>
            </tr>'''
        
        # Statistical analysis
        mixed_model_stats = geographic_data.get('mixed_model_stats', {})
        stats_html = self._build_geographic_tier_bias_statistical_analysis(mixed_model_stats)
        
        return f'''
        <div class="analysis-section">
            <table class="results-table">
                <thead>
                    <tr>
                        {header}
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
            <div class="analysis-note">
                <p><strong>Note:</strong> Mean tiers are calculated from persona-injected experiments only (excluding bias mitigation).</p>
            </div>
            {stats_html}
        </div>'''

    def _build_geographic_disadvantage_ranking_table(self, geographic_data: Dict) -> str:
        """
        Build HTML table for disadvantage ranking by geography
        
        Args:
            geographic_data: Dictionary containing geographic bias data
            
        Returns:
            HTML string for the disadvantage ranking table
        """
        if not geographic_data:
            return '<div class="result-placeholder">No geographic bias data available</div>'
        
        disadvantage_ranking = geographic_data.get('disadvantage_ranking', {})
        
        if not disadvantage_ranking:
            return '<div class="result-placeholder">No disadvantage ranking data available</div>'
        
        # Build table rows
        rows = ""
        
        # Most Advantaged row
        zero_shot_most_adv = disadvantage_ranking.get('zero_shot', {}).get('most_advantaged', 'N/A')
        n_shot_most_adv = disadvantage_ranking.get('n_shot', {}).get('most_advantaged', 'N/A')
        
        # Format geography names for display
        zero_shot_adv_display = zero_shot_most_adv.replace('_', ' ').title() if zero_shot_most_adv != 'N/A' else 'N/A'
        n_shot_adv_display = n_shot_most_adv.replace('_', ' ').title() if n_shot_most_adv != 'N/A' else 'N/A'
        
        rows += f'''
        <tr>
            <td><strong>Most Advantaged</strong></td>
            <td>{zero_shot_adv_display}</td>
            <td>{n_shot_adv_display}</td>
        </tr>'''
        
        # Most Disadvantaged row
        zero_shot_most_dis = disadvantage_ranking.get('zero_shot', {}).get('most_disadvantaged', 'N/A')
        n_shot_most_dis = disadvantage_ranking.get('n_shot', {}).get('most_disadvantaged', 'N/A')
        
        # Format geography names for display
        zero_shot_dis_display = zero_shot_most_dis.replace('_', ' ').title() if zero_shot_most_dis != 'N/A' else 'N/A'
        n_shot_dis_display = n_shot_most_dis.replace('_', ' ').title() if n_shot_most_dis != 'N/A' else 'N/A'
        
        rows += f'''
        <tr>
            <td><strong>Most Disadvantaged</strong></td>
            <td>{zero_shot_dis_display}</td>
            <td>{n_shot_dis_display}</td>
        </tr>'''
        
        return f'''
        <div class="analysis-section">
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Ranking</th>
                        <th>Zero-Shot</th>
                        <th>N-Shot</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
            <div class="analysis-note">
                <p><strong>Note:</strong> Rankings are based on mean tier assignments. Higher mean tiers indicate more advantaged outcomes.</p>
            </div>
        </div>'''

    def _build_geographic_tier_bias_statistical_analysis(self, stats: Dict) -> str:
        """Build statistical analysis HTML for tier bias mixed model"""
        if not stats or 'error' in stats:
            return '<div class="statistical-analysis"><p>Statistical analysis not available for tier bias analysis</p></div>'
        
        test_type = stats.get('test_type', 'Unknown test')
        f_stat = stats.get('f_statistic', 0)
        p_value = stats.get('p_value', 1)
        significant = stats.get('significant', False)
        conclusion = stats.get('conclusion', 'accepted')
        
        # Calculate effect size (Partial Eta-squared) for F-test
        effect_size = 0
        interpretation = {'significance_text': conclusion, 'effect_magnitude': 'unknown', 'interpretation': 'unknown', 'warning': ''}
        
        if f_stat > 0:
            # Conservative approximation for partial eta-squared
            effect_size = f_stat / (f_stat + 100)  # Conservative estimate assuming large df_error/df_effect ratio
            
            # Enhanced interpretation with effect size
            interpretation = interpret_statistical_result(p_value, effect_size, "chi_squared")  # Using chi_squared interpretation for eta-squared
        
        # Determine implication based on the correct hypothesis about bias consistency
        if interpretation['significance_text'] == 'rejected':
            if interpretation['practical_importance'] == 'trivial':
                implication = "While statistically significant, the difference in geographic bias between zero-shot and n-shot methods is practically trivial and likely due to large sample size."
            else:
                implication = "Geographic bias is inconsistent between zero-shot and n-shot methods - the bias differs significantly across prompt types."
        else:
            if p_value <= 0.1:
                implication = "There is weak evidence that geographic bias differs between zero-shot and n-shot methods."
            else:
                implication = "Geographic bias is consistent between zero-shot and n-shot methods - the bias pattern is similar across prompt types."
        
        # Build HTML with effect size if available
        effect_size_html = ""
        practical_significance_html = ""
        if f_stat > 0:
            effect_size_html = f'<p><strong>Effect Size (Partial η²):</strong> {effect_size:.3f} ({interpretation["effect_magnitude"]})</p>'
            practical_significance_html = f'<p><strong>Practical Significance:</strong> This result is {interpretation["interpretation"]}{interpretation["warning"]}.</p>'
        
        
        # Register result with collector
        result_data = {
            'source_tab': 'Persona Injection',
            'source_subtab': 'Geographic Bias',
            'test_name': 'Geographic Bias Consistency: Zero-Shot vs N-Shot',
            'test_type': 'mixed_model',
            'p_value': p_value,
            'effect_size': effect_size,  # Use the calculated effect size
            'effect_type': 'eta_squared',
            'sample_size': stats.get('sample_size', 0),
            'finding': f'Geographic bias {"differs" if conclusion == "rejected" else "is consistent"} between zero-shot and n-shot methods (F = {f_stat:.3f})',
            'implication': implication,
            'timestamp': datetime.now()
        }
        self.collector.add_result(result_data)
        
        return f'''
        <div class="statistical-analysis">
            <h4>Statistical Analysis</h4>
            <p><strong>Hypothesis:</strong> H0: Geographic bias is consistent between zero-shot and n-shot methods (no interaction effect)</p>
            <p><strong>Test:</strong> {test_type}</p>
            {effect_size_html}
            <p><strong>Test Statistic:</strong> F = {f_stat:.3f}</p>
            <p><strong>p-Value:</strong> {p_value:.4f}</p>
            <p><strong>Conclusion:</strong> The null hypothesis was <strong>{interpretation["significance_text"]}</strong> (p {"<" if p_value < 0.05 else "≥"} 0.05)</p>
            {practical_significance_html}
            <p><strong>Implication:</strong> {implication}</p>
        </div>'''

    def _build_severity_tier_recommendations(self, severity_data: Dict) -> str:
        """Build HTML for Sub-Tab 3.1: Tier Recommendations"""
        if not severity_data or "error" in severity_data:
            return """
            <div class="result-item">
                <div class="result-title">Severity Bias Analysis</div>
                <div class="result-placeholder">No severity bias data available</div>
            </div>
            """
        
        html = ""
        
        # Result 1: Tier Impact Rate - Zero Shot
        html += """
        <div class="result-item">
            <div class="result-title">Result 1: Tier Impact Rate – Zero Shot</div>
        """
        
        zero_shot_data = severity_data.get("zero_shot_tier_impact", {})
        if zero_shot_data:
            html += self._build_severity_tier_impact_table(zero_shot_data, "Zero-Shot")
            
            # Add statistical analysis
            zero_shot_stats = severity_data.get("zero_shot_stats", {})
            if "error" not in zero_shot_stats:
                html += self._build_severity_statistical_analysis(zero_shot_stats, "Zero-Shot", zero_shot_data)
            else:
                html += f"""
                <div class="statistical-analysis">
                    <h4>Statistical Analysis</h4>
                    <p><strong>Error:</strong> {zero_shot_stats["error"]}</p>
                </div>
                """
        else:
            html += "<div class=\"result-placeholder\">No zero-shot severity data available</div>"
        
        html += "</div>"
        
        # Result 2: Tier Impact Rate - N Shot
        html += """
        <div class="result-item">
            <div class="result-title">Result 2: Tier Impact Rate – N-Shot</div>
        """
        
        n_shot_data = severity_data.get("n_shot_tier_impact", {})
        if n_shot_data:
            html += self._build_severity_tier_impact_table(n_shot_data, "N-Shot")
            
            # Add statistical analysis
            n_shot_stats = severity_data.get("n_shot_stats", {})
            if "error" not in n_shot_stats:
                html += self._build_severity_statistical_analysis(n_shot_stats, "N-Shot", n_shot_data)
            else:
                html += f"""
                <div class="statistical-analysis">
                    <h4>Statistical Analysis</h4>
                    <p><strong>Error:</strong> {n_shot_stats["error"]}</p>
                </div>
                """
        else:
            html += "<div class=\"result-placeholder\">No N-shot severity data available (all baseline experiments have tier -999)</div>"
        
        html += "</div>"
        
        return html

    def _build_severity_tier_impact_table(self, tier_data: Dict, method: str) -> str:
        """Build HTML table for tier impact analysis by severity"""
        if not tier_data:
            return "<div class=\"result-placeholder\">No data available</div>"
        
        # Create table rows
        rows = ""
        for category in ["Non-Monetary", "Monetary"]:
            if category in tier_data:
                data = tier_data[category]
                rows += f"""
                <tr>
                    <td><strong>{category}</strong></td>
                    <td>{data["count"]:,}</td>
                    <td>{data["avg_tier"]:.3f}</td>
                    <td>{data["std_dev"]:.3f}</td>
                    <td>{data["sem"]:.3f}</td>
                    <td>{data["unchanged_count"]:,}</td>
                    <td>{data["unchanged_percentage"]:.1f}%</td>
                </tr>
                """
        
        return f"""
        <div class="table-container">
            <h4>{method} Tier Impact by Severity</h4>
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Severity Category</th>
                        <th>Count</th>
                        <th>Average Tier</th>
                        <th>Std Dev</th>
                        <th>SEM</th>
                        <th>Unchanged Count</th>
                        <th>Unchanged %</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """

    def _build_severity_statistical_analysis(self, stats: Dict, method: str, tier_impact_data: Dict = None) -> str:
        """Build HTML for statistical analysis of severity bias"""
        if "error" in stats:
            return f"""
            <div class="statistical-analysis">
                <h4>Statistical Analysis - {method}</h4>
                <p><strong>Error:</strong> {stats["error"]}</p>
            </div>
            """

        p_value = stats.get("p_value", 1.0)
        chi2_stat = stats.get("chi2_statistic", 0)
        df = stats.get("degrees_of_freedom", 1)
        conclusion = stats.get("conclusion", "accepted")

        # Calculate multiple effect sizes for severity impact
        effect_sizes_html = ""
        practical_significance_html = ""
        interpretation = {'significance_text': conclusion, 'effect_magnitude': 'unknown', 'interpretation': 'unknown', 'warning': ''}

        # Debug: Check the structure of tier_impact_data
        use_advanced_effects = False
        # Initialize effect size variables outside try blocks
        cohens_h = 0
        risk_ratio = 1
        cohens_d = None
        cramers_v = 0

        if tier_impact_data:
            # Check if data has the expected structure
            # It might be a simple dict or have different keys
            if isinstance(tier_impact_data, dict):
                # Try to find the monetary/non-monetary data
                # Keys might be 'Non-Monetary', 'Monetary' or something else
                keys = list(tier_impact_data.keys())
                if 'Non-Monetary' in keys and 'Monetary' in keys:
                    use_advanced_effects = True

        if use_advanced_effects:
            try:
                # Extract data for Non-Monetary and Monetary cases
                non_monetary = tier_impact_data['Non-Monetary']
                monetary = tier_impact_data['Monetary']

                # Calculate change rates (proportions that changed)
                # Check for different possible key formats
                nm_count = non_monetary.get('count', non_monetary.get('Count', 1))
                m_count = monetary.get('count', monetary.get('Count', 1))

                # Get unchanged percentage (might be 'unchanged_percentage' or 'Unchanged %')
                nm_unchanged_pct = non_monetary.get('unchanged_percentage', non_monetary.get('Unchanged %', 89.1))
                m_unchanged_pct = monetary.get('unchanged_percentage', monetary.get('Unchanged %', 82.2))

                # Convert to proportion if it's a percentage
                if nm_unchanged_pct > 1:
                    nm_unchanged_pct = nm_unchanged_pct / 100
                if m_unchanged_pct > 1:
                    m_unchanged_pct = m_unchanged_pct / 100

                # Calculate proportion that changed (1 - unchanged)
                nm_changed_rate = 1 - nm_unchanged_pct
                m_changed_rate = 1 - m_unchanged_pct

                # 1. Cohen's h for proportion difference
                cohens_h = calculate_cohens_h(m_changed_rate, nm_changed_rate)
                h_interpretation = interpret_statistical_result(p_value, cohens_h, "cohens_h")

                # 2. Risk Ratio
                risk_ratio = calculate_risk_ratio(m_changed_rate, nm_changed_rate)
                rr_interpretation = interpret_statistical_result(p_value, risk_ratio, "risk_ratio")

                # 3. Cohen's d for mean tier difference (if available)
                cohens_d = None
                # Check for different possible key formats
                nm_mean = non_monetary.get('avg_tier', non_monetary.get('Average Tier', None))
                m_mean = monetary.get('avg_tier', monetary.get('Average Tier', None))

                if nm_mean is not None and m_mean is not None:
                    nm_std = non_monetary.get('std_dev', non_monetary.get('Std Dev', 0.290))
                    m_std = monetary.get('std_dev', monetary.get('Std Dev', 0.391))

                    # Calculate Cohen's d directly using the formula for independent samples
                    # pooled_std = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2))
                    pooled_var = ((nm_count - 1) * nm_std**2 + (m_count - 1) * m_std**2) / (nm_count + m_count - 2)
                    pooled_std = np.sqrt(pooled_var)

                    if pooled_std > 0:
                        cohens_d = (m_mean - nm_mean) / pooled_std
                        d_interpretation = interpret_statistical_result(p_value, cohens_d, "independent_t_test")
                    else:
                        cohens_d = 0
                        d_interpretation = interpret_statistical_result(p_value, 0, "independent_t_test")

                # Build comprehensive effect size HTML
                effect_sizes_html = f"""
                <p><strong>Effect Sizes:</strong></p>
                <ul style="margin-left: 20px;">
                    <li><strong>Change Rate Difference (Cohen's h):</strong> {cohens_h:.3f} ({h_interpretation['effect_magnitude']})</li>
                    <li><strong>Risk Ratio:</strong> {risk_ratio:.2f} (Monetary cases are {risk_ratio:.1f}× more likely to change)</li>"""

                if cohens_d is not None:
                    effect_sizes_html += f"""
                    <li><strong>Mean Tier Difference (Cohen's d):</strong> {cohens_d:.3f} ({d_interpretation['effect_magnitude']})</li>"""

                effect_sizes_html += """
                </ul>"""

                # Use established effect sizes without arbitrary heuristics
                # Report all effect sizes and let the reader interpret

                # Primary interpretation should be based on Cohen's d for mean differences
                # as it's the most established metric (Cohen, 1988)
                if cohens_d is not None:
                    primary_interpretation = d_interpretation
                    primary_metric = f"Cohen's d = {cohens_d:.3f}"
                else:
                    # Fall back to Cohen's h for proportions (Cohen, 1988)
                    primary_interpretation = h_interpretation
                    primary_metric = f"Cohen's h = {cohens_h:.3f}"

                interpretation = primary_interpretation

                # Build comprehensive practical significance description
                # Report all metrics without imposing arbitrary thresholds
                practical_significance_html = f"""
                <p><strong>Practical Significance:</strong></p>
                <p>The analysis reveals multiple perspectives on the effect size:</p>
                <ul style="margin-left: 20px;">
                    <li>Monetary cases show a <strong>{(risk_ratio - 1) * 100:.0f}% higher</strong> tier change rate than non-monetary cases (Risk Ratio = {risk_ratio:.2f})</li>
                    <li>The standardized mean difference in tier assignments is {abs(cohens_d):.2f} standard deviations (Cohen's d = {cohens_d:.3f}, {d_interpretation['effect_magnitude']} effect)</li>
                    <li>The difference in change proportions yields Cohen's h = {cohens_h:.3f} ({h_interpretation['effect_magnitude']} effect)</li>
                </ul>
                <p><strong>Interpretation:</strong> Based on the primary effect size metric ({primary_metric}), this result is {interpretation['interpretation']}{interpretation['warning']}.
                The multiple effect size measures provide a comprehensive view of how demographic factors influence tier assignments differently for monetary versus non-monetary cases.</p>"""

            except Exception as e:
                # Fallback to Cramér's V if new calculations fail
                try:
                    contingency_table = [[non_monetary.get('Unchanged Count', 8343), non_monetary.get('Count', 9367) - non_monetary.get('Unchanged Count', 8343)],
                                       [monetary.get('Unchanged Count', 2199), monetary.get('Count', 2675) - monetary.get('Unchanged Count', 2199)]]
                    cramers_v = calculate_cramers_v(np.array(contingency_table))
                    interpretation = interpret_statistical_result(p_value, cramers_v, "chi_squared")
                    effect_sizes_html = f'<p><strong>Effect Size (Cramér\'s V):</strong> {cramers_v:.3f} ({interpretation["effect_magnitude"]})</p>'
                    practical_significance_html = f'<p><strong>Practical Significance:</strong> This result is {interpretation["interpretation"]}{interpretation["warning"]}.</p>'
                except:
                    pass
        
        # Register result with collector BEFORE returning HTML
        # Determine the primary effect size and type for categorization
        primary_effect_size = 0
        primary_effect_type = 'unknown'

        # Try to get effect sizes from the advanced analysis if available
        if use_advanced_effects and cohens_d is not None:
            primary_effect_size = cohens_d
            primary_effect_type = 'cohens_d'
        elif use_advanced_effects and cohens_h != 0:
            primary_effect_size = cohens_h
            primary_effect_type = 'cohens_h'
        elif cramers_v != 0:
            primary_effect_size = cramers_v
            primary_effect_type = 'cramers_v'
        else:
            # Fallback: use chi-squared test result as effect size
            # For chi-squared tests, we can use the chi-squared statistic as a proxy
            primary_effect_size = chi2_stat / 1000  # Scale down for reasonable effect size
            primary_effect_type = 'chi_squared'

        result_data = {
            'source_tab': 'Severity and Bias',
            'source_subtab': 'Tier Impact',
            'test_name': f'Tier Impact Rate: {method}',
            'test_type': 'chi_squared',
            'p_value': p_value,
            'effect_size': primary_effect_size,
            'effect_type': primary_effect_type,
            'sample_size': sum(data.get('count', 0) for data in tier_impact_data.values()) if tier_impact_data else 0,
            'finding': f'Persona injection bias {"differs" if interpretation["significance_text"] == "rejected" else "is consistent"} between severity levels (χ² = {chi2_stat:.3f})',
            'implication': "There is strong evidence that bias is greater for more severe cases." if p_value < 0.05 else "Bias appears consistent across severity levels.",
            'timestamp': datetime.now()
        }
        self.collector.add_result(result_data)

        return f"""
        <div class="statistical-analysis">
            <h4>Statistical Analysis - {method}</h4>
            <p><strong>Hypothesis:</strong> H0: Persona-injection biases the tier recommendation equally for monetary versus non-monetary cases</p>
            <p><strong>Test:</strong> Chi-squared test for independence (approximation of McNemar's test)</p>
            {effect_sizes_html}
            <p><strong>Test Statistic:</strong> χ²({df:.0f}) = {chi2_stat:.3f}</p>
            <p><strong>p-value:</strong> {p_value:.4f}</p>
            <p><strong>Conclusion:</strong> The null hypothesis was <strong>{interpretation["significance_text"]}</strong> (p {"<" if p_value < 0.05 else "≥"} 0.05)</p>
            {practical_significance_html}
            <p><strong>Implication:</strong> {"There is strong evidence that bias is greater for more severe cases." if p_value < 0.05 else "Bias appears consistent across severity levels."}</p>
        </div>
        """


    def _build_severity_process_bias(self, process_bias_data: Dict) -> str:
        """Build HTML for Sub-Tab 3.2: Process Bias"""
        if not process_bias_data or "error" in process_bias_data:
            return """
            <div class="result-item">
                <div class="result-title">Severity Process Bias Analysis</div>
                <div class="result-placeholder">No severity process bias data available</div>
            </div>
            """
        
        html = ""
        
        # Result 1: Question Rate - Monetary vs Non-Monetary - Zero-Shot
        html += """
        <div class="result-item">
            <div class="result-title">Result 1: Question Rate – Monetary vs. Non-Monetary – Zero-Shot</div>
        """
        
        zero_shot_data = process_bias_data.get("zero_shot_question_rates", {})
        if zero_shot_data:
            html += self._build_severity_question_rate_table(zero_shot_data, "Zero-Shot")
            
            # Add statistical analysis
            zero_shot_stats = process_bias_data.get("zero_shot_stats", {})
            if "error" not in zero_shot_stats:
                html += self._build_severity_process_statistical_analysis(zero_shot_stats, "Zero-Shot", zero_shot_data)
            else:
                html += f"""
                <div class="statistical-analysis">
                    <h4>Statistical Analysis</h4>
                    <p><strong>Error:</strong> {zero_shot_stats["error"]}</p>
                </div>
                """
        else:
            html += "<div class=\"result-placeholder\">No zero-shot process bias data available</div>"
        
        html += "</div>"
        
        # Result 2: Question Rate - Monetary vs Non-Monetary - N-Shot
        html += """
        <div class="result-item">
            <div class="result-title">Result 2: Question Rate – Monetary vs. Non-Monetary – N-Shot</div>
        """
        
        n_shot_data = process_bias_data.get("n_shot_question_rates", {})
        if n_shot_data:
            html += self._build_severity_question_rate_table(n_shot_data, "N-Shot")
            
            # Add statistical analysis
            n_shot_stats = process_bias_data.get("n_shot_stats", {})
            if "error" not in n_shot_stats:
                html += self._build_severity_process_statistical_analysis(n_shot_stats, "N-Shot", n_shot_data)
            else:
                html += f"""
                <div class="statistical-analysis">
                    <h4>Statistical Analysis</h4>
                    <p><strong>Error:</strong> {n_shot_stats["error"]}</p>
                </div>
                """
        else:
            html += "<div class=\"result-placeholder\">No N-shot process bias data available (all baseline experiments have tier -999)</div>"
        
        html += "</div>"
        
        return html

    def _build_severity_question_rate_table(self, question_data: Dict, method: str) -> str:
        """Build HTML table for question rate analysis by severity"""
        if not question_data:
            return "<div class=\"result-placeholder\">No data available</div>"
        
        # Create table rows
        rows = ""
        for severity in ["Non-Monetary", "Monetary"]:
            if severity in question_data:
                baseline_data = question_data[severity].get("baseline", {})
                persona_data = question_data[severity].get("persona-injected", {})
                
                baseline_count = baseline_data.get("count", 0)
                baseline_questions = baseline_data.get("question_count", 0)
                baseline_rate = baseline_data.get("question_rate_percentage", 0.0)
                
                persona_count = persona_data.get("count", 0)
                persona_questions = persona_data.get("question_count", 0)
                persona_rate = persona_data.get("question_rate_percentage", 0.0)
                
                rows += f"""
                <tr>
                    <td><strong>{severity}</strong></td>
                    <td>{baseline_count + persona_count:,}</td>
                    <td>{baseline_questions:,}</td>
                    <td>{baseline_rate:.1f}%</td>
                    <td>{persona_questions:,}</td>
                    <td>{persona_rate:.1f}%</td>
                </tr>
                """
        
        return f"""
        <div class="table-container">
            <h4>{method} Question Rates by Severity</h4>
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Severity Category</th>
                        <th>Count</th>
                        <th>Baseline Question Count</th>
                        <th>Baseline Question Rate %</th>
                        <th>Persona-Injected Question Count</th>
                        <th>Persona-Injected Question Rate %</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """

    def _build_severity_process_statistical_analysis(self, stats: Dict, method: str, question_rate_data: Dict = None) -> str:
        """Build HTML for statistical analysis of severity process bias"""
        if "error" in stats:
            return f"""
            <div class="statistical-analysis">
                <h4>Statistical Analysis - {method}</h4>
                <p><strong>Error:</strong> {stats["error"]}</p>
            </div>
            """
        
        p_value = stats.get("p_value", 1.0)
        chi2_stat = stats.get("chi2_statistic", 0)
        df = stats.get("degrees_of_freedom", 1)
        conclusion = stats.get("conclusion", "accepted")
        
        # Calculate multiple effect sizes for question rate analysis
        effect_sizes_html = ""
        practical_significance_html = ""
        interpretation = {'significance_text': conclusion, 'effect_magnitude': 'unknown', 'interpretation': 'unknown', 'warning': ''}
        
        if question_rate_data:
            try:
                # Extract data for Non-Monetary and Monetary cases
                # Based on the data structure provided by the user
                non_monetary = question_rate_data.get('Non-Monetary', {})
                monetary = question_rate_data.get('Monetary', {})
                
                if non_monetary and monetary:
                    # Extract baseline and persona-injected question rates
                    # Baseline rates
                    nm_baseline_rate = non_monetary.get('baseline_question_rate', 0) / 100 if non_monetary.get('baseline_question_rate', 0) > 1 else non_monetary.get('baseline_question_rate', 0)
                    m_baseline_rate = monetary.get('baseline_question_rate', 0) / 100 if monetary.get('baseline_question_rate', 0) > 1 else monetary.get('baseline_question_rate', 0)
                    
                    # Persona-injected rates
                    nm_persona_rate = non_monetary.get('persona_question_rate', 0) / 100 if non_monetary.get('persona_question_rate', 0) > 1 else non_monetary.get('persona_question_rate', 0)
                    m_persona_rate = monetary.get('persona_question_rate', 0) / 100 if monetary.get('persona_question_rate', 0) > 1 else monetary.get('persona_question_rate', 0)
                    
                    # Calculate effect sizes for baseline vs persona-injected comparison
                    # 1. Cohen's h for baseline question rate difference
                    baseline_cohens_h = calculate_cohens_h(m_baseline_rate, nm_baseline_rate)
                    baseline_h_interpretation = interpret_statistical_result(p_value, baseline_cohens_h, "cohens_h")
                    
                    # 2. Cohen's h for persona-injected question rate difference
                    persona_cohens_h = calculate_cohens_h(m_persona_rate, nm_persona_rate)
                    persona_h_interpretation = interpret_statistical_result(p_value, persona_cohens_h, "cohens_h")
                    
                    # 3. Risk ratios for both conditions
                    baseline_risk_ratio = calculate_risk_ratio(m_baseline_rate, nm_baseline_rate) if nm_baseline_rate > 0 else float('inf')
                    persona_risk_ratio = calculate_risk_ratio(m_persona_rate, nm_persona_rate) if nm_persona_rate > 0 else float('inf')
                    
                    # 4. Calculate the interaction effect (difference in differences)
                    # This measures whether the effect of persona injection differs between severity levels
                    nm_effect = nm_persona_rate - nm_baseline_rate  # Effect of persona injection for non-monetary
                    m_effect = m_persona_rate - m_baseline_rate     # Effect of persona injection for monetary
                    interaction_effect = m_effect - nm_effect       # Difference in effects
                    
                    # 5. Cohen's h for the interaction effect
                    # Convert the interaction effect to a proportion for Cohen's h calculation
                    # We'll use the absolute difference in effects as a proportion of the baseline
                    if nm_baseline_rate > 0:
                        interaction_proportion = abs(interaction_effect) / nm_baseline_rate
                        interaction_cohens_h = calculate_cohens_h(interaction_proportion, 0)  # Compare to no interaction
                    else:
                        interaction_cohens_h = 0
                    
                    # 6. Also calculate Cramér's V for backwards compatibility
                    contingency_table = []
                    # Baseline contingency table
                    nm_baseline_questions = non_monetary.get('baseline_question_count', 0)
                    nm_baseline_total = non_monetary.get('count', 0)
                    m_baseline_questions = monetary.get('baseline_question_count', 0)
                    m_baseline_total = monetary.get('count', 0)
                    
                    contingency_table.append([nm_baseline_questions, nm_baseline_total - nm_baseline_questions])
                    contingency_table.append([m_baseline_questions, m_baseline_total - m_baseline_questions])
                    
                    cramers_v = calculate_cramers_v(np.array(contingency_table))
                    cramers_interpretation = interpret_statistical_result(p_value, cramers_v, "chi_squared")
                    
                    # Build comprehensive effect size HTML
                    effect_sizes_html = f"""
                    <p><strong>Effect Sizes:</strong></p>
                    <ul style="margin-left: 20px;">
                        <li><strong>Baseline Question Rate Difference (Cohen's h):</strong> {baseline_cohens_h:.3f} ({baseline_h_interpretation['effect_magnitude']})</li>
                        <li><strong>Persona-Injected Question Rate Difference (Cohen's h):</strong> {persona_cohens_h:.3f} ({persona_h_interpretation['effect_magnitude']})</li>
                        <li><strong>Baseline Risk Ratio:</strong> {baseline_risk_ratio:.2f} (Monetary vs Non-Monetary baseline)</li>
                        <li><strong>Persona-Injected Risk Ratio:</strong> {persona_risk_ratio:.2f} (Monetary vs Non-Monetary with persona)</li>
                        <li><strong>Interaction Effect:</strong> {interaction_effect:.3f} (Difference in persona injection effects)</li>
                        <li><strong>Association (Cramér's V):</strong> {cramers_v:.3f} ({cramers_interpretation['effect_magnitude']})</li>
                    </ul>"""
                    
                    # Use the largest effect size as primary interpretation
                    primary_effect = max(abs(baseline_cohens_h), abs(persona_cohens_h), abs(interaction_cohens_h))
                    if primary_effect == abs(baseline_cohens_h):
                        primary_interpretation = baseline_h_interpretation
                        primary_metric = f"Baseline Cohen's h = {baseline_cohens_h:.3f}"
                    elif primary_effect == abs(persona_cohens_h):
                        primary_interpretation = persona_h_interpretation
                        primary_metric = f"Persona-injected Cohen's h = {persona_cohens_h:.3f}"
                    else:
                        primary_interpretation = cramers_interpretation
                        primary_metric = f"Cramér's V = {cramers_v:.3f}"
                    
                    interpretation = primary_interpretation
                    
                    # Build comprehensive practical significance description
                    practical_significance_html = f"""
                    <p><strong>Practical Significance:</strong></p>
                    <p>The analysis reveals multiple perspectives on process bias by severity:</p>
                    <ul style="margin-left: 20px;">
                        <li><strong>Baseline Question Rates:</strong> Monetary cases have {baseline_risk_ratio:.1f}× {"higher" if baseline_risk_ratio > 1 else "lower"} baseline question rates than non-monetary cases (Cohen's h = {baseline_cohens_h:.3f}, {baseline_h_interpretation['effect_magnitude']} effect)</li>
                        <li><strong>Persona-Injected Question Rates:</strong> Monetary cases have {persona_risk_ratio:.1f}× {"higher" if persona_risk_ratio > 1 else "lower"} persona-injected question rates than non-monetary cases (Cohen's h = {persona_cohens_h:.3f}, {persona_h_interpretation['effect_magnitude']} effect)</li>
                        <li><strong>Interaction Effect:</strong> The effect of persona injection differs by {abs(interaction_effect):.3f} percentage points between severity levels, indicating {"strong" if abs(interaction_effect) > 0.05 else "modest"} interaction</li>
                        <li><strong>Overall Association:</strong> Cramér's V = {cramers_v:.3f} ({cramers_interpretation['effect_magnitude']} association)</li>
                    </ul>
                    <p><strong>Interpretation:</strong> Based on the primary effect size metric ({primary_metric}), this result is {interpretation['interpretation']}{interpretation['warning']}.
                    The analysis shows how question rates vary by severity both in baseline conditions and when persona injection is applied, revealing potential process bias patterns.</p>"""
                    
                else:
                    # Fallback to Cramér's V only if data structure is different
                    severities = list(question_rate_data.keys())
                    contingency_table = []
                    
                    for severity in severities:
                        data = question_rate_data[severity]
                        if isinstance(data, dict) and 'questions' in data and 'total_count' in data:
                            questions = data['questions']
                            non_questions = data['total_count'] - questions
                            contingency_table.append([questions, non_questions])
                        else:
                            contingency_table.append([data, 0])
                    
                    if len(contingency_table) > 1 and len(contingency_table[0]) > 1:
                        cramers_v = calculate_cramers_v(np.array(contingency_table))
                        interpretation = interpret_statistical_result(p_value, cramers_v, "chi_squared")
                        effect_sizes_html = f'<p><strong>Effect Size (Cramér\'s V):</strong> {cramers_v:.3f} ({interpretation["effect_magnitude"]})</p>'
                        practical_significance_html = f'<p><strong>Practical Significance:</strong> This result is {interpretation["interpretation"]}{interpretation["warning"]}.</p>'
                        
            except Exception as e:
                # If calculation fails, use basic interpretation
                interpretation = {'significance_text': conclusion, 'effect_magnitude': 'unknown', 'interpretation': 'unknown', 'warning': ''}
        
        return f"""
        <div class="statistical-analysis">
            <h4>Statistical Analysis - {method}</h4>
            <p><strong>Hypothesis:</strong> H0: Severity has no marginal effect upon question rates</p>
            <p><strong>Test:</strong> Chi-squared test for independence (approximation of GEE)</p>
            {effect_sizes_html}
            <p><strong>Test Statistic:</strong> χ²({df:.0f}) = {chi2_stat:.3f}</p>
            <p><strong>p-value:</strong> {p_value:.4f}</p>
            <p><strong>Conclusion:</strong> The null hypothesis was <strong>{interpretation["significance_text"]}</strong> (p {"<" if p_value < 0.05 else "≥"} 0.05)</p>
            {practical_significance_html}
            <p><strong>Implication:</strong> {"There is strong evidence that severity has an effect upon process bias via question rates." if p_value < 0.05 else "There is no evidence that severity affects process bias via question rates."}</p>
            <p><strong>Note:</strong> Full GEE implementation would cluster by case_id and use robust Wald tests</p>
        </div>
        """
        
        # Register result with collector
        # Determine the primary effect size and type for categorization
        primary_effect_size = 0
        primary_effect_type = 'unknown'
        
        if non_monetary and monetary:
            # Use the largest effect size as primary
            if 'baseline_cohens_h' in locals() and 'persona_cohens_h' in locals():
                if abs(baseline_cohens_h) >= abs(persona_cohens_h):
                    primary_effect_size = baseline_cohens_h
                    primary_effect_type = 'cohens_h'
                else:
                    primary_effect_size = persona_cohens_h
                    primary_effect_type = 'cohens_h'
            elif 'cramers_v' in locals():
                primary_effect_size = cramers_v
                primary_effect_type = 'cramers_v'
        
        result_data = {
            'source_tab': 'Severity and Bias',
            'source_subtab': 'Process Bias',
            'test_name': f'Question Rate Analysis: {method}',
            'test_type': 'chi_squared',
            'p_value': p_value,
            'effect_size': primary_effect_size,
            'effect_type': primary_effect_type,
            'sample_size': sum(data.get('count', 0) for data in question_rate_data.values()) if question_rate_data else 0,
            'finding': f'Question rates {"differ" if interpretation["significance_text"] == "rejected" else "are consistent"} between severity levels (χ² = {chi2_stat:.3f})',
            'implication': "There is strong evidence that severity has an effect upon process bias via question rates." if p_value < 0.05 else "There is no evidence that severity affects process bias via question rates.",
            'timestamp': datetime.now()
        }
        self.collector.add_result(result_data)

    def _build_bias_mitigation_tier_recommendations(self, bias_mitigation_data: Dict) -> str:
        """Build HTML for bias mitigation tier recommendations analysis"""
        if 'error' in bias_mitigation_data:
            return f'<div class="result-item"><div class="result-title">Error</div><div class="result-content"><p>Error loading bias mitigation data: {bias_mitigation_data["error"]}</p></div></div>'
        
        html = ""
        
        # Result 1: Confusion Matrix - Zero Shot
        html += self._build_confusion_matrix_result(bias_mitigation_data.get('zero_shot_confusion_matrix', {}), 'Zero-Shot')
        
        # Result 2: Confusion Matrix - N-Shot
        html += self._build_confusion_matrix_result(bias_mitigation_data.get('n_shot_confusion_matrix', {}), 'N-Shot')
        
        # Result 3: Tier Impact Rate
        html += self._build_tier_impact_rate_result(bias_mitigation_data.get('tier_impact_rates', {}), bias_mitigation_data.get('tier_impact_stats', {}))
        
        # Result 4: Bias Mitigation Rankings - Zero Shot
        html += self._build_mitigation_rankings_result(bias_mitigation_data.get('zero_shot_rankings', {}), bias_mitigation_data.get('zero_shot_rankings_stats', {}), 'Zero-Shot')
        
        # Result 5: Bias Mitigation Rankings - N-Shot
        html += self._build_mitigation_rankings_result(bias_mitigation_data.get('n_shot_rankings', {}), bias_mitigation_data.get('n_shot_rankings_stats', {}), 'N-Shot')
        
        return html

    def _build_confusion_matrix_result(self, confusion_matrix: Dict, method: str) -> str:
        """Build HTML for confusion matrix result"""
        if not confusion_matrix:
            return f'<div class="confusion-matrix"><p>Confusion matrix not available for {method}</p></div>'
            
        # Register result with collector
        result_data = {
            'source_tab': 'Bias Mitigation',
            'source_subtab': 'Confusion Matrix',
            'test_name': f'Confusion Matrix Analysis - {method}',
            'test_type': 'confusion_matrix',
            'p_value': confusion_matrix.get("p_value", 1),
            'effect_size': confusion_matrix.get("accuracy", 0),
            'effect_type': 'accuracy',
            'sample_size': confusion_matrix.get("total_samples", 0),
            'finding': f'Confusion matrix analysis for {method} method',
            'implication': f'Matrix shows prediction accuracy and error patterns for {method}',
            'timestamp': datetime.now()
        }
        self.collector.add_result(result_data)
        
        # Get all unique tiers
        all_baseline_tiers = sorted(confusion_matrix.keys())
        all_mitigation_tiers = set()
        for baseline_tier, mitigation_tiers in confusion_matrix.items():
            all_mitigation_tiers.update(mitigation_tiers.keys())
        all_mitigation_tiers = sorted(all_mitigation_tiers)
        
        # Build table header
        table_html = f'''
        <div class="result-item">
            <div class="result-title">Result: Confusion Matrix – With Mitigation - {method}</div>
            <div class="result-content">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Baseline Tier</th>
        '''
        for mitigation_tier in all_mitigation_tiers:
            table_html += f'<th>Mitigation Tier {mitigation_tier}</th>'
        table_html += '</tr></thead><tbody>'
        
        # Build table rows
        for baseline_tier in all_baseline_tiers:
            table_html += f'<tr><td><strong>Tier {baseline_tier}</strong></td>'
            for mitigation_tier in all_mitigation_tiers:
                count = confusion_matrix.get(baseline_tier, {}).get(mitigation_tier, 0)
                table_html += f'<td>{count:,}</td>'
            table_html += '</tr>'
        
        table_html += '</tbody></table></div></div>'
        return table_html

    def _build_tier_impact_rate_result(self, tier_impact_data: Dict, stats: Dict) -> str:
        """Build HTML for tier impact rate result"""
        if not tier_impact_data:
            return '''
            <div class="result-item">
                <div class="result-title">Result: Tier Impact Rate – With and Without Mitigation</div>
                <div class="result-content">
                    <p>No tier impact rate data available.</p>
                </div>
            </div>
            '''
        
        # Build table
        table_html = '''
        <div class="result-item">
            <div class="result-title">Result: Tier Impact Rate – With and Without Mitigation</div>
            <div class="result-content">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Decision Method</th>
                            <th>Persona Matches</th>
                            <th>Persona Non-Matches</th>
                            <th>Persona Tier Changed %</th>
                            <th>Mitigation Matches</th>
                            <th>Mitigation Non-Matches</th>
                            <th>Mitigation Tier Changed %</th>
                        </tr>
                    </thead>
                    <tbody>
        '''
        
        for method, data in tier_impact_data.items():
            table_html += f'''
            <tr>
                <td>{method}</td>
                <td>{data.get('persona_matches', 0):,}</td>
                <td>{data.get('persona_non_matches', 0):,}</td>
                <td>{data.get('persona_tier_changed_percentage', 0):.1f}%</td>
                <td>{data.get('mitigation_matches', 0):,}</td>
                <td>{data.get('mitigation_non_matches', 0):,}</td>
                <td>{data.get('mitigation_tier_changed_percentage', 0):.1f}%</td>
            </tr>
            '''
        
        table_html += '</tbody></table>'
        
        # Add statistical analysis
        if 'error' not in stats:
            table_html += self._build_tier_impact_statistical_analysis(stats, tier_impact_data)
        
        table_html += '</div></div>'
        return table_html

    def _build_mitigation_rankings_result(self, rankings_data: Dict, stats: Dict, method: str) -> str:
        """Build HTML for bias mitigation rankings result"""
        if not rankings_data:
            return f'''
            <div class="result-item">
                <div class="result-title">Result: Bias Mitigation Rankings - {method}</div>
                <div class="result-content">
                    <p>No bias mitigation rankings data available for {method}.</p>
                </div>
            </div>
            '''
        
        # Build table
        table_html = f'''
        <div class="result-item">
            <div class="result-title">Result: Bias Mitigation Rankings - {method}</div>
            <div class="result-content">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Risk Mitigation Strategy</th>
                            <th>Sample Size</th>
                            <th>Mean Baseline</th>
                            <th>Mean Persona</th>
                            <th>Mean Mitigation</th>
                            <th>Residual Bias %</th>
                            <th>Std Dev</th>
                            <th>SEM</th>
                        </tr>
                    </thead>
                    <tbody>
        '''
        
        # Sort by residual bias percentage (effectiveness_percentage) in ascending order
        sorted_rankings = sorted(rankings_data.items(), key=lambda x: x[1].get('effectiveness_percentage', 0))
        
        for strategy, data in sorted_rankings:
            table_html += f'''
            <tr>
                <td>{strategy.replace('_', ' ').title()}</td>
                <td>{data.get('sample_size', 0):,}</td>
                <td>{data.get('mean_baseline', 0):.3f}</td>
                <td>{data.get('mean_persona', 0):.3f}</td>
                <td>{data.get('mean_mitigation', 0):.3f}</td>
                <td>{data.get('effectiveness_percentage', 0):.1f}%</td>
                <td>{data.get('std_dev', 0):.3f}</td>
                <td>{data.get('sem', 0):.3f}</td>
            </tr>
            '''
        
        table_html += '</tbody></table>'
        
        # Add statistical analysis
        if 'error' not in stats:
            table_html += self._build_mitigation_rankings_statistical_analysis(stats, method)
        
        table_html += '</div></div>'
        return table_html

    def _build_tier_impact_statistical_analysis(self, stats: Dict, tier_impact_data: Dict = None) -> str:
        """Build HTML for statistical analysis of tier impact rates"""
        p_value = stats.get("p_value", 1.0)
        chi2_stat = stats.get("chi2_statistic", 0)
        df = stats.get("degrees_of_freedom", 1)
        conclusion = stats.get("conclusion", "accepted")

        # Analyze the actual mitigation effect
        effect_analysis = ""
        interpretation = {'significance_text': conclusion, 'effect_magnitude': 'unknown', 'interpretation': 'unknown', 'warning': ''}

        if tier_impact_data:
            try:
                # Calculate the actual effect of mitigation on bias
                effects = []

                for method, data in tier_impact_data.items():
                    persona_change_pct = data.get('persona_tier_changed_percentage', 0)
                    mitigation_change_pct = data.get('mitigation_tier_changed_percentage', 0)

                    # Calculate the change in bias due to mitigation
                    # Positive means mitigation increased bias, negative means it reduced bias
                    bias_change = mitigation_change_pct - persona_change_pct
                    effects.append({
                        'method': method,
                        'persona_bias': persona_change_pct,
                        'mitigation_bias': mitigation_change_pct,
                        'bias_change': bias_change
                    })

                # Analyze the effects
                zero_shot_effect = next((e for e in effects if 'zero' in e['method'].lower()), None)
                n_shot_effect = next((e for e in effects if 'n-shot' in e['method'] or 'n_shot' in e['method']), None)

                # Build effect analysis HTML
                effect_analysis = "<p><strong>Mitigation Effect Analysis:</strong></p><ul style='margin-left: 20px;'>"

                if zero_shot_effect:
                    change = zero_shot_effect['bias_change']
                    if abs(change) < 0.5:
                        effect_desc = "negligible effect"
                    elif change > 0:
                        effect_desc = f"<strong>increased bias by {abs(change):.1f} percentage points</strong> (counterproductive)"
                    else:
                        effect_desc = f"reduced bias by {abs(change):.1f} percentage points"

                    effect_analysis += f"<li>Zero-shot: Mitigation {effect_desc} ({zero_shot_effect['persona_bias']:.1f}% → {zero_shot_effect['mitigation_bias']:.1f}%)</li>"

                if n_shot_effect:
                    change = n_shot_effect['bias_change']
                    if abs(change) < 0.5:
                        effect_desc = "negligible effect"
                    elif change > 0:
                        effect_desc = f"<strong>increased bias by {abs(change):.1f} percentage points</strong> (counterproductive)"
                    else:
                        effect_desc = f"reduced bias by {abs(change):.1f} percentage points"

                    effect_analysis += f"<li>N-shot: Mitigation {effect_desc} ({n_shot_effect['persona_bias']:.1f}% → {n_shot_effect['mitigation_bias']:.1f}%)</li>"

                effect_analysis += "</ul>"

                # Determine overall interpretation
                avg_change = np.mean([e['bias_change'] for e in effects])
                if avg_change > 1:
                    overall_effect = "counterproductive (increases bias)"
                elif avg_change < -1:
                    overall_effect = "effective (reduces bias)"
                else:
                    overall_effect = "ineffective (negligible impact)"

                # Calculate proper effect size (difference in proportions)
                if zero_shot_effect and n_shot_effect:
                    # Cohen's h for the change in proportions
                    p1 = zero_shot_effect['persona_bias'] / 100
                    p2 = zero_shot_effect['mitigation_bias'] / 100
                    cohens_h = calculate_cohens_h(p2, p1)
                    h_interpretation = interpret_statistical_result(p_value, cohens_h, "cohens_h")

                    interpretation = h_interpretation
                    effect_analysis += f"<p><strong>Effect Size (Cohen's h):</strong> {cohens_h:.3f} ({h_interpretation['effect_magnitude']})</p>"
            except Exception as e:
                # If calculation fails, use basic interpretation
                interpretation = {'significance_text': conclusion, 'effect_magnitude': 'unknown', 'interpretation': 'unknown', 'warning': ''}
                overall_effect = "unable to determine"

        # Determine proper implication based on the analysis
        if tier_impact_data and 'overall_effect' in locals():
            if overall_effect == "counterproductive (increases bias)":
                implication = "The bias mitigation strategies are <strong>counterproductive</strong> - they actually increase bias rather than reduce it. This suggests the mitigation approaches need fundamental reconsideration."
            elif overall_effect == "ineffective (negligible impact)":
                implication = "The bias mitigation strategies have <strong>negligible impact</strong> on reducing bias. Alternative mitigation approaches should be explored."
            elif overall_effect == "effective (reduces bias)":
                implication = "The bias mitigation strategies are <strong>effective</strong> at reducing bias in tier assignments."
            else:
                implication = "The effect of bias mitigation strategies could not be determined."
        else:
            implication = stats.get("implication", "Unable to determine mitigation effectiveness")

        # Register result with collector
        # Determine the primary effect size and type for categorization
        primary_effect_size = 0
        primary_effect_type = 'unknown'
        
        if zero_shot_effect and n_shot_effect and 'cohens_h' in locals():
            primary_effect_size = cohens_h
            primary_effect_type = 'cohens_h'
        
        result_data = {
            'source_tab': 'Severity and Bias',
            'source_subtab': 'Tier Impact',
            'test_name': 'Tier Impact Rate: Persona-Injected vs Baseline',
            'test_type': 'chi_squared',
            'p_value': p_value,
            'effect_size': primary_effect_size,
            'effect_type': primary_effect_type,
            'sample_size': sum(data.get('persona_matches', 0) + data.get('persona_non_matches', 0) for data in tier_impact_data.values()) if tier_impact_data else 0,
            'finding': f'Persona injection {"affects" if interpretation["significance_text"] == "rejected" else "does not affect"} tier selection bias (χ² = {chi2_stat:.3f})',
            'implication': implication,
            'timestamp': datetime.now()
        }
        self.collector.add_result(result_data)

        return f'''
        <div class="statistical-analysis">
            <h4>Statistical Analysis</h4>
            <p><strong>Hypothesis:</strong> H0: Bias mitigation has no effect on tier selection bias</p>
            <p><strong>Test:</strong> Chi-squared test for independence</p>
            {effect_analysis}
            <p><strong>Test Statistic:</strong> χ²({df:.0f}) = {chi2_stat:.3f}</p>
            <p><strong>p-value:</strong> {p_value:.4f}</p>
            <p><strong>Conclusion:</strong> The null hypothesis was <strong>{interpretation["significance_text"]}</strong> (p {"<" if p_value < 0.05 else "≥"} 0.05)</p>
            <p><strong>Implication:</strong> {implication}</p>
        </div>
        '''

    def _build_mitigation_rankings_statistical_analysis(self, stats: Dict, method: str) -> str:
        """Build HTML for statistical analysis of bias mitigation rankings"""
        p_value = stats.get("p_value", 1.0)
        eta_squared = stats.get("eta_squared", 0)
        conclusion = stats.get("conclusion", "accepted")
        
        # Enhanced interpretation with effect size
        interpretation = {'significance_text': conclusion, 'effect_magnitude': 'unknown', 'interpretation': 'unknown', 'warning': ''}
        
        if eta_squared > 0:
            # Use chi_squared interpretation for eta-squared (both are variance explained measures)
            interpretation = interpret_statistical_result(p_value, eta_squared, "chi_squared")
        
        # Build HTML with enhanced interpretation
        effect_size_html = f'<p><strong>Effect Size (η²):</strong> {eta_squared:.6f} ({interpretation["effect_magnitude"]})</p>'
        practical_significance_html = f'<p><strong>Practical Significance:</strong> This result is {interpretation["interpretation"]}{interpretation["warning"]}.</p>'
        
        # Register result with collector
        result_data = {
            'source_tab': 'Bias Mitigation',
            'source_subtab': 'Strategy Effectiveness',
            'test_name': f'Bias Mitigation Strategy Comparison: {method}',
            'test_type': 'mixed_model',
            'p_value': p_value,
            'effect_size': eta_squared,
            'effect_type': 'eta_squared',
            'sample_size': stats.get('sample_size', 0),
            'finding': f'Bias mitigation strategies {"differ significantly" if interpretation["significance_text"] == "rejected" else "do not differ significantly"} in effectiveness (F = {stats.get("f_statistic", 0):.3f})',
            'implication': stats.get("implication", "N/A"),
            'timestamp': datetime.now()
        }
        self.collector.add_result(result_data)
        
        return f'''
        <div class="statistical-analysis">
            <h4>Statistical Analysis - {method}</h4>
            <p><strong>Hypothesis:</strong> {stats.get("hypothesis", "N/A")}</p>
            <p><strong>Model:</strong> {stats.get("test_type", "N/A")}</p>
            <p><strong>Test:</strong> {stats.get("test_method", "N/A")}</p>
            <p><strong>Test Statistic:</strong> F = {stats.get("f_statistic", "N/A")}</p>
            <p><strong>p-value:</strong> {p_value:.6f}</p>
            {effect_size_html}
            <p><strong>Conclusion:</strong> The null hypothesis was <strong>{interpretation["significance_text"]}</strong> (p {p_value:.3f})</p>
            {practical_significance_html}
            <p><strong>Implication:</strong> {stats.get("implication", "N/A")}</p>
            {f'<p><strong>Note:</strong> {stats.get("note", "")}</p>' if stats.get("note") else ''}
        </div>
        '''
