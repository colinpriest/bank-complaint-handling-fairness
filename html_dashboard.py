#!/usr/bin/env python3
"""
HTML Dashboard Generator for Bank Complaint Handling Fairness Analysis

This module creates an interactive HTML dashboard with multiple tabs showing
fairness analysis results, persona injection effects, bias mitigation, and
ground truth accuracy metrics.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


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

        # Build complete HTML content
        html_content = self._build_html_structure(experiment_data)

        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"[SUCCESS] Dashboard generated: {output_file}")
        return str(output_file)

    def _build_html_structure(self, data: Dict[str, Any]) -> str:
        """Build complete HTML structure with all components"""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Fairness Dashboard - Bank Complaint Analysis</title>
    {self._get_css_styles()}
    {self._get_javascript()}
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

                <div class="result-item">
                    <div class="result-title">Result 1: Confusion Matrix – Zero Shot</div>
                    <div class="result-placeholder">[Placeholder: Confusion matrix for zero-shot tier predictions by complaint severity]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 2: Confusion Matrix – N-Shot</div>
                    <div class="result-placeholder">[Placeholder: Confusion matrix for n-shot tier predictions by complaint severity]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 3: Tier Impact Rate – Monetary vs. Non-Monetary</div>
                    <div class="result-placeholder">[Placeholder: Comparison of tier assignment rates for monetary vs non-monetary complaints]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 4: Mean Tier Impact– Monetary vs. Non-Monetary</div>
                    <div class="result-placeholder">[Placeholder: Average tier assignments for monetary vs non-monetary complaints]</div>
                </div>
            </div>
        </div>

        <div id="severity-process" class="sub-tab-content">
            <div class="section">
                <h2>Process Bias</h2>

                <div class="result-item">
                    <div class="result-title">Result 1: Question Rate – Monetary vs. Non-Monetary – Zero-Shot</div>
                    <div class="result-placeholder">[Placeholder: Information request rates by complaint type in zero-shot experiments]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 2: Question Rate – Monetary vs. Non-Monetary – N-Shot</div>
                    <div class="result-placeholder">[Placeholder: Information request rates by complaint type in n-shot experiments]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 3: Implied Stereotyping - Monetary vs. Non-Monetary</div>
                    <div class="result-placeholder">[Placeholder: Stereotyping analysis by complaint severity type]</div>
                </div>
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

                <div class="result-item">
                    <div class="result-title">Result 1: Confusion Matrix – With and Without Mitigation</div>
                    <div class="result-placeholder">[Placeholder: Confusion matrix comparing tier predictions with and without bias mitigation]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 2: Tier Impact Rate – With and Without Mitigation</div>
                    <div class="result-placeholder">[Placeholder: Tier assignment rate comparison with and without bias mitigation]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 3: Mean Tier Impact – With and Without Mitigation</div>
                    <div class="result-placeholder">[Placeholder: Average tier assignments with and without bias mitigation]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 4: Bias Mitigation Rankings</div>
                    <div class="result-placeholder">[Placeholder: Ranking of bias mitigation strategies by effectiveness]</div>
                </div>
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
            </div>
        </div>

        <div id="persona-ethnicity" class="sub-tab-content">
            <div class="section">
                <h2>Ethnicity Bias</h2>

                <div class="result-item">
                    <div class="result-title">Result 1: Mean Tier by Ethnicity and by Zero-Shot/N-Shot</div>
                    <div class="result-placeholder">[Placeholder: Average tier assignments broken down by ethnicity and shot type]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 2: Tier Distribution by Ethnicity and by Zero-Shot/N-Shot</div>
                    <div class="result-placeholder">[Placeholder: Tier distribution analysis by ethnicity and shot type]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 3: Bias Distribution by Ethnicity and by Zero-Shot/N-Shot</div>
                    <div class="result-placeholder">[Placeholder: Bias distribution analysis by ethnicity and shot type]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 4: Question Rate – Persona-Injected vs. Baseline – by Ethnicity and by Zero-Shot/N-Shot</div>
                    <div class="result-placeholder">[Placeholder: Information request rates by ethnicity and shot type]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 5: Disadvantage Ranking by Ethnicity and by Zero-Shot/N-Shot</div>
                    <div class="result-placeholder">[Placeholder: Ranking of ethnicity disadvantage by shot type]</div>
                </div>
            </div>
        </div>

        <div id="persona-geography" class="sub-tab-content">
            <div class="section">
                <h2>Geographic Bias</h2>

                <div class="result-item">
                    <div class="result-title">Result 1: Mean Tier by Geography and by Zero-Shot/N-Shot</div>
                    <div class="result-placeholder">[Placeholder: Average tier assignments broken down by geography and shot type]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 2: Tier Distribution by Geography and by Zero-Shot/N-Shot</div>
                    <div class="result-placeholder">[Placeholder: Tier distribution analysis by geography and shot type]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 3: Bias Distribution by Geography and by Zero-Shot/N-Shot</div>
                    <div class="result-placeholder">[Placeholder: Bias distribution analysis by geography and shot type]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 4: Question Rate – Persona-Injected vs. Baseline – by Geography and by Zero-Shot/N-Shot</div>
                    <div class="result-placeholder">[Placeholder: Information request rates by geography and shot type]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 5: Disadvantage Ranking by Geography and by Zero-Shot/N-Shot</div>
                    <div class="result-placeholder">[Placeholder: Ranking of geography disadvantage by shot type]</div>
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

        # Build statistical analysis conclusion
        stats_html = ""
        stats = distribution_data.get('statistical_analysis')
        if stats:
            chi2 = stats.get('chi2')
            p_value = stats.get('p_value')
            dof = stats.get('dof')

            if p_value is not None:
                is_significant = p_value < 0.05
                p_value_display = f"{p_value:.4f}" if p_value >= 0.0001 else "< 0.0001"
                
                implication_text = "The distributions of tier recommendations are not significantly different between baseline and persona-injected experiments."
                if is_significant:
                    implication_text = "The distributions of tier recommendations are significantly different, suggesting that persona injection influences the pattern of tier assignments."

                stats_html = f"""
                <div class="conclusion">
                    <h4>Statistical Analysis:</h4>
                    <p>H0: The tier distribution is independent of persona injection.</p>
                    <p>Test: Chi-squared test of independence</p>
                    <p>Test Statistic: χ²({dof}) = {chi2:.2f}</p>
                    <p>p-value: {p_value_display}</p>
                    <p>Conclusion: The null hypothesis is <strong>{'rejected' if is_significant else 'not rejected'}</strong> (p {'<' if p_value < 0.05 else '≥'} 0.05).</p>
                    <p>Implication: {implication_text}</p>
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
        
        # Add conclusion
        conclusion = ""
        if total_experiments > 0:
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
                # Calculate effect size (Cohen's d)
                pooled_std = stddev  # Since we're using the SD of differences
                cohens_d = (mean_persona - mean_baseline) / pooled_std if pooled_std > 0 else 0
                
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
                
                # Determine if the difference is statistically significant
                is_significant = p_value < 0.05
                
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
                if is_significant:
                    effect_description = ' '.join(filter(None, [magnitude, direction]))
                    implication_text = f"The LLM's recommended tier is {effect_description} when it sees humanizing attributes, {implication}."

                # Build the statistical report
                stats_html += f"""
                <div class="conclusion">
                    <h4>Statistical Analysis ({item.get('llm_method', 'N/A').title()}):</h4>
                    <p>H0: The mean tier is the same with and without persona injection</p>
                    <p>Test: Paired t-test</p>
                    <p>Effect Size (Cohen's d): {abs(cohens_d):.2f} ({'small' if 0.2 <= abs(cohens_d) < 0.5 else 'medium' if 0.5 <= abs(cohens_d) < 0.8 else 'large' if abs(cohens_d) >= 0.8 else 'negligible'})</p>
                    <p>Mean Difference: {mean_persona - mean_baseline:+.2f} (from {mean_baseline:.2f} to {mean_persona:.2f})</p>
                    <p>Test Statistic: t({df}) = {t_statistic:.4f}</p>
                    <p>p-value: {p_value_display}</p>
                    <p>Conclusion: The null hypothesis is <strong>{"rejected" if is_significant else "not rejected"}</strong> (p {'<' if p_value < 0.05 else '≥'} 0.05).</p>
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
                
                # Round very small p-values
                p_value_display = f"{p_value:.4f}" if p_value >= 0.0001 else "< 0.0001"
                
                # Determine significance
                is_significant = p_value < 0.05
                
                # Determine implication
                if is_significant:
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
                    <p>Test Statistic: χ²({dof}) = {chi2:.2f}</p>
                    <p>p-value: {p_value_display}</p>
                    <p>Conclusion: The null hypothesis is <strong>{'rejected' if is_significant else 'not rejected'}</strong> (p {'<' if p_value < 0.05 else '≥'} 0.05).</p>
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
                
                # Round very small p-values
                p_value_display = f"{p_value:.4f}" if p_value >= 0.0001 else "< 0.0001"
                
                # Determine significance
                is_significant = p_value < 0.05
                
                # Determine implication
                if is_significant:
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
                    <p>Test Statistic: χ²({dof}) = {chi2:.2f}</p>
                    <p>p-value: {p_value_display}</p>
                    <p>Conclusion: The null hypothesis is <strong>{'rejected' if is_significant else 'not rejected'}</strong> (p {'<' if p_value < 0.05 else '≥'} 0.05).</p>
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
                    <div class="result-placeholder">[Placeholder: Comprehensive accuracy comparison across all experimental conditions]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 2: Zero-Shot vs N-Shot Accuracy</div>
                    <div class="result-placeholder">[Placeholder: Direct comparison of zero-shot and n-shot accuracy performance]</div>
                </div>

                <div class="result-item">
                    <div class="result-title">Result 3: Confidence vs Accuracy Correlation</div>
                    <div class="result-placeholder">[Placeholder: Analysis of model confidence levels vs actual accuracy]</div>
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
        
        zero_shot_stats_html = self._build_gender_distribution_statistical_analysis(zero_shot_stats, "Zero-Shot")
        n_shot_stats_html = self._build_gender_distribution_statistical_analysis(n_shot_stats, "N-Shot")
        
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
        
        zero_shot_stats_html = self._build_gender_question_statistical_analysis(zero_shot_stats, "Zero-Shot")
        n_shot_stats_html = self._build_gender_question_statistical_analysis(n_shot_stats, "N-Shot")
        
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
        
        # Determine implication
        if conclusion == 'rejected':
            if mean_diff > 0:
                implication = f"The LLM's mean recommended tier is biased by gender, disadvantaging {gender2}s."
            else:
                implication = f"The LLM's mean recommended tier is biased by gender, disadvantaging {gender1}s."
        else:
            if p_value <= 0.1:
                implication = "There is weak evidence that the LLM's mean recommended tier is biased by gender."
            else:
                implication = "There is no evidence that the LLM's mean recommended tier is biased by gender."
        
        return f'''
        <div class="statistical-analysis">
            <h4>Statistical Analysis - {method}</h4>
            <p><strong>Hypothesis:</strong> H0: Persona injection does not affect mean tier assignment</p>
            <p><strong>Test:</strong> Paired t-test</p>
            <p><strong>Effect Size (Cohen's d):</strong> {cohens_d:.3f}</p>
            <p><strong>Mean Difference:</strong> {mean_diff:.3f}</p>
            <p><strong>Test Statistic:</strong> t({df:.0f}) = {t_stat:.3f}</p>
            <p><strong>p-value:</strong> {p_value:.4f}</p>
            <p><strong>Conclusion:</strong> The null hypothesis was {conclusion} (p {"<" if p_value < 0.05 else "≥"} 0.05)</p>
            <p><strong>Implication:</strong> {implication}</p>
        </div>'''

    def _build_gender_distribution_statistical_analysis(self, stats: Dict, method: str) -> str:
        """Build statistical analysis HTML for distribution comparison"""
        if not stats or 'error' in stats:
            return f'<div class="statistical-analysis"><p>Statistical analysis not available for {method}</p></div>'
        
        chi2 = stats.get('chi2_statistic', 0)
        p_value = stats.get('p_value', 1)
        df = stats.get('degrees_of_freedom', 0)
        conclusion = stats.get('conclusion', 'accepted')
        
        # Determine implication
        if conclusion == 'rejected':
            implication = "The LLM's recommended tiers are biased by gender."
        else:
            if p_value > 0.1:
                implication = "There is no evidence that the LLM's recommended tiers are biased by gender."
            else:
                implication = "There is weak evidence that the LLM's recommended tiers are biased by gender."
        
        return f'''
        <div class="statistical-analysis">
            <h4>Statistical Analysis - {method}</h4>
            <p><strong>Hypothesis:</strong> H0: Persona injection does not affect the distribution of tier assignments</p>
            <p><strong>Test:</strong> Chi-squared test</p>
            <p><strong>Test Statistic:</strong> χ²({df:.0f}) = {chi2:.3f}</p>
            <p><strong>p-value:</strong> {p_value:.4f}</p>
            <p><strong>Conclusion:</strong> The null hypothesis was {conclusion} (p {"<" if p_value < 0.05 else "≥"} 0.05)</p>
            <p><strong>Implication:</strong> {implication}</p>
        </div>'''

    def _build_gender_question_statistical_analysis(self, stats: Dict, method: str) -> str:
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
        
        # Determine implication
        if conclusion == 'rejected':
            if rate_diff > 0:
                implication = f"The LLM's questioning behavior is biased by gender, with {gender1}s being asked more questions."
            else:
                implication = f"The LLM's questioning behavior is biased by gender, with {gender2}s being asked more questions."
        else:
            if p_value > 0.1:
                implication = "There is no evidence that the LLM's questioning behavior is biased by gender."
            else:
                implication = "There is weak evidence that the LLM's questioning behavior is biased by gender."
        
        return f'''
        <div class="statistical-analysis">
            <h4>Statistical Analysis - {method}</h4>
            <p><strong>Hypothesis:</strong> H0: The question rate is the same across genders</p>
            <p><strong>Test:</strong> Chi-squared test of independence</p>
            <p><strong>Rate Difference:</strong> {rate_diff:.1f}%</p>
            <p><strong>Test Statistic:</strong> χ²({df:.0f}) = {chi2:.3f}</p>
            <p><strong>p-value:</strong> {p_value:.4f}</p>
            <p><strong>Conclusion:</strong> The null hypothesis was {conclusion} (p {"<" if p_value < 0.05 else "≥"} 0.05)</p>
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
        
        # Determine implication
        if conclusion == 'rejected':
            implication = "The LLM's recommended tiers are biased by an interaction of gender and LLM prompt."
        else:
            if p_value > 0.1:
                implication = "There is no evidence that the LLM's recommended tiers are biased by an interaction of gender and LLM prompt."
            else:
                implication = "There is weak evidence that the LLM's recommended tiers are biased by an interaction of gender and LLM prompt."
        
        return f'''
        <div class="statistical-analysis">
            <h4>Statistical Analysis</h4>
            <p><strong>Hypothesis:</strong> H0: For each gender, the within-case expected decision is the same for zero-shot and n-shot</p>
            <p><strong>Test:</strong> {test_type}</p>
            <p><strong>Test Statistic:</strong> F = {f_stat:.3f}</p>
            <p><strong>p-Value:</strong> {p_value:.4f}</p>
            <p><strong>Conclusion:</strong> The null hypothesis was {conclusion} (p {"<" if p_value < 0.05 else "≥"} 0.05)</p>
            <p><strong>Implication:</strong> {implication}</p>
        </div>'''