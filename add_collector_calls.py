#!/usr/bin/env python3
"""
Add collector calls to all statistical analysis methods in html_dashboard.py

This script adds self.collector.add_result() calls to all statistical analysis methods
that don't already have them, so the Headline Results tab will show all statistical results.
"""

import re

def add_collector_calls():
    """Add collector calls to statistical analysis methods"""
    
    # Read the file
    with open('html_dashboard.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # List of methods that need collector calls (excluding the 4 that already have them)
    methods_to_update = [
        '_build_gender_question_statistical_analysis',
        '_build_ethnicity_mean_statistical_analysis',
        '_build_ethnicity_distribution_statistical_analysis', 
        '_build_ethnicity_question_statistical_analysis',
        '_build_ethnicity_tier_bias_statistical_analysis',
        '_build_geographic_mean_statistical_analysis',
        '_build_geographic_distribution_statistical_analysis',
        '_build_geographic_question_statistical_analysis',
        '_build_geographic_tier_bias_statistical_analysis',
        '_build_severity_statistical_analysis',
        '_build_severity_process_statistical_analysis',
        '_build_tier_impact_statistical_analysis'
    ]
    
    # Template for collector call
    collector_template = '''
        # Register result with collector
        result_data = {{
            'source_tab': '{source_tab}',
            'source_subtab': '{source_subtab}',
            'test_name': '{test_name}',
            'test_type': '{test_type}',
            'p_value': {p_value},
            'effect_size': {effect_size},
            'effect_type': '{effect_type}',
            'sample_size': {sample_size},
            'finding': '{finding}',
            'implication': '{implication}',
            'timestamp': datetime.now()
        }}
        self.collector.add_result(result_data)
        
        '''
    
    # Method-specific configurations
    method_configs = {
        '_build_gender_question_statistical_analysis': {
            'source_tab': 'Persona Injection',
            'source_subtab': 'Gender Bias',
            'test_name': f'Question Rate Comparison: {{gender1}} vs {{gender2}}',
            'test_type': 'chi_squared',
            'p_value': 'p_value',
            'effect_size': 'cramers_v',
            'effect_type': 'cramers_v',
            'sample_size': 'sum(sum(row) for row in contingency_table) if question_data else 0',
            'finding': f'Question rate differs significantly between gender groups (χ² = {{chi2:.3f}})',
            'implication': 'implication'
        },
        '_build_ethnicity_mean_statistical_analysis': {
            'source_tab': 'Persona Injection',
            'source_subtab': 'Ethnicity Bias',
            'test_name': f'Mean Tier Comparison: {{ethnicity1}} vs {{ethnicity2}}',
            'test_type': 'paired_t_test',
            'p_value': 'p_value',
            'effect_size': 'cohens_d',
            'effect_type': 'cohens_d',
            'sample_size': 'sample_size',
            'finding': f'Mean tier differs significantly between ethnicity groups (t = {{t_stat:.3f}})',
            'implication': 'implication'
        },
        '_build_ethnicity_distribution_statistical_analysis': {
            'source_tab': 'Persona Injection',
            'source_subtab': 'Ethnicity Bias',
            'test_name': f'Tier Distribution Comparison: {{ethnicity1}} vs {{ethnicity2}}',
            'test_type': 'chi_squared',
            'p_value': 'p_value',
            'effect_size': 'cramers_v',
            'effect_type': 'cramers_v',
            'sample_size': 'sum(sum(row) for row in contingency_table) if distribution_data else 0',
            'finding': f'Tier distribution differs significantly between ethnicity groups (χ² = {{chi2:.3f}})',
            'implication': 'implication'
        },
        '_build_ethnicity_question_statistical_analysis': {
            'source_tab': 'Persona Injection',
            'source_subtab': 'Ethnicity Bias',
            'test_name': f'Question Rate Comparison: {{ethnicity1}} vs {{ethnicity2}}',
            'test_type': 'chi_squared',
            'p_value': 'p_value',
            'effect_size': 'cramers_v',
            'effect_type': 'cramers_v',
            'sample_size': 'sum(sum(row) for row in contingency_table) if question_data else 0',
            'finding': f'Question rate differs significantly between ethnicity groups (χ² = {{chi2:.3f}})',
            'implication': 'implication'
        },
        '_build_ethnicity_tier_bias_statistical_analysis': {
            'source_tab': 'Persona Injection',
            'source_subtab': 'Ethnicity Bias',
            'test_name': 'Ethnicity Bias Consistency: Zero-Shot vs N-Shot',
            'test_type': 'mixed_model',
            'p_value': 'stats.get("p_value", 1)',
            'effect_size': 'stats.get("partial_eta_squared", 0)',
            'effect_type': 'eta_squared',
            'sample_size': 'stats.get("sample_size", 0)',
            'finding': f'Ethnicity bias {{"differs" if stats.get("conclusion") == "rejected" else "is consistent"}} between zero-shot and n-shot methods (F = {{stats.get("f_statistic", 0):.3f}})',
            'implication': 'stats.get("implication", "N/A")'
        },
        '_build_geographic_mean_statistical_analysis': {
            'source_tab': 'Persona Injection',
            'source_subtab': 'Geographic Bias',
            'test_name': 'Mean Tier Comparison Across Geographies',
            'test_type': 'anova',
            'p_value': 'stats.get("p_value", 1)',
            'effect_size': 'stats.get("eta_squared", 0)',
            'effect_type': 'eta_squared',
            'sample_size': 'sum(len(group) for group in groups) if "groups" in locals() else 0',
            'finding': f'Mean tier differs significantly across geographies (F = {{stats.get("f_statistic", 0):.3f}})',
            'implication': 'stats.get("implication", "N/A")'
        },
        '_build_geographic_distribution_statistical_analysis': {
            'source_tab': 'Persona Injection',
            'source_subtab': 'Geographic Bias',
            'test_name': 'Tier Distribution Comparison Across Geographies',
            'test_type': 'chi_squared',
            'p_value': 'stats.get("p_value", 1)',
            'effect_size': 'stats.get("cramers_v", 0)',
            'effect_type': 'cramers_v',
            'sample_size': 'sum(sum(row) for row in contingency_table) if distribution_data else 0',
            'finding': f'Tier distribution differs significantly across geographies (χ² = {{stats.get("chi2_statistic", 0):.3f}})',
            'implication': 'stats.get("implication", "N/A")'
        },
        '_build_geographic_question_statistical_analysis': {
            'source_tab': 'Persona Injection',
            'source_subtab': 'Geographic Bias',
            'test_name': 'Question Rate Comparison Across Geographies',
            'test_type': 'chi_squared',
            'p_value': 'stats.get("p_value", 1)',
            'effect_size': 'stats.get("cramers_v", 0)',
            'effect_type': 'cramers_v',
            'sample_size': 'sum(sum(row) for row in contingency_table) if question_data else 0',
            'finding': f'Question rate differs significantly across geographies (χ² = {{stats.get("chi2_statistic", 0):.3f}})',
            'implication': 'stats.get("implication", "N/A")'
        },
        '_build_geographic_tier_bias_statistical_analysis': {
            'source_tab': 'Persona Injection',
            'source_subtab': 'Geographic Bias',
            'test_name': 'Geographic Bias Consistency: Zero-Shot vs N-Shot',
            'test_type': 'mixed_model',
            'p_value': 'stats.get("p_value", 1)',
            'effect_size': 'stats.get("partial_eta_squared", 0)',
            'effect_type': 'eta_squared',
            'sample_size': 'stats.get("sample_size", 0)',
            'finding': f'Geographic bias {{"differs" if stats.get("conclusion") == "rejected" else "is consistent"}} between zero-shot and n-shot methods (F = {{stats.get("f_statistic", 0):.3f}})',
            'implication': 'stats.get("implication", "N/A")'
        },
        '_build_severity_statistical_analysis': {
            'source_tab': 'Severity and Bias',
            'source_subtab': 'Severity Bias',
            'test_name': 'Tier Impact Rate: With vs Without Mitigation',
            'test_type': 'chi_squared',
            'p_value': 'stats.get("p_value", 1)',
            'effect_size': 'stats.get("cramers_v", 0)',
            'effect_type': 'cramers_v',
            'sample_size': 'sum(sum(row) for row in contingency_table) if tier_impact_data else 0',
            'finding': f'Bias mitigation {{"affects" if stats.get("conclusion") == "rejected" else "does not affect"}} tier selection bias (χ² = {{stats.get("chi2_statistic", 0):.3f}})',
            'implication': 'stats.get("implication", "N/A")'
        },
        '_build_severity_process_statistical_analysis': {
            'source_tab': 'Severity and Bias',
            'source_subtab': 'Process Bias',
            'test_name': 'Question Rate: With vs Without Mitigation',
            'test_type': 'chi_squared',
            'p_value': 'stats.get("p_value", 1)',
            'effect_size': 'stats.get("cramers_v", 0)',
            'effect_type': 'cramers_v',
            'sample_size': 'sum(sum(row) for row in contingency_table) if question_rate_data else 0',
            'finding': f'Bias mitigation {{"affects" if stats.get("conclusion") == "rejected" else "does not affect"}} question rate (χ² = {{stats.get("chi2_statistic", 0):.3f}})',
            'implication': 'stats.get("implication", "N/A")'
        },
        '_build_tier_impact_statistical_analysis': {
            'source_tab': 'Severity and Bias',
            'source_subtab': 'Tier Impact',
            'test_name': 'Tier Impact Rate: Persona-Injected vs Baseline',
            'test_type': 'chi_squared',
            'p_value': 'stats.get("p_value", 1)',
            'effect_size': 'stats.get("cramers_v", 0)',
            'effect_type': 'cramers_v',
            'sample_size': 'sum(sum(row) for row in contingency_table) if tier_impact_data else 0',
            'finding': f'Persona injection {{"affects" if stats.get("conclusion") == "rejected" else "does not affect"}} tier selection (χ² = {{stats.get("chi2_statistic", 0):.3f}})',
            'implication': 'stats.get("implication", "N/A")'
        }
    }
    
    # For each method, find the return statement and add collector call before it
    for method_name in methods_to_update:
        config = method_configs[method_name]
        
        # Format the collector template with the config
        collector_call = collector_template.format(**config)
        
        # Find the method definition
        method_pattern = rf'def {method_name}\(.*?\) -> str:'
        method_match = re.search(method_pattern, content, re.DOTALL)
        
        if method_match:
            # Find the return statement in this method
            method_start = method_match.start()
            method_end = method_start + len(method_match.group())
            
            # Look for the return statement after the method definition
            remaining_content = content[method_end:]
            return_pattern = r'return f\'\'\''
            return_match = re.search(return_pattern, remaining_content)
            
            if return_match:
                # Insert collector call before the return statement
                insert_pos = method_end + return_match.start()
                content = content[:insert_pos] + collector_call + content[insert_pos:]
                print(f"Added collector call to {method_name}")
            else:
                print(f"Could not find return statement in {method_name}")
        else:
            print(f"Could not find method {method_name}")
    
    # Write the updated content back to the file
    with open('html_dashboard.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Successfully added collector calls to all statistical analysis methods!")

if __name__ == "__main__":
    add_collector_calls()
