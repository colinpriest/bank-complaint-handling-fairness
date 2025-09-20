#!/usr/bin/env python3
"""
Extract accuracy analysis data from the database for dashboard.

This module extracts data for Sub-Tab 5.1: Overview of the Ground Truth Accuracy tab,
including overall accuracy comparison and zero-shot vs n-shot accuracy rates.
"""

import os
import psycopg2
from typing import Dict, Any, List, Tuple
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_db_connection():
    """Get database connection using environment variables"""
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_NAME', 'complaints_analysis'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', ''),
        'port': os.getenv('DB_PORT', '5432')
    }
    
    return psycopg2.connect(**db_config)

def extract_accuracy_data() -> Dict[str, Any]:
    """
    Extracts accuracy analysis data from the database for dashboard.
    
    Returns:
        Dictionary containing accuracy data for Sub-Tab 5.1: Overview
    """
    connection = get_db_connection()
    
    try:
        # Query 1: Overall Accuracy Comparison - Confusion Matrix Data
        confusion_matrix_query = """
        SELECT 
            gt.simplified_ground_truth_tier,
            e.llm_simplified_tier,
            e.decision_method,
            CASE 
                WHEN e.persona IS NULL AND e.risk_mitigation_strategy IS NULL THEN 'Baseline'
                WHEN e.persona IS NOT NULL AND e.risk_mitigation_strategy IS NULL THEN 'Persona-Injected'
                WHEN e.risk_mitigation_strategy IS NOT NULL THEN 'Bias Mitigation'
                ELSE 'Unknown'
            END as experiment_category,
            COUNT(*) as count
        FROM experiments e
        JOIN ground_truth gt ON e.case_id = gt.case_id
        WHERE e.llm_simplified_tier != -999
            AND gt.simplified_ground_truth_tier >= 0
        GROUP BY gt.simplified_ground_truth_tier, e.llm_simplified_tier, e.decision_method, experiment_category
        ORDER BY gt.simplified_ground_truth_tier, e.llm_simplified_tier, e.decision_method, experiment_category;
        """
        
        # Query 2: Zero-Shot vs N-Shot Accuracy Rates
        accuracy_rates_query = """
        SELECT 
            e.decision_method,
            CASE 
                WHEN e.persona IS NULL AND e.risk_mitigation_strategy IS NULL THEN 'Baseline'
                WHEN e.persona IS NOT NULL AND e.risk_mitigation_strategy IS NULL THEN 'Persona-Injected'
                WHEN e.risk_mitigation_strategy IS NOT NULL THEN 'Bias Mitigation'
                ELSE 'Unknown'
            END as experiment_category,
            COUNT(*) as sample_size,
            SUM(CASE WHEN e.llm_simplified_tier = gt.simplified_ground_truth_tier THEN 1 ELSE 0 END) as correct_count,
            ROUND(100.0 * SUM(CASE WHEN e.llm_simplified_tier = gt.simplified_ground_truth_tier THEN 1 ELSE 0 END) / COUNT(*), 2) as accuracy_percentage
        FROM experiments e
        JOIN ground_truth gt ON e.case_id = gt.case_id
        WHERE e.llm_simplified_tier != -999
            AND gt.simplified_ground_truth_tier >= 0
        GROUP BY e.decision_method, experiment_category
        ORDER BY e.decision_method, experiment_category;
        """
        
        # Query 3: Detailed accuracy data for statistical analysis
        detailed_accuracy_query = """
        SELECT 
            e.case_id,
            gt.simplified_ground_truth_tier,
            e.llm_simplified_tier,
            e.decision_method,
            CASE 
                WHEN e.persona IS NULL AND e.risk_mitigation_strategy IS NULL THEN 'Baseline'
                WHEN e.persona IS NOT NULL AND e.risk_mitigation_strategy IS NULL THEN 'Persona-Injected'
                WHEN e.risk_mitigation_strategy IS NOT NULL THEN 'Bias Mitigation'
                ELSE 'Unknown'
            END as experiment_category,
            CASE WHEN e.llm_simplified_tier = gt.simplified_ground_truth_tier THEN 1 ELSE 0 END as is_accurate
        FROM experiments e
        JOIN ground_truth gt ON e.case_id = gt.case_id
        WHERE e.llm_simplified_tier != -999
            AND gt.simplified_ground_truth_tier >= 0
        ORDER BY e.decision_method, experiment_category, e.case_id;
        """
        
        cursor = connection.cursor()
        
        # Execute confusion matrix query
        cursor.execute(confusion_matrix_query)
        confusion_matrix_data = cursor.fetchall()
        
        # Execute accuracy rates query
        cursor.execute(accuracy_rates_query)
        accuracy_rates_data = cursor.fetchall()
        
        # Execute detailed accuracy query
        cursor.execute(detailed_accuracy_query)
        detailed_accuracy_data = cursor.fetchall()
        
        # Process confusion matrix data
        confusion_matrix = {}
        for row in confusion_matrix_data:
            gt_tier, llm_tier, decision_method, experiment_category, count = row
            key = (decision_method, experiment_category)
            if key not in confusion_matrix:
                confusion_matrix[key] = {}
            if gt_tier not in confusion_matrix[key]:
                confusion_matrix[key][gt_tier] = {}
            confusion_matrix[key][gt_tier][llm_tier] = count
        
        # Process accuracy rates data
        accuracy_rates = {}
        for row in accuracy_rates_data:
            decision_method, experiment_category, sample_size, correct_count, accuracy_percentage = row
            key = (decision_method, experiment_category)
            accuracy_rates[key] = {
                'decision_method': decision_method,
                'experiment_category': experiment_category,
                'sample_size': sample_size,
                'correct_count': correct_count,
                'accuracy_percentage': accuracy_percentage
            }
        
        # Process detailed accuracy data for statistical analysis
        detailed_data = {}
        for row in detailed_accuracy_data:
            case_id, gt_tier, llm_tier, decision_method, experiment_category, is_accurate = row
            key = (decision_method, experiment_category)
            if key not in detailed_data:
                detailed_data[key] = []
            detailed_data[key].append({
                'case_id': case_id,
                'gt_tier': gt_tier,
                'llm_tier': llm_tier,
                'is_accurate': is_accurate
            })
        
        return {
            'confusion_matrix': confusion_matrix,
            'accuracy_rates': accuracy_rates,
            'detailed_data': detailed_data,
            'query_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error extracting accuracy data: {e}")
        return {
            'confusion_matrix': {},
            'accuracy_rates': {},
            'detailed_data': {},
            'error': str(e)
        }
    finally:
        if connection:
            connection.close()

def format_confusion_matrix_html(confusion_matrix: Dict, decision_method: str = 'zero-shot', experiment_category: str = 'Baseline') -> str:
    """
    Format confusion matrix data as HTML table.
    
    Args:
        confusion_matrix: Dictionary containing confusion matrix data
        decision_method: Decision method to filter by
        experiment_category: Experiment category to filter by
    
    Returns:
        HTML table string
    """
    key = (decision_method, experiment_category)
    if key not in confusion_matrix:
        return "<p>No data available for selected filters.</p>"
    
    matrix_data = confusion_matrix[key]
    
    # Get all unique tiers
    all_gt_tiers = sorted(set(gt_tier for gt_tier in matrix_data.keys()))
    all_llm_tiers = sorted(set(llm_tier for gt_tier in matrix_data.values() for llm_tier in gt_tier.keys()))
    
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

def format_accuracy_rates_html(accuracy_rates: Dict) -> str:
    """
    Format accuracy rates data as HTML table.
    
    Args:
        accuracy_rates: Dictionary containing accuracy rates data
    
    Returns:
        HTML table string
    """
    if not accuracy_rates:
        return "<p>No accuracy data available.</p>"
    
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
        html += f'      <td>{data["accuracy_percentage"]}%</td>\n'
        html += '    </tr>\n'
    
    html += '  </tbody>\n'
    html += '</table>\n'
    
    return html

if __name__ == "__main__":
    # Test the data extraction
    data = extract_accuracy_data()
    print("Accuracy data extracted successfully!")
    print(f"Confusion matrix keys: {list(data['confusion_matrix'].keys())}")
    print(f"Accuracy rates keys: {list(data['accuracy_rates'].keys())}")
    
    # Test HTML formatting
    if data['confusion_matrix']:
        html = format_confusion_matrix_html(data['confusion_matrix'])
        print("\nConfusion Matrix HTML:")
        print(html)
    
    if data['accuracy_rates']:
        html = format_accuracy_rates_html(data['accuracy_rates'])
        print("\nAccuracy Rates HTML:")
        print(html)
