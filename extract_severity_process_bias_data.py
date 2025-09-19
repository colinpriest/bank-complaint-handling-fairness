#!/usr/bin/env python3
"""
Extract Severity Process Bias Data for Sub-Tab 3.2: Process Bias

This script extracts data for analyzing how complaint severity affects process bias
(question rates) in LLM decision-making for both zero-shot and n-shot methods.
"""

import os
import psycopg2
from dotenv import load_dotenv
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# Load environment variables
load_dotenv()

def get_db_connection():
    """Establishes a new database connection."""
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'fairness_analysis'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', '')
    }
    return psycopg2.connect(**db_config)

def extract_severity_process_bias_data() -> Dict[str, Any]:
    """
    Extracts severity process bias data from the database for dashboard.
    """
    connection = get_db_connection()
    
    try:
        # Query 1: Question Rate - Monetary vs Non-Monetary - Zero-Shot
        zero_shot_query = """
        WITH baseline_data AS (
            SELECT 
                b.case_id,
                b.llm_simplified_tier as baseline_tier,
                'baseline' as experiment_type,
                CASE 
                    WHEN b.llm_simplified_tier IN (0, 1) THEN 'Non-Monetary'
                    WHEN b.llm_simplified_tier = 2 THEN 'Monetary'
                    ELSE 'Unknown'
                END as severity,
                CASE WHEN b.asks_for_info = true THEN 1 ELSE 0 END as asks_for_info_binary
            FROM baseline_experiments b
            WHERE b.decision_method = 'zero-shot'
                AND b.llm_simplified_tier != -999
        ),
        persona_data AS (
            SELECT 
                p.case_id,
                b.llm_simplified_tier as baseline_tier,
                'persona-injected' as experiment_type,
                CASE 
                    WHEN b.llm_simplified_tier IN (0, 1) THEN 'Non-Monetary'
                    WHEN b.llm_simplified_tier = 2 THEN 'Monetary'
                    ELSE 'Unknown'
                END as severity,
                CASE WHEN p.asks_for_info = true THEN 1 ELSE 0 END as asks_for_info_binary
            FROM persona_injected_experiments p
            JOIN baseline_experiments b ON p.case_id = b.case_id
            WHERE p.decision_method = 'zero-shot'
                AND b.decision_method = 'zero-shot'
                AND b.llm_simplified_tier != -999
        ),
        combined_data AS (
            SELECT * FROM baseline_data
            UNION ALL
            SELECT * FROM persona_data
        )
        SELECT 
            severity,
            experiment_type,
            COUNT(*) as count,
            SUM(asks_for_info_binary) as question_count,
            (SUM(asks_for_info_binary)::FLOAT / COUNT(*)) * 100 as question_rate_percentage
        FROM combined_data
        WHERE severity != 'Unknown'
        GROUP BY severity, experiment_type
        ORDER BY severity, experiment_type;
        """
        
        cursor = connection.cursor()
        cursor.execute(zero_shot_query)
        zero_shot_data = cursor.fetchall()
        
        # Query 2: Question Rate - Monetary vs Non-Monetary - N-Shot
        n_shot_query = """
        WITH baseline_data AS (
            SELECT 
                b.case_id,
                b.llm_simplified_tier as baseline_tier,
                'baseline' as experiment_type,
                CASE 
                    WHEN b.llm_simplified_tier IN (0, 1) THEN 'Non-Monetary'
                    WHEN b.llm_simplified_tier = 2 THEN 'Monetary'
                    ELSE 'Unknown'
                END as severity,
                CASE WHEN b.asks_for_info = true THEN 1 ELSE 0 END as asks_for_info_binary
            FROM baseline_experiments b
            WHERE b.decision_method = 'n-shot'
                AND b.llm_simplified_tier != -999
        ),
        persona_data AS (
            SELECT 
                p.case_id,
                b.llm_simplified_tier as baseline_tier,
                'persona-injected' as experiment_type,
                CASE 
                    WHEN b.llm_simplified_tier IN (0, 1) THEN 'Non-Monetary'
                    WHEN b.llm_simplified_tier = 2 THEN 'Monetary'
                    ELSE 'Unknown'
                END as severity,
                CASE WHEN p.asks_for_info = true THEN 1 ELSE 0 END as asks_for_info_binary
            FROM persona_injected_experiments p
            JOIN baseline_experiments b ON p.case_id = b.case_id
            WHERE p.decision_method = 'n-shot'
                AND b.decision_method = 'n-shot'
                AND b.llm_simplified_tier != -999
        ),
        combined_data AS (
            SELECT * FROM baseline_data
            UNION ALL
            SELECT * FROM persona_data
        )
        SELECT 
            severity,
            experiment_type,
            COUNT(*) as count,
            SUM(asks_for_info_binary) as question_count,
            (SUM(asks_for_info_binary)::FLOAT / COUNT(*)) * 100 as question_rate_percentage
        FROM combined_data
        WHERE severity != 'Unknown'
        GROUP BY severity, experiment_type
        ORDER BY severity, experiment_type;
        """
        
        cursor.execute(n_shot_query)
        n_shot_data = cursor.fetchall()
        
        # Process zero-shot data
        zero_shot_results = {}
        for row in zero_shot_data:
            severity, experiment_type, count, question_count, question_rate_percentage = row
            if severity not in zero_shot_results:
                zero_shot_results[severity] = {}
            zero_shot_results[severity][experiment_type] = {
                'count': int(count),
                'question_count': int(question_count),
                'question_rate_percentage': float(question_rate_percentage) if question_rate_percentage is not None else 0.0
            }
        
        # Process n-shot data
        n_shot_results = {}
        for row in n_shot_data:
            severity, experiment_type, count, question_count, question_rate_percentage = row
            if severity not in n_shot_results:
                n_shot_results[severity] = {}
            n_shot_results[severity][experiment_type] = {
                'count': int(count),
                'question_count': int(question_count),
                'question_rate_percentage': float(question_rate_percentage) if question_rate_percentage is not None else 0.0
            }
        
        # Perform statistical analysis for zero-shot
        zero_shot_stats = perform_gee_analysis(zero_shot_results, 'zero-shot')
        
        # Perform statistical analysis for n-shot
        n_shot_stats = perform_gee_analysis(n_shot_results, 'n-shot')
        
        result = {
            'zero_shot_question_rates': zero_shot_results,
            'n_shot_question_rates': n_shot_results,
            'zero_shot_stats': zero_shot_stats,
            'n_shot_stats': n_shot_stats
        }
        
        return result
        
    except Exception as e:
        print(f"Error extracting severity process bias data: {e}")
        return {'error': str(e)}
    finally:
        connection.close()

def perform_gee_analysis(question_rate_data: Dict, method: str) -> Dict:
    """
    Perform GEE (Generalized Estimating Equations) analysis for clustered logistic model.
    Note: This is a simplified implementation. In practice, you would use
    specialized libraries like statsmodels or R's gee package.
    """
    try:
        if not question_rate_data:
            return {'error': 'No data available for statistical analysis'}
        
        # For now, we'll use a chi-squared test as an approximation
        # In a full implementation, you would use GEE with clustering by case_id
        
        # Create contingency table for chi-squared test
        contingency_data = []
        for severity in ['Non-Monetary', 'Monetary']:
            if severity in question_rate_data:
                baseline_data = question_rate_data[severity].get('baseline', {})
                persona_data = question_rate_data[severity].get('persona-injected', {})
                
                baseline_questions = baseline_data.get('question_count', 0)
                baseline_total = baseline_data.get('count', 0)
                baseline_no_questions = baseline_total - baseline_questions
                
                persona_questions = persona_data.get('question_count', 0)
                persona_total = persona_data.get('count', 0)
                persona_no_questions = persona_total - persona_questions
                
                contingency_data.append([baseline_questions, baseline_no_questions])
                contingency_data.append([persona_questions, persona_no_questions])
        
        if len(contingency_data) < 2:
            return {'error': 'Insufficient data for statistical analysis'}
        
        # Perform chi-squared test
        contingency_table = np.array(contingency_data)
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Determine implication based on results
        if p_value < 0.05:
            implication = "There is strong evidence that severity has an effect upon process bias via question rates."
        elif p_value <= 0.1:
            implication = "There is weak evidence that severity has an effect upon process bias via question rates."
        else:
            implication = "There is no evidence that severity has an effect upon process bias via question rates."
        
        return {
            'test_type': 'Chi-squared test for independence (approximation of GEE)',
            'hypothesis': 'H0: Severity has no marginal effect upon question rates',
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significant': p_value < 0.05,
            'conclusion': 'rejected' if p_value < 0.05 else 'accepted',
            'implication': implication,
            'note': 'Full GEE implementation would cluster by case_id and use robust Wald tests'
        }
        
    except Exception as e:
        return {'error': f'Statistical analysis failed: {e}'}

if __name__ == "__main__":
    # Test the data extraction
    print("=== Testing Severity Process Bias Data Extraction ===")
    data = extract_severity_process_bias_data()
    
    if 'error' in data:
        print(f"Error: {data['error']}")
    else:
        print("\n--- Zero-Shot Question Rates ---")
        for severity, experiments in data['zero_shot_question_rates'].items():
            print(f"\n{severity}:")
            for exp_type, stats in experiments.items():
                print(f"  {exp_type}: Count={stats['count']}, Questions={stats['question_count']}, Rate={stats['question_rate_percentage']:.1f}%")
        
        print("\n--- N-Shot Question Rates ---")
        for severity, experiments in data['n_shot_question_rates'].items():
            print(f"\n{severity}:")
            for exp_type, stats in experiments.items():
                print(f"  {exp_type}: Count={stats['count']}, Questions={stats['question_count']}, Rate={stats['question_rate_percentage']:.1f}%")
        
        print("\n--- Zero-Shot Statistical Analysis ---")
        if 'error' not in data['zero_shot_stats']:
            stats = data['zero_shot_stats']
            print(f"Test: {stats['test_type']}")
            print(f"Chi² = {stats['chi2_statistic']:.3f}, p = {stats['p_value']:.3f}")
            print(f"Conclusion: {stats['conclusion']}")
            print(f"Implication: {stats['implication']}")
            if 'note' in stats:
                print(f"Note: {stats['note']}")
        
        print("\n--- N-Shot Statistical Analysis ---")
        if 'error' not in data['n_shot_stats']:
            stats = data['n_shot_stats']
            print(f"Test: {stats['test_type']}")
            print(f"Chi² = {stats['chi2_statistic']:.3f}, p = {stats['p_value']:.3f}")
            print(f"Conclusion: {stats['conclusion']}")
            print(f"Implication: {stats['implication']}")
            if 'note' in stats:
                print(f"Note: {stats['note']}")
