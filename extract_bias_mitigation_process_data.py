#!/usr/bin/env python3
"""
Extract Bias Mitigation Process Bias Data for Sub-Tab 4.2: Process Bias

This script extracts data for analyzing how bias mitigation strategies affect
question rates (process bias) in LLM decision-making for both zero-shot and n-shot methods.
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

def extract_bias_mitigation_process_data() -> Dict[str, Any]:
    """
    Extracts bias mitigation process bias data from the database for dashboard.

    Returns:
        Dictionary containing process bias analysis data for bias mitigation experiments
    """
    connection = get_db_connection()

    try:
        # Query 1: Question rates with and without mitigation - Zero Shot
        zero_shot_query = """
        WITH baseline_questions AS (
            SELECT
                case_id,
                CASE WHEN asks_for_info = true THEN 1 ELSE 0 END as question_asked
            FROM baseline_experiments
            WHERE decision_method = 'zero-shot'
        ),
        mitigation_questions AS (
            SELECT
                case_id,
                CASE WHEN asks_for_info = true THEN 1 ELSE 0 END as question_asked,
                risk_mitigation_strategy as bias_mitigation_strategy
            FROM bias_mitigation_experiments
            WHERE decision_method = 'zero-shot'
        )
        SELECT
            'baseline' as condition,
            COUNT(*) as total_cases,
            SUM(question_asked) as questions_asked,
            AVG(question_asked::float) as question_rate
        FROM baseline_questions

        UNION ALL

        SELECT
            'mitigation' as condition,
            COUNT(*) as total_cases,
            SUM(question_asked) as questions_asked,
            AVG(question_asked::float) as question_rate
        FROM mitigation_questions

        UNION ALL

        SELECT
            bias_mitigation_strategy as condition,
            COUNT(*) as total_cases,
            SUM(question_asked) as questions_asked,
            AVG(question_asked::float) as question_rate
        FROM mitigation_questions
        GROUP BY bias_mitigation_strategy
        ORDER BY condition;
        """

        # Query 2: Question rates with and without mitigation - N-Shot
        n_shot_query = """
        WITH baseline_questions AS (
            SELECT
                case_id,
                CASE WHEN asks_for_info = true THEN 1 ELSE 0 END as question_asked
            FROM baseline_experiments
            WHERE decision_method = 'n-shot'
        ),
        mitigation_questions AS (
            SELECT
                case_id,
                CASE WHEN asks_for_info = true THEN 1 ELSE 0 END as question_asked,
                risk_mitigation_strategy as bias_mitigation_strategy
            FROM bias_mitigation_experiments
            WHERE decision_method = 'n-shot'
        )
        SELECT
            'baseline' as condition,
            COUNT(*) as total_cases,
            SUM(question_asked) as questions_asked,
            AVG(question_asked::float) as question_rate
        FROM baseline_questions

        UNION ALL

        SELECT
            'mitigation' as condition,
            COUNT(*) as total_cases,
            SUM(question_asked) as questions_asked,
            AVG(question_asked::float) as question_rate
        FROM mitigation_questions

        UNION ALL

        SELECT
            bias_mitigation_strategy as condition,
            COUNT(*) as total_cases,
            SUM(question_asked) as questions_asked,
            AVG(question_asked::float) as question_rate
        FROM mitigation_questions
        GROUP BY bias_mitigation_strategy
        ORDER BY condition;
        """

        # Query 3: Statistical analysis data for zero-shot
        zero_shot_stats_query = """
        WITH baseline_questions AS (
            SELECT
                case_id,
                CASE WHEN asks_for_info = true THEN 1 ELSE 0 END as question_asked,
                'baseline' as condition
            FROM baseline_experiments
            WHERE decision_method = 'zero-shot'
        ),
        mitigation_questions AS (
            SELECT
                case_id,
                CASE WHEN asks_for_info = true THEN 1 ELSE 0 END as question_asked,
                'mitigation' as condition
            FROM bias_mitigation_experiments
            WHERE decision_method = 'zero-shot'
        ),
        combined AS (
            SELECT * FROM baseline_questions
            UNION ALL
            SELECT * FROM mitigation_questions
        )
        SELECT
            condition,
            SUM(question_asked) as questions_asked,
            COUNT(*) - SUM(question_asked) as questions_not_asked,
            COUNT(*) as total
        FROM combined
        GROUP BY condition
        ORDER BY condition;
        """

        # Query 4: Statistical analysis data for n-shot
        n_shot_stats_query = """
        WITH baseline_questions AS (
            SELECT
                case_id,
                CASE WHEN asks_for_info = true THEN 1 ELSE 0 END as question_asked,
                'baseline' as condition
            FROM baseline_experiments
            WHERE decision_method = 'n-shot'
        ),
        mitigation_questions AS (
            SELECT
                case_id,
                CASE WHEN asks_for_info = true THEN 1 ELSE 0 END as question_asked,
                'mitigation' as condition
            FROM bias_mitigation_experiments
            WHERE decision_method = 'n-shot'
        ),
        combined AS (
            SELECT * FROM baseline_questions
            UNION ALL
            SELECT * FROM mitigation_questions
        )
        SELECT
            condition,
            SUM(question_asked) as questions_asked,
            COUNT(*) - SUM(question_asked) as questions_not_asked,
            COUNT(*) as total
        FROM combined
        GROUP BY condition
        ORDER BY condition;
        """

        cursor = connection.cursor()

        # Execute zero-shot query
        cursor.execute(zero_shot_query)
        zero_shot_results = cursor.fetchall()
        zero_shot_columns = [desc[0] for desc in cursor.description]

        # Execute n-shot query
        cursor.execute(n_shot_query)
        n_shot_results = cursor.fetchall()
        n_shot_columns = [desc[0] for desc in cursor.description]

        # Execute statistical queries
        cursor.execute(zero_shot_stats_query)
        zero_shot_stats_results = cursor.fetchall()

        cursor.execute(n_shot_stats_query)
        n_shot_stats_results = cursor.fetchall()

        # Process zero-shot data
        zero_shot_data = {}
        for row in zero_shot_results:
            condition = row[0]
            zero_shot_data[condition] = {
                'total_cases': row[1],
                'questions_asked': row[2],
                'question_rate': float(row[3]) if row[3] is not None else 0.0
            }

        # Process n-shot data
        n_shot_data = {}
        for row in n_shot_results:
            condition = row[0]
            n_shot_data[condition] = {
                'total_cases': row[1],
                'questions_asked': row[2],
                'question_rate': float(row[3]) if row[3] is not None else 0.0
            }

        # Calculate statistical analysis for zero-shot
        zero_shot_stats = {}
        if len(zero_shot_stats_results) >= 2:
            try:
                # Create contingency table for chi-squared test
                contingency_table = []
                for row in zero_shot_stats_results:
                    contingency_table.append([row[1], row[2]])  # [questions_asked, questions_not_asked]

                contingency_array = np.array(contingency_table)

                if contingency_array.size > 0 and np.all(contingency_array >= 0):
                    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_array)

                    # Calculate effect size (Cramér's V)
                    n = np.sum(contingency_array)
                    cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_array.shape) - 1)))

                    zero_shot_stats = {
                        'chi2_statistic': float(chi2_stat),
                        'p_value': float(p_value),
                        'degrees_of_freedom': int(dof),
                        'cramers_v': float(cramers_v),
                        'contingency_table': contingency_table,
                        'conclusion': 'rejected' if p_value < 0.05 else 'accepted'
                    }
                else:
                    zero_shot_stats = {'error': 'Invalid contingency table data'}
            except Exception as e:
                zero_shot_stats = {'error': f'Statistical analysis failed: {str(e)}'}
        else:
            zero_shot_stats = {'error': 'Insufficient data for statistical analysis'}

        # Calculate statistical analysis for n-shot
        n_shot_stats = {}
        if len(n_shot_stats_results) >= 2:
            try:
                # Create contingency table for chi-squared test
                contingency_table = []
                for row in n_shot_stats_results:
                    contingency_table.append([row[1], row[2]])  # [questions_asked, questions_not_asked]

                contingency_array = np.array(contingency_table)

                if contingency_array.size > 0 and np.all(contingency_array >= 0):
                    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_array)

                    # Calculate effect size (Cramér's V)
                    n = np.sum(contingency_array)
                    cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_array.shape) - 1)))

                    n_shot_stats = {
                        'chi2_statistic': float(chi2_stat),
                        'p_value': float(p_value),
                        'degrees_of_freedom': int(dof),
                        'cramers_v': float(cramers_v),
                        'contingency_table': contingency_table,
                        'conclusion': 'rejected' if p_value < 0.05 else 'accepted'
                    }
                else:
                    n_shot_stats = {'error': 'Invalid contingency table data'}
            except Exception as e:
                n_shot_stats = {'error': f'Statistical analysis failed: {str(e)}'}
        else:
            n_shot_stats = {'error': 'Insufficient data for statistical analysis'}

        return {
            'zero_shot_question_rates': zero_shot_data,
            'n_shot_question_rates': n_shot_data,
            'zero_shot_stats': zero_shot_stats,
            'n_shot_stats': n_shot_stats
        }

    except Exception as e:
        return {'error': f'Database query failed: {str(e)}'}

    finally:
        connection.close()

if __name__ == "__main__":
    # Test the extraction function
    data = extract_bias_mitigation_process_data()
    print("Bias Mitigation Process Data:")
    for key, value in data.items():
        print(f"\n{key}:")
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")
        else:
            print(f"  {value}")