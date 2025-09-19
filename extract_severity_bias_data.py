#!/usr/bin/env python3
"""
Extract Severity Bias Data for Sub-Tab 3.1: Tier Recommendations

This script extracts data for analyzing how persona injection affects tier recommendations
based on case severity (monetary vs non-monetary cases).
"""

import os
import psycopg2
from dotenv import load_dotenv
from typing import Dict, List, Any
from scipy.stats import chi2_contingency
import numpy as np

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

def extract_severity_bias_data() -> Dict[str, Any]:
    """
    Extracts severity bias data from the database for dashboard.
    """
    connection = get_db_connection()
    
    try:
        # Query 1: Tier Impact Rate - Zero Shot
        zero_shot_query = """
        WITH tier_impact_data AS (
            SELECT 
                b.case_id,
                b.llm_simplified_tier as baseline_tier,
                p.llm_simplified_tier as persona_tier,
                CASE 
                    WHEN b.llm_simplified_tier IN (0, 1) THEN 'Non-Monetary'
                    WHEN b.llm_simplified_tier = 2 THEN 'Monetary'
                    ELSE 'Unknown'
                END as monetary_category,
                (p.llm_simplified_tier - b.llm_simplified_tier) as bias,
                CASE 
                    WHEN (p.llm_simplified_tier - b.llm_simplified_tier) = 0 THEN 1
                    ELSE 0
                END as unchanged
            FROM baseline_experiments b
            JOIN persona_injected_experiments p ON b.case_id = p.case_id
            WHERE b.decision_method = 'zero-shot'
                AND p.decision_method = 'zero-shot'
                AND b.llm_simplified_tier != -999
                AND p.llm_simplified_tier != -999
        )
        SELECT 
            monetary_category,
            COUNT(*) as count,
            AVG(persona_tier::FLOAT) as avg_tier,
            STDDEV(persona_tier::FLOAT) as std_dev,
            SUM(unchanged) as unchanged_count,
            (SUM(unchanged)::FLOAT / COUNT(*)) * 100 as unchanged_percentage
        FROM tier_impact_data
        WHERE monetary_category != 'Unknown'
        GROUP BY monetary_category
        ORDER BY monetary_category;
        """
        
        cursor = connection.cursor()
        cursor.execute(zero_shot_query)
        zero_shot_data = cursor.fetchall()
        
        # Query 2: Tier Impact Rate - N Shot
        n_shot_query = """
        WITH tier_impact_data AS (
            SELECT 
                b.case_id,
                b.llm_simplified_tier as baseline_tier,
                p.llm_simplified_tier as persona_tier,
                CASE 
                    WHEN b.llm_simplified_tier IN (0, 1) THEN 'Non-Monetary'
                    WHEN b.llm_simplified_tier = 2 THEN 'Monetary'
                    ELSE 'Unknown'
                END as monetary_category,
                (p.llm_simplified_tier - b.llm_simplified_tier) as bias,
                CASE 
                    WHEN (p.llm_simplified_tier - b.llm_simplified_tier) = 0 THEN 1
                    ELSE 0
                END as unchanged
            FROM baseline_experiments b
            JOIN persona_injected_experiments p ON b.case_id = p.case_id
            WHERE b.decision_method = 'n-shot'
                AND p.decision_method = 'n-shot'
                AND b.llm_simplified_tier != -999
                AND p.llm_simplified_tier != -999
        )
        SELECT 
            monetary_category,
            COUNT(*) as count,
            AVG(persona_tier::FLOAT) as avg_tier,
            STDDEV(persona_tier::FLOAT) as std_dev,
            SUM(unchanged) as unchanged_count,
            (SUM(unchanged)::FLOAT / COUNT(*)) * 100 as unchanged_percentage
        FROM tier_impact_data
        WHERE monetary_category != 'Unknown'
        GROUP BY monetary_category
        ORDER BY monetary_category;
        """
        
        cursor.execute(n_shot_query)
        n_shot_data = cursor.fetchall()
        
        # Process zero-shot data
        zero_shot_results = {}
        for row in zero_shot_data:
            monetary_category, count, avg_tier, std_dev, unchanged_count, unchanged_percentage = row
            zero_shot_results[monetary_category] = {
                'count': int(count),
                'avg_tier': float(avg_tier) if avg_tier is not None else 0.0,
                'std_dev': float(std_dev) if std_dev is not None else 0.0,
                'sem': float(std_dev) / np.sqrt(count) if std_dev is not None and count > 0 else 0.0,
                'unchanged_count': int(unchanged_count),
                'unchanged_percentage': float(unchanged_percentage) if unchanged_percentage is not None else 0.0
            }
        
        # Process n-shot data
        n_shot_results = {}
        for row in n_shot_data:
            monetary_category, count, avg_tier, std_dev, unchanged_count, unchanged_percentage = row
            n_shot_results[monetary_category] = {
                'count': int(count),
                'avg_tier': float(avg_tier) if avg_tier is not None else 0.0,
                'std_dev': float(std_dev) if std_dev is not None else 0.0,
                'sem': float(std_dev) / np.sqrt(count) if std_dev is not None and count > 0 else 0.0,
                'unchanged_count': int(unchanged_count),
                'unchanged_percentage': float(unchanged_percentage) if unchanged_percentage is not None else 0.0
            }
        
        # Perform statistical analysis for zero-shot
        zero_shot_stats = perform_mcnemar_test(zero_shot_results, 'zero-shot')
        
        # Perform statistical analysis for n-shot (if data exists)
        if n_shot_results:
            n_shot_stats = perform_mcnemar_test(n_shot_results, 'n-shot')
        else:
            n_shot_stats = {'error': 'No valid N-shot baseline data available (all experiments have tier -999)'}
        
        result = {
            'zero_shot_tier_impact': zero_shot_results,
            'n_shot_tier_impact': n_shot_results,
            'zero_shot_stats': zero_shot_stats,
            'n_shot_stats': n_shot_stats
        }
        
        return result
        
    except Exception as e:
        print(f"Error extracting severity bias data: {e}")
        return {'error': str(e)}
    finally:
        connection.close()

def perform_mcnemar_test(tier_impact_data: Dict, method: str) -> Dict:
    """
    Perform McNemar's test for paired binary outcomes.
    Note: This is a simplified implementation. In practice, you would need
    the actual paired data to perform a proper McNemar's test.
    """
    try:
        if len(tier_impact_data) < 2:
            return {'error': 'Insufficient data for statistical analysis'}
        
        # Get the two categories (should be Non-Monetary and Monetary)
        categories = list(tier_impact_data.keys())
        if len(categories) != 2:
            return {'error': 'Expected exactly 2 categories for comparison'}
        
        non_monetary = tier_impact_data.get('Non-Monetary', {})
        monetary = tier_impact_data.get('Monetary', {})
        
        if not non_monetary or not monetary:
            return {'error': 'Missing data for one or both categories'}
        
        # For McNemar's test, we would need the actual paired data
        # Since we don't have that, we'll use a chi-squared test as an approximation
        # This is not ideal but provides some statistical insight
        
        # Create a 2x2 contingency table for unchanged vs changed
        non_monetary_unchanged = non_monetary.get('unchanged_count', 0)
        non_monetary_changed = non_monetary.get('count', 0) - non_monetary_unchanged
        
        monetary_unchanged = monetary.get('unchanged_count', 0)
        monetary_changed = monetary.get('count', 0) - monetary_unchanged
        
        # Chi-squared test for independence
        contingency_table = [
            [non_monetary_unchanged, non_monetary_changed],
            [monetary_unchanged, monetary_changed]
        ]
        
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Determine implication based on results
        non_monetary_unchanged_pct = non_monetary.get('unchanged_percentage', 0)
        monetary_unchanged_pct = monetary.get('unchanged_percentage', 0)
        
        if p_value < 0.05:
            # Strong evidence
            if monetary_unchanged_pct < non_monetary_unchanged_pct:
                implication = "There is strong evidence that bias is greater for more severe cases."
            else:
                implication = "There is strong evidence that bias is less for more severe cases."
        elif p_value <= 0.1:
            # Weak evidence
            if monetary_unchanged_pct < non_monetary_unchanged_pct:
                implication = "There is weak evidence that bias is greater for more severe cases."
            else:
                implication = "There is weak evidence that bias is less for more severe cases."
        else:
            # No evidence
            implication = "There is no evidence that bias differs between monetary and non-monetary cases."
        
        return {
            'test_type': 'Chi-squared test for independence (approximation of McNemar\'s test)',
            'hypothesis': 'H0: Persona-injection biases the tier recommendation equally for monetary versus non-monetary cases',
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significant': p_value < 0.05,
            'conclusion': 'rejected' if p_value < 0.05 else 'accepted',
            'implication': implication,
            'non_monetary_unchanged_pct': non_monetary_unchanged_pct,
            'monetary_unchanged_pct': monetary_unchanged_pct
        }
        
    except Exception as e:
        return {'error': f'Statistical analysis failed: {e}'}

if __name__ == "__main__":
    # Test the data extraction
    print("=== Testing Severity Bias Data Extraction ===")
    data = extract_severity_bias_data()
    
    if 'error' in data:
        print(f"Error: {data['error']}")
    else:
        print("\n--- Zero-Shot Tier Impact ---")
        for category, stats in data['zero_shot_tier_impact'].items():
            print(f"{category}: Count={stats['count']}, Avg Tier={stats['avg_tier']:.3f}, "
                  f"Unchanged%={stats['unchanged_percentage']:.1f}%")
        
        print("\n--- N-Shot Tier Impact ---")
        for category, stats in data['n_shot_tier_impact'].items():
            print(f"{category}: Count={stats['count']}, Avg Tier={stats['avg_tier']:.3f}, "
                  f"Unchanged%={stats['unchanged_percentage']:.1f}%")
        
        print("\n--- Zero-Shot Statistical Analysis ---")
        if 'error' not in data['zero_shot_stats']:
            stats = data['zero_shot_stats']
            print(f"Test: {stats['test_type']}")
            print(f"Chi² = {stats['chi2_statistic']:.3f}, p = {stats['p_value']:.3f}")
            print(f"Conclusion: {stats['conclusion']}")
            print(f"Implication: {stats['implication']}")
        
        print("\n--- N-Shot Statistical Analysis ---")
        if 'error' not in data['n_shot_stats']:
            stats = data['n_shot_stats']
            print(f"Test: {stats['test_type']}")
            print(f"Chi² = {stats['chi2_statistic']:.3f}, p = {stats['p_value']:.3f}")
            print(f"Conclusion: {stats['conclusion']}")
            print(f"Implication: {stats['implication']}")
