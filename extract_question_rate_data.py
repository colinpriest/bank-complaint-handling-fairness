#!/usr/bin/env python3
"""
Extract Question Rate Data for Dashboard

This script extracts question rate data from the database to populate
the Process Bias sub-tab in the HTML dashboard.
"""

import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()

def get_db_connection():
    """Get database connection"""
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'fairness_analysis'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', '')
    }
    return psycopg2.connect(**db_config)

def extract_question_rate_data() -> Dict[str, Any]:
    """
    Extract question rate data from the database for Process Bias analysis
    
    Returns:
        Dictionary containing question rate data for dashboard
    """
    connection = get_db_connection()
    
    try:
        # Query 1: Zero-Shot Question Rate - Persona-Injected vs. Baseline
        zero_shot_query = """
        SELECT 
            SUM(CASE WHEN e.persona IS NULL AND e.risk_mitigation_strategy IS NULL 
                     AND e.decision_method = 'zero-shot' AND e.llm_simplified_tier != -999
                     AND e.asks_for_info = true THEN 1 ELSE 0 END) as baseline_questions,
            SUM(CASE WHEN e.persona IS NULL AND e.risk_mitigation_strategy IS NULL 
                     AND e.decision_method = 'zero-shot' AND e.llm_simplified_tier != -999
                     THEN 1 ELSE 0 END) as baseline_count,
            SUM(CASE WHEN e.persona IS NOT NULL AND e.risk_mitigation_strategy IS NULL 
                     AND e.decision_method = 'zero-shot' AND e.llm_simplified_tier != -999
                     AND e.asks_for_info = true THEN 1 ELSE 0 END) as persona_questions,
            SUM(CASE WHEN e.persona IS NOT NULL AND e.risk_mitigation_strategy IS NULL 
                     AND e.decision_method = 'zero-shot' AND e.llm_simplified_tier != -999
                     THEN 1 ELSE 0 END) as persona_count
        FROM experiments e
        WHERE e.decision_method = 'zero-shot' 
            AND e.llm_simplified_tier != -999
            AND e.risk_mitigation_strategy IS NULL;
        """
        
        # Query 2: N-Shot Question Rate - Persona-Injected vs. Baseline
        n_shot_query = """
        SELECT 
            SUM(CASE WHEN e.persona IS NULL AND e.risk_mitigation_strategy IS NULL 
                     AND e.decision_method = 'n-shot' AND e.llm_simplified_tier != -999
                     AND e.asks_for_info = true THEN 1 ELSE 0 END) as baseline_questions,
            SUM(CASE WHEN e.persona IS NULL AND e.risk_mitigation_strategy IS NULL 
                     AND e.decision_method = 'n-shot' AND e.llm_simplified_tier != -999
                     THEN 1 ELSE 0 END) as baseline_count,
            SUM(CASE WHEN e.persona IS NOT NULL AND e.risk_mitigation_strategy IS NULL 
                     AND e.decision_method = 'n-shot' AND e.llm_simplified_tier != -999
                     AND e.asks_for_info = true THEN 1 ELSE 0 END) as persona_questions,
            SUM(CASE WHEN e.persona IS NOT NULL AND e.risk_mitigation_strategy IS NULL 
                     AND e.decision_method = 'n-shot' AND e.llm_simplified_tier != -999
                     THEN 1 ELSE 0 END) as persona_count
        FROM experiments e
        WHERE e.decision_method = 'n-shot' 
            AND e.llm_simplified_tier != -999
            AND e.risk_mitigation_strategy IS NULL;
        """
        
        # Query 3: N-Shot vs Zero-Shot Comparison (Persona-Injected only)
        nshot_vs_zeroshot_query = """
        WITH zero_shot_persona AS (
            SELECT 
                SUM(CASE WHEN e.asks_for_info = true THEN 1 ELSE 0 END) as zero_shot_questions,
                COUNT(*) as zero_shot_count
            FROM experiments e
            WHERE e.decision_method = 'zero-shot' 
                AND e.persona IS NOT NULL 
                AND e.risk_mitigation_strategy IS NULL
                AND e.llm_simplified_tier != -999
        ),
        n_shot_persona AS (
            SELECT 
                SUM(CASE WHEN e.asks_for_info = true THEN 1 ELSE 0 END) as n_shot_questions,
                COUNT(*) as n_shot_count
            FROM experiments e
            WHERE e.decision_method = 'n-shot' 
                AND e.persona IS NOT NULL 
                AND e.risk_mitigation_strategy IS NULL
                AND e.llm_simplified_tier != -999
        )
        SELECT 
            COALESCE(z.zero_shot_questions, 0) as zero_shot_questions,
            COALESCE(z.zero_shot_count, 0) as zero_shot_count,
            COALESCE(n.n_shot_questions, 0) as n_shot_questions,
            COALESCE(n.n_shot_count, 0) as n_shot_count
        FROM zero_shot_persona z
        CROSS JOIN n_shot_persona n;
        """
        
        # Execute queries
        cursor = connection.cursor()
        
        # Zero-shot data
        cursor.execute(zero_shot_query)
        zero_shot_data = cursor.fetchone()
        
        # N-shot data
        cursor.execute(n_shot_query)
        n_shot_data = cursor.fetchone()
        
        # N-shot vs Zero-shot data
        cursor.execute(nshot_vs_zeroshot_query)
        nshot_vs_zeroshot_data = cursor.fetchone()
        
        # Build result dictionary with proper type conversion
        result = {
            'zero_shot_question_rate': {
                'baseline_count': float(zero_shot_data[1]) if zero_shot_data and zero_shot_data[1] is not None else 0,
                'baseline_questions': float(zero_shot_data[0]) if zero_shot_data and zero_shot_data[0] is not None else 0,
                'persona_count': float(zero_shot_data[3]) if zero_shot_data and zero_shot_data[3] is not None else 0,
                'persona_questions': float(zero_shot_data[2]) if zero_shot_data and zero_shot_data[2] is not None else 0
            },
            'n_shot_question_rate': {
                'baseline_count': float(n_shot_data[1]) if n_shot_data and n_shot_data[1] is not None else 0,
                'baseline_questions': float(n_shot_data[0]) if n_shot_data and n_shot_data[0] is not None else 0,
                'persona_count': float(n_shot_data[3]) if n_shot_data and n_shot_data[3] is not None else 0,
                'persona_questions': float(n_shot_data[2]) if n_shot_data and n_shot_data[2] is not None else 0
            },
            'nshot_vs_zeroshot_comparison': {
                'zero_shot_count': float(nshot_vs_zeroshot_data[1]) if nshot_vs_zeroshot_data and nshot_vs_zeroshot_data[1] is not None else 0,
                'zero_shot_questions': float(nshot_vs_zeroshot_data[0]) if nshot_vs_zeroshot_data and nshot_vs_zeroshot_data[0] is not None else 0,
                'n_shot_count': float(nshot_vs_zeroshot_data[3]) if nshot_vs_zeroshot_data and nshot_vs_zeroshot_data[3] is not None else 0,
                'n_shot_questions': float(nshot_vs_zeroshot_data[2]) if nshot_vs_zeroshot_data and nshot_vs_zeroshot_data[2] is not None else 0
            }
        }
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Failed to extract question rate data: {e}")
        return {
            'zero_shot_question_rate': {},
            'n_shot_question_rate': {},
            'nshot_vs_zeroshot_comparison': {}
        }
    finally:
        connection.close()

def main():
    """Test the question rate data extraction"""
    print("Extracting question rate data from database...")
    
    data = extract_question_rate_data()
    
    print("\nQuestion Rate Data:")
    print("=" * 50)
    
    for key, value in data.items():
        print(f"\n{key}:")
        for sub_key, sub_value in value.items():
            print(f"  {sub_key}: {sub_value:,}")

if __name__ == "__main__":
    main()
