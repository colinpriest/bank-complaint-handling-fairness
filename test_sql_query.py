#!/usr/bin/env python3
"""
Test the SQL query directly to see what data is returned
"""

import os
from dotenv import load_dotenv
import psycopg2

def main():
    load_dotenv()
    
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        database=os.getenv('DB_NAME', 'fairness_analysis'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', '')
    )
    
    cursor = conn.cursor()
    
    # Test the exact query from the extraction script
    zero_shot_mean_query = """
    SELECT 
        e.geography,
        AVG(e.llm_simplified_tier::FLOAT) as mean_tier,
        COUNT(*) as experiment_count,
        STDDEV(e.llm_simplified_tier::FLOAT) as std_dev
    FROM experiments e
    WHERE e.decision_method = 'zero-shot'
        AND e.persona IS NOT NULL
        AND e.risk_mitigation_strategy IS NULL
        AND e.llm_simplified_tier != -999
        AND e.geography IS NOT NULL
        AND e.geography IN ('rural', 'urban_affluent', 'urban_poor', 'urban_upper_middle', 'urban_working', 'suburban_upper_middle', 'suburban_working', 'suburban_poor', 'rural_upper_middle', 'rural_working', 'rural_poor')
    GROUP BY e.geography
    ORDER BY e.geography;
    """
    
    print("Testing zero-shot mean query:")
    cursor.execute(zero_shot_mean_query)
    results = cursor.fetchall()
    
    for row in results:
        geography, mean_tier, count, std_dev = row
        print(f"  {geography}: Mean={mean_tier:.3f}, Count={count}, StdDev={std_dev:.3f}")
    
    # Test n-shot query too
    n_shot_mean_query = """
    SELECT 
        e.geography,
        AVG(e.llm_simplified_tier::FLOAT) as mean_tier,
        COUNT(*) as experiment_count,
        STDDEV(e.llm_simplified_tier::FLOAT) as std_dev
    FROM experiments e
    WHERE e.decision_method = 'n-shot'
        AND e.persona IS NOT NULL
        AND e.risk_mitigation_strategy IS NULL
        AND e.llm_simplified_tier != -999
        AND e.geography IS NOT NULL
        AND e.geography IN ('rural', 'urban_affluent', 'urban_poor', 'urban_upper_middle', 'urban_working', 'suburban_upper_middle', 'suburban_working', 'suburban_poor', 'rural_upper_middle', 'rural_working', 'rural_poor')
    GROUP BY e.geography
    ORDER BY e.geography;
    """
    
    print("\nTesting n-shot mean query:")
    cursor.execute(n_shot_mean_query)
    results = cursor.fetchall()
    
    for row in results:
        geography, mean_tier, count, std_dev = row
        print(f"  {geography}: Mean={mean_tier:.3f}, Count={count}, StdDev={std_dev:.3f}")
    
    conn.close()

if __name__ == "__main__":
    main()
