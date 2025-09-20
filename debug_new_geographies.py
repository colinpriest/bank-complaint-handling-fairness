#!/usr/bin/env python3
"""
Debug why new geographic categories aren't showing up in the analysis
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
    
    # Check what's in the new geographic categories
    new_geographies = ['urban_upper_middle', 'urban_working', 'suburban_upper_middle', 'suburban_working', 'suburban_poor', 'rural_upper_middle', 'rural_working', 'rural_poor']
    
    for geo in new_geographies:
        print(f"\n=== {geo} ===")
        
        # Check total experiments
        cursor.execute("SELECT COUNT(*) FROM experiments WHERE geography = %s", (geo,))
        total = cursor.fetchone()[0]
        print(f"Total experiments: {total}")
        
        # Check by decision method
        cursor.execute("SELECT decision_method, COUNT(*) FROM experiments WHERE geography = %s GROUP BY decision_method", (geo,))
        for method, count in cursor.fetchall():
            print(f"  {method}: {count}")
        
        # Check by risk_mitigation_strategy
        cursor.execute("SELECT risk_mitigation_strategy, COUNT(*) FROM experiments WHERE geography = %s GROUP BY risk_mitigation_strategy", (geo,))
        for strategy, count in cursor.fetchall():
            print(f"  risk_mitigation_strategy={strategy}: {count}")
        
        # Check by llm_simplified_tier
        cursor.execute("SELECT llm_simplified_tier, COUNT(*) FROM experiments WHERE geography = %s GROUP BY llm_simplified_tier", (geo,))
        for tier, count in cursor.fetchall():
            print(f"  llm_simplified_tier={tier}: {count}")
        
        # Check experiments that meet the analysis criteria
        cursor.execute("""
            SELECT COUNT(*) FROM experiments e
            WHERE e.geography = %s
                AND e.decision_method = 'zero-shot'
                AND e.persona IS NOT NULL
                AND e.risk_mitigation_strategy IS NULL
                AND e.llm_simplified_tier != -999
        """, (geo,))
        zero_shot_count = cursor.fetchone()[0]
        print(f"  Zero-shot with NULL mitigation: {zero_shot_count}")
        
        cursor.execute("""
            SELECT COUNT(*) FROM experiments e
            WHERE e.geography = %s
                AND e.decision_method = 'n-shot'
                AND e.persona IS NOT NULL
                AND e.risk_mitigation_strategy IS NULL
                AND e.llm_simplified_tier != -999
        """, (geo,))
        n_shot_count = cursor.fetchone()[0]
        print(f"  N-shot with NULL mitigation: {n_shot_count}")
    
    conn.close()

if __name__ == "__main__":
    main()
