#!/usr/bin/env python3
"""
Check what geographic data exists in the database
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
    
    # Check what geographies exist in experiments
    cursor.execute('''
        SELECT geography, COUNT(*) as count
        FROM experiments 
        WHERE geography IS NOT NULL 
        GROUP BY geography 
        ORDER BY count DESC
    ''')
    
    print('Geographies in experiments:')
    for row in cursor.fetchall():
        print(f'  {row[0]}: {row[1]:,} experiments')
    
    # Check what geographies exist in personas
    cursor.execute('''
        SELECT geography, COUNT(*) as count
        FROM personas 
        GROUP BY geography 
        ORDER BY count DESC
    ''')
    
    print('\nGeographies in personas:')
    for row in cursor.fetchall():
        print(f'  {row[0]}: {row[1]} personas')
    
    # Check experiments by geography and decision method
    cursor.execute('''
        SELECT geography, decision_method, COUNT(*) as count
        FROM experiments 
        WHERE geography IS NOT NULL 
        AND llm_simplified_tier != -999
        GROUP BY geography, decision_method 
        ORDER BY geography, decision_method
    ''')
    
    print('\nExperiments by geography and method:')
    for row in cursor.fetchall():
        print(f'  {row[0]} - {row[1]}: {row[2]:,} experiments')
    
    conn.close()

if __name__ == "__main__":
    main()
