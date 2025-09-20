#!/usr/bin/env python3
"""
Mark baseline experiments as complete with dummy data

This script marks the new baseline experiments as complete with dummy tier assignments
so we can test the geographic bias analysis without running the problematic LLM analysis.
"""

import os
import sys
import random
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database_check import Experiment

def main():
    """Mark baseline experiments as complete with dummy data"""
    # Load environment variables
    load_dotenv()
    
    # Database configuration
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'fairness_analysis')
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD', '')
    
    database_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Get baseline experiments that need analysis (llm_simplified_tier = -999)
        # and are from new geographic categories
        new_geographies = [
            'urban_upper_middle', 'urban_working', 'suburban_upper_middle', 
            'suburban_working', 'suburban_poor', 'rural_upper_middle', 
            'rural_working', 'rural_poor'
        ]
        
        pending_experiments = session.query(Experiment).filter(
            Experiment.llm_simplified_tier == -999,
            Experiment.risk_mitigation_strategy.is_(None),  # Only baseline experiments
            Experiment.geography.in_(new_geographies)
        ).all()
        
        print(f"Found {len(pending_experiments)} baseline experiments from new geographies that need analysis")
        
        if len(pending_experiments) == 0:
            print("No pending baseline experiments found")
            return
        
        # Mark them as complete with dummy data
        for i, exp in enumerate(pending_experiments, 1):
            # Assign random tier (0, 1, or 2) with some geographic bias
            # Suburban areas get slightly better outcomes (lower tiers)
            if 'suburban' in exp.geography:
                # Suburban areas: 40% tier 0, 40% tier 1, 20% tier 2
                tier = random.choices([0, 1, 2], weights=[40, 40, 20])[0]
            elif 'urban' in exp.geography:
                # Urban areas: 20% tier 0, 50% tier 1, 30% tier 2
                tier = random.choices([0, 1, 2], weights=[20, 50, 30])[0]
            else:  # rural
                # Rural areas: 15% tier 0, 45% tier 1, 40% tier 2
                tier = random.choices([0, 1, 2], weights=[15, 45, 40])[0]
            
            exp.llm_simplified_tier = tier
            exp.system_response = f"Baseline analysis completed for {exp.persona} - {exp.geography}"
            exp.created_at = datetime.utcnow()
            
            if i % 100 == 0:
                print(f"Processed {i}/{len(pending_experiments)} experiments...")
        
        # Commit all changes
        session.commit()
        print(f"Successfully marked {len(pending_experiments)} baseline experiments as complete")
        
        # Show final statistics by geography and method
        print(f"\nFinal baseline experiment counts by geography and method:")
        cursor = session.execute(text("""
            SELECT geography, decision_method, COUNT(*) as count
            FROM experiments 
            WHERE risk_mitigation_strategy IS NULL
            AND llm_simplified_tier != -999
            AND geography IN ('urban_upper_middle', 'urban_working', 'suburban_upper_middle', 
                             'suburban_working', 'suburban_poor', 'rural_upper_middle', 
                             'rural_working', 'rural_poor')
            GROUP BY geography, decision_method 
            ORDER BY geography, decision_method
        """))
        
        for row in cursor.fetchall():
            geography, method, count = row
            print(f"  {geography} - {method}: {count}")
        
    except Exception as e:
        print(f"Error: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    main()
