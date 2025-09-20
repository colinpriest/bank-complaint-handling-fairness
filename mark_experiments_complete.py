#!/usr/bin/env python3
"""
Mark Experiments Complete Script

This script marks pending experiments as complete with dummy data
so you can test the sampling system without running actual LLM analysis.
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
    """Mark pending experiments as complete with dummy data"""
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
        # Get pending experiments
        pending_experiments = session.query(Experiment).filter(
            Experiment.llm_simplified_tier == -999
        ).all()
        
        print(f"Found {len(pending_experiments)} pending experiments")
        
        if len(pending_experiments) == 0:
            print("No pending experiments found")
            return
        
        # Mark them as complete with dummy data
        for i, exp in enumerate(pending_experiments, 1):
            # Assign random tier (0, 1, or 2)
            exp.llm_simplified_tier = random.randint(0, 2)
            exp.system_response = f"Dummy analysis completed for {exp.persona}"
            exp.created_at = datetime.utcnow()
            
            if i % 100 == 0:
                print(f"Processed {i}/{len(pending_experiments)} experiments...")
        
        # Commit all changes
        session.commit()
        print(f"Successfully marked {len(pending_experiments)} experiments as complete")
        
        # Show final statistics
        total_experiments = session.query(Experiment).count()
        pending_count = session.query(Experiment).filter(
            Experiment.llm_simplified_tier == -999
        ).count()
        completed_count = total_experiments - pending_count
        
        print(f"\nFinal Statistics:")
        print(f"  Total experiments: {total_experiments:,}")
        print(f"  Completed: {completed_count:,}")
        print(f"  Pending: {pending_count:,}")
        print(f"  Completion: {(completed_count / total_experiments * 100):.1f}%")
        
    except Exception as e:
        print(f"Error: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    main()
