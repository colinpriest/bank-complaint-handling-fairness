#!/usr/bin/env python3
"""
Create complete baseline experiments for new geographic categories

This script creates baseline experiments (risk_mitigation_strategy = NULL) 
for both zero-shot and n-shot methods for the new geographic categories.
"""

import os
import sys
import random
from datetime import datetime
from typing import Dict, List, Any
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database_check import Experiment, Persona, GroundTruth

class CompleteBaselineExperimentCreator:
    """Creates complete baseline experiments for new geographic categories"""
    
    def __init__(self):
        """Initialize with PostgreSQL database connection"""
        # Load environment variables
        load_dotenv()
        
        # Database configuration
        self.db_host = os.getenv('DB_HOST', 'localhost')
        self.db_port = os.getenv('DB_PORT', '5432')
        self.db_name = os.getenv('DB_NAME', 'fairness_analysis')
        self.db_user = os.getenv('DB_USER', 'postgres')
        self.db_password = os.getenv('DB_PASSWORD', '')
        
        self.database_url = f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        self.engine = create_engine(self.database_url)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # New geographic categories that need baseline experiments
        self.new_geographies = [
            'urban_upper_middle', 'urban_working', 'suburban_upper_middle', 
            'suburban_working', 'suburban_poor', 'rural_upper_middle', 
            'rural_working', 'rural_poor'
        ]

    def get_personas_for_geography(self, geography: str) -> List[Dict]:
        """Get all personas for a specific geography"""
        try:
            personas = self.session.query(Persona).filter(Persona.geography == geography).all()
            return [
                {
                    'id': p.id,
                    'key': p.key,
                    'ethnicity': p.ethnicity,
                    'gender': p.gender,
                    'geography': p.geography
                }
                for p in personas
            ]
        except Exception as e:
            print(f"Error getting personas for {geography}: {e}")
            return []

    def get_ground_truths(self, limit: int = 100) -> List[Dict]:
        """Get ground truth cases for baseline experiments"""
        try:
            ground_truths = self.session.query(GroundTruth).limit(limit).all()
            return [
                {
                    'case_id': gt.case_id,
                    'simplified_ground_truth_tier': gt.simplified_ground_truth_tier,
                    'consumer_complaint_text': gt.consumer_complaint_text
                }
                for gt in ground_truths
            ]
        except Exception as e:
            print(f"Error getting ground truths: {e}")
            return []

    def get_existing_baseline_experiments(self, geography: str) -> set:
        """Get existing baseline experiments for a geography to avoid duplicates"""
        try:
            experiments = self.session.query(Experiment).filter(
                Experiment.geography == geography,
                Experiment.risk_mitigation_strategy.is_(None)
            ).all()
            
            existing = set()
            for exp in experiments:
                key = (exp.persona, exp.case_id, exp.decision_method)
                existing.add(key)
            
            return existing
        except Exception as e:
            print(f"Error getting existing experiments for {geography}: {e}")
            return set()

    def create_baseline_experiments_for_geography(self, geography: str, experiments_per_persona_per_method: int = 50) -> Dict[str, int]:
        """Create baseline experiments for both zero-shot and n-shot for a specific geography"""
        print(f"Creating baseline experiments for {geography}...")
        
        # Get personas for this geography
        personas = self.get_personas_for_geography(geography)
        if not personas:
            print(f"No personas found for {geography}")
            return {'zero-shot': 0, 'n-shot': 0}
        
        # Get ground truth cases
        ground_truths = self.get_ground_truths(100)
        if not ground_truths:
            print("No ground truth cases found")
            return {'zero-shot': 0, 'n-shot': 0}
        
        # Get existing experiments to avoid duplicates
        existing_experiments = self.get_existing_baseline_experiments(geography)
        
        experiments_created = {'zero-shot': 0, 'n-shot': 0}
        decision_methods = ['zero-shot', 'n-shot']
        
        for persona in personas:
            for decision_method in decision_methods:
                for _ in range(experiments_per_persona_per_method):
                    # Randomly select ground truth
                    ground_truth = random.choice(ground_truths)
                    
                    # Check if this combination already exists
                    experiment_key = (persona['key'], ground_truth['case_id'], decision_method)
                    if experiment_key in existing_experiments:
                        continue
                    
                    # Create new baseline experiment
                    new_experiment = Experiment(
                        case_id=ground_truth['case_id'],
                        decision_method=decision_method,
                        llm_model='gpt-4o-mini',
                        llm_simplified_tier=-999,  # Mark as needing analysis
                        persona=persona['key'],
                        gender=persona['gender'],
                        ethnicity=persona['ethnicity'],
                        geography=persona['geography'],
                        risk_mitigation_strategy=None,  # Baseline experiment
                        created_at=datetime.utcnow()
                    )
                    
                    self.session.add(new_experiment)
                    existing_experiments.add(experiment_key)
                    experiments_created[decision_method] += 1
                    
                    # Commit in batches
                    if (experiments_created['zero-shot'] + experiments_created['n-shot']) % 50 == 0:
                        try:
                            self.session.commit()
                            print(f"  Created {experiments_created['zero-shot']} zero-shot, {experiments_created['n-shot']} n-shot experiments...")
                        except Exception as e:
                            self.session.rollback()
                            print(f"Error committing batch: {e}")
                            return experiments_created
        
        try:
            self.session.commit()
            print(f"Successfully created {experiments_created['zero-shot']} zero-shot and {experiments_created['n-shot']} n-shot baseline experiments for {geography}")
            return experiments_created
        except Exception as e:
            self.session.rollback()
            print(f"Error creating experiments for {geography}: {e}")
            return experiments_created

    def create_all_baseline_experiments(self, experiments_per_persona_per_method: int = 50) -> Dict[str, Dict[str, int]]:
        """Create baseline experiments for all new geographic categories"""
        print("Creating complete baseline experiments for new geographic categories...")
        print("=" * 70)
        
        results = {}
        total_created = {'zero-shot': 0, 'n-shot': 0}
        
        for geography in self.new_geographies:
            created = self.create_baseline_experiments_for_geography(geography, experiments_per_persona_per_method)
            results[geography] = created
            total_created['zero-shot'] += created['zero-shot']
            total_created['n-shot'] += created['n-shot']
        
        print(f"\nSummary:")
        print(f"Total baseline experiments created:")
        print(f"  Zero-shot: {total_created['zero-shot']}")
        print(f"  N-shot: {total_created['n-shot']}")
        print(f"  Total: {total_created['zero-shot'] + total_created['n-shot']}")
        
        print(f"\nBy geography:")
        for geography, counts in results.items():
            print(f"  {geography}: {counts['zero-shot']} zero-shot, {counts['n-shot']} n-shot")
        
        return results

    def get_baseline_experiment_counts(self) -> Dict[str, Dict[str, int]]:
        """Get counts of baseline experiments by geography and method"""
        try:
            cursor = self.session.execute(text("""
                SELECT geography, decision_method, COUNT(*) as count
                FROM experiments 
                WHERE risk_mitigation_strategy IS NULL
                AND llm_simplified_tier != -999
                GROUP BY geography, decision_method 
                ORDER BY geography, decision_method
            """))
            
            results = {}
            for row in cursor.fetchall():
                geography, method, count = row
                if geography not in results:
                    results[geography] = {}
                results[geography][method] = count
            
            return results
        except Exception as e:
            print(f"Error getting baseline experiment counts: {e}")
            return {}

    def close(self):
        """Close the database session"""
        self.session.close()

def main():
    """Main function to create complete baseline experiments"""
    creator = CompleteBaselineExperimentCreator()
    
    try:
        # Show current baseline experiment counts
        print("Current baseline experiment counts:")
        current_counts = creator.get_baseline_experiment_counts()
        for geo, methods in current_counts.items():
            zero_shot = methods.get('zero-shot', 0)
            n_shot = methods.get('n-shot', 0)
            print(f"  {geo}: {zero_shot} zero-shot, {n_shot} n-shot")
        print()
        
        # Create baseline experiments
        results = creator.create_all_baseline_experiments(experiments_per_persona_per_method=50)
        
        # Show final counts
        print("\nFinal baseline experiment counts:")
        final_counts = creator.get_baseline_experiment_counts()
        for geo, methods in final_counts.items():
            zero_shot = methods.get('zero-shot', 0)
            n_shot = methods.get('n-shot', 0)
            print(f"  {geo}: {zero_shot} zero-shot, {n_shot} n-shot")
        
        print(f"\n✅ Successfully created complete baseline experiments!")
        print("You can now run the geographic bias analysis to include all categories for both methods.")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        creator.close()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
