#!/usr/bin/env python3
"""
Geographic Persona Sampling Script

This script creates a practical experimental design that uses random sampling
to cover all geographic persona combinations without creating every possible
experiment at once.

Instead of creating all combinations (which would be massive), it:
1. Adds the new geographic categories to the personas table
2. Creates a sampling strategy that randomly selects combinations
3. Allows you to run experiments in batches over time
4. Tracks coverage to ensure all combinations are eventually tested
"""

import os
import sys
import json
import random
from datetime import datetime
from typing import Dict, List, Tuple, Any
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database_check import Persona, Experiment, GroundTruth, MitigationStrategy

class GeographicPersonaSampler:
    """Handles geographic persona expansion with sampling strategy"""
    
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
        
        # New geographic categories to add
        self.new_geographies = [
            'urban_upper_middle', 'urban_working', 'suburban_upper_middle', 
            'suburban_working', 'suburban_poor', 'rural_upper_middle', 
            'rural_working', 'rural_poor'
        ]
        
        # Existing geographies to map from
        self.existing_geographies = ['urban_affluent', 'urban_poor', 'rural']
        
        # Sampling parameters
        self.batch_size = 1000  # Experiments per batch
        self.max_batches = 10   # Maximum batches to create initially

    def get_existing_personas(self) -> List[Dict]:
        """Get all existing personas from the database"""
        try:
            personas = self.session.query(Persona).all()
            return [
                {
                    'id': p.id,
                    'key': p.key,
                    'ethnicity': p.ethnicity,
                    'gender': p.gender,
                    'geography': p.geography,
                    'language_style': p.language_style,
                    'typical_names': json.loads(p.typical_names) if p.typical_names else [],
                    'typical_locations': json.loads(p.typical_locations) if p.typical_locations else [],
                    'typical_companies': json.loads(p.typical_companies) if p.typical_companies else [],
                    'typical_products': json.loads(p.typical_products) if p.typical_products else [],
                    'typical_activities': json.loads(p.typical_activities) if p.typical_activities else [],
                    'geography_hints': json.loads(p.geography_hints) if p.geography_hints else [],
                    'typical_occupations': json.loads(p.typical_occupations) if p.typical_occupations else [],
                    'gender_hints': json.loads(p.gender_hints) if p.gender_hints else [],
                    'ethnicity_hints': json.loads(p.ethnicity_hints) if p.ethnicity_hints else []
                }
                for p in personas
            ]
        except Exception as e:
            print(f"Error getting existing personas: {e}")
            return []

    def get_geography_mapping(self) -> Dict[str, str]:
        """Map new geographies to existing ones for persona creation"""
        return {
            'urban_upper_middle': 'urban_affluent',
            'urban_working': 'urban_affluent', 
            'suburban_upper_middle': 'urban_affluent',
            'suburban_working': 'urban_poor',
            'suburban_poor': 'urban_poor',
            'rural_upper_middle': 'rural',
            'rural_working': 'rural',
            'rural_poor': 'rural'
        }

    def get_geography_hints(self, geography: str) -> List[str]:
        """Get appropriate hints for each geography"""
        hints_map = {
            'urban_upper_middle': ['in my downtown office', 'at the business district', 'in my condo', 'at the upscale restaurant'],
            'urban_working': ['on the subway', 'at my union hall', 'in my apartment', 'at the local diner'],
            'suburban_upper_middle': ['in my home office', 'at the country club', 'in my SUV', 'at the Whole Foods'],
            'suburban_working': ['in my car', 'at the strip mall', 'in my townhouse', 'at the chain restaurant'],
            'suburban_poor': ['on the bus', 'at the discount store', 'in my apartment', 'at the fast food place'],
            'rural_upper_middle': ['on my ranch', 'at the farmers market', 'in my truck', 'at the local cafe'],
            'rural_working': ['at the feed store', 'in my pickup', 'at the diner', 'on the farm'],
            'rural_poor': ['on the long drive into town', 'at the general store', 'in my old car', 'at the community center']
        }
        return hints_map.get(geography, [])

    def get_occupations(self, geography: str) -> List[str]:
        """Get appropriate occupations for each geography"""
        occupations_map = {
            'urban_upper_middle': ['executive', 'lawyer', 'doctor', 'consultant', 'banker'],
            'urban_working': ['teacher', 'nurse', 'mechanic', 'retail manager', 'office worker'],
            'suburban_upper_middle': ['engineer', 'manager', 'sales rep', 'accountant', 'realtor'],
            'suburban_working': ['truck driver', 'warehouse worker', 'retail worker', 'maintenance', 'security'],
            'suburban_poor': ['fast food worker', 'temp worker', 'day laborer', 'cleaner', 'cashier'],
            'rural_upper_middle': ['farmer', 'rancher', 'veterinarian', 'contractor', 'small business owner'],
            'rural_working': ['farm worker', 'mechanic', 'truck driver', 'construction', 'warehouse'],
            'rural_poor': ['farm worker', 'day laborer', 'retail worker', 'custodian', 'temp worker']
        }
        return occupations_map.get(geography, [])

    def create_new_personas(self) -> int:
        """Create new personas for the expanded geographic categories"""
        print("Creating new personas for expanded geographic categories...")
        
        existing_personas = self.get_existing_personas()
        geography_mapping = self.get_geography_mapping()
        
        # Get all unique ethnicity/gender combinations
        ethnicity_gender_combos = set()
        for persona in existing_personas:
            ethnicity_gender_combos.add((persona['ethnicity'], persona['gender']))
        
        new_personas_created = 0
        
        for ethnicity, gender in ethnicity_gender_combos:
            for new_geography in self.new_geographies:
                # Check if this persona already exists
                persona_key = f"{ethnicity}_{gender}_{new_geography}"
                if any(p['key'] == persona_key for p in existing_personas):
                    continue
                
                # Find a similar existing persona to copy from
                source_geography = geography_mapping[new_geography]
                source_persona = None
                for p in existing_personas:
                    if p['ethnicity'] == ethnicity and p['gender'] == gender and p['geography'] == source_geography:
                        source_persona = p
                        break
                
                if not source_persona:
                    print(f"Warning: No source persona found for {ethnicity}_{gender}_{source_geography}")
                    continue
                
                # Create new persona
                new_persona = Persona(
                    key=persona_key,
                    ethnicity=ethnicity,
                    gender=gender,
                    geography=new_geography,
                    language_style=source_persona['language_style'],
                    typical_names=json.dumps(source_persona['typical_names']),
                    typical_locations=json.dumps(source_persona['typical_locations']),
                    typical_companies=json.dumps(source_persona['typical_companies']),
                    typical_products=json.dumps(source_persona['typical_products']),
                    typical_activities=json.dumps(source_persona['typical_activities']),
                    geography_hints=json.dumps(self.get_geography_hints(new_geography)),
                    typical_occupations=json.dumps(self.get_occupations(new_geography)),
                    gender_hints=json.dumps(source_persona['gender_hints']),
                    ethnicity_hints=json.dumps(source_persona['ethnicity_hints'])
                )
                
                self.session.add(new_persona)
                new_personas_created += 1
        
        try:
            self.session.commit()
            print(f"Successfully created {new_personas_created} new personas")
            return new_personas_created
        except Exception as e:
            self.session.rollback()
            print(f"Error creating personas: {e}")
            return 0

    def get_all_personas(self) -> List[Dict]:
        """Get all personas including newly created ones"""
        return self.get_existing_personas()

    def get_ground_truths(self) -> List[Dict]:
        """Get all ground truth cases"""
        try:
            ground_truths = self.session.query(GroundTruth).all()
            return [
                {
                    'case_id': gt.case_id,
                    'raw_ground_truth_tier': gt.raw_ground_truth_tier,
                    'simplified_ground_truth_tier': gt.simplified_ground_truth_tier,
                    'example_id': gt.example_id,
                    'consumer_complaint_text': gt.consumer_complaint_text,
                    'complaint_category': gt.complaint_category,
                    'product': gt.product,
                    'sub_product': gt.sub_product,
                    'issue': gt.issue,
                    'sub_issue': gt.sub_issue,
                    'vector_embeddings': gt.vector_embeddings,
                    'created_at': gt.created_at
                }
                for gt in ground_truths
            ]
        except Exception as e:
            print(f"Error getting ground truths: {e}")
            return []

    def get_mitigation_strategies(self) -> List[Dict]:
        """Get all mitigation strategies"""
        try:
            strategies = self.session.query(MitigationStrategy).all()
            return [
                {
                    'id': ms.id,
                    'key': ms.key,
                    'name': ms.name,
                    'description': ms.description,
                    'prompt_modification': ms.prompt_modification,
                    'created_at': ms.created_at
                }
                for ms in strategies
            ]
        except Exception as e:
            print(f"Error getting mitigation strategies: {e}")
            return []

    def create_sampled_experiments(self, batch_size: int = None) -> int:
        """Create a batch of randomly sampled experiments"""
        if batch_size is None:
            batch_size = self.batch_size
            
        print(f"Creating batch of {batch_size} sampled experiments...")
        
        personas = self.get_all_personas()
        ground_truths = self.get_ground_truths()
        mitigation_strategies = self.get_mitigation_strategies()
        decision_methods = ['zero-shot', 'n-shot']
        
        if not personas or not ground_truths or not mitigation_strategies:
            print("Error: Missing required data (personas, ground_truths, or mitigation_strategies)")
            return 0
        
        # Get existing experiments to avoid duplicates
        existing_experiments = set()
        try:
            experiments = self.session.query(Experiment).all()
            for exp in experiments:
                key = (exp.persona, exp.case_id, exp.risk_mitigation_strategy, exp.decision_method)
                existing_experiments.add(key)
        except Exception as e:
            print(f"Error getting existing experiments: {e}")
            return 0
        
        experiments_created = 0
        attempts = 0
        max_attempts = batch_size * 10  # Prevent infinite loops
        
        while experiments_created < batch_size and attempts < max_attempts:
            attempts += 1
            
            # Randomly sample components
            persona = random.choice(personas)
            ground_truth = random.choice(ground_truths)
            mitigation_strategy = random.choice(mitigation_strategies)
            decision_method = random.choice(decision_methods)
            
            # Check if this combination already exists
            experiment_key = (persona['key'], ground_truth['case_id'], mitigation_strategy['key'], decision_method)
            if experiment_key in existing_experiments:
                continue
            
            # Create new experiment
            new_experiment = Experiment(
                case_id=ground_truth['case_id'],
                decision_method=decision_method,
                llm_model='gpt-4o-mini',  # Default model
                llm_simplified_tier=-999,  # Mark as needing analysis
                persona=persona['key'],
                gender=persona['gender'],
                ethnicity=persona['ethnicity'],
                geography=persona['geography'],
                risk_mitigation_strategy=mitigation_strategy['key'],
                created_at=datetime.utcnow()
            )
            
            self.session.add(new_experiment)
            existing_experiments.add(experiment_key)
            experiments_created += 1
            
            # Commit in batches to avoid memory issues
            if experiments_created % 100 == 0:
                try:
                    self.session.commit()
                    print(f"  Committed {experiments_created} experiments...")
                except Exception as e:
                    self.session.rollback()
                    print(f"Error committing batch: {e}")
                    return experiments_created
        
        try:
            self.session.commit()
            print(f"Successfully created {experiments_created} new experiments")
            return experiments_created
        except Exception as e:
            self.session.rollback()
            print(f"Error creating experiments: {e}")
            return experiments_created

    def get_sampling_statistics(self) -> Dict[str, Any]:
        """Get statistics about the sampling coverage"""
        try:
            # Get total possible combinations
            personas = self.get_all_personas()
            ground_truths = self.get_ground_truths()
            mitigation_strategies = self.get_mitigation_strategies()
            decision_methods = ['zero-shot', 'n-shot']
            
            total_possible = len(personas) * len(ground_truths) * len(mitigation_strategies) * len(decision_methods)
            
            # Get existing experiments
            existing_count = self.session.query(Experiment).count()
            
            # Get experiments by geography
            geography_stats = {}
            for persona in personas:
                geo = persona['geography']
                if geo not in geography_stats:
                    geography_stats[geo] = 0
                # Count experiments for this persona
                count = self.session.query(Experiment).filter(Experiment.persona == persona['key']).count()
                geography_stats[geo] += count
            
            return {
                'total_possible_combinations': total_possible,
                'existing_experiments': existing_count,
                'coverage_percentage': (existing_count / total_possible * 100) if total_possible > 0 else 0,
                'geography_stats': geography_stats,
                'personas_count': len(personas),
                'ground_truths_count': len(ground_truths),
                'mitigation_strategies_count': len(mitigation_strategies)
            }
        except Exception as e:
            print(f"Error getting sampling statistics: {e}")
            return {}

    def run_sampling_setup(self) -> bool:
        """Run the initial sampling setup"""
        print("Geographic Persona Sampling Setup")
        print("=" * 60)
        print("This script sets up a sampling-based experimental design.")
        print("Instead of creating all combinations, it uses random sampling")
        print("to cover all combinations over time.")
        print()
        
        try:
            # Step 1: Create new personas
            print("Step 1: Creating new personas...")
            personas_created = self.create_new_personas()
            if personas_created == 0:
                print("No new personas created. They may already exist.")
            
            # Step 2: Show current statistics
            print("\nStep 2: Current sampling statistics...")
            stats = self.get_sampling_statistics()
            print(f"Total possible combinations: {stats.get('total_possible_combinations', 0):,}")
            print(f"Existing experiments: {stats.get('existing_experiments', 0):,}")
            print(f"Coverage: {stats.get('coverage_percentage', 0):.2f}%")
            print(f"Personas: {stats.get('personas_count', 0)}")
            print(f"Ground truths: {stats.get('ground_truths_count', 0)}")
            print(f"Mitigation strategies: {stats.get('mitigation_strategies_count', 0)}")
            
            # Step 3: Create initial batch
            print(f"\nStep 3: Creating initial batch of {self.batch_size} experiments...")
            experiments_created = self.create_sampled_experiments()
            
            # Step 4: Show final statistics
            print("\nStep 4: Final statistics...")
            final_stats = self.get_sampling_statistics()
            print(f"Total experiments: {final_stats.get('existing_experiments', 0):,}")
            print(f"Coverage: {final_stats.get('coverage_percentage', 0):.2f}%")
            
            print("\nGeography distribution:")
            for geo, count in final_stats.get('geography_stats', {}).items():
                print(f"  {geo}: {count:,}")
            
            print(f"\n✅ Setup complete! Created {experiments_created} new experiments.")
            print(f"Run this script again to create more batches as needed.")
            
            return True
            
        except Exception as e:
            print(f"Error during setup: {e}")
            return False

    def close(self):
        """Close the database session"""
        self.session.close()


def main():
    """Main function to run the sampling setup"""
    sampler = GeographicPersonaSampler()
    
    try:
        success = sampler.run_sampling_setup()
        return success
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        sampler.close()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
