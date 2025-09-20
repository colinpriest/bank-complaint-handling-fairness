#!/usr/bin/env python3
"""
Script to add new geographic persona options to existing experiments

This script expands the geographic persona categories from 3 simple categories 
(urban_affluent, urban_poor, rural) to 9 comprehensive categories that capture 
both geographic location and socioeconomic status.

Current: 3 geographies × 4 ethnicities × 2 genders = 24 personas
New: 9 geographies × 4 ethnicities × 2 genders = 72 personas

The script will:
1. Add new personas to the personas table
2. Create new experiment records for the new personas
3. Preserve all existing experiment data
4. Only run LLM calls for the new experiments (not existing ones)
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database_check import Persona, Experiment, GroundTruth, MitigationStrategy

class GeographicPersonaExpander:
    """Handles the expansion of geographic persona options"""
    
    def __init__(self):
        """Initialize the expander with PostgreSQL database connection"""
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
        
        # Mapping from old to new geographic categories
        self.geography_mapping = {
            'urban_affluent': ['urban_upper_middle', 'urban_working'],
            'urban_poor': ['urban_poor'],  # Keep as is
            'rural': ['rural_upper_middle', 'rural_working', 'rural_poor']
        }
        
        # New geographic categories to add
        self.new_geographies = [
            'urban_upper_middle', 'urban_working', 'urban_poor',
            'suburban_upper_middle', 'suburban_working', 'suburban_poor',
            'rural_upper_middle', 'rural_working', 'rural_poor'
        ]
        
        # Geographic characteristics for new personas
        self.geography_characteristics = {
            'urban_upper_middle': {
                'locations': [('Manhattan, NY', '10013'), ('San Francisco, CA', '94105'), ('Boston, MA', '02101'), ('Seattle, WA', '98101')],
                'companies': ['JPMorgan Chase', 'Goldman Sachs', 'Morgan Stanley', 'Wells Fargo'],
                'products': ['Investment account', 'Private banking', 'Wealth management', 'Premium credit cards'],
                'language_style': 'formal',
                'typical_activities': ['a charity gala', 'a board meeting', 'a wine tasting', 'a professional conference'],
                'geography_hints': ['getting out of a taxi', 'after valet parked my car', 'in my office building', 'at the country club'],
                'typical_occupations': ['software engineer', 'investment banker', 'physician', 'attorney', 'marketing director']
            },
            'urban_working': {
                'locations': [('Brooklyn, NY', '11201'), ('Oakland, CA', '94607'), ('Cambridge, MA', '02139'), ('Tacoma, WA', '98402')],
                'companies': ['Bank of America', 'Chase', 'Citibank', 'PNC Bank'],
                'products': ['Checking account', 'Savings account', 'Auto loan', 'Credit card'],
                'language_style': 'standard',
                'typical_activities': ['a union meeting', 'a community event', 'a local restaurant', 'a neighborhood gathering'],
                'geography_hints': ['on the subway', 'at the local bank branch', 'in my neighborhood', 'at the community center'],
                'typical_occupations': ['teacher', 'nurse', 'electrician', 'mechanic', 'retail manager']
            },
            'urban_poor': {
                'locations': [('Bronx, NY', '10451'), ('Oakland, CA', '94601'), ('Roxbury, MA', '02119'), ('Tacoma, WA', '98404')],
                'companies': ['Check Into Cash', 'Western Union', 'MoneyGram', 'ACE Cash Express'],
                'products': ['Payday loan', 'Money transfer', 'Prepaid card', 'Check cashing'],
                'language_style': 'informal',
                'typical_activities': ['a temp agency', 'a food pantry', 'a community clinic', 'a job fair'],
                'geography_hints': ['on the bus', 'at the public library', 'in the housing project', 'at the corner store'],
                'typical_occupations': ['cashier', 'janitor', 'food service worker', 'security guard', 'day laborer']
            },
            'suburban_upper_middle': {
                'locations': [('Greenwich, CT', '06830'), ('Palo Alto, CA', '94301'), ('Newton, MA', '02458'), ('Bellevue, WA', '98004')],
                'companies': ['Chase Private Client', 'Wells Fargo Advisors', 'Merrill Lynch', 'UBS'],
                'products': ['Investment portfolio', 'Mortgage', '401k', 'Business banking'],
                'language_style': 'formal',
                'typical_activities': ['a golf tournament', 'a school board meeting', 'a country club dinner', 'a charity auction'],
                'geography_hints': ['in my home office', 'at the country club', 'driving my SUV', 'at the private school'],
                'typical_occupations': ['executive', 'doctor', 'lawyer', 'engineer', 'business owner']
            },
            'suburban_working': {
                'locations': [('Stamford, CT', '06901'), ('Fremont, CA', '94538'), ('Waltham, MA', '02451'), ('Renton, WA', '98055')],
                'companies': ['Bank of America', 'Chase', 'Wells Fargo', 'TD Bank'],
                'products': ['Checking account', 'Savings account', 'Personal loan', 'Credit card'],
                'language_style': 'standard',
                'typical_activities': ['a PTA meeting', 'a local sports game', 'a community festival', 'a neighborhood block party'],
                'geography_hints': ['in my driveway', 'at the local shopping center', 'driving to work', 'at the community pool'],
                'typical_occupations': ['teacher', 'nurse', 'police officer', 'firefighter', 'office manager']
            },
            'suburban_poor': {
                'locations': [('Bridgeport, CT', '06604'), ('Hayward, CA', '94541'), ('Lynn, MA', '01901'), ('Kent, WA', '98032')],
                'companies': ['Check Into Cash', 'Western Union', 'Advance America', 'Check City'],
                'products': ['Payday loan', 'Money transfer', 'Prepaid card', 'Check cashing'],
                'language_style': 'informal',
                'typical_activities': ['a food bank', 'a community health clinic', 'a job training program', 'a local church'],
                'geography_hints': ['on the bus', 'at the strip mall', 'in the apartment complex', 'at the community center'],
                'typical_occupations': ['retail worker', 'fast food worker', 'custodian', 'security guard', 'temp worker']
            },
            'rural_upper_middle': {
                'locations': [('Aspen, CO', '81611'), ('Napa, CA', '94558'), ('Jackson, WY', '83001'), ('Sedona, AZ', '86336')],
                'companies': ['First National Bank', 'Community Bank', 'Farm Credit', 'Local Credit Union'],
                'products': ['Investment account', 'Business loan', 'Farm loan', 'Wealth management'],
                'language_style': 'formal',
                'typical_activities': ['a wine tasting', 'a business meeting', 'a charity event', 'a cultural festival'],
                'geography_hints': ['at my ranch', 'in my home office', 'at the country club', 'driving my truck'],
                'typical_occupations': ['rancher', 'winery owner', 'business owner', 'doctor', 'lawyer']
            },
            'rural_working': {
                'locations': [('Bozeman, MT', '59715'), ('Fargo, ND', '58102'), ('Burlington, VT', '05401'), ('Ames, IA', '50010')],
                'companies': ['Regions Bank', 'Wells Fargo', 'Bank of America', 'Local Credit Union'],
                'products': ['Checking account', 'Savings account', 'Auto loan', 'Personal loan'],
                'language_style': 'colloquial',
                'typical_activities': ['a county fair', 'a church service', 'a local diner', 'a community meeting'],
                'geography_hints': ['on the long drive into town', 'at the feed store', 'in my pickup truck', 'at the local bank'],
                'typical_occupations': ['farmer', 'truck driver', 'mechanic', 'teacher', 'nurse']
            },
            'rural_poor': {
                'locations': [('Huntsville, AL', '35801'), ('Jackson, MS', '39201'), ('Fresno, CA', '93701'), ('Corpus Christi, TX', '78401')],
                'companies': ['Check Into Cash', 'Western Union', 'Advance America', 'Local Pawn Shop'],
                'products': ['Payday loan', 'Money transfer', 'Prepaid card', 'Check cashing'],
                'language_style': 'colloquial',
                'typical_activities': ['a food pantry', 'a community clinic', 'a job training program', 'a local church'],
                'geography_hints': ['on the long drive into town', 'at the general store', 'in my old car', 'at the community center'],
                'typical_occupations': ['farm worker', 'day laborer', 'retail worker', 'custodian', 'temp worker']
            }
        }

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

    def get_existing_experiments(self) -> List[Dict]:
        """Get all existing experiments from the database"""
        try:
            experiments = self.session.query(Experiment).all()
            return [
                {
                    'experiment_id': exp.experiment_id,
                    'case_id': exp.case_id,
                    'decision_method': exp.decision_method,
                    'llm_model': exp.llm_model,
                    'persona': exp.persona,
                    'gender': exp.gender,
                    'ethnicity': exp.ethnicity,
                    'geography': exp.geography,
                    'risk_mitigation_strategy': exp.risk_mitigation_strategy,
                    'system_prompt': exp.system_prompt,
                    'user_prompt': exp.user_prompt,
                    'system_response': exp.system_response,
                    'cache_id': exp.cache_id
                }
                for exp in experiments
            ]
        except Exception as e:
            print(f"Error getting existing experiments: {e}")
            return []

    def get_ground_truth_cases(self) -> List[Dict]:
        """Get all ground truth cases"""
        try:
            cases = self.session.query(GroundTruth).all()
            return [
                {
                    'case_id': case.case_id,
                    'consumer_complaint_text': case.consumer_complaint_text,
                    'simplified_ground_truth_tier': case.simplified_ground_truth_tier
                }
                for case in cases
            ]
        except Exception as e:
            print(f"Error getting ground truth cases: {e}")
            return []

    def get_mitigation_strategies(self) -> List[Dict]:
        """Get all mitigation strategies"""
        try:
            strategies = self.session.query(MitigationStrategy).all()
            return [
                {
                    'id': strategy.id,
                    'key': strategy.key,
                    'name': strategy.name,
                    'description': strategy.description,
                    'prompt_modification': strategy.prompt_modification
                }
                for strategy in strategies
            ]
        except Exception as e:
            print(f"Error getting mitigation strategies: {e}")
            return []

    def create_new_personas(self) -> int:
        """Create new personas for the expanded geographic categories"""
        print("Creating new personas for expanded geographic categories...")
        
        existing_personas = self.get_existing_personas()
        existing_keys = {p['key'] for p in existing_personas}
        
        new_personas_created = 0
        
        # Get all unique ethnicity-gender combinations from existing personas
        ethnicity_gender_combos = set()
        for persona in existing_personas:
            ethnicity_gender_combos.add((persona['ethnicity'], persona['gender']))
        
        for ethnicity, gender in ethnicity_gender_combos:
            for geography in self.new_geographies:
                new_key = f"{ethnicity}_{gender}_{geography}"
                
                if new_key in existing_keys:
                    continue  # Skip if already exists
                
                # Get characteristics for this geography
                geo_chars = self.geography_characteristics[geography]
                
                # Create new persona
                new_persona = Persona(
                    key=new_key,
                    ethnicity=ethnicity,
                    gender=gender,
                    geography=geography,
                    language_style=geo_chars['language_style'],
                    typical_names=json.dumps(geo_chars.get('typical_names', [])),
                    typical_locations=json.dumps([f"{loc[0]}, {loc[1]}" for loc in geo_chars['locations']]),
                    typical_companies=json.dumps(geo_chars['companies']),
                    typical_products=json.dumps(geo_chars['products']),
                    typical_activities=json.dumps(geo_chars['typical_activities']),
                    geography_hints=json.dumps(geo_chars['geography_hints']),
                    typical_occupations=json.dumps(geo_chars['typical_occupations']),
                    # Copy other fields from existing persona of same ethnicity/gender
                    gender_hints=json.dumps([]),  # Will be populated from existing
                    ethnicity_hints=json.dumps([])  # Will be populated from existing
                )
                
                # Copy gender and ethnicity hints from existing persona
                for existing_persona in existing_personas:
                    if existing_persona['ethnicity'] == ethnicity and existing_persona['gender'] == gender:
                        new_persona.gender_hints = json.dumps(existing_persona['gender_hints'])
                        new_persona.ethnicity_hints = json.dumps(existing_persona['ethnicity_hints'])
                        break
                
                self.session.add(new_persona)
                new_personas_created += 1
        
        try:
            self.session.commit()
            print(f"Successfully created {new_personas_created} new personas")
            return new_personas_created
        except Exception as e:
            self.session.rollback()
            print(f"Error creating new personas: {e}")
            return 0

    def create_new_experiments(self) -> int:
        """Create new experiments for the new personas"""
        print("Creating new experiments for new personas...")
        
        existing_experiments = self.get_existing_experiments()
        existing_experiment_keys = set()
        
        # Create a key for each existing experiment to avoid duplicates
        for exp in existing_experiments:
            key = f"{exp['case_id']}_{exp['decision_method']}_{exp['persona']}_{exp['risk_mitigation_strategy']}"
            existing_experiment_keys.add(key)
        
        ground_truth_cases = self.get_ground_truth_cases()
        mitigation_strategies = self.get_mitigation_strategies()
        new_personas = self.get_existing_personas()
        
        # Filter to only new personas (those with new geographies)
        new_personas = [p for p in new_personas if p['geography'] in self.new_geographies]
        
        new_experiments_created = 0
        
        for case in ground_truth_cases:
            for decision_method in ['zero-shot', 'n-shot']:
                for persona in new_personas:
                    # Create baseline experiment (no mitigation)
                    exp_key = f"{case['case_id']}_{decision_method}_{persona['key']}_None"
                    if exp_key not in existing_experiment_keys:
                        new_exp = Experiment(
                            case_id=case['case_id'],
                            decision_method=decision_method,
                            llm_model='gpt-4o-mini',
                            llm_simplified_tier=-999,  # Will be filled by LLM call
                            persona=persona['key'],
                            gender=persona['gender'],
                            ethnicity=persona['ethnicity'],
                            geography=persona['geography'],
                            risk_mitigation_strategy=None,
                            system_prompt="",  # Will be filled by LLM call
                            user_prompt="",    # Will be filled by LLM call
                            system_response="", # Will be filled by LLM call
                            cache_id=None
                        )
                        self.session.add(new_exp)
                        new_experiments_created += 1
                    
                    # Create experiments with mitigation strategies
                    for strategy in mitigation_strategies:
                        if strategy['key'].startswith('dpp'):  # Skip DPP strategies
                            continue
                            
                        exp_key = f"{case['case_id']}_{decision_method}_{persona['key']}_{strategy['key']}"
                        if exp_key not in existing_experiment_keys:
                            new_exp = Experiment(
                                case_id=case['case_id'],
                                decision_method=decision_method,
                                llm_model='gpt-4o-mini',
                                llm_simplified_tier=-999,  # Will be filled by LLM call
                                persona=persona['key'],
                                gender=persona['gender'],
                                ethnicity=persona['ethnicity'],
                                geography=persona['geography'],
                                risk_mitigation_strategy=strategy['key'],
                                system_prompt="",  # Will be filled by LLM call
                                user_prompt="",    # Will be filled by LLM call
                                system_response="", # Will be filled by LLM call
                                cache_id=None
                            )
                            self.session.add(new_exp)
                            new_experiments_created += 1
        
        try:
            self.session.commit()
            print(f"Successfully created {new_experiments_created} new experiments")
            return new_experiments_created
        except Exception as e:
            self.session.rollback()
            print(f"Error creating new experiments: {e}")
            return 0

    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the expansion"""
        existing_personas = self.get_existing_personas()
        existing_experiments = self.get_existing_experiments()
        
        # Count personas by geography
        persona_counts = {}
        for persona in existing_personas:
            geo = persona['geography']
            persona_counts[geo] = persona_counts.get(geo, 0) + 1
        
        # Count experiments by geography
        experiment_counts = {}
        for exp in existing_experiments:
            if exp['geography']:
                geo = exp['geography']
                experiment_counts[geo] = experiment_counts.get(geo, 0) + 1
        
        return {
            'total_personas': len(existing_personas),
            'total_experiments': len(existing_experiments),
            'persona_counts_by_geography': persona_counts,
            'experiment_counts_by_geography': experiment_counts
        }

    def run_expansion(self) -> bool:
        """Run the complete geographic persona expansion"""
        print("Starting geographic persona expansion...")
        print("=" * 60)
        
        # Get initial statistics
        initial_stats = self.get_statistics()
        print("Initial Statistics:")
        print(f"  Total personas: {initial_stats['total_personas']}")
        print(f"  Total experiments: {initial_stats['total_experiments']}")
        print(f"  Personas by geography: {initial_stats['persona_counts_by_geography']}")
        print()
        
        # Create new personas
        new_personas_count = self.create_new_personas()
        if new_personas_count == 0:
            print("No new personas created. Expansion may already be complete.")
            return True
        
        # Create new experiments
        new_experiments_count = self.create_new_experiments()
        
        # Get final statistics
        final_stats = self.get_statistics()
        print()
        print("Final Statistics:")
        print(f"  Total personas: {final_stats['total_personas']} (+{final_stats['total_personas'] - initial_stats['total_personas']})")
        print(f"  Total experiments: {final_stats['total_experiments']} (+{final_stats['total_experiments'] - initial_stats['total_experiments']})")
        print(f"  Personas by geography: {final_stats['persona_counts_by_geography']}")
        print()
        
        print("=" * 60)
        print("Geographic persona expansion completed successfully!")
        print(f"Created {new_personas_count} new personas and {new_experiments_count} new experiments")
        print()
        print("Next steps:")
        print("1. Run your LLM analysis on the new experiments (those with llm_simplified_tier = -999)")
        print("2. The new experiments will be automatically included in your fairness analysis")
        print("3. You can now analyze bias across 9 geographic categories instead of 3")
        
        return True

    def close(self):
        """Close the database session"""
        self.session.close()


def main():
    """Main function to run the geographic persona expansion"""
    print("Geographic Persona Expansion Script")
    print("=" * 60)
    print("This script will expand your geographic persona categories from 3 to 9 categories.")
    print("Current: urban_affluent, urban_poor, rural")
    print("New: urban_upper_middle, urban_working, urban_poor, suburban_upper_middle, suburban_working, suburban_poor, rural_upper_middle, rural_working, rural_poor")
    print()
    
    # Create expander and run expansion
    expander = GeographicPersonaExpander()
    
    try:
        success = expander.run_expansion()
        return success
    except Exception as e:
        print(f"Error during expansion: {e}")
        return False
    finally:
        expander.close()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
