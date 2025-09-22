#!/usr/bin/env python3
"""
Stochastic Experiment Creation Algorithm

This module implements the stochastic approach for creating experiments as described
in the experiment_sampling_specification.md document.

The algorithm follows this sequence:
1. Start with ground truth examples
2. Create baseline experiments (zero-shot and n-shot for each ground truth)
3. Create persona-injected experiments (stochastically sample 10 personas per baseline)
4. Create bias-mitigation experiments (for each persona-injected experiment)

This approach replaces the previous factorial combination approach with a more
efficient stochastic sampling method.
"""

import os
import sys
import random
from datetime import datetime
from typing import Dict, List, Any, Set, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database_check import Experiment, Persona, GroundTruth, MitigationStrategy

class StochasticExperimentCreator:
    """
    Creates experiments using the stochastic sampling approach.
    
    This class implements the algorithm described in experiment_sampling_specification.md:
    1. Ground truth examples → Baseline experiments → Persona-injected experiments → Bias-mitigation experiments
    """
    
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
        
        # Configuration
        self.personas_per_baseline = 10  # Number of personas to sample per baseline experiment
        self.llm_model = 'gpt-4o-mini'
        
    def get_ground_truth_examples(self, limit: int = None) -> List[Dict]:
        """
        Step 1: Get ground truth examples
        
        Args:
            limit: Maximum number of ground truth examples to retrieve
            
        Returns:
            List of ground truth examples with case_id, simplified_ground_truth_tier, etc.
        """
        try:
            query = self.session.query(GroundTruth).filter(
                GroundTruth.simplified_ground_truth_tier >= 0
            ).order_by(GroundTruth.case_id)
            
            if limit:
                query = query.limit(limit)
                
            ground_truths = query.all()
            
            return [
                {
                    'case_id': gt.case_id,
                    'simplified_ground_truth_tier': gt.simplified_ground_truth_tier,
                    'consumer_complaint_text': gt.consumer_complaint_text
                }
                for gt in ground_truths
            ]
        except Exception as e:
            print(f"Error getting ground truth examples: {e}")
            return []
    
    def get_all_personas(self) -> List[Dict]:
        """Get all available personas for stochastic sampling"""
        try:
            personas = self.session.query(Persona).all()
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
            print(f"Error getting personas: {e}")
            return []
    
    def get_mitigation_strategies(self) -> List[Dict]:
        """Get all bias mitigation strategies"""
        try:
            strategies = self.session.query(MitigationStrategy).filter(
                MitigationStrategy.key.notlike('%dpp%')  # Exclude DPP strategies
            ).all()
            return [
                {
                    'id': s.id,
                    'key': s.key,
                    'prompt_modification': s.prompt_modification
                }
                for s in strategies
            ]
        except Exception as e:
            print(f"Error getting mitigation strategies: {e}")
            return []
    
    def get_existing_experiments(self) -> Set[Tuple]:
        """Get existing experiments to avoid duplicates"""
        try:
            experiments = self.session.query(Experiment).all()
            existing = set()
            for exp in experiments:
                # Create a unique key for each experiment
                key = (
                    exp.case_id,
                    exp.decision_method,
                    exp.persona,
                    exp.risk_mitigation_strategy
                )
                existing.add(key)
            return existing
        except Exception as e:
            print(f"Error getting existing experiments: {e}")
            return set()
    
    def create_baseline_experiments(self, ground_truths: List[Dict], existing_experiments: Set[Tuple]) -> List[Dict]:
        """
        Step 2: Create baseline experiments
        
        For each ground truth example, create 2 baseline experiments:
        - One zero-shot experiment
        - One n-shot experiment
        
        Args:
            ground_truths: List of ground truth examples
            existing_experiments: Set of existing experiment keys to avoid duplicates
            
        Returns:
            List of created baseline experiments
        """
        print("Creating baseline experiments...")
        baseline_experiments = []
        decision_methods = ['zero-shot', 'n-shot']
        
        for ground_truth in ground_truths:
            for decision_method in decision_methods:
                # Check if this baseline experiment already exists
                experiment_key = (
                    ground_truth['case_id'],
                    decision_method,
                    None,  # No persona for baseline
                    None   # No mitigation strategy for baseline
                )
                
                if experiment_key in existing_experiments:
                    continue
                
                # Create baseline experiment
                baseline_experiment = Experiment(
                    case_id=ground_truth['case_id'],
                    decision_method=decision_method,
                    llm_model=self.llm_model,
                    llm_simplified_tier=-999,  # Mark as needing analysis
                    persona=None,  # No persona for baseline
                    gender=None,
                    ethnicity=None,
                    geography=None,
                    risk_mitigation_strategy=None,  # No mitigation for baseline
                    created_at=datetime.utcnow()
                )
                
                self.session.add(baseline_experiment)
                self.session.flush()  # Get the experiment_id
                
                baseline_experiments.append({
                    'experiment_id': baseline_experiment.experiment_id,
                    'case_id': ground_truth['case_id'],
                    'decision_method': decision_method,
                    'ground_truth_tier': ground_truth['simplified_ground_truth_tier']
                })
                
                existing_experiments.add(experiment_key)
        
        try:
            self.session.commit()
            print(f"Successfully created {len(baseline_experiments)} baseline experiments")
            return baseline_experiments
        except Exception as e:
            self.session.rollback()
            print(f"Error creating baseline experiments: {e}")
            return []
    
    def create_persona_injected_experiments(self, baseline_experiments: List[Dict], 
                                          personas: List[Dict], 
                                          existing_experiments: Set[Tuple]) -> List[Dict]:
        """
        Step 3: Create persona-injected experiments
        
        For each baseline experiment:
        1. Get its experiment number
        2. Set random seed to experiment number
        3. Randomly select 10 personas
        4. Create persona-injected experiments
        
        Args:
            baseline_experiments: List of baseline experiments
            personas: List of available personas
            existing_experiments: Set of existing experiment keys
            
        Returns:
            List of created persona-injected experiments
        """
        print("Creating persona-injected experiments...")
        persona_injected_experiments = []
        
        for baseline in baseline_experiments:
            experiment_id = baseline['experiment_id']
            case_id = baseline['case_id']
            decision_method = baseline['decision_method']
            
            # Set random seed to experiment number for reproducible sampling
            random.seed(experiment_id)
            
            # Randomly select 10 personas (or all if fewer than 10)
            selected_personas = random.sample(personas, min(self.personas_per_baseline, len(personas)))
            
            for persona in selected_personas:
                # Check if this persona-injected experiment already exists
                experiment_key = (
                    case_id,
                    decision_method,
                    persona['key'],
                    None  # No mitigation strategy for persona-injected
                )
                
                if experiment_key in existing_experiments:
                    continue
                
                # Create persona-injected experiment
                persona_experiment = Experiment(
                    case_id=case_id,
                    decision_method=decision_method,
                    llm_model=self.llm_model,
                    llm_simplified_tier=-999,  # Mark as needing analysis
                    persona=persona['key'],
                    gender=persona['gender'],
                    ethnicity=persona['ethnicity'],
                    geography=persona['geography'],
                    risk_mitigation_strategy=None,  # No mitigation for persona-injected
                    created_at=datetime.utcnow()
                )
                
                self.session.add(persona_experiment)
                self.session.flush()  # Get the experiment_id
                
                persona_injected_experiments.append({
                    'experiment_id': persona_experiment.experiment_id,
                    'case_id': case_id,
                    'decision_method': decision_method,
                    'persona_key': persona['key'],
                    'baseline_experiment_id': experiment_id
                })
                
                existing_experiments.add(experiment_key)
        
        try:
            self.session.commit()
            print(f"Successfully created {len(persona_injected_experiments)} persona-injected experiments")
            return persona_injected_experiments
        except Exception as e:
            self.session.rollback()
            print(f"Error creating persona-injected experiments: {e}")
            return []
    
    def create_bias_mitigation_experiments(self, persona_injected_experiments: List[Dict],
                                         mitigation_strategies: List[Dict],
                                         existing_experiments: Set[Tuple]) -> List[Dict]:
        """
        Step 4: Create bias-mitigation experiments
        
        For each persona-injected experiment:
        1. Get its experiment number
        2. Set the random seed to the experiment number
        3. Randomly select 3 bias strategies (from the several in the table)
        4. Create bias-mitigation experiments for each selected strategy
        
        Args:
            persona_injected_experiments: List of persona-injected experiments
            mitigation_strategies: List of mitigation strategies
            existing_experiments: Set of existing experiment keys
            
        Returns:
            List of created bias-mitigation experiments
        """
        print("Creating bias-mitigation experiments...")
        bias_mitigation_experiments = []
        
        for persona_exp in persona_injected_experiments:
            experiment_id = persona_exp['experiment_id']
            case_id = persona_exp['case_id']
            decision_method = persona_exp['decision_method']
            persona_key = persona_exp['persona_key']
            
            # Set random seed to experiment number for reproducibility
            random.seed(experiment_id)
            
            # Randomly select 3 bias strategies
            selected_strategies = random.sample(mitigation_strategies, min(3, len(mitigation_strategies)))
            
            for strategy in selected_strategies:
                # Check if this bias-mitigation experiment already exists
                experiment_key = (
                    case_id,
                    decision_method,
                    persona_key,
                    strategy['key']
                )
                
                if experiment_key in existing_experiments:
                    continue
                
                # Create bias-mitigation experiment
                mitigation_experiment = Experiment(
                    case_id=case_id,
                    decision_method=decision_method,
                    llm_model=self.llm_model,
                    llm_simplified_tier=-999,  # Mark as needing analysis
                    persona=persona_key,
                    gender=persona_exp.get('gender'),
                    ethnicity=persona_exp.get('ethnicity'),
                    geography=persona_exp.get('geography'),
                    risk_mitigation_strategy=strategy['key'],
                    created_at=datetime.utcnow()
                )
                
                self.session.add(mitigation_experiment)
                self.session.flush()  # Get the experiment_id
                
                bias_mitigation_experiments.append({
                    'experiment_id': mitigation_experiment.experiment_id,
                    'case_id': case_id,
                    'decision_method': decision_method,
                    'persona_key': persona_key,
                    'mitigation_strategy': strategy['key'],
                    'persona_experiment_id': experiment_id
                })
                
                existing_experiments.add(experiment_key)
        
        try:
            self.session.commit()
            print(f"Successfully created {len(bias_mitigation_experiments)} bias-mitigation experiments")
            return bias_mitigation_experiments
        except Exception as e:
            self.session.rollback()
            print(f"Error creating bias-mitigation experiments: {e}")
            return []
    
    def create_experiments_stochastic(self, ground_truth_limit: int = None) -> Dict[str, Any]:
        """
        Main method to create experiments using the stochastic approach
        
        Args:
            ground_truth_limit: Maximum number of ground truth examples to process
            
        Returns:
            Dictionary with creation statistics
        """
        print("Starting stochastic experiment creation...")
        print("=" * 60)
        
        # Step 1: Get ground truth examples
        print("Step 1: Getting ground truth examples...")
        ground_truths = self.get_ground_truth_examples(ground_truth_limit)
        if not ground_truths:
            print("No ground truth examples found!")
            return {'error': 'No ground truth examples found'}
        
        print(f"Found {len(ground_truths)} ground truth examples")
        
        # Get supporting data
        personas = self.get_all_personas()
        mitigation_strategies = self.get_mitigation_strategies()
        existing_experiments = self.get_existing_experiments()
        
        print(f"Found {len(personas)} personas")
        print(f"Found {len(mitigation_strategies)} mitigation strategies")
        print(f"Found {len(existing_experiments)} existing experiments")
        print()
        
        # Step 2: Create baseline experiments
        baseline_experiments = self.create_baseline_experiments(ground_truths, existing_experiments)
        
        # Step 3: Create persona-injected experiments
        persona_injected_experiments = self.create_persona_injected_experiments(
            baseline_experiments, personas, existing_experiments
        )
        
        # Step 4: Create bias-mitigation experiments
        bias_mitigation_experiments = self.create_bias_mitigation_experiments(
            persona_injected_experiments, mitigation_strategies, existing_experiments
        )
        
        # Calculate statistics
        total_created = len(baseline_experiments) + len(persona_injected_experiments) + len(bias_mitigation_experiments)
        
        statistics = {
            'ground_truth_examples': len(ground_truths),
            'baseline_experiments': len(baseline_experiments),
            'persona_injected_experiments': len(persona_injected_experiments),
            'bias_mitigation_experiments': len(bias_mitigation_experiments),
            'total_experiments_created': total_created,
            'personas_sampled_per_baseline': self.personas_per_baseline,
            'mitigation_strategies_used': len(mitigation_strategies)
        }
        
        print("\n" + "=" * 60)
        print("STOCHASTIC EXPERIMENT CREATION COMPLETE")
        print("=" * 60)
        print(f"Ground truth examples processed: {statistics['ground_truth_examples']}")
        print(f"Baseline experiments created: {statistics['baseline_experiments']}")
        print(f"Persona-injected experiments created: {statistics['persona_injected_experiments']}")
        print(f"Bias-mitigation experiments created: {statistics['bias_mitigation_experiments']}")
        print(f"Total experiments created: {statistics['total_experiments_created']}")
        print(f"Personas sampled per baseline: {statistics['personas_sampled_per_baseline']}")
        print(f"Mitigation strategies used: {statistics['mitigation_strategies_used']}")
        
        return statistics
    
    def get_experiment_statistics(self) -> Dict[str, Any]:
        """Get current experiment statistics from the database"""
        try:
            # Get counts by experiment type
            cursor = self.session.execute(text("""
                SELECT 
                    CASE 
                        WHEN persona IS NULL AND risk_mitigation_strategy IS NULL THEN 'baseline'
                        WHEN persona IS NOT NULL AND risk_mitigation_strategy IS NULL THEN 'persona_injected'
                        WHEN persona IS NOT NULL AND risk_mitigation_strategy IS NOT NULL THEN 'bias_mitigation'
                        ELSE 'other'
                    END as experiment_type,
                    decision_method,
                    COUNT(*) as count
                FROM experiments 
                GROUP BY experiment_type, decision_method
                ORDER BY experiment_type, decision_method
            """))
            
            stats = {
                'baseline': {'zero-shot': 0, 'n-shot': 0},
                'persona_injected': {'zero-shot': 0, 'n-shot': 0},
                'bias_mitigation': {'zero-shot': 0, 'n-shot': 0},
                'other': {'zero-shot': 0, 'n-shot': 0}
            }
            
            for row in cursor.fetchall():
                exp_type, method, count = row
                if exp_type in stats and method in stats[exp_type]:
                    stats[exp_type][method] = count
            
            # Calculate totals
            stats['total'] = sum(
                sum(methods.values()) for methods in stats.values() 
                if isinstance(methods, dict)
            )
            
            return stats
            
        except Exception as e:
            print(f"Error getting experiment statistics: {e}")
            return {}
    
    def close(self):
        """Close the database session"""
        self.session.close()

def main():
    """Main function to run stochastic experiment creation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create experiments using stochastic sampling approach')
    parser.add_argument('--ground-truth-limit', type=int, default=None,
                       help='Limit number of ground truth examples to process (default: all)')
    parser.add_argument('--show-stats', action='store_true',
                       help='Show current experiment statistics before creating new ones')
    
    args = parser.parse_args()
    
    creator = StochasticExperimentCreator()
    
    try:
        if args.show_stats:
            print("Current experiment statistics:")
            stats = creator.get_experiment_statistics()
            for exp_type, methods in stats.items():
                if isinstance(methods, dict):
                    print(f"  {exp_type}: {methods}")
                else:
                    print(f"  {exp_type}: {methods}")
            print()
        
        # Create experiments using stochastic approach
        results = creator.create_experiments_stochastic(args.ground_truth_limit)
        
        if 'error' in results:
            print(f"Error: {results['error']}")
            return False
        
        print(f"\n✅ Successfully created {results['total_experiments_created']} experiments using stochastic approach!")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        creator.close()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
