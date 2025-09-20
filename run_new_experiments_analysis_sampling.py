#!/usr/bin/env python3
"""
Run New Experiments Analysis - Sampling Version

This script runs LLM analysis on new geographic persona experiments
that were created using the sampling approach.

It processes experiments with llm_simplified_tier = -999 (pending analysis).
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database_check import Experiment

class NewExperimentsAnalyzer:
    """Handles LLM analysis for new geographic persona experiments"""
    
    def __init__(self):
        """Initialize the analyzer with PostgreSQL database connection"""
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
        
        # Import LLM analysis components
        try:
            # Import from the correct module name (with hyphens converted to underscores)
            import importlib.util
            spec = importlib.util.spec_from_file_location("bank_complaint_handling", "bank-complaint-handling.py")
            bank_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(bank_module)
            self.analyzer = bank_module.BankComplaintFairnessAnalyzer()
        except Exception as e:
            print(f"Warning: Could not import BankComplaintFairnessAnalyzer: {e}")
            self.analyzer = None

    def get_pending_experiments(self, limit: int = 100) -> List[Dict]:
        """Get experiments that need LLM analysis"""
        try:
            experiments = self.session.query(Experiment).filter(
                Experiment.llm_simplified_tier == -999
            ).limit(limit).all()
            
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
                    'system_response': exp.system_response
                }
                for exp in experiments
            ]
        except Exception as e:
            print(f"Error getting pending experiments: {e}")
            return []

    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get statistics about pending analysis"""
        try:
            total_experiments = self.session.query(Experiment).count()
            pending_experiments = self.session.query(Experiment).filter(
                Experiment.llm_simplified_tier == -999
            ).count()
            completed_experiments = total_experiments - pending_experiments
            
            # Get breakdown by geography
            geography_stats = {}
            for exp in self.session.query(Experiment).filter(Experiment.llm_simplified_tier == -999).all():
                geo = exp.geography or 'none'
                geography_stats[geo] = geography_stats.get(geo, 0) + 1
            
            return {
                'total_experiments': total_experiments,
                'pending_experiments': pending_experiments,
                'completed_experiments': completed_experiments,
                'completion_percentage': (completed_experiments / total_experiments * 100) if total_experiments > 0 else 0,
                'geography_stats': geography_stats
            }
        except Exception as e:
            print(f"Error getting analysis statistics: {e}")
            return {}

    def run_analysis_batch(self, batch_size: int = 50) -> int:
        """Run LLM analysis on a batch of pending experiments"""
        if not self.analyzer:
            print("Error: BankComplaintFairnessAnalyzer not available")
            return 0
        
        print(f"Running analysis on batch of {batch_size} experiments...")
        
        pending_experiments = self.get_pending_experiments(batch_size)
        if not pending_experiments:
            print("No pending experiments found")
            return 0
        
        print(f"Found {len(pending_experiments)} experiments to process")
        
        # Use the existing analyzer's run_all_experiments method
        # This will handle all the complex LLM analysis logic
        try:
            # The analyzer expects to connect to its own database
            # We need to make sure it's configured properly
            if hasattr(self.analyzer, 'connect_to_database'):
                if not self.analyzer.connect_to_database():
                    print("Error: Could not connect analyzer to database")
                    return 0
            
            # Run the analysis using the existing method
            # This will process all pending experiments (llm_simplified_tier = -999)
            success = self.analyzer.run_all_experiments(max_workers=1)  # Use single worker for safety
            
            if success:
                # Count how many experiments were actually processed
                remaining_pending = self.session.query(Experiment).filter(
                    Experiment.llm_simplified_tier == -999
                ).count()
                
                # Calculate how many were processed
                initial_pending = len(pending_experiments)
                processed_count = initial_pending - remaining_pending
                
                print(f"Successfully processed {processed_count} experiments")
                return processed_count
            else:
                print("Analysis failed")
                return 0
                
        except Exception as e:
            print(f"Error running analysis: {e}")
            return 0

    def run_analysis(self, max_batches: int = 10, batch_size: int = 50) -> bool:
        """Run LLM analysis on pending experiments"""
        print("New Geographic Persona Experiments Analysis")
        print("=" * 60)
        print("This script will run LLM analysis on new geographic persona experiments.")
        print("Only experiments with llm_simplified_tier = -999 will be processed.")
        print()
        
        try:
            # Get initial statistics
            stats = self.get_analysis_statistics()
            print("Current Analysis Status:")
            print(f"  Total experiments: {stats.get('total_experiments', 0):,}")
            print(f"  Completed: {stats.get('completed_experiments', 0):,}")
            print(f"  Pending: {stats.get('pending_experiments', 0):,}")
            print(f"  Completion: {stats.get('completion_percentage', 0):.1f}%")
            print()
            
            if stats.get('pending_experiments', 0) == 0:
                print("No pending experiments found. All experiments are complete!")
                return True
            
            print("Pending experiments by geography:")
            for geo, count in stats.get('geography_stats', {}).items():
                print(f"  {geo}: {count:,}")
            print()
            
            # Run analysis batches
            total_processed = 0
            for batch_num in range(1, max_batches + 1):
                print(f"Running batch {batch_num}/{max_batches}...")
                processed = self.run_analysis_batch(batch_size)
                total_processed += processed
                
                if processed < batch_size:
                    print(f"Only processed {processed} experiments (less than requested {batch_size})")
                    print("This may indicate we're running out of pending experiments.")
                    break
            
            # Show final statistics
            print(f"\nFinal Analysis Status:")
            final_stats = self.get_analysis_statistics()
            print(f"  Total experiments: {final_stats.get('total_experiments', 0):,}")
            print(f"  Completed: {final_stats.get('completed_experiments', 0):,}")
            print(f"  Pending: {final_stats.get('pending_experiments', 0):,}")
            print(f"  Completion: {final_stats.get('completion_percentage', 0):.1f}%")
            print(f"  Processed in this run: {total_processed:,}")
            
            print(f"\n✅ Analysis complete! Processed {total_processed} experiments.")
            
            return True
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            return False

    def close(self):
        """Close the database session"""
        self.session.close()


def main():
    """Main function to run the new experiments analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run LLM analysis on new geographic persona experiments')
    parser.add_argument('--max-batches', type=int, default=10,
                       help='Maximum number of batches to process (default: 10)')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Number of experiments per batch (default: 50)')
    
    args = parser.parse_args()
    
    analyzer = NewExperimentsAnalyzer()
    
    try:
        success = analyzer.run_analysis(args.max_batches, args.batch_size)
        return success
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        analyzer.close()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
