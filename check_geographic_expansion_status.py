#!/usr/bin/env python3
"""
Script to check the status of geographic persona expansion

This script will show you:
1. Current persona and experiment counts
2. Which geographic categories exist
3. How many experiments need LLM analysis
4. Progress towards the full 9-category expansion
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database_check import Persona, Experiment, GroundTruth, MitigationStrategy

class GeographicExpansionStatus:
    """Check the status of geographic persona expansion"""
    
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
        
        # Define the target geographic categories
        self.target_geographies = [
            'urban_upper_middle', 'urban_working', 'urban_poor',
            'suburban_upper_middle', 'suburban_working', 'suburban_poor',
            'rural_upper_middle', 'rural_working', 'rural_poor'
        ]
        
        # Define the original geographic categories
        self.original_geographies = ['urban_affluent', 'urban_poor', 'rural']

    def get_persona_statistics(self) -> Dict[str, Any]:
        """Get statistics about personas"""
        try:
            personas = self.session.query(Persona).all()
            
            # Count by geography
            geography_counts = {}
            ethnicity_gender_counts = {}
            
            for persona in personas:
                geo = persona.geography
                ethnicity_gender = f"{persona.ethnicity}_{persona.gender}"
                
                geography_counts[geo] = geography_counts.get(geo, 0) + 1
                ethnicity_gender_counts[ethnicity_gender] = ethnicity_gender_counts.get(ethnicity_gender, 0) + 1
            
            return {
                'total_personas': len(personas),
                'geography_counts': geography_counts,
                'ethnicity_gender_counts': ethnicity_gender_counts,
                'unique_geographies': list(geography_counts.keys()),
                'unique_ethnicity_genders': list(ethnicity_gender_counts.keys())
            }
        except Exception as e:
            print(f"Error getting persona statistics: {e}")
            return {}

    def get_experiment_statistics(self) -> Dict[str, Any]:
        """Get statistics about experiments"""
        try:
            experiments = self.session.query(Experiment).all()
            
            # Count by geography
            geography_counts = {}
            pending_by_geography = {}
            completed_by_geography = {}
            
            for exp in experiments:
                if exp.geography:
                    geo = exp.geography
                    geography_counts[geo] = geography_counts.get(geo, 0) + 1
                    
                    if exp.llm_simplified_tier == -999:
                        pending_by_geography[geo] = pending_by_geography.get(geo, 0) + 1
                    else:
                        completed_by_geography[geo] = completed_by_geography.get(geo, 0) + 1
            
            # Count pending vs completed
            total_pending = sum(pending_by_geography.values())
            total_completed = sum(completed_by_geography.values())
            
            return {
                'total_experiments': len(experiments),
                'total_pending': total_pending,
                'total_completed': total_completed,
                'completion_percentage': (total_completed / len(experiments) * 100) if experiments else 0,
                'geography_counts': geography_counts,
                'pending_by_geography': pending_by_geography,
                'completed_by_geography': completed_by_geography
            }
        except Exception as e:
            print(f"Error getting experiment statistics: {e}")
            return {}

    def check_expansion_status(self) -> Dict[str, Any]:
        """Check the overall expansion status"""
        persona_stats = self.get_persona_statistics()
        experiment_stats = self.get_experiment_statistics()
        
        current_geographies = set(persona_stats.get('unique_geographies', []))
        target_geographies = set(self.target_geographies)
        original_geographies = set(self.original_geographies)
        
        # Check what's missing
        missing_geographies = target_geographies - current_geographies
        new_geographies = current_geographies - original_geographies
        
        # Calculate expected counts
        expected_personas = len(persona_stats.get('unique_ethnicity_genders', [])) * len(target_geographies)
        expected_experiments = expected_personas * 2 * 8  # 2 methods × 8 strategies (including baseline)
        
        return {
            'current_geographies': sorted(current_geographies),
            'target_geographies': sorted(target_geographies),
            'missing_geographies': sorted(missing_geographies),
            'new_geographies': sorted(new_geographies),
            'expansion_complete': len(missing_geographies) == 0,
            'expected_personas': expected_personas,
            'expected_experiments': expected_experiments,
            'persona_stats': persona_stats,
            'experiment_stats': experiment_stats
        }

    def print_status_report(self):
        """Print a comprehensive status report"""
        status = self.check_expansion_status()
        
        print("Geographic Persona Expansion Status Report")
        print("=" * 60)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Overall status
        print("OVERALL STATUS:")
        if status['expansion_complete']:
            print("✓ Geographic expansion is COMPLETE")
        else:
            print("⚠ Geographic expansion is INCOMPLETE")
        print(f"  Current geographies: {len(status['current_geographies'])}")
        print(f"  Target geographies: {len(status['target_geographies'])}")
        print(f"  Missing geographies: {len(status['missing_geographies'])}")
        print()
        
        # Persona statistics
        persona_stats = status['persona_stats']
        print("PERSONA STATISTICS:")
        print(f"  Total personas: {persona_stats['total_personas']}")
        print(f"  Expected personas: {status['expected_personas']}")
        print(f"  Personas by geography:")
        for geo in sorted(persona_stats['geography_counts'].keys()):
            count = persona_stats['geography_counts'][geo]
            status_icon = "✓" if geo in status['target_geographies'] else "⚠"
            print(f"    {status_icon} {geo}: {count}")
        print()
        
        # Experiment statistics
        experiment_stats = status['experiment_stats']
        print("EXPERIMENT STATISTICS:")
        print(f"  Total experiments: {experiment_stats['total_experiments']}")
        print(f"  Expected experiments: {status['expected_experiments']}")
        print(f"  Completed: {experiment_stats['total_completed']}")
        print(f"  Pending: {experiment_stats['total_pending']}")
        print(f"  Completion: {experiment_stats['completion_percentage']:.1f}%")
        print()
        
        # Geography breakdown
        print("GEOGRAPHY BREAKDOWN:")
        for geo in sorted(experiment_stats['geography_counts'].keys()):
            total = experiment_stats['geography_counts'][geo]
            completed = experiment_stats['completed_by_geography'].get(geo, 0)
            pending = experiment_stats['pending_by_geography'].get(geo, 0)
            completion = (completed / total * 100) if total > 0 else 0
            
            status_icon = "✓" if geo in status['target_geographies'] else "⚠"
            print(f"    {status_icon} {geo}: {completed}/{total} completed ({completion:.1f}%)")
        print()
        
        # Missing geographies
        if status['missing_geographies']:
            print("MISSING GEOGRAPHIES:")
            for geo in status['missing_geographies']:
                print(f"  - {geo}")
            print()
        
        # Recommendations
        print("RECOMMENDATIONS:")
        if not status['expansion_complete']:
            print("1. Run 'python add_geographic_persona_options.py' to add missing personas and experiments")
        if experiment_stats['total_pending'] > 0:
            print("2. Run 'python run_new_experiments_analysis.py' to complete LLM analysis on pending experiments")
        if status['expansion_complete'] and experiment_stats['total_pending'] == 0:
            print("✓ All geographic persona expansion is complete!")
            print("  You can now run your full fairness analysis with 9 geographic categories.")
        print()

    def close(self):
        """Close the database session"""
        self.session.close()


def main():
    """Main function to check expansion status"""
    # Create status checker
    checker = GeographicExpansionStatus()
    
    try:
        checker.print_status_report()
        return True
    except Exception as e:
        print(f"Error checking status: {e}")
        return False
    finally:
        checker.close()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
