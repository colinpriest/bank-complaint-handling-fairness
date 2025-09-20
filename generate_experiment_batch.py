#!/usr/bin/env python3
"""
Generate Experiment Batch Script

This script generates additional batches of randomly sampled experiments
to increase coverage of the experimental design over time.

Usage:
    python generate_experiment_batch.py [batch_size]
    
Example:
    python generate_experiment_batch.py 500
"""

import os
import sys
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from add_geographic_persona_sampling import GeographicPersonaSampler

def main():
    """Main function to generate experiment batches"""
    parser = argparse.ArgumentParser(description='Generate additional experiment batches')
    parser.add_argument('batch_size', type=int, nargs='?', default=1000,
                       help='Number of experiments to create in this batch (default: 1000)')
    parser.add_argument('--max-batches', type=int, default=1,
                       help='Maximum number of batches to create (default: 1)')
    
    args = parser.parse_args()
    
    print(f"Generating {args.max_batches} batch(es) of {args.batch_size} experiments each...")
    print("=" * 60)
    
    sampler = GeographicPersonaSampler()
    
    try:
        # Show initial statistics
        print("Initial statistics:")
        stats = sampler.get_sampling_statistics()
        print(f"  Total experiments: {stats.get('existing_experiments', 0):,}")
        print(f"  Coverage: {stats.get('coverage_percentage', 0):.2f}%")
        print()
        
        total_created = 0
        
        for batch_num in range(1, args.max_batches + 1):
            print(f"Creating batch {batch_num}/{args.max_batches}...")
            created = sampler.create_sampled_experiments(args.batch_size)
            total_created += created
            
            if created < args.batch_size:
                print(f"Only created {created} experiments (less than requested {args.batch_size})")
                print("This may indicate we're approaching full coverage or hitting duplicates.")
                break
        
        # Show final statistics
        print(f"\nFinal statistics:")
        final_stats = sampler.get_sampling_statistics()
        print(f"  Total experiments: {final_stats.get('existing_experiments', 0):,}")
        print(f"  Coverage: {final_stats.get('coverage_percentage', 0):.2f}%")
        print(f"  Created in this run: {total_created:,}")
        
        print(f"\n✅ Successfully created {total_created} new experiments!")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        sampler.close()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
