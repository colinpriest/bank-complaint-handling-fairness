#!/usr/bin/env python3
"""
GPT-4o-mini Fairness Analysis

This script runs fairness analysis using only the gpt-4o-mini model.
Results are saved to the gpt_4o_mini_results directory.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict

# Add the fairness_analysis package to the path
sys.path.insert(0, str(Path(__file__).parent))

from fairness_analysis import AdvancedFairnessAnalyzer
import shutil
import glob


def clear_cache(cache_dir: str = "data_cache"):
    """Clear all cache files"""
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"[CACHE] Cleared cache directory: {cache_dir}")
    else:
        print(f"[CACHE] Cache directory does not exist: {cache_dir}")


def show_cache_info(cache_dir: str = "data_cache"):
    """Show cache information"""
    if not os.path.exists(cache_dir):
        print(f"[CACHE] Cache directory does not exist: {cache_dir}")
        return
    
    cache_files = glob.glob(os.path.join(cache_dir, "**", "*.json"), recursive=True)
    total_size = 0
    
    print(f"[CACHE] Cache directory: {cache_dir}")
    print(f"[CACHE] Total cache files: {len(cache_files)}")
    
    # Group by provider
    providers = {}
    for file_path in cache_files:
        provider = os.path.basename(os.path.dirname(file_path))
        if provider not in providers:
            providers[provider] = []
        providers[provider].append(file_path)
        total_size += os.path.getsize(file_path)
    
    print(f"[CACHE] Total cache size: {total_size / (1024*1024):.2f} MB")
    print(f"[CACHE] Providers:")
    for provider, files in providers.items():
        provider_size = sum(os.path.getsize(f) for f in files)
        print(f"  {provider}: {len(files)} files, {provider_size / (1024*1024):.2f} MB")


def load_cfpb_cases_with_sampling_index(sample_size: int) -> List[Dict]:
    """
    Load CFPB cases using a sampling index system for reproducible case selection.
    
    Steps:
    1. Check if sampling index file exists
    2. If not, create it by:
       a) Reading whole CFPB data
       b) Filtering cases with known outcome tiers
       c) Jumbling case IDs with seed 42
       d) Storing in sampling index file
    3. Load cases for the requested sample_size from the index
    """
    import json
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    sampling_index_file = Path("cfpb_sampling_index.json")
    
    # Step 1: Check if sampling index file exists
    if not sampling_index_file.exists():
        print("[SAMPLING] Creating CFPB sampling index file...")
        create_sampling_index_file(sampling_index_file)
    
    # Step 2: Load sampling index
    with open(sampling_index_file, 'r') as f:
        sampling_data = json.load(f)
    
    jumbled_case_ids = sampling_data['jumbled_case_ids']
    total_available = len(jumbled_case_ids)
    
    print(f"[SAMPLING] Total available CFPB cases: {total_available}")
    print(f"[SAMPLING] Requesting {sample_size} cases")
    
    if sample_size > total_available:
        print(f"[ERROR] Requested {sample_size} cases but only {total_available} available")
        return []
    
    # Step 3: Get case IDs for the requested sample size
    selected_case_ids = jumbled_case_ids[:sample_size]
    print(f"[SAMPLING] Selected case IDs: {selected_case_ids[:5]}..." if len(selected_case_ids) > 5 else f"[SAMPLING] Selected case IDs: {selected_case_ids}")
    
    # Step 4: Load the actual CFPB data for these case IDs
    try:
        from fairness_analysis.data_loader import DataLoader
        data_loader = DataLoader()
        
        # Load all CFPB data (we need to load more to get the specific cases we want)
        cfpb_df = data_loader.load_expanded_cfpb_data(sample_size=total_available * 2)
        
        # Convert case IDs to integers for indexing
        selected_indices = [int(case_id) for case_id in selected_case_ids]
        
        # Filter to get only the selected cases
        selected_df = cfpb_df.iloc[selected_indices]
        
        # Convert to baseline record format
        baseline_records = []
        for idx, (original_idx, row) in enumerate(selected_df.iterrows()):
            baseline_record = {
                'case_id': f"cfpb_{selected_case_ids[idx]}",
                'narrative': row.get('Consumer complaint narrative', ''),
                'issue': row.get('Issue', ''),
                'sub_issue': row.get('Sub-issue', ''),
                'product': row.get('Product', ''),
                'sub_product': row.get('Sub-product', ''),
                'company_response': row.get('Company response to consumer', ''),
                'timely_response': row.get('Timely response?', ''),
                'consumer_disputed': row.get('Consumer disputed?', ''),
                'date_received': row.get('Date received', ''),
                'variant': 'NC',
                'group_label': 'baseline',
                'group_text': 'No demographic context'
            }
            baseline_records.append(baseline_record)
        
        print(f"[SAMPLING] Loaded {len(baseline_records)} CFPB cases as templates")
        return baseline_records
        
    except Exception as e:
        print(f"[ERROR] Failed to load CFPB data: {e}")
        return []


def create_sampling_index_file(sampling_index_file: Path):
    """Create the sampling index file with jumbled case IDs"""
    import json
    import pandas as pd
    import numpy as np
    
    print("[SAMPLING] Reading whole CFPB dataset...")
    
    try:
        from fairness_analysis.data_loader import DataLoader
        data_loader = DataLoader()
        
        # Load all CFPB data
        cfpb_df = data_loader.load_expanded_cfpb_data(sample_size=50000)  # Load a large number
        
        print(f"[SAMPLING] Loaded {len(cfpb_df)} total CFPB cases")
        
        # Filter out cases that don't have known outcome tiers
        # (This would need to be implemented based on your outcome tier logic)
        # For now, we'll assume all cases are valid
        valid_cases = cfpb_df.copy()
        
        print(f"[SAMPLING] Valid cases with known outcome tiers: {len(valid_cases)}")
        
        # Set RNG seed to 42
        rng = np.random.RandomState(42)
        
        # Get all case indices
        case_indices = list(range(len(valid_cases)))
        
        # Jumble the case IDs without replacement
        jumbled_case_ids = rng.choice(case_indices, size=len(case_indices), replace=False)
        jumbled_case_ids = [str(idx) for idx in jumbled_case_ids]
        
        print(f"[SAMPLING] Jumbled {len(jumbled_case_ids)} case IDs")
        
        # Store in sampling index file
        sampling_data = {
            'total_cases': len(valid_cases),
            'jumbled_case_ids': jumbled_case_ids,
            'created_timestamp': pd.Timestamp.now().isoformat(),
            'rng_seed': 42
        }
        
        with open(sampling_index_file, 'w') as f:
            json.dump(sampling_data, f, indent=2)
        
        print(f"[SAMPLING] Created sampling index file: {sampling_index_file}")
        print(f"[SAMPLING] First 10 jumbled case IDs: {jumbled_case_ids[:10]}")
        
    except Exception as e:
        print(f"[ERROR] Failed to create sampling index: {e}")
        raise


def generate_experiments_with_all_strategies(sample_size: int = 100, threads_per_model: int = 10):
    """Generate 200 complaints with all personas and evaluate them with LLM"""
    import json
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    # Set up random seed for reproducibility
    rng = np.random.RandomState(42)
    
    # Helper to convert numpy types for JSON serialization
    def convert_numpy(obj):
        """Recursively convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    print(f"[EXPERIMENT] Generating {sample_size} complaints with comprehensive persona coverage...")
    print(f"[EXPERIMENT] Target: 10 personas per complaint for comprehensive Complaint Categories analysis")
    
    # Always load fresh CFPB data for templates (don't reuse existing data)
    runs_file = Path("out/runs.jsonl")
    baseline_records = load_cfpb_cases_with_sampling_index(sample_size)
    
    print(f"[EXPERIMENT] Using {len(baseline_records)} baseline records as templates")
    
    # Get all available personas
    try:
        from complaints_llm_fairness_harness import DEMOGRAPHIC_PERSONAS, generate_realistic_narrative
        import uuid
        all_available_personas = list(DEMOGRAPHIC_PERSONAS.keys())
        print(f"[EXPERIMENT] Available personas: {len(all_available_personas)}")
    except ImportError:
        print("[ERROR] Cannot import DEMOGRAPHIC_PERSONAS")
        return
    
    # Generate sample_size complaints with 10 personas each
    print(f"[EXPERIMENT] Generating {sample_size} new complaints...")
    
    new_records = []
    for i in range(sample_size):
        # Use baseline records in order to ensure unique case_ids
        template = baseline_records[i]
        
        # Generate case_id based on the CFPB case ID from sampling index
        cfpb_case_id = template['case_id'].split('_')[1]  # Extract the CFPB case ID from template
        new_case_id = f"exp_{cfpb_case_id}"
        
        # Create baseline record
        baseline_record = template.copy()
        baseline_record['case_id'] = new_case_id
        baseline_record['group_label'] = 'baseline'
        baseline_record['group_text'] = 'No demographic context'
        baseline_record['variant'] = 'NC'
        new_records.append(baseline_record)
        
        # Select 10 random personas for this complaint (without replacement)
        selected_personas = rng.choice(all_available_personas, size=10, replace=False)
        
        # Create persona records
        for persona_key in selected_personas:
            persona = DEMOGRAPHIC_PERSONAS[persona_key]
            
            # Create persona-specific record
            name = rng.choice(persona["names"])
            location_data = persona["locations"][rng.randint(0, len(persona["locations"]) - 1)]
            location, zip_code = location_data
            company = rng.choice(persona["companies"])
            product = rng.choice(persona["products"])
            style = persona["language_style"]
            
            # Generate realistic narrative
            base_narrative = template.get("narrative", "")
            persona_narrative = generate_realistic_narrative(
                base_narrative, style, name, location, product
            )
            
            # Create multiple records for this persona:
            # 1. Standard persona record (G variant) - for ALL personas
            # 2. One randomly selected fairness strategy variant - for ALL personas
            
            # Standard persona record (G variant) - always generated
            new_record = template.copy()
            new_record["case_id"] = new_case_id
            new_record["group_label"] = persona_key
            new_record["group_text"] = f"{name} from {location}"
            new_record["variant"] = "G"  # Standard persona variant
            new_record["product"] = product
            new_record["company"] = company
            new_record["state"] = location.split(", ")[1] if ", " in location else "CA"
            new_record["narrative"] = persona_narrative
            new_records.append(new_record)
            
            # One randomly selected fairness strategy variant (including persona_fairness)
            fairness_strategies = [
                "persona_fairness",
                "structured_extraction",
                "roleplay", 
                "consequentialist",
                "perspective",
                "minimal",
                "chain_of_thought"
            ]
            
            # Randomly select one fairness strategy for this persona
            selected_strategy = rng.choice(fairness_strategies)
            strategy_record = template.copy()
            strategy_record["case_id"] = new_case_id
            strategy_record["group_label"] = persona_key
            strategy_record["group_text"] = f"{name} from {location}"
            strategy_record["variant"] = selected_strategy
            strategy_record["product"] = product
            strategy_record["company"] = company
            strategy_record["state"] = location.split(", ")[1] if ", " in location else "CA"
            strategy_record["narrative"] = persona_narrative
            new_records.append(strategy_record)
        
        if (i + 1) % 10 == 0:
            print(f"[EXPERIMENT] Generated {i + 1}/{sample_size} complaints")
    
    print(f"[EXPERIMENT] Generated {len(new_records)} total records ({sample_size} complaints × 31 records each: 1 baseline + 10 personas × 3 variants each)")
    
    if new_records:
        # Ensure the output directory exists
        runs_file.parent.mkdir(parents=True, exist_ok=True)
        
        # First, append new records to runs.jsonl (without remedy_tier)
        print("[EXPERIMENT] Appending new records to out/runs.jsonl...")
        with open(runs_file, 'a') as f:
            for record in new_records:
                safe_record = convert_numpy(record)
                f.write(json.dumps(safe_record, ensure_ascii=False) + '\n')
        
        print(f"[EXPERIMENT] Successfully added {len(new_records)} records")
        
        # Now evaluate the new records with LLM (only if not in cache)
        print("[EXPERIMENT] Checking cache and evaluating new records with LLM...")
        from complaints_llm_fairness_harness import run_dialog, LLMClient, PairRecord
        
        # Initialize LLM client
        client = LLMClient(model_id="gpt-4o-mini", provider="openai")
        
        # Convert new records to PairRecord format and evaluate with multithreading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        evaluated_records = []
        lock = threading.Lock()
        
        # Cache usage tracking
        cache_stats = {
            'total_processed': 0,
            'cache_hits': 0,
            'llm_calls': 0,
            'errors': 0
        }
        
        def process_record(record_data):
            i, record = record_data
            # Process ALL records (both baseline and persona records)
            with lock:
                print(f"[EXPERIMENT] Processing record {i+1}/{len(new_records)}: {record.get('case_id')} - {record.get('group_label')}")
            
            # Convert to PairRecord format
            pr = PairRecord(
                pair_id=str(uuid.uuid4()),
                case_id=record['case_id'],
                group_label=record['group_label'],
                variant=record['variant'],
                group_text=record['group_text'],
                narrative=record['narrative'],
                product=record.get('product', ''),
                company=record.get('company', ''),
                state=record.get('state', ''),
                issue=record.get('issue', ''),
                year=record.get('year', '2023')
            )
            
            # Check if this record already exists in cache with retry logic
            max_retries = 2
            retry_count = 0
            
            while retry_count <= max_retries:
                try:
                    # Get cache stats before the call
                    cache_hits_before = client.cache_hits
                    api_calls_before = client.api_calls
                    
                    # The run_dialog function will check cache automatically
                    result = run_dialog(client, pr, i)
                    
                    # Check if this was a cache hit or new LLM call
                    cache_hits_after = client.cache_hits
                    api_calls_after = client.api_calls
                    
                    was_cache_hit = cache_hits_after > cache_hits_before
                    was_llm_call = api_calls_after > api_calls_before
                    
                    # Update cache stats
                    with lock:
                        cache_stats['total_processed'] += 1
                        if was_cache_hit:
                            cache_stats['cache_hits'] += 1
                            cache_source = "CACHE"
                        elif was_llm_call:
                            cache_stats['llm_calls'] += 1
                            cache_source = "LLM"
                        else:
                            cache_source = "UNKNOWN"
                        
                        # Print progress with cache info
                        hit_rate = (cache_stats['cache_hits'] / cache_stats['total_processed'] * 100) if cache_stats['total_processed'] > 0 else 0
                        retry_info = f" (retry {retry_count})" if retry_count > 0 else ""
                        print(f"[EXPERIMENT] Record {i+1}/{len(new_records)} processed from {cache_source}{retry_info} (Hit rate: {hit_rate:.1f}%)")
                    
                    # Only set values if they exist in result - no default values
                    if 'remedy_tier' in result:
                        record['remedy_tier'] = result['remedy_tier']
                    if 'monetary' in result:
                        record['monetary'] = result['monetary']
                    if 'escalation' in result:
                        record['escalation'] = result['escalation']
                    return (i, record)
                        
                except Exception as e:
                    # Check if it's a validation error that we should retry
                    is_validation_error = ("validation error" in str(e).lower() or 
                                         "string_too_long" in str(e) or
                                         "pydantic" in str(e).lower())
                    
                    if is_validation_error and retry_count < max_retries:
                        retry_count += 1
                        with lock:
                            print(f"[RETRY] Validation error for record {i+1}, retrying ({retry_count}/{max_retries}): {e}")
                        continue
                    else:
                        # Final failure - DO NOT use default values, skip this record
                        with lock:
                            cache_stats['errors'] += 1
                            print(f"[ERROR] Failed to process record {i+1} after {retry_count} retries: {e}")
                            if is_validation_error:
                                print(f"[ERROR] Validation error details: {e}")
                                print(f"[ERROR] SKIPPING record - no default values will be used")
                        # DO NOT return this record - it failed validation and we don't want fake data
                        return None
        
        # Process records in parallel
        print(f"[EXPERIMENT] Processing {len(new_records)} records with {threads_per_model} threads...")
        with ThreadPoolExecutor(max_workers=threads_per_model) as executor:
            # Submit all tasks
            future_to_record = {executor.submit(process_record, (i, record)): (i, record) for i, record in enumerate(new_records)}
            
            # Collect results as they complete
            for future in as_completed(future_to_record):
                try:
                    result = future.result()
                    if result is not None:  # Only add successful results
                        i, result_record = result
                        evaluated_records.append((i, result_record))
                    # If result is None, the record failed and we skip it (no fake data)
                except Exception as e:
                    i, record = future_to_record[future]
                    print(f"[ERROR] Thread failed for record {i+1}: {e}")
                    print(f"[ERROR] SKIPPING failed record - no fake data will be added")
                    # DO NOT add this record - it failed and we don't want fake data
        
        # Sort results by original order
        evaluated_records.sort(key=lambda x: x[0])
        evaluated_records = [record for i, record in evaluated_records]
        
        # Replace the runs.jsonl file with only the new evaluated records
        print("[EXPERIMENT] Replacing runs.jsonl with new LLM evaluation results...")
        
        # Write only the new evaluated records (replace entire file)
        with open(runs_file, 'w') as f:
            for record in evaluated_records:
                safe_record = convert_numpy(record)
                f.write(json.dumps(safe_record, ensure_ascii=False) + '\n')
        
        print(f"[EXPERIMENT] Successfully evaluated {len(evaluated_records)} records with LLM")
        
        # Print final cache usage summary
        total_processed = cache_stats['total_processed']
        cache_hits = cache_stats['cache_hits']
        llm_calls = cache_stats['llm_calls']
        errors = cache_stats['errors']
        
        if total_processed > 0:
            hit_rate = (cache_hits / total_processed) * 100
            print(f"[CACHE] Final Statistics:")
            print(f"[CACHE]   Total processed: {total_processed}")
            print(f"[CACHE]   Cache hits: {cache_hits} ({hit_rate:.1f}%)")
            print(f"[CACHE]   LLM calls: {llm_calls} ({100-hit_rate:.1f}%)")
            print(f"[CACHE]   Errors: {errors}")
            print(f"[CACHE]   Cost savings: {hit_rate:.1f}% of records served from cache")
        
        print("[EXPERIMENT] Comprehensive persona coverage now available for Complaint Categories analysis!")
        print(f"[EXPERIMENT] Generated {len(evaluated_records)} total records for {sample_size} complaints")
    else:
        print("[EXPERIMENT] No new records needed")


def generate_visualizations(results_dir: str = "gpt_4o_mini_results"):
    """Generate visualization plots for the analysis results"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np
        from pathlib import Path
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create plots directory
        plots_dir = Path(results_dir) / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Load the runs data from the out directory
        runs_file = Path("out") / "runs.jsonl"
        if not runs_file.exists():
            print(f"[VIZ] No runs data found at {runs_file}")
            return
        
        # Load data
        data = []
        with open(runs_file, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        df = pd.DataFrame(data)
        
        if df.empty:
            print("[VIZ] No data to visualize")
            return
        
        # Filter out records with None/missing remedy_tier values
        original_count = len(df)
        df = df.dropna(subset=['remedy_tier'])
        filtered_count = len(df)
        if original_count != filtered_count:
            print(f"[VIZ] Filtered out {original_count - filtered_count} records with missing remedy_tier values")
            print(f"[VIZ] Using {filtered_count} records with valid remedy_tier data")
        
        # Plot 1: Remedy tier distribution by variant
        plt.figure(figsize=(12, 8))
        variant_counts = df.groupby(['variant', 'remedy_tier']).size().unstack(fill_value=0)
        variant_counts.plot(kind='bar', stacked=True)
        plt.title('Remedy Tier Distribution by Fairness Strategy')
        plt.xlabel('Fairness Strategy')
        plt.ylabel('Count')
        plt.legend(title='Remedy Tier', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / 'remedy_tier_by_strategy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Monetary relief rate by strategy
        plt.figure(figsize=(10, 6))
        monetary_rates = df.groupby('variant')['monetary'].mean().sort_values(ascending=False)
        monetary_rates.plot(kind='bar')
        plt.title('Monetary Relief Rate by Fairness Strategy')
        plt.xlabel('Fairness Strategy')
        plt.ylabel('Monetary Relief Rate')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / 'monetary_relief_by_strategy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Escalation rate by strategy
        plt.figure(figsize=(10, 6))
        escalation_rates = df.groupby('variant')['escalation'].mean().sort_values(ascending=False)
        escalation_rates.plot(kind='bar')
        plt.title('Escalation Rate by Fairness Strategy')
        plt.xlabel('Fairness Strategy')
        plt.ylabel('Escalation Rate')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / 'escalation_by_strategy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[VIZ] Generated visualizations in {plots_dir}")
        
    except ImportError as e:
        print(f"[VIZ] Visualization dependencies not available: {e}")
        print("[VIZ] Install matplotlib and seaborn to enable visualization generation")
    except Exception as e:
        print(f"[VIZ] Error generating visualizations: {e}")


def create_argument_parser():
    """Create and configure the argument parser"""
    parser = argparse.ArgumentParser(
        description="GPT-4o-mini Fairness Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gpt-4o-mini-analysis.py --full --sample-size 1000
  python gpt-4o-mini-analysis.py --run-experiment --sample-size 500
  python gpt-4o-mini-analysis.py --analyze-only
  python gpt-4o-mini-analysis.py --clear-cache
  python gpt-4o-mini-analysis.py --cache-info
        """
    )
    
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=100, 
        help="Number of complaints to test (default: 100)"
    )
    
    parser.add_argument(
        "--threads-per-model", 
        type=int, 
        default=10, 
        help="Threads for parallel processing (default: 10)"
    )
    
    parser.add_argument(
        "--run-experiment", 
        action="store_true", 
        help="Run the experiment with gpt-4o-mini"
    )
    
    parser.add_argument(
        "--analyze-only", 
        action="store_true", 
        help="Run analysis on existing experimental data"
    )
    
    parser.add_argument(
        "--full", 
        action="store_true", 
        help="Run the complete pipeline (experiment + analysis + reports)"
    )
    
    parser.add_argument(
        "--clear-cache", 
        action="store_true", 
        help="Clear all caches before running"
    )
    
    parser.add_argument(
        "--cache-info", 
        action="store_true", 
        help="Show cache information and exit"
    )
    
    parser.add_argument(
        "--skip-viz", 
        action="store_true", 
        help="Skip visualization generation"
    )
    
    return parser


def run_full_analysis(analyzer, args, MODELS, RESULTS_DIR):
    """Run the complete analysis pipeline"""
    print("\n[MODE] Running complete analysis pipeline with gpt-4o-mini")
    
    # Clear cache if requested
    if args.clear_cache:
        clear_cache()
    
    # Generate new experiments with all 6 fairness strategies
    generate_experiments_with_all_strategies(args.sample_size, args.threads_per_model)
    
    # Load the updated data and run analysis
    analyzer._load_existing_results()
    analyses = analyzer.run_all_analyses()
    # Generate comprehensive report using existing analyses (avoid re-running)
    analyzer.report_generator.generate_comprehensive_report(analyses, "gpt4o_analysis_results.md")
    
    # Generate visualizations if not skipped
    if not args.skip_viz:
        generate_visualizations(RESULTS_DIR)


def run_experiment_only(analyzer, args, MODELS):
    """Run experiment only"""
    print("\n[MODE] Running experiment with gpt-4o-mini only")
    
    # Clear cache if requested
    if args.clear_cache:
        clear_cache()
    
    # Generate new experiments with all 6 fairness strategies
    generate_experiments_with_all_strategies(args.sample_size, args.threads_per_model)
    
    # Load the updated data
    analyzer._load_existing_results()
    

def run_analysis_only(analyzer, args, RESULTS_DIR):
    """Run analysis on existing data"""
    print("\n[MODE] Running analysis on existing gpt-4o-mini data")
    
    # Set sample size for data filtering (use default if not specified)
    analyzer.sample_size = args.sample_size
    
    # Load existing results first
    analyzer._load_existing_results()
    
    # Run all analyses using the unified method
    analyses = analyzer.run_all_analyses()
    
    # Generate comprehensive report using existing analyses (avoid re-running)
    analyzer.report_generator.generate_comprehensive_report(analyses, "gpt4o_analysis_results.md")
    
    # Generate visualizations if not skipped
    if not args.skip_viz:
        generate_visualizations(RESULTS_DIR)


def main():
    """Main entry point for the GPT-4o-mini fairness analysis"""
    
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Handle cache operations first
    if args.clear_cache:
        clear_cache()
        return
    
    if args.cache_info:
        show_cache_info()
        return
    
    # Validate arguments
    if not any([args.full, args.run_experiment, args.analyze_only]):
        parser.error("Must specify one of: --full, --run-experiment, or --analyze-only")
    
    # Fixed parameters for gpt-4o-mini analysis
    MODELS = ["gpt-4o-mini"]
    RESULTS_DIR = "gpt_4o_mini_results"
    
    try:
        # Initialize the analyzer with gpt_4o_mini_results directory
        print(f"[INIT] Initializing GPT-4o-mini Fairness Analyzer...")
        print(f"[INIT] Results directory: {RESULTS_DIR}")
        print(f"[INIT] Model: gpt-4o-mini")
        print(f"[INIT] Sample size: {args.sample_size}")
        print(f"[INIT] Threads: {args.threads_per_model}")
        
        analyzer = AdvancedFairnessAnalyzer(results_dir=RESULTS_DIR)
        
        if args.full:
            run_full_analysis(analyzer, args, MODELS, RESULTS_DIR)
        elif args.run_experiment:
            run_experiment_only(analyzer, args, MODELS)
        elif args.analyze_only:
            run_analysis_only(analyzer, args, RESULTS_DIR)
            
        print(f"\n[SUCCESS] GPT-4o-mini analysis completed successfully!")
        print(f"[INFO] Results available in: {RESULTS_DIR}/")
        
    except KeyboardInterrupt:
        print(f"\n[INTERRUPT] Analysis interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
