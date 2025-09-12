#!/usr/bin/env python3
"""
GPT-4o-mini Fairness Analysis

This script runs fairness analysis using only the gpt-4o-mini model.
Results are saved to the gpt_4o_mini_results directory.
"""

import argparse
import json
import sys
from pathlib import Path

# Add the fairness_analysis package to the path
sys.path.insert(0, str(Path(__file__).parent))

from fairness_analysis import AdvancedFairnessAnalyzer


def main():
    """Main entry point for the GPT-4o-mini fairness analysis"""
    
    parser = argparse.ArgumentParser(
        description="GPT-4o-mini Fairness Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gpt-4o-mini-analysis.py --full --sample-size 1000
  python gpt-4o-mini-analysis.py --run-experiment --sample-size 500
  python gpt-4o-mini-analysis.py --analyze-only
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
    
    args = parser.parse_args()
    
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
            print("\n[MODE] Running complete analysis pipeline with gpt-4o-mini")
            # Override the default models in the full analysis
            analyzer.run_enhanced_experiment(
                models=MODELS,
                sample_size=args.sample_size,
                threads_per_model=args.threads_per_model
            )
            # Then run analysis and generate report
            analyzer.run_all_analyses()
            analyzer.generate_comprehensive_report("gpt4o_analysis_results.md")
            
        elif args.run_experiment:
            print("\n[MODE] Running experiment with gpt-4o-mini only")
            analyzer.run_enhanced_experiment(
                models=MODELS, 
                sample_size=args.sample_size,
                threads_per_model=args.threads_per_model
            )
            
        elif args.analyze_only:
            print("\n[MODE] Running analysis on existing gpt-4o-mini data")
            
            # Set sample size for data filtering (use default if not specified)
            analyzer.sample_size = args.sample_size
            
            # Load existing results first
            analyzer._load_existing_results()
            
            # Run individual analyses
            print("Running statistical analyses...")
            
            analyses = {
                "demographic_injection": analyzer.statistical_analyzer.analyze_demographic_injection_effect(
                    analyzer.raw_results
                ),
                "gender_effects": analyzer.statistical_analyzer.analyze_gender_effects(
                    analyzer.raw_results
                ),
                "ethnicity_effects": analyzer.statistical_analyzer.analyze_ethnicity_effects(
                    analyzer.raw_results
                ),
                "geography_effects": analyzer.statistical_analyzer.analyze_geography_effects(
                    analyzer.raw_results
                ),
                "granular_bias": analyzer.statistical_analyzer.analyze_granular_bias(
                    analyzer.raw_results
                ),
                "bias_directional_consistency": analyzer.statistical_analyzer.analyze_bias_directional_consistency(
                    analyzer.raw_results
                ),
                "fairness_strategies": analyzer.statistical_analyzer.analyze_fairness_strategies(
                    analyzer.raw_results
                ),
                "process_fairness": analyzer.statistical_analyzer.analyze_process_fairness(
                    analyzer.raw_results
                ),
                "severity_context": analyzer.statistical_analyzer.analyze_severity_context(
                    analyzer.raw_results
                ),
                "scaling_laws": analyzer.statistical_analyzer.analyze_scaling_laws(
                    analyzer.persona_results
                )
            }
            
            # Save individual analysis results
            for name, results in analyses.items():
                output_file = analyzer.results_dir / f"{name}_analysis.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    # Convert numpy types for JSON serialization
                    import numpy as np
                    
                    def convert_numpy(obj):
                        """Convert numpy types to native Python types"""
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
                    
                    json.dump(convert_numpy(results), f, indent=2)
                print(f"[SAVE] {name} analysis saved to {output_file}")
            
            # Generate comprehensive report
            analyzer.generate_comprehensive_report("gpt4o_analysis_results.md")
            
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