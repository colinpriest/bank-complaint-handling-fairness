#!/usr/bin/env python3
"""
Advanced LLM Fairness Analysis - Clean Modular Version

This is a clean, refactored version of the fairness analysis tool.
The original monolithic file has been broken into modular components.
"""

import argparse
import json
import sys
from pathlib import Path

# Add the fairness_analysis package to the path
sys.path.insert(0, str(Path(__file__).parent))

from fairness_analysis import AdvancedFairnessAnalyzer


def main():
    """Main entry point for the advanced fairness analysis"""
    
    parser = argparse.ArgumentParser(
        description="Advanced LLM Fairness Analysis - Modular Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python advanced_fairness_analysis_v2.py --full --sample-size 1000
  python advanced_fairness_analysis_v2.py --run-experiment --models gpt-4o claude-3-5-sonnet
  python advanced_fairness_analysis_v2.py --analyze-only
        """
    )
    
    parser.add_argument(
        "--models", 
        nargs="+", 
        help="Models to test (e.g., gpt-4o, claude-3-5-sonnet, gemini-2.5-pro)"
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
        default=5, 
        help="Threads per model for parallel processing (default: 5)"
    )
    
    parser.add_argument(
        "--run-experiment", 
        action="store_true", 
        help="Run the enhanced experiment with bias detection"
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
        "--results-dir",
        type=str,
        default="advanced_results",
        help="Directory to save results (default: advanced_results)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.full, args.run_experiment, args.analyze_only]):
        parser.error("Must specify one of: --full, --run-experiment, or --analyze-only")
    
    try:
        # Initialize the analyzer
        print(f"[INIT] Initializing Advanced Fairness Analyzer...")
        print(f"[INIT] Results directory: {args.results_dir}")
        print(f"[INIT] Sample size: {args.sample_size}")
        print(f"[INIT] Threads per model: {args.threads_per_model}")
        
        analyzer = AdvancedFairnessAnalyzer(results_dir=args.results_dir)
        
        if args.full:
            print("\n[MODE] Running complete analysis pipeline")
            analyzer.run_full_analysis(
                sample_size=args.sample_size,
                threads_per_model=args.threads_per_model
            )
            
        elif args.run_experiment:
            print("\n[MODE] Running enhanced experiment only")
            analyzer.run_enhanced_experiment(
                models=args.models, 
                sample_size=args.sample_size,
                threads_per_model=args.threads_per_model
            )
            
        elif args.analyze_only:
            print("\n[MODE] Running analysis on existing data")
            
            # Load raw experimental data
            analyzer._load_raw_results()
            
            # Run individual analyses
            print("Running statistical analyses...")
            
            # Combine baseline and persona results for methods that need both
            combined_results = (analyzer.baseline_results or []) + (analyzer.persona_results or [])
            
            analyses = {
                "granular_bias": analyzer.statistical_analyzer.analyze_granular_bias(
                    analyzer.persona_results
                ),
                "process_fairness": analyzer.statistical_analyzer.analyze_process_fairness(
                    analyzer.raw_results
                ),
                "severity_context": analyzer.statistical_analyzer.analyze_severity_context(
                    analyzer.persona_results
                ),
                "severity_bias_variation": analyzer.statistical_analyzer.analyze_severity_bias_variation(
                    combined_results
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
            analyzer.generate_comprehensive_report("multimodel_analysis_results.md")
            
        print(f"\n[SUCCESS] Analysis completed successfully!")
        print(f"[INFO] Results available in: {analyzer.results_dir}")
        
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