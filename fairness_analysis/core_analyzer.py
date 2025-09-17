"""
Core AdvancedFairnessAnalyzer class - main orchestrator
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from .data_loader import DataLoader
from .statistical_analyzer import StatisticalAnalyzer
from .report_generator import ReportGenerator
from .experiment_runner import ExperimentRunner


class AdvancedFairnessAnalyzer:
    """Main class for conducting advanced fairness analysis of LLMs"""
    
    def __init__(self, results_dir: str = "advanced_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.report_generator = ReportGenerator(self.results_dir)
        self.experiment_runner = ExperimentRunner(results_dir)
        
        # Data storage
        self.baseline_results = None
        self.persona_results = None
        self.strategy_results = None
        self.raw_results = None
        self.sample_size = None  # Store sample size for data filtering
        
    def run_enhanced_experiment(self, models: Optional[List[str]] = None, 
                              sample_size: int = 100, threads_per_model: int = 5):
        """Run enhanced experiment with bias detection"""
        # Store sample size for data filtering
        self.sample_size = sample_size
        return self.experiment_runner.run_enhanced_experiment(
            models=models,
            sample_size=sample_size,
            threads_per_model=threads_per_model
        )
        
    def run_all_analyses(self, use_cache: bool = False) -> Dict:
        """Run all statistical analyses"""
        print("\n[ANALYSIS] Running all statistical analyses...")
        
        # Only load data if we don't already have it loaded
        if self.raw_results is None or len(self.raw_results) == 0:
            if use_cache:
                self._load_existing_results()
            else:
                # Load fresh raw results for analysis
                self._load_raw_results()
        
        analyses = {}
        
        try:
            # Run individual analyses
            analyses["ground_truth"] = self.statistical_analyzer.analyze_ground_truth(
                self.raw_results
            )
            analyses["demographic_injection"] = self.statistical_analyzer.analyze_demographic_injection_effect(
                self.raw_results
            )
            analyses["gender_effects"] = self.statistical_analyzer.analyze_gender_effects(
                self.raw_results
            )
            analyses["ethnicity_effects"] = self.statistical_analyzer.analyze_ethnicity_effects(
                self.raw_results
            )
            analyses["geography_effects"] = self.statistical_analyzer.analyze_geography_effects(
                self.raw_results
            )
            analyses["granular_bias"] = self.statistical_analyzer.analyze_granular_bias(
                self.raw_results
            )
            analyses["bias_directional_consistency"] = self.statistical_analyzer.analyze_bias_directional_consistency(
                self.raw_results
            )
            analyses["fairness_strategies"] = self.statistical_analyzer.analyze_fairness_strategies(
                self.raw_results
            )
            analyses["process_fairness"] = self.statistical_analyzer.analyze_process_fairness(
                self.raw_results
            )
            analyses["severity_bias_variation"] = self.statistical_analyzer.analyze_severity_bias_variation(
                self.raw_results
            )
            analyses["severity_context"] = self.statistical_analyzer.analyze_severity_context(
                self.raw_results
            )
            analyses["model_scaling"] = self.statistical_analyzer.analyze_scaling_laws(
                self.persona_results
            )
            
            print(f"[OK] Completed {len(analyses)} analyses")
            
        except Exception as e:
            print(f"[ERROR] Analysis failed: {e}")
            
        return analyses
    
    def _load_raw_results(self):
        """Load only raw experimental results without cached analysis data"""
        # Load raw experimental results from runs.jsonl
        runs_file = Path("out/runs.jsonl")
        if runs_file.exists():
            print(f"[DATA] Loaded experimental results from: {runs_file}")
            with open(runs_file, 'r', encoding='utf-8') as f:
                self.raw_results = []
                for line in f:
                    if line.strip():
                        self.raw_results.append(json.loads(line))
            
            # Apply sample size filtering if specified
            if self.sample_size and len(self.raw_results) > self.sample_size:
                print(f"[DATA] Filtering to sample size: {self.sample_size} (from {len(self.raw_results)} total records)")
                self.raw_results = self._filter_to_experiment_samples(self.raw_results, self.sample_size)
        else:
            print("[WARNING] No experimental results found. Run experiments first.")
            self.raw_results = []
        
    def generate_comprehensive_report(self, filename: str = "advanced_research_summary.md", use_cache: bool = False) -> str:
        """Generate comprehensive analysis report"""
        analyses = self.run_all_analyses(use_cache=use_cache)
        return self.report_generator.generate_comprehensive_report(analyses, filename)
        
    def generate_directional_fairness_report(self) -> str:
        """Generate directional fairness report"""
        analyses = self.run_all_analyses()
        return self.report_generator.generate_directional_fairness_report(analyses)
        
    def run_full_analysis(self, sample_size: int = 100, threads_per_model: int = 5):
        """Run complete advanced analysis pipeline"""
        print("\n[START] Starting Advanced Fairness Analysis Pipeline")
        
        # Run experiments with enhanced threading
        self.run_enhanced_experiment(sample_size=sample_size, threads_per_model=threads_per_model)
        
        # Generate all analyses
        analyses = self.run_all_analyses()
        
        # Generate reports
        report_path = self.generate_comprehensive_report()
        
        print(f"\n[OK] Advanced analysis complete!")
        print(f"[DATA] Results saved to: {self.results_dir}")
        print(f"[REPORT] Report available at: {report_path}")
        
        return analyses
        
    def _load_existing_results(self):
        """Load existing experimental results"""
        # Map of analysis types to actual data files
        data_file_mapping = {
            "raw_results": "runs.jsonl",  # Standard runs.jsonl file
            "persona_results": "persona_model_interactions_analysis.json",
            "strategy_results": "strategy_accuracy_equalization_analysis.json",
            "directional_analysis": "directional_fairness_analysis.json",
            "ground_truth": "ground_truth_validation_analysis.json"
        }
        
        # Check for data in the results directory first, then fall back to main out directory
        local_runs_file = self.results_dir / "runs.jsonl"
        out_dir = Path("out")

        if local_runs_file.exists():
            # Use the local results directory runs.jsonl if it exists (mark as absolute to avoid double-prefixing)
            data_file_mapping["raw_results"] = f"ABSOLUTE:{str(local_runs_file)}"
            # print(f"[DEBUG] Found local experimental data at {local_runs_file}")
        elif out_dir.exists() and (out_dir / "runs.jsonl").exists():
            # Fall back to the main experimental results if no local file (mark as absolute)
            data_file_mapping["raw_results"] = f"ABSOLUTE:{str(out_dir / 'runs.jsonl')}"
            # print(f"[DEBUG] Using main experimental data at {out_dir / 'runs.jsonl'}")
        
        files_found = []
        for data_type, filename in data_file_mapping.items():
            # Handle both relative and absolute paths
            if str(filename).startswith('ABSOLUTE:'):
                # Remove the ABSOLUTE: prefix and use as-is
                file_path = Path(filename[9:])
            elif Path(filename).is_absolute() or str(filename).startswith(('out/', 'advanced_results/', 'out\\', 'advanced_results\\')):
                file_path = Path(filename)
            else:
                file_path = self.results_dir / filename
                
            # print(f"[DEBUG] Checking {data_type}: {filename} -> {file_path} (exists: {file_path.exists()})")
            if file_path.exists():
                files_found.append(filename)
                # print(f"[DEBUG] Loading {data_type} from {file_path}")
                try:
                    if filename.endswith('.jsonl'):
                        # Handle JSONL files (like enhanced_runs.jsonl)
                        data = []
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                if line.strip():
                                    data.append(json.loads(line))
                        
                        # Apply sample size filtering if specified
                        if data_type == "raw_results" and self.sample_size and len(data) > self.sample_size:
                            print(f"[DATA] Filtering to sample size: {self.sample_size} (from {len(data)} total records)")
                            # Filter to use only the specific complaints that were selected for the experiment
                            data = self._filter_to_experiment_samples(data, self.sample_size)
                    else:
                        # Handle regular JSON files
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                    
                    if data_type == "raw_results":
                        self.raw_results = data
                    elif data_type == "persona_results":
                        self.persona_results = data
                    elif data_type == "strategy_results":
                        self.strategy_results = data
                    elif data_type == "directional_analysis":
                        self.directional_results = data
                    elif data_type == "ground_truth":
                        self.ground_truth_results = data
                except Exception as e:
                    print(f"[WARNING] Failed to load {filename}: {e}")
        
        # Try to construct baseline results from persona data
        if self.persona_results and 'discrimination_matrix' in self.persona_results:
            # Extract baseline values from the persona analysis
            self.baseline_results = {"extracted_from_persona_analysis": True}
            
        if files_found:
            print(f"[DATA] Loaded experimental results from: {', '.join(files_found)}")
        else:
            print("[WARNING] No existing experimental results found")
            
    def _filter_to_experiment_samples(self, data: List[Dict], sample_size: int) -> List[Dict]:
        """Filter data to use only the specific complaints that were selected for the experiment"""
        
        # Load the pairs.jsonl to get the specific complaints that were selected
        pairs_file = Path("out/pairs.jsonl")
        if not pairs_file.exists():
            print(f"[WARNING] pairs.jsonl not found, using first {sample_size} records")
            return data[:sample_size]
        
        # Load the specific case_ids that were selected for the experiment
        selected_case_ids = set()
        with open(pairs_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    pair_data = json.loads(line)
                    selected_case_ids.add(pair_data.get('case_id'))
        
        # Filter the data to only include results for the selected complaints
        filtered_data = [record for record in data if record.get('case_id') in selected_case_ids]
        
        # Limit to the first N unique complaints (sample_size)
        unique_case_ids = set()
        limited_data = []
        for record in filtered_data:
            case_id = record.get('case_id')
            if case_id not in unique_case_ids and len(unique_case_ids) < sample_size:
                unique_case_ids.add(case_id)
                limited_data.append(record)
            elif case_id in unique_case_ids:
                limited_data.append(record)
        filtered_data = limited_data
        
        print(f"[DATA] Filtered to {len(filtered_data)} records from {len(unique_case_ids)} unique complaints (sample_size={sample_size})")
        return filtered_data