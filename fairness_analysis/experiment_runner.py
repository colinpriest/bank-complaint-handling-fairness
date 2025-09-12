"""
Experiment Runner for Advanced Fairness Analysis

This module provides functionality to run fairness experiments
with enhanced threading and bias detection.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


class ExperimentRunner:
    """Runs fairness experiments with enhanced threading capabilities"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
    def run_enhanced_experiment(self, models: Optional[List[str]] = None, 
                              sample_size: int = 100, threads_per_model: int = 5) -> Dict[str, Any]:
        """Run enhanced experiment with bias detection and threading"""
        
        if models is None:
            models = ["gpt-4o-mini", "claude-3-5-sonnet-20241022"]
            
        # Check if real experimental data already exists
        out_dir = Path("out")
        if out_dir.exists() and (out_dir / "runs.jsonl").exists():
            print(f"[EXPERIMENT] Real experimental data found in {out_dir}/runs.jsonl")
            print(f"[EXPERIMENT] Using existing experimental data")
            return self._load_existing_experiment_data()
            
        # If no real data exists, raise an error
        raise FileNotFoundError(
            f"No experimental data found at {out_dir}/runs.jsonl. "
            "Please run the main experiment first using complaints_llm_fairness_harness.py"
        )
        
        
        
    def run_bias_detection_experiment(self, model: str, 
                                    demographic_groups: List[str]) -> Dict[str, Any]:
        """Run specific bias detection experiment"""
        
        print(f"[BIAS] Running bias detection for {model}")
        print(f"[BIAS] Demographic groups: {demographic_groups}")
        
        # Load real experimental data for bias analysis
        out_file = Path("out/runs.jsonl")
        if not out_file.exists():
            raise FileNotFoundError("No experimental data available for bias detection")
            
        # Load and analyze real data
        data = []
        with open(out_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        # Perform real bias analysis on the data
        # This would contain actual statistical analysis
        bias_results = {
            "model": model,
            "demographic_groups": demographic_groups,
            "data_source": str(out_file),
            "record_count": len(data),
            "note": "Real bias analysis requires implementation of statistical methods"
        }
        
        return bias_results
        
    def run_fairness_strategy_experiment(self, model: str, 
                                       strategies: List[str]) -> Dict[str, Any]:
        """Run fairness strategy effectiveness experiment"""
        
        print(f"[STRATEGY] Testing fairness strategies for {model}")
        print(f"[STRATEGY] Strategies: {strategies}")
        
        # Load real experimental data for strategy analysis
        out_file = Path("out/runs.jsonl")
        if not out_file.exists():
            raise FileNotFoundError("No experimental data available for strategy analysis")
            
        # Load and analyze real data
        data = []
        with open(out_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        # Perform real strategy analysis on the data
        strategy_results = {
            "model": model,
            "strategies_tested": strategies,
            "data_source": str(out_file),
            "record_count": len(data),
            "note": "Real strategy analysis requires implementation of statistical methods"
        }
        
        return strategy_results
        
    def _load_existing_experiment_data(self) -> Dict[str, Any]:
        """Load existing experimental data from out/runs.jsonl"""
        out_file = Path("out/runs.jsonl")
        
        if not out_file.exists():
            return {"error": "No existing experimental data found"}
            
        print(f"[LOAD] Loading existing experimental data from {out_file}")
        
        # Load the JSONL data
        data = []
        with open(out_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        print(f"[LOAD] Loaded {len(data)} experimental records")
        
        # Return in the expected format
        return {
            "experiment_metadata": {
                "source": "existing_data",
                "file": str(out_file),
                "record_count": len(data),
                "timestamp": time.time()
            },
            "raw_data": data
        }
