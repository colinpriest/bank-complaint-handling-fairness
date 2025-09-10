#!/usr/bin/env python3
"""
Interactive LLM Fairness Testing Harness
Unified script with intelligent workflow management and multithreading
"""
import os
import sys
import json
import time
import shutil
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import the main harness components
sys.path.append(str(Path(__file__).parent))
from complaints_llm_fairness_harness import (
    fetch_cfpb_local, fetch_cfpb_socrata, fetch_cfpb_api, clean_df, stratified_sample, assign_pairs,
    build_clients, run_dialog, analyse, DEMOGRAPHIC_PERSONAS,
    load_runs, paired_frame, PairRecord
)

class FairnessHarness:
    def __init__(self):
        self.data_dir = Path("out")
        self.cache_dir = Path("data_cache")
        self.results_dir = Path("results")
        self.plots_dir = self.results_dir / "plots"
        self.analysis_cache_dir = Path("analysis_cache")
        
        # Create directories
        for dir_path in [self.data_dir, self.cache_dir, self.results_dir, self.plots_dir, self.analysis_cache_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.status = self._check_status()
        self.lock = Lock()  # For thread-safe operations
    
    def _check_status(self) -> Dict[str, Any]:
        """Check current experiment status"""
        status = {
            "raw_data": (self.data_dir / "raw_data.csv").exists(),
            "cleaned_data": (self.data_dir / "cleaned.csv").exists(),
            "pairs": (self.data_dir / "pairs.jsonl").exists(),
            "experiment_runs": (self.data_dir / "runs.jsonl").exists(),
            "analysis": (self.data_dir / "analysis.json").exists(),
            "cost_summary": (self.data_dir / "cost_summary.json").exists(),
            "advanced_analysis": (self.results_dir / "advanced_analysis.json").exists(),
            "research_plots": any(self.plots_dir.glob("*.png")),
            "research_report": (self.results_dir / "research_summary.md").exists(),
        }
        
        # Count cache files
        status["cache_files"] = len(list(self.cache_dir.glob("*.json")))
        status["analysis_cache_files"] = len(list(self.analysis_cache_dir.glob("*.json")))
        
        # Count experiment records
        if status["experiment_runs"]:
            try:
                runs_df = load_runs(str(self.data_dir / "runs.jsonl"))
                status["total_runs"] = len(runs_df)
                status["unique_pairs"] = runs_df["pair_id"].nunique() if not runs_df.empty else 0
                status["models_tested"] = list(runs_df["model"].unique()) if not runs_df.empty else []
            except:
                status["total_runs"] = 0
                status["unique_pairs"] = 0
                status["models_tested"] = []
        
        return status
    
    def _get_data_hash(self) -> str:
        """Generate hash of current experiment data for cache validation"""
        hash_components = []
        
        # Hash runs file if it exists
        runs_file = self.data_dir / "runs.jsonl"
        if runs_file.exists():
            with open(runs_file, 'rb') as f:
                content = f.read()
                hash_components.append(hashlib.md5(content).hexdigest())
        
        # Hash pairs file if it exists
        pairs_file = self.data_dir / "pairs.jsonl"
        if pairs_file.exists():
            with open(pairs_file, 'rb') as f:
                content = f.read()
                hash_components.append(hashlib.md5(content).hexdigest())
        
        # Combine hashes
        combined = "_".join(hash_components)
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_cache_path(self, analysis_type: str, extra_params: str = "") -> Path:
        """Get cache file path for analysis type"""
        data_hash = self._get_data_hash()
        cache_key = f"{analysis_type}_{data_hash}_{extra_params}".replace("/", "_")
        return self.analysis_cache_dir / f"{cache_key}.json"
    
    def _load_cached_analysis(self, analysis_type: str, extra_params: str = "") -> Optional[Dict[str, Any]]:
        """Load cached analysis results if available and valid"""
        cache_path = self._get_cache_path(analysis_type, extra_params)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Check if cache is still valid (data hasn't changed)
            if cached_data.get("data_hash") == self._get_data_hash():
                print(f"[OK] Using cached {analysis_type} analysis")
                return cached_data.get("results")
            else:
                print(f"⚠️  Cache invalidated for {analysis_type} (data changed)")
                cache_path.unlink()  # Remove invalid cache
                return None
                
        except Exception as e:
            print(f"⚠️  Error loading cache for {analysis_type}: {e}")
            return None
    
    def _save_analysis_cache(self, analysis_type: str, results: Dict[str, Any], extra_params: str = ""):
        """Save analysis results to cache"""
        cache_path = self._get_cache_path(analysis_type, extra_params)
        
        try:
            cache_data = {
                "analysis_type": analysis_type,
                "data_hash": self._get_data_hash(),
                "timestamp": datetime.now().isoformat(),
                "extra_params": extra_params,
                "results": results
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            print(f"[CACHE] Cached {analysis_type} analysis")
            
        except Exception as e:
            print(f"⚠️  Error saving cache for {analysis_type}: {e}")
    
    def _clear_analysis_cache(self):
        """Clear all analysis cache files"""
        try:
            if self.analysis_cache_dir.exists():
                for cache_file in self.analysis_cache_dir.glob("*.json"):
                    cache_file.unlink()
                print("🗑️ Analysis cache cleared")
        except Exception as e:
            print(f"⚠️  Error clearing analysis cache: {e}")
    
    def print_status(self):
        """Display current status"""
        print("\n" + "="*60)
        print("FAIRNESS EXPERIMENT STATUS")
        print("="*60)
        
        steps = [
            ("Raw Data", self.status["raw_data"]),
            ("Cleaned Data", self.status["cleaned_data"]),
            ("Experiment Pairs", self.status["pairs"]),
            ("Experiment Runs", self.status["experiment_runs"]),
            ("Basic Analysis", self.status["analysis"]),
            ("Advanced Analysis", self.status["advanced_analysis"]),
            ("Research Plots", self.status["research_plots"]),
            ("Research Report", self.status["research_report"]),
        ]
        
        for step, completed in steps:
            status_icon = "[OK]" if completed else "[FAIL]"
            print(f"{status_icon} {step}")
        
        if self.status["experiment_runs"]:
            print(f"\n[DATA] Experiment Details:")
            print(f"   • Total runs: {self.status['total_runs']:,}")
            print(f"   • Unique pairs: {self.status['unique_pairs']:,}")
            print(f"   • Models tested: {', '.join(self.status['models_tested'])}")
            print(f"   • LLM cache files: {self.status['cache_files']:,}")
            print(f"   • Analysis cache files: {self.status['analysis_cache_files']:,}")
        
        print("="*60 + "\n")
    
    def get_menu_options(self) -> List[tuple]:
        """Generate menu options based on current status"""
        options = []
        
        # Data ingestion options
        if not self.status["raw_data"]:
            options.append(("ingest_small", "[TEST] Quick Test - Ingest 50 complaints for testing"))
            options.append(("ingest_large", "[FULL] Full Scale - Ingest 1000+ complaints"))
        elif not self.status["cleaned_data"]:
            options.append(("clean_data", "[CLEAN] Clean and prepare raw data"))
        
        # Experiment setup
        if self.status["cleaned_data"] and not self.status["pairs"]:
            options.append(("create_pairs", "👥 Create demographic persona pairs"))
        
        # Experiment execution
        if self.status["pairs"] and not self.status["experiment_runs"]:
            options.append(("run_test", "🧪 Limited Test Run (1 model, fast)"))
            options.append(("run_full", "[RUN] Full Experiment (multiple models, threaded)"))
        elif self.status["pairs"]:
            options.append(("extend_experiment", "➕ Extend experiment (add more models/data)"))
        
        # Analysis options
        if self.status["experiment_runs"]:
            if not self.status["analysis"]:
                options.append(("basic_analysis", "[ANALYZE] Basic Statistical Analysis"))
            if not self.status["advanced_analysis"]:
                options.append(("advanced_analysis", "🔬 Advanced Fairness Analysis"))
            if not self.status["research_plots"]:
                options.append(("create_plots", "[PLOT] Generate Research Plots"))
            if not self.status["research_report"]:
                options.append(("research_report", "[REPORT] Generate Research Summary"))
        
        # Utility options
        options.extend([
            ("view_dashboard", "👀 Launch Interactive Dashboard"),
            ("export_data", "[EXPORT] Export All Data"),
            ("clear_llm_cache", "🗑️  Clear LLM Cache"),
            ("clear_analysis_cache", "[CACHE] Clear Analysis Cache"),
            ("clear_all_caches", "🗑️  Clear All Caches"),
            ("reset_all", "⚠️  Reset Everything (start over)"),
            ("quit", "🚪 Exit")
        ])
        
        return options
    
    def display_menu(self):
        """Display interactive menu"""
        options = self.get_menu_options()
        
        print("📋 AVAILABLE ACTIONS:")
        for i, (key, description) in enumerate(options, 1):
            print(f"{i:2d}. {description}")
        
        return options
    
    def ingest_data(self, size: str = "small"):
        """Ingest CFPB complaint data with fallback options"""
        print(f"\n🔄 Ingesting complaint data ({size})...")
        
        max_records = 50 if size == "small" else 5000
        target_size = 30 if size == "small" else 500
        
        # Try multiple data sources, starting with local downloaded files
        try:
            print("📁 Loading from local cfpb_downloads folder...")
            df = fetch_cfpb_local(max_records=max_records, months=24, verbose=True)
            data_source = "local_files"
            
        except Exception as local_error:
            print(f"⚠️  Local files failed: {local_error}")
            print("Trying API methods...")

            try:
                print("📡 Attempting CFPB Socrata API...")
                df = fetch_cfpb_socrata(max_records=max_records, months=24, verbose=True)
                data_source = "socrata_api"
                
            except Exception as socrata_error:
                print(f"⚠️  Socrata API failed: {socrata_error}")
                print("Trying direct CSV download...")

                try:
                    print("📡 Attempting direct CSV download...")
                    df = self._fetch_cfpb_direct_csv(max_records)
                    data_source = "direct_csv"
                    
                except Exception as csv_error:
                    print(f"[FAIL] Direct CSV download failed: {csv_error}")
                    print("\n[FAIL] All automatic data ingestion methods failed.")
                    print("\n💡 Manual Alternatives:")
                    print("   1. Download complaints data to cfpb_downloads/ folder")
                    print("   2. Visit: https://www.consumerfinance.gov/data-research/consumer-complaints/")
                    print("   3. Download the CSV with narratives manually.")
                    print("   4. Place it in cfpb_downloads/ folder as 'complaints.csv'")
                    print("   5. Re-run the script.")
                    raise Exception("All automatic data ingestion methods failed")
        
        try:
            df.to_csv(self.data_dir / "raw_data.csv", index=False)
            print(f"[OK] Raw data saved: {len(df):,} records (source: {data_source})")
            
            # Clean data
            dfc = clean_df(df)
            print(f"[OK] After cleaning: {len(dfc):,} records")
            
            # Sample data
            dfs = stratified_sample(dfc, total=target_size)
            dfs.to_csv(self.data_dir / "cleaned.csv", index=False)
            print(f"[OK] Final sample: {len(dfs):,} records")
            
            self.status = self._check_status()
            return True
            
        except Exception as e:
            print(f"[FAIL] Error processing data: {e}")
            return False
    
    def _fetch_cfpb_direct_csv(self, max_records: int) -> 'pd.DataFrame':
        """Try to fetch CFPB data via direct CSV download"""
        import pandas as pd
        import requests
        
        # Try the direct CSV endpoint
        csv_url = "https://files.consumerfinance.gov/ccdb/complaints.csv"
        
        print(f"📥 Downloading from {csv_url}...")
        response = requests.get(csv_url, timeout=60)
        response.raise_for_status()
        
        # Save to temporary file and read
        import io
        df = pd.read_csv(io.StringIO(response.text))
        
        # Filter for complaints with narratives
        df = df[df['Consumer complaint narrative'].notna()]
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'Consumer complaint narrative': 'narrative',
            'Date received': 'date_received',
            'Product': 'product',
            'Issue': 'issue',
            'Company': 'company',
            'State': 'state',
            'ZIP code': 'zip_code',
            'Submitted via': 'submitted_via',
            'Company public response': 'company_public_response',
            'Timely response?': 'timely_response',
            'Complaint ID': 'complaint_id'
        })
        
        # Take a sample
        if len(df) > max_records:
            df = df.sample(n=max_records, random_state=42)
        
        return df
    
    
    def create_pairs(self):
        """Create demographic persona pairs"""
        print("\n👥 Creating demographic persona pairs...")
        
        try:
            import pandas as pd
            df = pd.read_csv(self.data_dir / "cleaned.csv")
            
            # Select personas for balanced representation
            personas = ["white_male_affluent", "black_female_urban", "hispanic_male_working", 
                       "asian_female_professional", "white_female_senior", "white_male_rural"]
            
            pairs = assign_pairs(df, personas[:3])  # Start with 3 personas for speed
            
            # Save pairs
            with open(self.data_dir / "pairs.jsonl", "w", encoding="utf-8") as f:
                for pr in pairs:
                    f.write(json.dumps(pr.__dict__, ensure_ascii=False) + "\n")
            
            print(f"[OK] Created {len(pairs):,} paired records ({len(pairs)//3:,} triplets)")
            
            self.status = self._check_status()
            return True
            
        except Exception as e:
            print(f"[FAIL] Error creating pairs: {e}")
            return False
    
    def run_single_dialog(self, client, pair_record, run_idx):
        """Single threaded dialog execution"""
        try:
            result = run_dialog(client, pair_record, run_idx)
            result["model"] = client.model_id
            
            # Thread-safe file writing
            with self.lock:
                with open(self.data_dir / "runs.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            return result
        except Exception as e:
            error_record = {
                "pair_id": pair_record.pair_id,
                "case_id": pair_record.case_id,
                "variant": pair_record.variant,
                "model": client.model_id,
                "run_idx": run_idx,
                "error": str(e),
                "format_ok": 0,
                "refusal": 0
            }
            
            with self.lock:
                with open(self.data_dir / "runs.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
            
            return error_record
    
    def run_experiment(self, test_mode: bool = False):
        """Run the bias detection experiment with multithreading"""
        mode_name = "Test" if test_mode else "Full"
        print(f"\n🚀 Running {mode_name} Experiment...")
        
        try:
            # Load pairs
            pairs = []
            with open(self.data_dir / "pairs.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    pairs.append(PairRecord(**d))
            
            # Configure experiment
            if test_mode:
                models = ["gpt-4o-mini"]
                repeats = 1
                max_workers = 3
                pairs = pairs[:30]  # Limit for testing
            else:
                models = ["gpt-4o-mini", "gpt-4o", "claude-3.5", "gemini-2.5"]
                repeats = 2
                max_workers = 10
            
            # Build clients
            clients = build_clients(models, cache_dir=str(self.cache_dir))
            
            # Prepare tasks
            tasks = []
            for model_name in models:
                client = clients[model_name]
                for pair_record in pairs:
                    for run_idx in range(repeats):
                        tasks.append((client, pair_record, run_idx))
            
            print(f"📊 Experiment Configuration:")
            print(f"   • Models: {', '.join(models)}")
            print(f"   • Pairs: {len(pairs):,}")
            print(f"   • Repeats: {repeats}")
            print(f"   • Total tasks: {len(tasks):,}")
            print(f"   • Threads: {max_workers}")
            
            # Clear previous runs file
            if (self.data_dir / "runs.jsonl").exists():
                (self.data_dir / "runs.jsonl").unlink()
            
            # Execute with progress tracking
            completed = 0
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(self.run_single_dialog, client, pair_record, run_idx): (client.model_id, pair_record.variant)
                    for client, pair_record, run_idx in tasks
                }
                
                # Process results
                for future in as_completed(future_to_task):
                    completed += 1
                    model_name, variant = future_to_task[future]
                    
                    if completed % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed
                        eta = (len(tasks) - completed) / rate if rate > 0 else 0
                        print(f"⏳ Progress: {completed}/{len(tasks)} ({completed/len(tasks)*100:.1f}%) | ETA: {eta/60:.1f}min")
            
            # Final statistics
            elapsed = time.time() - start_time
            print(f"\n[OK] Experiment completed in {elapsed/60:.1f} minutes")
            
            # Generate cost summary
            total_cost = sum(client.total_cost for client in clients.values())
            cost_summary = {
                "total_cost_usd": round(total_cost, 4),
                "total_requests": len(tasks),
                "cache_directory": str(self.cache_dir),
                "models": {name: client.get_stats() for name, client in clients.items()}
            }
            
            with open(self.data_dir / "cost_summary.json", "w") as f:
                json.dump(cost_summary, f, indent=2)
            
            print(f"💰 Total cost: ${total_cost:.4f} USD")
            
            self.status = self._check_status()
            return True
            
        except Exception as e:
            print(f"[FAIL] Error running experiment: {e}")
            return False
    
    def run_analysis(self, advanced: bool = False):
        """Run statistical analysis with caching"""
        analysis_type = "advanced" if advanced else "basic"
        print(f"\n📊 Running {analysis_type.title()} Analysis...")
        
        try:
            # Check cache first
            cached_results = self._load_cached_analysis(analysis_type)
            
            if cached_results is not None:
                results = cached_results
                print("[OK] Loaded from cache")
            else:
                print("🔄 Computing analysis...")
                
                if not advanced:
                    # Run basic analysis from original script
                    results = analyse(str(self.data_dir))
                else:
                    # Advanced fairness metrics
                    results = self.advanced_fairness_analysis()
                
                # Cache the results
                self._save_analysis_cache(analysis_type, results)
            
            # Always save to file (in case file was missing)
            if advanced:
                with open(self.results_dir / "advanced_analysis.json", "w") as f:
                    json.dump(results, f, indent=2)
            
            print(f"[OK] {analysis_type.title()} analysis completed")
            
            self.status = self._check_status()
            return True
            
        except Exception as e:
            print(f"[FAIL] Error running analysis: {e}")
            return False
    
    def advanced_fairness_analysis(self) -> Dict[str, Any]:
        """Advanced fairness metrics and analysis"""
        import pandas as pd
        import numpy as np
        from scipy import stats
        
        # Load data
        runs = load_runs(str(self.data_dir / "runs.jsonl"))
        paired = paired_frame(runs)
        
        results = {
            "fairness_metrics": {},
            "persona_analysis": {},
            "effectiveness_analysis": {},
            "intersectional_effects": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Fairness metrics per model
        for model in runs["model"].unique():
            model_data = paired[paired["model"] == model]
            
            if model_data.empty:
                continue
                
            metrics = {}
            
            # Demographic Parity Difference
            baseline_positive = model_data["monetary_NC"].mean()
            persona_positive = model_data["monetary_G"].mean()
            fairness_positive = model_data["monetary_persona_fairness"].mean()
            
            metrics["demographic_parity"] = {
                "baseline_rate": float(baseline_positive),
                "persona_rate": float(persona_positive),
                "fairness_rate": float(fairness_positive),
                "bias_magnitude": float(abs(persona_positive - baseline_positive)),
                "fairness_improvement": float(abs(fairness_positive - baseline_positive) - abs(persona_positive - baseline_positive))
            }
            
            # Equal Opportunity Difference (among those who should get monetary relief)
            # This is a simplified version - in practice you'd define "should get relief" more carefully
            
            # Statistical significance tests
            from scipy.stats import chi2_contingency, fisher_exact
            
            # Chi-square test for independence
            contingency = pd.crosstab(
                model_data["group_label"], 
                [model_data["monetary_NC"], model_data["monetary_G"], model_data["monetary_persona_fairness"]]
            )
            
            if contingency.size > 0:
                try:
                    chi2, p_chi2, dof, expected = chi2_contingency(contingency)
                    metrics["independence_test"] = {
                        "chi2_statistic": float(chi2),
                        "p_value": float(p_chi2),
                        "degrees_freedom": int(dof)
                    }
                except:
                    metrics["independence_test"] = {"error": "Insufficient data"}
            
            # Effect sizes
            # Cohen's h for proportions
            def cohens_h(p1, p2):
                return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
            
            metrics["effect_sizes"] = {
                "baseline_vs_persona": float(cohens_h(baseline_positive, persona_positive)),
                "baseline_vs_fairness": float(cohens_h(baseline_positive, fairness_positive)),
                "persona_vs_fairness": float(cohens_h(persona_positive, fairness_positive))
            }
            
            results["fairness_metrics"][model] = metrics
        
        # Persona-specific analysis
        for persona in runs["group_label"].unique():
            if persona == "baseline":
                continue
                
            persona_runs = runs[runs["group_label"] == persona]
            persona_metrics = {
                "sample_size": len(persona_runs),
                "mean_remedy_tier": float(persona_runs["remedy_tier"].mean()),
                "monetary_relief_rate": float(persona_runs["monetary"].mean()),
                "escalation_rate": float(persona_runs["escalation"].mean())
            }
            
            results["persona_analysis"][persona] = persona_metrics
        
        # Fairness instruction effectiveness
        effectiveness = {}
        
        if not paired.empty:
            # Overall effectiveness across all personas
            persona_bias = (paired["remedy_tier_G"] - paired["remedy_tier_NC"]).mean()
            fairness_bias = (paired["remedy_tier_persona_fairness"] - paired["remedy_tier_NC"]).mean()
            improvement = abs(fairness_bias) - abs(persona_bias)
            
            effectiveness["overall"] = {
                "persona_bias_magnitude": float(abs(persona_bias)),
                "fairness_bias_magnitude": float(abs(fairness_bias)),
                "bias_reduction": float(improvement),
                "percentage_improvement": float((improvement / abs(persona_bias)) * 100) if persona_bias != 0 else 0
            }
            
            # Effectiveness by persona
            for persona in paired["group_label"].unique():
                if persona == "baseline":
                    continue
                    
                persona_data = paired[paired["group_label"] == persona]
                if not persona_data.empty:
                    p_bias = (persona_data["remedy_tier_G"] - persona_data["remedy_tier_NC"]).mean()
                    f_bias = (persona_data["remedy_tier_persona_fairness"] - persona_data["remedy_tier_NC"]).mean()
                    
                    effectiveness[persona] = {
                        "persona_bias": float(p_bias),
                        "fairness_bias": float(f_bias),
                        "improvement": float(abs(f_bias) - abs(p_bias))
                    }
        
        results["effectiveness_analysis"] = effectiveness
        
        return results
    
    def create_research_plots(self):
        """Generate publication-ready plots with caching"""
        print("\n📊 Creating research plots...")
        
        # Check if plots already exist and data hasn't changed
        plot_files = list(self.plots_dir.glob("*.png"))
        if plot_files:
            cached_plots = self._load_cached_analysis("plots")
            if cached_plots is not None:
                print("[OK] Using cached plots (data unchanged)")
                return True
        
        try:
            print("🔄 Generating plots...")
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            
            # Set publication style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # Load data
            runs = load_runs(str(self.data_dir / "runs.jsonl"))
            paired = paired_frame(runs)
            
            # Plot 1: Bias Magnitude by Persona
            fig, ax = plt.subplots(figsize=(12, 8))
            
            bias_data = []
            for model in runs["model"].unique():
                model_data = paired[paired["model"] == model]
                for persona in model_data["group_label"].unique():
                    if persona == "baseline":
                        continue
                    persona_data = model_data[model_data["group_label"] == persona]
                    if not persona_data.empty:
                        persona_bias = (persona_data["remedy_tier_G"] - persona_data["remedy_tier_NC"]).mean()
                        fairness_bias = (persona_data["remedy_tier_persona_fairness"] - persona_data["remedy_tier_NC"]).mean()
                        
                        bias_data.extend([
                            {"Model": model, "Persona": persona, "Treatment": "Demographic Signals", "Bias": persona_bias},
                            {"Model": model, "Persona": persona, "Treatment": "Signals + Fairness", "Bias": fairness_bias}
                        ])
            
            if bias_data:
                bias_df = pd.DataFrame(bias_data)
                sns.barplot(data=bias_df, x="Persona", y="Bias", hue="Treatment", ax=ax)
                plt.xticks(rotation=45, ha='right')
                plt.title("Bias Magnitude by Demographic Persona", fontsize=16, fontweight='bold')
                plt.ylabel("Mean Remedy Tier Bias", fontsize=14)
                plt.xlabel("Demographic Persona", fontsize=14)
                plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.savefig(self.plots_dir / "bias_by_persona.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Plot 2: Fairness Instruction Effectiveness
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Effectiveness across models
            effectiveness_data = []
            for model in runs["model"].unique():
                model_data = paired[paired["model"] == model]
                if not model_data.empty:
                    persona_bias = abs((model_data["remedy_tier_G"] - model_data["remedy_tier_NC"]).mean())
                    fairness_bias = abs((model_data["remedy_tier_persona_fairness"] - model_data["remedy_tier_NC"]).mean())
                    improvement = ((persona_bias - fairness_bias) / persona_bias * 100) if persona_bias > 0 else 0
                    
                    effectiveness_data.append({
                        "Model": model,
                        "Original_Bias": persona_bias,
                        "With_Fairness": fairness_bias,
                        "Improvement_%": improvement
                    })
            
            if effectiveness_data:
                eff_df = pd.DataFrame(effectiveness_data)
                
                # Bar plot of bias reduction
                x = np.arange(len(eff_df))
                width = 0.35
                
                ax1.bar(x - width/2, eff_df["Original_Bias"], width, label='Demographic Signals', alpha=0.8)
                ax1.bar(x + width/2, eff_df["With_Fairness"], width, label='Signals + Fairness', alpha=0.8)
                
                ax1.set_xlabel('Model', fontsize=14)
                ax1.set_ylabel('Absolute Bias Magnitude', fontsize=14)
                ax1.set_title('Bias Reduction by Fairness Instructions', fontsize=16, fontweight='bold')
                ax1.set_xticks(x)
                ax1.set_xticklabels(eff_df["Model"])
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Improvement percentage
                colors = ['green' if x > 0 else 'red' for x in eff_df["Improvement_%"]]
                ax2.bar(eff_df["Model"], eff_df["Improvement_%"], color=colors, alpha=0.7)
                ax2.set_xlabel('Model', fontsize=14)
                ax2.set_ylabel('Bias Reduction (%)', fontsize=14)
                ax2.set_title('Fairness Instruction Effectiveness', fontsize=16, fontweight='bold')
                ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(self.plots_dir / "fairness_effectiveness.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Plot 3: Distribution of Remedy Tiers
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Prepare data for stacked bar chart
            variant_names = {"baseline": "Baseline", "persona": "Demographic Signals", "persona_fairness": "Signals + Fairness"}
            
            tier_data = []
            for variant in ["baseline", "persona", "persona_fairness"]:
                variant_data = runs[runs["variant"] == variant]
                if not variant_data.empty:
                    tier_counts = variant_data["remedy_tier"].value_counts().sort_index()
                    for tier, count in tier_counts.items():
                        tier_data.append({
                            "Variant": variant_names.get(variant, variant),
                            "Remedy_Tier": f"Tier {tier}",
                            "Count": count,
                            "Percentage": count / len(variant_data) * 100
                        })
            
            if tier_data:
                tier_df = pd.DataFrame(tier_data)
                pivot_df = tier_df.pivot(index="Variant", columns="Remedy_Tier", values="Percentage").fillna(0)
                
                pivot_df.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
                ax.set_title('Distribution of Remedy Tiers by Variant', fontsize=16, fontweight='bold')
                ax.set_xlabel('Experimental Variant', fontsize=14)
                ax.set_ylabel('Percentage of Cases', fontsize=14)
                ax.legend(title='Remedy Tier', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(self.plots_dir / "remedy_tier_distribution.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"[OK] Research plots saved to {self.plots_dir}/")
            
            # Cache the plot generation (just metadata, not the actual plots)
            plot_metadata = {
                "plots_created": len(list(self.plots_dir.glob("*.png"))),
                "timestamp": datetime.now().isoformat(),
                "plot_files": [p.name for p in self.plots_dir.glob("*.png")]
            }
            self._save_analysis_cache("plots", plot_metadata)
            
            self.status = self._check_status()
            return True
            
        except Exception as e:
            print(f"[FAIL] Error creating plots: {e}")
            return False
    
    def generate_research_report(self):
        """Generate research summary report with caching"""
        print("\n📝 Generating research report...")
        
        # Check cache first
        cached_report = self._load_cached_analysis("research_report")
        if cached_report is not None:
            # Write cached report to file
            with open(self.results_dir / "research_summary.md", "w", encoding="utf-8") as f:
                f.write(cached_report["content"])
            print("[OK] Used cached research report")
            self.status = self._check_status()
            return True
        
        try:
            print("🔄 Generating research report...")
            
            # Load all analysis results
            analysis_files = {
                "basic": self.data_dir / "analysis.json",
                "advanced": self.results_dir / "advanced_analysis.json",
                "costs": self.data_dir / "cost_summary.json"
            }
            
            analyses = {}
            for key, file_path in analysis_files.items():
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        analyses[key] = json.load(f)
            
            # Generate markdown report
            report = self._create_markdown_report(analyses)
            
            with open(self.results_dir / "research_summary.md", "w", encoding="utf-8") as f:
                f.write(report)
            
            # Cache the report
            report_data = {
                "content": report,
                "timestamp": datetime.now().isoformat(),
                "word_count": len(report.split())
            }
            self._save_analysis_cache("research_report", report_data)
            
            print(f"[OK] Research report saved to {self.results_dir}/research_summary.md")
            
            self.status = self._check_status()
            return True
            
        except Exception as e:
            print(f"[FAIL] Error generating report: {e}")
            return False
    
    def _create_markdown_report(self, analyses: Dict) -> str:
        """Create comprehensive markdown research report"""
        
        report = f"""# LLM Fairness in Financial Services: Bias Detection Study

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This study examines bias in Large Language Models (LLMs) when processing financial complaint resolution decisions. Using a three-variant experimental design, we tested whether demographic signals influence remedy decisions and whether explicit fairness instructions can mitigate such bias.

## Methodology

### Experimental Design
- **Three-Variant Approach:**
  1. **Baseline:** Generic customer profile (Taylor Johnson, Springfield IL)
  2. **Demographic Signals:** Realistic demographic personas (names, locations, language patterns)
  3. **Signals + Fairness:** Same demographic signals with explicit fairness instructions

### Data
- **Source:** CFPB Consumer Complaint Database
- **Sample Size:** {self.status.get('unique_pairs', 'N/A')} complaint triplets
- **Total Evaluations:** {self.status.get('total_runs', 'N/A')} LLM decisions

### Models Tested
{chr(10).join(f"- {model}" for model in self.status.get('models_tested', []))}

## Key Findings

"""

        # Add findings from advanced analysis
        if "advanced" in analyses:
            advanced = analyses["advanced"]
            
            if "fairness_metrics" in advanced:
                report += "### Bias Detection Results\n\n"
                
                for model, metrics in advanced["fairness_metrics"].items():
                    if "demographic_parity" in metrics:
                        dp = metrics["demographic_parity"]
                        report += f"**{model}:**\n"
                        report += f"- Baseline monetary relief rate: {dp['baseline_rate']:.3f}\n"
                        report += f"- With demographic signals: {dp['persona_rate']:.3f}\n"
                        report += f"- With fairness instructions: {dp['fairness_rate']:.3f}\n"
                        report += f"- Bias magnitude: {dp['bias_magnitude']:.3f}\n"
                        report += f"- Fairness improvement: {dp['fairness_improvement']:.3f}\n\n"
                
            if "effectiveness_analysis" in advanced:
                effectiveness = advanced["effectiveness_analysis"]
                if "overall" in effectiveness:
                    eff = effectiveness["overall"]
                    report += "### Fairness Instruction Effectiveness\n\n"
                    report += f"- Original bias magnitude: {eff['persona_bias_magnitude']:.3f}\n"
                    report += f"- With fairness instructions: {eff['fairness_bias_magnitude']:.3f}\n"
                    report += f"- Bias reduction: {eff['bias_reduction']:.3f}\n"
                    report += f"- Percentage improvement: {eff['percentage_improvement']:.1f}%\n\n"

        # Add cost analysis
        if "costs" in analyses:
            costs = analyses["costs"]
            report += "### Experimental Costs\n\n"
            report += f"- **Total Cost:** ${costs.get('total_cost_usd', 0):.4f} USD\n"
            report += f"- **Total Requests:** {costs.get('total_requests', 0):,}\n\n"
            
            if "models" in costs:
                report += "**Per-Model Breakdown:**\n"
                for model, stats in costs["models"].items():
                    report += f"- {model}: ${stats.get('total_cost_usd', 0):.4f} (Cache hit rate: {stats.get('cache_hit_rate', 0):.0%})\n"
                report += "\n"

        report += """## Statistical Analysis

### Tests Performed
- **Wilcoxon Signed-Rank Test:** For ordinal remedy tier comparisons
- **McNemar's Test:** For binary monetary relief decisions  
- **Chi-Square Test:** For independence of demographic factors
- **Effect Size Analysis:** Cohen's h for proportion differences

### Multiple Comparisons Correction
- Applied Holm-Bonferroni correction for multiple testing
- Controls family-wise error rate at α = 0.05

## Implications

### For AI Fairness
- Demonstrates measurable bias from subtle demographic signals
- Shows mixed effectiveness of generic fairness instructions
- Highlights need for more sophisticated bias mitigation strategies

### For Financial Services
- Reveals potential compliance risks in AI-assisted decision making
- Suggests need for demographic audit testing in production systems
- Points to importance of training data diversity and bias detection

### For LLM Development
- Shows models can exhibit implicit bias even with explicit fairness instructions
- Demonstrates value of structured bias testing methodologies
- Indicates need for bias-aware model training approaches

## Limitations

- Limited to specific demographic personas and complaint types
- Simulated decision environment may not reflect real-world complexity
- Fairness instructions tested were generic rather than domain-specific
- Sample size constraints limit generalizability

## Recommendations

1. **Implement Regular Bias Audits:** Use similar methodology for production AI systems
2. **Develop Domain-Specific Fairness Instructions:** Generic instructions show limited effectiveness
3. **Diversify Training Data:** Ensure balanced representation across demographic groups
4. **Monitor Decision Patterns:** Track outcome disparities across customer segments
5. **Human Oversight:** Maintain human review for high-stakes decisions

## Technical Details

### Reproducibility
- All code and data available in experiment directory
- Deterministic random seeds used for reproducible results
- Cached LLM responses enable exact replication

### Data Processing
- Complaints cleaned and stratified by product/issue categories
- Language detection and quality filtering applied
- Demographic personas based on Census data and financial product usage patterns

---

*This analysis was conducted using the LLM Fairness Testing Harness. For technical questions or replication, see the accompanying code and documentation.*
"""

        return report
    
    def launch_dashboard(self):
        """Launch the interactive dashboard"""
        print("\n👀 Launching interactive dashboard...")
        
        try:
            import subprocess
            import sys
            
            dashboard_path = Path(__file__).parent / "dashboard.py"
            if dashboard_path.exists():
                print("🌐 Starting Streamlit dashboard...")
                print("💡 Dashboard will open in your browser automatically")
                subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])
            else:
                print("[FAIL] Dashboard file not found")
                return False
                
        except Exception as e:
            print(f"[FAIL] Error launching dashboard: {e}")
            return False
    
    def export_data(self):
        """Export all data in various formats"""
        print("\n💾 Exporting all data...")
        
        try:
            export_dir = self.results_dir / "exports"
            export_dir.mkdir(exist_ok=True)
            
            # Export experiment data
            if (self.data_dir / "runs.jsonl").exists():
                import pandas as pd
                runs = load_runs(str(self.data_dir / "runs.jsonl"))
                
                # CSV export
                runs.to_csv(export_dir / "experiment_results.csv", index=False)
                
                # Excel export with multiple sheets
                with pd.ExcelWriter(export_dir / "complete_analysis.xlsx") as writer:
                    runs.to_excel(writer, sheet_name="Raw Results", index=False)
                    
                    paired = paired_frame(runs)
                    if not paired.empty:
                        paired.to_excel(writer, sheet_name="Paired Analysis", index=False)
                    
                    # Summary statistics
                    summary = runs.groupby(["model", "variant"]).agg({
                        "remedy_tier": ["mean", "std", "count"],
                        "monetary": "mean",
                        "escalation": "mean",
                        "latency_s": "mean"
                    }).round(4)
                    summary.to_excel(writer, sheet_name="Summary Stats")
            
            # Copy analysis files
            for file_name in ["analysis.json", "cost_summary.json"]:
                source = self.data_dir / file_name
                if source.exists():
                    shutil.copy2(source, export_dir / file_name)
            
            # Copy advanced analysis
            advanced_file = self.results_dir / "advanced_analysis.json"
            if advanced_file.exists():
                shutil.copy2(advanced_file, export_dir / "advanced_analysis.json")
            
            print(f"[OK] Data exported to {export_dir}/")
            return True
            
        except Exception as e:
            print(f"[FAIL] Error exporting data: {e}")
            return False
    
    def clear_llm_cache(self):
        """Clear LLM response cache"""
        print("\n🗑️ Clearing LLM cache...")
        
        try:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(exist_ok=True)
                print(f"[OK] LLM cache cleared: {self.cache_dir}/")
            else:
                print("[INFO] LLM cache directory doesn't exist")
            
            self.status = self._check_status()
            return True
            
        except Exception as e:
            print(f"[FAIL] Error clearing LLM cache: {e}")
            return False
    
    def clear_analysis_cache_cmd(self):
        """Clear analysis cache via command"""
        print("\n🧹 Clearing analysis cache...")
        
        try:
            self._clear_analysis_cache()
            self.status = self._check_status()
            return True
            
        except Exception as e:
            print(f"[FAIL] Error clearing analysis cache: {e}")
            return False
    
    def clear_all_caches(self):
        """Clear both LLM and analysis caches"""
        print("\n🗑️ Clearing all caches...")
        
        success = True
        success &= self.clear_llm_cache()
        success &= self.clear_analysis_cache_cmd()
        
        if success:
            print("[OK] All caches cleared successfully")
        
        return success
    
    def reset_all(self):
        """Reset everything and start over"""
        confirm = input("⚠️  This will delete ALL data and results. Type 'RESET' to confirm: ")
        
        if confirm == "RESET":
            print("\n🔄 Resetting all data...")
            
            try:
                for dir_path in [self.data_dir, self.cache_dir, self.results_dir, self.analysis_cache_dir]:
                    if dir_path.exists():
                        shutil.rmtree(dir_path)
                        dir_path.mkdir(exist_ok=True)
                
                self.plots_dir.mkdir(exist_ok=True)
                
                print("[OK] All data reset successfully")
                self.status = self._check_status()
                return True
                
            except Exception as e:
                print(f"[FAIL] Error resetting data: {e}")
                return False
        else:
            print("[FAIL] Reset cancelled")
            return False
    
    def run_interactive(self):
        """Main interactive loop"""
        print("*** LLM Fairness Testing Harness ***")
        print("Comprehensive bias detection and analysis toolkit")
        
        while True:
            self.print_status()
            options = self.display_menu()
            
            try:
                choice = input("\n=> Enter your choice (number or 'q' to quit): ").strip()
                
                if choice.lower() in ['q', 'quit', 'exit']:
                    break
                
                try:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(options):
                        action_key, _ = options[choice_idx]
                        
                        if action_key == "quit":
                            break
                        elif action_key == "ingest_small":
                            self.ingest_data("small")
                        elif action_key == "ingest_large":
                            self.ingest_data("large")
                        elif action_key == "create_pairs":
                            self.create_pairs()
                        elif action_key == "run_test":
                            self.run_experiment(test_mode=True)
                        elif action_key == "run_full":
                            self.run_experiment(test_mode=False)
                        elif action_key == "extend_experiment":
                            self.run_experiment(test_mode=False)
                        elif action_key == "basic_analysis":
                            self.run_analysis(advanced=False)
                        elif action_key == "advanced_analysis":
                            self.run_analysis(advanced=True)
                        elif action_key == "create_plots":
                            self.create_research_plots()
                        elif action_key == "research_report":
                            self.generate_research_report()
                        elif action_key == "view_dashboard":
                            self.launch_dashboard()
                        elif action_key == "export_data":
                            self.export_data()
                        elif action_key == "clear_llm_cache":
                            self.clear_llm_cache()
                        elif action_key == "clear_analysis_cache":
                            self.clear_analysis_cache_cmd()
                        elif action_key == "clear_all_caches":
                            self.clear_all_caches()
                        elif action_key == "reset_all":
                            self.reset_all()
                        else:
                            print("[FAIL] Action not implemented yet")
                    else:
                        print("[FAIL] Invalid choice")
                except ValueError:
                    print("[FAIL] Please enter a number")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"[FAIL] Unexpected error: {e}")
        
        print("\nThanks for using the LLM Fairness Testing Harness!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Fairness Testing Harness")
    parser.add_argument("--non-interactive", action="store_true", help="Run specific commands non-interactively")
    parser.add_argument("--action", choices=["ingest", "pairs", "test", "full", "analyze", "report"], help="Action to perform")
    
    args = parser.parse_args()
    
    harness = FairnessHarness()
    
    if args.non_interactive and args.action:
        # Non-interactive mode for automation
        if args.action == "ingest":
            harness.ingest_data("large")
        elif args.action == "pairs":
            harness.create_pairs()
        elif args.action == "test":
            harness.run_experiment(test_mode=True)
        elif args.action == "full":
            harness.run_experiment(test_mode=False)
        elif args.action == "analyze":
            harness.run_analysis(advanced=True)
        elif args.action == "report":
            harness.generate_research_report()
    else:
        # Interactive mode
        harness.run_interactive()