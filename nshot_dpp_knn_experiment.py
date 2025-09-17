#!/usr/bin/env python3
"""
N-Shot DPP + k-NN Optimization Experiment

This experiment finds optimal combinations of:
- n: Number of DPP (Determinantal Point Process) examples for diversity
- k: Number of k-NN (k Nearest Neighbor) examples for relevance

The approach combines diversity (DPP) and relevance (k-NN) selection strategies
to potentially outperform single-parameter alpha optimization.
"""

import sys
import os
import numpy as np
import psycopg2
from typing import List, Dict, Tuple, Optional
import openai
from pathlib import Path
import json
from datetime import datetime
from dotenv import load_dotenv
import instructor
from pydantic import BaseModel, Field
from enum import Enum
import time

# Load environment variables
load_dotenv()

# Add project path
sys.path.insert(0, str(Path(__file__).parent))

from nshot_optimisation import NShotOptimisation
from nshot_prompt_generator import NShotPromptGenerator, BiasStrategy


class DecisionConfidence(str, Enum):
    """Confidence levels for decision making"""
    CONFIDENT = "confident"
    NEED_MORE_INFO = "need_more_info"
    UNCERTAIN = "uncertain"


class ProcessAnalysis(BaseModel):
    """Pydantic model for process discrimination analysis"""
    tier: int = Field(..., description="Remedy tier: 0=No Action, 1=Non-Monetary Action, 2=Monetary Action", ge=0, le=2)
    confidence: DecisionConfidence = Field(..., description="Confidence in decision or need for more information")
    reasoning: str = Field(..., description="Brief explanation of decision and confidence level", max_length=1000)
    information_needed: Optional[str] = Field(None, description="If need_more_info, what specific information is needed?", max_length=150)


class NShotDPPKNNExperiment:
    """Class to run DPP + k-NN optimization experiments"""

    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'fairness_analysis'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }
        self.optimizer = NShotOptimisation(random_state=42)
        self.prompt_generator = NShotPromptGenerator()

        # Set up OpenAI client with Instructor for structured outputs
        openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.client = instructor.from_openai(openai_client)

    def connect_to_database(self):
        """Connect to PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            self.cursor = self.connection.cursor()
            print("[INFO] Connected to database")
            return True
        except Exception as e:
            print(f"[ERROR] Database connection failed: {e}")
            return False

    def create_dpp_knn_results_table(self):
        """Create table to store DPP + k-NN optimization results"""
        try:
            create_table_sql = """
                CREATE TABLE IF NOT EXISTS nshot_dpp_knn_results (
                    id SERIAL PRIMARY KEY,
                    experiment_timestamp TIMESTAMP NOT NULL,
                    experiment_type VARCHAR(50) DEFAULT 'dpp_knn',

                    -- Parameter combination
                    n_dpp INTEGER NOT NULL,  -- Number of DPP examples
                    k_nn INTEGER NOT NULL,   -- Number of k-NN examples
                    total_examples INTEGER NOT NULL,  -- n_dpp + k_nn

                    -- Sample information
                    sample_size INTEGER NOT NULL,
                    total_ground_truth_examples INTEGER,

                    -- Performance metrics
                    correct_predictions INTEGER NOT NULL,
                    total_predictions INTEGER NOT NULL,
                    accuracy_score FLOAT NOT NULL,

                    -- Process discrimination metrics
                    confident_decisions INTEGER DEFAULT 0,
                    uncertain_decisions INTEGER DEFAULT 0,
                    need_info_decisions INTEGER DEFAULT 0,
                    confident_rate FLOAT,
                    uncertain_rate FLOAT,
                    need_info_rate FLOAT,

                    -- Timing and metadata
                    execution_time_seconds FLOAT,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_dpp_knn_experiment ON nshot_dpp_knn_results(experiment_timestamp);
                CREATE INDEX IF NOT EXISTS idx_dpp_knn_params ON nshot_dpp_knn_results(n_dpp, k_nn);
                CREATE INDEX IF NOT EXISTS idx_dpp_knn_accuracy ON nshot_dpp_knn_results(accuracy_score);
                CREATE INDEX IF NOT EXISTS idx_dpp_knn_total ON nshot_dpp_knn_results(total_examples);
            """

            self.cursor.execute(create_table_sql)
            self.connection.commit()
            print("[SUCCESS] Created nshot_dpp_knn_results table")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to create table: {e}")
            return False

    def get_ground_truth_examples(self, limit: int = 500) -> List[Dict]:
        """Get ground truth examples with embeddings"""
        query = """
        SELECT case_id, consumer_complaint_text, simplified_ground_truth_tier,
               vector_embeddings, product, complaint_category
        FROM ground_truth
        WHERE simplified_ground_truth_tier >= 0
        AND vector_embeddings IS NOT NULL
        AND vector_embeddings != ''
        ORDER BY case_id
        LIMIT %s
        """

        try:
            self.cursor.execute(query, (limit,))
            results = self.cursor.fetchall()

            examples = []
            for row in results:
                case_id, complaint_text, tier, embedding_json, product, category = row

                # Convert JSON string to numpy array
                if embedding_json:
                    try:
                        embedding_list = json.loads(embedding_json)
                        embedding = np.array(embedding_list, dtype=np.float32)
                    except:
                        embedding = None
                else:
                    embedding = None

                examples.append({
                    'case_id': case_id,
                    'complaint_text': complaint_text,
                    'tier': tier,
                    'embedding': embedding,
                    'product': product or 'Unknown',
                    'category': category or 'Unknown'
                })

            print(f"[INFO] Loaded {len(examples)} ground truth examples with embeddings")
            return examples

        except Exception as e:
            print(f"[ERROR] Failed to load ground truth examples: {e}")
            return []

    def select_dpp_examples(self, candidates: List[Dict], n_dpp: int, lambda_diversity: float = 0.8) -> List[int]:
        """Select n_dpp examples using HybridDPP-based diversity selection."""
        if n_dpp == 0 or len(candidates) == 0:
            return []

        indexed_candidates = [
            (idx, ex) for idx, ex in enumerate(candidates) if ex.get('embedding') is not None
        ]
        if not indexed_candidates:
            return []

        embeddings = np.array([ex['embedding'] for _, ex in indexed_candidates], dtype=np.float32)
        selector = HybridDPP(embeddings, random_state=42)
        num_to_select = min(n_dpp, len(indexed_candidates))
        local_indices = selector.select(num_to_select)
        return [indexed_candidates[i][0] for i in local_indices]

    def select_knn_examples(self, query_embedding: np.ndarray, candidates: List[Dict], k_nn: int) -> List[int]:
        """
        Select k_nn examples using k-nearest neighbor selection based on cosine similarity
        """
        if k_nn == 0 or len(candidates) == 0:
            return []

        candidate_embeddings = np.array([ex['embedding'] for ex in candidates])

        # Calculate cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_embedding.reshape(1, -1), candidate_embeddings).flatten()

        # Get top k_nn most similar
        top_k_indices = np.argsort(similarities)[-k_nn:][::-1]  # Descending order

        return top_k_indices.tolist()

    def combine_dpp_knn_examples(self, query_embedding: np.ndarray, candidates: List[Dict],
                                n_dpp: int, k_nn: int) -> List[Dict]:
        """
        Combine DPP and k-NN example selection
        """
        if n_dpp == 0 and k_nn == 0:
            return []

        selected_examples = []
        used_indices = set()

        # First, select DPP examples for diversity
        if n_dpp > 0:
            dpp_indices = self.select_dpp_examples(candidates, n_dpp)
            for idx in dpp_indices:
                if idx < len(candidates):  # Safety check
                    selected_examples.append(candidates[idx])
                    used_indices.add(idx)

        # Then, select k-NN examples for relevance (avoiding duplicates)
        if k_nn > 0:
            # Filter out already selected candidates
            remaining_candidates = [candidates[i] for i in range(len(candidates)) if i not in used_indices]

            if remaining_candidates:
                knn_indices = self.select_knn_examples(query_embedding, remaining_candidates, k_nn)

                # Map back to original indices
                remaining_map = {new_idx: orig_idx for new_idx, orig_idx in enumerate(range(len(candidates))) if orig_idx not in used_indices}

                for new_idx in knn_indices:
                    if new_idx in remaining_map:
                        orig_idx = remaining_map[new_idx]
                        selected_examples.append(candidates[orig_idx])

        return selected_examples

    def call_gpt4o_mini_analysis(self, system_prompt: str, user_prompt: str) -> Optional[ProcessAnalysis]:
        """Send prompt to GPT-4o-mini for analysis"""
        try:
            analysis = self.client.chat.completions.create(
                model="gpt-4o-mini",
                response_model=ProcessAnalysis,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_retries=2
            )
            return analysis
        except Exception as e:
            print(f"[ERROR] GPT-4o-mini call failed: {e}")
            return None

    def run_dpp_knn_optimization(self, sample_size: int = 500):
        """Run DPP + k-NN optimization experiment"""

        print("=" * 80)
        print("N-SHOT DPP + K-NN OPTIMIZATION EXPERIMENT")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Connect to database
        if not self.connect_to_database():
            return

        # Create results table
        if not self.create_dpp_knn_results_table():
            return

        # Load ground truth examples
        print(f"\n[STEP 1] Loading {sample_size} ground truth examples...")
        ground_truth_examples = self.get_ground_truth_examples(sample_size)

        if not ground_truth_examples:
            print("[ERROR] No ground truth examples loaded")
            return

        examples_with_embeddings = [ex for ex in ground_truth_examples if ex['embedding'] is not None]
        print(f"[INFO] {len(examples_with_embeddings)} examples have embeddings")

        # Parameter grids
        n_dpp_grid = list(range(0, 11))  # 0 to 10 DPP examples
        k_nn_grid = list(range(0, 11))   # 0 to 10 k-NN examples

        print(f"\n[STEP 2] Running DPP + k-NN parameter grid search...")
        print(f"[INFO] n_dpp grid: {n_dpp_grid}")
        print(f"[INFO] k_nn grid: {k_nn_grid}")
        print(f"[INFO] Total combinations: {len(n_dpp_grid) * len(k_nn_grid)}")

        param_results = []
        experiment_timestamp = datetime.now()

        # Test on subset for meaningful accuracy metrics
        test_examples = examples_with_embeddings[:100]

        combination_count = 0
        total_combinations = len(n_dpp_grid) * len(k_nn_grid)

        for n_dpp in n_dpp_grid:
            for k_nn in k_nn_grid:
                combination_count += 1
                total_shots = n_dpp + k_nn

                print(f"\n[INFO] Testing combination {combination_count}/{total_combinations}: n_dpp={n_dpp}, k_nn={k_nn} (total={total_shots})")

                start_time = time.time()
                correct_predictions = 0
                total_predictions = 0

                confident_decisions = 0
                uncertain_decisions = 0
                need_info_decisions = 0

                for i, target_example in enumerate(test_examples):
                    if i % 25 == 0:
                        print(f"  Processing example {i+1}/{len(test_examples)}")

                    target_embedding = target_example['embedding']
                    target_complaint = target_example['complaint_text']
                    target_tier = target_example['tier']

                    # Create candidate pool (exclude target)
                    candidates = [ex for ex in examples_with_embeddings if ex['case_id'] != target_example['case_id']]

                    if len(candidates) < total_shots:
                        continue

                    # Select examples using DPP + k-NN combination
                    selected_examples = self.combine_dpp_knn_examples(
                        target_embedding, candidates, n_dpp, k_nn
                    )

                    # Convert to format expected by prompt generator
                    nshot_examples = []
                    for ex in selected_examples:
                        nshot_examples.append({
                            'complaint_text': ex['complaint_text'],
                            'tier': ex['tier']
                        })

                    # Create target case
                    target_case = {'complaint_text': target_complaint}

                    # Generate prompts
                    system_prompt, user_prompt = self.prompt_generator.generate_prompts(
                        target_case=target_case,
                        nshot_examples=nshot_examples,
                        persona=None,
                        bias_strategy=BiasStrategy.CHAIN_OF_THOUGHT
                    )

                    # Get prediction
                    analysis = self.call_gpt4o_mini_analysis(system_prompt, user_prompt)

                    if analysis:
                        predicted_tier = analysis.tier
                        is_correct = predicted_tier == target_tier

                        if is_correct:
                            correct_predictions += 1
                        total_predictions += 1

                        # Track confidence levels
                        if analysis.confidence == DecisionConfidence.CONFIDENT:
                            confident_decisions += 1
                        elif analysis.confidence == DecisionConfidence.UNCERTAIN:
                            uncertain_decisions += 1
                        elif analysis.confidence == DecisionConfidence.NEED_MORE_INFO:
                            need_info_decisions += 1

                execution_time = time.time() - start_time
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

                # Calculate confidence rates
                confident_rate = confident_decisions / total_predictions if total_predictions > 0 else 0
                uncertain_rate = uncertain_decisions / total_predictions if total_predictions > 0 else 0
                need_info_rate = need_info_decisions / total_predictions if total_predictions > 0 else 0

                # Store detailed results
                param_result = {
                    'n_dpp': n_dpp,
                    'k_nn': k_nn,
                    'total_shots': total_shots,
                    'correct': correct_predictions,
                    'total': total_predictions,
                    'accuracy': accuracy,
                    'confident_decisions': confident_decisions,
                    'uncertain_decisions': uncertain_decisions,
                    'need_info_decisions': need_info_decisions,
                    'confident_rate': confident_rate,
                    'uncertain_rate': uncertain_rate,
                    'need_info_rate': need_info_rate,
                    'execution_time': execution_time
                }
                param_results.append(param_result)

                print(f"  Result: {correct_predictions}/{total_predictions} = {accuracy:.3f} accuracy ({execution_time:.1f}s)")

                # Save to database immediately
                self.save_dpp_knn_result(param_result, experiment_timestamp, len(examples_with_embeddings), len(test_examples))

        # Find and display best results
        print(f"\n[STEP 3] Analysis of DPP + k-NN optimization results...")

        if param_results:
            # Best overall accuracy
            best_result = max(param_results, key=lambda x: x['accuracy'])
            print(f"\n[SUCCESS] Best parameters found:")
            print(f"  n_dpp = {best_result['n_dpp']}")
            print(f"  k_nn = {best_result['k_nn']}")
            print(f"  total_shots = {best_result['total_shots']}")
            print(f"  Accuracy = {best_result['accuracy']:.3f} ({best_result['correct']}/{best_result['total']})")

            # Top 5 results
            print(f"\nTop 5 DPP + k-NN combinations:")
            sorted_results = sorted(param_results, key=lambda x: x['accuracy'], reverse=True)
            for i, result in enumerate(sorted_results[:5], 1):
                print(f"  {i}. n_dpp={result['n_dpp']}, k_nn={result['k_nn']}: {result['accuracy']:.3f} accuracy")

            # Save summary results
            self.save_dpp_knn_summary(experiment_timestamp, best_result, len(param_results))

        print(f"\n[SUCCESS] DPP + k-NN optimization complete!")
        print(f"[INFO] Results saved to nshot_dpp_knn_results table")

    def save_dpp_knn_result(self, result: Dict, experiment_timestamp: datetime,
                           total_ground_truth: int, sample_size: int):
        """Save individual DPP + k-NN result to database"""
        try:
            insert_query = """
                INSERT INTO nshot_dpp_knn_results (
                    experiment_timestamp, experiment_type, n_dpp, k_nn, total_examples,
                    sample_size, total_ground_truth_examples,
                    correct_predictions, total_predictions, accuracy_score,
                    confident_decisions, uncertain_decisions, need_info_decisions,
                    confident_rate, uncertain_rate, need_info_rate,
                    execution_time_seconds, notes, created_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                )
            """

            self.cursor.execute(insert_query, (
                experiment_timestamp,
                'dpp_knn',
                result['n_dpp'],
                result['k_nn'],
                result['total_shots'],
                sample_size,
                total_ground_truth,
                result['correct'],
                result['total'],
                result['accuracy'],
                result['confident_decisions'],
                result['uncertain_decisions'],
                result['need_info_decisions'],
                result['confident_rate'],
                result['uncertain_rate'],
                result['need_info_rate'],
                result['execution_time'],
                f"DPP={result['n_dpp']}, k-NN={result['k_nn']}, total={result['total_shots']}"
            ))

            self.connection.commit()

        except Exception as e:
            print(f"[WARNING] Failed to save result for n_dpp={result['n_dpp']}, k_nn={result['k_nn']}: {e}")

    def save_dpp_knn_summary(self, experiment_timestamp: datetime, best_result: Dict, total_combinations: int):
        """Save experiment summary to JSON file"""
        output_file = f"dpp_knn_optimization_results_{experiment_timestamp.strftime('%Y%m%d_%H%M%S')}.json"

        output_data = {
            'experiment_timestamp': experiment_timestamp.isoformat(),
            'experiment_type': 'dpp_knn_optimization',
            'total_combinations_tested': total_combinations,
            'best_parameters': {
                'n_dpp': best_result['n_dpp'],
                'k_nn': best_result['k_nn'],
                'total_shots': best_result['total_shots'],
                'accuracy': best_result['accuracy'],
                'confident_rate': best_result['confident_rate'],
                'execution_time': best_result['execution_time']
            },
            'parameter_ranges': {
                'n_dpp_range': '0-10',
                'k_nn_range': '0-10',
                'total_combinations': 121
            }
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"[INFO] Summary saved to: {output_file}")

    def close_connection(self):
        """Close database connection"""
        if hasattr(self, 'cursor'):
            self.cursor.close()
        if hasattr(self, 'connection'):
            self.connection.close()


def main():
    """Main function"""
    experiment = NShotDPPKNNExperiment()

    try:
        experiment.run_dpp_knn_optimization(sample_size=500)
    except KeyboardInterrupt:
        print(f"\n[INTERRUPT] Experiment interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Experiment failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        experiment.close_connection()


if __name__ == "__main__":
    main()