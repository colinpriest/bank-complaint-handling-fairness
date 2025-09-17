#!/usr/bin/env python3
"""
N-Shot Optimization Experiment

This script implements the algorithm to find optimal n and alpha parameters
by testing them against real ground truth data and measuring LLM accuracy.
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

# Load environment variables
load_dotenv()

# Add project path
sys.path.insert(0, str(Path(__file__).parent))

from nshot_optimisation import NShotOptimisation
from database_check import DatabaseCheck
from nshot_prompt_generator import NShotPromptGenerator
import hashlib


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


class NShotOptimizationExperiment:
    """Class to run n-shot optimization experiments"""

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

    def get_ground_truth_examples(self, limit: int = 500) -> List[Dict]:
        """
        Step 1: Get the first 500 ground truth examples, filtering out where tier < 0
        """
        query = """
        SELECT case_id, consumer_complaint_text, simplified_ground_truth_tier, vector_embeddings
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
                case_id, complaint_text, tier, embedding_json = row

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
                    'embedding': embedding
                })

            print(f"[INFO] Loaded {len(examples)} ground truth examples (tier >= 0)")
            return examples

        except Exception as e:
            print(f"[ERROR] Failed to load ground truth examples: {e}")
            return []

    def create_prompt_with_examples(self, nshot_examples: List[Dict], target_complaint: str) -> Tuple[str, str]:
        """
        Create GPT-4o-mini prompt using the NShotPromptGenerator
        """
        # Convert format for prompt generator
        target_case = {'complaint_text': target_complaint}

        # Convert nshot examples format if needed
        formatted_examples = []
        for example in nshot_examples:
            formatted_examples.append({
                'complaint_text': example['complaint_text'],
                'tier': example['tier']
            })

        # Generate prompts using the new prompt generator
        system_prompt, user_prompt = self.prompt_generator.generate_prompts(
            target_case=target_case,
            nshot_examples=formatted_examples
        )

        return system_prompt, user_prompt

    def call_gpt4o_mini(self, system_prompt: str, user_prompt: str) -> Optional[ProcessAnalysis]:
        """
        Send prompt to GPT-4o-mini and get process discrimination analysis using Instructor
        """
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

    def run_experiment(self, sample_size: int = 500):
        """
        Main experiment runner following the pseudocode:

        Step 1: Get first 500 ground truth examples (tier >= 0)
        Step 2: For each alpha and n parameter pair
                  For each ground truth example
                    Find optimal nshot examples
                    Create prompt with nshot examples
                    Get GPT-4o-mini prediction
                Count accuracy rate
        Step 3: Find optimal n and alpha with highest accuracy
        """
        print("="*80)
        print("N-SHOT OPTIMIZATION EXPERIMENT")
        print("="*80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Connect to database
        if not self.connect_to_database():
            return

        # Step 1: Get ground truth examples
        print(f"\n[STEP 1] Loading {sample_size} ground truth examples...")
        ground_truth_examples = self.get_ground_truth_examples(sample_size)

        if not ground_truth_examples:
            print("[ERROR] No ground truth examples loaded")
            return

        # Check embeddings
        examples_with_embeddings = [ex for ex in ground_truth_examples if ex['embedding'] is not None]
        print(f"[INFO] {len(examples_with_embeddings)} examples have embeddings")

        if len(examples_with_embeddings) < 50:
            print("[ERROR] Not enough examples with embeddings for meaningful n-shot optimization")
            return

        # Step 2: Parameter grid search
        print(f"\n[STEP 2] Running parameter grid search...")

        # Parameter grids (full range as originally specified)
        n_grid = [0, 5, 6, 7, 8, 9, 10]  # Full n-shot range
        alpha_grid = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]  # Full alpha range (0.3 to 0.7 step 0.05)

        print(f"[INFO] n grid: {n_grid}")
        print(f"[INFO] alpha grid: {alpha_grid}")
        print(f"[INFO] Total combinations: {len(n_grid) * len(alpha_grid)}")

        results = []
        param_results = []

        for n in n_grid:
            print(f"\n[INFO] Testing n = {n}")

            # For n=0, alpha doesn't matter - only test alpha=0.5 once
            if n == 0:
                alpha_values = [0.5]  # Only test one alpha value for zero-shot
                print("[INFO] n=0: alpha parameter irrelevant, using alpha=0.5")
            else:
                alpha_values = alpha_grid

            for alpha in alpha_values:
                print(f"[INFO] Testing n={n}, alpha={alpha:.2f}")

                correct_predictions = 0
                total_predictions = 0

                # Test on subset of examples for meaningful accuracy metrics
                test_examples = examples_with_embeddings[:100]

                for i, target_example in enumerate(test_examples):
                    if i % 25 == 0:
                        print(f"  Processing example {i+1}/{len(test_examples)}")

                    target_embedding = target_example['embedding']
                    target_complaint = target_example['complaint_text']
                    target_tier = target_example['tier']

                    # Create candidate pool (all other examples)
                    candidates = [ex for ex in examples_with_embeddings if ex['case_id'] != target_example['case_id']]

                    if len(candidates) < n:
                        print(f"  [WARNING] Not enough candidates ({len(candidates)}) for n={n}")
                        continue

                    if n == 0:
                        # Zero-shot case
                        nshot_examples = []
                    else:
                        # Find optimal n-shot examples using joint optimization
                        candidate_embeddings = np.array([ex['embedding'] for ex in candidates])
                        selected_indices = self.optimizer.joint_optimization(
                            target_embedding, candidate_embeddings, n, lambda_param=alpha
                        )
                        nshot_examples = [candidates[idx] for idx in selected_indices]

                    # Create prompt
                    system_prompt, user_prompt = self.create_prompt_with_examples(nshot_examples, target_complaint)

                    # Get process analysis
                    analysis = self.call_gpt4o_mini(system_prompt, user_prompt)

                    if analysis is not None:
                        total_predictions += 1
                        if analysis.tier == target_tier:
                            correct_predictions += 1

                        # Store detailed result for database saving
                        result = {
                            'n': n,
                            'alpha': alpha,
                            'case_id': target_example['case_id'],
                            'ground_truth_tier': target_tier,
                            'predicted_tier': analysis.tier,
                            'confidence': analysis.confidence.value,
                            'reasoning': analysis.reasoning,
                            'information_needed': analysis.information_needed,
                            'asks_for_info': analysis.confidence == DecisionConfidence.NEED_MORE_INFO,
                            'correct_prediction': analysis.tier == target_tier,
                            'system_prompt': system_prompt,
                            'user_prompt': user_prompt
                        }
                        results.append(result)

                # Calculate accuracy for this parameter combination
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

                param_result = {
                    'n': n,
                    'alpha': alpha,
                    'correct': correct_predictions,
                    'total': total_predictions,
                    'accuracy': accuracy
                }
                param_results.append(param_result)
                print(f"  Result: {correct_predictions}/{total_predictions} = {accuracy:.3f} accuracy")

        # Step 3: Find optimal parameters
        print(f"\n[STEP 3] Finding optimal parameters...")

        if not param_results:
            print("[ERROR] No parameter results to analyze")
            return

        # Find best result
        best_result = max(param_results, key=lambda x: x['accuracy'])

        print(f"\n[SUCCESS] Optimal parameters found:")
        print(f"  n = {best_result['n']}")
        print(f"  alpha = {best_result['alpha']:.2f}")
        print(f"  Accuracy = {best_result['accuracy']:.3f} ({best_result['correct']}/{best_result['total']})")

        # Save results to database and file
        self.save_results_to_database(results)
        self.save_detailed_results_to_database(param_results, len(examples_with_embeddings))
        self.save_results(param_results, best_result)

        # Show top 5 results
        print(f"\nTop 5 parameter combinations:")
        sorted_results = sorted(param_results, key=lambda x: x['accuracy'], reverse=True)
        for i, result in enumerate(sorted_results[:5], 1):
            print(f"  {i}. n={result['n']}, alpha={result['alpha']:.2f}: {result['accuracy']:.3f} accuracy")

    def save_results_to_database(self, results: List[Dict]):
        """Save n-shot optimization results to database (nshot_optimisation table only)"""
        try:
            from database_check import DatabaseCheck

            # Create database checker to get session
            db_checker = DatabaseCheck()
            if not db_checker.connect_to_database():
                print("[ERROR] Could not connect to database for saving results")
                return

            print(f"[INFO] Saving n-shot optimization results to database...")

            # Save optimization summary only (not individual experiment records)
            self.save_optimization_results_to_database(results, db_checker)

            # Optionally save to LLM cache for future reuse
            self.save_to_llm_cache(results, db_checker)

            db_checker.close_connection()

        except Exception as e:
            print(f"[ERROR] Failed to save results to database: {e}")

    def save_to_llm_cache(self, results: List[Dict], db_checker):
        """Save LLM responses to cache for future reuse"""
        try:
            from sqlalchemy import text

            print(f"[INFO] Saving {len(results)} responses to LLM cache...")

            saved_count = 0
            for result in results:
                try:
                    # Create hash for caching
                    prompt_text = f"{result['system_prompt']}\n{result['user_prompt']}"
                    request_hash = hashlib.sha256(prompt_text.encode()).hexdigest()

                    # Check if already exists
                    check_query = text("SELECT id FROM llm_cache WHERE request_hash = :hash")
                    existing = db_checker.session.execute(check_query, {'hash': request_hash}).fetchone()

                    if not existing:
                        # Insert into LLM cache
                        cache_insert = text("""
                            INSERT INTO llm_cache (
                                request_hash, model_name, temperature, max_tokens,
                                prompt_text, system_prompt, case_id,
                                response_text, response_json, remedy_tier,
                                process_confidence, information_needed, asks_for_info, reasoning,
                                created_at
                            ) VALUES (
                                :request_hash, :model_name, :temperature, :max_tokens,
                                :prompt_text, :system_prompt, :case_id,
                                :response_text, :response_json, :remedy_tier,
                                :process_confidence, :information_needed, :asks_for_info, :reasoning,
                                NOW()
                            )
                        """)

                        response_json = json.dumps({
                            'tier': result['predicted_tier'],
                            'confidence': result['confidence'],
                            'reasoning': result['reasoning'],
                            'information_needed': result['information_needed']
                        })

                        db_checker.session.execute(cache_insert, {
                            'request_hash': request_hash,
                            'model_name': 'gpt-4o-mini',
                            'temperature': 0.0,
                            'max_tokens': None,
                            'prompt_text': prompt_text,
                            'system_prompt': result['system_prompt'],
                            'case_id': result['case_id'],
                            'response_text': f"Tier: {result['predicted_tier']}, Confidence: {result['confidence']}",
                            'response_json': response_json,
                            'remedy_tier': result['predicted_tier'],
                            'process_confidence': result['confidence'],
                            'information_needed': result['information_needed'],
                            'asks_for_info': result['asks_for_info'],
                            'reasoning': result['reasoning']
                        })

                        saved_count += 1

                except Exception as e:
                    print(f"[WARNING] Failed to cache result for case {result['case_id']}: {e}")
                    continue

            db_checker.session.commit()
            print(f"[SUCCESS] Saved {saved_count} new responses to LLM cache")

        except Exception as e:
            print(f"[ERROR] Failed to save to LLM cache: {e}")

    def save_optimization_results_to_database(self, results: List[Dict], db_checker):
        """Save n-shot optimization summary to database"""
        try:
            from sqlalchemy import text

            # Calculate metrics by parameter combination
            param_combinations = {}
            for result in results:
                key = (result['n'], result['alpha'])
                if key not in param_combinations:
                    param_combinations[key] = {
                        'correct': 0,
                        'total': 0,
                        'asks_for_info': 0,
                        'confident': 0,
                        'uncertain': 0,
                        'need_info': 0
                    }

                param_combinations[key]['total'] += 1
                if result['correct_prediction']:
                    param_combinations[key]['correct'] += 1
                if result['asks_for_info']:
                    param_combinations[key]['asks_for_info'] += 1

                confidence = result['confidence']
                if confidence == 'confident':
                    param_combinations[key]['confident'] += 1
                elif confidence == 'uncertain':
                    param_combinations[key]['uncertain'] += 1
                elif confidence == 'need_more_info':
                    param_combinations[key]['need_info'] += 1

            # Find best performing combination
            best_accuracy = 0
            best_params = None
            for (n, alpha), metrics in param_combinations.items():
                accuracy = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = (n, alpha, metrics)

            if best_params:
                n, alpha, metrics = best_params
                total = metrics['total']

                # Insert optimization result
                insert_query = text("""
                    INSERT INTO nshot_optimisation (
                        optimal_n, optimal_alpha, experiment_type, sample_size,
                        accuracy_score, information_request_rate,
                        confident_rate, uncertain_rate, need_info_rate,
                        created_at
                    ) VALUES (
                        :optimal_n, :optimal_alpha, :experiment_type, :sample_size,
                        :accuracy_score, :information_request_rate,
                        :confident_rate, :uncertain_rate, :need_info_rate,
                        NOW()
                    )
                """)

                db_checker.session.execute(insert_query, {
                    'optimal_n': n,
                    'optimal_alpha': alpha,
                    'experiment_type': 'process_discrimination',
                    'sample_size': total,
                    'accuracy_score': best_accuracy,
                    'information_request_rate': metrics['asks_for_info'] / total,
                    'confident_rate': metrics['confident'] / total,
                    'uncertain_rate': metrics['uncertain'] / total,
                    'need_info_rate': metrics['need_info'] / total
                })

                db_checker.session.commit()
                print(f"[SUCCESS] Saved optimization results: n={n}, alpha={alpha:.2f}, accuracy={best_accuracy:.3f}")

        except Exception as e:
            print(f"[ERROR] Failed to save optimization results: {e}")

    def save_results(self, results: List[Dict], best_result: Dict):
        """Save experiment results to file"""
        output_file = f"nshot_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output_data = {
            'experiment_timestamp': datetime.now().isoformat(),
            'best_parameters': best_result,
            'all_results': results,
            'summary': {
                'total_combinations_tested': len(results),
                'best_n': best_result['n'],
                'best_alpha': best_result['alpha'],
                'best_accuracy': best_result['accuracy']
            }
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n[INFO] Results saved to: {output_file}")

    def save_detailed_results_to_database(self, param_results: List[Dict], total_ground_truth: int):
        """Save detailed parameter combination results to nshot_optimization_results table"""
        try:
            import psycopg2

            db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', 5432)),
                'database': os.getenv('DB_NAME', 'fairness_analysis'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', '')
            }

            connection = psycopg2.connect(**db_config)
            cursor = connection.cursor()

            experiment_timestamp = datetime.now()

            print(f"[INFO] Saving {len(param_results)} detailed parameter results to database...")

            for result in param_results:
                # Calculate process discrimination metrics (if available)
                confident_decisions = result.get('confident_decisions', 0)
                uncertain_decisions = result.get('uncertain_decisions', 0)
                need_info_decisions = result.get('need_info_decisions', 0)

                total_decisions = result['total']
                confident_rate = confident_decisions / total_decisions if total_decisions > 0 else 0
                uncertain_rate = uncertain_decisions / total_decisions if total_decisions > 0 else 0
                need_info_rate = need_info_decisions / total_decisions if total_decisions > 0 else 0

                # Insert detailed result
                insert_query = """
                    INSERT INTO nshot_optimization_results (
                        experiment_timestamp, experiment_type, n_value, alpha_value,
                        sample_size, total_ground_truth_examples,
                        correct_predictions, total_predictions, accuracy_score,
                        confident_decisions, uncertain_decisions, need_info_decisions,
                        confident_rate, uncertain_rate, need_info_rate,
                        execution_time_seconds, notes, created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                    )
                """

                cursor.execute(insert_query, (
                    experiment_timestamp,
                    'accuracy',  # experiment_type
                    result['n'],  # n_value
                    result['alpha'],  # alpha_value
                    result['total'],  # sample_size
                    total_ground_truth,  # total_ground_truth_examples
                    result['correct'],  # correct_predictions
                    result['total'],  # total_predictions
                    result['accuracy'],  # accuracy_score
                    confident_decisions,
                    uncertain_decisions,
                    need_info_decisions,
                    confident_rate,
                    uncertain_rate,
                    need_info_rate,
                    None,  # execution_time_seconds (not tracked yet)
                    f"n={result['n']}, alpha={result['alpha']:.2f}"  # notes
                ))

            connection.commit()
            cursor.close()
            connection.close()

            print(f"[SUCCESS] Saved {len(param_results)} detailed parameter results")

        except Exception as e:
            print(f"[ERROR] Failed to save detailed results: {e}")

    def close_connection(self):
        """Close database connection"""
        if hasattr(self, 'cursor'):
            self.cursor.close()
        if hasattr(self, 'connection'):
            self.connection.close()


def main():
    """Main function"""
    experiment = NShotOptimizationExperiment()

    try:
        experiment.run_experiment(sample_size=500)
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