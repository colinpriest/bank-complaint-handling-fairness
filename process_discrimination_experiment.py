#!/usr/bin/env python3
"""
Process Discrimination Experiment

This script tests whether the LLM shows process discrimination by examining
if it wants to ask for more information before making tier decisions, and
whether this varies systematically across different complaint types or demographics.
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


class ProcessDiscriminationExperiment:
    """Class to run process discrimination experiments"""

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
        """Get ground truth examples with additional metadata for analysis"""
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

            print(f"[INFO] Loaded {len(examples)} ground truth examples (tier >= 0)")
            return examples

        except Exception as e:
            print(f"[ERROR] Failed to load ground truth examples: {e}")
            return []

    def create_process_prompt(self, nshot_examples: List[Dict], target_complaint: str) -> Tuple[str, str]:
        """
        Create prompt that tests for process discrimination using NShotPromptGenerator
        """
        # Convert nshot_examples to the format expected by NShotPromptGenerator
        formatted_examples = []
        for example in nshot_examples:
            formatted_examples.append({
                'complaint_text': example['complaint_text'],
                'tier': example['tier']
            })

        # Create target case
        target_case = {
            'complaint_text': target_complaint
        }

        # Use NShotPromptGenerator with process-focused bias strategy
        system_prompt, user_prompt = self.prompt_generator.generate_prompts(
            target_case=target_case,
            nshot_examples=formatted_examples,
            persona=None,
            bias_strategy=BiasStrategy.CHAIN_OF_THOUGHT  # Encourages step-by-step thinking
        )

        return system_prompt, user_prompt

    def call_gpt4o_mini_process(self, system_prompt: str, user_prompt: str) -> Optional[ProcessAnalysis]:
        """
        Send prompt to GPT-4o-mini for process discrimination analysis
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

    def run_process_experiment(self, sample_size: int = 100):
        """
        Run process discrimination experiment
        """
        print("="*80)
        print("PROCESS DISCRIMINATION EXPERIMENT")
        print("="*80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Connect to database
        if not self.connect_to_database():
            return

        # Get ground truth examples
        print(f"\n[STEP 1] Loading {sample_size} ground truth examples...")
        ground_truth_examples = self.get_ground_truth_examples(sample_size)

        if not ground_truth_examples:
            print("[ERROR] No ground truth examples loaded")
            return

        examples_with_embeddings = [ex for ex in ground_truth_examples if ex['embedding'] is not None]
        print(f"[INFO] {len(examples_with_embeddings)} examples have embeddings")

        # Test different n values for process discrimination patterns
        print(f"\n[STEP 2] Testing process discrimination across n-shot configurations...")

        results = []
        n_values = [0, 5]  # Test zero-shot vs 5-shot

        for n in n_values:
            print(f"\n[INFO] Testing n = {n}")

            # Test on subset for analysis
            test_examples = examples_with_embeddings[:20]

            for i, target_example in enumerate(test_examples):
                print(f"  Processing example {i+1}/{len(test_examples)}")

                target_embedding = target_example['embedding']
                target_complaint = target_example['complaint_text']
                target_tier = target_example['tier']

                # Create candidate pool
                candidates = [ex for ex in examples_with_embeddings if ex['case_id'] != target_example['case_id']]

                if n == 0:
                    nshot_examples = []
                else:
                    if len(candidates) >= n:
                        candidate_embeddings = np.array([ex['embedding'] for ex in candidates])
                        selected_indices = self.optimizer.joint_optimization(
                            target_embedding, candidate_embeddings, n, lambda_param=0.5
                        )
                        nshot_examples = [candidates[idx] for idx in selected_indices]
                    else:
                        nshot_examples = candidates[:n] if candidates else []

                # Get process analysis
                system_prompt, user_prompt = self.create_process_prompt(nshot_examples, target_complaint)
                analysis = self.call_gpt4o_mini_process(system_prompt, user_prompt)

                if analysis:
                    result = {
                        'n': n,
                        'case_id': target_example['case_id'],
                        'ground_truth_tier': target_tier,
                        'predicted_tier': analysis.tier,
                        'confidence': analysis.confidence.value,
                        'reasoning': analysis.reasoning,
                        'information_needed': analysis.information_needed,
                        'product': target_example['product'],
                        'category': target_example['category'],
                        'correct_prediction': analysis.tier == target_tier,
                        'asks_for_info': analysis.confidence == DecisionConfidence.NEED_MORE_INFO
                    }
                    results.append(result)

        # Analyze results for process discrimination patterns
        self.analyze_process_discrimination(results)
        self.save_process_results(results)

    def analyze_process_discrimination(self, results: List[Dict]):
        """Analyze results for process discrimination patterns"""
        print(f"\n[STEP 3] Analyzing process discrimination patterns...")

        if not results:
            print("[ERROR] No results to analyze")
            return

        # Overall statistics
        total_results = len(results)
        asks_for_info_count = sum(1 for r in results if r['asks_for_info'])
        confident_count = sum(1 for r in results if r['confidence'] == 'confident')
        uncertain_count = sum(1 for r in results if r['confidence'] == 'uncertain')

        print(f"\nOVERALL STATISTICS:")
        print(f"  Total cases analyzed: {total_results}")
        print(f"  Asks for more info: {asks_for_info_count} ({asks_for_info_count/total_results*100:.1f}%)")
        print(f"  Confident decisions: {confident_count} ({confident_count/total_results*100:.1f}%)")
        print(f"  Uncertain decisions: {uncertain_count} ({uncertain_count/total_results*100:.1f}%)")

        # Analysis by n-shot configuration
        print(f"\nBY N-SHOT CONFIGURATION:")
        for n in sorted(set(r['n'] for r in results)):
            n_results = [r for r in results if r['n'] == n]
            n_asks_info = sum(1 for r in n_results if r['asks_for_info'])
            n_accuracy = sum(1 for r in n_results if r['correct_prediction']) / len(n_results)

            print(f"  n={n}: {len(n_results)} cases, {n_asks_info} ask for info ({n_asks_info/len(n_results)*100:.1f}%), accuracy: {n_accuracy:.3f}")

        # Analysis by product type
        print(f"\nBY PRODUCT TYPE:")
        products = {}
        for result in results:
            product = result['product']
            if product not in products:
                products[product] = []
            products[product].append(result)

        for product, prod_results in products.items():
            if len(prod_results) >= 2:  # Only show products with multiple cases
                asks_info = sum(1 for r in prod_results if r['asks_for_info'])
                accuracy = sum(1 for r in prod_results if r['correct_prediction']) / len(prod_results)
                print(f"  {product}: {len(prod_results)} cases, {asks_info} ask for info ({asks_info/len(prod_results)*100:.1f}%), accuracy: {accuracy:.3f}")

        # Analysis by tier
        print(f"\nBY GROUND TRUTH TIER:")
        for tier in [0, 1, 2]:
            tier_results = [r for r in results if r['ground_truth_tier'] == tier]
            if tier_results:
                asks_info = sum(1 for r in tier_results if r['asks_for_info'])
                accuracy = sum(1 for r in tier_results if r['correct_prediction']) / len(tier_results)
                print(f"  Tier {tier}: {len(tier_results)} cases, {asks_info} ask for info ({asks_info/len(tier_results)*100:.1f}%), accuracy: {accuracy:.3f}")

    def save_process_results(self, results: List[Dict]):
        """Save process discrimination results"""
        output_file = f"process_discrimination_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output_data = {
            'experiment_timestamp': datetime.now().isoformat(),
            'experiment_type': 'process_discrimination',
            'total_cases': len(results),
            'results': results,
            'summary': {
                'total_asks_for_info': sum(1 for r in results if r['asks_for_info']),
                'overall_accuracy': sum(1 for r in results if r['correct_prediction']) / len(results) if results else 0,
                'confidence_distribution': {
                    'confident': sum(1 for r in results if r['confidence'] == 'confident'),
                    'need_more_info': sum(1 for r in results if r['confidence'] == 'need_more_info'),
                    'uncertain': sum(1 for r in results if r['confidence'] == 'uncertain')
                }
            }
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n[INFO] Process discrimination results saved to: {output_file}")

    def close_connection(self):
        """Close database connection"""
        if hasattr(self, 'cursor'):
            self.cursor.close()
        if hasattr(self, 'connection'):
            self.connection.close()


def main():
    """Main function"""
    experiment = ProcessDiscriminationExperiment()

    try:
        experiment.run_process_experiment(sample_size=100)
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