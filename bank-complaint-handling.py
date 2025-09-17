#!/usr/bin/env python3
"""
Bank Complaint Handling Fairness Analysis - Main Script (Refactored)

This script brings together all experiments and analyses with PostgreSQL integration.
It handles database setup, ground truth data loading, and fairness analysis execution
using the NShotPromptGenerator infrastructure with DPP+k-NN configuration (n=1, k=4).

Usage:
    python bank-complaint-handling.py [options]

Requirements:
    - PostgreSQL installed and running
    - CFPB complaints data (complaints.csv)
    - Required Python packages (see requirements.txt)
    - OpenAI API key configured
"""

import sys
import os
import numpy as np
import psycopg2
from typing import List, Dict, Optional
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

from database_check import DatabaseCheck
from nshot_prompt_generator import NShotPromptGenerator, BiasStrategy
from hybrid_dpp import HybridDPP


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


class BankComplaintFairnessAnalyzer:
    """Main analyzer using NShotPromptGenerator with DPP+k-NN configuration"""

    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'fairness_analysis'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }

        # Initialize components using NShotPromptGenerator infrastructure
        self.prompt_generator = NShotPromptGenerator()

        # Configure DPP+k-NN parameters: n=1 (DPP), k=4 (k-NN)
        self.n_dpp = 1
        self.k_nn = 4
        self.total_examples = self.n_dpp + self.k_nn

        # Set up OpenAI client with Instructor for structured outputs
        openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.client = instructor.from_openai(openai_client)

    def setup_database(self):
        """Setup database using existing DatabaseCheck infrastructure"""
        print("="*80)
        print("BANK COMPLAINT HANDLING FAIRNESS ANALYSIS")
        print("="*80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Initialize database checker
        db_checker = DatabaseCheck()

        try:
            # Step 1: Check PostgreSQL installation
            if not db_checker.check_postgresql_installation():
                print("\n[CRITICAL] PostgreSQL is required but not found. Please install PostgreSQL and try again.")
                return False

            # Step 2: Check database existence and create if needed
            print(f"\n[INFO] Checking database '{db_checker.db_name}'...")

            if not db_checker.check_database_exists(db_checker.db_name):
                print(f"[INFO] Database '{db_checker.db_name}' does not exist")
                if not db_checker.create_database(db_checker.db_name):
                    print(f"\n[CRITICAL] Could not create database '{db_checker.db_name}'. Please check your PostgreSQL configuration.")
                    return False
            else:
                print(f"[INFO] Database '{db_checker.db_name}' already exists")

            # Step 3: Connect to database and setup tables
            if not db_checker.connect_to_database():
                print("\n[CRITICAL] Could not connect to database.")
                return False

            if not db_checker.setup_database_tables():
                print("\n[CRITICAL] Could not set up database tables.")
                return False

            # Step 4: Populate static tables
            print("\n[INFO] Setting up static tables...")

            if not db_checker.populate_personas():
                print("[WARNING] Could not populate personas table")

            if not db_checker.populate_mitigation_strategies():
                print("[WARNING] Could not populate mitigation strategies table")

            # Step 5: Check ground truth table
            print(f"\n[INFO] Checking ground truth table...")

            if not db_checker.check_ground_truth_table():
                print("[INFO] Ground truth table is empty or does not exist")
                if not db_checker.create_ground_truth_table():
                    print("[WARNING] Could not create or populate ground truth table")
                    print("[INFO] You can manually populate it later using CFPB data")
            else:
                print("[INFO] Ground truth table already has data")
                print("[INFO] Skipping recreation - using existing data")

            # Step 6: Check and setup LLM cache table
            print(f"\n[INFO] Checking LLM cache table...")
            db_checker.check_llm_cache_table()

            # Step 7: Check and generate vector embeddings
            db_checker.check_and_generate_embeddings()

            # Step 8: Check experiment table
            print(f"\n[INFO] Checking experiments table...")
            db_checker.check_experiment_table()

            # Step 9: Check nshot_optimisation table
            print(f"\n[INFO] Checking nshot_optimisation table...")
            db_checker.check_nshot_optimisation_table()

            # Summary
            print("\n" + "="*80)
            print("DATABASE SETUP COMPLETE")
            print("="*80)

            # Show table counts
            counts = db_checker.get_table_counts()

            print(f"Database: {db_checker.db_name}")
            print(f"Tables created:")
            print(f"  - personas: {counts.get('personas', 0)} records")
            print(f"  - mitigation_strategies: {counts.get('mitigation_strategies', 0)} records")
            print(f"  - ground_truth: {counts.get('ground_truth', 0):,} records")
            print(f"  - llm_cache: {counts.get('llm_cache', 0):,} records")
            print(f"  - experiments: {counts.get('experiments', 0):,} records")
            print(f"  - nshot_optimisation: {counts.get('nshot_optimisation', 0):,} records")

            if counts.get('ground_truth', 0) > 0:
                print(f"\n[SUCCESS] Database is ready for fairness analysis!")
                return True
            else:
                print(f"\n[INFO] Database setup complete, but ground truth data needs to be populated")
                print(f"[INFO] Place CFPB complaints.csv file in cfpb_downloads/ directory and run again")
                return False

        except Exception as e:
            print(f"\n[CRITICAL] Database setup failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            db_checker.close_connection()

    def connect_to_database(self):
        """Connect to PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            self.cursor = self.connection.cursor()
            print("[INFO] Connected to database for analysis")
            return True
        except Exception as e:
            print(f"[ERROR] Database connection failed: {e}")
            return False

    def get_ground_truth_examples(self, limit: int = 1000) -> List[Dict]:
        """Get ground truth examples with embeddings for DPP+k-NN selection"""
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

    def select_dpp_examples(self, candidates: List[Dict], n_dpp: int) -> List[int]:
        """Select n_dpp examples using HybridDPP-based diversity selection"""
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
        """Select k_nn examples using k-nearest neighbor selection based on cosine similarity"""
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
        """Combine DPP and k-NN example selection (n=1, k=4 configuration)"""
        if n_dpp == 0 and k_nn == 0:
            return []

        selected_examples = []
        used_indices = set()

        # First, select DPP examples for diversity (n=1)
        if n_dpp > 0:
            dpp_indices = self.select_dpp_examples(candidates, n_dpp)
            for idx in dpp_indices:
                if idx < len(candidates):
                    selected_examples.append(candidates[idx])
                    used_indices.add(idx)

        # Then, select k-NN examples for relevance (k=4, avoiding duplicates)
        if k_nn > 0:
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

    def call_gpt4o_mini_analysis(self, system_prompt: str, user_prompt: str, case_id: Optional[int] = None) -> Optional[ProcessAnalysis]:
        """Send prompt to GPT-4o-mini for analysis with caching (matching nshot optimization pattern)"""
        import hashlib

        # Create hash for caching (same pattern as nshot optimization)
        prompt_text = f"{system_prompt}\n\n{user_prompt}"
        request_hash = hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()

        try:
            # Check cache first
            self.cursor.execute("SELECT response_json FROM llm_cache WHERE request_hash = %s", (request_hash,))
            cached_result = self.cursor.fetchone()

            if cached_result:
                # Parse cached response
                try:
                    cached_json = json.loads(cached_result[0])
                    analysis = ProcessAnalysis(
                        tier=cached_json['tier'],
                        confidence=cached_json['confidence'],
                        reasoning=cached_json['reasoning'],
                        information_needed=cached_json.get('information_needed')
                    )
                    return analysis
                except Exception as e:
                    print(f"[WARNING] Failed to parse cached result: {e}")

            # Make API call if not cached
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

            # Cache the result (same pattern as nshot optimization)
            if analysis:
                try:
                    response_json = {
                        'tier': analysis.tier,
                        'confidence': analysis.confidence.value,
                        'reasoning': analysis.reasoning,
                        'information_needed': analysis.information_needed
                    }

                    insert_cache_sql = """
                        INSERT INTO llm_cache (
                            request_hash, model_name, temperature, prompt_text, system_prompt,
                            case_id, response_text, response_json, remedy_tier,
                            process_confidence, information_needed, asks_for_info, reasoning,
                            created_at
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                        )
                        ON CONFLICT (request_hash) DO NOTHING
                    """

                    self.cursor.execute(insert_cache_sql, (
                        request_hash,
                        "gpt-4o-mini",
                        0.0,
                        prompt_text,
                        system_prompt,
                        case_id,
                        analysis.reasoning,  # response_text
                        json.dumps(response_json),
                        analysis.tier,
                        analysis.confidence.value,
                        analysis.information_needed,
                        analysis.information_needed is not None,  # asks_for_info
                        analysis.reasoning
                    ))
                    self.connection.commit()

                except Exception as e:
                    print(f"[WARNING] Failed to cache result: {e}")

            return analysis
        except Exception as e:
            print(f"[ERROR] GPT-4o-mini call failed: {e}")
            return None

    def run_fairness_analysis(self, sample_size: int = 100):
        """Run fairness analysis using DPP+k-NN configuration with NShotPromptGenerator"""

        print("\n" + "="*80)
        print("FAIRNESS ANALYSIS WITH DPP+k-NN PROMPT GENERATION")
        print("="*80)
        print(f"Configuration: n_dpp={self.n_dpp}, k_nn={self.k_nn}, total_examples={self.total_examples}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Connect to database
        if not self.connect_to_database():
            return

        # Load ground truth examples
        print(f"\n[STEP 1] Loading ground truth examples...")
        ground_truth_examples = self.get_ground_truth_examples(limit=1000)

        if not ground_truth_examples:
            print("[ERROR] No ground truth examples loaded")
            return

        examples_with_embeddings = [ex for ex in ground_truth_examples if ex['embedding'] is not None]
        print(f"[INFO] {len(examples_with_embeddings)} examples have embeddings")

        # Run analysis on sample
        print(f"\n[STEP 2] Running fairness analysis on {sample_size} examples...")
        test_examples = examples_with_embeddings[:sample_size]

        correct_predictions = 0
        total_predictions = 0
        confident_decisions = 0
        uncertain_decisions = 0
        need_info_decisions = 0

        results = []

        for i, target_example in enumerate(test_examples):
            if i % 10 == 0:
                print(f"  Processing example {i+1}/{len(test_examples)}")

            target_embedding = target_example['embedding']
            target_complaint = target_example['complaint_text']
            target_tier = target_example['tier']

            # Create candidate pool (exclude target)
            candidates = [ex for ex in examples_with_embeddings if ex['case_id'] != target_example['case_id']]

            if len(candidates) < self.total_examples:
                continue

            # Select examples using DPP + k-NN combination (n=1, k=4)
            selected_examples = self.combine_dpp_knn_examples(
                target_embedding, candidates, self.n_dpp, self.k_nn
            )

            # Convert to format expected by NShotPromptGenerator
            nshot_examples = []
            for ex in selected_examples:
                nshot_examples.append({
                    'complaint_text': ex['complaint_text'],
                    'tier': ex['tier']
                })

            # Create target case
            target_case = {'complaint_text': target_complaint}

            # Generate prompts using NShotPromptGenerator infrastructure
            system_prompt, user_prompt = self.prompt_generator.generate_prompts(
                target_case=target_case,
                nshot_examples=nshot_examples,
                persona=None,  # Could add persona injection here
                bias_strategy=BiasStrategy.CHAIN_OF_THOUGHT
            )

            # Get prediction (include case_id for caching consistency)
            analysis = self.call_gpt4o_mini_analysis(system_prompt, user_prompt, target_example['case_id'])

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

                results.append({
                    'case_id': target_example['case_id'],
                    'target_tier': target_tier,
                    'predicted_tier': predicted_tier,
                    'correct': is_correct,
                    'confidence': analysis.confidence.value,
                    'reasoning': analysis.reasoning
                })

        # Calculate final metrics
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        confident_rate = confident_decisions / total_predictions if total_predictions > 0 else 0
        uncertain_rate = uncertain_decisions / total_predictions if total_predictions > 0 else 0
        need_info_rate = need_info_decisions / total_predictions if total_predictions > 0 else 0

        # Display results
        print(f"\n[STEP 3] Analysis Results...")
        print(f"="*50)
        print(f"Configuration: DPP+k-NN (n={self.n_dpp}, k={self.k_nn})")
        print(f"Sample size: {total_predictions} examples")
        print(f"Accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
        print(f"Confident decisions: {confident_rate:.3f} ({confident_decisions}/{total_predictions})")
        print(f"Uncertain decisions: {uncertain_rate:.3f} ({uncertain_decisions}/{total_predictions})")
        print(f"Need more info: {need_info_rate:.3f} ({need_info_decisions}/{total_predictions})")

        # Save results
        output_file = f"bank_complaint_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_data = {
            'experiment_timestamp': datetime.now().isoformat(),
            'configuration': {
                'n_dpp': self.n_dpp,
                'k_nn': self.k_nn,
                'total_examples': self.total_examples,
                'bias_strategy': 'chain_of_thought'
            },
            'metrics': {
                'accuracy': accuracy,
                'confident_rate': confident_rate,
                'uncertain_rate': uncertain_rate,
                'need_info_rate': need_info_rate,
                'sample_size': total_predictions
            },
            'results': results
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n[SUCCESS] Analysis complete! Results saved to: {output_file}")

    def close_connection(self):
        """Close database connection"""
        if hasattr(self, 'cursor'):
            self.cursor.close()
        if hasattr(self, 'connection'):
            self.connection.close()


def main():
    """Main function to run bank complaint handling setup and analysis"""
    analyzer = BankComplaintFairnessAnalyzer()

    try:
        # Step 1: Setup database
        print("Setting up database infrastructure...")
        if not analyzer.setup_database():
            print("[ERROR] Database setup failed - cannot proceed with analysis")
            sys.exit(1)

        # Step 2: Run fairness analysis with DPP+k-NN configuration
        print("\nRunning fairness analysis with DPP+k-NN configuration...")
        analyzer.run_fairness_analysis(sample_size=100)

    except KeyboardInterrupt:
        print(f"\n[INTERRUPT] Analysis interrupted by user")
    except Exception as e:
        print(f"\n[CRITICAL] Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        analyzer.close_connection()


if __name__ == "__main__":
    main()