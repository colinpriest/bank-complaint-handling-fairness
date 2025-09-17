#!/usr/bin/env python3
"""
Multithreaded N-Shot Optimization Experiments

This module provides multithreaded versions of both optimization approaches:
1. Single Alpha Optimization (n + alpha grid search)
2. DPP + k-NN Optimization (n_dpp + k_nn grid search)

Uses ThreadPoolExecutor for parallel parameter combination testing.
"""

import sys
import os
import numpy as np
import psycopg2

# Set matplotlib backend before importing pyplot to avoid TkInter threading issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Callable
import openai
from pathlib import Path
import json
from datetime import datetime
from dotenv import load_dotenv
import instructor
from pydantic import BaseModel, Field
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue

# Load environment variables
load_dotenv()

# Add project path
sys.path.insert(0, str(Path(__file__).parent))

from nshot_optimisation import NShotOptimisation
from nshot_prompt_generator import NShotPromptGenerator, BiasStrategy
from hybrid_similarity import HybridSimilarityCalculator, HybridKNNSelector, HybridSimilarityPresets
from hybrid_dpp import HybridDPP


def get_product_category_group(product: str) -> str:
    """
    Map detailed product categories to 5 major groups:
    1. Credit Reporting - credit reports, credit repair
    2. Banking - checking, savings, money transfers, prepaid cards
    3. Lending - mortgages, credit cards, student loans, auto loans, personal loans
    4. Debt Collection - debt collection services
    5. Other - everything else
    """
    if not product:
        return "Other"

    product_lower = product.lower().strip()

    # Credit Reporting (65% of data)
    if 'credit report' in product_lower or 'credit repair' in product_lower:
        return "Credit Reporting"

    # Debt Collection
    elif 'debt collection' in product_lower:
        return "Debt Collection"

    # Banking Services
    elif any(term in product_lower for term in [
        'checking', 'savings', 'account', 'bank account',
        'money transfer', 'virtual currency', 'money service',
        'prepaid card', 'money transfers'
    ]):
        return "Banking"

    # Lending Products
    elif any(term in product_lower for term in [
        'mortgage', 'credit card', 'student loan', 'vehicle loan',
        'payday loan', 'title loan', 'personal loan', 'consumer loan',
        'advance loan', 'lease'
    ]):
        return "Lending"

    # Other/Miscellaneous
    else:
        return "Other"


def get_issue_category_group(issue: str) -> str:
    """
    Map detailed issue categories to 5 major groups:
    1. Credit Report Issues - incorrect info, improper use, investigation problems
    2. Debt Collection Issues - collection attempts, communication, false statements
    3. Account Management - managing accounts, transactions, payment issues
    4. Fraud/Disputes - fraud, scams, billing disputes
    5. Other - everything else
    """
    if not issue:
        return "Other"

    issue_lower = issue.lower().strip()

    # Credit Report Issues (~65% of data)
    if any(term in issue_lower for term in [
        'incorrect information', 'improper use', 'investigation into',
        'credit report', 'credit reporting'
    ]):
        return "Credit Report Issues"

    # Debt Collection Issues
    elif any(term in issue_lower for term in [
        'collect debt', 'debt not owed', 'notification about debt',
        'false statements', 'representation', 'negative or legal action',
        'communication tactics'
    ]):
        return "Debt Collection Issues"

    # Account Management
    elif any(term in issue_lower for term in [
        'managing', 'account', 'payment process', 'transaction',
        'lender', 'servicer', 'struggling to pay', 'loan', 'lease'
    ]):
        return "Account Management"

    # Fraud/Disputes
    elif any(term in issue_lower for term in [
        'fraud', 'scam', 'purchase shown', 'billing', 'dispute',
        'charging your account'
    ]):
        return "Fraud/Disputes"

    # Other
    else:
        return "Other"


def get_combined_category_key(product: str, issue: str) -> str:
    """Get combined product+issue category key for statistics"""
    product_group = get_product_category_group(product)
    issue_group = get_issue_category_group(issue)
    return f"{product_group} + {issue_group}"


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


class MultithreadedNShotOptimizer:
    """Base class for multithreaded n-shot optimization experiments"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'fairness_analysis'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }
        self.optimizer = NShotOptimisation(random_state=42)
        self.prompt_generator = NShotPromptGenerator()

        # Thread-local storage for database connections
        self.thread_local = threading.local()

        # Cache for category statistics to avoid repeated calculations
        self.category_stats_cache = {}
        self.cache_lock = threading.Lock()

        # Set up OpenAI client with Instructor for structured outputs
        openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.client = instructor.from_openai(openai_client)

        print(f"[INFO] Initialized multithreaded optimizer with {max_workers} workers")

    def get_thread_db_connection(self):
        """Get thread-local database connection"""
        if not hasattr(self.thread_local, 'connection'):
            self.thread_local.connection = psycopg2.connect(**self.db_config)
            self.thread_local.cursor = self.thread_local.connection.cursor()
        return self.thread_local.connection, self.thread_local.cursor

    def close_thread_db_connection(self):
        """Close thread-local database connection"""
        if hasattr(self.thread_local, 'cursor'):
            self.thread_local.cursor.close()
        if hasattr(self.thread_local, 'connection'):
            self.thread_local.connection.close()

    def precompute_similarities(self, test_examples: List[Dict], all_examples: List[Dict]) -> np.ndarray:
        """Pre-compute cosine similarities between test and candidate examples"""
        print("[INFO] Pre-computing cosine similarities...")

        # Extract embeddings
        test_embeddings = np.array([ex['embedding'] for ex in test_examples])
        candidate_embeddings = np.array([ex['embedding'] for ex in all_examples])

        # Compute all pairwise similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(test_embeddings, candidate_embeddings)

        print(f"[SUCCESS] Pre-computed {similarities.shape[0]}x{similarities.shape[1]} similarity matrix")
        return similarities

    def load_stratified_sample(self, target_size: int = 1000, min_per_category: int = 5) -> List[Dict]:
        """Load stratified sample of examples by product+issue categories"""
        connection = psycopg2.connect(**self.db_config)
        cursor = connection.cursor()

        print(f"[INFO] Loading stratified sample of {target_size} examples with min {min_per_category} per category...")

        # First, get distribution of product+issue combinations
        category_query = """
            SELECT product, issue, COUNT(*) as count
            FROM ground_truth
            WHERE simplified_ground_truth_tier >= 0
            AND vector_embeddings IS NOT NULL
            AND vector_embeddings != ''
            GROUP BY product, issue
            HAVING COUNT(*) >= %s
            ORDER BY count DESC
        """

        cursor.execute(category_query, (min_per_category,))
        categories = cursor.fetchall()

        # Filter out categories with too few examples
        print(f"[INFO] Found {len(categories)} categories with at least {min_per_category} examples")

        total_count = sum(count for _, _, count in categories)

        # Calculate samples per category (proportional to size, but ensure minimum)
        all_examples = []
        samples_allocated = 0

        for product, issue, count in categories:
            # Ensure we get at least min_per_category examples from each category
            proportion = count / total_count
            samples_for_category = max(min_per_category, int(target_size * proportion))

            # Adjust last category to reach exact target
            if samples_allocated + samples_for_category > target_size:
                samples_for_category = max(min_per_category, target_size - samples_allocated)

            # Sample from this category
            sample_query = """
                SELECT case_id, consumer_complaint_text, simplified_ground_truth_tier,
                       vector_embeddings, product, sub_product, issue, sub_issue, complaint_category
                FROM ground_truth
                WHERE simplified_ground_truth_tier >= 0
                AND vector_embeddings IS NOT NULL
                AND vector_embeddings != ''
                AND product = %s
                AND issue = %s
                ORDER BY RANDOM()
                LIMIT %s
            """

            cursor.execute(sample_query, (product, issue, samples_for_category))
            category_results = cursor.fetchall()

            for row in category_results:
                case_id, complaint_text, tier, embedding_json, prod, sub_prod, iss, sub_iss, category = row

                if embedding_json:
                    try:
                        embedding_list = json.loads(embedding_json)
                        embedding = np.array(embedding_list, dtype=np.float32)
                    except:
                        embedding = None
                else:
                    embedding = None

                if embedding is not None:
                    all_examples.append({
                        'case_id': case_id,
                        'complaint_text': complaint_text,
                        'tier': tier,
                        'embedding': embedding,
                        'product': prod,
                        'sub_product': sub_prod,
                        'issue': iss,
                        'sub_issue': sub_iss,
                        'complaint_category': category
                    })

            samples_allocated += samples_for_category
            if samples_allocated >= target_size:
                break

        cursor.close()
        connection.close()

        print(f"[SUCCESS] Loaded {len(all_examples)} stratified examples covering {len(categories)} product+issue categories")
        return all_examples

    def load_ground_truth_examples(self, limit: int = None, stratified_sample: int = None) -> List[Dict]:
        """Load ground truth examples with embeddings (main thread only)"""
        # Use stratified sampling if requested
        if stratified_sample is not None:
            return self.load_stratified_sample(stratified_sample)

        connection = psycopg2.connect(**self.db_config)
        cursor = connection.cursor()

        if limit is None:
            # Load all available examples
            query = """
            SELECT case_id, consumer_complaint_text, simplified_ground_truth_tier,
                   vector_embeddings, product, sub_product, issue, sub_issue, complaint_category
            FROM ground_truth
            WHERE simplified_ground_truth_tier >= 0
            AND vector_embeddings IS NOT NULL
            AND vector_embeddings != ''
            ORDER BY case_id
            """
        else:
            # Load limited examples
            query = """
            SELECT case_id, consumer_complaint_text, simplified_ground_truth_tier,
                   vector_embeddings, product, sub_product, issue, sub_issue, complaint_category
            FROM ground_truth
            WHERE simplified_ground_truth_tier >= 0
            AND vector_embeddings IS NOT NULL
            AND vector_embeddings != ''
            ORDER BY case_id
            LIMIT %s
            """

        try:
            if limit is None:
                cursor.execute(query)
            else:
                cursor.execute(query, (limit,))
            results = cursor.fetchall()

            examples = []
            for row in results:
                case_id, complaint_text, tier, embedding_json, product, sub_product, issue, sub_issue, category = row

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
                    'product': product or '',
                    'sub_product': sub_product or '',
                    'issue': issue or '',
                    'sub_issue': sub_issue or '',
                    'category': category or ''
                })

            cursor.close()
            connection.close()

            examples_with_embeddings = [ex for ex in examples if ex['embedding'] is not None]
            print(f"[INFO] Loaded {len(examples_with_embeddings)} ground truth examples with embeddings")
            return examples_with_embeddings

        except Exception as e:
            print(f"[ERROR] Failed to load ground truth examples: {e}")
            cursor.close()
            connection.close()
            return []

    def calculate_combined_category_tier_statistics(self, product: str, issue: str) -> Dict[str, float]:
        """Calculate tier distribution statistics for combined product+issue category (cached)"""
        # Get the combined category key for caching
        combined_key = get_combined_category_key(product, issue)

        # Check cache first (thread-safe)
        with self.cache_lock:
            if combined_key in self.category_stats_cache:
                return self.category_stats_cache[combined_key]

        try:
            connection, cursor = self.get_thread_db_connection()

            product_group = get_product_category_group(product)
            issue_group = get_issue_category_group(issue)

            # Build SQL conditions for both product and issue groups
            product_condition = self._build_product_condition(product_group)
            issue_condition = self._build_issue_condition(issue_group)

            # Query to get tier distribution for the combined category
            query = f"""
                SELECT simplified_ground_truth_tier, COUNT(*) as count
                FROM ground_truth
                WHERE {product_condition}
                  AND {issue_condition}
                  AND simplified_ground_truth_tier >= 0
                  AND vector_embeddings IS NOT NULL
                  AND vector_embeddings != ''
                GROUP BY simplified_ground_truth_tier
                ORDER BY simplified_ground_truth_tier
            """

            cursor.execute(query)
            results = cursor.fetchall()

            # Initialize counts
            tier_counts = {0: 0, 1: 0, 2: 0}
            total_count = 0

            # Process results
            for tier, count in results:
                if tier in tier_counts:
                    tier_counts[tier] = count
                    total_count += count

            # Calculate percentages (fall back to global stats if no data)
            if total_count == 0:
                category_stats = {
                    'tier_0_percent': 65.6,
                    'tier_1_percent': 31.8,
                    'tier_2_percent': 2.6
                }
            else:
                category_stats = {
                    'tier_0_percent': (tier_counts[0] / total_count) * 100,
                    'tier_1_percent': (tier_counts[1] / total_count) * 100,
                    'tier_2_percent': (tier_counts[2] / total_count) * 100
                }

            # Cache the result (thread-safe) and log only once
            with self.cache_lock:
                if combined_key not in self.category_stats_cache:
                    self.category_stats_cache[combined_key] = category_stats
                    if total_count > 0:
                        print(f"[STATS] '{combined_key}': T0={category_stats['tier_0_percent']:.1f}%, "
                              f"T1={category_stats['tier_1_percent']:.1f}%, "
                              f"T2={category_stats['tier_2_percent']:.1f}% (n={total_count})")
                    else:
                        print(f"[STATS] '{combined_key}': Using global statistics (no data)")

            return category_stats

        except Exception as e:
            print(f"[ERROR] Failed to calculate combined category statistics for '{combined_key}': {e}")
            # Fall back to global statistics
            fallback_stats = {
                'tier_0_percent': 65.6,
                'tier_1_percent': 31.8,
                'tier_2_percent': 2.6
            }
            # Cache the fallback too
            with self.cache_lock:
                self.category_stats_cache[combined_key] = fallback_stats
            return fallback_stats

    def _build_product_condition(self, product_group: str) -> str:
        """Build SQL condition for product group"""
        if product_group == "Credit Reporting":
            return "(LOWER(product) LIKE '%credit report%' OR LOWER(product) LIKE '%credit repair%')"
        elif product_group == "Debt Collection":
            return "LOWER(product) LIKE '%debt collection%'"
        elif product_group == "Banking":
            return """(LOWER(product) LIKE '%checking%' OR LOWER(product) LIKE '%savings%'
                      OR LOWER(product) LIKE '%account%' OR LOWER(product) LIKE '%money transfer%'
                      OR LOWER(product) LIKE '%virtual currency%' OR LOWER(product) LIKE '%money service%'
                      OR LOWER(product) LIKE '%prepaid card%')"""
        elif product_group == "Lending":
            return """(LOWER(product) LIKE '%mortgage%' OR LOWER(product) LIKE '%credit card%'
                      OR LOWER(product) LIKE '%student loan%' OR LOWER(product) LIKE '%vehicle loan%'
                      OR LOWER(product) LIKE '%payday loan%' OR LOWER(product) LIKE '%title loan%'
                      OR LOWER(product) LIKE '%personal loan%' OR LOWER(product) LIKE '%consumer loan%'
                      OR LOWER(product) LIKE '%advance loan%' OR LOWER(product) LIKE '%lease%')"""
        else:  # "Other"
            return "1=1"  # Match all (will be handled by global stats fallback)

    def _build_issue_condition(self, issue_group: str) -> str:
        """Build SQL condition for issue group"""
        if issue_group == "Credit Report Issues":
            return """(LOWER(issue) LIKE '%incorrect information%' OR LOWER(issue) LIKE '%improper use%'
                      OR LOWER(issue) LIKE '%investigation into%' OR LOWER(issue) LIKE '%credit report%'
                      OR LOWER(issue) LIKE '%credit reporting%')"""
        elif issue_group == "Debt Collection Issues":
            return """(LOWER(issue) LIKE '%collect debt%' OR LOWER(issue) LIKE '%debt not owed%'
                      OR LOWER(issue) LIKE '%notification about debt%' OR LOWER(issue) LIKE '%false statements%'
                      OR LOWER(issue) LIKE '%representation%' OR LOWER(issue) LIKE '%negative or legal action%'
                      OR LOWER(issue) LIKE '%communication tactics%')"""
        elif issue_group == "Account Management":
            return """(LOWER(issue) LIKE '%managing%' OR LOWER(issue) LIKE '%account%'
                      OR LOWER(issue) LIKE '%payment process%' OR LOWER(issue) LIKE '%transaction%'
                      OR LOWER(issue) LIKE '%lender%' OR LOWER(issue) LIKE '%servicer%'
                      OR LOWER(issue) LIKE '%struggling to pay%' OR LOWER(issue) LIKE '%loan%'
                      OR LOWER(issue) LIKE '%lease%')"""
        elif issue_group == "Fraud/Disputes":
            return """(LOWER(issue) LIKE '%fraud%' OR LOWER(issue) LIKE '%scam%'
                      OR LOWER(issue) LIKE '%purchase shown%' OR LOWER(issue) LIKE '%billing%'
                      OR LOWER(issue) LIKE '%dispute%' OR LOWER(issue) LIKE '%charging your account%')"""
        else:  # "Other"
            return "1=1"  # Match all (will be handled by global stats fallback)

    def calculate_grouped_category_tier_statistics(self, product: str) -> Dict[str, float]:
        """Calculate tier distribution statistics for a grouped product category"""
        try:
            connection, cursor = self.get_thread_db_connection()

            # Get the category group
            category_group = get_product_category_group(product)

            # Build the SQL condition based on category group
            if category_group == "Credit Reporting":
                condition = "(LOWER(product) LIKE '%credit report%' OR LOWER(product) LIKE '%credit repair%')"
            elif category_group == "Debt Collection":
                condition = "LOWER(product) LIKE '%debt collection%'"
            elif category_group == "Banking":
                condition = """(LOWER(product) LIKE '%checking%' OR LOWER(product) LIKE '%savings%'
                              OR LOWER(product) LIKE '%account%' OR LOWER(product) LIKE '%money transfer%'
                              OR LOWER(product) LIKE '%virtual currency%' OR LOWER(product) LIKE '%money service%'
                              OR LOWER(product) LIKE '%prepaid card%')"""
            elif category_group == "Lending":
                condition = """(LOWER(product) LIKE '%mortgage%' OR LOWER(product) LIKE '%credit card%'
                              OR LOWER(product) LIKE '%student loan%' OR LOWER(product) LIKE '%vehicle loan%'
                              OR LOWER(product) LIKE '%payday loan%' OR LOWER(product) LIKE '%title loan%'
                              OR LOWER(product) LIKE '%personal loan%' OR LOWER(product) LIKE '%consumer loan%'
                              OR LOWER(product) LIKE '%advance loan%' OR LOWER(product) LIKE '%lease%')"""
            else:
                # For "Other" category, use global statistics
                return {
                    'tier_0_percent': 65.6,
                    'tier_1_percent': 31.8,
                    'tier_2_percent': 2.6
                }

            # Query to get tier distribution for the category group
            query = f"""
                SELECT simplified_ground_truth_tier, COUNT(*) as count
                FROM ground_truth
                WHERE {condition}
                  AND simplified_ground_truth_tier >= 0
                  AND vector_embeddings IS NOT NULL
                  AND vector_embeddings != ''
                GROUP BY simplified_ground_truth_tier
                ORDER BY simplified_ground_truth_tier
            """

            cursor.execute(query)
            results = cursor.fetchall()

            # Initialize counts
            tier_counts = {0: 0, 1: 0, 2: 0}
            total_count = 0

            # Process results
            for tier, count in results:
                if tier in tier_counts:
                    tier_counts[tier] = count
                    total_count += count

            # Calculate percentages (fall back to global stats if no data)
            if total_count == 0:
                print(f"[WARNING] No data found for category group '{category_group}'. Using global statistics.")
                return {
                    'tier_0_percent': 65.6,
                    'tier_1_percent': 31.8,
                    'tier_2_percent': 2.6
                }

            category_stats = {
                'tier_0_percent': (tier_counts[0] / total_count) * 100,
                'tier_1_percent': (tier_counts[1] / total_count) * 100,
                'tier_2_percent': (tier_counts[2] / total_count) * 100
            }

            print(f"[INFO] Category group '{category_group}' stats: "
                  f"T0={category_stats['tier_0_percent']:.1f}%, "
                  f"T1={category_stats['tier_1_percent']:.1f}%, "
                  f"T2={category_stats['tier_2_percent']:.1f}% "
                  f"(n={total_count})")

            return category_stats

        except Exception as e:
            print(f"[ERROR] Failed to calculate category statistics for '{product}': {e}")
            # Fall back to global statistics
            return {
                'tier_0_percent': 65.6,
                'tier_1_percent': 31.8,
                'tier_2_percent': 2.6
            }

    def call_gpt4o_mini_analysis(self, system_prompt: str, user_prompt: str, case_id: Optional[int] = None) -> Optional[ProcessAnalysis]:
        """Send prompt to GPT-4o-mini for analysis with caching"""
        import hashlib

        # Create hash for caching
        prompt_text = f"{system_prompt}\n\n{user_prompt}"
        request_hash = hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()

        try:
            # Check cache first
            connection, cursor = self.get_thread_db_connection()

            cursor.execute("SELECT response_json FROM llm_cache WHERE request_hash = %s", (request_hash,))
            cached_result = cursor.fetchone()

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

            # Cache the result
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

                    cursor.execute(insert_cache_sql, (
                        request_hash,
                        'gpt-4o-mini',
                        0.0,
                        prompt_text,
                        system_prompt,
                        case_id,
                        f"Tier: {analysis.tier}, Confidence: {analysis.confidence.value}",
                        json.dumps(response_json),
                        analysis.tier,
                        analysis.confidence.value,
                        analysis.information_needed,
                        analysis.confidence.value == 'need_more_info',
                        analysis.reasoning
                    ))

                    connection.commit()

                except Exception as e:
                    print(f"[WARNING] Failed to cache result: {e}")

            return analysis

        except Exception as e:
            print(f"[ERROR] GPT-4o-mini call failed: {e}")
            return None


class MultithreadedSingleAlphaOptimizer(MultithreadedNShotOptimizer):
    """Multithreaded version of single alpha optimization"""

    def generate_alpha_heatmap(self, results: List[Dict], experiment_timestamp: datetime):
        """Generate 2D heatmap for n vs alpha optimization results"""
        try:
            print(f"\n[HEATMAP] Generating alpha optimization heatmap...")

            # Create result matrix
            n_values = sorted(list(set([r['n'] for r in results])))
            alpha_values = sorted(list(set([r['alpha'] for r in results])))

            # Initialize matrix with NaN
            accuracy_matrix = np.full((len(n_values), len(alpha_values)), np.nan)

            # Fill matrix with accuracy values
            for result in results:
                n_idx = n_values.index(result['n'])
                alpha_idx = alpha_values.index(result['alpha'])
                accuracy_matrix[n_idx, alpha_idx] = result['accuracy']

            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(accuracy_matrix,
                       xticklabels=[f"{a:.2f}" for a in alpha_values],
                       yticklabels=[f"{n}" for n in n_values],
                       annot=True,
                       fmt='.3f',
                       cmap='viridis',
                       cbar_kws={'label': 'Accuracy'})

            plt.title(f'N-Shot Alpha Optimization Accuracy Heatmap\n{experiment_timestamp.strftime("%Y-%m-%d %H:%M:%S")}')
            plt.xlabel('Alpha Parameter')
            plt.ylabel('N-Shot Examples')
            plt.tight_layout()

            # Save plot
            timestamp_str = experiment_timestamp.strftime("%Y%m%d_%H%M%S")

            # Create directory if it doesn't exist
            output_dir = Path("nshot_optimisation")
            output_dir.mkdir(exist_ok=True)

            filename = f"alpha_optimization_heatmap_{timestamp_str}.png"
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"[SUCCESS] Alpha heatmap saved as: {filepath}")

        except Exception as e:
            print(f"[ERROR] Failed to generate alpha heatmap: {e}")

    def create_alpha_results_table(self):
        """Create/verify table for alpha optimization results"""
        connection = psycopg2.connect(**self.db_config)
        cursor = connection.cursor()

        # Check if table exists, create if not
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'nshot_optimization_results'
            );
        """)

        if not cursor.fetchone()[0]:
            print("[ERROR] nshot_optimization_results table doesn't exist. Run add_nshot_results_table.py first.")
            cursor.close()
            connection.close()
            return False

        cursor.close()
        connection.close()
        return True

    def precompute_incremental_selections(self, test_examples: List[Dict], all_examples: List[Dict],
                                          n_grid: List[int], alpha_grid: List[float]) -> Tuple[Dict, List[Optional[Dict]]]:
        """Pre-compute incremental selections for all alpha and test example combinations."""
        max_n = max(n_grid) if n_grid else 0
        alpha_values = sorted(set(list(alpha_grid) + [0.0]))
        cache: Dict[Tuple[float, int, int], Tuple[int, ...]] = {}
        precomputed_candidate_data: List[Optional[Dict]] = []
        total_selection_steps = 0

        print("[CACHE] Preparing candidate pools for alpha optimization...")

        max_candidates_per_test = getattr(self, "max_alpha_candidates", 250)
        self.max_alpha_candidates = max_candidates_per_test

        from sklearn.metrics.pairwise import cosine_similarity

        for test_idx, target_example in enumerate(test_examples):
            query_embedding = target_example.get('embedding')
            if query_embedding is None:
                precomputed_candidate_data.append(None)
                continue

            candidates = [ex for ex in all_examples if ex['case_id'] != target_example['case_id']]
            if not candidates:
                precomputed_candidate_data.append(None)
                continue

            candidate_embeddings = np.array([ex['embedding'] for ex in candidates], dtype=np.float32)
            query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)

            query_similarities = cosine_similarity(candidate_embeddings, query_embedding).flatten()

            if max_candidates_per_test and len(candidates) > max_candidates_per_test:
                top_indices = np.argsort(query_similarities)[::-1][:max_candidates_per_test]
            else:
                top_indices = np.arange(len(candidates))

            filtered_candidates = [candidates[i] for i in top_indices]
            filtered_embeddings = candidate_embeddings[top_indices]
            filtered_similarities = query_similarities[top_indices].astype(np.float32)

            norms = np.linalg.norm(filtered_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized_embeddings = (filtered_embeddings / norms).astype(np.float32)

            precomputed_candidate_data.append({
                'examples': filtered_candidates,
                'embeddings': normalized_embeddings,
                'query_similarity': filtered_similarities
            })

        print(f"[CACHE] Computing incremental selections for {len(alpha_values)} alpha values...")

        for alpha_idx, alpha in enumerate(alpha_values):
            print(f"[CACHE] Alpha {alpha_idx + 1}/{len(alpha_values)}: {alpha:.2f}")
            cache_start = time.time()

            for test_idx, candidate_data in enumerate(precomputed_candidate_data):
                if candidate_data is None:
                    for n in n_grid:
                        cache[(alpha, test_idx, n)] = tuple()
                    continue

                query_sim = candidate_data['query_similarity']
                embeddings = candidate_data['embeddings']
                candidate_count = len(query_sim)

                if candidate_count == 0:
                    for n in n_grid:
                        cache[(alpha, test_idx, n)] = tuple()
                    continue

                cache[(alpha, test_idx, 0)] = tuple()

                selection_order: List[int] = []
                selected_mask = np.zeros(candidate_count, dtype=bool)
                similarity_sums = np.zeros(candidate_count, dtype=np.float32)

                for current_n in range(1, max_n + 1):
                    if current_n > candidate_count:
                        if current_n in n_grid:
                            cache[(alpha, test_idx, current_n)] = tuple(selection_order)
                        continue

                    best_idx = -1
                    best_score = -np.inf
                    denom = current_n - 1

                    for idx in range(candidate_count):
                        if selected_mask[idx]:
                            continue

                        if denom == 0:
                            diversity_score = 1.0
                        else:
                            avg_similarity = similarity_sums[idx] / denom
                            diversity_score = 1.0 - avg_similarity

                        score = alpha * query_sim[idx] + (1.0 - alpha) * diversity_score
                        if score > best_score:
                            best_score = score
                            best_idx = idx

                    if best_idx == -1:
                        if current_n in n_grid:
                            cache[(alpha, test_idx, current_n)] = tuple(selection_order)
                        continue

                    selected_mask[best_idx] = True
                    selection_order.append(best_idx)
                    if current_n in n_grid:
                        cache[(alpha, test_idx, current_n)] = tuple(selection_order)

                    selected_vector = embeddings[best_idx]
                    similarity_updates = embeddings @ selected_vector
                    similarity_sums += similarity_updates

                    total_selection_steps += 1

                for n in n_grid:
                    if n == 0:
                        continue
                    cache.setdefault((alpha, test_idx, n), tuple(selection_order))

            cache_time = time.time() - cache_start
            print(f"[CACHE] Alpha {alpha:.2f} completed in {cache_time:.1f}s")

        print(f"[SUCCESS] Incremental selection cache built with {total_selection_steps} selection steps")
        return cache, precomputed_candidate_data

    def test_single_parameter_combination_with_cache(self, params: Tuple, global_selection_cache: Dict,
                                                     precomputed_candidate_data: List[Optional[Dict]]) -> Dict:
        """Test a single (n, alpha) parameter combination using pre-computed selections."""
        n, alpha, test_examples, all_examples, experiment_timestamp = params

        start_time = time.time()
        correct_predictions = 0
        total_predictions = 0
        confident_decisions = 0
        uncertain_decisions = 0
        need_info_decisions = 0

        try:
            for test_idx, target_example in enumerate(test_examples):
                target_complaint = target_example['complaint_text']
                target_tier = target_example['tier']

                if n == 0:
                    nshot_examples = []
                else:
                    candidate_data = precomputed_candidate_data[test_idx] if test_idx < len(precomputed_candidate_data) else None
                    if not candidate_data:
                        continue

                    selection_key = (alpha, test_idx, n)
                    selected_indices = global_selection_cache.get(selection_key, tuple())
                    selected_subset = selected_indices[:n]

                    if len(selected_subset) < n:
                        continue

                    candidates = candidate_data['examples']
                    nshot_examples = []
                    for idx in selected_subset:
                        if idx < len(candidates):
                            candidate = candidates[idx]
                            nshot_examples.append({
                                'complaint_text': candidate['complaint_text'],
                                'tier': candidate['tier']
                            })

                    if len(nshot_examples) < n:
                        continue

                # Calculate category-specific tier statistics using product+issue
                target_product = target_example.get('product', '')
                target_issue = target_example.get('issue', '')
                category_tier_stats = self.calculate_combined_category_tier_statistics(target_product, target_issue)

                system_prompt, user_prompt = self.prompt_generator.generate_prompts(
                    target_case={'complaint_text': target_complaint},
                    nshot_examples=nshot_examples,
                    persona=None,
                    bias_strategy=BiasStrategy.CHAIN_OF_THOUGHT,
                    category_tier_stats=category_tier_stats
                )

                analysis = self.call_gpt4o_mini_analysis(system_prompt, user_prompt, target_example.get('case_id'))

                if analysis is None:
                    continue

                total_predictions += 1

                predicted_tier = analysis.tier
                if predicted_tier == target_tier:
                    correct_predictions += 1

                if analysis.confidence.lower() == 'confident':
                    confident_decisions += 1
                elif analysis.confidence.lower() == 'uncertain':
                    uncertain_decisions += 1
                elif analysis.information_needed:
                    need_info_decisions += 1

        except Exception as e:
            print(f"[ERROR] Alpha test failed for n={n}, alpha={alpha}: {e}")

        execution_time = time.time() - start_time
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        result = {
            'n': n,
            'alpha': alpha,
            'correct': correct_predictions,
            'total': total_predictions,
            'accuracy': accuracy,
            'confident_decisions': confident_decisions,
            'uncertain_decisions': uncertain_decisions,
            'need_info_decisions': need_info_decisions,
            'execution_time': execution_time,
            'experiment_timestamp': experiment_timestamp
        }

        self.save_alpha_result_to_db(result, len(all_examples), len(test_examples))

        return result
    def test_single_parameter_combination(self, params: Tuple) -> Dict:
        """Test a single (n, alpha) parameter combination"""
        n, alpha, test_examples, all_examples, experiment_timestamp = params

        start_time = time.time()
        correct_predictions = 0
        total_predictions = 0
        confident_decisions = 0
        uncertain_decisions = 0
        need_info_decisions = 0

        try:
            for target_example in test_examples:
                target_embedding = target_example['embedding']
                target_complaint = target_example['complaint_text']
                target_tier = target_example['tier']

                # Create candidate pool
                candidates = [ex for ex in all_examples if ex['case_id'] != target_example['case_id']]

                if len(candidates) < n:
                    continue

                if n == 0:
                    nshot_examples = []
                else:
                    # Select examples using joint optimization
                    candidate_embeddings = np.array([ex['embedding'] for ex in candidates])
                    selected_indices = self.optimizer.joint_optimization(
                        target_embedding, candidate_embeddings, n, lambda_param=alpha
                    )
                    nshot_examples = []
                    for idx in selected_indices:
                        if idx < len(candidates):
                            ex = candidates[idx]
                            nshot_examples.append({
                                'complaint_text': ex['complaint_text'],
                                'tier': ex['tier']
                            })

                # Create target case
                target_case = {'complaint_text': target_complaint}

                # Calculate category-specific tier statistics using product+issue
                target_product = target_example.get('product', '')
                target_issue = target_example.get('issue', '')
                category_tier_stats = self.calculate_combined_category_tier_statistics(target_product, target_issue)

                # Generate prompts
                system_prompt, user_prompt = self.prompt_generator.generate_prompts(
                    target_case=target_case,
                    nshot_examples=nshot_examples,
                    persona=None,
                    bias_strategy=BiasStrategy.CHAIN_OF_THOUGHT,
                    category_tier_stats=category_tier_stats
                )

                # Get prediction
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

        except Exception as e:
            print(f"[ERROR] Error in parameter combination n={n}, alpha={alpha}: {e}")

        execution_time = time.time() - start_time
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        result = {
            'n': n,
            'alpha': alpha,
            'correct': correct_predictions,
            'total': total_predictions,
            'accuracy': accuracy,
            'confident_decisions': confident_decisions,
            'uncertain_decisions': uncertain_decisions,
            'need_info_decisions': need_info_decisions,
            'execution_time': execution_time,
            'experiment_timestamp': experiment_timestamp
        }

        # Save to database immediately
        self.save_alpha_result_to_db(result, len(all_examples), len(test_examples))

        return result

    def save_alpha_result_to_db(self, result: Dict, total_ground_truth: int, sample_size: int):
        """Save single alpha optimization result to database"""
        try:
            connection, cursor = self.get_thread_db_connection()

            confident_rate = result['confident_decisions'] / result['total'] if result['total'] > 0 else 0
            uncertain_rate = result['uncertain_decisions'] / result['total'] if result['total'] > 0 else 0
            need_info_rate = result['need_info_decisions'] / result['total'] if result['total'] > 0 else 0

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
                result['experiment_timestamp'],
                'multithreaded_alpha',
                result['n'],
                result['alpha'],
                sample_size,
                total_ground_truth,
                result['correct'],
                result['total'],
                result['accuracy'],
                result['confident_decisions'],
                result['uncertain_decisions'],
                result['need_info_decisions'],
                confident_rate,
                uncertain_rate,
                need_info_rate,
                result['execution_time'],
                f"Multithreaded: n={result['n']}, alpha={result['alpha']:.2f}"
            ))

            connection.commit()

        except Exception as e:
            print(f"[WARNING] Failed to save result for n={result['n']}, alpha={result['alpha']}: {e}")

    def run_multithreaded_alpha_optimization(self, sample_size: int = 500):
        """Run multithreaded single alpha optimization"""

        print("=" * 80)
        print("MULTITHREADED N-SHOT ALPHA OPTIMIZATION")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Using {self.max_workers} worker threads")

        # Verify table exists
        if not self.create_alpha_results_table():
            return

        # Load data
        print(f"\n[STEP 1] Loading ground truth examples...")

        # Load separate pools for testing and candidates to avoid data leakage
        print("[INFO] Loading test examples...")
        test_examples = self.load_ground_truth_examples(limit=250)  # Fixed test set
        if not test_examples:
            print("[ERROR] No test examples loaded")
            return
        print(f"[INFO] Loaded {len(test_examples)} test examples")

        print("[INFO] Loading candidate pool with stratified sampling...")
        # Load extra samples to account for test example removal
        all_examples = self.load_ground_truth_examples(stratified_sample=1250)  # Stratified candidates
        if not all_examples:
            print("[ERROR] No candidate examples loaded")
            return

        # Remove any test examples from candidate pool to prevent data leakage
        test_case_ids = {ex['case_id'] for ex in test_examples}
        all_examples = [ex for ex in all_examples if ex['case_id'] not in test_case_ids]

        print(f"[INFO] Loaded {len(all_examples)} stratified examples as candidate pool (after removing test examples)")

        # Parameter grids
        n_grid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        alpha_grid = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

        # Create parameter combinations (optimized for n=0)
        param_combinations = []
        experiment_timestamp = datetime.now()

        for n in n_grid:
            if n == 0:
                # Only test alpha=0 for zero-shot (no examples, so no diversity/relevance trade-off)
                param_combinations.append((n, 0.0, test_examples, all_examples, experiment_timestamp))
            else:
                # Test all alpha values for n > 0
                for alpha in alpha_grid:
                    param_combinations.append((n, alpha, test_examples, all_examples, experiment_timestamp))

        # Pre-compute incremental selections for efficiency
        print(f"\n[STEP 2] Pre-computing incremental selections...")
        global_selection_cache, precomputed_candidate_data = self.precompute_incremental_selections(
            test_examples, all_examples, n_grid, alpha_grid
        )

        print(f"\n[STEP 3] Testing {len(param_combinations)} parameter combinations with {self.max_workers} threads...")

        results = []
        completed_count = 0

        # Run multithreaded optimization
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks with global selection cache
            future_to_params = {
                executor.submit(
                    self.test_single_parameter_combination_with_cache,
                    params,
                    global_selection_cache,
                    precomputed_candidate_data
                ): params
                for params in param_combinations
            }

            # Process completed tasks
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                n, alpha = params[0], params[1]

                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1

                    print(f"[PROGRESS] {completed_count}/{len(param_combinations)}: n={n}, alpha={alpha:.2f} -> {result['accuracy']:.3f} accuracy ({result['execution_time']:.1f}s)")

                except Exception as e:
                    print(f"[ERROR] Parameter combination n={n}, alpha={alpha:.2f} failed: {e}")

        # Find and display best results
        if results:
            # Sort by accuracy (descending), then by total examples (n, ascending for tie-breaking)
            sorted_results = sorted(results, key=lambda x: (-x['accuracy'], x['n']))

            best_result = sorted_results[0]
            print(f"\n[SUCCESS] Best multithreaded alpha optimization:")
            print(f"  n = {best_result['n']}")
            print(f"  alpha = {best_result['alpha']:.2f}")
            print(f"  Accuracy = {best_result['accuracy']:.3f} ({best_result['correct']}/{best_result['total']})")

            # Display top 5 results
            print(f"\n[TOP 5] Alpha Optimization Results (highest accuracy → lowest):")
            print("=" * 70)
            for i, result in enumerate(sorted_results[:5], 1):
                accuracy_pct = result['accuracy'] * 100
                print(f"{i:2}. n={result['n']:2}, alpha={result['alpha']:.2f} → "
                      f"{accuracy_pct:5.1f}% accuracy ({result['correct']:2}/{result['total']:2}) "
                      f"[{result['execution_time']:.1f}s]")

        print(f"\n[SUCCESS] Multithreaded alpha optimization complete!")

        # Generate heatmap
        self.generate_alpha_heatmap(results, experiment_timestamp)


class MultithreadedDPPKNNOptimizer(MultithreadedNShotOptimizer):
    """Multithreaded version of DPP + k-NN optimization with hybrid similarity"""

    def generate_dpp_knn_heatmap(self, results: List[Dict], experiment_timestamp: datetime):
        """Generate 2D heatmap for DPP vs k-NN optimization results"""
        try:
            print(f"\n[HEATMAP] Generating DPP + k-NN optimization heatmap...")

            # Create result matrix
            n_dpp_values = sorted(list(set([r['n_dpp'] for r in results])))
            k_nn_values = sorted(list(set([r['k_nn'] for r in results])))

            # Initialize matrix with NaN
            accuracy_matrix = np.full((len(n_dpp_values), len(k_nn_values)), np.nan)

            # Fill matrix with accuracy values
            for result in results:
                n_dpp_idx = n_dpp_values.index(result['n_dpp'])
                k_nn_idx = k_nn_values.index(result['k_nn'])
                accuracy_matrix[n_dpp_idx, k_nn_idx] = result['accuracy']

            # Create heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(accuracy_matrix,
                       xticklabels=[f"{k}" for k in k_nn_values],
                       yticklabels=[f"{n}" for n in n_dpp_values],
                       annot=True,
                       fmt='.3f',
                       cmap='viridis',
                       cbar_kws={'label': 'Accuracy'})

            plt.title(f'DPP + k-NN Optimization Accuracy Heatmap\n{experiment_timestamp.strftime("%Y-%m-%d %H:%M:%S")}\n(Excluded combinations > 10 total examples)')
            plt.xlabel('k-NN Examples')
            plt.ylabel('DPP Examples')
            plt.tight_layout()

            # Save plot
            timestamp_str = experiment_timestamp.strftime("%Y%m%d_%H%M%S")

            # Create directory if it doesn't exist
            output_dir = Path("nshot_optimisation")
            output_dir.mkdir(exist_ok=True)

            filename = f"dpp_knn_optimization_heatmap_{timestamp_str}.png"
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"[SUCCESS] DPP + k-NN heatmap saved as: {filepath}")

        except Exception as e:
            print(f"[ERROR] Failed to generate DPP + k-NN heatmap: {e}")

    def __init__(self, max_workers: int = 4, use_hybrid_similarity: bool = True):
        """Initialize with optional hybrid similarity"""
        super().__init__(max_workers)

        # Set up hybrid similarity calculator
        self.use_hybrid_similarity = use_hybrid_similarity
        if use_hybrid_similarity:
            # Use balanced configuration by default
            self.similarity_calculator = HybridSimilarityPresets.balanced()
            self.hybrid_calculator = self.similarity_calculator  # Keep backward compatibility
            self.hybrid_selector = HybridKNNSelector(self.similarity_calculator)
            print("[INFO] Using hybrid similarity (semantic + categorical features)")
        else:
            self.similarity_calculator = None
            self.hybrid_calculator = None
            self.hybrid_selector = None
            print("[INFO] Using pure semantic similarity")

        self.global_dpp_selector: Optional[HybridDPP] = None
        self.global_dpp_examples: List[Dict] = []

    def create_dpp_knn_results_table(self):
        """Create/verify table for DPP + k-NN optimization results"""
        connection = psycopg2.connect(**self.db_config)
        cursor = connection.cursor()

        try:
            create_table_sql = """
                CREATE TABLE IF NOT EXISTS nshot_dpp_knn_results (
                    id SERIAL PRIMARY KEY,
                    experiment_timestamp TIMESTAMP NOT NULL,
                    experiment_type VARCHAR(50) DEFAULT 'multithreaded_dpp_knn',

                    -- Parameter combination
                    n_dpp INTEGER NOT NULL,
                    k_nn INTEGER NOT NULL,
                    total_examples INTEGER NOT NULL,

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

                CREATE INDEX IF NOT EXISTS idx_dpp_knn_mt_experiment ON nshot_dpp_knn_results(experiment_timestamp);
                CREATE INDEX IF NOT EXISTS idx_dpp_knn_mt_params ON nshot_dpp_knn_results(n_dpp, k_nn);
                CREATE INDEX IF NOT EXISTS idx_dpp_knn_mt_accuracy ON nshot_dpp_knn_results(accuracy_score);
            """

            cursor.execute(create_table_sql)
            connection.commit()
            cursor.close()
            connection.close()
            return True

        except Exception as e:
            print(f"[ERROR] Failed to create DPP + k-NN table: {e}")
            cursor.close()
            connection.close()
            return False

    def _initialize_global_dpp_selector(self, all_examples: List[Dict], random_state: int = 42):
        """Prepare the global HybridDPP selector across the candidate pool."""
        valid = [ex for ex in all_examples if ex.get('embedding') is not None]
        if valid:
            embeddings = np.array([ex['embedding'] for ex in valid], dtype=np.float32)
            self.global_dpp_examples = valid
            from hybrid_dpp import HybridDPP
            self.global_dpp_selector = HybridDPP(embeddings, random_state=random_state)
        else:
            self.global_dpp_examples = []
            self.global_dpp_selector = None

    def _get_global_dpp_examples(self, n: int, exclude_case_id: Optional[int] = None) -> List[Dict]:
        """Retrieve up to n globally selected DPP examples, optionally excluding a case."""
        if n <= 0 or not self.global_dpp_selector or not self.global_dpp_examples:
            return []

        selector = self.global_dpp_selector
        examples = self.global_dpp_examples
        max_items = getattr(selector, "num_items", len(examples))
        if max_items == 0:
            return []

        used_case_ids = set()
        if exclude_case_id is not None:
            used_case_ids.add(exclude_case_id)

        collected: List[Dict] = []
        attempted_sizes = set()
        buffer = 0

        while len(collected) < n and buffer <= max(0, max_items - n):
            size = min(n + buffer, max_items)
            if size in attempted_sizes:
                break
            attempted_sizes.add(size)

            indices = selector.select(size)
            for idx in indices:
                if idx >= len(examples):
                    continue
                ex = examples[idx]
                case_id = ex.get('case_id')
                if case_id in used_case_ids:
                    continue
                used_case_ids.add(case_id)
                collected.append(ex)
                if len(collected) == n:
                    break

            if len(collected) >= n or size == max_items:
                break
            buffer += 2

        return collected[:n]

    def precompute_dpp_knn_resources(self,
                                     test_examples: List[Dict],
                                     all_examples: List[Dict],
                                     max_n_dpp: int,
                                     max_k_nn: int,
                                     max_total_examples: int) -> List[Optional[Dict]]:
        """Pre-compute shared candidate pools and selection orders for DPP + k-NN."""
        print("[CACHE] Building shared candidate pools for DPP + k-NN optimization...")
        print(f"[CACHE] Target max_n_dpp={max_n_dpp}, max_k_nn={max_k_nn}, max_total={max_total_examples}")

        max_candidates_per_test = getattr(self, "max_dpp_knn_candidates", 300)
        self.max_dpp_knn_candidates = max_candidates_per_test

        precomputed_data: List[Optional[Dict]] = []
        from sklearn.metrics.pairwise import cosine_similarity

        for test_idx, target_example in enumerate(test_examples):
            target_case_id = target_example.get('case_id')

            filtered_candidates: List[Dict] = []
            knn_order: Tuple[int, ...] = tuple()

            knn_candidates = [
                ex for ex in all_examples
                if ex.get('case_id') != target_case_id and ex.get('embedding') is not None
            ]

            if max_k_nn > 0 and knn_candidates:
                if self.use_hybrid_similarity and self.similarity_calculator:
                    ranked = self.similarity_calculator.rank_candidates(target_example, knn_candidates)
                    ordered_indices = [idx for idx, _ in ranked]
                else:
                    query_embedding = target_example.get('embedding')
                    if query_embedding is not None:
                        candidate_embeddings = np.array([
                            ex['embedding'] for ex in knn_candidates
                        ], dtype=np.float32)
                        similarity_scores = cosine_similarity(
                            np.asarray(query_embedding, dtype=np.float32).reshape(1, -1),
                            candidate_embeddings
                        ).flatten()
                        ordered_indices = np.argsort(similarity_scores)[::-1].tolist()
                    else:
                        ordered_indices = []

                if ordered_indices:
                    if max_candidates_per_test and len(ordered_indices) > max_candidates_per_test:
                        ordered_indices = ordered_indices[:max_candidates_per_test]
                    filtered_candidates = [knn_candidates[idx] for idx in ordered_indices]
                    knn_order = tuple(range(len(filtered_candidates)))

            precomputed_data.append({
                'case_id': target_case_id,
                'knn_examples': filtered_candidates,
                'knn_order': knn_order
            })

            if (test_idx + 1) % 25 == 0 or test_idx == len(test_examples) - 1:
                print(f"[CACHE] Prepared {test_idx + 1}/{len(test_examples)} candidate pools")

        print("[SUCCESS] Shared candidate pools ready for DPP + k-NN optimization")
        return precomputed_data

    def build_precomputed_dpp_knn_selection(self,
                                            candidate_data: Optional[Dict],
                                            n_dpp: int,
                                            k_nn: int) -> Optional[List[Dict]]:
        """Create n-shot examples from precomputed candidate data for given (n_dpp, k_nn)."""
        total_required = n_dpp + k_nn
        if total_required == 0:
            return []

        candidate_data = candidate_data or {}
        case_id = candidate_data.get('case_id')

        final_examples: List[Dict] = []
        used_case_ids = set()

        if n_dpp > 0:
            for ex in self._get_global_dpp_examples(n_dpp, exclude_case_id=case_id):
                final_examples.append({
                    'complaint_text': ex['complaint_text'],
                    'tier': ex['tier']
                })
                used_case_ids.add(ex.get('case_id'))

        if k_nn > 0:
            knn_examples = candidate_data.get('knn_examples') or []
            knn_order = candidate_data.get('knn_order') or tuple()
            for idx in knn_order:
                if len(final_examples) >= total_required:
                    break
                if idx >= len(knn_examples):
                    continue
                candidate = knn_examples[idx]
                cid = candidate.get('case_id')
                if cid in used_case_ids:
                    continue
                final_examples.append({
                    'complaint_text': candidate['complaint_text'],
                    'tier': candidate['tier']
                })
                used_case_ids.add(cid)
                if len(final_examples) >= total_required:
                    break

        if total_required > 0 and not final_examples:
            return None

        return final_examples[:total_required]

    def test_dpp_knn_combination_with_precomputed(self,
                                                  params: Tuple,
                                                  precomputed_candidate_data: List[Optional[Dict]]) -> Dict:
        """Test a single (n_dpp, k_nn) combination using precomputed candidate pools."""
        n_dpp, k_nn, test_examples, total_candidate_count, experiment_timestamp = params

        start_time = time.time()
        correct_predictions = 0
        total_predictions = 0
        confident_decisions = 0
        uncertain_decisions = 0
        need_info_decisions = 0

        try:
            for test_idx, target_example in enumerate(test_examples):
                candidate_data = precomputed_candidate_data[test_idx] if test_idx < len(precomputed_candidate_data) else None

                if n_dpp == 0 and k_nn == 0:
                    nshot_examples: Optional[List[Dict]] = []
                else:
                    nshot_examples = self.build_precomputed_dpp_knn_selection(candidate_data, n_dpp, k_nn)

                if nshot_examples is None:
                    continue

                # Calculate category-specific tier statistics using product+issue
                target_product = target_example.get('product', '')
                target_issue = target_example.get('issue', '')
                category_tier_stats = self.calculate_combined_category_tier_statistics(target_product, target_issue)

                system_prompt, user_prompt = self.prompt_generator.generate_prompts(
                    target_case={'complaint_text': target_example['complaint_text']},
                    nshot_examples=nshot_examples,
                    persona=None,
                    bias_strategy=BiasStrategy.CHAIN_OF_THOUGHT,
                    category_tier_stats=category_tier_stats
                )

                analysis = self.call_gpt4o_mini_analysis(system_prompt, user_prompt, target_example.get('case_id'))

                if analysis is None:
                    continue

                total_predictions += 1

                if analysis.tier == target_example['tier']:
                    correct_predictions += 1

                if analysis.confidence == DecisionConfidence.CONFIDENT:
                    confident_decisions += 1
                elif analysis.confidence == DecisionConfidence.UNCERTAIN:
                    uncertain_decisions += 1
                elif analysis.confidence == DecisionConfidence.NEED_MORE_INFO or analysis.information_needed:
                    need_info_decisions += 1

        except Exception as e:
            print(f"[ERROR] Error in DPP + k-NN combination n_dpp={n_dpp}, k_nn={k_nn}: {e}")

        execution_time = time.time() - start_time
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        result = {
            'n_dpp': n_dpp,
            'k_nn': k_nn,
            'total_shots': n_dpp + k_nn,
            'correct': correct_predictions,
            'total': total_predictions,
            'accuracy': accuracy,
            'confident_decisions': confident_decisions,
            'uncertain_decisions': uncertain_decisions,
            'need_info_decisions': need_info_decisions,
            'execution_time': execution_time,
            'experiment_timestamp': experiment_timestamp
        }

        self.save_dpp_knn_result_to_db(result, total_candidate_count, len(test_examples))

        return result

    def combine_dpp_knn_examples(self, query_example: Dict, candidates: List[Dict],
                                n_dpp: int, k_nn: int) -> List[Dict]:
        """Combine DPP and k-NN example selection with hybrid similarity"""
        if n_dpp == 0 and k_nn == 0:
            return []

        selected_examples = []
        used_indices = set()

        # First, select DPP examples for diversity
        if n_dpp > 0:
            dpp_indices = self.select_dpp_examples(candidates, n_dpp)
            for idx in dpp_indices:
                if idx < len(candidates):
                    selected_examples.append(candidates[idx])
                    used_indices.add(idx)

        # Then, select k-NN examples for relevance (using hybrid similarity)
        if k_nn > 0:
            remaining_candidates = [candidates[i] for i in range(len(candidates)) if i not in used_indices]
            if remaining_candidates:
                knn_indices = self.select_knn_examples(query_example, remaining_candidates, k_nn)
                remaining_map = {new_idx: orig_idx for new_idx, orig_idx in enumerate(range(len(candidates))) if orig_idx not in used_indices}

                for new_idx in knn_indices:
                    if new_idx in remaining_map:
                        orig_idx = remaining_map[new_idx]
                        selected_examples.append(candidates[orig_idx])

        return selected_examples

    def select_dpp_examples(self, candidates: List[Dict], n_dpp: int) -> List[int]:
        """Select DPP examples from candidate list using HybridDPP."""
        if n_dpp == 0 or len(candidates) == 0:
            return []

        indexed_candidates = [
            (idx, ex) for idx, ex in enumerate(candidates) if ex.get('embedding') is not None
        ]
        if not indexed_candidates:
            return []

        embeddings = np.array([ex['embedding'] for _, ex in indexed_candidates], dtype=np.float32)
        from hybrid_dpp import HybridDPP
        selector = HybridDPP(embeddings, random_state=42)
        num_to_select = min(n_dpp, len(indexed_candidates))
        local_indices = selector.select(num_to_select)
        return [indexed_candidates[i][0] for i in local_indices]

    def select_knn_examples(self, query_example: Dict, candidates: List[Dict], k_nn: int) -> List[int]:
        """Select k-NN examples using hybrid or pure semantic similarity"""
        if k_nn == 0 or len(candidates) == 0:
            return []

        if self.use_hybrid_similarity and self.hybrid_selector:
            # Use hybrid similarity
            selected_indices = self.hybrid_selector.select_examples(
                query_example, candidates, k_nn, use_pure_knn=True
            )
            return selected_indices
        else:
            # Use pure semantic similarity (original implementation)
            query_embedding = query_example.get('embedding')
            if query_embedding is None:
                return []

            candidate_embeddings = np.array([ex['embedding'] for ex in candidates])

            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embedding.reshape(1, -1), candidate_embeddings).flatten()
            top_k_indices = np.argsort(similarities)[-k_nn:][::-1]

            return top_k_indices.tolist()

    def save_dpp_knn_result_to_db(self, result: Dict, total_ground_truth: int, sample_size: int):
        """Save DPP + k-NN result to database"""
        try:
            connection, cursor = self.get_thread_db_connection()

            confident_rate = result['confident_decisions'] / result['total'] if result['total'] > 0 else 0
            uncertain_rate = result['uncertain_decisions'] / result['total'] if result['total'] > 0 else 0
            need_info_rate = result['need_info_decisions'] / result['total'] if result['total'] > 0 else 0

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

            cursor.execute(insert_query, (
                result['experiment_timestamp'],
                'multithreaded_dpp_knn',
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
                confident_rate,
                uncertain_rate,
                need_info_rate,
                result['execution_time'],
                f"Multithreaded: DPP={result['n_dpp']}, k-NN={result['k_nn']}"
            ))

            connection.commit()

        except Exception as e:
            print(f"[WARNING] Failed to save DPP + k-NN result: {e}")

    def run_multithreaded_dpp_knn_optimization(self, sample_size: int = 500):
        """Run multithreaded DPP + k-NN optimization"""

        print("=" * 80)
        print("MULTITHREADED DPP + K-NN OPTIMIZATION")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Using {self.max_workers} worker threads")

        # Create table
        if not self.create_dpp_knn_results_table():
            return

        # Load data
        print(f"\n[STEP 1] Loading ground truth examples...")

        # Load separate pools for testing and candidates to avoid data leakage
        print("[INFO] Loading test examples...")
        test_examples = self.load_ground_truth_examples(limit=250)  # Fixed test set
        if not test_examples:
            print("[ERROR] No test examples loaded")
            return
        print(f"[INFO] Loaded {len(test_examples)} test examples")

        print("[INFO] Loading candidate pool with stratified sampling...")
        # Load extra samples to account for test example removal
        all_examples = self.load_ground_truth_examples(stratified_sample=1250)  # Stratified candidates
        if not all_examples:
            print("[ERROR] No candidate examples loaded")
            return

        # Remove any test examples from candidate pool to prevent data leakage
        test_case_ids = {ex['case_id'] for ex in test_examples}
        all_examples = [ex for ex in all_examples if ex['case_id'] not in test_case_ids]

        print(f"[INFO] Loaded {len(all_examples)} stratified examples as candidate pool (after removing test examples)")
        print(f"[INFO] DPP caching optimization enabled for faster computation")

        self._initialize_global_dpp_selector(all_examples)

        # Parameter grids
        n_dpp_grid = list(range(0, 11))  # 0 to 10
        k_nn_grid = list(range(0, 11))   # 0 to 10

        # Create parameter combinations
        param_combinations = []
        experiment_timestamp = datetime.now()

        for n_dpp in n_dpp_grid:
            for k_nn in k_nn_grid:
                # Skip combinations where total examples > 10 (too slow and inaccurate)
                if n_dpp + k_nn <= 10:
                    param_combinations.append((n_dpp, k_nn, test_examples, len(all_examples), experiment_timestamp))

        print(f"\n[STEP 2] Pre-computing shared candidate caches...")

        max_n_dpp = max(n_dpp_grid)
        max_k_nn = max(k_nn_grid)
        max_total_examples = max((n_dpp + k_nn) for n_dpp, k_nn, *_ in param_combinations) if param_combinations else 0

        precomputed_candidate_data = self.precompute_dpp_knn_resources(
            test_examples,
            all_examples,
            max_n_dpp=max_n_dpp,
            max_k_nn=max_k_nn,
            max_total_examples=max_total_examples
        )

        print(f"\n[STEP 3] Testing {len(param_combinations)} DPP + k-NN combinations with {self.max_workers} threads...")

        results = []
        completed_count = 0

        # Run multithreaded optimization
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks with shared precomputed candidate data
            future_to_params = {
                executor.submit(
                    self.test_dpp_knn_combination_with_precomputed,
                    params,
                    precomputed_candidate_data
                ): params
                for params in param_combinations
            }

            # Process completed tasks
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                n_dpp, k_nn = params[0], params[1]

                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1

                    print(f"[PROGRESS] {completed_count}/{len(param_combinations)}: n_dpp={n_dpp}, k_nn={k_nn} (total={result['total_shots']}) -> {result['accuracy']:.3f} accuracy ({result['execution_time']:.1f}s)")

                except Exception as e:
                    print(f"[ERROR] DPP + k-NN combination n_dpp={n_dpp}, k_nn={k_nn} failed: {e}")

        # Find and display best results
        if results:
            # Sort by accuracy (descending), then by total_shots (ascending for tie-breaking)
            sorted_results = sorted(results, key=lambda x: (-x['accuracy'], x['total_shots']))

            best_result = sorted_results[0]
            print(f"\n[SUCCESS] Best multithreaded DPP + k-NN optimization:")
            print(f"  n_dpp = {best_result['n_dpp']}")
            print(f"  k_nn = {best_result['k_nn']}")
            print(f"  total_shots = {best_result['total_shots']}")
            print(f"  Accuracy = {best_result['accuracy']:.3f} ({best_result['correct']}/{best_result['total']})")

            # Display top 5 results
            print(f"\n[TOP 5] DPP + k-NN Optimization Results (highest accuracy → lowest):")
            print("=" * 80)
            for i, result in enumerate(sorted_results[:5], 1):
                accuracy_pct = result['accuracy'] * 100
                print(f"{i:2}. n_dpp={result['n_dpp']:2}, k_nn={result['k_nn']:2} (total={result['total_shots']:2}) → "
                      f"{accuracy_pct:5.1f}% accuracy ({result['correct']:2}/{result['total']:2}) "
                      f"[{result['execution_time']:.1f}s]")

        print(f"\n[SUCCESS] Multithreaded DPP + k-NN optimization complete!")

        # Generate heatmap
        self.generate_dpp_knn_heatmap(results, experiment_timestamp)


class MultithreadedCategoryFilteredOptimizer(MultithreadedNShotOptimizer):
    """Category-filtered n-shot optimization: filter by product/issue, then use DPP + k-NN"""

    def generate_category_filtered_heatmap(self, results: List[Dict], experiment_timestamp: datetime):
        """Generate 2D heatmap for category-filtered n vs k optimization results"""
        try:
            print(f"\n[HEATMAP] Generating category-filtered optimization heatmap...")

            # Create result matrix
            n_values = sorted(list(set([r['n'] for r in results])))
            k_values = sorted(list(set([r['k'] for r in results])))

            # Initialize matrix with NaN
            accuracy_matrix = np.full((len(n_values), len(k_values)), np.nan)

            # Fill matrix with accuracy values
            for result in results:
                n_idx = n_values.index(result['n'])
                k_idx = k_values.index(result['k'])
                accuracy_matrix[n_idx, k_idx] = result['accuracy']

            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(accuracy_matrix,
                       xticklabels=[f"{k}" for k in k_values],
                       yticklabels=[f"{n}" for n in n_values],
                       annot=True,
                       fmt='.3f',
                       cmap='viridis',
                       cbar_kws={'label': 'Accuracy'})

            plt.title(f'Category-Filtered N-Shot Optimization Accuracy Heatmap\n{experiment_timestamp.strftime("%Y-%m-%d %H:%M:%S")}\n(Product/Issue filtered + DPP + k-NN)')
            plt.xlabel('k-NN Examples')
            plt.ylabel('DPP Examples per Tier')
            plt.tight_layout()

            # Save plot
            timestamp_str = experiment_timestamp.strftime("%Y%m%d_%H%M%S")

            # Create directory if it doesn't exist
            output_dir = Path("nshot_optimisation")
            output_dir.mkdir(exist_ok=True)

            filename = f"category_filtered_optimization_heatmap_{timestamp_str}.png"
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"[SUCCESS] Category-filtered heatmap saved as: {filepath}")

        except Exception as e:
            print(f"[ERROR] Failed to generate category-filtered heatmap: {e}")

    def create_category_filtered_results_table(self):
        """Create/verify table for category-filtered optimization results"""
        connection = psycopg2.connect(**self.db_config)
        cursor = connection.cursor()

        try:
            # Create table if it doesn't exist
            create_table_query = """
                CREATE TABLE IF NOT EXISTS nshot_category_filtered_results (
                    id SERIAL PRIMARY KEY,
                    experiment_timestamp TIMESTAMP,
                    experiment_type VARCHAR(50),
                    n_value INTEGER,
                    k_value INTEGER,
                    sample_size INTEGER,
                    total_ground_truth_examples INTEGER,
                    correct_predictions INTEGER,
                    total_predictions INTEGER,
                    accuracy_score DECIMAL(5,4),
                    confident_decisions INTEGER,
                    uncertain_decisions INTEGER,
                    need_info_decisions INTEGER,
                    confident_rate DECIMAL(5,4),
                    uncertain_rate DECIMAL(5,4),
                    need_info_rate DECIMAL(5,4),
                    execution_time_seconds DECIMAL(8,2),
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """
            cursor.execute(create_table_query)
            connection.commit()
            print("[SUCCESS] Category-filtered results table verified/created")

            cursor.close()
            connection.close()
            return True

        except Exception as e:
            print(f"[ERROR] Failed to create category-filtered results table: {e}")
            cursor.close()
            connection.close()
            return False

    def filter_examples_by_category(self, query_example: Dict, all_examples: List[Dict]) -> List[Dict]:
        """Filter examples to match query's product and issue category"""
        query_product = query_example.get('product', '').lower().strip()
        query_issue = query_example.get('issue', '').lower().strip()

        filtered_examples = []
        for example in all_examples:
            # Skip the query example itself
            if example['case_id'] == query_example['case_id']:
                continue

            example_product = example.get('product', '').lower().strip()
            example_issue = example.get('issue', '').lower().strip()

            # Must match both product and issue
            if example_product == query_product and example_issue == query_issue:
                filtered_examples.append(example)

        return filtered_examples

    def calculate_category_tier_statistics(self, product: str, issue: str) -> Dict[str, float]:
        """Calculate tier distribution statistics for a specific product/issue category"""
        try:
            connection, cursor = self.get_thread_db_connection()

            # Query to get tier distribution for the specific category
            query = """
                SELECT simplified_ground_truth_tier, COUNT(*) as count
                FROM ground_truth
                WHERE LOWER(TRIM(product)) = LOWER(TRIM(%s))
                  AND LOWER(TRIM(issue)) = LOWER(TRIM(%s))
                  AND simplified_ground_truth_tier >= 0
                  AND vector_embeddings IS NOT NULL
                  AND vector_embeddings != ''
                GROUP BY simplified_ground_truth_tier
                ORDER BY simplified_ground_truth_tier
            """

            cursor.execute(query, (product, issue))
            results = cursor.fetchall()

            # Initialize counts
            tier_counts = {0: 0, 1: 0, 2: 0}
            total_count = 0

            # Process results
            for tier, count in results:
                if tier in tier_counts:
                    tier_counts[tier] = count
                    total_count += count

            # Calculate percentages (fall back to global stats if no data)
            if total_count == 0:
                print(f"[WARNING] No data found for category product='{product}', issue='{issue}'. Using global statistics.")
                return {
                    'tier_0_percent': 65.6,
                    'tier_1_percent': 31.8,
                    'tier_2_percent': 2.6
                }

            category_stats = {
                'tier_0_percent': (tier_counts[0] / total_count) * 100,
                'tier_1_percent': (tier_counts[1] / total_count) * 100,
                'tier_2_percent': (tier_counts[2] / total_count) * 100
            }

            print(f"[INFO] Category stats for {product}/{issue}: "
                  f"Tier 0: {category_stats['tier_0_percent']:.1f}%, "
                  f"Tier 1: {category_stats['tier_1_percent']:.1f}%, "
                  f"Tier 2: {category_stats['tier_2_percent']:.1f}% "
                  f"(n={total_count})")

            return category_stats

        except Exception as e:
            print(f"[ERROR] Failed to calculate category statistics for {product}/{issue}: {e}")
            # Fall back to global statistics
            return {
                'tier_0_percent': 65.6,
                'tier_1_percent': 31.8,
                'tier_2_percent': 2.6
            }

    def select_dpp_examples_by_tier(self, examples: List[Dict], n_per_tier: int) -> List[Dict]:
        """Select n DPP examples for each unique remedy tier in the examples."""
        if n_per_tier == 0:
            return []

        examples_by_tier: Dict[str, List[Dict]] = {}
        for example in examples:
            tier = example.get('tier', 'unknown')
            examples_by_tier.setdefault(tier, []).append(example)

        selected_examples: List[Dict] = []

        for tier_examples in examples_by_tier.values():
            valid = [ex for ex in tier_examples if ex.get('embedding') is not None]
            if not valid:
                continue

            if len(valid) <= n_per_tier:
                selected_examples.extend(valid)
                continue

            embeddings = np.array([ex['embedding'] for ex in valid], dtype=np.float32)
            from hybrid_dpp import HybridDPP
            selector = HybridDPP(embeddings, random_state=42)
            indices = selector.select(min(n_per_tier, len(valid)))
            for idx in indices:
                if idx < len(valid):
                    selected_examples.append(valid[idx])

        return selected_examples

    def select_knn_examples(self, query_example: Dict, candidates: List[Dict], k: int) -> List[Dict]:
        """Select k nearest neighbor examples using hybrid similarity"""
        if k == 0 or len(candidates) == 0:
            return []

        if hasattr(self, 'similarity_calculator') and self.use_hybrid_similarity:
            # Use hybrid similarity
            similarities = []
            for candidate in candidates:
                sim = self.similarity_calculator.calculate_hybrid_similarity(query_example, candidate)
                similarities.append(sim)

            # Get top k
            k_actual = min(k, len(candidates))
            top_k_indices = np.argsort(similarities)[-k_actual:][::-1]
            return [candidates[i] for i in top_k_indices]
        else:
            # Use pure semantic similarity
            query_embedding = query_example.get('embedding')
            if query_embedding is None:
                return []

            candidate_embeddings = np.array([ex['embedding'] for ex in candidates])

            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embedding.reshape(1, -1), candidate_embeddings).flatten()

            k_actual = min(k, len(candidates))
            top_k_indices = np.argsort(similarities)[-k_actual:][::-1]

            return [candidates[i] for i in top_k_indices]

    def test_category_filtered_combination(self, params: Tuple) -> Dict:
        """Test a single (n, k) parameter combination with category filtering"""
        n, k, test_examples, all_examples, experiment_timestamp = params

        start_time = time.time()
        correct_predictions = 0
        total_predictions = 0
        confident_decisions = 0
        uncertain_decisions = 0
        need_info_decisions = 0

        try:
            for target_example in test_examples:
                target_tier = target_example['tier']

                # Step 1: Filter examples by product and issue category
                filtered_candidates = self.filter_examples_by_category(target_example, all_examples)

                # Skip if we don't have enough examples for the requested configuration
                # We need at least n*3 (for up to 3 tiers) + k examples
                min_required = k
                if n > 0:
                    # Count unique tiers to get actual requirement
                    unique_tiers = len(set(ex.get('tier', 0) for ex in filtered_candidates))
                    min_required = n * min(unique_tiers, 3) + k

                # Skip if insufficient examples (but allow zero-shot)
                if (n > 0 or k > 0) and len(filtered_candidates) < min_required:
                    # Skip this test case - not enough examples in category
                    continue

                # Step 2: Select n DPP examples per tier using pure diversity (category-static, not query-specific)
                dpp_examples = []
                if n > 0 and len(filtered_candidates) > 0:
                    dpp_examples = self.select_dpp_examples_by_tier(filtered_candidates, n)

                # Step 3: Select k nearest neighbors from filtered candidates (excluding DPP examples)
                knn_examples = []
                if k > 0:
                    remaining_candidates = [ex for ex in filtered_candidates
                                          if ex['case_id'] not in [dpp_ex['case_id'] for dpp_ex in dpp_examples]]
                    if remaining_candidates:
                        knn_examples = self.select_knn_examples(target_example, remaining_candidates, k)

                # Combine examples
                nshot_examples = dpp_examples + knn_examples

                # Generate prompt and get prediction
                target_case = {'complaint_text': target_example['complaint_text']}
                formatted_examples = [
                    {
                        'complaint_text': ex['complaint_text'],
                        'tier': ex['tier']
                    }
                    for ex in nshot_examples
                ]

                # Calculate category-specific tier statistics using product+issue combination
                target_product = target_example.get('product', '')
                target_issue = target_example.get('issue', '')
                category_tier_stats = self.calculate_combined_category_tier_statistics(target_product, target_issue)

                system_prompt, user_prompt = self.prompt_generator.generate_prompts(
                    target_case=target_case,
                    nshot_examples=formatted_examples,
                    persona=None,
                    bias_strategy=BiasStrategy.CHAIN_OF_THOUGHT,
                    category_tier_stats=category_tier_stats
                )

                # Get LLM prediction
                analysis = self.call_gpt4o_mini_analysis(system_prompt, user_prompt, target_example.get('case_id'))

                if analysis is None:
                    continue

                total_predictions += 1

                # Check accuracy
                predicted_tier = analysis.tier
                if predicted_tier == target_tier:
                    correct_predictions += 1

                # Count confidence levels
                if analysis.confidence.lower() == 'confident':
                    confident_decisions += 1
                elif analysis.confidence.lower() == 'uncertain':
                    uncertain_decisions += 1
                elif analysis.information_needed:
                    need_info_decisions += 1

        except Exception as e:
            print(f"[ERROR] Category-filtered test failed for n={n}, k={k}: {e}")

        # Calculate metrics
        execution_time = time.time() - start_time

        # If no predictions were made, return None or NaN for accuracy
        if total_predictions == 0:
            accuracy = float('nan')  # Use NaN to indicate no data available
        else:
            accuracy = correct_predictions / total_predictions

        result = {
            'n': n,
            'k': k,
            'correct': correct_predictions,
            'total': total_predictions,
            'accuracy': accuracy,
            'confident_decisions': confident_decisions,
            'uncertain_decisions': uncertain_decisions,
            'need_info_decisions': need_info_decisions,
            'execution_time': execution_time,
            'experiment_timestamp': experiment_timestamp
        }

        # Only save to database if we have actual results
        if total_predictions > 0:
            self.save_category_filtered_result_to_db(result, len(all_examples), len(test_examples))

        return result

    def save_category_filtered_result_to_db(self, result: Dict, total_ground_truth: int, sample_size: int):
        """Save category-filtered result to database"""
        try:
            connection, cursor = self.get_thread_db_connection()

            confident_rate = result['confident_decisions'] / result['total'] if result['total'] > 0 else 0
            uncertain_rate = result['uncertain_decisions'] / result['total'] if result['total'] > 0 else 0
            need_info_rate = result['need_info_decisions'] / result['total'] if result['total'] > 0 else 0

            insert_query = """
                INSERT INTO nshot_category_filtered_results (
                    experiment_timestamp, experiment_type, n_value, k_value,
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
                result['experiment_timestamp'],
                'category_filtered',
                result['n'],
                result['k'],
                sample_size,
                total_ground_truth,
                result['correct'],
                result['total'],
                result['accuracy'],
                result['confident_decisions'],
                result['uncertain_decisions'],
                result['need_info_decisions'],
                confident_rate,
                uncertain_rate,
                need_info_rate,
                result['execution_time'],
                f"Category-filtered: n={result['n']}, k={result['k']}"
            ))

            connection.commit()

        except Exception as e:
            print(f"[WARNING] Failed to save category-filtered result for n={result['n']}, k={result['k']}: {e}")

    def run_multithreaded_category_filtered_optimization(self, sample_size: int = 500):
        """Run multithreaded category-filtered optimization"""

        print("=" * 80)
        print("MULTITHREADED CATEGORY-FILTERED OPTIMIZATION")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Using {self.max_workers} worker threads")

        # Create table
        if not self.create_category_filtered_results_table():
            return

        # Load data
        print(f"\n[STEP 1] Loading ground truth examples...")

        # Load separate pools for testing and candidates to avoid data leakage
        print("[INFO] Loading test examples...")
        test_examples = self.load_ground_truth_examples(limit=250)  # Fixed test set
        if not test_examples:
            print("[ERROR] No test examples loaded")
            return
        print(f"[INFO] Loaded {len(test_examples)} test examples")

        print("[INFO] Loading candidate pool with enhanced stratified sampling (min 20 per category)...")
        # Load with higher minimum per category to ensure sufficient examples for filtering
        all_examples = self.load_stratified_sample(target_size=2500, min_per_category=20)
        if not all_examples:
            print("[ERROR] No candidate examples loaded")
            return

        # Remove any test examples from candidate pool to prevent data leakage
        test_case_ids = {ex['case_id'] for ex in test_examples}
        all_examples = [ex for ex in all_examples if ex['case_id'] not in test_case_ids]

        print(f"[INFO] Loaded {len(all_examples)} stratified examples as candidate pool (after removing test examples)")

        # Parameter grids
        n_grid = [0, 1, 2, 3, 4, 5]  # DPP examples per tier
        k_grid = [0, 1, 2, 3, 4, 5]  # k-NN examples

        # Create parameter combinations
        param_combinations = []
        experiment_timestamp = datetime.now()

        for n in n_grid:
            for k in k_grid:
                param_combinations.append((n, k, test_examples, all_examples, experiment_timestamp))

        print(f"\n[STEP 2] Testing {len(param_combinations)} category-filtered combinations with {self.max_workers} threads...")

        results = []
        completed_count = 0

        # Run multithreaded optimization
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_params = {
                executor.submit(self.test_category_filtered_combination, params): params
                for params in param_combinations
            }

            # Process completed tasks
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                n, k = params[0], params[1]

                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1

                    print(f"[PROGRESS] {completed_count}/{len(param_combinations)}: n={n}, k={k} -> {result['accuracy']:.3f} accuracy ({result['execution_time']:.1f}s)")

                except Exception as e:
                    print(f"[ERROR] Category-filtered combination n={n}, k={k} failed: {e}")

        # Find and display best results
        if results:
            # Sort by accuracy (descending), then by total examples (n+k, ascending for tie-breaking)
            sorted_results = sorted(results, key=lambda x: (-x['accuracy'], x['n'] + x['k']))

            best_result = sorted_results[0]
            print(f"\n[SUCCESS] Best multithreaded category-filtered optimization:")
            print(f"  n = {best_result['n']}")
            print(f"  k = {best_result['k']}")
            print(f"  Accuracy = {best_result['accuracy']:.3f} ({best_result['correct']}/{best_result['total']})")

            # Display top 5 results
            print(f"\n[TOP 5] Category-Filtered Optimization Results (highest accuracy → lowest):")
            print("=" * 70)
            for i, result in enumerate(sorted_results[:5], 1):
                accuracy_pct = result['accuracy'] * 100
                total_examples = result['n'] + result['k']
                print(f"{i:2}. n={result['n']:1}, k={result['k']:1} (total={total_examples:1}) → "
                      f"{accuracy_pct:5.1f}% accuracy ({result['correct']:2}/{result['total']:2}) "
                      f"[{result['execution_time']:.1f}s]")

        print(f"\n[SUCCESS] Multithreaded category-filtered optimization complete!")

        # Generate heatmap
        self.generate_category_filtered_heatmap(results, experiment_timestamp)


def main():
    """Main function to run multithreaded optimizations with optional bypassing"""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run multithreaded n-shot optimization experiments')
    parser.add_argument('--skip-alpha', action='store_true',
                        help='Skip alpha optimization (useful for testing other optimizations)')
    parser.add_argument('--skip-dpp-knn', action='store_true',
                        help='Skip DPP + k-NN optimization')
    parser.add_argument('--skip-category', action='store_true',
                        help='Skip category-filtered optimization')
    parser.add_argument('--workers', type=int, default=12,
                        help='Number of worker threads (default: 12)')
    parser.add_argument('--sample-size', type=int, default=500,
                        help='Sample size for testing (default: 500)')

    args = parser.parse_args()

    print("=" * 80)
    print("MULTITHREADED N-SHOT OPTIMIZATION SUITE")
    print("=" * 80)

    # Configuration summary
    optimizations_to_run = []
    if not args.skip_alpha:
        optimizations_to_run.append("Alpha Optimization")
    if not args.skip_dpp_knn:
        optimizations_to_run.append("DPP + k-NN Optimization")
    if not args.skip_category:
        optimizations_to_run.append("Category-Filtered Optimization")

    print(f"Workers: {args.workers}")
    print(f"Sample size: {args.sample_size}")
    print(f"Optimizations to run: {', '.join(optimizations_to_run) if optimizations_to_run else 'None'}")
    print()

    if not optimizations_to_run:
        print("[WARNING] All optimizations skipped. Nothing to run.")
        return

    # Set number of worker threads
    max_workers = args.workers

    # Initialize variables for sharing between optimizations
    dpp_knn_optimizer = None

    # Run Alpha Optimization
    if not args.skip_alpha:
        print("\n" + "="*50)
        print("RUNNING MULTITHREADED ALPHA OPTIMIZATION")
        print("="*50)

        alpha_optimizer = MultithreadedSingleAlphaOptimizer(max_workers=max_workers)
        try:
            alpha_optimizer.run_multithreaded_alpha_optimization(sample_size=args.sample_size)
        except Exception as e:
            print(f"[ERROR] Alpha optimization failed: {e}")
    else:
        print("\n[SKIPPED] Alpha optimization skipped by user request")

    # Run DPP + k-NN Optimization
    if not args.skip_dpp_knn:
        print("\n" + "="*50)
        print("RUNNING MULTITHREADED DPP + K-NN OPTIMIZATION")
        print("="*50)

        dpp_knn_optimizer = MultithreadedDPPKNNOptimizer(max_workers=max_workers)
        try:
            dpp_knn_optimizer.run_multithreaded_dpp_knn_optimization(sample_size=args.sample_size)
        except Exception as e:
            print(f"[ERROR] DPP + k-NN optimization failed: {e}")
    else:
        print("\n[SKIPPED] DPP + k-NN optimization skipped by user request")

    # Run Category-Filtered Optimization
    if not args.skip_category:
        print("\n" + "="*50)
        print("RUNNING MULTITHREADED CATEGORY-FILTERED OPTIMIZATION")
        print("="*50)

        category_filtered_optimizer = MultithreadedCategoryFilteredOptimizer(max_workers=max_workers)

        # Set up hybrid similarity if available from DPP optimizer
        if dpp_knn_optimizer and hasattr(dpp_knn_optimizer, 'similarity_calculator'):
            category_filtered_optimizer.similarity_calculator = dpp_knn_optimizer.similarity_calculator
            category_filtered_optimizer.use_hybrid_similarity = True
        else:
            # Initialize our own hybrid similarity if DPP wasn't run
            category_filtered_optimizer.use_hybrid_similarity = True

        try:
            category_filtered_optimizer.run_multithreaded_category_filtered_optimization(sample_size=args.sample_size)
        except Exception as e:
            print(f"[ERROR] Category-filtered optimization failed: {e}")
    else:
        print("\n[SKIPPED] Category-filtered optimization skipped by user request")

    print("\n" + "="*80)
    print("ALL MULTITHREADED OPTIMIZATIONS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()






