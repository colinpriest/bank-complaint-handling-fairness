#!/usr/bin/env python3
"""
Tier-Stratified DPP+k-NN Hybrid Algorithm

Enhanced version of the DPP+k-NN algorithm that ensures at least one example from each tier
is included in the n-shot examples. When a tier is missing, adds the most central example
of that tier for the specific product.
"""

import numpy as np
import psycopg2
import os
from dotenv import load_dotenv
from typing import List, Optional, Dict, Tuple, Set
import json
from hybrid_dpp import HybridDPP

load_dotenv()


class TierStratifiedDPP:
    """
    Tier-stratified DPP+k-NN selector that ensures representation from all tiers.

    Pre-computes central examples for each tier-product combination and uses them
    to fill gaps when DPP+k-NN selection misses certain tiers.
    """

    def __init__(self, db_config: Dict, cache_file: str = "tier_central_examples.json", debug: bool = False):
        """
        Initialize the tier-stratified selector.

        Args:
            db_config: Database configuration dictionary
            cache_file: File to cache pre-computed central examples
            debug: Whether to print debug messages
        """
        self.db_config = db_config
        self.cache_file = cache_file
        self.debug = debug
        self.central_examples_cache = {}

        # Load or compute central examples
        self._load_or_compute_central_examples()

    def _get_db_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_config)

    def _load_or_compute_central_examples(self):
        """Load central examples from cache or compute them"""
        if os.path.exists(self.cache_file):
            if self.debug:
                print(f"Loading central examples from {self.cache_file}")
            try:
                with open(self.cache_file, 'r') as f:
                    self.central_examples_cache = json.load(f)
                if self.debug:
                    print(f"Loaded {len(self.central_examples_cache)} product categories")
                return
            except Exception as e:
                if self.debug:
                    print(f"Error loading cache: {e}. Recomputing...")

        if self.debug:
            print("Computing central examples for each tier-product combination...")
        self._compute_central_examples()

    def _compute_central_examples(self):
        """
        Pre-compute the most central example for each tier-product combination.

        Central example = example with minimum average cosine distance to all other
        examples in the same tier and product category.
        """
        connection = self._get_db_connection()
        cursor = connection.cursor()

        try:
            # Get all product categories
            cursor.execute("""
                SELECT DISTINCT product
                FROM ground_truth
                WHERE product IS NOT NULL
                ORDER BY product
            """)
            products = [row[0] for row in cursor.fetchall()]

            if self.debug:
                print(f"Processing {len(products)} products...")

            for product_idx, product in enumerate(products):
                if self.debug:
                    print(f"Processing product {product_idx + 1}/{len(products)}: {product}")

                product_cache = {}

                # For each tier
                for tier in [0, 1, 2]:
                    # Get all examples for this tier-product combination
                    cursor.execute("""
                        SELECT
                            gt.case_id,
                            gt.consumer_complaint_text,
                            gt.vector_embeddings
                        FROM ground_truth gt
                        WHERE gt.simplified_ground_truth_tier = %s
                        AND gt.product = %s
                        AND gt.vector_embeddings IS NOT NULL
                        AND gt.consumer_complaint_text IS NOT NULL
                    """, (tier, product))

                    examples = cursor.fetchall()

                    if len(examples) == 0:
                        if self.debug:
                            print(f"  No examples found for Tier {tier}")
                        continue

                    if len(examples) == 1:
                        case_id, complaint_text, embedding_str = examples[0]
                        product_cache[f"tier_{tier}"] = {
                            'case_id': case_id,
                            'complaint_text': complaint_text,
                            'tier': tier,
                            'centrality_score': 1.0  # Only one example
                        }
                        if self.debug:
                            print(f"  Tier {tier}: 1 example (automatic central)")
                        continue

                    if self.debug:
                        print(f"  Tier {tier}: {len(examples)} examples, computing centrality...")

                    # Extract embeddings
                    embeddings = []
                    valid_examples = []

                    for case_id, complaint_text, embedding_str in examples:
                        try:
                            # Parse embedding string (assuming JSON format)
                            embedding = json.loads(embedding_str)
                            if isinstance(embedding, list) and len(embedding) > 0:
                                embeddings.append(np.array(embedding))
                                valid_examples.append((case_id, complaint_text, tier))
                        except (json.JSONDecodeError, ValueError) as e:
                            if self.debug:
                                print(f"    Warning: Invalid embedding for case {case_id}: {e}")
                            continue

                    if len(embeddings) == 0:
                        if self.debug:
                            print(f"  Tier {tier}: No valid embeddings found")
                        continue

                    # Compute central example
                    embeddings_matrix = np.array(embeddings)
                    central_idx = self._find_central_example(embeddings_matrix)

                    case_id, complaint_text, tier = valid_examples[central_idx]

                    # Calculate centrality score (average cosine similarity to all others)
                    central_embedding = embeddings_matrix[central_idx]
                    similarities = []
                    for i, emb in enumerate(embeddings_matrix):
                        if i != central_idx:
                            sim = np.dot(central_embedding, emb) / (np.linalg.norm(central_embedding) * np.linalg.norm(emb))
                            similarities.append(sim)

                    centrality_score = np.mean(similarities) if similarities else 1.0

                    product_cache[f"tier_{tier}"] = {
                        'case_id': case_id,
                        'complaint_text': complaint_text,
                        'tier': tier,
                        'centrality_score': float(centrality_score)
                    }

                    if self.debug:
                        print(f"  Tier {tier}: Central example found (case_id={case_id}, centrality={centrality_score:.3f})")

                self.central_examples_cache[product] = product_cache

            # Save cache
            with open(self.cache_file, 'w') as f:
                json.dump(self.central_examples_cache, f, indent=2)

            if self.debug:
                print(f"Central examples computed and saved to {self.cache_file}")

        finally:
            connection.close()

    def _find_central_example(self, embeddings: np.ndarray) -> int:
        """
        Find the most central example in a set of embeddings.

        Args:
            embeddings: Array of embeddings (n_examples, embedding_dim)

        Returns:
            Index of the most central example
        """
        if len(embeddings) == 1:
            return 0

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = embeddings / norms

        # Compute cosine similarity matrix
        similarity_matrix = normalized @ normalized.T

        # Find example with highest average similarity to all others
        # (excluding self-similarity)
        avg_similarities = []
        for i in range(len(embeddings)):
            similarities = similarity_matrix[i]
            # Exclude self-similarity
            other_similarities = np.concatenate([similarities[:i], similarities[i+1:]])
            avg_similarities.append(np.mean(other_similarities))

        return int(np.argmax(avg_similarities))

    def select_stratified_examples(self,
                                   candidates: List[Dict],
                                   query_embedding: np.ndarray,
                                   target_product: str,
                                   n_dpp: int = 1,
                                   k_nn: int = 2) -> List[Dict]:
        """
        Select examples using tier-stratified DPP+k-NN algorithm.

        Args:
            candidates: List of candidate examples with embeddings and tier info
            query_embedding: Embedding of the query/target case
            target_product: Product category for the target case
            n_dpp: Number of examples to select via DPP
            k_nn: Number of examples to select via k-NN

        Returns:
            List of selected examples ensuring tier representation
        """
        total_examples = n_dpp + k_nn

        if total_examples == 0:
            return []

        # Step 1: Apply standard DPP+k-NN selection
        selected_examples = self._apply_hybrid_dpp_knn(
            candidates, query_embedding, n_dpp, k_nn
        )

        # Step 2: Check tier representation
        selected_tiers = set(example.get('tier', -1) for example in selected_examples)
        missing_tiers = set([0, 1, 2]) - selected_tiers

        if not missing_tiers:
            if self.debug:
                print(f"All tiers represented in selection: {sorted(selected_tiers)}")
            return selected_examples

        if self.debug:
            print(f"Missing tiers: {sorted(missing_tiers)}, filling with central examples")

        # Step 3: Fill missing tiers with central examples
        for missing_tier in sorted(missing_tiers):
            central_example = self._get_central_example(target_product, missing_tier)

            if central_example:
                # Replace the least relevant example or add if we have room
                if len(selected_examples) < total_examples:
                    selected_examples.append(central_example)
                    if self.debug:
                        print(f"Added central Tier {missing_tier} example (case_id={central_example.get('case_id')})")
                else:
                    # Replace the least relevant example
                    # (For simplicity, replace the last k-NN selected example)
                    if len(selected_examples) > 0:
                        replaced = selected_examples.pop()
                        selected_examples.append(central_example)
                        if self.debug:
                            print(f"Replaced example (case_id={replaced.get('case_id')}) with central Tier {missing_tier} example (case_id={central_example.get('case_id')})")

        # Ensure we don't exceed the target number of examples
        selected_examples = selected_examples[:total_examples]

        final_tiers = [example.get('tier', -1) for example in selected_examples]
        if self.debug:
            print(f"Final tier distribution: {sorted(set(final_tiers))}")

        return selected_examples

    def _apply_hybrid_dpp_knn(self,
                              candidates: List[Dict],
                              query_embedding: np.ndarray,
                              n_dpp: int,
                              k_nn: int) -> List[Dict]:
        """Apply the original hybrid DPP+k-NN selection"""
        selected_examples = []
        used_indices = set()

        # Extract embeddings from candidates
        embeddings = []
        for candidate in candidates:
            embedding = candidate.get('embedding')
            if embedding is not None:
                embeddings.append(np.array(embedding))
            else:
                embeddings.append(np.zeros(query_embedding.shape[0]))  # Fallback

        if len(embeddings) == 0:
            return []

        embeddings_matrix = np.array(embeddings)

        # DPP selection for diversity
        if n_dpp > 0 and len(candidates) > 0:
            try:
                dpp_selector = HybridDPP(embeddings_matrix, random_state=42)
                dpp_indices = dpp_selector.select(n_dpp)

                for idx in dpp_indices:
                    if idx < len(candidates):
                        selected_examples.append(candidates[idx])
                        used_indices.add(idx)
            except Exception as e:
                if self.debug:
                    print(f"DPP selection failed: {e}, falling back to random selection")
                # Fallback to random selection
                available_indices = list(range(len(candidates)))
                np.random.shuffle(available_indices)
                for idx in available_indices[:n_dpp]:
                    selected_examples.append(candidates[idx])
                    used_indices.add(idx)

        # k-NN selection for relevance
        if k_nn > 0:
            remaining_candidates = [candidates[i] for i in range(len(candidates)) if i not in used_indices]
            remaining_embeddings = [embeddings[i] for i in range(len(embeddings)) if i not in used_indices]

            if remaining_candidates and len(remaining_embeddings) > 0:
                # Compute cosine similarities to query
                remaining_matrix = np.array(remaining_embeddings)

                # Normalize embeddings
                query_norm = np.linalg.norm(query_embedding)
                remaining_norms = np.linalg.norm(remaining_matrix, axis=1)

                # Avoid division by zero
                if query_norm > 0 and np.all(remaining_norms > 0):
                    similarities = np.dot(remaining_matrix, query_embedding) / (remaining_norms * query_norm)
                else:
                    similarities = np.random.random(len(remaining_candidates))  # Fallback

                # Select top k_nn most similar
                top_indices = np.argsort(similarities)[-k_nn:][::-1]

                for idx in top_indices:
                    if idx < len(remaining_candidates):
                        selected_examples.append(remaining_candidates[idx])

        return selected_examples

    def _get_central_example(self, product: str, tier: int) -> Optional[Dict]:
        """
        Get the pre-computed central example for a specific product-tier combination.

        Args:
            product: Product category
            tier: Target tier (0, 1, or 2)

        Returns:
            Central example dictionary or None if not found
        """
        product_cache = self.central_examples_cache.get(product, {})
        tier_key = f"tier_{tier}"

        if tier_key in product_cache:
            central_data = product_cache[tier_key]

            # Convert to the expected format
            return {
                'case_id': central_data['case_id'],
                'complaint_text': central_data['complaint_text'],
                'tier': central_data['tier'],
                'centrality_score': central_data['centrality_score'],
                'is_central_fallback': True  # Mark as a central fallback
            }

        # Fallback: try to find any example from this tier in the database
        if self.debug:
            print(f"No cached central example for {product} Tier {tier}, searching database...")
        return self._find_fallback_example(product, tier)

    def _find_fallback_example(self, product: str, tier: int) -> Optional[Dict]:
        """Find a fallback example from the database"""
        connection = self._get_db_connection()
        cursor = connection.cursor()

        try:
            cursor.execute("""
                SELECT case_id, consumer_complaint_text
                FROM ground_truth
                WHERE simplified_ground_truth_tier = %s
                AND product = %s
                AND consumer_complaint_text IS NOT NULL
                LIMIT 1
            """, (tier, product))

            result = cursor.fetchone()
            if result:
                case_id, complaint_text = result
                return {
                    'case_id': case_id,
                    'complaint_text': complaint_text,
                    'tier': tier,
                    'is_fallback': True
                }

        except Exception as e:
            if self.debug:
                print(f"Error finding fallback example: {e}")

        finally:
            connection.close()

        return None

    def get_cache_stats(self) -> Dict:
        """Get statistics about the cached central examples"""
        stats = {
            'total_products': len(self.central_examples_cache),
            'tier_coverage': {},
            'products_with_full_coverage': 0
        }

        for tier in [0, 1, 2]:
            stats['tier_coverage'][f'tier_{tier}'] = 0

        for product, product_cache in self.central_examples_cache.items():
            tiers_covered = set()
            for tier in [0, 1, 2]:
                tier_key = f'tier_{tier}'
                if tier_key in product_cache:
                    stats['tier_coverage'][tier_key] += 1
                    tiers_covered.add(tier)

            if len(tiers_covered) == 3:
                stats['products_with_full_coverage'] += 1

        return stats


def test_tier_stratified_dpp():
    """Test the tier-stratified DPP algorithm"""
    # Database configuration
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'fairness_analysis'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', '')
    }

    print("Initializing Tier-Stratified DPP...")
    selector = TierStratifiedDPP(db_config, debug=False)

    # Show cache statistics
    stats = selector.get_cache_stats()
    print("\nCache Statistics:")
    print(f"Total products: {stats['total_products']}")
    print(f"Products with full tier coverage: {stats['products_with_full_coverage']}")
    for tier, count in stats['tier_coverage'].items():
        print(f"  {tier}: {count} products")

    # Test example selection
    print("\nTesting example selection...")

    # Mock candidates (normally would come from database)
    mock_candidates = [
        {'case_id': 1, 'tier': 0, 'complaint_text': 'No action needed example', 'embedding': np.random.random(50)},
        {'case_id': 2, 'tier': 0, 'complaint_text': 'Another no action example', 'embedding': np.random.random(50)},
        {'case_id': 3, 'tier': 1, 'complaint_text': 'Non-monetary action example', 'embedding': np.random.random(50)},
        # Note: No Tier 2 examples to test the filling mechanism
    ]

    query_embedding = np.random.random(50)
    target_product = "Credit card or prepaid card"  # Common product

    selected = selector.select_stratified_examples(
        candidates=mock_candidates,
        query_embedding=query_embedding,
        target_product=target_product,
        n_dpp=1,
        k_nn=2
    )

    print(f"\nSelected {len(selected)} examples:")
    for i, example in enumerate(selected):
        print(f"  {i+1}. Case {example.get('case_id')}, Tier {example.get('tier')}, "
              f"Central: {example.get('is_central_fallback', False)}")


if __name__ == "__main__":
    test_tier_stratified_dpp()