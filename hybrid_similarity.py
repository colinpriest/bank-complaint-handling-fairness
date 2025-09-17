#!/usr/bin/env python3
"""
Hybrid Similarity Module for N-Shot Example Selection

This module implements a hybrid similarity metric that combines:
1. Semantic similarity (vector embedding cosine similarity)
2. Categorical feature matching (product, issue, company, etc.)
3. Numerical feature proximity (if applicable)

The hybrid approach can significantly improve example selection by considering
both the semantic content and structured metadata of complaints.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from enum import Enum


class SimilarityWeights:
    """Default weights for different similarity components"""
    SEMANTIC = 0.6  # Vector embedding similarity
    PRODUCT = 0.15  # Product category match
    ISSUE = 0.15    # Issue type match
    SUB_PRODUCT = 0.05  # Sub-product match
    SUB_ISSUE = 0.05   # Sub-issue match

    # Optional additional features
    COMPANY = 0.0   # Company match (usually 0 as we want cross-company generalization)
    STATE = 0.0     # State match (usually 0 to avoid geographic bias)


class HybridSimilarityCalculator:
    """
    Calculate hybrid similarity between complaints using both
    semantic embeddings and categorical features.
    """

    def __init__(self,
                 semantic_weight: float = SimilarityWeights.SEMANTIC,
                 product_weight: float = SimilarityWeights.PRODUCT,
                 issue_weight: float = SimilarityWeights.ISSUE,
                 sub_product_weight: float = SimilarityWeights.SUB_PRODUCT,
                 sub_issue_weight: float = SimilarityWeights.SUB_ISSUE,
                 company_weight: float = SimilarityWeights.COMPANY,
                 state_weight: float = SimilarityWeights.STATE):
        """
        Initialize hybrid similarity calculator with custom weights.

        Args:
            semantic_weight: Weight for vector embedding similarity (0-1)
            product_weight: Weight for product category match (0-1)
            issue_weight: Weight for issue type match (0-1)
            sub_product_weight: Weight for sub-product match (0-1)
            sub_issue_weight: Weight for sub-issue match (0-1)
            company_weight: Weight for company match (0-1)
            state_weight: Weight for state match (0-1)
        """
        # Normalize weights to sum to 1
        total_weight = (semantic_weight + product_weight + issue_weight +
                       sub_product_weight + sub_issue_weight +
                       company_weight + state_weight)

        if total_weight > 0:
            self.semantic_weight = semantic_weight / total_weight
            self.product_weight = product_weight / total_weight
            self.issue_weight = issue_weight / total_weight
            self.sub_product_weight = sub_product_weight / total_weight
            self.sub_issue_weight = sub_issue_weight / total_weight
            self.company_weight = company_weight / total_weight
            self.state_weight = state_weight / total_weight
        else:
            # Fallback to semantic only
            self.semantic_weight = 1.0
            self.product_weight = 0.0
            self.issue_weight = 0.0
            self.sub_product_weight = 0.0
            self.sub_issue_weight = 0.0
            self.company_weight = 0.0
            self.state_weight = 0.0

        print(f"[INFO] Hybrid similarity weights (normalized):")
        print(f"  - Semantic: {self.semantic_weight:.3f}")
        print(f"  - Product: {self.product_weight:.3f}")
        print(f"  - Issue: {self.issue_weight:.3f}")
        print(f"  - Sub-product: {self.sub_product_weight:.3f}")
        print(f"  - Sub-issue: {self.sub_issue_weight:.3f}")
        if self.company_weight > 0:
            print(f"  - Company: {self.company_weight:.3f}")
        if self.state_weight > 0:
            print(f"  - State: {self.state_weight:.3f}")

    def calculate_categorical_similarity(self, query: Dict, candidate: Dict) -> float:
        """
        Calculate similarity based on categorical feature matches.

        Args:
            query: Query complaint dictionary
            candidate: Candidate complaint dictionary

        Returns:
            Weighted categorical similarity score (0-1)
        """
        similarity = 0.0

        # Product match
        if self.product_weight > 0:
            query_product = query.get('product', '').lower().strip()
            candidate_product = candidate.get('product', '').lower().strip()
            if query_product and candidate_product:
                if query_product == candidate_product:
                    similarity += self.product_weight
                # Partial credit for related products
                elif self._are_products_related(query_product, candidate_product):
                    similarity += self.product_weight * 0.5

        # Issue match
        if self.issue_weight > 0:
            query_issue = query.get('issue', '').lower().strip()
            candidate_issue = candidate.get('issue', '').lower().strip()
            if query_issue and candidate_issue:
                if query_issue == candidate_issue:
                    similarity += self.issue_weight
                # Partial credit for similar issues
                elif self._are_issues_related(query_issue, candidate_issue):
                    similarity += self.issue_weight * 0.5

        # Sub-product match
        if self.sub_product_weight > 0:
            query_sub = query.get('sub_product', '').lower().strip()
            candidate_sub = candidate.get('sub_product', '').lower().strip()
            if query_sub and candidate_sub and query_sub == candidate_sub:
                similarity += self.sub_product_weight

        # Sub-issue match
        if self.sub_issue_weight > 0:
            query_sub_issue = query.get('sub_issue', '').lower().strip()
            candidate_sub_issue = candidate.get('sub_issue', '').lower().strip()
            if query_sub_issue and candidate_sub_issue and query_sub_issue == candidate_sub_issue:
                similarity += self.sub_issue_weight

        # Company match (usually not used)
        if self.company_weight > 0:
            query_company = query.get('company', '').lower().strip()
            candidate_company = candidate.get('company', '').lower().strip()
            if query_company and candidate_company and query_company == candidate_company:
                similarity += self.company_weight

        # State match (usually not used)
        if self.state_weight > 0:
            query_state = query.get('state', '').lower().strip()
            candidate_state = candidate.get('state', '').lower().strip()
            if query_state and candidate_state and query_state == candidate_state:
                similarity += self.state_weight

        return similarity

    def _are_products_related(self, product1: str, product2: str) -> bool:
        """
        Check if two products are related (for partial credit).

        This uses domain knowledge about product relationships.
        """
        # Define product groups
        credit_products = {'credit card', 'prepaid card', 'credit card or prepaid card'}
        loan_products = {'mortgage', 'student loan', 'vehicle loan or lease', 'payday loan',
                        'personal loan', 'debt collection', 'payday loan, title loan, or personal loan'}
        deposit_products = {'checking or savings account', 'bank account or service'}

        product_groups = [credit_products, loan_products, deposit_products]

        for group in product_groups:
            if product1 in group and product2 in group:
                return True

        return False

    def _are_issues_related(self, issue1: str, issue2: str) -> bool:
        """
        Check if two issues are related (for partial credit).

        This uses domain knowledge about issue relationships.
        """
        # Define issue groups
        fee_issues = {'fees', 'late fees', 'overdraft fees', 'unexpected fees', 'other fees'}
        fraud_issues = {'fraud', 'unauthorized transactions', 'identity theft', 'scam'}
        account_issues = {'account opening', 'account closure', 'account management', 'managing an account'}
        payment_issues = {'payment', 'billing', 'transaction', 'incorrect transaction amount'}

        issue_groups = [fee_issues, fraud_issues, account_issues, payment_issues]

        for group in issue_groups:
            if any(keyword in issue1 for keyword in group) and any(keyword in issue2 for keyword in group):
                return True

        return False

    def calculate_hybrid_similarity(self, query: Dict, candidate: Dict) -> float:
        """
        Calculate hybrid similarity combining semantic and categorical features.

        Args:
            query: Query complaint dictionary with 'embedding' and categorical fields
            candidate: Candidate complaint dictionary with 'embedding' and categorical fields

        Returns:
            Hybrid similarity score (0-1)
        """
        total_similarity = 0.0

        # Semantic similarity from embeddings
        if self.semantic_weight > 0:
            query_emb = query.get('embedding')
            candidate_emb = candidate.get('embedding')

            if query_emb is not None and candidate_emb is not None:
                # Ensure embeddings are 2D arrays for cosine_similarity
                if query_emb.ndim == 1:
                    query_emb = query_emb.reshape(1, -1)
                if candidate_emb.ndim == 1:
                    candidate_emb = candidate_emb.reshape(1, -1)

                semantic_sim = cosine_similarity(query_emb, candidate_emb)[0, 0]
                # Normalize cosine similarity from [-1, 1] to [0, 1]
                semantic_sim = (semantic_sim + 1) / 2
                total_similarity += self.semantic_weight * semantic_sim

        # Categorical similarity
        categorical_sim = self.calculate_categorical_similarity(query, candidate)
        total_similarity += categorical_sim

        return total_similarity

    def rank_candidates(self, query: Dict, candidates: List[Dict],
                       top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        Rank candidates by hybrid similarity to query.

        Args:
            query: Query complaint dictionary
            candidates: List of candidate complaint dictionaries
            top_k: If specified, return only top k candidates

        Returns:
            List of (index, similarity_score) tuples, sorted by similarity descending
        """
        similarities = []

        for i, candidate in enumerate(candidates):
            sim = self.calculate_hybrid_similarity(query, candidate)
            similarities.append((i, sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            return similarities[:top_k]

        return similarities

    def select_diverse_relevant_examples(self, query: Dict, candidates: List[Dict],
                                        n_examples: int, diversity_weight: float = 0.3) -> List[int]:
        """
        Select examples that are both relevant (hybrid similarity) and diverse.

        Uses a greedy algorithm that balances relevance to query with diversity among selected examples.

        Args:
            query: Query complaint dictionary
            candidates: List of candidate complaint dictionaries
            n_examples: Number of examples to select
            diversity_weight: Weight for diversity vs relevance (0 = pure relevance, 1 = pure diversity)

        Returns:
            List of selected candidate indices
        """
        if n_examples <= 0 or len(candidates) == 0:
            return []

        if n_examples >= len(candidates):
            return list(range(len(candidates)))

        selected_indices = []
        remaining_indices = list(range(len(candidates)))

        # Calculate relevance scores for all candidates
        relevance_scores = []
        for candidate in candidates:
            relevance = self.calculate_hybrid_similarity(query, candidate)
            relevance_scores.append(relevance)

        # Iteratively select examples
        for _ in range(n_examples):
            best_score = -float('inf')
            best_idx = None

            for idx in remaining_indices:
                # Relevance score (hybrid similarity to query)
                relevance = relevance_scores[idx]

                # Diversity score (minimum similarity to already selected examples)
                if selected_indices:
                    diversity = float('inf')
                    for selected_idx in selected_indices:
                        sim = self.calculate_hybrid_similarity(
                            candidates[idx],
                            candidates[selected_idx]
                        )
                        diversity = min(diversity, 1 - sim)  # Convert similarity to distance
                else:
                    diversity = 1.0  # First selection has maximum diversity

                # Combined score
                score = (1 - diversity_weight) * relevance + diversity_weight * diversity

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

        return selected_indices


class HybridKNNSelector:
    """
    Enhanced k-NN selector using hybrid similarity for n-shot optimization.
    """

    def __init__(self, similarity_calculator: Optional[HybridSimilarityCalculator] = None):
        """
        Initialize hybrid k-NN selector.

        Args:
            similarity_calculator: Custom similarity calculator, or use default if None
        """
        if similarity_calculator is None:
            self.similarity_calculator = HybridSimilarityCalculator()
        else:
            self.similarity_calculator = similarity_calculator

    def select_examples(self, query: Dict, candidates: List[Dict], k: int,
                        use_pure_knn: bool = False) -> List[int]:
        """
        Select k examples using hybrid similarity.

        Args:
            query: Query complaint dictionary
            candidates: List of candidate complaint dictionaries
            k: Number of examples to select
            use_pure_knn: If True, select k most similar; if False, use diverse selection

        Returns:
            List of selected candidate indices
        """
        if use_pure_knn:
            # Pure k-NN: select k most similar
            ranked = self.similarity_calculator.rank_candidates(query, candidates, top_k=k)
            return [idx for idx, _ in ranked]
        else:
            # Diverse selection: balance relevance and diversity
            return self.similarity_calculator.select_diverse_relevant_examples(
                query, candidates, k, diversity_weight=0.3
            )


# Configuration presets for different use cases
class HybridSimilarityPresets:
    """Predefined weight configurations for different scenarios"""

    @staticmethod
    def semantic_only():
        """Pure semantic similarity (current approach)"""
        return HybridSimilarityCalculator(
            semantic_weight=1.0,
            product_weight=0.0,
            issue_weight=0.0
        )

    @staticmethod
    def balanced():
        """Balanced semantic and categorical"""
        return HybridSimilarityCalculator(
            semantic_weight=0.6,
            product_weight=0.15,
            issue_weight=0.15,
            sub_product_weight=0.05,
            sub_issue_weight=0.05
        )

    @staticmethod
    def category_focused():
        """Emphasize categorical matches"""
        return HybridSimilarityCalculator(
            semantic_weight=0.3,
            product_weight=0.3,
            issue_weight=0.3,
            sub_product_weight=0.05,
            sub_issue_weight=0.05
        )

    @staticmethod
    def product_specific():
        """Strong product matching for domain-specific learning"""
        return HybridSimilarityCalculator(
            semantic_weight=0.5,
            product_weight=0.35,
            issue_weight=0.10,
            sub_product_weight=0.05
        )


if __name__ == "__main__":
    # Test the hybrid similarity calculator
    print("Testing Hybrid Similarity Calculator")
    print("=" * 50)

    # Create test data
    query = {
        'embedding': np.random.randn(384),
        'product': 'credit card',
        'issue': 'unexpected fees',
        'sub_product': 'general purpose card',
        'complaint_text': 'I was charged a fee that I did not expect'
    }

    candidates = [
        {
            'embedding': np.random.randn(384),
            'product': 'credit card',  # Same product
            'issue': 'late fees',  # Related issue
            'sub_product': 'general purpose card',
            'complaint_text': 'They charged me a late fee unfairly'
        },
        {
            'embedding': np.random.randn(384),
            'product': 'mortgage',  # Different product category
            'issue': 'loan servicing',  # Unrelated issue
            'sub_product': 'conventional',
            'complaint_text': 'My mortgage payment was processed incorrectly'
        },
        {
            'embedding': np.random.randn(384),
            'product': 'prepaid card',  # Related product
            'issue': 'unexpected fees',  # Same issue
            'sub_product': 'general purpose card',
            'complaint_text': 'Hidden fees on my prepaid card'
        }
    ]

    # Test different configurations
    print("\n1. Balanced Configuration:")
    calc_balanced = HybridSimilarityPresets.balanced()
    for i, candidate in enumerate(candidates):
        sim = calc_balanced.calculate_hybrid_similarity(query, candidate)
        print(f"   Candidate {i+1}: {sim:.3f} (Product: {candidate['product']}, Issue: {candidate['issue']})")

    print("\n2. Category-Focused Configuration:")
    calc_category = HybridSimilarityPresets.category_focused()
    for i, candidate in enumerate(candidates):
        sim = calc_category.calculate_hybrid_similarity(query, candidate)
        print(f"   Candidate {i+1}: {sim:.3f} (Product: {candidate['product']}, Issue: {candidate['issue']})")

    print("\n3. Ranking with Balanced Configuration:")
    ranked = calc_balanced.rank_candidates(query, candidates)
    for idx, sim in ranked:
        print(f"   Rank: Candidate {idx+1} - Similarity: {sim:.3f}")

    print("\n[SUCCESS] Hybrid similarity calculator tested successfully!")