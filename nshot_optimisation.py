#!/usr/bin/env python3
"""
N-Shot Optimization Module

This module contains the NShotOptimisation class which handles optimization
of n-shot learning parameters including alpha tuning and example selection
for the bank complaint handling fairness analysis system.
"""

import numpy as np
from typing import Callable, Tuple, List, Optional
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity


class NShotOptimisation:
    """
    N-shot optimization class for tuning parameters and selecting examples
    for optimal few-shot learning performance.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize NShotOptimisation.

        Args:
            random_state: Random seed for reproducible results
        """
        self.random_state = random_state
        np.random.seed(random_state)

    def joint_optimization(self, query_emb: np.ndarray, candidate_embs: np.ndarray,
                          n: int, lambda_param: float = 0.5) -> List[int]:
        """
        Selects n examples with joint relevance-diversity greedy optimization.

        Args:
            query_emb: numpy array with shape (embedding_dim,)
            candidate_embs: numpy array with shape (num_candidates, embedding_dim)
            n: number of examples to select
            lambda_param: trade-off between relevance (to query) and diversity (among group)

        Returns:
            list of selected example indices
        """
        if len(candidate_embs) == 0:
            return []

        num_candidates = candidate_embs.shape[0]

        if n >= num_candidates:
            return list(range(num_candidates))

        selected = []
        candidate_indices = list(range(num_candidates))

        # Relevance: similarity to query
        query_sim = cosine_similarity(candidate_embs, query_emb.reshape(1, -1)).flatten()

        # Diversity: similarity among candidates
        diversity_matrix = cosine_similarity(candidate_embs)

        for _ in range(n):
            best_score = -np.inf
            best_idx = None

            for i in candidate_indices:
                # If first selection, diversity is maximal by definition
                if not selected:
                    diversity_score = 1.0
                else:
                    # 1 - mean similarity with already selected examples = diversity
                    sim_to_selected = diversity_matrix[i, selected]
                    diversity_score = 1.0 - np.mean(sim_to_selected)
                total_score = lambda_param * query_sim[i] + (1.0 - lambda_param) * diversity_score
                if total_score > best_score:
                    best_score = total_score
                    best_idx = i
            selected.append(best_idx)
            candidate_indices.remove(best_idx)

        return selected

    def tune_alpha(self, query_emb: np.ndarray, candidate_embs: np.ndarray,
                   eval_fn: Callable[[List[int]], float]) -> Tuple[int, float, float]:
        """
        Tune both n and alpha parameters with grid search

        Args:
            query_emb: numpy array of query embedding
            candidate_embs: numpy array of candidate embeddings
            eval_fn: function to evaluate selected indices, should return a score

        Returns:
            Tuple of (best_n, best_alpha, best_score)
        """
        alpha_grid = np.arange(0.3, 0.71, 0.05)
        n_grid = [0, 5, 6, 7, 8, 9, 10]

        # Limit n_grid to available candidates
        max_candidates = len(candidate_embs)
        n_grid = [n for n in n_grid if n <= max_candidates]

        best_score = -np.inf
        best_alpha = None
        best_n = None

        print(f"[INFO] Tuning n and alpha parameters")
        print(f"[INFO] n grid: {n_grid}")
        print(f"[INFO] alpha grid: {alpha_grid}")

        for n in n_grid:
            print(f"\n[INFO] Testing n = {n}")

            if n == 0:
                # Special case: n=0 means no examples selected
                try:
                    score = eval_fn([])
                    print(f"[INFO] n=0: Score {score:.4f} (no examples selected)")

                    if score > best_score:
                        best_score = score
                        best_n = n
                        best_alpha = 0.5  # Default alpha for n=0 case

                except Exception as e:
                    print(f"[WARNING] Error with n=0: {str(e)}")
                continue

            # Test all alpha values for current n
            for alpha in alpha_grid:
                try:
                    selected_indices = self.joint_optimization(query_emb, candidate_embs, n, lambda_param=alpha)
                    score = eval_fn(selected_indices)

                    print(f"[INFO] n={n}, alpha={alpha:.2f}: Score {score:.4f} (selected {len(selected_indices)} examples)")

                    if score > best_score:
                        best_score = score
                        best_alpha = alpha
                        best_n = n

                except Exception as e:
                    print(f"[WARNING] Error with n={n}, alpha={alpha:.2f}: {str(e)}")
                    continue

        if best_n is None:
            print("[WARNING] No valid parameters found, using defaults")
            best_n = min(5, max_candidates) if max_candidates > 0 else 0
            best_alpha = 0.5
            best_score = -1.0

        print(f"\n[SUCCESS] Best parameters: n={best_n}, alpha={best_alpha:.2f}, score={best_score:.4f}")
        return best_n, best_alpha, best_score

    def tune_n_and_alpha(self, query_emb: np.ndarray, candidate_embs: np.ndarray,
                        n_range: Tuple[int, int], eval_fn: Callable[[List[int]], float]) -> Tuple[int, float, float]:
        """
        Tune both n (number of examples) and alpha parameter.

        Args:
            query_emb: numpy array of query embedding
            candidate_embs: numpy array of candidate embeddings
            n_range: tuple of (min_n, max_n) for number of examples to try
            eval_fn: function to evaluate selected indices, should return a score

        Returns:
            Tuple of (best_n, best_alpha, best_score)
        """
        min_n, max_n = n_range
        max_n = min(max_n, len(candidate_embs))  # Can't select more than available

        if min_n > max_n:
            print(f"[WARNING] min_n ({min_n}) > max_n ({max_n}), adjusting min_n")
            min_n = max_n

        best_score = -np.inf
        best_n = None
        best_alpha = None

        print(f"[INFO] Tuning n from {min_n} to {max_n} and alpha from 0.3 to 0.7")

        for n in range(min_n, max_n + 1):
            print(f"\n[INFO] Testing n = {n}")

            try:
                alpha, score = self.tune_alpha(query_emb, candidate_embs, n, eval_fn)

                print(f"[INFO] n = {n}: Best alpha = {alpha:.2f}, Score = {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_n = n
                    best_alpha = alpha

            except Exception as e:
                print(f"[WARNING] Error with n = {n}: {str(e)}")
                continue

        if best_n is None:
            print("[WARNING] No valid n found, using defaults")
            best_n = min_n
            best_alpha = 0.5
            best_score = -1.0

        print(f"\n[SUCCESS] Optimal parameters: n = {best_n}, alpha = {best_alpha:.2f}, score = {best_score:.4f}")
        return best_n, best_alpha, best_score

    def select_examples(self, query_emb: np.ndarray, candidate_embs: np.ndarray,
                       n: int, alpha: float = 0.5) -> List[int]:
        """
        Select n examples using the specified alpha parameter.

        Args:
            query_emb: numpy array of query embedding
            candidate_embs: numpy array of candidate embeddings
            n: number of examples to select
            alpha: balance parameter between similarity and diversity

        Returns:
            List of selected indices
        """
        return self.joint_optimization(query_emb, candidate_embs, n, lambda_param=alpha)

    def evaluate_selection_quality(self, selected_indices: List[int],
                                  query_emb: np.ndarray, candidate_embs: np.ndarray) -> dict:
        """
        Evaluate the quality of a selection by computing various metrics.

        Args:
            selected_indices: List of selected example indices
            query_emb: Query embedding
            candidate_embs: All candidate embeddings

        Returns:
            Dictionary with quality metrics
        """
        if len(selected_indices) == 0:
            return {
                'avg_similarity_to_query': 0.0,
                'min_similarity_to_query': 0.0,
                'max_similarity_to_query': 0.0,
                'avg_diversity': 0.0,
                'min_diversity': 0.0,
                'coverage_score': 0.0
            }

        # Ensure query_emb is 2D
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)

        selected_embs = candidate_embs[selected_indices]

        # Similarity to query
        query_similarities = cosine_similarity(query_emb, selected_embs).flatten()

        # Diversity among selected examples
        if len(selected_indices) > 1:
            pairwise_similarities = cosine_similarity(selected_embs, selected_embs)
            # Get upper triangular matrix (excluding diagonal)
            upper_tri = np.triu(pairwise_similarities, k=1)
            diversities = 1.0 - upper_tri[upper_tri > 0]
            avg_diversity = np.mean(diversities)
            min_diversity = np.min(diversities)
        else:
            avg_diversity = 1.0
            min_diversity = 1.0

        # Coverage score (how well the selection covers the candidate space)
        if len(candidate_embs) > len(selected_indices):
            # For each non-selected candidate, find max similarity to selected examples
            non_selected_indices = [i for i in range(len(candidate_embs)) if i not in selected_indices]
            non_selected_embs = candidate_embs[non_selected_indices]

            coverage_similarities = cosine_similarity(non_selected_embs, selected_embs)
            max_coverage_per_candidate = np.max(coverage_similarities, axis=1)
            coverage_score = np.mean(max_coverage_per_candidate)
        else:
            coverage_score = 1.0

        return {
            'avg_similarity_to_query': float(np.mean(query_similarities)),
            'min_similarity_to_query': float(np.min(query_similarities)),
            'max_similarity_to_query': float(np.max(query_similarities)),
            'avg_diversity': float(avg_diversity),
            'min_diversity': float(min_diversity),
            'coverage_score': float(coverage_score),
            'num_selected': len(selected_indices)
        }