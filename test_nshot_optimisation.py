#!/usr/bin/env python3
"""
Test the NShotOptimisation class
"""

import numpy as np
from nshot_optimisation import NShotOptimisation

def test_nshot_optimisation():
    """Test basic NShotOptimisation functionality"""

    print("="*80)
    print("TESTING NSHOT_OPTIMISATION CLASS")
    print("="*80)

    # Initialize NShotOptimisation
    optimizer = NShotOptimisation(random_state=42)

    # Create synthetic test data
    print("\n[INFO] Creating synthetic test data...")
    np.random.seed(42)

    # Create a query embedding
    query_emb = np.random.randn(384)  # Match sentence transformer dimension

    # Create candidate embeddings (some similar to query, some diverse)
    num_candidates = 50
    candidate_embs = np.random.randn(num_candidates, 384)

    # Make some candidates more similar to query
    for i in range(5):
        candidate_embs[i] = query_emb + 0.1 * np.random.randn(384)

    print(f"[INFO] Created query embedding: shape {query_emb.shape}")
    print(f"[INFO] Created candidate embeddings: shape {candidate_embs.shape}")

    # Test joint optimization
    print("\n[INFO] Testing joint_optimization...")
    n = 5
    alpha = 0.5

    selected_indices = optimizer.joint_optimization(query_emb, candidate_embs, n, alpha)
    print(f"[INFO] Selected {len(selected_indices)} indices: {selected_indices}")

    # Test different alpha values
    print("\n[INFO] Testing different alpha values...")
    for test_alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        indices = optimizer.joint_optimization(query_emb, candidate_embs, n, test_alpha)
        print(f"  Alpha {test_alpha}: selected {indices}")

    # Test evaluation metrics
    print("\n[INFO] Testing evaluation metrics...")
    quality_metrics = optimizer.evaluate_selection_quality(selected_indices, query_emb, candidate_embs)
    print(f"[INFO] Quality metrics:")
    for metric, value in quality_metrics.items():
        print(f"  - {metric}: {value:.4f}")

    # Test with simple evaluation function
    print("\n[INFO] Testing tune_alpha...")

    def simple_eval_fn(indices):
        """Simple evaluation function - higher score for more diverse selections"""
        if len(indices) <= 1:
            return 0.0

        selected_embs = candidate_embs[indices]
        from sklearn.metrics.pairwise import cosine_similarity

        # Calculate average pairwise diversity
        pairwise_sim = cosine_similarity(selected_embs, selected_embs)
        upper_tri = np.triu(pairwise_sim, k=1)
        avg_diversity = 1.0 - np.mean(upper_tri[upper_tri > 0])

        return avg_diversity

    try:
        best_n, best_alpha, best_score = optimizer.tune_alpha(query_emb, candidate_embs, simple_eval_fn)
        print(f"[SUCCESS] Best n: {best_n}, Best alpha: {best_alpha:.2f}, Best score: {best_score:.4f}")
    except Exception as e:
        print(f"[ERROR] tune_alpha failed: {str(e)}")

    # Test edge cases
    print("\n[INFO] Testing edge cases...")

    # Empty candidates
    try:
        empty_result = optimizer.joint_optimization(query_emb, np.array([]).reshape(0, 384), 5, 0.5)
        print(f"[INFO] Empty candidates result: {empty_result}")
    except Exception as e:
        print(f"[WARNING] Empty candidates error: {str(e)}")

    # More n than candidates
    try:
        large_n_result = optimizer.joint_optimization(query_emb, candidate_embs[:3], 10, 0.5)
        print(f"[INFO] Large n result: {large_n_result}")
    except Exception as e:
        print(f"[WARNING] Large n error: {str(e)}")

    print("\n[SUCCESS] NShotOptimisation class testing completed!")

if __name__ == "__main__":
    test_nshot_optimisation()