#!/usr/bin/env python3
"""
Test that all three optimization algorithms use category-specific tier statistics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from multithreaded_nshot_optimization import (
    MultithreadedSingleAlphaOptimizer,
    MultithreadedDPPKNNOptimizer,
    MultithreadedCategoryFilteredOptimizer,
    get_product_category_group
)
from datetime import datetime


def test_category_grouping():
    """Test the product category grouping function"""
    print("=" * 80)
    print("TESTING CATEGORY GROUPING")
    print("=" * 80)

    test_products = [
        "Credit reporting, credit repair services, or other personal consumer reports",
        "Debt collection",
        "Checking or savings account",
        "Mortgage",
        "Credit card or prepaid card",
        "Student loan",
        "Money transfer, virtual currency, or money service",
        "Vehicle loan or lease",
        "Other financial service"
    ]

    for product in test_products:
        group = get_product_category_group(product)
        print(f"{product[:50]:50} -> {group}")

    print()


def test_algorithm_category_stats(optimizer, algorithm_name, test_size=3):
    """Test that an algorithm uses category-specific statistics"""
    print(f"\nTESTING {algorithm_name}")
    print("-" * 60)

    # Load a small test set
    test_examples = optimizer.load_ground_truth_examples(limit=test_size)
    if not test_examples:
        print(f"[ERROR] No test examples loaded")
        return

    all_examples = optimizer.load_ground_truth_examples(limit=50)

    # Remove test examples from candidates
    test_case_ids = {ex['case_id'] for ex in test_examples}
    all_examples = [ex for ex in all_examples if ex['case_id'] not in test_case_ids]

    print(f"[INFO] Loaded {len(test_examples)} test, {len(all_examples)} candidate examples")

    # Test category statistics calculation for each test example
    print(f"\n[INFO] Category statistics for test examples:")
    for i, example in enumerate(test_examples):
        product = example.get('product', '')
        group = get_product_category_group(product)

        # Calculate category stats
        stats = optimizer.calculate_grouped_category_tier_statistics(product)

        print(f"\nExample {i+1}:")
        print(f"  Product: {product[:60]}")
        print(f"  Group: {group}")
        print(f"  Stats: T0={stats['tier_0_percent']:.1f}%, T1={stats['tier_1_percent']:.1f}%, T2={stats['tier_2_percent']:.1f}%")

    # Run a minimal test to ensure prompt generation works
    if algorithm_name == "ALPHA OPTIMIZATION":
        params = (1, 0.5, test_examples[:1], all_examples, datetime.now())
        try:
            result = optimizer.test_single_parameter_combination(params)
            if result['total'] > 0:
                print(f"\n[SUCCESS] {algorithm_name} executed with category-specific stats")
            else:
                print(f"\n[WARNING] {algorithm_name} made no predictions")
        except Exception as e:
            print(f"\n[ERROR] {algorithm_name} test failed: {e}")

    elif algorithm_name == "DPP+K-NN OPTIMIZATION":
        # Initialize global DPP selector
        optimizer._initialize_global_dpp_selector(all_examples)

        params = (1, 1, test_examples[:1], len(all_examples), datetime.now())
        precomputed_data = optimizer.precompute_dpp_knn_resources(
            test_examples[:1], all_examples, 1, 1, 2
        )
        try:
            result = optimizer.test_dpp_knn_combination_with_precomputed(params, precomputed_data)
            if result['total'] > 0:
                print(f"\n[SUCCESS] {algorithm_name} executed with category-specific stats")
            else:
                print(f"\n[WARNING] {algorithm_name} made no predictions")
        except Exception as e:
            print(f"\n[ERROR] {algorithm_name} test failed: {e}")

    elif algorithm_name == "CATEGORY-FILTERED OPTIMIZATION":
        params = (1, 1, test_examples[:1], all_examples, datetime.now())
        try:
            result = optimizer.test_category_filtered_combination(params)
            if result['total'] > 0:
                print(f"\n[SUCCESS] {algorithm_name} executed with category-specific stats")
            else:
                print(f"\n[WARNING] {algorithm_name} made no predictions")
        except Exception as e:
            print(f"\n[ERROR] {algorithm_name} test failed: {e}")


def main():
    """Test all three algorithms with category-specific statistics"""
    print("=" * 80)
    print("TESTING ALL ALGORITHMS WITH CATEGORY-SPECIFIC STATISTICS")
    print("=" * 80)

    # Test category grouping
    test_category_grouping()

    # Test Alpha Optimization
    alpha_optimizer = MultithreadedSingleAlphaOptimizer(max_workers=1)
    test_algorithm_category_stats(alpha_optimizer, "ALPHA OPTIMIZATION")

    # Test DPP+k-NN Optimization
    dpp_knn_optimizer = MultithreadedDPPKNNOptimizer(max_workers=1)
    test_algorithm_category_stats(dpp_knn_optimizer, "DPP+K-NN OPTIMIZATION")

    # Test Category-Filtered Optimization
    category_optimizer = MultithreadedCategoryFilteredOptimizer(max_workers=1)
    test_algorithm_category_stats(category_optimizer, "CATEGORY-FILTERED OPTIMIZATION")

    print("\n" + "=" * 80)
    print("ALL ALGORITHMS TESTED WITH CATEGORY-SPECIFIC STATISTICS")
    print("=" * 80)


if __name__ == "__main__":
    main()