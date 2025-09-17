#!/usr/bin/env python3
"""
Test the category-filtered optimization with category-specific statistics integration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from multithreaded_nshot_optimization import MultithreadedCategoryFilteredOptimizer
from datetime import datetime


def test_category_filtered_integration():
    """Test a small category-filtered optimization run to verify integration works"""
    print("=" * 80)
    print("TESTING CATEGORY-FILTERED OPTIMIZATION INTEGRATION")
    print("=" * 80)

    # Initialize optimizer
    optimizer = MultithreadedCategoryFilteredOptimizer(max_workers=1)

    # Create table if needed
    print("[STEP 1] Verifying database table...")
    if not optimizer.create_category_filtered_results_table():
        print("[ERROR] Failed to create/verify database table")
        return

    # Load a small sample for testing
    print("[STEP 2] Loading test examples...")
    try:
        test_examples = optimizer.load_ground_truth_examples(limit=5)  # Very small sample
        if not test_examples:
            print("[ERROR] No test examples loaded")
            return
        print(f"[INFO] Loaded {len(test_examples)} test examples")

        all_examples = optimizer.load_ground_truth_examples(limit=100)  # Small candidate pool
        if not all_examples:
            print("[ERROR] No candidate examples loaded")
            return

        # Remove test examples from candidates
        test_case_ids = {ex['case_id'] for ex in test_examples}
        all_examples = [ex for ex in all_examples if ex['case_id'] not in test_case_ids]
        print(f"[INFO] Loaded {len(all_examples)} candidate examples")

        # Test a single parameter combination
        print("[STEP 3] Testing single parameter combination (n=1, k=1)...")

        experiment_timestamp = datetime.now()
        params = (1, 1, test_examples, all_examples, experiment_timestamp)

        result = optimizer.test_category_filtered_combination(params)

        print(f"[RESULT] Test completed:")
        print(f"  n={result['n']}, k={result['k']}")
        print(f"  Correct: {result['correct']}/{result['total']}")
        print(f"  Accuracy: {result['accuracy']:.3f}")
        print(f"  Execution time: {result['execution_time']:.1f}s")

        if result['total'] > 0:
            print(f"[SUCCESS] Category-filtered optimization with category-specific statistics completed successfully!")
        else:
            print(f"[WARNING] No predictions made - this might be due to insufficient category examples")

    except Exception as e:
        print(f"[ERROR] Integration test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("CATEGORY-FILTERED INTEGRATION TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    test_category_filtered_integration()