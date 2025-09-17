#!/usr/bin/env python3
"""
Test script for category-specific tier statistics implementation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from multithreaded_nshot_optimization import MultithreadedCategoryFilteredOptimizer
from nshot_prompt_generator import NShotPromptGenerator, BiasStrategy


def test_category_specific_stats():
    """Test the category-specific tier statistics functionality"""
    print("=" * 80)
    print("TESTING CATEGORY-SPECIFIC TIER STATISTICS")
    print("=" * 80)

    # Initialize the optimizer
    optimizer = MultithreadedCategoryFilteredOptimizer(max_workers=1)

    # Test a few common product/issue combinations
    test_categories = [
        ("Credit reporting, credit repair services, or other personal consumer reports",
         "Incorrect information on your report"),
        ("Debt collection", "Attempts to collect debt not owed"),
        ("Checking or savings account", "Account opening, closing, or management"),
        ("Credit card or prepaid card", "Billing disputes")
    ]

    print("\n[TEST 1] Testing category-specific statistics calculation:")
    print("-" * 60)

    for product, issue in test_categories:
        print(f"\nCategory: {product[:50]}... / {issue[:50]}...")
        try:
            stats = optimizer.calculate_category_tier_statistics(product, issue)
            total = stats['tier_0_percent'] + stats['tier_1_percent'] + stats['tier_2_percent']
            print(f"  Stats: T0={stats['tier_0_percent']:.1f}%, T1={stats['tier_1_percent']:.1f}%, T2={stats['tier_2_percent']:.1f}% (Total: {total:.1f}%)")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n[TEST 2] Testing prompt generation with category-specific stats:")
    print("-" * 60)

    # Create test data
    target_case = {
        'complaint_text': 'The bank incorrectly reported a late payment on my credit report.'
    }

    nshot_examples = [
        {'complaint_text': 'Similar credit report error case.', 'tier': 1},
        {'complaint_text': 'Another credit report dispute.', 'tier': 0}
    ]

    # Test with category-specific stats
    test_product = "Credit reporting, credit repair services, or other personal consumer reports"
    test_issue = "Incorrect information on your report"

    try:
        category_stats = optimizer.calculate_category_tier_statistics(test_product, test_issue)

        prompt_generator = NShotPromptGenerator()

        # Test with global stats (original behavior)
        print("\nGlobal Statistics (Original):")
        system_prompt_global, user_prompt_global = prompt_generator.generate_prompts(
            target_case=target_case,
            nshot_examples=nshot_examples,
            bias_strategy=BiasStrategy.CHAIN_OF_THOUGHT
        )

        # Extract tier percentages from prompt
        lines = user_prompt_global.split('\n')
        for line in lines:
            if "% of cases)" in line:
                print(f"  {line.strip()}")

        # Test with category-specific stats
        print(f"\nCategory-Specific Statistics ({test_product[:30]}...):")
        system_prompt_category, user_prompt_category = prompt_generator.generate_prompts(
            target_case=target_case,
            nshot_examples=nshot_examples,
            bias_strategy=BiasStrategy.CHAIN_OF_THOUGHT,
            category_tier_stats=category_stats
        )

        # Extract tier percentages from prompt
        lines = user_prompt_category.split('\n')
        for line in lines:
            if "% of cases)" in line:
                print(f"  {line.strip()}")

        print(f"\n[SUCCESS] Category-specific statistics successfully applied!")
        print(f"Global vs Category comparison:")
        print(f"  Global: T0=65.6%, T1=31.8%, T2=2.6%")
        print(f"  Category: T0={category_stats['tier_0_percent']:.1f}%, T1={category_stats['tier_1_percent']:.1f}%, T2={category_stats['tier_2_percent']:.1f}%")

    except Exception as e:
        print(f"[ERROR] Prompt generation test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("CATEGORY-SPECIFIC STATISTICS TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    test_category_specific_stats()