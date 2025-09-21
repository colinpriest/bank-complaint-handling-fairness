#!/usr/bin/env python3
"""
Test the tier-stratified DPP integration in the bank complaint handling system.
"""

import os
import sys
import psycopg2
from dotenv import load_dotenv
import numpy as np
import json

# Add the current directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tier_stratified_dpp import TierStratifiedDPP

load_dotenv()

def test_integration():
    """Test the tier-stratified DPP integration"""
    print("=== Testing Tier-Stratified DPP Integration ===\n")

    # Database configuration
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'fairness_analysis'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', '')
    }

    # Initialize selector
    print("1. Initializing Tier-Stratified DPP...")
    selector = TierStratifiedDPP(db_config, debug=False)

    # Show cache stats
    stats = selector.get_cache_stats()
    print(f"   - Cached central examples for {stats['total_products']} products")
    print(f"   - Products with full tier coverage: {stats['products_with_full_coverage']}")

    # Test with real data from database
    print("\n2. Testing with real database examples...")

    connection = psycopg2.connect(**db_config)
    cursor = connection.cursor()

    try:
        # Get a sample case for testing
        cursor.execute("""
            SELECT case_id, consumer_complaint_text, product, vector_embeddings
            FROM ground_truth
            WHERE vector_embeddings IS NOT NULL
            AND consumer_complaint_text IS NOT NULL
            AND product = 'Credit card or prepaid card'
            LIMIT 1
        """)

        result = cursor.fetchone()
        if not result:
            print("   No suitable test case found!")
            return

        test_case_id, test_complaint, test_product, test_embedding = result
        test_embedding = np.array(json.loads(test_embedding))

        print(f"   - Test case: {test_case_id}")
        print(f"   - Product: {test_product}")
        print(f"   - Complaint length: {len(test_complaint)} characters")

        # Get candidate examples (excluding the test case)
        cursor.execute("""
            SELECT case_id, consumer_complaint_text, simplified_ground_truth_tier, vector_embeddings
            FROM ground_truth
            WHERE case_id != %s
            AND simplified_ground_truth_tier >= 0
            AND vector_embeddings IS NOT NULL
            AND consumer_complaint_text IS NOT NULL
            LIMIT 100
        """, (test_case_id,))

        candidates = []
        tier_counts = {0: 0, 1: 0, 2: 0}

        for case_id, complaint, tier, embedding in cursor.fetchall():
            candidates.append({
                'case_id': case_id,
                'complaint_text': complaint,
                'tier': tier,
                'embedding': np.array(json.loads(embedding))
            })
            if tier in tier_counts:
                tier_counts[tier] += 1

        print(f"   - Candidate pool: {len(candidates)} examples")
        print(f"   - Tier distribution: T0={tier_counts[0]}, T1={tier_counts[1]}, T2={tier_counts[2]}")

        # Test tier-stratified selection
        print("\n3. Testing tier-stratified selection...")

        selected_examples = selector.select_stratified_examples(
            candidates=candidates,
            query_embedding=test_embedding,
            target_product=test_product,
            n_dpp=1,
            k_nn=2
        )

        print(f"   - Selected {len(selected_examples)} examples:")
        selected_tiers = []
        for i, example in enumerate(selected_examples):
            tier = example.get('tier', -1)
            case_id = example.get('case_id', 'unknown')
            is_central = example.get('is_central_fallback', False)
            selected_tiers.append(tier)

            print(f"     {i+1}. Case {case_id}, Tier {tier}, Central: {is_central}")

        # Check tier representation
        unique_tiers = set(selected_tiers)
        print(f"   - Unique tiers represented: {sorted(unique_tiers)}")

        if len(unique_tiers) >= 2:
            print("   [SUCCESS] Multiple tiers represented!")
        else:
            print("   [WARNING] Limited tier diversity")

        # Test comparison with unfiltered selection
        print("\n4. Comparing with unfiltered candidates...")

        # Create a biased candidate pool (mostly Tier 0)
        biased_candidates = [c for c in candidates if c['tier'] == 0][:8]  # Only Tier 0 examples
        biased_candidates.extend([c for c in candidates if c['tier'] == 1][:2])  # Few Tier 1

        print(f"   - Biased pool: {len(biased_candidates)} examples")
        biased_tier_counts = {}
        for c in biased_candidates:
            tier = c['tier']
            biased_tier_counts[tier] = biased_tier_counts.get(tier, 0) + 1
        print(f"   - Biased distribution: {biased_tier_counts}")

        selected_biased = selector.select_stratified_examples(
            candidates=biased_candidates,
            query_embedding=test_embedding,
            target_product=test_product,
            n_dpp=1,
            k_nn=2
        )

        print(f"   - Selected from biased pool:")
        biased_selected_tiers = []
        for i, example in enumerate(selected_biased):
            tier = example.get('tier', -1)
            case_id = example.get('case_id', 'unknown')
            is_central = example.get('is_central_fallback', False)
            biased_selected_tiers.append(tier)

            print(f"     {i+1}. Case {case_id}, Tier {tier}, Central: {is_central}")

        biased_unique_tiers = set(biased_selected_tiers)
        print(f"   - Tiers in biased selection: {sorted(biased_unique_tiers)}")

        if 2 in biased_unique_tiers:
            print("   [SUCCESS] Tier 2 added via central example fallback!")

        print("\n=== Integration Test Complete ===")

    finally:
        connection.close()

if __name__ == "__main__":
    test_integration()