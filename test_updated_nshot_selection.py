#!/usr/bin/env python3
"""
Test the updated n-shot selection with tier stratification in a realistic scenario.
"""

import os
import sys
import psycopg2
from dotenv import load_dotenv
import numpy as np
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib.util
spec = importlib.util.spec_from_file_location("bank_complaint_handling", "bank-complaint-handling.py")
bank_complaint_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bank_complaint_module)
BankComplaintFairnessAnalyzer = bank_complaint_module.BankComplaintFairnessAnalyzer

load_dotenv()

def test_updated_nshot():
    """Test n-shot selection with the new tier-stratified algorithm"""
    print("=== Testing Updated N-Shot Selection ===\n")

    try:
        # Initialize the analyzer (this will load the tier-stratified selector)
        print("1. Initializing Bank Complaint Analyzer...")
        analyzer = BankComplaintFairnessAnalyzer()

        print("   - Analyzer initialized successfully")
        print("   - Tier-stratified selector loaded")

        # Test a sample case
        print("\n2. Testing n-shot example selection...")

        # Get a random case from the database to test
        connection = analyzer.get_thread_db_connection()[0]
        cursor = connection.cursor()

        cursor.execute("""
            SELECT case_id, consumer_complaint_text, product
            FROM ground_truth
            WHERE vector_embeddings IS NOT NULL
            AND consumer_complaint_text IS NOT NULL
            AND product = 'Credit card or prepaid card'
            LIMIT 1
        """)

        result = cursor.fetchone()
        if not result:
            print("   No test case found!")
            return

        case_id, complaint_text, product = result
        print(f"   - Test case ID: {case_id}")
        print(f"   - Product: {product}")
        print(f"   - Complaint length: {len(complaint_text)} chars")

        # Get examples with embeddings for this test
        examples_with_embeddings = analyzer.get_ground_truth_examples_with_embeddings()
        print(f"   - Available examples: {len(examples_with_embeddings)}")

        # Create target case
        target_example = None
        for ex in examples_with_embeddings:
            if ex['case_id'] == case_id:
                target_example = ex
                break

        if not target_example:
            print("   Target case not found in examples!")
            return

        # Test the selection process
        print("\n3. Running tier-stratified selection...")

        # Extract embedding and create candidate pool
        target_embedding = target_example['embedding']
        candidates = [ex for ex in examples_with_embeddings if ex['case_id'] != case_id]

        print(f"   - Target embedding shape: {target_embedding.shape}")
        print(f"   - Candidate pool size: {len(candidates)}")

        # Check tier distribution in candidates
        tier_counts = {0: 0, 1: 0, 2: 0, -1: 0}
        for candidate in candidates[:100]:  # Sample first 100
            tier = candidate.get('tier', -1)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        print(f"   - Sample tier distribution: T0={tier_counts[0]}, T1={tier_counts[1]}, T2={tier_counts[2]}")

        # Test the selection method directly
        selected_examples = analyzer.combine_dpp_knn_examples(
            target_embedding,
            candidates[:100],  # Use first 100 candidates
            analyzer.n_dpp,
            analyzer.k_nn,
            product
        )

        print(f"\n4. Selection Results:")
        print(f"   - Selected {len(selected_examples)} examples")

        selected_tiers = []
        for i, example in enumerate(selected_examples):
            tier = example.get('tier', -1)
            case_id = example.get('case_id', 'unknown')
            is_central = example.get('is_central_fallback', False)
            selected_tiers.append(tier)

            print(f"     {i+1}. Case {case_id}, Tier {tier}, Central: {is_central}")

        unique_tiers = set(selected_tiers)
        print(f"   - Unique tiers: {sorted(unique_tiers)}")

        # Check if all tiers are represented
        expected_tiers = {0, 1, 2}
        if expected_tiers.issubset(unique_tiers):
            print("   [SUCCESS] All tiers represented!")
        else:
            missing = expected_tiers - unique_tiers
            print(f"   [WARNING] Missing tiers: {missing}")

        print("\n=== Test Complete ===")

    except Exception as e:
        print(f"   [ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_updated_nshot()