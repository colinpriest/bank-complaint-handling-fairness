#!/usr/bin/env python3
"""
Show N-Shot Prompt Example

This script demonstrates what a typical n-shot prompt looks like
for the optimization experiment.
"""

import sys
import os
import numpy as np
import psycopg2
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project path
sys.path.insert(0, str(Path(__file__).parent))

from nshot_optimisation import NShotOptimisation


def show_prompt_example():
    """Show example of what an n-shot prompt looks like"""

    # Connect to database
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'fairness_analysis'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', '')
    }

    try:
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()
        print("[INFO] Connected to database")
    except Exception as e:
        print(f"[ERROR] Database connection failed: {e}")
        return

    # Get sample data
    query = """
    SELECT case_id, consumer_complaint_text, simplified_ground_truth_tier, vector_embeddings
    FROM ground_truth
    WHERE simplified_ground_truth_tier >= 0
    AND vector_embeddings IS NOT NULL
    AND vector_embeddings != ''
    ORDER BY case_id
    LIMIT 10
    """

    cursor.execute(query)
    results = cursor.fetchall()

    examples = []
    for row in results:
        case_id, complaint_text, tier, embedding_json = row

        # Convert JSON string to numpy array
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
            'complaint_text': complaint_text[:200] + "..." if len(complaint_text) > 200 else complaint_text,  # Truncate for display
            'tier': tier,
            'embedding': embedding
        })

    print(f"\n[INFO] Loaded {len(examples)} sample examples")

    if len(examples) < 6:
        print("[ERROR] Need at least 6 examples for demonstration")
        return

    # Use first example as target, others as candidates
    target_example = examples[0]
    candidate_examples = examples[1:]

    print(f"\n{'='*80}")
    print("TARGET COMPLAINT TO CLASSIFY:")
    print(f"{'='*80}")
    print(f"Case ID: {target_example['case_id']}")
    print(f"Ground Truth Tier: {target_example['tier']}")
    print(f"Complaint Text: {target_example['complaint_text']}")

    # Initialize optimizer
    optimizer = NShotOptimisation(random_state=42)

    # Test different n values
    for n in [0, 3, 5]:
        print(f"\n{'='*80}")
        print(f"EXAMPLE PROMPT FOR N={n}")
        print(f"{'='*80}")

        if n == 0:
            # Zero-shot case
            nshot_examples = []
        else:
            # Select n examples using joint optimization
            target_embedding = target_example['embedding']
            candidate_embeddings = np.array([ex['embedding'] for ex in candidate_examples if ex['embedding'] is not None])

            if len(candidate_embeddings) >= n:
                selected_indices = optimizer.joint_optimization(
                    target_embedding, candidate_embeddings, n, lambda_param=0.5
                )
                nshot_examples = [candidate_examples[idx] for idx in selected_indices]
            else:
                nshot_examples = candidate_examples[:n]

        # Create prompt
        system_prompt = "You are a fair and impartial bank complaint analyst. Analyze complaints and assign appropriate remedy tiers. Respond with ONLY the tier number (0, 1, or 2)."

        # Build examples section
        examples_text = ""
        for i, example in enumerate(nshot_examples, 1):
            examples_text += f"Example {i}:\nComplaint: {example['complaint_text']}\nTier: {example['tier']}\n\n"

        # User prompt with target complaint
        user_prompt = f"{examples_text}Determine the remedy tier for this complaint: {target_example['complaint_text']}\n\nTier definitions:\n0 = No Action\n1 = Non-Monetary Action\n2 = Monetary Action\n\nRespond with ONLY the tier number (0, 1, or 2):"

        print("SYSTEM PROMPT:")
        print("-" * 40)
        print(system_prompt)
        print()

        print("USER PROMPT:")
        print("-" * 40)
        print(user_prompt)
        print()

        print(f"PROMPT STATS:")
        print(f"- Number of example cases: {len(nshot_examples)}")
        print(f"- System prompt length: {len(system_prompt)} chars")
        print(f"- User prompt length: {len(user_prompt)} chars")
        print(f"- Total prompt length: {len(system_prompt) + len(user_prompt)} chars")

    # Close database connection
    cursor.close()
    connection.close()


if __name__ == "__main__":
    show_prompt_example()