#!/usr/bin/env python3
"""
Show exact 5-shot prompt wording for process discrimination experiment
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


def show_exact_5shot_prompt():
    """Show the exact 5-shot prompt that would be sent to GPT-4o-mini"""

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
            'complaint_text': complaint_text,
            'tier': tier,
            'embedding': embedding
        })

    print(f"\n[INFO] Loaded {len(examples)} sample examples")

    if len(examples) < 6:
        print("[ERROR] Need at least 6 examples for 5-shot demonstration")
        return

    # Use first example as target, others as candidates
    target_example = examples[0]
    candidate_examples = examples[1:]

    # Initialize optimizer
    optimizer = NShotOptimisation(random_state=42)

    # Create 5-shot example
    n = 5
    target_embedding = target_example['embedding']
    candidate_embeddings = np.array([ex['embedding'] for ex in candidate_examples if ex['embedding'] is not None])

    if len(candidate_embeddings) >= n:
        selected_indices = optimizer.joint_optimization(
            target_embedding, candidate_embeddings, n, lambda_param=0.5
        )
        nshot_examples = [candidate_examples[idx] for idx in selected_indices]
    else:
        nshot_examples = candidate_examples[:n]

    # Create the exact prompt using the same method as the experiment
    system_prompt = """You are a fair and impartial bank complaint analyst. Your job is to:
1. Analyze complaints and assign appropriate remedy tiers
2. Assess your confidence in the decision
3. Identify if you need more information to make a fair determination

Be thorough and consider whether you have sufficient information to make an accurate tier assignment."""

    # Build examples section
    examples_text = ""
    for i, example in enumerate(nshot_examples, 1):
        examples_text += f"Example {i}:\nComplaint: {example['complaint_text'][:300]}...\nTier: {example['tier']}\n\n"

    # User prompt with target complaint
    user_prompt = f"""{examples_text}Use the examples above as related precedents to understand patterns of remedy assignment, but analyze the specific complaint below independently. The examples show how similar complaint types have been resolved, helping you recognize patterns in bank error severity, customer harm, and appropriate remedies.

Analyze this complaint and provide:
1. Remedy tier assignment
2. Your confidence level
3. Brief reasoning
4. Any additional information needed (if applicable)

Complaint: {target_example['complaint_text']}

Tier definitions and context:
0 = No Action (65.6% of cases)
   - The bank will not take any remedial action
   - Complaint is resolved, no bank error found, or complaint is frivolous
   - Customer concerns addressed through explanation only

1 = Non-Monetary Action (31.8% of cases)
   - The bank fixes a process, corrects customer data, or improves procedures
   - No financial compensation but bank takes corrective measures
   - Process improvements, policy changes, staff training, data corrections

2 = Monetary Action (2.6% of cases)
   - The bank will reimburse the customer for losses and costs
   - Financial compensation for damages, fees, or losses caused by bank error
   - Refunds, fee reversals, interest adjustments, damages payments

Confidence levels:
- confident: You have sufficient information to make an accurate tier assignment
- need_more_info: You need additional information to make a fair determination
- uncertain: The complaint is ambiguous or borderline between tiers

Provide your analysis with tier, confidence, reasoning, and any information needed."""

    # Show the exact messages array
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    print("EXACT 5-SHOT PROMPT FOR PROCESS DISCRIMINATION:")
    print("=" * 80)
    print("client.chat.completions.create(")
    print("    model='gpt-4o-mini',")
    print("    response_model=ProcessAnalysis,")
    print("    messages=")
    print(json.dumps(messages, indent=4))
    print("    temperature=0.0,")
    print("    max_retries=2")
    print(")")
    print()
    print("TARGET CASE INFO:")
    print(f"Case ID: {target_example['case_id']}")
    print(f"Ground Truth Tier: {target_example['tier']}")
    print(f"Selected Examples: {[ex['case_id'] for ex in nshot_examples]}")
    print(f"Selected Example Tiers: {[ex['tier'] for ex in nshot_examples]}")
    print()
    print("EXPECTED RESPONSE FORMAT:")
    print("{")
    print('  "tier": 0,')
    print('  "confidence": "confident",')
    print('  "reasoning": "This appears to be a debt collection harassment case...",')
    print('  "information_needed": null')
    print("}")

    cursor.close()
    connection.close()


if __name__ == "__main__":
    show_exact_5shot_prompt()