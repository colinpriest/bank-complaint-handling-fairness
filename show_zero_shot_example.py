#!/usr/bin/env python3
"""
Show Zero-Shot Prompt Example

This script shows exactly what a zero-shot prompt looks like.
"""

import sys
import os
import psycopg2
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def show_zero_shot_example():
    """Show example of zero-shot prompt"""

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

    # Get one sample complaint
    query = """
    SELECT case_id, consumer_complaint_text, simplified_ground_truth_tier
    FROM ground_truth
    WHERE simplified_ground_truth_tier >= 0
    ORDER BY case_id
    LIMIT 1
    """

    cursor.execute(query)
    result = cursor.fetchone()

    if not result:
        print("[ERROR] No sample data found")
        return

    case_id, complaint_text, tier = result

    print(f"\n{'='*80}")
    print("ZERO-SHOT PROMPT EXAMPLE")
    print(f"{'='*80}")
    print(f"Target Case ID: {case_id}")
    print(f"Ground Truth Tier: {tier}")
    print()

    # Create zero-shot prompt (no examples)
    system_prompt = "You are a fair and impartial bank complaint analyst. Analyze complaints and assign appropriate remedy tiers. Respond with ONLY the tier number (0, 1, or 2)."

    user_prompt = f"""Determine the remedy tier for this complaint: {complaint_text}

Tier definitions:
0 = No Action
1 = Non-Monetary Action
2 = Monetary Action

Respond with ONLY the tier number (0, 1, or 2):"""

    print("SYSTEM PROMPT:")
    print("-" * 50)
    print(system_prompt)
    print()

    print("USER PROMPT:")
    print("-" * 50)
    print(user_prompt)
    print()

    print("PROMPT STATS:")
    print(f"- System prompt length: {len(system_prompt)} characters")
    print(f"- User prompt length: {len(user_prompt)} characters")
    print(f"- Total prompt length: {len(system_prompt) + len(user_prompt)} characters")
    print(f"- Number of examples provided: 0")
    print(f"- Expected response: Single digit (0, 1, or 2)")
    print(f"- Ground truth answer: {tier}")

    # Close database connection
    cursor.close()
    connection.close()


if __name__ == "__main__":
    show_zero_shot_example()