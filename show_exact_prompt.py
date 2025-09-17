#!/usr/bin/env python3
"""
Show Exact Prompt - displays the literal messages sent to GPT-4o-mini
"""

import sys
import os
import psycopg2
from pathlib import Path
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

def show_exact_prompt():
    """Show the exact messages array sent to GPT-4o-mini"""

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
    case_id, complaint_text, tier = result

    # Create the exact prompts as they would be sent
    system_prompt = "You are a fair and impartial bank complaint analyst. Analyze complaints and assign appropriate remedy tiers. Respond with ONLY the tier number (0, 1, or 2)."

    user_prompt = f"""Determine the remedy tier for this complaint: {complaint_text}

Tier definitions:
0 = No Action
1 = Non-Monetary Action
2 = Monetary Action

Respond with ONLY the tier number (0, 1, or 2):"""

    # Show the exact messages array
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    print("EXACT API CALL TO GPT-4O-MINI:")
    print("=" * 80)
    print("client.chat.completions.create(")
    print("    model='gpt-4o-mini',")
    print("    response_model=TierPrediction,")
    print("    messages=")
    print(json.dumps(messages, indent=4))
    print("    temperature=0.0,")
    print("    max_retries=2")
    print(")")
    print()
    print("GROUND TRUTH ANSWER:", tier)
    print("CASE ID:", case_id)

    cursor.close()
    connection.close()

if __name__ == "__main__":
    show_exact_prompt()