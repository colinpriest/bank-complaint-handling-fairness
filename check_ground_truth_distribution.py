#!/usr/bin/env python3
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'fairness_analysis'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', '')
    }
    return psycopg2.connect(**db_config)

connection = get_db_connection()
cursor = connection.cursor()

print("=== GROUND TRUTH TIER DISTRIBUTION ===")
cursor.execute("""
SELECT
    simplified_ground_truth_tier,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as percentage
FROM ground_truth
GROUP BY simplified_ground_truth_tier
ORDER BY simplified_ground_truth_tier;
""")

total_cases = 0
for row in cursor.fetchall():
    tier, count, percentage = row
    total_cases += count
    print(f"Tier {tier}: {count} cases ({percentage}%)")

print(f"\nTotal ground truth cases: {total_cases}")

print("\n=== N-SHOT EXAMPLES USED IN EXPERIMENTS ===")
# Check what examples were actually used in n-shot experiments
cursor.execute("""
SELECT DISTINCT user_prompt FROM baseline_experiments
WHERE decision_method = 'n-shot'
LIMIT 3;
""")

results = cursor.fetchall()
if results:
    print("Sample n-shot prompts to check example distribution:")
    for i, (prompt,) in enumerate(results):
        print(f"\n--- Sample {i+1} ---")
        # Look for example patterns in the prompt
        lines = prompt.split('\n')
        example_count = 0
        tier_pattern = {}

        for line in lines:
            if line.startswith('Example') and 'Tier:' in line:
                example_count += 1
                # Extract tier from line like "Tier: 0"
                tier_part = line.split('Tier:')[-1].strip()
                if tier_part.isdigit():
                    tier = int(tier_part)
                    tier_pattern[tier] = tier_pattern.get(tier, 0) + 1

        print(f"Examples found: {example_count}")
        if tier_pattern:
            print(f"Tier distribution: {tier_pattern}")
        else:
            # Try different pattern
            print("First 500 chars of prompt:")
            print(prompt[:500] + "..." if len(prompt) > 500 else prompt)

connection.close()