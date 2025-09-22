#!/usr/bin/env python3
import os
import psycopg2
from dotenv import load_dotenv
import re

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

print("=== ANALYZING N-SHOT EXAMPLE DISTRIBUTIONS ===")

# Get sample n-shot prompts
cursor.execute("""
SELECT user_prompt FROM baseline_experiments
WHERE decision_method = 'n-shot'
LIMIT 10;
""")

tier_distributions = []
for i, (prompt,) in enumerate(cursor.fetchall()):
    print(f"\n--- Analyzing Prompt {i+1} ---")

    # Look for example patterns
    lines = prompt.split('\n')
    tier_counts = {0: 0, 1: 0, 2: 0}
    example_found = False

    for line in lines:
        if line.strip().startswith('Tier:'):
            example_found = True
            tier_match = re.search(r'Tier:\s*(\d+)', line)
            if tier_match:
                tier = int(tier_match.group(1))
                if tier in tier_counts:
                    tier_counts[tier] += 1

    if example_found:
        total_examples = sum(tier_counts.values())
        print(f"Examples found: {total_examples}")
        print(f"Tier distribution: {tier_counts}")
        if total_examples > 0:
            percentages = {t: (count/total_examples)*100 for t, count in tier_counts.items()}
            print(f"Percentages: T0={percentages[0]:.1f}%, T1={percentages[1]:.1f}%, T2={percentages[2]:.1f}%")
            tier_distributions.append(tier_counts)
    else:
        print("No tier examples found in expected format")
        # Show a sample to debug
        relevant_lines = [line for line in lines if 'Example' in line or 'Tier' in line][:10]
        print("Sample lines containing 'Example' or 'Tier':")
        for line in relevant_lines:
            print(f"  {line}")

# Calculate average distribution
if tier_distributions:
    print(f"\n=== SUMMARY ACROSS {len(tier_distributions)} PROMPTS ===")
    avg_counts = {0: 0, 1: 0, 2: 0}
    for dist in tier_distributions:
        for tier, count in dist.items():
            avg_counts[tier] += count

    total_avg = sum(avg_counts.values())
    if total_avg > 0:
        print(f"Average examples per prompt: {total_avg / len(tier_distributions):.1f}")
        print(f"Average tier distribution:")
        for tier in [0, 1, 2]:
            avg_pct = (avg_counts[tier] / total_avg) * 100
            print(f"  Tier {tier}: {avg_pct:.1f}%")

        print(f"\nExpected distribution (from ground truth):")
        print(f"  Tier 0: 65.2% (actual ground truth)")
        print(f"  Tier 1: 31.6% (actual ground truth)")
        print(f"  Tier 2: 2.6% (actual ground truth)")

connection.close()