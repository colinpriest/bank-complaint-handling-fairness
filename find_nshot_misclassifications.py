#!/usr/bin/env python3
"""
Find n-shot experiments where LLM chose tier 1 but ground truth was tier 0
"""

import os
import psycopg2
from dotenv import load_dotenv
import json

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

def find_tier1_misclassifications():
    """Find cases where n-shot predicted tier 1 but ground truth was tier 0"""

    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        # Query for n-shot tier 1 predictions with tier 0 ground truth
        query = """
        SELECT
            be.case_id,
            be.user_prompt,
            be.system_prompt,
            be.system_response,
            be.llm_simplified_tier,
            gt.simplified_ground_truth_tier,
            gt.consumer_complaint_text,
            gt.product,
            gt.issue,
            gt.sub_product,
            gt.sub_issue,
            be.reasoning,
            be.information_needed
        FROM baseline_experiments be
        JOIN ground_truth gt ON be.case_id = gt.case_id
        WHERE be.decision_method = 'n-shot'
        AND be.llm_simplified_tier = 1
        AND gt.simplified_ground_truth_tier = 0
        LIMIT 5
        """

        cursor.execute(query)
        results = cursor.fetchall()

        print(f"Found {len(results)} n-shot tier 1 misclassifications (should be tier 0)\n")
        print("="*80)

        misclassified_cases = []

        for i, row in enumerate(results, 1):
            case_id = row[0]
            user_prompt = row[1]
            system_prompt = row[2]
            llm_response = row[3]
            predicted_tier = row[4]
            true_tier = row[5]
            complaint_text = row[6]
            product = row[7]
            issue = row[8]
            sub_product = row[9]
            sub_issue = row[10]
            reasoning = row[11]
            info_needed = row[12]

            print(f"\n{'='*80}")
            print(f"CASE {i}: Case ID {case_id}")
            print(f"{'='*80}")
            print(f"\nProduct: {product}")
            print(f"Sub-product: {sub_product}")
            print(f"Issue: {issue}")
            print(f"Sub-issue: {sub_issue}")
            print(f"\nPredicted: Tier {predicted_tier} (Non-monetary action)")
            print(f"Actual: Tier {true_tier} (No action required)")

            print(f"\n--- COMPLAINT ---")
            print(f"{complaint_text[:500]}..." if len(complaint_text) > 500 else complaint_text)

            if reasoning:
                print(f"\n--- LLM REASONING ---")
                print(f"{reasoning[:500]}..." if len(reasoning) > 500 else reasoning)

            print(f"\n--- LLM RESPONSE ---")
            print(f"{llm_response[:500]}..." if len(llm_response) > 500 else llm_response)

            # Extract n-shot examples from the user prompt
            print(f"\n--- N-SHOT EXAMPLES PROVIDED ---")
            if "Example" in user_prompt:
                # Try to extract the examples
                lines = user_prompt.split('\n')
                example_started = False
                example_text = []
                for line in lines:
                    if line.strip().startswith('Example'):
                        example_started = True
                        if example_text:  # Print previous example
                            print('\n'.join(example_text[:10]))  # First 10 lines
                            example_text = []
                        print(f"\n{line}")
                    elif example_started and line.strip().startswith('Tier:'):
                        print(f"{line}")
                        example_started = False
                    elif example_started:
                        example_text.append(line)

            # Save case data for analysis
            case_data = {
                'case_id': case_id,
                'product': product,
                'issue': issue,
                'complaint': complaint_text,
                'sub_issue': sub_issue,
                'predicted_tier': predicted_tier,
                'true_tier': true_tier,
                'llm_response': llm_response,
                'reasoning': reasoning,
                'info_needed': info_needed
            }
            misclassified_cases.append(case_data)

        # Save to JSON for ChatGPT analysis
        with open('nshot_misclassifications.json', 'w') as f:
            json.dump(misclassified_cases, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Saved {len(misclassified_cases)} cases to nshot_misclassifications.json for analysis")

        return misclassified_cases

    except Exception as e:
        print(f"Error: {e}")
        return []

    finally:
        connection.close()

if __name__ == "__main__":
    cases = find_tier1_misclassifications()