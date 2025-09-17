#!/usr/bin/env python3
"""
Regenerate Experiment Data with New NShotPromptGenerator

This script regenerates experiment data using the improved NShotPromptGenerator
to ensure all experiments use consistent, high-quality prompts with proper
precedent instructions and enhanced tier context.
"""

import sys
import os
import psycopg2
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Add project path
sys.path.insert(0, str(Path(__file__).parent))

from nshot_prompt_generator import NShotPromptGenerator, BiasStrategy, create_persona_dict
from database_check import DatabaseCheck


def regenerate_experiment_data():
    """Regenerate experiment data using the new prompt generator"""

    print("="*80)
    print("REGENERATING EXPERIMENT DATA WITH NEW PROMPT GENERATOR")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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

    # Verify experiments table is empty
    cursor.execute("SELECT COUNT(*) FROM experiments")
    count = cursor.fetchone()[0]
    if count > 0:
        print(f"[WARNING] Experiments table has {count} records. Clear first? (y/n)")
        response = input().lower()
        if response == 'y':
            cursor.execute("DELETE FROM experiments")
            connection.commit()
            print("[INFO] Cleared experiments table")
        else:
            print("[INFO] Keeping existing data, will append new experiments")

    # Get sample ground truth data for demonstration
    cursor.execute("""
        SELECT case_id, consumer_complaint_text, simplified_ground_truth_tier
        FROM ground_truth
        WHERE simplified_ground_truth_tier >= 0
        ORDER BY case_id
        LIMIT 10
    """)

    sample_cases = cursor.fetchall()
    print(f"[INFO] Loaded {len(sample_cases)} sample cases for demonstration")

    # Initialize prompt generator
    generator = NShotPromptGenerator()

    # Test different experiment configurations
    configurations = [
        # Zero-shot experiments
        {'method': 'zero-shot', 'n': 0, 'persona': None, 'bias_strategy': None},
        {'method': 'zero-shot', 'n': 0, 'persona': 'hispanic_female', 'bias_strategy': None},
        {'method': 'zero-shot', 'n': 0, 'persona': 'white_male', 'bias_strategy': BiasStrategy.PERSONA_FAIRNESS},

        # N-shot experiments
        {'method': 'n-shot', 'n': 5, 'persona': None, 'bias_strategy': None},
        {'method': 'n-shot', 'n': 5, 'persona': 'black_female', 'bias_strategy': BiasStrategy.CHAIN_OF_THOUGHT},
        {'method': 'n-shot', 'n': 3, 'persona': 'asian_male', 'bias_strategy': BiasStrategy.STRUCTURED_EXTRACTION},
    ]

    print(f"\n[INFO] Testing {len(configurations)} experiment configurations...")

    experiment_id = 1

    for config in configurations:
        print(f"\n[CONFIG] {config}")

        for i, (case_id, complaint_text, ground_truth_tier) in enumerate(sample_cases):
            if i >= 3:  # Limit to 3 cases per configuration for demo
                break

            # Create target case
            target_case = {'complaint_text': complaint_text}

            # Create n-shot examples (use other cases)
            nshot_examples = []
            if config['n'] > 0:
                other_cases = [(cid, text, tier) for cid, text, tier in sample_cases if cid != case_id]
                for j in range(min(config['n'], len(other_cases))):
                    nshot_examples.append({
                        'complaint_text': other_cases[j][1],
                        'tier': other_cases[j][2]
                    })

            # Create persona if specified
            persona = None
            if config['persona']:
                if config['persona'] == 'hispanic_female':
                    persona = create_persona_dict(
                        name="Maria Rodriguez",
                        ethnicity="Hispanic",
                        gender="female",
                        location="Phoenix, AZ"
                    )
                elif config['persona'] == 'white_male':
                    persona = create_persona_dict(
                        name="John Smith",
                        ethnicity="White",
                        gender="male",
                        location="Dallas, TX"
                    )
                elif config['persona'] == 'black_female':
                    persona = create_persona_dict(
                        name="Keisha Washington",
                        ethnicity="Black",
                        gender="female",
                        location="Atlanta, GA"
                    )
                elif config['persona'] == 'asian_male':
                    persona = create_persona_dict(
                        name="David Chen",
                        ethnicity="Asian",
                        gender="male",
                        location="San Francisco, CA"
                    )

            # Generate prompts
            try:
                system_prompt, user_prompt = generator.generate_prompts(
                    target_case=target_case,
                    nshot_examples=nshot_examples,
                    persona=persona,
                    bias_strategy=config['bias_strategy']
                )

                # Store in database (simulate)
                decision_method = f"{config['method']}-{config['n']}" if config['n'] > 0 else config['method']

                insert_query = """
                    INSERT INTO experiments (
                        case_id, decision_method, llm_model, llm_simplified_tier,
                        persona, risk_mitigation_strategy, system_prompt, user_prompt,
                        system_response, process_confidence, reasoning, created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                    )
                """

                cursor.execute(insert_query, (
                    case_id,
                    decision_method,
                    'gpt-4o-mini',
                    -1,  # Placeholder - would be filled by actual LLM call
                    config['persona'],
                    config['bias_strategy'].value if config['bias_strategy'] else None,
                    system_prompt,
                    user_prompt,
                    'PLACEHOLDER - would be filled by LLM response',
                    'confident',  # Placeholder
                    'PLACEHOLDER - would be filled by LLM reasoning'
                ))

                print(f"    Generated experiment {experiment_id} for case {case_id}")
                experiment_id += 1

            except Exception as e:
                print(f"    [ERROR] Failed to generate prompts for case {case_id}: {e}")
                continue

    # Commit all changes
    connection.commit()

    # Show final count
    cursor.execute("SELECT COUNT(*) FROM experiments")
    final_count = cursor.fetchone()[0]

    print(f"\n[SUCCESS] Generated {final_count} experiment records")
    print("Experiments table now contains samples using the new NShotPromptGenerator")

    # Show sample of generated prompts
    print(f"\n[INFO] Sample system prompt:")
    print("="*50)
    print(system_prompt)

    print(f"\n[INFO] Sample user prompt (truncated):")
    print("="*50)
    print(user_prompt[:500] + "..." if len(user_prompt) > 500 else user_prompt)

    cursor.close()
    connection.close()


if __name__ == "__main__":
    regenerate_experiment_data()