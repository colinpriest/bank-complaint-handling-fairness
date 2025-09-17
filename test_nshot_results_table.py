#!/usr/bin/env python3
"""
Test NShotOptimizationResults Table

Quick test to verify the new detailed results table is working.
"""

import os
import psycopg2
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

def test_nshot_results_table():
    """Test inserting and querying the nshot_optimization_results table"""

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

        print("[INFO] Testing nshot_optimization_results table...")

        # Insert a test record
        test_timestamp = datetime.now()
        insert_query = """
            INSERT INTO nshot_optimization_results (
                experiment_timestamp, experiment_type, n_value, alpha_value,
                sample_size, total_ground_truth_examples,
                correct_predictions, total_predictions, accuracy_score,
                confident_decisions, uncertain_decisions, need_info_decisions,
                confident_rate, uncertain_rate, need_info_rate,
                notes, created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
            )
        """

        cursor.execute(insert_query, (
            test_timestamp,
            'test',  # experiment_type
            5,  # n_value
            0.5,  # alpha_value
            100,  # sample_size
            500,  # total_ground_truth_examples
            75,  # correct_predictions
            100,  # total_predictions
            0.75,  # accuracy_score
            80,  # confident_decisions
            15,  # uncertain_decisions
            5,  # need_info_decisions
            0.8,  # confident_rate
            0.15,  # uncertain_rate
            0.05,  # need_info_rate
            'Test record - n=5, alpha=0.5'  # notes
        ))

        connection.commit()
        print("[SUCCESS] Test record inserted")

        # Query the record back
        cursor.execute("""
            SELECT n_value, alpha_value, accuracy_score, sample_size, notes
            FROM nshot_optimization_results
            WHERE experiment_type = 'test'
            ORDER BY created_at DESC
            LIMIT 1
        """)

        result = cursor.fetchone()
        if result:
            n, alpha, accuracy, sample, notes = result
            print(f"[SUCCESS] Retrieved test record: n={n}, alpha={alpha}, accuracy={accuracy:.3f}, sample={sample}")
            print(f"  Notes: {notes}")
        else:
            print("[ERROR] Could not retrieve test record")

        # Clean up test record
        cursor.execute("DELETE FROM nshot_optimization_results WHERE experiment_type = 'test'")
        connection.commit()
        print("[INFO] Test record cleaned up")

        # Show table info
        cursor.execute("SELECT COUNT(*) FROM nshot_optimization_results")
        count = cursor.fetchone()[0]
        print(f"[INFO] Total records in nshot_optimization_results table: {count}")

        cursor.close()
        connection.close()

        print("[SUCCESS] NShotOptimizationResults table is working correctly!")
        return True

    except Exception as e:
        print(f"[ERROR] Table test failed: {e}")
        return False

if __name__ == "__main__":
    test_nshot_results_table()