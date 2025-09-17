#!/usr/bin/env python3
"""
Check Experiments Table Schema

This script checks what columns exist in the experiments table
and adds any missing columns needed for process discrimination.
"""

import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_and_update_experiments_table():
    """Check and update experiments table schema"""

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

        # Check what columns exist in experiments table
        cursor.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'experiments'
            ORDER BY ordinal_position;
        """)

        existing_columns = cursor.fetchall()
        print("[INFO] Current experiments table columns:")
        for col_name, data_type in existing_columns:
            print(f"  - {col_name}: {data_type}")

        # Check if process discrimination columns exist
        column_names = [col[0] for col in existing_columns]

        missing_columns = []

        # Required columns for process discrimination
        required_columns = {
            'process_confidence': 'VARCHAR(20)',
            'information_needed': 'TEXT',
            'asks_for_info': 'BOOLEAN DEFAULT FALSE',
            'reasoning': 'TEXT'
        }

        for col_name, col_type in required_columns.items():
            if col_name not in column_names:
                missing_columns.append((col_name, col_type))

        if missing_columns:
            print(f"\n[INFO] Adding {len(missing_columns)} missing columns...")

            for col_name, col_type in missing_columns:
                try:
                    alter_sql = f"ALTER TABLE experiments ADD COLUMN {col_name} {col_type};"
                    print(f"[INFO] Executing: {alter_sql}")
                    cursor.execute(alter_sql)
                    connection.commit()
                    print(f"[SUCCESS] Added column: {col_name}")
                except Exception as e:
                    print(f"[ERROR] Failed to add column {col_name}: {e}")

        else:
            print("\n[INFO] All required columns already exist!")

        # Verify final state
        cursor.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'experiments'
            ORDER BY ordinal_position;
        """)

        final_columns = cursor.fetchall()
        print(f"\n[INFO] Final experiments table has {len(final_columns)} columns:")
        for col_name, data_type in final_columns:
            print(f"  - {col_name}: {data_type}")

        cursor.close()
        connection.close()
        return True

    except Exception as e:
        print(f"[ERROR] Database operation failed: {e}")
        return False

if __name__ == "__main__":
    check_and_update_experiments_table()