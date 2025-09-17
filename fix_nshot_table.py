#!/usr/bin/env python3
"""
Fix NShotOptimisation Table Schema

This script adds missing columns to the nshot_optimisation table.
"""

import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fix_nshot_table():
    """Add missing columns to nshot_optimisation table"""

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

        # Check what columns exist in nshot_optimisation table
        cursor.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'nshot_optimisation'
            ORDER BY ordinal_position;
        """)

        existing_columns = cursor.fetchall()
        print("[INFO] Current nshot_optimisation table columns:")
        for col_name, data_type in existing_columns:
            print(f"  - {col_name}: {data_type}")

        # Check if required columns exist
        column_names = [col[0] for col in existing_columns]

        missing_columns = []

        # Required columns from database_check.py schema
        required_columns = {
            'experiment_type': "VARCHAR(50) DEFAULT 'accuracy'",
            'sample_size': 'INTEGER',
            'accuracy_score': 'FLOAT',
            'information_request_rate': 'FLOAT',
            'confident_rate': 'FLOAT',
            'uncertain_rate': 'FLOAT',
            'need_info_rate': 'FLOAT'
        }

        for col_name, col_type in required_columns.items():
            if col_name not in column_names:
                missing_columns.append((col_name, col_type))

        if missing_columns:
            print(f"\n[INFO] Adding {len(missing_columns)} missing columns...")

            for col_name, col_type in missing_columns:
                try:
                    alter_sql = f"ALTER TABLE nshot_optimisation ADD COLUMN {col_name} {col_type};"
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
            WHERE table_name = 'nshot_optimisation'
            ORDER BY ordinal_position;
        """)

        final_columns = cursor.fetchall()
        print(f"\n[INFO] Final nshot_optimisation table has {len(final_columns)} columns:")
        for col_name, data_type in final_columns:
            print(f"  - {col_name}: {data_type}")

        cursor.close()
        connection.close()
        return True

    except Exception as e:
        print(f"[ERROR] Database operation failed: {e}")
        return False

if __name__ == "__main__":
    fix_nshot_table()