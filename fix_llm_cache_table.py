#!/usr/bin/env python3
"""
Fix LLMCache Table Schema

This script adds missing process discrimination columns to the llm_cache table.
"""

import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fix_llm_cache_table():
    """Add missing columns to llm_cache table"""

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

        # Check what columns exist in llm_cache table
        cursor.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'llm_cache'
            ORDER BY ordinal_position;
        """)

        existing_columns = cursor.fetchall()
        print("[INFO] Current llm_cache table columns:")
        for col_name, data_type in existing_columns:
            print(f"  - {col_name}: {data_type}")

        # Check if required columns exist
        column_names = [col[0] for col in existing_columns]

        missing_columns = []

        # Required columns from database_check.py LLMCache schema
        required_columns = {
            'system_prompt': 'TEXT',
            'case_id': 'INTEGER',
            'response_text': 'TEXT',
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
                    alter_sql = f"ALTER TABLE llm_cache ADD COLUMN {col_name} {col_type};"
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
            WHERE table_name = 'llm_cache'
            ORDER BY ordinal_position;
        """)

        final_columns = cursor.fetchall()
        print(f"\n[INFO] Final llm_cache table has {len(final_columns)} columns:")
        for col_name, data_type in final_columns:
            print(f"  - {col_name}: {data_type}")

        cursor.close()
        connection.close()
        return True

    except Exception as e:
        print(f"[ERROR] Database operation failed: {e}")
        return False

if __name__ == "__main__":
    fix_llm_cache_table()