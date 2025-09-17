#!/usr/bin/env python3
"""
Quick Database Table Setup Script

This script creates all database tables with the latest schema including
process discrimination fields in the experiments table.
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from database_check import Base

# Load environment variables
load_dotenv()

def setup_tables():
    """Create all database tables with updated schema"""

    # Database configuration
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'fairness_analysis'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', '')
    }

    database_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"

    try:
        print(f"[INFO] Connecting to database: {db_config['database']}")
        engine = create_engine(database_url, echo=False)

        # Create all tables (this will add missing columns to existing tables)
        print("[INFO] Creating/updating database tables...")
        Base.metadata.create_all(engine)

        print("[SUCCESS] Database tables created/updated successfully!")
        print("[INFO] Tables now include:")
        print("  - ground_truth (with vector_embeddings)")
        print("  - experiments (with process_confidence, information_needed, asks_for_info, reasoning)")
        print("  - personas")
        print("  - mitigation_strategies")
        print("  - nshot_optimisation")
        print("  - llm_cache")

        return True

    except Exception as e:
        print(f"[ERROR] Failed to create database tables: {e}")
        return False

if __name__ == "__main__":
    setup_tables()