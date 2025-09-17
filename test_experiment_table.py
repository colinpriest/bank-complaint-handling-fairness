#!/usr/bin/env python3
"""
Test experiment table creation
"""

import os
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'fairness_analysis')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')

database_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(database_url, echo=False)

# Check if experiments table exists
inspector = inspect(engine)
tables = inspector.get_table_names()

print("="*80)
print("DATABASE TABLES")
print("="*80)

for table in sorted(tables):
    print(f"  - {table}")

print("\n" + "="*80)
print("EXPERIMENTS TABLE")
print("="*80)

if 'experiments' in tables:
    print("[SUCCESS] Experiments table exists")

    # Get column information
    columns = inspector.get_columns('experiments')
    print(f"\nTable has {len(columns)} columns:")
    for col in columns:
        print(f"  - {col['name']:<30} {str(col['type']):<20} {'NOT NULL' if not col['nullable'] else 'NULL'}")

    # Check row count
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM experiments"))
        count = result.scalar()
        print(f"\nRecords in table: {count:,}")

    # Check indexes
    indexes = inspector.get_indexes('experiments')
    if indexes:
        print(f"\nIndexes ({len(indexes)}):")
        for idx in indexes:
            print(f"  - {idx['name']}: {', '.join(idx['column_names'])}")
else:
    print("[INFO] Experiments table does not exist yet")
    print("[INFO] Run bank-complaint-handling.py to create it")