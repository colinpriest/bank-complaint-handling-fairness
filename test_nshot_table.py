#!/usr/bin/env python3
"""
Test nshot_optimisation table creation
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

# Check if nshot_optimisation table exists
inspector = inspect(engine)
tables = inspector.get_table_names()

print("="*80)
print("NSHOT_OPTIMISATION TABLE TEST")
print("="*80)

if 'nshot_optimisation' in tables:
    print("[SUCCESS] nshot_optimisation table exists")

    # Get column information
    columns = inspector.get_columns('nshot_optimisation')
    print(f"\nTable has {len(columns)} columns:")
    for col in columns:
        print(f"  - {col['name']:<20} {str(col['type']):<20} {'NOT NULL' if not col['nullable'] else 'NULL'}")

    # Check row count
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM nshot_optimisation"))
        count = result.scalar()
        print(f"\nRecords in table: {count:,}")

    # Check indexes
    indexes = inspector.get_indexes('nshot_optimisation')
    if indexes:
        print(f"\nIndexes ({len(indexes)}):")
        for idx in indexes:
            print(f"  - {idx['name']}: {', '.join(idx['column_names'])}")
    else:
        print("\nNo indexes found")

    # Test inserting a sample record
    print("\n[INFO] Testing record insertion...")
    with engine.connect() as conn:
        try:
            conn.execute(text("""
                INSERT INTO nshot_optimisation (optimal_n, optimal_alpha)
                VALUES (5, 0.75)
                ON CONFLICT DO NOTHING
            """))
            conn.commit()

            # Check the inserted record
            result = conn.execute(text("SELECT * FROM nshot_optimisation LIMIT 1"))
            row = result.fetchone()
            if row:
                print(f"  Sample record: optimal_n={row[1]}, optimal_alpha={row[2]}")

            print("[SUCCESS] Record insertion successful")
        except Exception as e:
            print(f"[ERROR] Record insertion failed: {str(e)}")

else:
    print("[INFO] nshot_optimisation table does not exist yet")
    print("[INFO] Run bank-complaint-handling.py to create it")

print("\n" + "="*80)
print("ALL TABLES IN DATABASE")
print("="*80)
for table in sorted(tables):
    print(f"  - {table}")