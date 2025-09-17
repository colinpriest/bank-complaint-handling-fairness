#!/usr/bin/env python3
"""
Test embedding generation for a small sample
"""

import os
from sqlalchemy import create_engine, text
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
Session = sessionmaker(bind=engine)
session = Session()

# Check how many records have embeddings
result = session.execute(text("""
    SELECT
        COUNT(*) as total,
        COUNT(vector_embeddings) as with_embeddings,
        COUNT(*) - COUNT(vector_embeddings) as missing_embeddings
    FROM ground_truth
"""))

row = result.fetchone()
print(f"Total records: {row[0]:,}")
print(f"With embeddings: {row[1]:,}")
print(f"Missing embeddings: {row[2]:,}")

# Check a sample embedding
result = session.execute(text("""
    SELECT
        example_id,
        LENGTH(vector_embeddings) as embedding_length,
        LEFT(vector_embeddings, 100) as embedding_sample
    FROM ground_truth
    WHERE vector_embeddings IS NOT NULL
    LIMIT 1
"""))

row = result.fetchone()
if row:
    print(f"\nSample embedding (example_id={row[0]}):")
    print(f"  Length: {row[1]} characters")
    print(f"  Sample: {row[2]}...")

session.close()