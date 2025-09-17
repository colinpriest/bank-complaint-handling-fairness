#!/usr/bin/env python3
"""Test PostgreSQL connection before running full setup"""

import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'fairness_analysis')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')

print("Database Configuration:")
print(f"  Host: {DB_HOST}")
print(f"  Port: {DB_PORT}")
print(f"  Database: {DB_NAME}")
print(f"  User: {DB_USER}")
print(f"  Password: {'*' * len(DB_PASSWORD) if DB_PASSWORD else '(not set)'}")

# Create database URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

try:
    print("\nTesting connection...")
    engine = create_engine(DATABASE_URL, echo=False)
    connection = engine.connect()
    print("✓ Successfully connected to PostgreSQL!")

    # Test query
    result = connection.execute("SELECT version()")
    version = result.fetchone()[0]
    print(f"✓ PostgreSQL version: {version}")

    connection.close()
    print("\n[SUCCESS] Database is ready. You can run database_setup.py")

except Exception as e:
    print(f"\n✗ Connection failed: {str(e)}")
    print("\nTroubleshooting steps:")
    print("1. Ensure PostgreSQL is installed and running")
    print("2. Create the database: CREATE DATABASE fairness_analysis;")
    print("3. Update .env file with correct credentials")
    print("4. Ensure the PostgreSQL server accepts connections")