#!/usr/bin/env python3
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'fairness_analysis'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', '')
    }
    return psycopg2.connect(**db_config)

connection = get_db_connection()
cursor = connection.cursor()

print("=== BASELINE_EXPERIMENTS COLUMNS ===")
cursor.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'baseline_experiments' ORDER BY ordinal_position;")
for row in cursor.fetchall():
    print(f"{row[0]}: {row[1]}")

print("\n=== GROUND_TRUTH COLUMNS ===")
cursor.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'ground_truth' ORDER BY ordinal_position;")
for row in cursor.fetchall():
    print(f"{row[0]}: {row[1]}")

print("\n=== SAMPLE BASELINE_EXPERIMENTS DATA ===")
cursor.execute("SELECT * FROM baseline_experiments LIMIT 3;")
columns = [desc[0] for desc in cursor.description]
print("Columns:", ", ".join(columns))
for row in cursor.fetchall():
    print("Row:", row)

connection.close()