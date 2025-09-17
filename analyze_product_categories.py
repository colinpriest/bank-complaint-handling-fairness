#!/usr/bin/env python3
"""
Analyze product categories in the database to determine best grouping strategy
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def analyze_product_categories():
    """Analyze product distribution in the database"""

    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'fairness_analysis'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', '')
    }

    connection = psycopg2.connect(**db_config)
    cursor = connection.cursor()

    # Get product distribution
    query = """
        SELECT product, COUNT(*) as count
        FROM ground_truth
        WHERE simplified_ground_truth_tier >= 0
        AND vector_embeddings IS NOT NULL
        AND vector_embeddings != ''
        GROUP BY product
        ORDER BY count DESC
    """

    cursor.execute(query)
    results = cursor.fetchall()

    print("=" * 80)
    print("PRODUCT CATEGORY DISTRIBUTION")
    print("=" * 80)

    total_count = sum(count for _, count in results)

    for i, (product, count) in enumerate(results, 1):
        percentage = (count / total_count) * 100
        print(f"{i:2}. {product[:60]:60} {count:6} ({percentage:5.1f}%)")

    print(f"\nTotal unique products: {len(results)}")
    print(f"Total complaints: {total_count}")

    cursor.close()
    connection.close()

if __name__ == "__main__":
    analyze_product_categories()