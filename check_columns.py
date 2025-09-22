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

cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'baseline_experiments' ORDER BY ordinal_position;")
columns = [row[0] for row in cursor.fetchall()]
print("baseline_experiments columns:", columns)

cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'ground_truth' ORDER BY ordinal_position;")
columns = [row[0] for row in cursor.fetchall()]
print("\nground_truth columns:", columns)

connection.close()