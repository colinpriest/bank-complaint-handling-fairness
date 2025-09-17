#!/usr/bin/env python3
"""
Add NShotOptimizationResults Table

This script creates a new table to store detailed results from each
n-shot optimization parameter combination for visualization and analysis.
"""

import os
import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, Float, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

# Load environment variables
load_dotenv()

# Create SQLAlchemy base
Base = declarative_base()

class NShotOptimizationResults(Base):
    """Table to store detailed results for each n-shot optimization parameter combination"""
    __tablename__ = 'nshot_optimization_results'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Experiment metadata
    experiment_timestamp = Column(DateTime, nullable=False)
    experiment_type = Column(String(50), default='accuracy')  # 'accuracy' or 'process_discrimination'

    # Parameter combination
    n_value = Column(Integer, nullable=False)  # Number of shots (0, 5, 6, 7, 8, 9, 10)
    alpha_value = Column(Float, nullable=False)  # Alpha parameter (0.3 to 0.7)

    # Sample information
    sample_size = Column(Integer, nullable=False)  # Number of test cases used
    total_ground_truth_examples = Column(Integer)  # Total examples available

    # Performance metrics
    correct_predictions = Column(Integer, nullable=False)  # Number of correct predictions
    total_predictions = Column(Integer, nullable=False)  # Number of total predictions
    accuracy_score = Column(Float, nullable=False)  # Accuracy (correct/total)

    # Process discrimination metrics (for process discrimination experiments)
    confident_decisions = Column(Integer, default=0)  # Number of confident decisions
    uncertain_decisions = Column(Integer, default=0)  # Number of uncertain decisions
    need_info_decisions = Column(Integer, default=0)  # Number of "need more info" decisions

    confident_rate = Column(Float)  # Rate of confident decisions
    uncertain_rate = Column(Float)  # Rate of uncertain decisions
    need_info_rate = Column(Float)  # Rate of needing more information

    # Timing information
    execution_time_seconds = Column(Float)  # Time taken for this parameter combination

    # Additional metadata
    notes = Column(Text)  # Any additional notes about this run
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_nshot_results_experiment', 'experiment_timestamp'),
        Index('idx_nshot_results_params', 'n_value', 'alpha_value'),
        Index('idx_nshot_results_accuracy', 'accuracy_score'),
        Index('idx_nshot_results_type', 'experiment_type'),
    )


def create_nshot_results_table():
    """Create the NShotOptimizationResults table"""

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

        # Check if table already exists
        try:
            connection = psycopg2.connect(**db_config)
            cursor = connection.cursor()

            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'nshot_optimization_results'
                );
            """)

            table_exists = cursor.fetchone()[0]

            if table_exists:
                print("[INFO] nshot_optimization_results table already exists")

                # Show current schema
                cursor.execute("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = 'nshot_optimization_results'
                    ORDER BY ordinal_position;
                """)

                columns = cursor.fetchall()
                print(f"[INFO] Current table has {len(columns)} columns:")
                for col_name, data_type, nullable in columns:
                    print(f"  - {col_name}: {data_type} ({'NULL' if nullable == 'YES' else 'NOT NULL'})")

            cursor.close()
            connection.close()

        except Exception as e:
            print(f"[WARNING] Could not check existing table: {e}")

        # Create table (will only create if it doesn't exist)
        print("[INFO] Creating nshot_optimization_results table...")
        Base.metadata.create_all(engine)

        print("[SUCCESS] NShotOptimizationResults table created/verified successfully!")

        print("\n[INFO] Table schema:")
        print("  - Stores detailed results for each n-shot parameter combination")
        print("  - Enables visualization of accuracy vs n and alpha")
        print("  - Tracks process discrimination metrics")
        print("  - Includes timing and metadata for analysis")
        print("  - Indexed for efficient querying and visualization")

        return True

    except Exception as e:
        print(f"[ERROR] Failed to create table: {e}")
        return False


if __name__ == "__main__":
    create_nshot_results_table()