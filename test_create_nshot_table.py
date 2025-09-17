#!/usr/bin/env python3
"""
Test creating just the nshot_optimisation table
"""

import os
from sqlalchemy import create_engine, Column, Integer, Float, DateTime, Index
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Database configuration
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'fairness_analysis')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')

# SQLAlchemy base
Base = declarative_base()

class NShotOptimisation(Base):
    """N-shot optimization parameters table"""
    __tablename__ = 'nshot_optimisation'

    id = Column(Integer, primary_key=True, autoincrement=True)
    optimal_n = Column(Integer, nullable=False)  # Optimal number of shots
    optimal_alpha = Column(Float, nullable=False)  # Optimal alpha parameter
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_nshot_optimal_n', 'optimal_n'),
        Index('idx_nshot_optimal_alpha', 'optimal_alpha'),
    )

def test_create_table():
    """Test creating the nshot_optimisation table"""

    print("="*80)
    print("CREATING NSHOT_OPTIMISATION TABLE")
    print("="*80)

    database_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(database_url, echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Create the table
        print("[INFO] Creating nshot_optimisation table...")
        Base.metadata.create_all(engine)
        print("[SUCCESS] Table created successfully")

        # Test the table
        count = session.query(NShotOptimisation).count()
        print(f"[INFO] Current record count: {count}")

        # Insert a test record
        print("[INFO] Inserting test record...")
        test_record = NShotOptimisation(
            optimal_n=5,
            optimal_alpha=0.75
        )
        session.add(test_record)
        session.commit()

        # Verify insertion
        count = session.query(NShotOptimisation).count()
        print(f"[INFO] Record count after insertion: {count}")

        # Show the record
        record = session.query(NShotOptimisation).first()
        if record:
            print(f"[INFO] Sample record: optimal_n={record.optimal_n}, optimal_alpha={record.optimal_alpha}")

        print("\n[SUCCESS] nshot_optimisation table is working correctly!")

        session.close()

    except Exception as e:
        print(f"[ERROR] Failed to create table: {str(e)}")
        session.rollback()
        session.close()

if __name__ == "__main__":
    test_create_table()