#!/usr/bin/env python3
"""
Standalone script to generate embeddings for ground_truth table
"""

import os
import json
import sys
from sqlalchemy import create_engine, text, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from datetime import datetime
import warnings

# Suppress TensorFlow warnings and set environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Suppress TensorFlow logging
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

# Define the GroundTruth model
Base = declarative_base()

class GroundTruth(Base):
    """Ground truth table with CFPB complaint data and remedy tiers"""
    __tablename__ = 'ground_truth'

    case_id = Column(Integer, primary_key=True)
    raw_ground_truth_tier = Column(String(100), nullable=False)
    simplified_ground_truth_tier = Column(Integer, nullable=False)
    example_id = Column(Integer, unique=True, nullable=False)
    consumer_complaint_text = Column(Text, nullable=False)
    complaint_category = Column(String(200))
    product = Column(String(200))
    sub_product = Column(String(200))
    issue = Column(String(500))
    sub_issue = Column(String(500))
    vector_embeddings = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

# Load environment variables
load_dotenv()

# Database configuration
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'fairness_analysis')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')

def generate_embeddings(batch_size=100, model_name='all-MiniLM-L6-v2'):
    """Generate embeddings for records missing them"""

    print("="*80)
    print("GENERATING VECTOR EMBEDDINGS")
    print("="*80)

    # Connect to database
    database_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(database_url, echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Check current status
    total_count = session.query(GroundTruth).count()
    missing_count = session.query(GroundTruth).filter(
        (GroundTruth.vector_embeddings == None) |
        (GroundTruth.vector_embeddings == '')
    ).count()

    print(f"Total records: {total_count:,}")
    print(f"Records with embeddings: {total_count - missing_count:,}")
    print(f"Records missing embeddings: {missing_count:,}")

    if missing_count == 0:
        print("\n[SUCCESS] All records already have embeddings!")
        return

    print(f"\n[INFO] Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"[INFO] Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Process in batches
    print(f"\n[INFO] Processing {missing_count:,} records in batches of {batch_size}...")

    processed = 0
    while True:
        # Get batch of records without embeddings
        records = session.query(GroundTruth).filter(
            (GroundTruth.vector_embeddings == None) |
            (GroundTruth.vector_embeddings == '')
        ).limit(batch_size).all()

        if not records:
            break

        # Prepare texts for embedding
        texts = []
        for record in records:
            text_parts = []

            # Primary content
            if record.consumer_complaint_text:
                text_parts.append(record.consumer_complaint_text)

            # Add metadata for context
            if record.product:
                text_parts.append(f"Product: {record.product}")
            if record.sub_product:
                text_parts.append(f"Sub-product: {record.sub_product}")
            if record.issue:
                text_parts.append(f"Issue: {record.issue}")
            if record.sub_issue:
                text_parts.append(f"Sub-issue: {record.sub_issue}")
            if record.complaint_category:
                text_parts.append(f"Category: {record.complaint_category}")

            combined_text = " ".join(text_parts)
            texts.append(combined_text)

        # Generate embeddings
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=batch_size)

        # Update records
        for record, embedding in zip(records, embeddings):
            embedding_json = json.dumps(embedding.tolist())
            record.vector_embeddings = embedding_json

        # Commit batch
        session.commit()
        processed += len(records)

        # Progress update
        percentage = (processed / missing_count) * 100
        print(f"[PROGRESS] {processed:,}/{missing_count:,} ({percentage:.1f}%)")

        # Emergency stop check
        if processed >= missing_count:
            break

    # Final verification
    final_missing = session.query(GroundTruth).filter(
        (GroundTruth.vector_embeddings == None) |
        (GroundTruth.vector_embeddings == '')
    ).count()

    print("\n" + "="*80)
    print("EMBEDDING GENERATION COMPLETE")
    print("="*80)
    print(f"Processed: {processed:,} records")
    print(f"Remaining without embeddings: {final_missing:,}")

    if final_missing == 0:
        print("\n[SUCCESS] All records now have vector embeddings!")
    else:
        print(f"\n[WARNING] {final_missing:,} records still missing embeddings")

    session.close()

if __name__ == "__main__":
    # Allow batch size to be passed as argument
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    generate_embeddings(batch_size=batch_size)