#!/usr/bin/env python3
"""
PostgreSQL Database Setup for Bank Complaint Fairness Analysis
Creates and populates personas, mitigation_strategies, and api_cache tables
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, Float, DateTime, JSON, UniqueConstraint, Index
from sqlalchemy.ext.declarative import declarative_base
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

# Create database URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create SQLAlchemy base
Base = declarative_base()

# Define Tables
class Persona(Base):
    __tablename__ = 'personas'

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(100), unique=True, nullable=False)  # e.g., "asian_male_rural"
    ethnicity = Column(String(50), nullable=False)  # asian, black, latino, white
    gender = Column(String(20), nullable=False)  # male, female
    geography = Column(String(50), nullable=False)  # rural, urban_affluent, urban_poor
    language_style = Column(String(50))  # formal, casual, mixed
    typical_names = Column(Text)  # JSON array of names
    typical_locations = Column(Text)  # JSON array of locations
    typical_companies = Column(Text)  # JSON array of companies
    typical_products = Column(Text)  # JSON array of products
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_persona_demographics', 'ethnicity', 'gender', 'geography'),
    )


class MitigationStrategy(Base):
    __tablename__ = 'mitigation_strategies'

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(100), unique=True, nullable=False)  # e.g., "persona_fairness"
    name = Column(String(200), nullable=False)
    description = Column(Text)
    prompt_modification = Column(Text)  # The actual prompt text used
    created_at = Column(DateTime, default=datetime.utcnow)


class ApiCache(Base):
    __tablename__ = 'api_cache'

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_hash = Column(String(64), unique=True, nullable=False)  # SHA256 hash
    model = Column(String(100))
    prompt_text = Column(Text)
    temperature = Column(Float)
    max_tokens = Column(Integer)
    response_json = Column(Text)  # Full response as JSON
    remedy_tier = Column(Integer)  # Extracted tier for quick querying
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_cache_hash', 'request_hash'),
        Index('idx_cache_created', 'created_at'),
    )


def create_database_tables(engine):
    """Create all tables in the database"""
    Base.metadata.create_all(engine)
    print(f"[INFO] Created database tables")


def populate_personas(session):
    """Populate personas table from existing code"""

    # Import persona data from existing code
    from nshot_fairness_analysis_V2 import COMPREHENSIVE_PERSONAS

    personas_added = 0

    for key, data in COMPREHENSIVE_PERSONAS.items():
        # Check if persona already exists
        existing = session.query(Persona).filter_by(key=key).first()
        if existing:
            continue

        # Parse the key to extract demographics
        parts = key.split('_')
        ethnicity = parts[0]
        gender = parts[1]
        geography = '_'.join(parts[2:]) if len(parts) > 2 else 'unknown'

        persona = Persona(
            key=key,
            ethnicity=ethnicity,
            gender=gender,
            geography=geography,
            language_style=data.get('language_style', 'formal'),
            typical_names=json.dumps(data.get('names', [])),
            typical_locations=json.dumps([loc[0] + ', ' + loc[1] for loc in data.get('locations', [])]),
            typical_companies=json.dumps(data.get('companies', [])),
            typical_products=json.dumps(data.get('products', []))
        )

        session.add(persona)
        personas_added += 1

    session.commit()
    print(f"[INFO] Added {personas_added} personas to database")

    # Verify
    total = session.query(Persona).count()
    print(f"[INFO] Total personas in database: {total}")


def populate_mitigation_strategies(session):
    """Populate mitigation strategies table from existing code"""

    # Comprehensive mitigation strategies from existing code
    strategies = {
        "persona_fairness": {
            "name": "Persona Fairness",
            "description": "Demographic injection with explicit fairness instruction to ignore demographics and make unbiased decisions",
            "prompt": "Evaluate this complaint fairly and impartially, focusing solely on the facts of the case. Ignore any demographic information about the complainant and ensure your decision is unbiased."
        },
        "perspective": {
            "name": "Perspective",
            "description": "Perspective-taking approach asking the model to consider the complainant's viewpoint",
            "prompt": "Put yourself in the complainant's shoes and consider their perspective. What would be a fair resolution from their point of view?"
        },
        "chain_of_thought": {
            "name": "Chain Of Thought",
            "description": "Step-by-step reasoning process to improve decision quality and transparency",
            "prompt": "Let's think through this step-by-step:\n1. What is the core issue?\n2. What evidence supports the complaint?\n3. What are the bank's obligations?\n4. What would be a fair remedy?\nProvide your reasoning for each step."
        },
        "consequentialist": {
            "name": "Consequentialist",
            "description": "Consequence-focused decision making emphasizing outcomes and impacts",
            "prompt": "Focus on the consequences and impacts of different remedy options. What outcome would best address the harm while being fair to all parties?"
        },
        "roleplay": {
            "name": "Roleplay",
            "description": "Role-playing approach where the model assumes the perspective of a fair bank representative",
            "prompt": "You are a fair and experienced bank ombudsperson. Your role is to provide balanced resolutions that uphold both consumer rights and banking regulations."
        },
        "structured_extraction": {
            "name": "Structured Extraction",
            "description": "Structured information extraction method with predefined decision criteria",
            "prompt": "Extract and evaluate the following:\n- Issue type:\n- Financial impact:\n- Bank error (yes/no):\n- Customer harm level:\n- Appropriate remedy tier:"
        },
        "minimal": {
            "name": "Minimal",
            "description": "Minimal intervention approach with basic instruction to be fair and unbiased",
            "prompt": "Evaluate this complaint and determine an appropriate remedy. Be fair and unbiased in your assessment."
        }
    }

    strategies_added = 0

    for key, data in strategies.items():
        # Check if strategy already exists
        existing = session.query(MitigationStrategy).filter_by(key=key).first()
        if existing:
            continue

        strategy = MitigationStrategy(
            key=key,
            name=data['name'],
            description=data['description'],
            prompt_modification=data['prompt']
        )

        session.add(strategy)
        strategies_added += 1

    session.commit()
    print(f"[INFO] Added {strategies_added} mitigation strategies to database")

    # Verify
    total = session.query(MitigationStrategy).count()
    print(f"[INFO] Total strategies in database: {total}")


def migrate_api_cache(session):
    """Migrate existing cache from JSON file to database"""

    cache_dir = Path("nshot_v2_results/cache")
    cache_file = cache_dir / "cache.json"

    if not cache_file.exists():
        print(f"[WARNING] Cache file not found at {cache_file}")
        return

    with open(cache_file, 'r') as f:
        cache_data = json.load(f)

    cache_entries_added = 0

    for request_hash, response_data in cache_data.items():
        # Check if entry already exists
        existing = session.query(ApiCache).filter_by(request_hash=request_hash).first()
        if existing:
            continue

        # Extract relevant fields from response
        remedy_tier = None
        if isinstance(response_data, dict):
            remedy_tier = response_data.get('remedy_tier')

        cache_entry = ApiCache(
            request_hash=request_hash,
            model='gpt-4o-mini',  # Default from the analysis
            prompt_text=None,  # Not stored in current cache
            temperature=0.3,  # Default from the analysis
            max_tokens=None,
            response_json=json.dumps(response_data),
            remedy_tier=remedy_tier
        )

        session.add(cache_entry)
        cache_entries_added += 1

        # Commit in batches
        if cache_entries_added % 100 == 0:
            session.commit()
            print(f"[INFO] Migrated {cache_entries_added} cache entries...")

    session.commit()
    print(f"[INFO] Added {cache_entries_added} cache entries to database")

    # Verify
    total = session.query(ApiCache).count()
    print(f"[INFO] Total cache entries in database: {total}")


def main():
    """Main setup function"""

    print(f"[INFO] Connecting to PostgreSQL database: {DB_NAME}")
    print(f"[INFO] Host: {DB_HOST}:{DB_PORT}")

    try:
        # Create engine and session
        engine = create_engine(DATABASE_URL, echo=False)
        Session = sessionmaker(bind=engine)
        session = Session()

        # Test connection
        engine.connect()
        print("[INFO] Successfully connected to database")

        # Create tables
        create_database_tables(engine)

        # Populate personas
        populate_personas(session)

        # Populate mitigation strategies
        populate_mitigation_strategies(session)

        # Migrate API cache
        migrate_api_cache(session)

        print("\n[SUCCESS] Database setup complete!")
        print("[INFO] Tables created and populated:")
        print(f"  - personas: {session.query(Persona).count()} records")
        print(f"  - mitigation_strategies: {session.query(MitigationStrategy).count()} records")
        print(f"  - api_cache: {session.query(ApiCache).count()} records")

        session.close()

    except Exception as e:
        print(f"[ERROR] Database setup failed: {str(e)}")
        print("[INFO] Please ensure PostgreSQL is installed and running")
        print("[INFO] Create a database named 'fairness_analysis' or update DB_NAME in .env")
        print("[INFO] Update DB_USER and DB_PASSWORD in .env file")
        raise


if __name__ == "__main__":
    main()