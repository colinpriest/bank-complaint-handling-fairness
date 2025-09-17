#!/usr/bin/env python3
"""
Test experiment setup with a small sample
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

# Calculate expected experiment count
print("="*80)
print("EXPERIMENT SETUP CALCULATION")
print("="*80)

# Get counts
total_cases = session.execute(text("SELECT COUNT(*) FROM ground_truth")).scalar()
total_personas = session.execute(text("SELECT COUNT(*) FROM personas")).scalar()
total_strategies = session.execute(text("SELECT COUNT(*) FROM mitigation_strategies")).scalar()

print(f"Ground truth cases: {total_cases:,}")
print(f"Personas: {total_personas}")
print(f"Mitigation strategies: {total_strategies}")

# Calculate experiment counts per case
# For each case:
# 1. Baseline (1 per case)
# 2. Persona-injected (1 per persona per case)
# 3. Persona + mitigation (personas × non-DPP strategies per case)

# Check for DPP strategies
strategies = session.execute(text("SELECT key FROM mitigation_strategies")).fetchall()
non_dpp_strategies = [s[0] for s in strategies if 'dpp' not in s[0].lower()]
print(f"Non-DPP strategies: {len(non_dpp_strategies)} ({', '.join(non_dpp_strategies)})")

# Per case calculations for each method (zero-shot and n-shot)
baseline_per_case = 1
persona_per_case = total_personas
mitigated_per_case = total_personas * len(non_dpp_strategies)

experiments_per_case = baseline_per_case + persona_per_case + mitigated_per_case
print(f"\nExperiments per case per method:")
print(f"  - Baseline: {baseline_per_case}")
print(f"  - Persona-injected: {persona_per_case}")
print(f"  - Persona + mitigated: {mitigated_per_case}")
print(f"  - Total per case per method: {experiments_per_case}")

# Total experiments (both zero-shot and n-shot)
total_zero_shot = total_cases * experiments_per_case
total_n_shot = total_cases * experiments_per_case
total_experiments = total_zero_shot + total_n_shot

print(f"\nTotal experiments:")
print(f"  - Zero-shot: {total_zero_shot:,}")
print(f"  - N-shot: {total_n_shot:,}")
print(f"  - Grand total: {total_experiments:,}")

# Check current experiment count
current_experiments = session.execute(text("SELECT COUNT(*) FROM experiments")).scalar()
print(f"\nCurrent experiments in database: {current_experiments:,}")

if current_experiments == 0:
    print("\n[INFO] No experiments exist. They will be created when running the main script.")
else:
    print(f"\n[INFO] Experiments already exist. Expected vs actual:")
    print(f"  Expected: {total_experiments:,}")
    print(f"  Actual: {current_experiments:,}")

    if current_experiments == total_experiments:
        print("  ✅ Counts match!")
    else:
        print("  ⚠️  Counts don't match")

session.close()