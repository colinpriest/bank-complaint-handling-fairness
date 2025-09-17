#!/usr/bin/env python3
"""
Test experiment setup with just 10 cases to verify logic
"""

import os
from sqlalchemy import create_engine, text, Column, Integer, String, Text, DateTime
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

# Define models
Base = declarative_base()

class GroundTruth(Base):
    __tablename__ = 'ground_truth'
    case_id = Column(Integer, primary_key=True)
    consumer_complaint_text = Column(Text, nullable=False)

class Persona(Base):
    __tablename__ = 'personas'
    id = Column(Integer, primary_key=True)
    key = Column(String(100), unique=True, nullable=False)
    ethnicity = Column(String(50), nullable=False)
    gender = Column(String(20), nullable=False)
    geography = Column(String(50), nullable=False)

class MitigationStrategy(Base):
    __tablename__ = 'mitigation_strategies'
    id = Column(Integer, primary_key=True)
    key = Column(String(100), unique=True, nullable=False)
    name = Column(String(200), nullable=False)
    prompt_modification = Column(Text)

class Experiment(Base):
    __tablename__ = 'experiments'
    experiment_id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Integer, nullable=False)
    decision_method = Column(String(50), nullable=False)
    llm_model = Column(String(100), nullable=False)
    llm_simplified_tier = Column(Integer, nullable=False)
    persona = Column(String(100))
    gender = Column(String(20))
    ethnicity = Column(String(50))
    geography = Column(String(50))
    risk_mitigation_strategy = Column(String(100))
    system_prompt = Column(Text)
    user_prompt = Column(Text)
    system_response = Column(Text)
    cache_id = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

def test_small_experiment_setup():
    """Test experiment setup with just 10 cases"""

    database_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(database_url, echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()

    print("="*80)
    print("SMALL SCALE EXPERIMENT SETUP TEST")
    print("="*80)

    # Clear any existing test experiments
    print("[INFO] Clearing existing experiments...")
    session.query(Experiment).delete()
    session.commit()

    # Get first 10 ground truth records
    ground_truth_records = session.query(GroundTruth).limit(10).all()
    personas = session.query(Persona).all()
    strategies = session.query(MitigationStrategy).all()

    print(f"[INFO] Testing with {len(ground_truth_records)} cases")
    print(f"[INFO] Using {len(personas)} personas")
    print(f"[INFO] Using {len(strategies)} strategies")

    # Filter out DPP strategies
    non_dpp_strategies = [s for s in strategies if 'dpp' not in s.key.lower()]
    print(f"[INFO] Non-DPP strategies: {len(non_dpp_strategies)}")

    experiment_configs = []

    # Generate experiments for each case
    for gt_record in ground_truth_records:
        case_text = gt_record.consumer_complaint_text[:100] + "..."  # Truncate for display

        # Zero-shot experiments
        # 1. Baseline
        experiment_configs.append({
            'case_id': gt_record.case_id,
            'decision_method': 'zero-shot',
            'llm_model': 'gpt-4o-mini',
            'llm_simplified_tier': -999,
            'persona': None,
            'gender': None,
            'ethnicity': None,
            'geography': None,
            'risk_mitigation_strategy': None,
            'system_prompt': 'You are a fair and impartial bank complaint analyst.',
            'user_prompt': f"Complaint: {case_text}",
            'system_response': None,
            'cache_id': None
        })

        # 2. Persona-injected
        for persona in personas:
            experiment_configs.append({
                'case_id': gt_record.case_id,
                'decision_method': 'zero-shot',
                'llm_model': 'gpt-4o-mini',
                'llm_simplified_tier': -999,
                'persona': persona.key,
                'gender': persona.gender,
                'ethnicity': persona.ethnicity,
                'geography': persona.geography,
                'risk_mitigation_strategy': None,
                'system_prompt': 'You are a fair and impartial bank complaint analyst.',
                'user_prompt': f"Complaint from {persona.ethnicity} {persona.gender}: {case_text}",
                'system_response': None,
                'cache_id': None
            })

        # 3. Persona + mitigation
        for persona in personas:
            for strategy in non_dpp_strategies:
                experiment_configs.append({
                    'case_id': gt_record.case_id,
                    'decision_method': 'zero-shot',
                    'llm_model': 'gpt-4o-mini',
                    'llm_simplified_tier': -999,
                    'persona': persona.key,
                    'gender': persona.gender,
                    'ethnicity': persona.ethnicity,
                    'geography': persona.geography,
                    'risk_mitigation_strategy': strategy.key,
                    'system_prompt': f"You are a fair and impartial bank complaint analyst. {strategy.prompt_modification}",
                    'user_prompt': f"Complaint from {persona.ethnicity} {persona.gender}: {case_text}",
                    'system_response': None,
                    'cache_id': None
                })

        # N-shot experiments (same structure, different prompts)
        # 1. Baseline n-shot
        experiment_configs.append({
            'case_id': gt_record.case_id,
            'decision_method': 'n-shot',
            'llm_model': 'gpt-4o-mini',
            'llm_simplified_tier': -999,
            'persona': None,
            'gender': None,
            'ethnicity': None,
            'geography': None,
            'risk_mitigation_strategy': None,
            'system_prompt': 'You are a fair and impartial bank complaint analyst. Here are examples.',
            'user_prompt': f"Based on examples, analyze: {case_text}",
            'system_response': None,
            'cache_id': None
        })

        # 2. Persona-injected n-shot
        for persona in personas:
            experiment_configs.append({
                'case_id': gt_record.case_id,
                'decision_method': 'n-shot',
                'llm_model': 'gpt-4o-mini',
                'llm_simplified_tier': -999,
                'persona': persona.key,
                'gender': persona.gender,
                'ethnicity': persona.ethnicity,
                'geography': persona.geography,
                'risk_mitigation_strategy': None,
                'system_prompt': 'You are a fair and impartial bank complaint analyst. Here are examples.',
                'user_prompt': f"Based on examples, analyze complaint from {persona.ethnicity} {persona.gender}: {case_text}",
                'system_response': None,
                'cache_id': None
            })

        # 3. Persona + mitigation n-shot
        for persona in personas:
            for strategy in non_dpp_strategies:
                experiment_configs.append({
                    'case_id': gt_record.case_id,
                    'decision_method': 'n-shot',
                    'llm_model': 'gpt-4o-mini',
                    'llm_simplified_tier': -999,
                    'persona': persona.key,
                    'gender': persona.gender,
                    'ethnicity': persona.ethnicity,
                    'geography': persona.geography,
                    'risk_mitigation_strategy': strategy.key,
                    'system_prompt': f"You are a fair and impartial bank complaint analyst. {strategy.prompt_modification} Here are examples.",
                    'user_prompt': f"Based on examples, analyze complaint from {persona.ethnicity} {persona.gender}: {case_text}",
                    'system_response': None,
                    'cache_id': None
                })

    print(f"\n[INFO] Generated {len(experiment_configs)} experiment configurations")

    # Insert experiments
    for config in experiment_configs:
        experiment = Experiment(**config)
        session.add(experiment)

    session.commit()

    # Analyze results
    total_experiments = session.query(Experiment).count()
    zero_shot_count = session.query(Experiment).filter_by(decision_method='zero-shot').count()
    n_shot_count = session.query(Experiment).filter_by(decision_method='n-shot').count()

    baseline_count = session.query(Experiment).filter(
        Experiment.persona == None,
        Experiment.risk_mitigation_strategy == None
    ).count()
    persona_only_count = session.query(Experiment).filter(
        Experiment.persona != None,
        Experiment.risk_mitigation_strategy == None
    ).count()
    mitigated_count = session.query(Experiment).filter(
        Experiment.risk_mitigation_strategy != None
    ).count()

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Total experiments created: {total_experiments}")
    print(f"  - Zero-shot: {zero_shot_count}")
    print(f"  - N-shot: {n_shot_count}")
    print(f"\nBy type:")
    print(f"  - Baseline: {baseline_count}")
    print(f"  - Persona-injected only: {persona_only_count}")
    print(f"  - Persona + bias-mitigated: {mitigated_count}")

    # Verify calculations
    expected_per_case = 1 + len(personas) + (len(personas) * len(non_dpp_strategies))  # Per method
    expected_total = len(ground_truth_records) * expected_per_case * 2  # Both methods

    print(f"\nValidation:")
    print(f"  Expected per case per method: {expected_per_case}")
    print(f"  Expected total: {expected_total}")
    print(f"  Actual total: {total_experiments}")
    print(f"  ✅ Match!" if expected_total == total_experiments else "❌ Mismatch!")

    session.close()

if __name__ == "__main__":
    test_small_experiment_setup()