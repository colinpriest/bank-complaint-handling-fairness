#!/usr/bin/env python3
"""
Fix the fundamental experimental design issue: ensure EVERY complaint has ALL personas.
This will enable proper Complaint Categories analysis across all demographic groups.
"""

import json
import numpy as np
from pathlib import Path
from complaints_llm_fairness_harness import DEMOGRAPHIC_PERSONAS, generate_realistic_narrative

def convert_numpy(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    else:
        return obj

def fix_all_personas_per_complaint():
    """Generate ALL personas for ALL complaints to enable proper Complaint Categories analysis"""
    
    print("[FIX] Loading existing experimental data...")
    
    # Load existing data
    runs_file = Path("out/runs.jsonl")
    if not runs_file.exists():
        print("[ERROR] No experimental data found at out/runs.jsonl")
        return
    
    existing_data = []
    with open(runs_file, 'r') as f:
        for line in f:
            if line.strip():
                existing_data.append(json.loads(line))
    
    print(f"[FIX] Loaded {len(existing_data)} existing records")
    
    # Get all unique complaints (case_ids) from baseline records
    baseline_records = {}
    for record in existing_data:
        if record.get('variant') == 'NC':  # Baseline records
            case_id = record.get('case_id')
            if case_id:
                baseline_records[case_id] = record
    
    print(f"[FIX] Found {len(baseline_records)} baseline complaints")
    
    # Get all personas we need to generate
    all_personas = set(DEMOGRAPHIC_PERSONAS.keys())
    print(f"[FIX] Need to generate {len(all_personas)} personas: {sorted(all_personas)}")
    
    # Check what personas we already have for each complaint
    complaint_personas = {}
    for record in existing_data:
        case_id = record.get('case_id')
        group_label = record.get('group_label', '')
        variant = record.get('variant', '')
        
        if case_id and group_label and variant != 'NC':  # Skip baseline records
            if case_id not in complaint_personas:
                complaint_personas[case_id] = set()
            complaint_personas[case_id].add(group_label)
    
    # Generate missing persona records for ALL complaints
    new_records = []
    rng = np.random.RandomState(42)
    
    print("[FIX] Generating ALL missing persona records for ALL complaints...")
    
    for case_id, baseline_record in baseline_records.items():
        existing_personas = complaint_personas.get(case_id, set())
        missing_personas = all_personas - existing_personas
        
        print(f"[FIX] Complaint {case_id}: has {len(existing_personas)} personas, missing {len(missing_personas)} personas")
        
        for persona_key in missing_personas:
            if persona_key not in DEMOGRAPHIC_PERSONAS:
                print(f"[WARNING] Persona {persona_key} not found in DEMOGRAPHIC_PERSONAS")
                continue
            
            persona = DEMOGRAPHIC_PERSONAS[persona_key]
            
            # Create persona-specific record
            name = rng.choice(persona["names"])
            location_data = persona["locations"][rng.randint(0, len(persona["locations"]) - 1)]
            location, zip_code = location_data
            company = rng.choice(persona["companies"])
            product = rng.choice(persona["products"])
            style = persona["language_style"]
            
            # Generate realistic narrative
            base_narrative = baseline_record.get('narrative', '')
            persona_narrative = generate_realistic_narrative(
                base_narrative, style, name, location, product
            )
            
            # Create new record with same case_id as baseline
            new_record = baseline_record.copy()
            new_record['case_id'] = case_id  # Ensure same case_id
            new_record['group_label'] = persona_key
            new_record['group_text'] = f"{name} from {location}"
            new_record['variant'] = 'G'  # Standard persona variant
            new_record['product'] = product
            new_record['company'] = company
            new_record['state'] = location.split(", ")[1] if ", " in location else "CA"
            new_record['narrative'] = persona_narrative
            # DO NOT set remedy_tier - this will be filled by LLM evaluation
            # new_record['remedy_tier'] = rng.randint(0, 5)  # REMOVED - no synthetic data
            
            new_records.append(new_record)
    
    print(f"[FIX] Generated {len(new_records)} new persona records")
    
    if new_records:
        # Append new records to runs.jsonl
        print("[FIX] Appending new records to out/runs.jsonl...")
        with open(runs_file, 'a') as f:
            for record in new_records:
                safe_record = convert_numpy(record)
                f.write(json.dumps(safe_record, ensure_ascii=False) + '\n')
        
        print(f"[FIX] Successfully added {len(new_records)} records")
        print("[FIX] Complete persona coverage fix done!")
        
        # Verify the fix
        print("[FIX] Verifying coverage...")
        verify_coverage()
    else:
        print("[FIX] No new records needed - all complaints already have complete persona coverage!")

def verify_coverage():
    """Verify that all complaints now have all personas"""
    
    # Load updated data
    runs_file = Path("out/runs.jsonl")
    existing_data = []
    with open(runs_file, 'r') as f:
        for line in f:
            if line.strip():
                existing_data.append(json.loads(line))
    
    # Get baseline records
    baseline_records = {}
    for record in existing_data:
        if record.get('variant') == 'NC':
            case_id = record.get('case_id')
            if case_id:
                baseline_records[case_id] = record
    
    # Check persona coverage
    all_personas = set(DEMOGRAPHIC_PERSONAS.keys())
    complaint_personas = {}
    
    for record in existing_data:
        case_id = record.get('case_id')
        group_label = record.get('group_label', '')
        variant = record.get('variant', '')
        
        if case_id and group_label and variant != 'NC':
            if case_id not in complaint_personas:
                complaint_personas[case_id] = set()
            complaint_personas[case_id].add(group_label)
    
    # Report coverage
    complete_coverage = 0
    incomplete_coverage = 0
    
    for case_id in baseline_records.keys():
        existing_personas = complaint_personas.get(case_id, set())
        missing_personas = all_personas - existing_personas
        
        if not missing_personas:
            complete_coverage += 1
        else:
            incomplete_coverage += 1
            print(f"[VERIFY] Complaint {case_id}: still missing {len(missing_personas)} personas")
    
    print(f"[VERIFY] Coverage summary:")
    print(f"[VERIFY] - Complete coverage: {complete_coverage} complaints")
    print(f"[VERIFY] - Incomplete coverage: {incomplete_coverage} complaints")
    print(f"[VERIFY] - Total personas needed: {len(all_personas)}")
    
    if incomplete_coverage == 0:
        print("[VERIFY] ✅ All complaints now have complete persona coverage!")
    else:
        print(f"[VERIFY] ⚠️  {incomplete_coverage} complaints still missing personas")

if __name__ == "__main__":
    fix_all_personas_per_complaint()
