#!/usr/bin/env python3
"""
Test ground truth validation on our LLM fairness data
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

def test_ground_truth_validation():
    """Test ground truth validation analysis"""
    
    print("Loading LLM results data...")
    runs_file = Path("advanced_results/enhanced_runs.jsonl")
    
    if not runs_file.exists():
        print("Enhanced runs file not found!")
        return
    
    # Load sample of LLM data
    runs = []
    with open(runs_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < 2000:  # Just first 2000 for testing
                runs.append(json.loads(line))
            else:
                break
    
    df = pd.DataFrame(runs)
    print(f"Loaded {len(df)} LLM results")
    print(f"Available columns: {list(df.columns)}")
    
    # Load CFPB data sample
    print("\nLoading CFPB ground truth data...")
    cfpb_file = "cfpb_downloads/complaints.csv"
    
    try:
        cfpb_df = pd.read_csv(cfpb_file, nrows=1000)  # Small sample
        print(f"Loaded {len(cfpb_df)} CFPB complaints")
        
        # Show CFPB outcome distribution
        print("\nCFPB outcome distribution:")
        outcome_counts = cfpb_df["Company response to consumer"].value_counts()
        for outcome, count in outcome_counts.items():
            print(f"  {outcome}: {count} ({count/len(cfpb_df)*100:.1f}%)")
        
        # Create outcome to tier mapping
        outcome_to_tier = {
            "Closed with monetary relief": 1.0,        # Highest remedy
            "Closed with non-monetary relief": 2.0,    # Middle remedy
            "Closed with explanation": 3.0,            # Lowest remedy
        }
        
        cfpb_df["ground_truth_tier"] = cfpb_df["Company response to consumer"].map(outcome_to_tier)
        cfpb_complete = cfpb_df.dropna(subset=["ground_truth_tier"])
        
        print(f"\nCFPB resolved cases: {len(cfpb_complete)}")
        print("Ground truth tier distribution:")
        tier_counts = cfpb_complete["ground_truth_tier"].value_counts().sort_index()
        for tier, count in tier_counts.items():
            print(f"  Tier {tier}: {count} ({count/len(cfpb_complete)*100:.1f}%)")
        
        # Compare with LLM predictions
        print("\nLLM prediction distribution:")
        llm_tier_counts = df["remedy_tier"].value_counts().sort_index()
        for tier, count in llm_tier_counts.items():
            print(f"  Tier {tier}: {count} ({count/len(df)*100:.1f}%)")
        
        # Statistical comparison
        llm_mean_tier = df["remedy_tier"].mean()
        cfpb_mean_tier = cfpb_complete["ground_truth_tier"].mean()
        
        print(f"\nMean comparison:")
        print(f"  LLM mean tier: {llm_mean_tier:.2f}")
        print(f"  CFPB mean tier: {cfpb_mean_tier:.2f}")
        print(f"  Difference: {llm_mean_tier - cfpb_mean_tier:.2f}")
        
        # Check if we can match by case_id
        if "case_id" in df.columns:
            print(f"\nSample case_ids from LLM data:")
            sample_case_ids = df["case_id"].head(5).tolist()
            for case_id in sample_case_ids:
                print(f"  {case_id}")
            
            print(f"\nSample CFPB Complaint IDs:")
            sample_cfpb_ids = cfpb_df["Complaint ID"].head(5).tolist()
            for complaint_id in sample_cfpb_ids:
                print(f"  {complaint_id}")
        
        # Try to find actual matches
        matches = 0
        if "case_id" in df.columns:
            for case_id in df["case_id"].head(100):  # Check first 100
                if str(case_id) in cfpb_df["Complaint ID"].astype(str).values:
                    matches += 1
            
            print(f"\nDirect ID matches found: {matches}/100 checked")
        
        # Analysis by demographic groups
        print(f"\nLLM bias by demographic group:")
        group_bias = df.groupby("group_label")["remedy_tier"].agg(['mean', 'count']).round(3)
        for group, stats in group_bias.iterrows():
            print(f"  {group}: {stats['mean']} (n={stats['count']})")
        
    except FileNotFoundError:
        print(f"CFPB data file not found: {cfpb_file}")
    except Exception as e:
        print(f"Error loading CFPB data: {e}")

if __name__ == "__main__":
    test_ground_truth_validation()