#!/usr/bin/env python3
"""
Ground Truth Remedy Tiers Analysis

This script reads all CFPB cases, categorizes them into remedy tiers based on 
company responses, and shows a count table for each remedy tier.

The remedy tier system based on actual CFPB data categories:
- Tier 0: No relief (Closed, Closed without relief)
- Tier 1: Explanation only (Closed with explanation)
- Tier 2: Non-monetary relief (Closed with non-monetary relief)
- Tier 3: Monetary relief (Closed with monetary relief)
- Ambiguous: Closed with relief (unclear if monetary or non-monetary)
- Undecided/Still Open: In progress, Untimely response

Note: The CFPB public dataset does not provide monetary amounts, only categorical indicators.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter


def load_cfpb_data(cfpb_path: str = "cfpb_downloads/complaints.csv") -> pd.DataFrame:
    """Load CFPB complaints data"""
    print(f"Loading CFPB data from {cfpb_path}...")
    
    try:
        # Load with low_memory=False to handle mixed types
        df = pd.read_csv(cfpb_path, low_memory=False, dtype=str)
        print(f"Loaded {len(df):,} total complaints")
        return df
    except FileNotFoundError:
        print(f"Error: CFPB data file not found at {cfpb_path}")
        raise


def categorize_remedy_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """Categorize cases into remedy tiers based on company responses"""
    
    # Define remedy tier mapping based on actual CFPB data categories
    response_to_tier = {
        # Tier 0: No relief
        'Closed': 0,
        'Closed without relief': 0,
        
        # Tier 1: Explanation only
        'Closed with explanation': 1,
        
        # Tier 2: Non-monetary relief
        'Closed with non-monetary relief': 2,
        
        # Tier 3: Monetary relief
        'Closed with monetary relief': 3,
        
        # Ambiguous category (unclear if monetary or non-monetary)
        'Closed with relief': 'Ambiguous',
        
        # Undecided/Still Open cases
        'In progress': 'Undecided/Still Open',
        'Untimely response': 'Undecided/Still Open'
    }
    
    # Map company responses to remedy tiers
    df['remedy_tier'] = df['Company response to consumer'].map(response_to_tier)
    
    # Handle any unmapped responses
    unmapped_count = df['remedy_tier'].isna().sum()
    if unmapped_count > 0:
        print(f"Warning: {unmapped_count} cases have unmapped company responses")
        print("Unmapped responses:")
        unmapped_responses = df[df['remedy_tier'].isna()]['Company response to consumer'].value_counts()
        print(unmapped_responses)
        # Assign unmapped cases to 'Undecided/Still Open'
        df['remedy_tier'] = df['remedy_tier'].fillna('Undecided/Still Open')
    
    return df


def create_tier_labels() -> dict:
    """Create human-readable labels for remedy tiers"""
    return {
        0: "Tier 0: No relief",
        1: "Tier 1: Explanation only", 
        2: "Tier 2: Non-monetary relief",
        3: "Tier 3: Monetary relief",
        'Ambiguous': "Ambiguous: Closed with relief",
        'Undecided/Still Open': "Undecided/Still Open"
    }


def generate_count_table(df: pd.DataFrame) -> None:
    """Generate and display count table for remedy tiers"""
    
    tier_labels = create_tier_labels()
    
    # Count cases by remedy tier
    tier_counts = df['remedy_tier'].value_counts()
    
    # Calculate percentages
    total_cases = len(df)
    tier_percentages = (tier_counts / total_cases * 100).round(2)
    
    print("\n" + "="*80)
    print("CFPB COMPLAINTS BY REMEDY TIER")
    print("="*80)
    print(f"Total Cases: {total_cases:,}")
    print()
    
    # Create formatted table
    print(f"{'Remedy Tier':<35} {'Count':<12} {'Percentage':<12}")
    print("-" * 60)
    
    # Sort tiers in logical order
    tier_order = [0, 1, 2, 3, 'Ambiguous', 'Undecided/Still Open']
    
    for tier in tier_order:
        if tier in tier_counts.index:
            count = tier_counts[tier]
            percentage = tier_percentages[tier]
            label = tier_labels[tier]
            print(f"{label:<35} {count:>10,} {percentage:>10.2f}%")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Count resolved vs undecided cases
    resolved_tiers = [0, 1, 2, 3]  # Clear categories only
    resolved_count = sum(tier_counts.get(tier, 0) for tier in resolved_tiers)
    ambiguous_count = tier_counts.get('Ambiguous', 0)
    undecided_count = tier_counts.get('Undecided/Still Open', 0)
    
    print(f"Resolved Cases (clear categories): {resolved_count:,} ({resolved_count/total_cases*100:.2f}%)")
    print(f"Ambiguous Cases: {ambiguous_count:,} ({ambiguous_count/total_cases*100:.2f}%)")
    print(f"Undecided/Still Open: {undecided_count:,} ({undecided_count/total_cases*100:.2f}%)")
    
    # Calculate mean tier for resolved cases (excluding ambiguous and undecided)
    resolved_df = df[df['remedy_tier'].isin(resolved_tiers)]
    if len(resolved_df) > 0:
        mean_tier = resolved_df['remedy_tier'].astype(float).mean()
        print(f"Mean Remedy Tier (resolved cases): {mean_tier:.2f}")
    
    # Show distribution of monetary vs non-monetary remedies
    monetary_tiers = [3]  # Monetary relief only
    non_monetary_tiers = [0, 1, 2]  # No relief, explanation, non-monetary relief
    
    monetary_count = sum(tier_counts.get(tier, 0) for tier in monetary_tiers)
    non_monetary_count = sum(tier_counts.get(tier, 0) for tier in non_monetary_tiers)
    
    print(f"\nMonetary Relief (Tier 3): {monetary_count:,} ({monetary_count/resolved_count*100:.2f}% of resolved)")
    print(f"Non-Monetary Relief (Tiers 0-2): {non_monetary_count:,} ({non_monetary_count/resolved_count*100:.2f}% of resolved)")


def main():
    """Main function to run the ground truth tiers analysis"""
    print("CFPB Ground Truth Remedy Tiers Analysis")
    print("="*50)
    
    # Load CFPB data
    df = load_cfpb_data()
    
    # Categorize into remedy tiers
    df = categorize_remedy_tiers(df)
    
    # Generate count table
    generate_count_table(df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
