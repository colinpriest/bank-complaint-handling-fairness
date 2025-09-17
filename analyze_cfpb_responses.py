#!/usr/bin/env python3
"""
Analyze raw CFPB company response values to see all unique categories
"""

import pandas as pd
from collections import Counter

def analyze_company_responses(cfpb_path: str = "cfpb_downloads/complaints.csv"):
    """Load CFPB data and analyze company response categories"""

    print("Loading CFPB data...")
    df = pd.read_csv(cfpb_path, low_memory=False, dtype=str)
    print(f"Total complaints loaded: {len(df):,}\n")

    # Get all unique company responses
    company_responses = df['Company response to consumer'].value_counts()

    print("="*80)
    print("ALL UNIQUE CFPB COMPANY RESPONSE VALUES")
    print("="*80)
    print(f"{'Company Response':<50} {'Count':<15} {'Percentage':<10}")
    print("-"*80)

    total = len(df)
    for response, count in company_responses.items():
        percentage = (count / total) * 100
        print(f"{str(response):<50} {count:<15,} {percentage:.2f}%")

    print("-"*80)
    print(f"Total unique response types: {len(company_responses)}")

    # Check for null/missing values
    missing = df['Company response to consumer'].isna().sum()
    if missing > 0:
        print(f"\nMissing/NaN values: {missing:,} ({missing/total*100:.2f}%)")

    # Also check with consumer narratives
    print("\n" + "="*80)
    print("RESPONSES FOR COMPLAINTS WITH NARRATIVES")
    print("="*80)

    df_with_narrative = df[df['Consumer complaint narrative'].notna()]
    print(f"Complaints with narratives: {len(df_with_narrative):,}\n")

    narrative_responses = df_with_narrative['Company response to consumer'].value_counts()

    print(f"{'Company Response':<50} {'Count':<15} {'Percentage':<10}")
    print("-"*80)

    total_with_narrative = len(df_with_narrative)
    for response, count in narrative_responses.items():
        percentage = (count / total_with_narrative) * 100
        print(f"{str(response):<50} {count:<15,} {percentage:.2f}%")

    print("-"*80)
    print(f"Total unique response types (with narratives): {len(narrative_responses)}")

    missing_narrative = df_with_narrative['Company response to consumer'].isna().sum()
    if missing_narrative > 0:
        print(f"\nMissing/NaN values (with narratives): {missing_narrative:,} ({missing_narrative/total_with_narrative*100:.2f}%)")

if __name__ == "__main__":
    analyze_company_responses()