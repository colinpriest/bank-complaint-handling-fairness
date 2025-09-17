#!/usr/bin/env python3
"""
Company Public Response Analysis

This script reads all CFPB cases and analyzes the "Company public response" field,
showing the count of cases for each unique value in that field.
"""

import pandas as pd
import numpy as np
from pathlib import Path


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


def analyze_company_public_responses(df: pd.DataFrame) -> None:
    """Analyze and display company public response categories"""
    
    print("\n" + "="*100)
    print("CFPB COMPANY PUBLIC RESPONSE ANALYSIS")
    print("="*100)
    print(f"Total Cases: {len(df):,}")
    print()
    
    # Get value counts for Company public response field
    response_counts = df['Company public response'].value_counts(dropna=False)
    
    # Calculate percentages
    total_cases = len(df)
    response_percentages = (response_counts / total_cases * 100).round(2)
    
    # Create formatted table
    print(f"{'Company Public Response':<80} {'Count':<12} {'Percentage':<12}")
    print("-" * 105)
    
    for response, count in response_counts.items():
        percentage = response_percentages[response]
        
        # Handle NaN values
        if pd.isna(response):
            response_display = "No public response (NaN)"
        else:
            # Truncate very long responses for display
            response_display = str(response)
            if len(response_display) > 75:
                response_display = response_display[:72] + "..."
        
        print(f"{response_display:<80} {count:>10,} {percentage:>10.2f}%")
    
    # Summary statistics
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    
    # Count cases with and without public responses
    has_public_response = df['Company public response'].notna().sum()
    no_public_response = df['Company public response'].isna().sum()
    
    print(f"Cases with public response: {has_public_response:,} ({has_public_response/total_cases*100:.2f}%)")
    print(f"Cases without public response: {no_public_response:,} ({no_public_response/total_cases*100:.2f}%)")
    
    # Analyze the most common responses
    print(f"\nMost common public responses:")
    print("-" * 50)
    
    # Get top 5 responses (excluding NaN)
    top_responses = response_counts.head(5)
    for i, (response, count) in enumerate(top_responses.items(), 1):
        if pd.isna(response):
            continue
        percentage = response_percentages[response]
        print(f"{i}. {str(response)[:60]}{'...' if len(str(response)) > 60 else ''}")
        print(f"   Count: {count:,} ({percentage:.2f}%)")
        print()
    
    # Categorize responses by type
    print("Response Categories:")
    print("-" * 50)
    
    # Define response categories
    response_categories = {
        'No Response': ['No public response (NaN)', 'Company chooses not to provide a public response'],
        'Appropriate Action': ['Company believes it acted appropriately as authorized by contract or law'],
        'Disputes Facts': ['Company disputes the facts presented in the complaint'],
        'Misunderstanding': ['Company believes the complaint is the result of a misunderstanding'],
        'Third Party': ['Company believes complaint caused principally by actions of third party outside the control or direction of the company'],
        'Isolated Error': ['Company believes the complaint is the result of an isolated error'],
        'Opportunity to Answer': ['Company believes the complaint provided an opportunity to answer consumer\'s questions'],
        'Can\'t Verify': ['Company can\'t verify or dispute the facts in the complaint'],
        'Opportunity for Improvement': ['Company believes complaint represents an opportunity for improvement to better serve consumers'],
        'Discontinued Policy': ['Company believes complaint relates to a discontinued policy or procedure'],
        'Responded but No Public Response': ['Company has responded to the consumer and the CFPB and chooses not to provide a public response']
    }
    
    # Count by category
    category_counts = {}
    for category, keywords in response_categories.items():
        count = 0
        for keyword in keywords:
            if keyword == 'No public response (NaN)':
                count += df['Company public response'].isna().sum()
            else:
                count += (df['Company public response'] == keyword).sum()
        category_counts[category] = count
    
    # Display category summary
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            percentage = count / total_cases * 100
            print(f"{category}: {count:,} ({percentage:.2f}%)")


def main():
    """Main function to run the company public response analysis"""
    print("CFPB Company Public Response Analysis")
    print("="*50)
    
    # Load CFPB data
    df = load_cfpb_data()
    
    # Analyze company public responses
    analyze_company_public_responses(df)
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()

