#!/usr/bin/env python3
"""
Test the geographic extraction to see what data is returned
"""

from extract_geographic_bias_data import extract_geographic_bias_data

def main():
    print("Testing geographic bias data extraction...")
    
    data = extract_geographic_bias_data()
    
    if 'error' in data:
        print(f"Error: {data['error']}")
        return
    
    print("\nZero-Shot Mean Tier Data:")
    for geo, stats in data.get('zero_shot_mean_tier', {}).items():
        print(f"  {geo}: Mean={stats['mean_tier']:.3f}, Count={stats['count']}, StdDev={stats['std_dev']:.3f}")
    
    print("\nN-Shot Mean Tier Data:")
    for geo, stats in data.get('n_shot_mean_tier', {}).items():
        print(f"  {geo}: Mean={stats['mean_tier']:.3f}, Count={stats['count']}, StdDev={stats['std_dev']:.3f}")
    
    print("\nTier Bias Summary:")
    for geo, methods in data.get('tier_bias_summary', {}).items():
        print(f"  {geo}:")
        for method, stats in methods.items():
            print(f"    {method}: Mean={stats['mean_tier']:.3f}, Count={stats['count']}")

if __name__ == "__main__":
    main()
