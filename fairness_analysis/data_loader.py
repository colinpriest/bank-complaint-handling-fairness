"""
Data loading and preprocessing functionality
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np


class DataLoader:
    """Handles loading and preprocessing of CFPB data"""
    
    def __init__(self):
        self.cache_dir = Path("data_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    def load_expanded_cfpb_data(self, sample_size: int = 1000, 
                              cfpb_path: Optional[str] = None) -> pd.DataFrame:
        """Load expanded CFPB data for comprehensive analysis"""
        
        if cfpb_path is None:
            cfpb_path = "cfpb_downloads/complaints.csv"
            
        print(f"[DATA] Loading CFPB data from {cfpb_path}")
        
        try:
            # Load the data with proper dtype handling to avoid warnings
            df = pd.read_csv(cfpb_path, low_memory=False, dtype=str)
            print(f"[DATA] Loaded {len(df):,} total complaints")
            
            # Filter for relevant columns
            required_columns = [
                'Date received', 'Product', 'Sub-product', 'Issue', 'Sub-issue',
                'Consumer complaint narrative', 'Company response to consumer',
                'Timely response?', 'Consumer disputed?'
            ]
            
            # Check which columns exist
            available_columns = [col for col in required_columns if col in df.columns]
            df = df[available_columns]
            
            # Filter for complaints with narratives
            if 'Consumer complaint narrative' in df.columns:
                df = df.dropna(subset=['Consumer complaint narrative'])
                df = df[df['Consumer complaint narrative'].str.len() > 50]  # Meaningful length
                
            print(f"[DATA] Filtered to {len(df):,} complaints with narratives")
            
            # Sample the requested size
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
                print(f"[DATA] Sampled {sample_size:,} complaints")
                
            # Add severity classification
            df = self._classify_severity(df)
            
            # Note: Ground truth tiers should come from real experimental data
            # df = self._add_ground_truth_tiers(df)  # Removed - use real data instead
            
            return df
            
        except FileNotFoundError:
            print(f"[ERROR] CFPB data file not found at {cfpb_path}")
            raise FileNotFoundError(f"CFPB data file not found at {cfpb_path}. Please ensure the data file exists.")
            
    def _classify_severity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify complaint severity based on content"""
        def classify_severity(narrative: str, issue: str = "") -> str:
            narrative_lower = str(narrative).lower()
            issue_lower = str(issue).lower()
            
            # High severity keywords
            high_severity = ['fraud', 'identity theft', 'unauthorized', 'stolen', 'foreclosure', 'eviction']
            if any(keyword in narrative_lower for keyword in high_severity):
                return 'high'
                
            # Medium severity keywords  
            medium_severity = ['dispute', 'error', 'incorrect', 'wrong', 'mistake', 'problem']
            if any(keyword in narrative_lower for keyword in medium_severity):
                return 'medium'
                
            return 'low'
        
        df['severity'] = df.apply(
            lambda row: classify_severity(
                row.get('Consumer complaint narrative', ''),
                row.get('Issue', '')
            ), axis=1
        )
        
        return df
        
        

