#!/usr/bin/env python3
"""
Intersectional Bias Analysis for Bank Complaint Handling Fairness

This script analyzes how different demographic intersections (combinations of
ethnicity, gender, and geography) affect LLM tier assignments, and compares
actual bias patterns to expected disadvantage based on sociological research.
"""

import os
import sys
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from scipy import stats
from typing import Dict, List, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

class IntersectionalBiasAnalyzer:
    """Analyze intersectional bias in LLM tier assignments"""

    def __init__(self):
        """Initialize database connection and analysis parameters"""
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'fairness_analysis'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }

        # Define expected disadvantage scores based on sociological research
        # Higher score = more expected disadvantage/bias
        self.disadvantage_scores = {
            # Ethnicity scores (0-3 scale)
            'ethnicity': {
                'black': 3,      # Historically highest discrimination
                'latino': 2.5,   # High discrimination, especially in financial services
                'asian': 1.5,    # Model minority myth but still face discrimination
                'white': 0       # Reference group, systemic advantages
            },
            # Gender scores (0-2 scale)
            'gender': {
                'female': 2,     # Gender discrimination in financial services
                'male': 0        # Reference group
            },
            # Geography scores (0-3 scale)
            'geography': {
                'urban_poor': 3,     # Highest economic disadvantage
                'rural': 2,          # Limited access to services
                'urban_affluent': 0  # Most advantaged
            }
        }

        # Define interaction penalties (additional disadvantage for specific combinations)
        self.interaction_penalties = {
            ('black', 'female'): 1.5,      # Intersectional discrimination
            ('black', 'male', 'urban_poor'): 2.0,  # Criminalization stereotypes
            ('latino', 'female', 'rural'): 1.0,    # Language/cultural barriers
            ('asian', 'female'): 0.5,      # Model minority pressure
            ('black', 'female', 'urban_poor'): 2.5,  # Triple disadvantage
            ('latino', 'male', 'urban_poor'): 1.5,   # Economic + ethnic stereotypes
        }

    def connect_to_database(self) -> psycopg2.extensions.connection:
        """Establish database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            print(f"[INFO] Connected to database '{self.db_config['database']}'")
            return conn
        except Exception as e:
            print(f"[ERROR] Failed to connect to database: {e}")
            sys.exit(1)

    def fetch_experiment_data(self, conn) -> pd.DataFrame:
        """Fetch all experiment data with baseline comparisons"""
        query = """
        SELECT
            e.experiment_id,
            e.case_id,
            e.decision_method,
            e.persona,
            e.ethnicity,
            e.gender,
            e.geography,
            e.risk_mitigation_strategy,
            e.llm_simplified_tier as persona_tier,
            b.llm_simplified_tier as baseline_tier,
            g.simplified_ground_truth_tier as ground_truth_tier
        FROM experiments e
        JOIN baseline_experiments b ON e.case_id = b.case_id
            AND e.decision_method = b.decision_method
        LEFT JOIN ground_truth g ON e.case_id = g.case_id
        WHERE e.persona IS NOT NULL
          AND e.risk_mitigation_strategy IS NULL
          AND e.llm_simplified_tier >= 0
          AND b.llm_simplified_tier >= 0
        ORDER BY e.case_id, e.persona
        """

        df = pd.read_sql_query(query, conn)
        print(f"[INFO] Loaded {len(df):,} experiment records")

        # Calculate tier differences
        df['tier_diff_from_baseline'] = df['persona_tier'] - df['baseline_tier']
        df['tier_diff_from_truth'] = df['persona_tier'] - df['ground_truth_tier']

        return df

    def calculate_disadvantage_score(self, ethnicity: str, gender: str, geography: str) -> float:
        """Calculate expected disadvantage score for a persona"""
        # Base scores
        score = (self.disadvantage_scores['ethnicity'].get(ethnicity, 0) +
                self.disadvantage_scores['gender'].get(gender, 0) +
                self.disadvantage_scores['geography'].get(geography, 0))

        # Add interaction penalties
        for combo, penalty in self.interaction_penalties.items():
            if len(combo) == 2 and ethnicity in combo and gender in combo:
                score += penalty
            elif len(combo) == 3 and ethnicity in combo and gender in combo and geography in combo:
                score += penalty

        return score

    def analyze_intersectional_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze bias patterns across intersectional groups"""
        results = {}

        # Group by full persona (all intersections)
        persona_groups = df.groupby('persona')

        for persona, group_data in persona_groups:
            # Parse persona components
            parts = persona.split('_')
            ethnicity = parts[0] if len(parts) > 0 else 'unknown'
            gender = parts[1] if len(parts) > 1 else 'unknown'
            geography = '_'.join(parts[2:]) if len(parts) > 2 else 'unknown'

            # Calculate metrics
            results[persona] = {
                'ethnicity': ethnicity,
                'gender': gender,
                'geography': geography,
                'sample_size': len(group_data),
                'mean_tier': group_data['persona_tier'].mean(),
                'baseline_mean_tier': group_data['baseline_tier'].mean(),
                'tier_bias': group_data['tier_diff_from_baseline'].mean(),
                'tier_bias_std': group_data['tier_diff_from_baseline'].std(),
                'downgrade_rate': (group_data['tier_diff_from_baseline'] < 0).mean(),
                'upgrade_rate': (group_data['tier_diff_from_baseline'] > 0).mean(),
                'no_change_rate': (group_data['tier_diff_from_baseline'] == 0).mean(),
                'disadvantage_score': self.calculate_disadvantage_score(ethnicity, gender, geography)
            }

            # Statistical test vs baseline
            if len(group_data) > 5:
                baseline_tiers = group_data['baseline_tier'].values
                persona_tiers = group_data['persona_tier'].values

                # Paired t-test since same cases
                if len(baseline_tiers) == len(persona_tiers):
                    t_stat, p_value = stats.ttest_rel(persona_tiers, baseline_tiers)
                    results[persona]['t_statistic'] = t_stat
                    results[persona]['p_value'] = p_value
                    results[persona]['significant'] = p_value < 0.05
                else:
                    results[persona]['t_statistic'] = np.nan
                    results[persona]['p_value'] = np.nan
                    results[persona]['significant'] = False

        return results

    def analyze_by_demographic_dimension(self, df: pd.DataFrame) -> Dict:
        """Analyze patterns by individual demographic dimensions"""
        dimensions = {}

        # Analyze by ethnicity
        ethnicity_groups = df.groupby('ethnicity')
        dimensions['ethnicity'] = {}
        for ethnicity, group in ethnicity_groups:
            dimensions['ethnicity'][ethnicity] = {
                'mean_tier': group['persona_tier'].mean(),
                'mean_bias': group['tier_diff_from_baseline'].mean(),
                'sample_size': len(group),
                'downgrade_rate': (group['tier_diff_from_baseline'] < 0).mean()
            }

        # Analyze by gender
        gender_groups = df.groupby('gender')
        dimensions['gender'] = {}
        for gender, group in gender_groups:
            dimensions['gender'][gender] = {
                'mean_tier': group['persona_tier'].mean(),
                'mean_bias': group['tier_diff_from_baseline'].mean(),
                'sample_size': len(group),
                'downgrade_rate': (group['tier_diff_from_baseline'] < 0).mean()
            }

        # Analyze by geography
        geography_groups = df.groupby('geography')
        dimensions['geography'] = {}
        for geography, group in geography_groups:
            dimensions['geography'][geography] = {
                'mean_tier': group['persona_tier'].mean(),
                'mean_bias': group['tier_diff_from_baseline'].mean(),
                'sample_size': len(group),
                'downgrade_rate': (group['tier_diff_from_baseline'] < 0).mean()
            }

        return dimensions

    def analyze_interaction_effects(self, df: pd.DataFrame) -> Dict:
        """Analyze two-way and three-way interaction effects"""
        interactions = {}

        # Two-way interactions
        for col1, col2 in [('ethnicity', 'gender'), ('ethnicity', 'geography'), ('gender', 'geography')]:
            interaction_key = f"{col1}_x_{col2}"
            grouped = df.groupby([col1, col2])
            interactions[interaction_key] = {}

            for (val1, val2), group in grouped:
                combo_key = f"{val1}_{val2}"
                interactions[interaction_key][combo_key] = {
                    'mean_tier': group['persona_tier'].mean(),
                    'mean_bias': group['tier_diff_from_baseline'].mean(),
                    'sample_size': len(group),
                    'variance': group['persona_tier'].var()
                }

        # Three-way interaction (full intersectional analysis)
        grouped = df.groupby(['ethnicity', 'gender', 'geography'])
        interactions['full_intersection'] = {}

        for (ethnicity, gender, geography), group in grouped:
            combo_key = f"{ethnicity}_{gender}_{geography}"
            interactions['full_intersection'][combo_key] = {
                'mean_tier': group['persona_tier'].mean(),
                'mean_bias': group['tier_diff_from_baseline'].mean(),
                'sample_size': len(group),
                'variance': group['persona_tier'].var()
            }

        return interactions

    def rank_personas_by_disadvantage(self, results: Dict) -> pd.DataFrame:
        """Rank personas by expected vs actual disadvantage"""
        rankings = []

        for persona, metrics in results.items():
            rankings.append({
                'persona': persona,
                'expected_disadvantage_score': metrics['disadvantage_score'],
                'actual_mean_tier': metrics['mean_tier'],
                'actual_tier_bias': metrics['tier_bias'],
                'downgrade_rate': metrics['downgrade_rate'],
                'upgrade_rate': metrics['upgrade_rate'],
                'significant_bias': metrics.get('significant', False),
                'p_value': metrics.get('p_value', 1.0)
            })

        df_rankings = pd.DataFrame(rankings)

        # Sort by expected disadvantage
        df_rankings = df_rankings.sort_values('expected_disadvantage_score', ascending=False)
        df_rankings['expected_rank'] = range(1, len(df_rankings) + 1)

        # Sort by actual bias (lower tier = more disadvantage in this context)
        df_rankings = df_rankings.sort_values('actual_mean_tier', ascending=True)
        df_rankings['actual_rank'] = range(1, len(df_rankings) + 1)

        # Calculate rank correlation
        correlation, p_value = stats.spearmanr(df_rankings['expected_disadvantage_score'],
                                              -df_rankings['actual_mean_tier'])  # Negative because lower tier = worse

        print(f"\n[CORRELATION] Spearman correlation between expected and actual disadvantage: {correlation:.3f} (p={p_value:.4f})")

        return df_rankings.sort_values('expected_disadvantage_score', ascending=False)

    def plot_intersectional_bias(self, df_rankings: pd.DataFrame):
        """Create visualizations of intersectional bias patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Expected vs Actual Disadvantage
        ax1 = axes[0, 0]
        ax1.scatter(df_rankings['expected_disadvantage_score'],
                   -df_rankings['actual_mean_tier'],  # Negative so higher = more disadvantage
                   alpha=0.6, s=100)

        # Add labels for extreme cases
        for idx, row in df_rankings.head(3).iterrows():
            ax1.annotate(row['persona'].split('_')[0][:3] + '_' + row['persona'].split('_')[1][:1],
                        (row['expected_disadvantage_score'], -row['actual_mean_tier']),
                        fontsize=8)

        ax1.set_xlabel('Expected Disadvantage Score')
        ax1.set_ylabel('Actual Disadvantage (Negative Mean Tier)')
        ax1.set_title('Expected vs Actual Disadvantage')
        ax1.grid(True, alpha=0.3)

        # Add correlation line
        z = np.polyfit(df_rankings['expected_disadvantage_score'],
                      -df_rankings['actual_mean_tier'], 1)
        p = np.poly1d(z)
        ax1.plot(df_rankings['expected_disadvantage_score'],
                p(df_rankings['expected_disadvantage_score']),
                "r--", alpha=0.5, label=f'Trend line')
        ax1.legend()

        # Plot 2: Downgrade Rates by Persona
        ax2 = axes[0, 1]
        top_personas = df_rankings.nlargest(10, 'downgrade_rate')
        ax2.barh(range(len(top_personas)), top_personas['downgrade_rate'])
        ax2.set_yticks(range(len(top_personas)))
        ax2.set_yticklabels([p.replace('_', ' ').title() for p in top_personas['persona']], fontsize=8)
        ax2.set_xlabel('Downgrade Rate (Tier Reduction from Baseline)')
        ax2.set_title('Top 10 Personas by Downgrade Rate')
        ax2.grid(True, alpha=0.3, axis='x')

        # Plot 3: Bias Distribution
        ax3 = axes[1, 0]
        ax3.hist(df_rankings['actual_tier_bias'], bins=20, edgecolor='black', alpha=0.7)
        ax3.axvline(x=0, color='red', linestyle='--', label='No Bias')
        ax3.set_xlabel('Tier Bias (Persona - Baseline)')
        ax3.set_ylabel('Number of Personas')
        ax3.set_title('Distribution of Tier Bias Across Personas')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Ranking Comparison
        ax4 = axes[1, 1]
        ax4.scatter(df_rankings['expected_rank'], df_rankings['actual_rank'], alpha=0.6, s=100)

        # Add diagonal line for perfect correlation
        max_rank = len(df_rankings)
        ax4.plot([1, max_rank], [1, max_rank], 'r--', alpha=0.5, label='Perfect Correlation')

        # Add labels for outliers (big differences in ranking)
        rank_diff = abs(df_rankings['expected_rank'] - df_rankings['actual_rank'])
        outliers = df_rankings[rank_diff > 10]
        for idx, row in outliers.iterrows():
            ax4.annotate(row['persona'].split('_')[0][:3] + '_' + row['persona'].split('_')[1][:1],
                        (row['expected_rank'], row['actual_rank']),
                        fontsize=8)

        ax4.set_xlabel('Expected Disadvantage Rank')
        ax4.set_ylabel('Actual Disadvantage Rank')
        ax4.set_title('Expected vs Actual Disadvantage Rankings')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('intersectional_bias_analysis.png', dpi=150, bbox_inches='tight')
        print("[INFO] Saved visualization to 'intersectional_bias_analysis.png'")
        plt.show()

    def print_detailed_report(self, df_rankings: pd.DataFrame, dimensions: Dict, interactions: Dict):
        """Print detailed analysis report"""
        print("\n" + "="*80)
        print("INTERSECTIONAL BIAS ANALYSIS REPORT")
        print("="*80)

        # Top disadvantaged personas (expected)
        print("\n[EXPECTED] Top 5 Most Disadvantaged Personas (by sociological research):")
        print("-" * 60)
        for _, row in df_rankings.head(5).iterrows():
            print(f"  {row['persona']:<30} Score: {row['expected_disadvantage_score']:.1f}")

        # Top disadvantaged personas (actual)
        print("\n[ACTUAL] Top 5 Most Disadvantaged Personas (by LLM tier assignment):")
        print("-" * 60)
        actual_sorted = df_rankings.sort_values('actual_mean_tier', ascending=True)
        for _, row in actual_sorted.head(5).iterrows():
            print(f"  {row['persona']:<30} Mean Tier: {row['actual_mean_tier']:.3f}")

        # Personas with significant bias
        print("\n[SIGNIFICANT] Personas with Statistically Significant Bias (p < 0.05):")
        print("-" * 60)
        significant = df_rankings[df_rankings['significant_bias'] == True]
        if len(significant) > 0:
            for _, row in significant.iterrows():
                bias_direction = "disadvantaged" if row['actual_tier_bias'] < 0 else "advantaged"
                print(f"  {row['persona']:<30} Bias: {row['actual_tier_bias']:+.3f} ({bias_direction})")
        else:
            print("  No personas showed statistically significant bias")

        # Dimension-level patterns
        print("\n[DIMENSIONS] Bias Patterns by Individual Demographics:")
        print("-" * 60)

        for dimension, groups in dimensions.items():
            print(f"\n  {dimension.upper()}:")
            sorted_groups = sorted(groups.items(), key=lambda x: x[1]['mean_bias'])
            for group, metrics in sorted_groups:
                print(f"    {group:<20} Bias: {metrics['mean_bias']:+.4f} (n={metrics['sample_size']})")

        # Surprising inversions (expected disadvantage but actual advantage)
        print("\n[INVERSIONS] Personas with Unexpected Bias Direction:")
        print("-" * 60)
        inversions = df_rankings[
            ((df_rankings['expected_disadvantage_score'] > 5) & (df_rankings['actual_tier_bias'] > 0.05)) |
            ((df_rankings['expected_disadvantage_score'] < 3) & (df_rankings['actual_tier_bias'] < -0.05))
        ]

        if len(inversions) > 0:
            for _, row in inversions.iterrows():
                expected = "disadvantaged" if row['expected_disadvantage_score'] > 5 else "advantaged"
                actual = "advantaged" if row['actual_tier_bias'] > 0 else "disadvantaged"
                print(f"  {row['persona']:<30} Expected: {expected}, Actual: {actual}")
        else:
            print("  No major inversions detected")

        # Summary statistics
        print("\n[SUMMARY] Overall Intersectional Bias Patterns:")
        print("-" * 60)
        print(f"  Total unique personas analyzed: {len(df_rankings)}")
        print(f"  Personas with tier downgrades: {(df_rankings['downgrade_rate'] > 0.5).sum()}")
        print(f"  Personas with tier upgrades: {(df_rankings['upgrade_rate'] > 0.5).sum()}")
        print(f"  Mean absolute bias: {abs(df_rankings['actual_tier_bias']).mean():.4f}")
        print(f"  Correlation (expected vs actual): See visualization")

    def run_analysis(self):
        """Run complete intersectional bias analysis"""
        # Connect to database
        conn = self.connect_to_database()

        try:
            # Fetch experiment data
            df = self.fetch_experiment_data(conn)

            # Analyze intersectional patterns
            print("\n[ANALYSIS] Computing intersectional bias patterns...")
            results = self.analyze_intersectional_patterns(df)

            # Analyze by individual dimensions
            dimensions = self.analyze_by_demographic_dimension(df)

            # Analyze interaction effects
            interactions = self.analyze_interaction_effects(df)

            # Rank personas
            df_rankings = self.rank_personas_by_disadvantage(results)

            # Save rankings to CSV
            df_rankings.to_csv('persona_disadvantage_rankings.csv', index=False)
            print("[INFO] Saved rankings to 'persona_disadvantage_rankings.csv'")

            # Print report
            self.print_detailed_report(df_rankings, dimensions, interactions)

            # Create visualizations
            self.plot_intersectional_bias(df_rankings)

            # Export detailed results
            detailed_results = pd.DataFrame.from_dict(results, orient='index')
            detailed_results.to_csv('intersectional_bias_detailed.csv')
            print("[INFO] Saved detailed results to 'intersectional_bias_detailed.csv'")

        finally:
            conn.close()
            print("\n[INFO] Analysis complete!")


def main():
    """Main execution function"""
    analyzer = IntersectionalBiasAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()