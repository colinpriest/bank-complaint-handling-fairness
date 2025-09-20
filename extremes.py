#!/usr/bin/env python3
"""
Extreme Tier Analysis for Bank Complaint Handling Fairness

This script analyzes which personas are most likely to receive extreme tier
assignments (Tier 0: No Action or Tier 2: Monetary Action), revealing potential
biases in how LLMs handle edge cases for different demographic groups.
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
from matplotlib.patches import Rectangle
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

class ExtremeTierAnalyzer:
    """Analyze extreme tier assignments (0 and 2) across personas"""

    def __init__(self):
        """Initialize database connection and analysis parameters"""
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'fairness_analysis'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }

        # Define tier meanings for clarity
        self.tier_meanings = {
            0: "No Action (complaint dismissed/no bank error)",
            1: "Non-Monetary Action (process fix/data correction)",
            2: "Monetary Action (refund/compensation)"
        }

        # Color scheme for visualizations
        self.colors = {
            'tier_0': '#FF6B6B',  # Red for dismissal
            'tier_1': '#95E77E',  # Green for non-monetary
            'tier_2': '#4ECDC4',  # Teal for monetary
            'baseline': '#FFD93D', # Yellow for baseline
            'ground_truth': '#6C5CE7' # Purple for ground truth
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

    def fetch_extreme_tier_data(self, conn) -> pd.DataFrame:
        """Fetch data focusing on extreme tier assignments"""
        query = """
        WITH tier_data AS (
            SELECT
                e.experiment_id,
                e.case_id,
                e.decision_method,
                e.persona,
                e.ethnicity,
                e.gender,
                e.geography,
                e.llm_simplified_tier as persona_tier,
                b.llm_simplified_tier as baseline_tier,
                g.simplified_ground_truth_tier as ground_truth_tier,
                -- Flag extreme tiers
                CASE WHEN e.llm_simplified_tier = 0 THEN 1 ELSE 0 END as is_tier_0,
                CASE WHEN e.llm_simplified_tier = 2 THEN 1 ELSE 0 END as is_tier_2,
                CASE WHEN b.llm_simplified_tier = 0 THEN 1 ELSE 0 END as baseline_is_tier_0,
                CASE WHEN b.llm_simplified_tier = 2 THEN 1 ELSE 0 END as baseline_is_tier_2,
                CASE WHEN g.simplified_ground_truth_tier = 0 THEN 1 ELSE 0 END as truth_is_tier_0,
                CASE WHEN g.simplified_ground_truth_tier = 2 THEN 1 ELSE 0 END as truth_is_tier_2
            FROM experiments e
            JOIN baseline_experiments b ON e.case_id = b.case_id
                AND e.decision_method = b.decision_method
            LEFT JOIN ground_truth g ON e.case_id = g.case_id
            WHERE e.persona IS NOT NULL
              AND e.risk_mitigation_strategy IS NULL
              AND e.llm_simplified_tier >= 0
              AND b.llm_simplified_tier >= 0
        )
        SELECT * FROM tier_data
        ORDER BY case_id, persona
        """

        df = pd.read_sql_query(query, conn)
        print(f"[INFO] Loaded {len(df):,} experiment records")

        # Calculate transition patterns
        df['shifted_to_0'] = ((df['baseline_tier'] != 0) & (df['persona_tier'] == 0)).astype(int)
        df['shifted_from_0'] = ((df['baseline_tier'] == 0) & (df['persona_tier'] != 0)).astype(int)
        df['shifted_to_2'] = ((df['baseline_tier'] != 2) & (df['persona_tier'] == 2)).astype(int)
        df['shifted_from_2'] = ((df['baseline_tier'] == 2) & (df['persona_tier'] != 2)).astype(int)

        return df

    def analyze_extreme_tier_propensity(self, df: pd.DataFrame) -> Dict:
        """Analyze propensity for extreme tier assignments by persona"""
        results = {}

        # Group by persona
        for persona in df['persona'].unique():
            if pd.isna(persona):
                continue

            persona_data = df[df['persona'] == persona]

            # Parse persona components
            parts = persona.split('_')
            ethnicity = parts[0] if len(parts) > 0 else 'unknown'
            gender = parts[1] if len(parts) > 1 else 'unknown'
            geography = '_'.join(parts[2:]) if len(parts) > 2 else 'unknown'

            # Calculate metrics
            total_cases = len(persona_data)

            results[persona] = {
                'ethnicity': ethnicity,
                'gender': gender,
                'geography': geography,
                'total_cases': total_cases,

                # Tier 0 metrics (No Action)
                'tier_0_count': persona_data['is_tier_0'].sum(),
                'tier_0_rate': persona_data['is_tier_0'].mean(),
                'baseline_tier_0_rate': persona_data['baseline_is_tier_0'].mean(),
                'tier_0_lift': (persona_data['is_tier_0'].mean() /
                               max(persona_data['baseline_is_tier_0'].mean(), 0.001)),
                'shifted_to_0_count': persona_data['shifted_to_0'].sum(),
                'shifted_to_0_rate': persona_data['shifted_to_0'].mean(),

                # Tier 2 metrics (Monetary Action)
                'tier_2_count': persona_data['is_tier_2'].sum(),
                'tier_2_rate': persona_data['is_tier_2'].mean(),
                'baseline_tier_2_rate': persona_data['baseline_is_tier_2'].mean(),
                'tier_2_lift': (persona_data['is_tier_2'].mean() /
                               max(persona_data['baseline_is_tier_2'].mean(), 0.001)),
                'shifted_to_2_count': persona_data['shifted_to_2'].sum(),
                'shifted_to_2_rate': persona_data['shifted_to_2'].mean(),

                # Combined extreme metrics
                'extreme_tier_rate': (persona_data['is_tier_0'].sum() +
                                     persona_data['is_tier_2'].sum()) / total_cases,
                'baseline_extreme_rate': ((persona_data['baseline_is_tier_0'].sum() +
                                         persona_data['baseline_is_tier_2'].sum()) / total_cases),

                # Polarization index (how much more extreme than baseline)
                'polarization_index': abs(persona_data['is_tier_0'].mean() -
                                        persona_data['baseline_is_tier_0'].mean()) +
                                     abs(persona_data['is_tier_2'].mean() -
                                        persona_data['baseline_is_tier_2'].mean())
            }

            # Statistical tests for significance
            # Test for Tier 0 difference
            if persona_data['baseline_is_tier_0'].sum() > 0:
                chi2_0, p_val_0 = stats.chi2_contingency([
                    [persona_data['is_tier_0'].sum(), total_cases - persona_data['is_tier_0'].sum()],
                    [persona_data['baseline_is_tier_0'].sum(), total_cases - persona_data['baseline_is_tier_0'].sum()]
                ])[:2]
                results[persona]['tier_0_p_value'] = p_val_0
                results[persona]['tier_0_significant'] = p_val_0 < 0.05
            else:
                results[persona]['tier_0_p_value'] = 1.0
                results[persona]['tier_0_significant'] = False

            # Test for Tier 2 difference
            if persona_data['baseline_is_tier_2'].sum() > 0:
                chi2_2, p_val_2 = stats.chi2_contingency([
                    [persona_data['is_tier_2'].sum(), total_cases - persona_data['is_tier_2'].sum()],
                    [persona_data['baseline_is_tier_2'].sum(), total_cases - persona_data['baseline_is_tier_2'].sum()]
                ])[:2]
                results[persona]['tier_2_p_value'] = p_val_2
                results[persona]['tier_2_significant'] = p_val_2 < 0.05
            else:
                results[persona]['tier_2_p_value'] = 1.0
                results[persona]['tier_2_significant'] = False

        return results

    def analyze_demographic_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze extreme tier patterns by demographic dimensions"""
        patterns = {}

        # Analyze by individual dimensions
        for dimension in ['ethnicity', 'gender', 'geography']:
            patterns[dimension] = {}

            for value in df[dimension].unique():
                if pd.isna(value):
                    continue

                dim_data = df[df[dimension] == value]

                patterns[dimension][value] = {
                    'sample_size': len(dim_data),
                    'tier_0_rate': dim_data['is_tier_0'].mean(),
                    'tier_2_rate': dim_data['is_tier_2'].mean(),
                    'extreme_rate': (dim_data['is_tier_0'].sum() + dim_data['is_tier_2'].sum()) / len(dim_data),
                    'baseline_tier_0_rate': dim_data['baseline_is_tier_0'].mean(),
                    'baseline_tier_2_rate': dim_data['baseline_is_tier_2'].mean(),
                    'tier_0_vs_2_ratio': (dim_data['is_tier_0'].sum() /
                                         max(dim_data['is_tier_2'].sum(), 1))
                }

        # Analyze intersections for extreme patterns
        patterns['intersections'] = {}

        # Most interesting intersections
        for ethnicity in ['black', 'white', 'asian', 'latino']:
            for gender in ['male', 'female']:
                for geography in ['urban_affluent', 'urban_poor', 'rural']:
                    intersection_data = df[
                        (df['ethnicity'] == ethnicity) &
                        (df['gender'] == gender) &
                        (df['geography'] == geography)
                    ]

                    if len(intersection_data) > 0:
                        key = f"{ethnicity}_{gender}_{geography}"
                        patterns['intersections'][key] = {
                            'tier_0_rate': intersection_data['is_tier_0'].mean(),
                            'tier_2_rate': intersection_data['is_tier_2'].mean(),
                            'sample_size': len(intersection_data)
                        }

        return patterns

    def identify_extreme_outliers(self, results: Dict, df: pd.DataFrame) -> Dict:
        """Identify personas that are outliers in extreme tier assignment"""
        outliers = {
            'tier_0_champions': [],  # Most likely to dismiss complaints
            'tier_2_champions': [],  # Most likely to award monetary relief
            'polarized': [],         # High rates of both extremes
            'moderate': []           # Avoid extremes
        }

        # Calculate percentiles for context
        tier_0_rates = [r['tier_0_rate'] for r in results.values()]
        tier_2_rates = [r['tier_2_rate'] for r in results.values()]

        tier_0_p75 = np.percentile(tier_0_rates, 75)
        tier_0_p25 = np.percentile(tier_0_rates, 25)
        tier_2_p75 = np.percentile(tier_2_rates, 75)
        tier_2_p25 = np.percentile(tier_2_rates, 25)

        for persona, metrics in results.items():
            # Tier 0 champions (high dismissal rate)
            if metrics['tier_0_rate'] > tier_0_p75 and metrics['tier_2_rate'] < tier_2_p25:
                outliers['tier_0_champions'].append({
                    'persona': persona,
                    'tier_0_rate': metrics['tier_0_rate'],
                    'tier_2_rate': metrics['tier_2_rate'],
                    'polarization': metrics['polarization_index']
                })

            # Tier 2 champions (high monetary relief)
            elif metrics['tier_2_rate'] > tier_2_p75 and metrics['tier_0_rate'] < tier_0_p25:
                outliers['tier_2_champions'].append({
                    'persona': persona,
                    'tier_0_rate': metrics['tier_0_rate'],
                    'tier_2_rate': metrics['tier_2_rate'],
                    'polarization': metrics['polarization_index']
                })

            # Polarized (high rates of both extremes)
            elif metrics['tier_0_rate'] > tier_0_p75 and metrics['tier_2_rate'] > tier_2_p75:
                outliers['polarized'].append({
                    'persona': persona,
                    'tier_0_rate': metrics['tier_0_rate'],
                    'tier_2_rate': metrics['tier_2_rate'],
                    'polarization': metrics['polarization_index']
                })

            # Moderate (avoid extremes)
            elif metrics['tier_0_rate'] < tier_0_p25 and metrics['tier_2_rate'] < tier_2_p25:
                outliers['moderate'].append({
                    'persona': persona,
                    'tier_0_rate': metrics['tier_0_rate'],
                    'tier_2_rate': metrics['tier_2_rate'],
                    'polarization': metrics['polarization_index']
                })

        # Sort each category by relevant metric
        outliers['tier_0_champions'].sort(key=lambda x: x['tier_0_rate'], reverse=True)
        outliers['tier_2_champions'].sort(key=lambda x: x['tier_2_rate'], reverse=True)
        outliers['polarized'].sort(key=lambda x: x['polarization'], reverse=True)
        outliers['moderate'].sort(key=lambda x: x['polarization'])

        return outliers

    def create_visualizations(self, results: Dict, patterns: Dict, outliers: Dict, df: pd.DataFrame):
        """Create comprehensive visualizations of extreme tier patterns"""
        fig = plt.figure(figsize=(20, 12))

        # Create a grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Plot 1: Scatter plot of Tier 0 vs Tier 2 rates
        ax1 = fig.add_subplot(gs[0, :2])
        personas = list(results.keys())
        tier_0_rates = [results[p]['tier_0_rate'] for p in personas]
        tier_2_rates = [results[p]['tier_2_rate'] for p in personas]

        # Color by geography
        colors_map = {'urban_affluent': 'gold', 'urban_poor': 'darkred', 'rural': 'green'}
        colors_scatter = [colors_map.get(results[p]['geography'], 'gray') for p in personas]

        scatter = ax1.scatter(tier_0_rates, tier_2_rates, c=colors_scatter, alpha=0.6, s=100)

        # Add quadrant lines based on baseline rates
        baseline_tier_0 = df['baseline_is_tier_0'].mean()
        baseline_tier_2 = df['baseline_is_tier_2'].mean()
        ax1.axvline(baseline_tier_0, color='red', linestyle='--', alpha=0.5, label=f'Baseline Tier 0: {baseline_tier_0:.3f}')
        ax1.axhline(baseline_tier_2, color='blue', linestyle='--', alpha=0.5, label=f'Baseline Tier 2: {baseline_tier_2:.3f}')

        # Label outliers
        for category, items in outliers.items():
            if category != 'moderate' and items:
                for item in items[:3]:  # Top 3 from each category
                    idx = personas.index(item['persona'])
                    ax1.annotate(item['persona'].split('_')[0][:3] + '_' +
                               item['persona'].split('_')[1][:1],
                               (tier_0_rates[idx], tier_2_rates[idx]),
                               fontsize=8, alpha=0.7)

        ax1.set_xlabel('Tier 0 Rate (No Action)')
        ax1.set_ylabel('Tier 2 Rate (Monetary Action)')
        ax1.set_title('Extreme Tier Assignment Patterns by Persona')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Bar chart of top Tier 0 personas
        ax2 = fig.add_subplot(gs[0, 2])
        top_tier_0 = sorted(results.items(), key=lambda x: x[1]['tier_0_rate'], reverse=True)[:10]
        personas_0 = [p[0].replace('_', '\n') for p in top_tier_0]
        rates_0 = [p[1]['tier_0_rate'] for p in top_tier_0]

        bars = ax2.barh(range(len(personas_0)), rates_0, color=self.colors['tier_0'])
        ax2.set_yticks(range(len(personas_0)))
        ax2.set_yticklabels(personas_0, fontsize=6)
        ax2.set_xlabel('Tier 0 Rate')
        ax2.set_title('Top 10 Personas: Tier 0 (Dismissal)', fontweight='bold')
        ax2.axvline(baseline_tier_0, color='black', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3, axis='x')

        # Plot 3: Bar chart of top Tier 2 personas
        ax3 = fig.add_subplot(gs[1, 2])
        top_tier_2 = sorted(results.items(), key=lambda x: x[1]['tier_2_rate'], reverse=True)[:10]
        personas_2 = [p[0].replace('_', '\n') for p in top_tier_2]
        rates_2 = [p[1]['tier_2_rate'] for p in top_tier_2]

        bars = ax3.barh(range(len(personas_2)), rates_2, color=self.colors['tier_2'])
        ax3.set_yticks(range(len(personas_2)))
        ax3.set_yticklabels(personas_2, fontsize=6)
        ax3.set_xlabel('Tier 2 Rate')
        ax3.set_title('Top 10 Personas: Tier 2 (Monetary)', fontweight='bold')
        ax3.axvline(baseline_tier_2, color='black', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3, axis='x')

        # Plot 4: Heatmap of extreme rates by demographics
        ax4 = fig.add_subplot(gs[1, :2])

        # Create matrix for heatmap
        ethnicities = ['white', 'black', 'asian', 'latino']
        geographies = ['urban_affluent', 'urban_poor', 'rural']

        # Calculate extreme rate differences for each combination
        heatmap_data = np.zeros((len(ethnicities), len(geographies)))
        for i, eth in enumerate(ethnicities):
            for j, geo in enumerate(geographies):
                # Average across genders for this eth/geo combination
                matching = [p for p, r in results.items()
                          if r['ethnicity'] == eth and r['geography'] == geo]
                if matching:
                    avg_extreme = np.mean([results[p]['extreme_tier_rate'] -
                                         results[p]['baseline_extreme_rate']
                                         for p in matching])
                    heatmap_data[i, j] = avg_extreme

        im = ax4.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', vmin=-0.2, vmax=0.2)
        ax4.set_xticks(range(len(geographies)))
        ax4.set_xticklabels(geographies)
        ax4.set_yticks(range(len(ethnicities)))
        ax4.set_yticklabels(ethnicities)
        ax4.set_title('Extreme Tier Bias by Ethnicity × Geography\n(Difference from Baseline)')

        # Add text annotations
        for i in range(len(ethnicities)):
            for j in range(len(geographies)):
                text = ax4.text(j, i, f'{heatmap_data[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=9)

        plt.colorbar(im, ax=ax4)

        # Plot 5: Distribution comparison
        ax5 = fig.add_subplot(gs[2, :])

        # Prepare data for grouped bar chart
        dimensions = ['ethnicity', 'gender', 'geography']
        tier_0_by_dim = {}
        tier_2_by_dim = {}

        for dim in dimensions:
            tier_0_by_dim[dim] = {k: v['tier_0_rate'] for k, v in patterns[dim].items()}
            tier_2_by_dim[dim] = {k: v['tier_2_rate'] for k, v in patterns[dim].items()}

        # Create grouped bars
        x = np.arange(len(dimensions))
        width = 0.35

        # Calculate averages for each dimension
        tier_0_avgs = [np.mean(list(tier_0_by_dim[d].values())) for d in dimensions]
        tier_2_avgs = [np.mean(list(tier_2_by_dim[d].values())) for d in dimensions]

        bars1 = ax5.bar(x - width/2, tier_0_avgs, width, label='Tier 0 (No Action)',
                       color=self.colors['tier_0'])
        bars2 = ax5.bar(x + width/2, tier_2_avgs, width, label='Tier 2 (Monetary)',
                       color=self.colors['tier_2'])

        ax5.set_xlabel('Demographic Dimension')
        ax5.set_ylabel('Average Rate')
        ax5.set_title('Extreme Tier Rates by Demographic Dimension')
        ax5.set_xticks(x)
        ax5.set_xticklabels(dimensions)
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')

        # Add baseline reference lines
        ax5.axhline(baseline_tier_0, color='red', linestyle='--', alpha=0.3)
        ax5.axhline(baseline_tier_2, color='blue', linestyle='--', alpha=0.3)

        # Plot 6: Polarization Index
        ax6 = fig.add_subplot(gs[2, 2])
        polarization_scores = sorted([(p, r['polarization_index'])
                                     for p, r in results.items()],
                                    key=lambda x: x[1], reverse=True)[:10]

        personas_pol = [p[0].replace('_', '\n') for p in polarization_scores]
        scores_pol = [p[1] for p in polarization_scores]

        bars = ax6.barh(range(len(personas_pol)), scores_pol, color='purple')
        ax6.set_yticks(range(len(personas_pol)))
        ax6.set_yticklabels(personas_pol, fontsize=6)
        ax6.set_xlabel('Polarization Index')
        ax6.set_title('Most Polarized Personas', fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x')

        plt.suptitle('Extreme Tier Assignment Analysis: Who Gets Dismissed vs Compensated?',
                    fontsize=16, fontweight='bold')
        plt.savefig('extreme_tier_analysis.png', dpi=150, bbox_inches='tight')
        print("[INFO] Saved visualization to 'extreme_tier_analysis.png'")
        plt.show()

    def print_detailed_report(self, results: Dict, patterns: Dict, outliers: Dict, df: pd.DataFrame):
        """Print comprehensive extreme tier analysis report"""
        print("\n" + "="*80)
        print("EXTREME TIER ASSIGNMENT ANALYSIS")
        print("="*80)

        # Overall statistics
        print("\n[OVERALL] Extreme Tier Statistics:")
        print("-" * 60)
        overall_tier_0 = df['is_tier_0'].mean()
        overall_tier_2 = df['is_tier_2'].mean()
        baseline_tier_0 = df['baseline_is_tier_0'].mean()
        baseline_tier_2 = df['baseline_is_tier_2'].mean()

        print(f"  Persona Tier 0 Rate: {overall_tier_0:.3f} (Baseline: {baseline_tier_0:.3f})")
        print(f"  Persona Tier 2 Rate: {overall_tier_2:.3f} (Baseline: {baseline_tier_2:.3f})")
        print(f"  Total Extreme Rate: {overall_tier_0 + overall_tier_2:.3f}")

        # Most likely to dismiss (Tier 0)
        print("\n[DISMISSIVE] Top 10 Personas Most Likely to Get Tier 0 (No Action):")
        print("-" * 60)
        top_dismissive = sorted(results.items(),
                               key=lambda x: x[1]['tier_0_rate'],
                               reverse=True)[:10]

        for persona, metrics in top_dismissive:
            lift = metrics['tier_0_lift']
            print(f"  {persona:<30} Rate: {metrics['tier_0_rate']:.3f} "
                  f"(Lift: {lift:.1f}x)")

        # Most likely to get monetary relief (Tier 2)
        print("\n[GENEROUS] Top 10 Personas Most Likely to Get Tier 2 (Monetary):")
        print("-" * 60)
        top_generous = sorted(results.items(),
                            key=lambda x: x[1]['tier_2_rate'],
                            reverse=True)[:10]

        for persona, metrics in top_generous:
            lift = metrics['tier_2_lift']
            print(f"  {persona:<30} Rate: {metrics['tier_2_rate']:.3f} "
                  f"(Lift: {lift:.1f}x)")

        # Most polarized personas
        print("\n[POLARIZED] Personas with Highest Extreme Tier Variance:")
        print("-" * 60)
        if outliers['polarized']:
            for item in outliers['polarized'][:5]:
                print(f"  {item['persona']:<30} "
                      f"T0: {item['tier_0_rate']:.3f}, T2: {item['tier_2_rate']:.3f}, "
                      f"Polarization: {item['polarization']:.3f}")
        else:
            print("  No highly polarized personas found")

        # Moderate personas (avoid extremes)
        print("\n[MODERATE] Personas Least Likely to Get Extreme Tiers:")
        print("-" * 60)
        if outliers['moderate']:
            for item in outliers['moderate'][:5]:
                print(f"  {item['persona']:<30} "
                      f"T0: {item['tier_0_rate']:.3f}, T2: {item['tier_2_rate']:.3f}")
        else:
            print("  No consistently moderate personas found")

        # Demographic patterns
        print("\n[DEMOGRAPHICS] Extreme Tier Patterns by Dimension:")
        print("-" * 60)

        for dimension in ['ethnicity', 'gender', 'geography']:
            print(f"\n  {dimension.upper()}:")
            dim_data = patterns[dimension]

            # Sort by tier 0 rate
            sorted_dim = sorted(dim_data.items(),
                              key=lambda x: x[1]['tier_0_rate'],
                              reverse=True)

            for value, metrics in sorted_dim:
                t0_rate = metrics['tier_0_rate']
                t2_rate = metrics['tier_2_rate']
                ratio = metrics['tier_0_vs_2_ratio']
                print(f"    {value:<20} T0: {t0_rate:.3f}, T2: {t2_rate:.3f} "
                      f"(T0/T2 ratio: {ratio:.1f})")

        # Significant shifts
        print("\n[SHIFTS] Personas with Significant Tier Shifts:")
        print("-" * 60)

        # Biggest shifts to Tier 0
        shifts_to_0 = sorted(results.items(),
                            key=lambda x: x[1]['shifted_to_0_rate'],
                            reverse=True)[:5]
        print("\n  Shifted TO Tier 0 (became dismissive):")
        for persona, metrics in shifts_to_0:
            if metrics['shifted_to_0_rate'] > 0:
                print(f"    {persona:<30} Shift rate: {metrics['shifted_to_0_rate']:.3f}")

        # Biggest shifts to Tier 2
        shifts_to_2 = sorted(results.items(),
                            key=lambda x: x[1]['shifted_to_2_rate'],
                            reverse=True)[:5]
        print("\n  Shifted TO Tier 2 (became generous):")
        for persona, metrics in shifts_to_2:
            if metrics['shifted_to_2_rate'] > 0:
                print(f"    {persona:<30} Shift rate: {metrics['shifted_to_2_rate']:.3f}")

        # Statistical significance summary
        print("\n[SIGNIFICANCE] Statistically Significant Extreme Tier Biases:")
        print("-" * 60)

        sig_tier_0 = [p for p, m in results.items() if m.get('tier_0_significant', False)]
        sig_tier_2 = [p for p, m in results.items() if m.get('tier_2_significant', False)]

        print(f"  Personas with significant Tier 0 bias: {len(sig_tier_0)}")
        print(f"  Personas with significant Tier 2 bias: {len(sig_tier_2)}")

        # Key insights
        print("\n[INSIGHTS] Key Findings:")
        print("-" * 60)

        # Find patterns
        rural_t0 = np.mean([m['tier_0_rate'] for p, m in results.items()
                          if 'rural' in p])
        urban_poor_t2 = np.mean([m['tier_2_rate'] for p, m in results.items()
                                if 'urban_poor' in p])

        print(f"  • Rural personas average Tier 0 rate: {rural_t0:.3f}")
        print(f"  • Urban poor personas average Tier 2 rate: {urban_poor_t2:.3f}")

        # Gender differences in extremes
        male_t0 = np.mean([m['tier_0_rate'] for p, m in results.items()
                         if 'male' in p])
        female_t0 = np.mean([m['tier_0_rate'] for p, m in results.items()
                           if 'female' in p])

        print(f"  • Male vs Female Tier 0 rates: {male_t0:.3f} vs {female_t0:.3f}")

    def run_analysis(self):
        """Run complete extreme tier analysis"""
        # Connect to database
        conn = self.connect_to_database()

        try:
            # Fetch data
            df = self.fetch_extreme_tier_data(conn)

            # Analyze extreme tier propensity
            print("\n[ANALYSIS] Computing extreme tier patterns...")
            results = self.analyze_extreme_tier_propensity(df)

            # Analyze demographic patterns
            patterns = self.analyze_demographic_patterns(df)

            # Identify outliers
            outliers = self.identify_extreme_outliers(results, df)

            # Save results to CSV
            results_df = pd.DataFrame.from_dict(results, orient='index')
            results_df.to_csv('extreme_tier_analysis.csv')
            print("[INFO] Saved detailed results to 'extreme_tier_analysis.csv'")

            # Print report
            self.print_detailed_report(results, patterns, outliers, df)

            # Create visualizations
            self.create_visualizations(results, patterns, outliers, df)

        finally:
            conn.close()
            print("\n[INFO] Analysis complete!")


def main():
    """Main execution function"""
    analyzer = ExtremeTierAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()