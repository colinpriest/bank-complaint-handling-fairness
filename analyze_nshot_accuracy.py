#!/usr/bin/env python3
"""
Analyze N-Shot vs Zero-Shot Accuracy Issues

This script analyzes the database to identify why n-shot accuracy is lower than zero-shot.
"""

import os
import psycopg2
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Load environment variables
load_dotenv()

def get_db_connection():
    """Establishes a new database connection."""
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'fairness_analysis'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', '')
    }
    return psycopg2.connect(**db_config)

def analyze_nshot_accuracy():
    """
    Analyzes n-shot vs zero-shot accuracy to identify root causes.
    """
    connection = get_db_connection()

    try:
        cursor = connection.cursor()

        # Query 1: Overall accuracy comparison
        overall_accuracy_query = """
        SELECT
            be.decision_method,
            COUNT(*) as total_cases,
            SUM(CASE WHEN be.llm_simplified_tier = gt.simplified_ground_truth_tier THEN 1 ELSE 0 END) as correct_decisions,
            AVG(CASE WHEN be.llm_simplified_tier = gt.simplified_ground_truth_tier THEN 1.0 ELSE 0.0 END) as accuracy_rate
        FROM baseline_experiments be
        JOIN ground_truth gt ON be.case_id = gt.case_id
        WHERE be.decision_method != 'baseline'
        GROUP BY be.decision_method
        ORDER BY be.decision_method;
        """

        # Query 2: Accuracy by ground truth tier
        tier_accuracy_query = """
        SELECT
            be.decision_method,
            gt.simplified_ground_truth_tier,
            COUNT(*) as total_cases,
            SUM(CASE WHEN be.llm_simplified_tier = gt.simplified_ground_truth_tier THEN 1 ELSE 0 END) as correct_decisions,
            AVG(CASE WHEN be.llm_simplified_tier = gt.simplified_ground_truth_tier THEN 1.0 ELSE 0.0 END) as accuracy_rate
        FROM baseline_experiments be
        JOIN ground_truth gt ON be.case_id = gt.case_id
        WHERE be.decision_method != 'baseline'
        GROUP BY be.decision_method, gt.simplified_ground_truth_tier
        ORDER BY be.decision_method, gt.simplified_ground_truth_tier;
        """

        # Query 3: Decision distribution analysis
        decision_distribution_query = """
        SELECT
            be.decision_method,
            be.llm_simplified_tier,
            COUNT(*) as decisions_made,
            AVG(CASE WHEN be.llm_simplified_tier = gt.simplified_ground_truth_tier THEN 1.0 ELSE 0.0 END) as accuracy_for_this_decision
        FROM baseline_experiments be
        JOIN ground_truth gt ON be.case_id = gt.case_id
        WHERE be.decision_method != 'baseline'
        GROUP BY be.decision_method, be.llm_simplified_tier
        ORDER BY be.decision_method, be.llm_simplified_tier;
        """

        # Query 4: Demographic bias in accuracy
        demographic_accuracy_query = """
        SELECT
            be.decision_method,
            dc.gender,
            dc.ethnicity,
            dc.geographic_tier,
            COUNT(*) as total_cases,
            AVG(CASE WHEN be.llm_simplified_tier = gt.simplified_ground_truth_tier THEN 1.0 ELSE 0.0 END) as accuracy_rate
        FROM baseline_experiments be
        JOIN ground_truth gt ON be.case_id = gt.case_id
        JOIN demographic_cases dc ON be.case_id = dc.case_id
        WHERE be.decision_method != 'baseline'
        GROUP BY be.decision_method, dc.gender, dc.ethnicity, dc.geographic_tier
        HAVING COUNT(*) >= 5
        ORDER BY be.decision_method, dc.gender, dc.ethnicity, dc.geographic_tier;
        """

        # Query 5: Confusion matrix data
        confusion_matrix_query = """
        SELECT
            be.decision_method,
            gt.simplified_ground_truth_tier as true_tier,
            be.llm_simplified_tier as predicted_tier,
            COUNT(*) as count
        FROM baseline_experiments be
        JOIN ground_truth gt ON be.case_id = gt.case_id
        WHERE be.decision_method != 'baseline'
        GROUP BY be.decision_method, gt.simplified_ground_truth_tier, be.llm_simplified_tier
        ORDER BY be.decision_method, gt.simplified_ground_truth_tier, be.llm_simplified_tier;
        """

        print("=== OVERALL ACCURACY COMPARISON ===")
        cursor.execute(overall_accuracy_query)
        overall_results = cursor.fetchall()
        for row in overall_results:
            method, total, correct, accuracy = row
            print(f"{method}: {correct}/{total} = {accuracy:.4f} ({accuracy*100:.2f}%)")

        print("\n=== ACCURACY BY GROUND TRUTH TIER ===")
        cursor.execute(tier_accuracy_query)
        tier_results = cursor.fetchall()

        tier_data = {}
        for row in tier_results:
            method, tier, total, correct, accuracy = row
            if method not in tier_data:
                tier_data[method] = {}
            tier_data[method][tier] = {
                'total': total,
                'correct': correct,
                'accuracy': accuracy
            }

        for method in sorted(tier_data.keys()):
            print(f"\n{method.upper()}:")
            for tier in sorted(tier_data[method].keys()):
                data = tier_data[method][tier]
                print(f"  Tier {tier}: {data['correct']}/{data['total']} = {data['accuracy']:.4f} ({data['accuracy']*100:.2f}%)")

        print("\n=== DECISION DISTRIBUTION ANALYSIS ===")
        cursor.execute(decision_distribution_query)
        distribution_results = cursor.fetchall()

        dist_data = {}
        for row in distribution_results:
            method, decision_tier, count, accuracy = row
            if method not in dist_data:
                dist_data[method] = {}
            dist_data[method][decision_tier] = {
                'count': count,
                'accuracy': accuracy
            }

        for method in sorted(dist_data.keys()):
            print(f"\n{method.upper()} Decision Distribution:")
            total_decisions = sum(data['count'] for data in dist_data[method].values())
            for tier in sorted(dist_data[method].keys()):
                data = dist_data[method][tier]
                percentage = (data['count'] / total_decisions) * 100
                print(f"  Decided Tier {tier}: {data['count']} ({percentage:.1f}%) - Accuracy: {data['accuracy']:.4f}")

        print("\n=== CONFUSION MATRIX ANALYSIS ===")
        cursor.execute(confusion_matrix_query)
        confusion_results = cursor.fetchall()

        confusion_data = {}
        for row in confusion_results:
            method, true_tier, pred_tier, count = row
            if method not in confusion_data:
                confusion_data[method] = {}
            if true_tier not in confusion_data[method]:
                confusion_data[method][true_tier] = {}
            confusion_data[method][true_tier][pred_tier] = count

        for method in sorted(confusion_data.keys()):
            print(f"\n{method.upper()} Confusion Matrix:")
            print("True\\Pred", end="")
            all_tiers = sorted(set(
                tier for true_data in confusion_data[method].values()
                for tier in true_data.keys()
            ))
            for tier in all_tiers:
                print(f"\tT{tier}", end="")
            print()

            for true_tier in sorted(confusion_data[method].keys()):
                print(f"True T{true_tier}", end="")
                for pred_tier in all_tiers:
                    count = confusion_data[method][true_tier].get(pred_tier, 0)
                    print(f"\t{count}", end="")
                print()

        print("\n=== DEMOGRAPHIC ACCURACY PATTERNS ===")
        cursor.execute(demographic_accuracy_query)
        demo_results = cursor.fetchall()

        demo_data = {}
        for row in demo_results:
            method, gender, ethnicity, geo_tier, total, accuracy = row
            if method not in demo_data:
                demo_data[method] = []
            demo_data[method].append({
                'gender': gender,
                'ethnicity': ethnicity,
                'geo_tier': geo_tier,
                'total': total,
                'accuracy': accuracy
            })

        for method in sorted(demo_data.keys()):
            print(f"\n{method.upper()} Accuracy by Demographics (cases >= 5):")
            # Sort by accuracy to see patterns
            sorted_demo = sorted(demo_data[method], key=lambda x: x['accuracy'])

            print("Lowest accuracy groups:")
            for item in sorted_demo[:5]:
                print(f"  {item['gender']}, {item['ethnicity']}, Geo-T{item['geo_tier']}: {item['accuracy']:.4f} (n={item['total']})")

            print("Highest accuracy groups:")
            for item in sorted_demo[-5:]:
                print(f"  {item['gender']}, {item['ethnicity']}, Geo-T{item['geo_tier']}: {item['accuracy']:.4f} (n={item['total']})")

        # Calculate accuracy differences to identify problem areas
        print("\n=== ANALYSIS SUMMARY ===")

        # Get overall accuracy difference
        zero_shot_acc = next((acc for method, _, _, acc in overall_results if method == 'zero-shot'), 0)
        n_shot_acc = next((acc for method, _, _, acc in overall_results if method == 'n-shot'), 0)

        print(f"Zero-shot accuracy: {zero_shot_acc:.4f}")
        print(f"N-shot accuracy: {n_shot_acc:.4f}")
        print(f"Accuracy difference: {zero_shot_acc - n_shot_acc:.4f} ({((zero_shot_acc - n_shot_acc) / zero_shot_acc * 100):.1f}% worse)")

        # Identify which tiers are most problematic for n-shot
        if 'n-shot' in tier_data and 'zero-shot' in tier_data:
            print("\nTier-specific accuracy differences (zero-shot - n-shot):")
            for tier in sorted(tier_data['zero-shot'].keys()):
                if tier in tier_data['n-shot']:
                    zero_acc = tier_data['zero-shot'][tier]['accuracy']
                    n_acc = tier_data['n-shot'][tier]['accuracy']
                    diff = zero_acc - n_acc
                    print(f"  Tier {tier}: {diff:.4f} ({diff/zero_acc*100:.1f}% worse)")

        return {
            'overall_accuracy': overall_results,
            'tier_accuracy': tier_results,
            'decision_distribution': distribution_results,
            'demographic_accuracy': demo_results,
            'confusion_matrix': confusion_results
        }

    except Exception as e:
        print(f"Error analyzing accuracy: {str(e)}")
        return None

    finally:
        connection.close()

if __name__ == "__main__":
    analyze_nshot_accuracy()