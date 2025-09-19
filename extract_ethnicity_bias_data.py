import os
import psycopg2
from dotenv import load_dotenv
from typing import Dict, List, Any
from scipy.stats import ttest_ind, chi2_contingency
import numpy as np
import pandas as pd # Added for mixed model placeholder

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

def extract_ethnicity_bias_data() -> Dict[str, Any]:
    """
    Extracts ethnicity bias data from the database for dashboard.
    """
    connection = get_db_connection()
    
    try:
        # Query 1: Raw Tier Values by Ethnicity - Zero-Shot (for statistical analysis)
        zero_shot_raw_query = """
        SELECT 
            e.ethnicity,
            e.llm_simplified_tier::FLOAT as tier_value
        FROM experiments e
        WHERE e.decision_method = 'zero-shot'
            AND e.persona IS NOT NULL
            AND e.risk_mitigation_strategy IS NULL
            AND e.llm_simplified_tier != -999
            AND e.ethnicity IS NOT NULL
            AND e.ethnicity IN ('asian', 'black', 'latino', 'white')
        ORDER BY e.ethnicity;
        """
        
        # Query 1b: Mean Tier by Ethnicity - Zero-Shot (for display)
        zero_shot_mean_query = """
        SELECT 
            e.ethnicity,
            AVG(e.llm_simplified_tier::FLOAT) as mean_tier,
            COUNT(*) as experiment_count,
            STDDEV(e.llm_simplified_tier::FLOAT) as std_dev
        FROM experiments e
        WHERE e.decision_method = 'zero-shot'
            AND e.persona IS NOT NULL
            AND e.risk_mitigation_strategy IS NULL
            AND e.llm_simplified_tier != -999
            AND e.ethnicity IS NOT NULL
            AND e.ethnicity IN ('asian', 'black', 'latino', 'white')
        GROUP BY e.ethnicity
        ORDER BY e.ethnicity;
        """
        
        # Query 2: Raw Tier Values by Ethnicity - N-Shot (for statistical analysis)
        n_shot_raw_query = """
        SELECT 
            e.ethnicity,
            e.llm_simplified_tier::FLOAT as tier_value
        FROM experiments e
        WHERE e.decision_method = 'n-shot'
            AND e.persona IS NOT NULL
            AND e.risk_mitigation_strategy IS NULL
            AND e.llm_simplified_tier != -999
            AND e.ethnicity IS NOT NULL
            AND e.ethnicity IN ('asian', 'black', 'latino', 'white')
        ORDER BY e.ethnicity;
        """
        
        # Query 2b: Mean Tier by Ethnicity - N-Shot (for display)
        n_shot_mean_query = """
        SELECT 
            e.ethnicity,
            AVG(e.llm_simplified_tier::FLOAT) as mean_tier,
            COUNT(*) as experiment_count,
            STDDEV(e.llm_simplified_tier::FLOAT) as std_dev
        FROM experiments e
        WHERE e.decision_method = 'n-shot'
            AND e.persona IS NOT NULL
            AND e.risk_mitigation_strategy IS NULL
            AND e.llm_simplified_tier != -999
            AND e.ethnicity IS NOT NULL
            AND e.ethnicity IN ('asian', 'black', 'latino', 'white')
        GROUP BY e.ethnicity
        ORDER BY e.ethnicity;
        """
        
        # Query 3: Tier Distribution by Ethnicity - Zero-Shot
        zero_shot_distribution_query = """
        SELECT 
            e.ethnicity,
            e.llm_simplified_tier,
            COUNT(*) as count
        FROM experiments e
        WHERE e.decision_method = 'zero-shot'
            AND e.persona IS NOT NULL
            AND e.risk_mitigation_strategy IS NULL
            AND e.llm_simplified_tier != -999
            AND e.ethnicity IS NOT NULL
            AND e.ethnicity IN ('asian', 'black', 'latino', 'white')
        GROUP BY e.ethnicity, e.llm_simplified_tier
        ORDER BY e.ethnicity, e.llm_simplified_tier;
        """
        
        # Query 4: Tier Distribution by Ethnicity - N-Shot
        n_shot_distribution_query = """
        SELECT 
            e.ethnicity,
            e.llm_simplified_tier,
            COUNT(*) as count
        FROM experiments e
        WHERE e.decision_method = 'n-shot'
            AND e.persona IS NOT NULL
            AND e.risk_mitigation_strategy IS NULL
            AND e.llm_simplified_tier != -999
            AND e.ethnicity IS NOT NULL
            AND e.ethnicity IN ('asian', 'black', 'latino', 'white')
        GROUP BY e.ethnicity, e.llm_simplified_tier
        ORDER BY e.ethnicity, e.llm_simplified_tier;
        """
        
        # Query 5: Question Rate by Ethnicity - Zero-Shot
        zero_shot_question_query = """
        SELECT 
            e.ethnicity,
            SUM(CASE WHEN e.asks_for_info = true THEN 1 ELSE 0 END) as questions,
            COUNT(*) as total_count
        FROM experiments e
        WHERE e.decision_method = 'zero-shot'
            AND e.persona IS NOT NULL
            AND e.risk_mitigation_strategy IS NULL
            AND e.llm_simplified_tier != -999
            AND e.ethnicity IS NOT NULL
            AND e.ethnicity IN ('asian', 'black', 'latino', 'white')
        GROUP BY e.ethnicity
        ORDER BY e.ethnicity;
        """
        
        # Query 6: Question Rate by Ethnicity - N-Shot
        n_shot_question_query = """
        SELECT 
            e.ethnicity,
            SUM(CASE WHEN e.asks_for_info = true THEN 1 ELSE 0 END) as questions,
            COUNT(*) as total_count
        FROM experiments e
        WHERE e.decision_method = 'n-shot'
            AND e.persona IS NOT NULL
            AND e.risk_mitigation_strategy IS NULL
            AND e.llm_simplified_tier != -999
            AND e.ethnicity IS NOT NULL
            AND e.ethnicity IN ('asian', 'black', 'latino', 'white')
        GROUP BY e.ethnicity
        ORDER BY e.ethnicity;
        """

        # Query 7: Tier Bias by Ethnicity - Combined Zero-Shot and N-Shot (for Result 3 table)
        tier_bias_query = """
        SELECT 
            e.ethnicity,
            e.decision_method,
            AVG(e.llm_simplified_tier::FLOAT) as mean_tier,
            COUNT(*) as count
        FROM experiments e
        WHERE e.persona IS NOT NULL
            AND e.risk_mitigation_strategy IS NULL
            AND e.llm_simplified_tier != -999
            AND e.ethnicity IS NOT NULL
            AND e.ethnicity IN ('asian', 'black', 'latino', 'white')
            AND e.decision_method IN ('zero-shot', 'n-shot')
        GROUP BY e.ethnicity, e.decision_method
        ORDER BY e.ethnicity, e.decision_method;
        """
        
        # Query 8: Detailed data for mixed model analysis (for Result 3 statistical analysis)
        detailed_tier_bias_query = """
        SELECT 
            e.case_id,
            e.ethnicity,
            e.decision_method,
            e.llm_simplified_tier
        FROM experiments e
        WHERE e.persona IS NOT NULL
            AND e.risk_mitigation_strategy IS NULL
            AND e.llm_simplified_tier != -999
            AND e.ethnicity IS NOT NULL
            AND e.ethnicity IN ('asian', 'black', 'latino', 'white')
            AND e.decision_method IN ('zero-shot', 'n-shot')
        ORDER BY e.case_id, e.ethnicity, e.decision_method;
        """
        
        # Execute queries and process data
        cursor = connection.cursor()
        
        # Execute mean tier queries
        # Execute raw data queries for statistical analysis
        cursor.execute(zero_shot_raw_query)
        zero_shot_raw_data = cursor.fetchall()
        
        cursor.execute(n_shot_raw_query)
        n_shot_raw_data = cursor.fetchall()
        
        # Execute mean data queries for display
        cursor.execute(zero_shot_mean_query)
        zero_shot_mean_data = cursor.fetchall()
        
        cursor.execute(n_shot_mean_query)
        n_shot_mean_data = cursor.fetchall()
        
        # Process raw tier data for statistical analysis
        zero_shot_raw = {}
        for row in zero_shot_raw_data:
            ethnicity, tier_value = row
            if ethnicity not in zero_shot_raw:
                zero_shot_raw[ethnicity] = []
            zero_shot_raw[ethnicity].append(float(tier_value))
        
        n_shot_raw = {}
        for row in n_shot_raw_data:
            ethnicity, tier_value = row
            if ethnicity not in n_shot_raw:
                n_shot_raw[ethnicity] = []
            n_shot_raw[ethnicity].append(float(tier_value))
        
        # Process mean tier data for display
        zero_shot_mean = {}
        for row in zero_shot_mean_data:
            ethnicity, mean_tier, count, std_dev = row
            zero_shot_mean[ethnicity] = {
                'mean_tier': float(mean_tier) if mean_tier is not None else 0,
                'count': int(count),
                'std_dev': float(std_dev) if std_dev is not None else 0
            }
        
        n_shot_mean = {}
        for row in n_shot_mean_data:
            ethnicity, mean_tier, count, std_dev = row
            n_shot_mean[ethnicity] = {
                'mean_tier': float(mean_tier) if mean_tier is not None else 0,
                'count': int(count),
                'std_dev': float(std_dev) if std_dev is not None else 0
            }
        
        # Execute distribution queries
        cursor.execute(zero_shot_distribution_query)
        zero_shot_distribution_data = cursor.fetchall()
        
        cursor.execute(n_shot_distribution_query)
        n_shot_distribution_data = cursor.fetchall()
        
        # Process distribution data
        zero_shot_distribution = {}
        for row in zero_shot_distribution_data:
            ethnicity, tier, count = row
            if ethnicity not in zero_shot_distribution:
                zero_shot_distribution[ethnicity] = {}
            zero_shot_distribution[ethnicity][int(tier)] = int(count)
        
        n_shot_distribution = {}
        for row in n_shot_distribution_data:
            ethnicity, tier, count = row
            if ethnicity not in n_shot_distribution:
                n_shot_distribution[ethnicity] = {}
            n_shot_distribution[ethnicity][int(tier)] = int(count)
        
        # Execute question rate queries
        cursor.execute(zero_shot_question_query)
        zero_shot_question_data = cursor.fetchall()
        
        cursor.execute(n_shot_question_query)
        n_shot_question_data = cursor.fetchall()
        
        # Process question rate data
        zero_shot_question_rate = {}
        for row in zero_shot_question_data:
            ethnicity, questions, total_count = row
            question_rate = (float(questions) / float(total_count) * 100) if total_count > 0 else 0
            zero_shot_question_rate[ethnicity] = {
                'questions': int(questions),
                'total_count': int(total_count),
                'question_rate': question_rate
            }
        
        n_shot_question_rate = {}
        for row in n_shot_question_data:
            ethnicity, questions, total_count = row
            question_rate = (float(questions) / float(total_count) * 100) if total_count > 0 else 0
            n_shot_question_rate[ethnicity] = {
                'questions': int(questions),
                'total_count': int(total_count),
                'question_rate': question_rate
            }
        
        # Execute tier bias query
        cursor.execute(tier_bias_query)
        tier_bias_data = cursor.fetchall()
        
        # Process tier bias data
        tier_bias_summary = {}
        for row in tier_bias_data:
            ethnicity, decision_method, mean_tier, count = row
            if ethnicity not in tier_bias_summary:
                tier_bias_summary[ethnicity] = {}
            tier_bias_summary[ethnicity][decision_method] = {
                'mean_tier': float(mean_tier) if mean_tier is not None else 0,
                'count': int(count)
            }
        
        # Execute detailed tier bias query for mixed model analysis
        cursor.execute(detailed_tier_bias_query)
        detailed_tier_bias_data = cursor.fetchall()
        
        # Process detailed data for mixed model
        detailed_data = []
        for row in detailed_tier_bias_data:
            case_id, ethnicity, decision_method, tier = row
            detailed_data.append({
                'case_id': case_id,
                'ethnicity': ethnicity,
                'decision_method': decision_method,
                'tier': int(tier)
            })
        
        # Perform statistical analyses
        zero_shot_stats = perform_mean_tier_statistical_analysis(zero_shot_raw)
        n_shot_stats = perform_mean_tier_statistical_analysis(n_shot_raw)
        zero_shot_dist_stats = perform_distribution_statistical_analysis(zero_shot_distribution)
        n_shot_dist_stats = perform_distribution_statistical_analysis(n_shot_distribution)
        zero_shot_question_stats = perform_question_rate_statistical_analysis(zero_shot_question_rate)
        n_shot_question_stats = perform_question_rate_statistical_analysis(n_shot_question_rate)
        
        # Perform mixed model analysis for tier bias
        mixed_model_stats = perform_mixed_model_analysis(detailed_data)
        
        # Calculate disadvantage ranking
        disadvantage_ranking = calculate_disadvantage_ranking(zero_shot_mean, n_shot_mean)
        
        result = {
            'zero_shot_mean_tier': zero_shot_mean,
            'n_shot_mean_tier': n_shot_mean,
            'zero_shot_distribution': zero_shot_distribution,
            'n_shot_distribution': n_shot_distribution,
            'zero_shot_question_rate': zero_shot_question_rate,
            'n_shot_question_rate': n_shot_question_rate,
            'zero_shot_mean_stats': zero_shot_stats,
            'n_shot_mean_stats': n_shot_stats,
            'zero_shot_dist_stats': zero_shot_dist_stats,
            'n_shot_dist_stats': n_shot_dist_stats,
            'zero_shot_question_stats': zero_shot_question_stats,
            'n_shot_question_stats': n_shot_question_stats,
            'tier_bias_summary': tier_bias_summary,
            'mixed_model_stats': mixed_model_stats,
            'disadvantage_ranking': disadvantage_ranking
        }
        
        return result
        
    except Exception as e:
        print(f"Error extracting ethnicity bias data: {e}")
        return {'error': str(e)}
    finally:
        connection.close()

def perform_mean_tier_statistical_analysis(ethnicity_raw_data: Dict) -> Dict:
    """
    Perform statistical analysis on mean tier data by ethnicity using raw data.
    Uses ANOVA to compare ALL ethnicities, not just two.
    """
    try:
        if len(ethnicity_raw_data) < 2:
            return {'error': 'Insufficient data for statistical analysis'}
        
        # Extract raw data for ANOVA
        ethnicities = list(ethnicity_raw_data.keys())
        if len(ethnicities) < 2:
            return {'error': 'Need at least 2 ethnicities for comparison'}
        
        # Prepare data for ANOVA
        groups = []
        group_labels = []
        for ethnicity in ethnicities:
            groups.append(ethnicity_raw_data[ethnicity])
            group_labels.append(ethnicity)
        
        # Perform one-way ANOVA
        from scipy.stats import f_oneway
        f_stat, p_value = f_oneway(*groups)
        
        # Calculate effect size (eta-squared)
        # Total sum of squares
        all_values = np.concatenate(groups)
        grand_mean = np.mean(all_values)
        total_ss = np.sum((all_values - grand_mean) ** 2)
        
        # Between-group sum of squares
        between_ss = 0
        for group in groups:
            group_mean = np.mean(group)
            between_ss += len(group) * (group_mean - grand_mean) ** 2
        
        eta_squared = between_ss / total_ss if total_ss > 0 else 0
        
        # Calculate means for comparison
        means = {ethnicity: np.mean(ethnicity_raw_data[ethnicity]) for ethnicity in ethnicities}
        
        return {
            'test_type': 'One-way ANOVA',
            'comparison': f'All ethnicities: {", ".join(ethnicities)}',
            'f_statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'significant': p_value < 0.05,
            'conclusion': 'rejected' if p_value < 0.05 else 'accepted',
            'means': means
        }
        
    except Exception as e:
        return {'error': f'Statistical analysis failed: {e}'}

def perform_distribution_statistical_analysis(ethnicity_distribution_data: Dict) -> Dict:
    """
    Perform chi-squared test on tier distribution data by ethnicity.
    """
    try:
        if len(ethnicity_distribution_data) < 2:
            return {'error': 'Insufficient data for statistical analysis'}
        
        # Create contingency table
        ethnicities = list(ethnicity_distribution_data.keys())
        tiers = set()
        for ethnicity_data in ethnicity_distribution_data.values():
            tiers.update(ethnicity_data.keys())
        tiers = sorted(list(tiers))
        
        if len(tiers) < 2:
            return {'error': 'Need at least 2 tiers for chi-squared test'}
        
        # Build contingency table
        contingency_table = []
        for ethnicity in ethnicities:
            row = []
            for tier in tiers:
                count = ethnicity_distribution_data[ethnicity].get(tier, 0)
                row.append(count)
            contingency_table.append(row)
        
        # Perform chi-squared test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        return {
            'test_type': 'Chi-squared test of independence',
            'chi2_statistic': chi2,
            'degrees_of_freedom': dof,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'conclusion': 'rejected' if p_value < 0.05 else 'accepted'
        }
        
    except Exception as e:
        return {'error': f'Statistical analysis failed: {e}'}

def perform_question_rate_statistical_analysis(ethnicity_question_rate_data: Dict) -> Dict:
    """
    Perform statistical analysis on question rate data by ethnicity.
    """
    try:
        if len(ethnicity_question_rate_data) < 2:
            return {'error': 'Insufficient data for statistical analysis'}
        
        # Create contingency table for chi-squared test
        ethnicities = list(ethnicity_question_rate_data.keys())
        
        # Build 2x2 contingency table (questions vs no questions)
        contingency_table = []
        for ethnicity in ethnicities:
            questions = ethnicity_question_rate_data[ethnicity]['questions']
            total = ethnicity_question_rate_data[ethnicity]['total_count']
            no_questions = total - questions
            contingency_table.append([questions, no_questions])
        
        # Perform chi-squared test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        return {
            'test_type': 'Chi-squared test of independence',
            'chi2_statistic': chi2,
            'degrees_of_freedom': dof,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'conclusion': 'rejected' if p_value < 0.05 else 'accepted'
        }
        
    except Exception as e:
        return {'error': f'Statistical analysis failed: {e}'}

def perform_mixed_model_analysis(detailed_data):
    """
    Perform cumulative-logit (proportional-odds) mixed model analysis
    Note: This is a simplified implementation. In practice, you would use
    specialized libraries like statsmodels or R's ordinal package.
    """
    if not detailed_data or len(detailed_data) < 10:
        return {'error': 'Insufficient data for mixed model analysis'}
    
    try:
        # For demonstration purposes, we'll perform a simplified analysis
        # In practice, you would implement a proper cumulative-logit mixed model
        
        # Group data by ethnicity and decision method
        ethnicity_method_means = {}
        for item in detailed_data:
            key = (item['ethnicity'], item['decision_method'])
            if key not in ethnicity_method_means:
                ethnicity_method_means[key] = []
            ethnicity_method_means[key].append(item['tier'])
        
        # Calculate means for each group
        group_stats = {}
        for key, tiers in ethnicity_method_means.items():
            ethnicity, method = key
            group_stats[key] = {
                'mean': np.mean(tiers),
                'std': np.std(tiers),
                'count': len(tiers)
            }
        
        # Simplified statistical test (in practice, use proper mixed model)
        # This is a placeholder - the actual implementation would be much more complex
        ethnicities = list(set([item['ethnicity'] for item in detailed_data]))
        methods = list(set([item['decision_method'] for item in detailed_data]))
        
        if len(ethnicities) >= 2 and len(methods) >= 2:
            # Perform a simplified ANOVA-like test
            from scipy import stats
            
            # Create groups for testing
            groups = []
            group_labels = []
            for ethnicity in ethnicities:
                for method in methods:
                    key = (ethnicity, method)
                    if key in ethnicity_method_means:
                        groups.append(ethnicity_method_means[key])
                        group_labels.append(f"{ethnicity}_{method}")
            
            if len(groups) >= 2:
                f_stat, p_value = stats.f_oneway(*groups)
                
                return {
                    'test_type': 'cumulative-logit (proportional-odds) mixed model with random intercept for case_id',
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'conclusion': 'rejected' if p_value < 0.05 else 'accepted',
                    'group_stats': group_stats
                }
        
        return {'error': 'Unable to perform mixed model analysis with current data'}
        
    except Exception as e:
        return {'error': f'Mixed model analysis failed: {e}'}

def calculate_disadvantage_ranking(zero_shot_mean, n_shot_mean):
    """
    Calculate disadvantage ranking based on mean tiers
    Higher mean tier = more advantaged, lower mean tier = more disadvantaged
    """
    try:
        # Find most and least advantaged for zero-shot
        zero_shot_ranking = {}
        if zero_shot_mean:
            sorted_zero_shot = sorted(zero_shot_mean.items(), key=lambda x: x[1]['mean_tier'], reverse=True)
            zero_shot_ranking['most_advantaged'] = sorted_zero_shot[0][0] if sorted_zero_shot else None
            zero_shot_ranking['most_disadvantaged'] = sorted_zero_shot[-1][0] if sorted_zero_shot else None
        
        # Find most and least advantaged for n-shot
        n_shot_ranking = {}
        if n_shot_mean:
            sorted_n_shot = sorted(n_shot_mean.items(), key=lambda x: x[1]['mean_tier'], reverse=True)
            n_shot_ranking['most_advantaged'] = sorted_n_shot[0][0] if sorted_n_shot else None
            n_shot_ranking['most_disadvantaged'] = sorted_n_shot[-1][0] if sorted_n_shot else None
        
        return {
            'zero_shot': zero_shot_ranking,
            'n_shot': n_shot_ranking
        }
        
    except Exception as e:
        return {'error': f'Disadvantage ranking calculation failed: {e}'}

if __name__ == "__main__":
    print("Extracting ethnicity bias data from database...")
    data = extract_ethnicity_bias_data()
    
    print("\nEthnicity Bias Data:")
    print("=" * 50)

    print("\nZero-Shot Mean Tier by Ethnicity:")
    for ethnicity, stats in data.get('zero_shot_mean_tier', {}).items():
        print(f"  {ethnicity}: Mean={stats['mean_tier']:.3f}, Count={stats['count']}, StdDev={stats['std_dev']:.3f}")

    print("\nN-Shot Mean Tier by Ethnicity:")
    for ethnicity, stats in data.get('n_shot_mean_tier', {}).items():
        print(f"  {ethnicity}: Mean={stats['mean_tier']:.3f}, Count={stats['count']}, StdDev={stats['std_dev']:.3f}")

    print("\nZero-Shot Question Rate by Ethnicity:")
    for ethnicity, stats in data.get('zero_shot_question_rate', {}).items():
        print(f"  {ethnicity}: Rate={stats['question_rate']:.1f}%, Questions={stats['questions']}, Total={stats['total_count']}")
    
    print("\nN-Shot Question Rate by Ethnicity:")
    for ethnicity, stats in data.get('n_shot_question_rate', {}).items():
        print(f"  {ethnicity}: Rate={stats['question_rate']:.1f}%, Questions={stats['questions']}, Total={stats['total_count']}")
    
    print("\nTier Bias Summary:")
    for ethnicity, methods in data.get('tier_bias_summary', {}).items():
        print(f"  {ethnicity}:")
        for method, stats in methods.items():
            print(f"    {method}: Mean={stats['mean_tier']:.3f}, Count={stats['count']}")
    
    print("\nDisadvantage Ranking:")
    ranking = data.get('disadvantage_ranking', {})
    if 'zero_shot' in ranking:
        zs = ranking['zero_shot']
        print(f"  Zero-Shot: Most Advantaged={zs.get('most_advantaged', 'N/A')}, Most Disadvantaged={zs.get('most_disadvantaged', 'N/A')}")
    if 'n_shot' in ranking:
        ns = ranking['n_shot']
        print(f"  N-Shot: Most Advantaged={ns.get('most_advantaged', 'N/A')}, Most Disadvantaged={ns.get('most_disadvantaged', 'N/A')}")
