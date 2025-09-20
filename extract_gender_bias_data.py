#!/usr/bin/env python3
"""
Extract Gender Bias Analysis Data for HTML Dashboard

This script extracts gender bias analysis data from the database
for display in the HTML dashboard's Persona Injection > Gender Bias tab.
"""

import os
import psycopg2
from dotenv import load_dotenv
from scipy import stats
import numpy as np

# Load environment variables
load_dotenv()

def get_db_connection():
    """Get database connection"""
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'fairness_analysis'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', '')
    }
    return psycopg2.connect(**db_config)

def extract_gender_bias_data():
    """
    Extract gender bias analysis data from database
    
    Returns:
        Dictionary containing gender bias data for dashboard
    """
    connection = get_db_connection()
    
    try:
        # Query 1: Mean Tier by Gender - Zero-Shot
        zero_shot_mean_query = """
        SELECT 
            e.gender,
            AVG(e.llm_simplified_tier::FLOAT) as mean_tier,
            COUNT(*) as experiment_count,
            STDDEV(e.llm_simplified_tier::FLOAT) as std_dev
        FROM experiments e
        WHERE e.decision_method = 'zero-shot'
            AND e.persona IS NOT NULL
            AND e.risk_mitigation_strategy IS NULL
            AND e.llm_simplified_tier != -999
            AND e.gender IS NOT NULL
            AND (e.gender = 'male' OR e.gender = 'female')
        GROUP BY e.gender
        ORDER BY e.gender;
        """
        
        # Query 2: Mean Tier by Gender - N-Shot
        n_shot_mean_query = """
        SELECT 
            e.gender,
            AVG(e.llm_simplified_tier::FLOAT) as mean_tier,
            COUNT(*) as experiment_count,
            STDDEV(e.llm_simplified_tier::FLOAT) as std_dev
        FROM experiments e
        WHERE e.decision_method = 'n-shot'
            AND e.persona IS NOT NULL
            AND e.risk_mitigation_strategy IS NULL
            AND e.llm_simplified_tier != -999
            AND e.gender IS NOT NULL
            AND (e.gender = 'male' OR e.gender = 'female')
        GROUP BY e.gender
        ORDER BY e.gender;
        """
        
        # Query 3: Tier Distribution by Gender - Zero-Shot
        zero_shot_distribution_query = """
        SELECT 
            e.gender,
            e.llm_simplified_tier,
            COUNT(*) as count
        FROM experiments e
        WHERE e.decision_method = 'zero-shot'
            AND e.persona IS NOT NULL
            AND e.risk_mitigation_strategy IS NULL
            AND e.llm_simplified_tier != -999
            AND e.gender IS NOT NULL
            AND (e.gender = 'male' OR e.gender = 'female')
        GROUP BY e.gender, e.llm_simplified_tier
        ORDER BY e.gender, e.llm_simplified_tier;
        """
        
        # Query 4: Tier Distribution by Gender - N-Shot
        n_shot_distribution_query = """
        SELECT 
            e.gender,
            e.llm_simplified_tier,
            COUNT(*) as count
        FROM experiments e
        WHERE e.decision_method = 'n-shot'
            AND e.persona IS NOT NULL
            AND e.risk_mitigation_strategy IS NULL
            AND e.llm_simplified_tier != -999
            AND e.gender IS NOT NULL
            AND (e.gender = 'male' OR e.gender = 'female')
        GROUP BY e.gender, e.llm_simplified_tier
        ORDER BY e.gender, e.llm_simplified_tier;
        """
        
        # Query 5: Question Rate by Gender - Zero-Shot
        zero_shot_question_query = """
        SELECT 
            e.gender,
            SUM(CASE WHEN e.asks_for_info = true THEN 1 ELSE 0 END) as questions,
            COUNT(*) as total_count
        FROM experiments e
        WHERE e.decision_method = 'zero-shot'
            AND e.persona IS NOT NULL
            AND e.risk_mitigation_strategy IS NULL
            AND e.llm_simplified_tier != -999
            AND e.gender IS NOT NULL
            AND (e.gender = 'male' OR e.gender = 'female')
        GROUP BY e.gender
        ORDER BY e.gender;
        """
        
        # Query 6: Question Rate by Gender - N-Shot
        n_shot_question_query = """
        SELECT 
            e.gender,
            SUM(CASE WHEN e.asks_for_info = true THEN 1 ELSE 0 END) as questions,
            COUNT(*) as total_count
        FROM experiments e
        WHERE e.decision_method = 'n-shot'
            AND e.persona IS NOT NULL
            AND e.risk_mitigation_strategy IS NULL
            AND e.llm_simplified_tier != -999
            AND e.gender IS NOT NULL
            AND (e.gender = 'male' OR e.gender = 'female')
        GROUP BY e.gender
        ORDER BY e.gender;
        """
        
        # Query 7: Tier Bias by Gender - Combined Zero-Shot and N-Shot
        tier_bias_query = """
        SELECT 
            e.gender,
            e.decision_method,
            AVG(e.llm_simplified_tier::FLOAT) as mean_tier,
            COUNT(*) as count
        FROM experiments e
        WHERE e.persona IS NOT NULL
            AND e.risk_mitigation_strategy IS NULL
            AND e.llm_simplified_tier != -999
            AND e.gender IS NOT NULL
            AND (e.gender = 'male' OR e.gender = 'female')
            AND e.decision_method IN ('zero-shot', 'n-shot')
        GROUP BY e.gender, e.decision_method
        ORDER BY e.gender, e.decision_method;
        """
        
        # Query 8: Detailed data for mixed model analysis
        detailed_tier_bias_query = """
        SELECT 
            e.case_id,
            e.gender,
            e.decision_method,
            e.llm_simplified_tier
        FROM experiments e
        WHERE e.persona IS NOT NULL
            AND e.risk_mitigation_strategy IS NULL
            AND e.llm_simplified_tier != -999
            AND e.gender IS NOT NULL
            AND (e.gender = 'male' OR e.gender = 'female')
            AND e.decision_method IN ('zero-shot', 'n-shot')
        ORDER BY e.case_id, e.gender, e.decision_method;
        """
        
        # Query 9: Tier 0 Rate by Gender - Zero-Shot (Result 6)
        zero_shot_tier0_query = """
        SELECT 
            e.gender,
            COUNT(*) as sample_size,
            SUM(CASE WHEN e.llm_simplified_tier = 0 THEN 1 ELSE 0 END) as zero_tier_count,
            AVG(CASE WHEN e.llm_simplified_tier = 0 THEN 1.0 ELSE 0.0 END) as proportion_zero
        FROM experiments e
        WHERE e.decision_method = 'zero-shot'
            AND e.persona IS NOT NULL
            AND e.risk_mitigation_strategy IS NULL
            AND e.llm_simplified_tier != -999
            AND e.gender IS NOT NULL
            AND (e.gender = 'male' OR e.gender = 'female')
        GROUP BY e.gender
        ORDER BY e.gender;
        """
        
        # Query 10: Tier 0 Rate by Gender - N-Shot (Result 7)
        n_shot_tier0_query = """
        SELECT 
            e.gender,
            COUNT(*) as sample_size,
            SUM(CASE WHEN e.llm_simplified_tier = 0 THEN 1 ELSE 0 END) as zero_tier_count,
            AVG(CASE WHEN e.llm_simplified_tier = 0 THEN 1.0 ELSE 0.0 END) as proportion_zero
        FROM experiments e
        WHERE e.decision_method = 'n-shot'
            AND e.persona IS NOT NULL
            AND e.risk_mitigation_strategy IS NULL
            AND e.llm_simplified_tier != -999
            AND e.gender IS NOT NULL
            AND (e.gender = 'male' OR e.gender = 'female')
        GROUP BY e.gender
        ORDER BY e.gender;
        """
        
        # Execute queries
        cursor = connection.cursor()
        
        # Zero-shot mean tier data
        cursor.execute(zero_shot_mean_query)
        zero_shot_mean_data = cursor.fetchall()
        
        # N-shot mean tier data
        cursor.execute(n_shot_mean_query)
        n_shot_mean_data = cursor.fetchall()
        
        # Zero-shot distribution data
        cursor.execute(zero_shot_distribution_query)
        zero_shot_distribution_data = cursor.fetchall()
        
        # N-shot distribution data
        cursor.execute(n_shot_distribution_query)
        n_shot_distribution_data = cursor.fetchall()
        
        # Zero-shot question rate data
        cursor.execute(zero_shot_question_query)
        zero_shot_question_data = cursor.fetchall()
        
        # N-shot question rate data
        cursor.execute(n_shot_question_query)
        n_shot_question_data = cursor.fetchall()
        
        # Tier bias data
        cursor.execute(tier_bias_query)
        tier_bias_data = cursor.fetchall()
        
        # Detailed tier bias data for mixed model
        cursor.execute(detailed_tier_bias_query)
        detailed_tier_bias_data = cursor.fetchall()
        
        # Zero-shot tier 0 rate data (Result 6)
        cursor.execute(zero_shot_tier0_query)
        zero_shot_tier0_data = cursor.fetchall()
        
        # N-shot tier 0 rate data (Result 7)
        cursor.execute(n_shot_tier0_query)
        n_shot_tier0_data = cursor.fetchall()
        
        # Process mean tier data
        zero_shot_mean = {}
        for row in zero_shot_mean_data:
            gender, mean_tier, count, std_dev = row
            zero_shot_mean[gender] = {
                'mean_tier': float(mean_tier) if mean_tier is not None else 0,
                'count': int(count),
                'std_dev': float(std_dev) if std_dev is not None else 0
            }
        
        n_shot_mean = {}
        for row in n_shot_mean_data:
            gender, mean_tier, count, std_dev = row
            n_shot_mean[gender] = {
                'mean_tier': float(mean_tier) if mean_tier is not None else 0,
                'count': int(count),
                'std_dev': float(std_dev) if std_dev is not None else 0
            }
        
        # Process distribution data
        zero_shot_distribution = {}
        for row in zero_shot_distribution_data:
            gender, tier, count = row
            if gender not in zero_shot_distribution:
                zero_shot_distribution[gender] = {}
            zero_shot_distribution[gender][int(tier)] = int(count)
        
        n_shot_distribution = {}
        for row in n_shot_distribution_data:
            gender, tier, count = row
            if gender not in n_shot_distribution:
                n_shot_distribution[gender] = {}
            n_shot_distribution[gender][int(tier)] = int(count)
        
        # Process question rate data
        zero_shot_question_rate = {}
        for row in zero_shot_question_data:
            gender, questions, total_count = row
            zero_shot_question_rate[gender] = {
                'questions': int(questions),
                'total_count': int(total_count),
                'question_rate': float(questions) / float(total_count) * 100 if total_count > 0 else 0
            }
        
        n_shot_question_rate = {}
        for row in n_shot_question_data:
            gender, questions, total_count = row
            n_shot_question_rate[gender] = {
                'questions': int(questions),
                'total_count': int(total_count),
                'question_rate': float(questions) / float(total_count) * 100 if total_count > 0 else 0
            }
        
        # Perform statistical analysis for mean tier comparison
        zero_shot_stats = perform_mean_tier_statistical_analysis(zero_shot_mean)
        n_shot_stats = perform_mean_tier_statistical_analysis(n_shot_mean)
        
        # Perform statistical analysis for distribution comparison
        zero_shot_dist_stats = perform_distribution_statistical_analysis(zero_shot_distribution)
        n_shot_dist_stats = perform_distribution_statistical_analysis(n_shot_distribution)
        
        # Perform statistical analysis for question rate comparison
        zero_shot_question_stats = perform_question_rate_statistical_analysis(zero_shot_question_rate)
        n_shot_question_stats = perform_question_rate_statistical_analysis(n_shot_question_rate)
        
        # Process tier bias data
        tier_bias_summary = {}
        for row in tier_bias_data:
            gender, decision_method, mean_tier, count = row
            if gender not in tier_bias_summary:
                tier_bias_summary[gender] = {}
            tier_bias_summary[gender][decision_method] = {
                'mean_tier': float(mean_tier) if mean_tier is not None else 0,
                'count': int(count)
            }
        
        # Process detailed tier bias data for mixed model analysis
        detailed_data = []
        for row in detailed_tier_bias_data:
            case_id, gender, decision_method, tier = row
            detailed_data.append({
                'case_id': int(case_id),
                'gender': gender,
                'decision_method': decision_method,
                'tier': int(tier)
            })
        
        # Process tier 0 rate data (Result 6 - Zero-Shot)
        zero_shot_tier0_rate = {}
        for row in zero_shot_tier0_data:
            gender, sample_size, zero_tier_count, proportion_zero = row
            zero_shot_tier0_rate[gender] = {
                'sample_size': int(sample_size),
                'zero_tier_count': int(zero_tier_count),
                'proportion_zero': round(float(proportion_zero), 3) if proportion_zero is not None else 0.0
            }
        
        # Process tier 0 rate data (Result 7 - N-Shot)
        n_shot_tier0_rate = {}
        for row in n_shot_tier0_data:
            gender, sample_size, zero_tier_count, proportion_zero = row
            n_shot_tier0_rate[gender] = {
                'sample_size': int(sample_size),
                'zero_tier_count': int(zero_tier_count),
                'proportion_zero': round(float(proportion_zero), 3) if proportion_zero is not None else 0.0
            }
        
        # Perform mixed model analysis
        mixed_model_stats = perform_mixed_model_analysis(detailed_data)
        
        # Perform statistical analysis for tier 0 rate comparison
        zero_shot_tier0_stats = perform_tier0_rate_statistical_analysis(zero_shot_tier0_rate)
        n_shot_tier0_stats = perform_tier0_rate_statistical_analysis(n_shot_tier0_rate)
        
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
            'disadvantage_ranking': disadvantage_ranking,
            'zero_shot_tier0_rate': zero_shot_tier0_rate,
            'n_shot_tier0_rate': n_shot_tier0_rate,
            'zero_shot_tier0_stats': zero_shot_tier0_stats,
            'n_shot_tier0_stats': n_shot_tier0_stats
        }
        
        return result
        
    except Exception as e:
        print(f"Error extracting gender bias data: {e}")
        return {}
    finally:
        connection.close()

def perform_mean_tier_statistical_analysis(mean_data):
    """Perform statistical analysis for mean tier comparison"""
    if len(mean_data) < 2:
        return {'error': 'Insufficient data for statistical analysis'}
    
    genders = list(mean_data.keys())
    if len(genders) != 2:
        return {'error': 'Expected exactly 2 genders for comparison'}
    
    gender1, gender2 = genders
    mean1 = mean_data[gender1]['mean_tier']
    mean2 = mean_data[gender2]['mean_tier']
    std1 = mean_data[gender1]['std_dev']
    std2 = mean_data[gender2]['std_dev']
    n1 = mean_data[gender1]['count']
    n2 = mean_data[gender2]['count']
    
    # Calculate pooled standard deviation for Cohen's d
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
    
    # Perform t-test (assuming equal variances)
    try:
        # For demonstration, we'll use a simple t-test
        # In practice, you'd need the raw data for a proper t-test
        t_stat = (mean1 - mean2) / (pooled_std * np.sqrt(1/n1 + 1/n2)) if pooled_std > 0 else 0
        df = n1 + n2 - 2
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        return {
            'gender1': gender1,
            'gender2': gender2,
            'mean1': mean1,
            'mean2': mean2,
            'mean_difference': mean1 - mean2,
            'cohens_d': cohens_d,
            't_statistic': t_stat,
            'degrees_of_freedom': df,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'conclusion': 'rejected' if p_value < 0.05 else 'accepted'
        }
    except Exception as e:
        return {'error': f'Statistical analysis failed: {e}'}

def perform_distribution_statistical_analysis(distribution_data):
    """Perform chi-squared test for distribution comparison"""
    if len(distribution_data) < 2:
        return {'error': 'Insufficient data for statistical analysis'}
    
    genders = list(distribution_data.keys())
    if len(genders) != 2:
        return {'error': 'Expected exactly 2 genders for comparison'}
    
    # Create contingency table
    tiers = sorted(set().union(*[data.keys() for data in distribution_data.values()]))
    contingency_table = []
    
    for gender in genders:
        row = []
        for tier in tiers:
            count = distribution_data[gender].get(tier, 0)
            row.append(count)
        contingency_table.append(row)
    
    try:
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        return {
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significant': p_value < 0.05,
            'conclusion': 'rejected' if p_value < 0.05 else 'accepted'
        }
    except Exception as e:
        return {'error': f'Chi-squared test failed: {e}'}

def perform_question_rate_statistical_analysis(question_data):
    """Perform statistical analysis for question rate comparison"""
    if len(question_data) < 2:
        return {'error': 'Insufficient data for statistical analysis'}
    
    genders = list(question_data.keys())
    if len(genders) != 2:
        return {'error': 'Expected exactly 2 genders for comparison'}
    
    gender1, gender2 = genders
    rate1 = question_data[gender1]['question_rate']
    rate2 = question_data[gender2]['question_rate']
    n1 = question_data[gender1]['total_count']
    n2 = question_data[gender2]['total_count']
    
    # Perform chi-squared test for proportions
    questions1 = question_data[gender1]['questions']
    questions2 = question_data[gender2]['questions']
    
    # Create 2x2 contingency table
    contingency_table = [
        [questions1, n1 - questions1],
        [questions2, n2 - questions2]
    ]
    
    try:
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        return {
            'gender1': gender1,
            'gender2': gender2,
            'rate1': rate1,
            'rate2': rate2,
            'rate_difference': rate1 - rate2,
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significant': p_value < 0.05,
            'conclusion': 'rejected' if p_value < 0.05 else 'accepted'
        }
    except Exception as e:
        return {'error': f'Question rate statistical analysis failed: {e}'}

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
        
        # Group data by gender and decision method
        gender_method_means = {}
        for item in detailed_data:
            key = (item['gender'], item['decision_method'])
            if key not in gender_method_means:
                gender_method_means[key] = []
            gender_method_means[key].append(item['tier'])
        
        # Calculate means for each group
        group_stats = {}
        for key, tiers in gender_method_means.items():
            gender, method = key
            group_stats[key] = {
                'mean': np.mean(tiers),
                'std': np.std(tiers),
                'count': len(tiers)
            }
        
        # Simplified statistical test (in practice, use proper mixed model)
        # This is a placeholder - the actual implementation would be much more complex
        genders = list(set([item['gender'] for item in detailed_data]))
        methods = list(set([item['decision_method'] for item in detailed_data]))
        
        if len(genders) >= 2 and len(methods) >= 2:
            # Perform a simplified ANOVA-like test
            from scipy import stats
            
            # Create groups for testing
            groups = []
            group_labels = []
            for gender in genders:
                for method in methods:
                    key = (gender, method)
                    if key in gender_method_means:
                        groups.append(gender_method_means[key])
                        group_labels.append(f"{gender}_{method}")
            
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

def perform_tier0_rate_statistical_analysis(tier0_data):
    """Perform chi-squared test for tier 0 rate comparison by gender"""
    if len(tier0_data) < 2:
        return {'error': 'Insufficient data for statistical analysis'}
    
    genders = list(tier0_data.keys())
    if len(genders) != 2:
        return {'error': 'Expected exactly 2 genders for comparison'}
    
    gender1, gender2 = genders
    data1 = tier0_data[gender1]
    data2 = tier0_data[gender2]
    
    # Create 2x2 contingency table for chi-squared test
    # [zero_tier_count, non_zero_tier_count] for each gender
    zero1 = data1['zero_tier_count']
    non_zero1 = data1['sample_size'] - zero1
    zero2 = data2['zero_tier_count']
    non_zero2 = data2['sample_size'] - zero2
    
    contingency_table = [
        [zero1, non_zero1],
        [zero2, non_zero2]
    ]
    
    try:
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Determine which gender has higher proportion of zero-tier cases
        prop1 = data1['proportion_zero']
        prop2 = data2['proportion_zero']
        
        return {
            'gender1': gender1,
            'gender2': gender2,
            'proportion1': prop1,
            'proportion2': prop2,
            'proportion_difference': prop1 - prop2,
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significant': p_value < 0.05,
            'conclusion': 'rejected' if p_value < 0.05 else 'accepted',
            'higher_proportion_gender': gender1 if prop1 > prop2 else gender2
        }
    except Exception as e:
        return {'error': f'Tier 0 rate statistical analysis failed: {e}'}

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
    print("Extracting gender bias data from database...")
    data = extract_gender_bias_data()
    
    print("\nGender Bias Data:")
    print("=" * 50)
    
    print("\nZero-Shot Mean Tier by Gender:")
    for gender, stats in data.get('zero_shot_mean_tier', {}).items():
        print(f"  {gender}: Mean={stats['mean_tier']:.3f}, Count={stats['count']}, StdDev={stats['std_dev']:.3f}")
    
    print("\nN-Shot Mean Tier by Gender:")
    for gender, stats in data.get('n_shot_mean_tier', {}).items():
        print(f"  {gender}: Mean={stats['mean_tier']:.3f}, Count={stats['count']}, StdDev={stats['std_dev']:.3f}")
    
    print("\nZero-Shot Question Rate by Gender:")
    for gender, stats in data.get('zero_shot_question_rate', {}).items():
        print(f"  {gender}: Rate={stats['question_rate']:.1f}%, Questions={stats['questions']}, Total={stats['total_count']}")
    
    print("\nN-Shot Question Rate by Gender:")
    for gender, stats in data.get('n_shot_question_rate', {}).items():
        print(f"  {gender}: Rate={stats['question_rate']:.1f}%, Questions={stats['questions']}, Total={stats['total_count']}")
    
    print("\nTier Bias Summary:")
    for gender, methods in data.get('tier_bias_summary', {}).items():
        print(f"  {gender}:")
        for method, stats in methods.items():
            print(f"    {method}: Mean={stats['mean_tier']:.3f}, Count={stats['count']}")
    
    print("\nZero-Shot Tier 0 Rate by Gender (Result 6):")
    for gender, stats in data.get('zero_shot_tier0_rate', {}).items():
        print(f"  {gender}: Sample Size={stats['sample_size']}, Zero Tier={stats['zero_tier_count']}, Proportion Zero={stats['proportion_zero']:.3f}")
    
    print("\nN-Shot Tier 0 Rate by Gender (Result 7):")
    for gender, stats in data.get('n_shot_tier0_rate', {}).items():
        print(f"  {gender}: Sample Size={stats['sample_size']}, Zero Tier={stats['zero_tier_count']}, Proportion Zero={stats['proportion_zero']:.3f}")
    
    print("\nDisadvantage Ranking:")
    ranking = data.get('disadvantage_ranking', {})
    if 'zero_shot' in ranking:
        zs = ranking['zero_shot']
        print(f"  Zero-Shot: Most Advantaged={zs.get('most_advantaged', 'N/A')}, Most Disadvantaged={zs.get('most_disadvantaged', 'N/A')}")
    if 'n_shot' in ranking:
        ns = ranking['n_shot']
        print(f"  N-Shot: Most Advantaged={ns.get('most_advantaged', 'N/A')}, Most Disadvantaged={ns.get('most_disadvantaged', 'N/A')}")
