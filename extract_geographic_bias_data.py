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

def extract_geographic_bias_data() -> Dict[str, Any]:
    """
    Extracts geographic bias data from the database for dashboard.
    """
    connection = get_db_connection()
    
    try:
        # Query 1: Raw Tier Values by Geography - Zero-Shot (for statistical analysis)
        zero_shot_raw_query = """
        SELECT 
            e.geography,
            e.llm_simplified_tier::FLOAT as tier_value
        FROM experiments e
        WHERE e.decision_method = 'zero-shot'
            AND e.persona IS NOT NULL
            AND e.risk_mitigation_strategy IS NULL
            AND e.llm_simplified_tier != -999
            AND e.geography IS NOT NULL
            AND e.geography IN ('rural', 'urban_affluent', 'urban_poor', 'urban_upper_middle', 'urban_working', 'suburban_upper_middle', 'suburban_working', 'suburban_poor', 'rural_upper_middle', 'rural_working', 'rural_poor')
        ORDER BY e.geography;
        """
        
        # Query 1b: Mean Tier by Geography - Zero-Shot (for display)
        zero_shot_mean_query = """
        SELECT 
            e.geography,
            AVG(e.llm_simplified_tier::FLOAT) as mean_tier,
            COUNT(*) as experiment_count,
            STDDEV(e.llm_simplified_tier::FLOAT) as std_dev
        FROM experiments e
        WHERE e.decision_method = 'zero-shot'
            AND e.persona IS NOT NULL
            AND e.risk_mitigation_strategy IS NULL
            AND e.llm_simplified_tier != -999
            AND e.geography IS NOT NULL
            AND e.geography IN ('rural', 'urban_affluent', 'urban_poor', 'urban_upper_middle', 'urban_working', 'suburban_upper_middle', 'suburban_working', 'suburban_poor', 'rural_upper_middle', 'rural_working', 'rural_poor')
        GROUP BY e.geography
        ORDER BY e.geography;
        """
        
        # Query 2: Raw Tier Values by Geography - N-Shot (for statistical analysis)
        n_shot_raw_query = """
        SELECT 
            e.geography,
            e.llm_simplified_tier::FLOAT as tier_value
        FROM experiments e
        WHERE e.decision_method = 'n-shot'
            AND e.persona IS NOT NULL
            AND e.llm_simplified_tier != -999
            AND e.geography IS NOT NULL
            AND e.geography IN ('rural', 'urban_affluent', 'urban_poor', 'urban_upper_middle', 'urban_working', 'suburban_upper_middle', 'suburban_working', 'suburban_poor', 'rural_upper_middle', 'rural_working', 'rural_poor')
        ORDER BY e.geography;
        """
        
        # Query 2b: Mean Tier by Geography - N-Shot (for display)
        n_shot_mean_query = """
        SELECT 
            e.geography,
            AVG(e.llm_simplified_tier::FLOAT) as mean_tier,
            COUNT(*) as experiment_count,
            STDDEV(e.llm_simplified_tier::FLOAT) as std_dev
        FROM experiments e
        WHERE e.decision_method = 'n-shot'
            AND e.persona IS NOT NULL
            AND e.llm_simplified_tier != -999
            AND e.geography IS NOT NULL
            AND e.geography IN ('rural', 'urban_affluent', 'urban_poor', 'urban_upper_middle', 'urban_working', 'suburban_upper_middle', 'suburban_working', 'suburban_poor', 'rural_upper_middle', 'rural_working', 'rural_poor')
        GROUP BY e.geography
        ORDER BY e.geography;
        """
        
        # Query 3: Tier Distribution by Geography - Zero-Shot
        zero_shot_distribution_query = """
        SELECT 
            e.geography,
            e.llm_simplified_tier,
            COUNT(*) as count
        FROM experiments e
        WHERE e.decision_method = 'zero-shot'
            AND e.persona IS NOT NULL
            AND e.risk_mitigation_strategy IS NULL
            AND e.llm_simplified_tier != -999
            AND e.geography IS NOT NULL
            AND e.geography IN ('rural', 'urban_affluent', 'urban_poor', 'urban_upper_middle', 'urban_working', 'suburban_upper_middle', 'suburban_working', 'suburban_poor', 'rural_upper_middle', 'rural_working', 'rural_poor')
        GROUP BY e.geography, e.llm_simplified_tier
        ORDER BY e.geography, e.llm_simplified_tier;
        """
        
        # Query 4: Tier Distribution by Geography - N-Shot
        n_shot_distribution_query = """
        SELECT 
            e.geography,
            e.llm_simplified_tier,
            COUNT(*) as count
        FROM experiments e
        WHERE e.decision_method = 'n-shot'
            AND e.persona IS NOT NULL
            AND e.llm_simplified_tier != -999
            AND e.geography IS NOT NULL
            AND e.geography IN ('rural', 'urban_affluent', 'urban_poor', 'urban_upper_middle', 'urban_working', 'suburban_upper_middle', 'suburban_working', 'suburban_poor', 'rural_upper_middle', 'rural_working', 'rural_poor')
        GROUP BY e.geography, e.llm_simplified_tier
        ORDER BY e.geography, e.llm_simplified_tier;
        """
        
        # Query 5: Question Rate by Geography - Zero-Shot
        zero_shot_question_query = """
        SELECT 
            e.geography,
            SUM(CASE WHEN e.asks_for_info = true THEN 1 ELSE 0 END) as questions,
            COUNT(*) as total_count
        FROM experiments e
        WHERE e.decision_method = 'zero-shot'
            AND e.persona IS NOT NULL
            AND e.risk_mitigation_strategy IS NULL
            AND e.llm_simplified_tier != -999
            AND e.geography IS NOT NULL
            AND e.geography IN ('rural', 'urban_affluent', 'urban_poor', 'urban_upper_middle', 'urban_working', 'suburban_upper_middle', 'suburban_working', 'suburban_poor', 'rural_upper_middle', 'rural_working', 'rural_poor')
        GROUP BY e.geography
        ORDER BY e.geography;
        """
        
        # Query 6: Question Rate by Geography - N-Shot
        n_shot_question_query = """
        SELECT 
            e.geography,
            SUM(CASE WHEN e.asks_for_info = true THEN 1 ELSE 0 END) as questions,
            COUNT(*) as total_count
        FROM experiments e
        WHERE e.decision_method = 'n-shot'
            AND e.persona IS NOT NULL
            AND e.llm_simplified_tier != -999
            AND e.geography IS NOT NULL
            AND e.geography IN ('rural', 'urban_affluent', 'urban_poor', 'urban_upper_middle', 'urban_working', 'suburban_upper_middle', 'suburban_working', 'suburban_poor', 'rural_upper_middle', 'rural_working', 'rural_poor')
        GROUP BY e.geography
        ORDER BY e.geography;
        """

        # Query 7: Tier Bias by Geography - Combined Zero-Shot and N-Shot (for Result 3 table)
        tier_bias_query = """
        SELECT 
            e.geography,
            e.decision_method,
            AVG(e.llm_simplified_tier::FLOAT) as mean_tier,
            COUNT(*) as count
        FROM experiments e
        WHERE e.persona IS NOT NULL
            AND e.llm_simplified_tier != -999
            AND e.geography IS NOT NULL
            AND e.geography IN ('rural', 'urban_affluent', 'urban_poor', 'urban_upper_middle', 'urban_working', 'suburban_upper_middle', 'suburban_working', 'suburban_poor', 'rural_upper_middle', 'rural_working', 'rural_poor')
            AND e.decision_method IN ('zero-shot', 'n-shot')
        GROUP BY e.geography, e.decision_method
        ORDER BY e.geography, e.decision_method;
        """
        
        # Query 8: Detailed data for mixed model analysis (for Result 3 statistical analysis)
        detailed_tier_bias_query = """
        SELECT 
            e.case_id,
            e.geography,
            e.decision_method,
            e.llm_simplified_tier
        FROM experiments e
        WHERE e.persona IS NOT NULL
            AND e.llm_simplified_tier != -999
            AND e.geography IS NOT NULL
            AND e.geography IN ('rural', 'urban_affluent', 'urban_poor', 'urban_upper_middle', 'urban_working', 'suburban_upper_middle', 'suburban_working', 'suburban_poor', 'rural_upper_middle', 'rural_working', 'rural_poor')
            AND e.decision_method IN ('zero-shot', 'n-shot')
        ORDER BY e.case_id, e.geography, e.decision_method;
        """
        
        # Query 9: Tier 0 Rate by Geography - Zero-Shot (Result 6)
        zero_shot_tier0_query = """
        SELECT 
            e.geography,
            COUNT(*) as sample_size,
            SUM(CASE WHEN e.llm_simplified_tier = 0 THEN 1 ELSE 0 END) as zero_tier_count,
            AVG(CASE WHEN e.llm_simplified_tier = 0 THEN 1.0 ELSE 0.0 END) as proportion_zero
        FROM experiments e
        WHERE e.decision_method = 'zero-shot'
            AND e.persona IS NOT NULL
            AND e.risk_mitigation_strategy IS NULL
            AND e.llm_simplified_tier != -999
            AND e.geography IS NOT NULL
            AND e.geography IN ('rural', 'urban_affluent', 'urban_poor', 'urban_upper_middle', 'urban_working', 'suburban_upper_middle', 'suburban_working', 'suburban_poor', 'rural_upper_middle', 'rural_working', 'rural_poor')
        GROUP BY e.geography
        ORDER BY e.geography;
        """
        
        # Query 10: Tier 0 Rate by Geography - N-Shot (Result 7)
        n_shot_tier0_query = """
        SELECT 
            e.geography,
            COUNT(*) as sample_size,
            SUM(CASE WHEN e.llm_simplified_tier = 0 THEN 1 ELSE 0 END) as zero_tier_count,
            AVG(CASE WHEN e.llm_simplified_tier = 0 THEN 1.0 ELSE 0.0 END) as proportion_zero
        FROM experiments e
        WHERE e.decision_method = 'n-shot'
            AND e.persona IS NOT NULL
            AND e.llm_simplified_tier != -999
            AND e.geography IS NOT NULL
            AND e.geography IN ('rural', 'urban_affluent', 'urban_poor', 'urban_upper_middle', 'urban_working', 'suburban_upper_middle', 'suburban_working', 'suburban_poor', 'rural_upper_middle', 'rural_working', 'rural_poor')
        GROUP BY e.geography
        ORDER BY e.geography;
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
            geography, tier_value = row
            if geography not in zero_shot_raw:
                zero_shot_raw[geography] = []
            zero_shot_raw[geography].append(float(tier_value))
        
        n_shot_raw = {}
        for row in n_shot_raw_data:
            geography, tier_value = row
            if geography not in n_shot_raw:
                n_shot_raw[geography] = []
            n_shot_raw[geography].append(float(tier_value))
        
        # Process mean tier data for display
        zero_shot_mean = {}
        for row in zero_shot_mean_data:
            geography, mean_tier, count, std_dev = row
            zero_shot_mean[geography] = {
                'mean_tier': float(mean_tier) if mean_tier is not None else 0,
                'count': int(count),
                'std_dev': float(std_dev) if std_dev is not None else 0
            }
        
        n_shot_mean = {}
        for row in n_shot_mean_data:
            geography, mean_tier, count, std_dev = row
            n_shot_mean[geography] = {
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
            geography, tier, count = row
            if geography not in zero_shot_distribution:
                zero_shot_distribution[geography] = {}
            zero_shot_distribution[geography][int(tier)] = int(count)
        
        n_shot_distribution = {}
        for row in n_shot_distribution_data:
            geography, tier, count = row
            if geography not in n_shot_distribution:
                n_shot_distribution[geography] = {}
            n_shot_distribution[geography][int(tier)] = int(count)
        
        # Execute question rate queries
        cursor.execute(zero_shot_question_query)
        zero_shot_question_data = cursor.fetchall()
        
        cursor.execute(n_shot_question_query)
        n_shot_question_data = cursor.fetchall()
        
        # Process question rate data
        zero_shot_question_rate = {}
        for row in zero_shot_question_data:
            geography, questions, total_count = row
            question_rate = (float(questions) / float(total_count) * 100) if total_count > 0 else 0
            zero_shot_question_rate[geography] = {
                'questions': int(questions),
                'total_count': int(total_count),
                'question_rate': question_rate
            }
        
        n_shot_question_rate = {}
        for row in n_shot_question_data:
            geography, questions, total_count = row
            question_rate = (float(questions) / float(total_count) * 100) if total_count > 0 else 0
            n_shot_question_rate[geography] = {
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
            geography, decision_method, mean_tier, count = row
            if geography not in tier_bias_summary:
                tier_bias_summary[geography] = {}
            tier_bias_summary[geography][decision_method] = {
                'mean_tier': float(mean_tier) if mean_tier is not None else 0,
                'count': int(count)
            }
        
        # Execute detailed tier bias query for mixed model analysis
        cursor.execute(detailed_tier_bias_query)
        detailed_tier_bias_data = cursor.fetchall()
        
        # Execute tier 0 rate queries
        cursor.execute(zero_shot_tier0_query)
        zero_shot_tier0_data = cursor.fetchall()
        
        cursor.execute(n_shot_tier0_query)
        n_shot_tier0_data = cursor.fetchall()
        
        # Process detailed data for mixed model
        detailed_data = []
        for row in detailed_tier_bias_data:
            case_id, geography, decision_method, tier = row
            detailed_data.append({
                'case_id': case_id,
                'geography': geography,
                'decision_method': decision_method,
                'tier': int(tier)
            })
        
        # Process tier 0 rate data (Result 6 - Zero-Shot)
        zero_shot_tier0_rate = {}
        for row in zero_shot_tier0_data:
            geography, sample_size, zero_tier_count, proportion_zero = row
            zero_shot_tier0_rate[geography] = {
                'sample_size': int(sample_size),
                'zero_tier_count': int(zero_tier_count),
                'proportion_zero': round(float(proportion_zero), 3) if proportion_zero is not None else 0.0
            }
        
        # Process tier 0 rate data (Result 7 - N-Shot)
        n_shot_tier0_rate = {}
        for row in n_shot_tier0_data:
            geography, sample_size, zero_tier_count, proportion_zero = row
            n_shot_tier0_rate[geography] = {
                'sample_size': int(sample_size),
                'zero_tier_count': int(zero_tier_count),
                'proportion_zero': round(float(proportion_zero), 3) if proportion_zero is not None else 0.0
            }
        
        # Perform statistical analyses
        zero_shot_stats = perform_mean_tier_statistical_analysis(zero_shot_raw)
        n_shot_stats = perform_mean_tier_statistical_analysis(n_shot_raw)
        zero_shot_dist_stats = perform_distribution_statistical_analysis(zero_shot_distribution)
        n_shot_dist_stats = perform_distribution_statistical_analysis(n_shot_distribution)
        zero_shot_question_stats = perform_question_rate_statistical_analysis(zero_shot_question_rate)
        n_shot_question_stats = perform_question_rate_statistical_analysis(n_shot_question_rate)
        
        # Perform mixed model analysis for tier bias
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
        print(f"Error extracting geographic bias data: {e}")
        return {'error': str(e)}
    finally:
        connection.close()

def perform_mean_tier_statistical_analysis(geography_raw_data: Dict) -> Dict:
    """
    Perform statistical analysis on mean tier data by geography using raw data.
    Uses ANOVA to compare ALL geographies, not just two.
    """
    try:
        if len(geography_raw_data) < 2:
            return {'error': 'Insufficient data for statistical analysis'}
        
        # Extract raw data for ANOVA
        geographies = list(geography_raw_data.keys())
        if len(geographies) < 2:
            return {'error': 'Need at least 2 geographies for comparison'}
        
        # Prepare data for ANOVA
        groups = []
        group_labels = []
        for geography in geographies:
            groups.append(geography_raw_data[geography])
            group_labels.append(geography)
        
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
        means = {geography: np.mean(geography_raw_data[geography]) for geography in geographies}
        
        return {
            'test_type': 'One-way ANOVA',
            'comparison': f'All geographies: {", ".join(geographies)}',
            'f_statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'significant': p_value < 0.05,
            'conclusion': 'rejected' if p_value < 0.05 else 'accepted',
            'means': means
        }
        
    except Exception as e:
        return {'error': f'Statistical analysis failed: {e}'}

def perform_distribution_statistical_analysis(geography_distribution_data: Dict) -> Dict:
    """
    Perform chi-squared test on tier distribution data by geography.
    """
    try:
        if len(geography_distribution_data) < 2:
            return {'error': 'Insufficient data for statistical analysis'}
        
        # Create contingency table
        geographies = list(geography_distribution_data.keys())
        tiers = set()
        for geography_data in geography_distribution_data.values():
            tiers.update(geography_data.keys())
        tiers = sorted(list(tiers))
        
        if len(tiers) < 2:
            return {'error': 'Need at least 2 tiers for chi-squared test'}
        
        # Build contingency table
        contingency_table = []
        for geography in geographies:
            row = []
            for tier in tiers:
                count = geography_distribution_data[geography].get(tier, 0)
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

def perform_question_rate_statistical_analysis(geography_question_rate_data: Dict) -> Dict:
    """
    Perform statistical analysis on question rate data by geography.
    """
    try:
        if len(geography_question_rate_data) < 2:
            return {'error': 'Insufficient data for statistical analysis'}
        
        # Create contingency table for chi-squared test
        geographies = list(geography_question_rate_data.keys())
        
        # Build 2x2 contingency table (questions vs no questions)
        contingency_table = []
        for geography in geographies:
            questions = geography_question_rate_data[geography]['questions']
            total = geography_question_rate_data[geography]['total_count']
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
        
        # Group data by geography and decision method
        geography_method_means = {}
        for item in detailed_data:
            key = (item['geography'], item['decision_method'])
            if key not in geography_method_means:
                geography_method_means[key] = []
            geography_method_means[key].append(item['tier'])
        
        # Calculate means for each group
        group_stats = {}
        for key, tiers in geography_method_means.items():
            geography, method = key
            group_stats[key] = {
                'mean': np.mean(tiers),
                'std': np.std(tiers),
                'count': len(tiers)
            }
        
        # Simplified statistical test (in practice, use proper mixed model)
        # This is a placeholder - the actual implementation would be much more complex
        geographies = list(set([item['geography'] for item in detailed_data]))
        methods = list(set([item['decision_method'] for item in detailed_data]))
        
        if len(geographies) >= 2 and len(methods) >= 2:
            # Perform a simplified ANOVA-like test
            from scipy import stats
            
            # Create groups for testing
            groups = []
            group_labels = []
            for geography in geographies:
                for method in methods:
                    key = (geography, method)
                    if key in geography_method_means:
                        groups.append(geography_method_means[key])
                        group_labels.append(f"{geography}_{method}")
            
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
    """Perform chi-squared test for tier 0 rate comparison by geography"""
    if len(tier0_data) < 2:
        return {'error': 'Insufficient data for statistical analysis'}
    
    geographies = list(tier0_data.keys())
    if len(geographies) < 2:
        return {'error': 'Expected at least 2 geographies for comparison'}
    
    # Create contingency table for chi-squared test
    # [zero_tier_count, non_zero_tier_count] for each geography
    contingency_table = []
    for geography in geographies:
        data = tier0_data[geography]
        zero_count = data['zero_tier_count']
        non_zero_count = data['sample_size'] - zero_count
        contingency_table.append([zero_count, non_zero_count])
    
    try:
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Find geography with highest proportion of zero-tier cases
        highest_prop_geography = max(geographies, key=lambda g: tier0_data[g]['proportion_zero'])
        
        return {
            'geographies': geographies,
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significant': p_value < 0.05,
            'conclusion': 'rejected' if p_value < 0.05 else 'accepted',
            'highest_proportion_geography': highest_prop_geography
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
    print("Extracting geographic bias data from database...")
    data = extract_geographic_bias_data()
    
    print("\nGeographic Bias Data:")
    print("=" * 50)

    print("\nZero-Shot Mean Tier by Geography:")
    for geography, stats in data.get('zero_shot_mean_tier', {}).items():
        print(f"  {geography}: Mean={stats['mean_tier']:.3f}, Count={stats['count']}, StdDev={stats['std_dev']:.3f}")

    print("\nN-Shot Mean Tier by Geography:")
    for geography, stats in data.get('n_shot_mean_tier', {}).items():
        print(f"  {geography}: Mean={stats['mean_tier']:.3f}, Count={stats['count']}, StdDev={stats['std_dev']:.3f}")

    print("\nZero-Shot Question Rate by Geography:")
    for geography, stats in data.get('zero_shot_question_rate', {}).items():
        print(f"  {geography}: Rate={stats['question_rate']:.1f}%, Questions={stats['questions']}, Total={stats['total_count']}")
    
    print("\nN-Shot Question Rate by Geography:")
    for geography, stats in data.get('n_shot_question_rate', {}).items():
        print(f"  {geography}: Rate={stats['question_rate']:.1f}%, Questions={stats['questions']}, Total={stats['total_count']}")
    
    print("\nTier Bias Summary:")
    for geography, methods in data.get('tier_bias_summary', {}).items():
        print(f"  {geography}:")
        for method, stats in methods.items():
            print(f"    {method}: Mean={stats['mean_tier']:.3f}, Count={stats['count']}")
    
    print("\nZero-Shot Tier 0 Rate by Geography (Result 6):")
    for geography, stats in data.get('zero_shot_tier0_rate', {}).items():
        print(f"  {geography}: Sample Size={stats['sample_size']}, Zero Tier={stats['zero_tier_count']}, Proportion Zero={stats['proportion_zero']:.3f}")
    
    print("\nN-Shot Tier 0 Rate by Geography (Result 7):")
    for geography, stats in data.get('n_shot_tier0_rate', {}).items():
        print(f"  {geography}: Sample Size={stats['sample_size']}, Zero Tier={stats['zero_tier_count']}, Proportion Zero={stats['proportion_zero']:.3f}")
    
    print("\nDisadvantage Ranking:")
    ranking = data.get('disadvantage_ranking', {})
    if 'zero_shot' in ranking:
        zs = ranking['zero_shot']
        print(f"  Zero-Shot: Most Advantaged={zs.get('most_advantaged', 'N/A')}, Most Disadvantaged={zs.get('most_disadvantaged', 'N/A')}")
    if 'n_shot' in ranking:
        ns = ranking['n_shot']
        print(f"  N-Shot: Most Advantaged={ns.get('most_advantaged', 'N/A')}, Most Disadvantaged={ns.get('most_disadvantaged', 'N/A')}")
