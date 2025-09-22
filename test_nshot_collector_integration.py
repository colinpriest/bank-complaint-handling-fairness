#!/usr/bin/env python3
"""
Test that the n-shot vs zero-shot disparity now gets properly escalated to
the "Statistically Significant and Material" tab
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from html_dashboard import HTMLDashboard, StatisticalResultCollector

def test_nshot_collector_integration():
    """Test that severe n-shot disparity gets added to material findings"""
    print("=== Testing N-Shot Collector Integration ===\n")

    # Initialize collector
    collector = StatisticalResultCollector()

    # Create dashboard with collector
    dashboard = HTMLDashboard()

    # Test data from the user's example (severe disparity)
    comparison_data = {
        'zero_shot_count': 10000,
        'zero_shot_questions': 113,
        'n_shot_count': 10000,
        'n_shot_questions': 5
    }

    print("1. Testing n-shot vs zero-shot table generation...")

    # Generate the table (this should now add to collector)
    table_html = dashboard._build_nshot_vs_zeroshot_table(comparison_data)

    print("   - Table generated successfully")

    # Check if result was added to collector
    print("\n2. Checking collector results...")

    material_results = dashboard.collector.results['material']
    trivial_results = dashboard.collector.results['trivial']

    print(f"   - Material results: {len(material_results)}")
    print(f"   - Trivial results: {len(trivial_results)}")

    # Look for our n-shot result
    nshot_result = None
    for result in material_results:
        if 'N-Shot vs Zero-Shot' in result.get('test_name', ''):
            nshot_result = result
            break

    if nshot_result:
        print("   [SUCCESS] N-Shot disparity found in MATERIAL results!")
        print(f"     - Test name: {nshot_result['test_name']}")
        print(f"     - P-value: {nshot_result['p_value']}")
        print(f"     - Effect size: {nshot_result['effect_size']:.3f}")
        print(f"     - Effect type: {nshot_result['effect_type']}")
        print(f"     - Finding: {nshot_result['finding']}")
        print(f"     - Implication: {nshot_result['implication']}")

        # Check metadata
        metadata = nshot_result.get('metadata', {})
        if metadata:
            print(f"     - Disparity ratio: {metadata.get('disparity_ratio', 'N/A'):.1f}x")
            print(f"     - Reduction: {metadata.get('reduction_percentage', 'N/A'):.0f}%")
            print(f"     - Severity: {metadata.get('severity', 'N/A')}")
    else:
        # Check if it's in trivial (it shouldn't be)
        nshot_trivial = None
        for result in trivial_results:
            if 'N-Shot vs Zero-Shot' in result.get('test_name', ''):
                nshot_trivial = result
                break

        if nshot_trivial:
            print("   [ERROR] N-Shot disparity incorrectly classified as TRIVIAL!")
            print(f"     - Effect size: {nshot_trivial['effect_size']:.3f}")
            print(f"     - Effect type: {nshot_trivial['effect_type']}")
        else:
            print("   [ERROR] N-Shot disparity not found in collector at all!")

    print("\n3. Testing categorization logic...")

    # Test the categorization directly
    test_result = {
        'p_value': 0.0001,
        'effect_size': 0.044,  # Equity ratio from our example
        'effect_type': 'equity_ratio'
    }

    category = dashboard.collector._categorize_result(test_result)
    print(f"   - Equity ratio 0.044 categorized as: {category}")

    if category == 'material':
        print("   [SUCCESS] Severe disparity correctly categorized as MATERIAL")
    else:
        print("   [ERROR] Severe disparity incorrectly categorized")

    # Test borderline case
    borderline_result = {
        'p_value': 0.001,
        'effect_size': 0.75,  # Just above 80% rule but not severe
        'effect_type': 'equity_ratio'
    }

    borderline_category = dashboard.collector._categorize_result(borderline_result)
    print(f"   - Equity ratio 0.75 categorized as: {borderline_category}")

    # Test acceptable case
    acceptable_result = {
        'p_value': 0.001,
        'effect_size': 0.85,  # Above 80% threshold
        'effect_type': 'equity_ratio'
    }

    acceptable_category = dashboard.collector._categorize_result(acceptable_result)
    print(f"   - Equity ratio 0.85 categorized as: {acceptable_category}")

    print("\n4. Summary:")
    if nshot_result and category == 'material':
        print("   [SUCCESS] Severe n-shot disparity now properly escalates to Material tab")
        print("   [SUCCESS] 22.6x questioning disparity will appear in headline results")
    else:
        print("   [ERROR] Integration not working correctly")

    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_nshot_collector_integration()