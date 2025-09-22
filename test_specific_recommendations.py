#!/usr/bin/env python3
"""
Test script to verify that AI commentary generates specific, research-driven recommendations
instead of generic governance advice.
"""

import os
from ai_commentary_generator import AICommentaryGenerator

def test_specific_recommendations():
    """Test that the AI generates specific recommendations based on research findings"""
    
    print("Testing for specific, research-driven recommendations...")
    
    try:
        generator = AICommentaryGenerator()
        
        # Test with different types of findings to ensure specificity
        test_findings = [
            {
                'finding': 'Persona injection bias differs between severity levels (Cohen\'s d = 2.239)',
                'test_name': 'Severity-Dependent Bias Analysis',
                'p_value': 0.001,
                'effect_size': 2.239,
                'effect_type': 'cohens_d',
                'sample_size': 20000,
                'implication': 'SEVERE bias amplification for high-stakes decisions'
            },
            {
                'finding': 'Geographic bias: Suburban vs Urban question rate disparity (12.3% vs 8.1%)',
                'test_name': 'Geographic Question Rate Analysis',
                'p_value': 0.005,
                'effect_size': 0.042,
                'effect_type': 'disparity_rate',
                'sample_size': 15000,
                'implication': 'MATERIAL geographic discrimination in questioning'
            },
            {
                'finding': 'Method inconsistency: Zero-shot vs N-shot bias patterns differ significantly',
                'test_name': 'Method-Dependent Bias Consistency',
                'p_value': 0.001,
                'effect_size': 0.156,
                'effect_type': 'eta_squared',
                'sample_size': 25000,
                'implication': 'Prompting method significantly affects fairness outcomes'
            }
        ]
        
        print("\n🔍 Testing enhanced commentary for severity-dependent bias...")
        commentary1 = generator.generate_enhanced_commentary(test_findings[0])
        print("✅ Generated commentary for severity-dependent bias")
        print(f"   Length: {len(commentary1)} characters")
        
        # Check for specific terms that should be present
        specific_terms = ['severity', 'high-stakes', 'Cohen', '2.239']
        found_terms = [term for term in specific_terms if term.lower() in commentary1.lower()]
        print(f"   Specific terms found: {found_terms}")
        
        print("\n🔍 Testing enhanced commentary for geographic bias...")
        commentary2 = generator.generate_enhanced_commentary(test_findings[1])
        print("✅ Generated commentary for geographic bias")
        print(f"   Length: {len(commentary2)} characters")
        
        # Check for specific terms
        specific_terms = ['geographic', 'suburban', 'urban', 'socioeconomic']
        found_terms = [term for term in specific_terms if term.lower() in commentary2.lower()]
        print(f"   Specific terms found: {found_terms}")
        
        print("\n🔍 Testing enhanced commentary for method inconsistency...")
        commentary3 = generator.generate_enhanced_commentary(test_findings[2])
        print("✅ Generated commentary for method inconsistency")
        print(f"   Length: {len(commentary3)} characters")
        
        # Check for specific terms
        specific_terms = ['zero-shot', 'n-shot', 'method', 'prompting']
        found_terms = [term for term in specific_terms if term.lower() in commentary3.lower()]
        print(f"   Specific terms found: {found_terms}")
        
        print("\n📊 Testing executive summary with multiple findings...")
        summary = generator.generate_executive_summary(test_findings, 5)
        print("✅ Generated executive summary")
        print(f"   Length: {len(summary)} characters")
        
        # Check for specific strategic recommendations
        specific_recommendations = [
            'severity-dependent', 'high-stakes', 'geographic', 'socioeconomic',
            'zero-shot', 'n-shot', 'effect size', 'filtering'
        ]
        found_recommendations = [rec for rec in specific_recommendations if rec.lower() in summary.lower()]
        print(f"   Specific recommendations found: {found_recommendations}")
        
        # Check that generic terms are NOT heavily emphasized
        generic_terms = ['fairness audit', 'ethics committee', 'monitoring protocol', 'governance framework']
        generic_found = [term for term in generic_terms if term.lower() in summary.lower()]
        print(f"   Generic terms found (should be minimal): {generic_found}")
        
        if len(found_recommendations) > len(generic_found):
            print("\n🎉 SUCCESS: AI commentary is generating specific, research-driven recommendations!")
        else:
            print("\n⚠️  WARNING: AI commentary may still be too generic")
        
        print("\n📋 Sample of generated strategic recommendations:")
        # Extract the strategic recommendations section
        if '<h4>Strategic Recommendations</h4>' in summary:
            start = summary.find('<h4>Strategic Recommendations</h4>')
            end = summary.find('</p>', start)
            if end > start:
                recommendations_section = summary[start:end+4]
                print(recommendations_section)
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")

if __name__ == "__main__":
    test_specific_recommendations()
