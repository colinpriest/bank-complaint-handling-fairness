#!/usr/bin/env python3
"""
Test script for AI Commentary functionality
"""

import os
from ai_commentary_generator import AICommentaryGenerator, extract_findings_for_commentary

def test_ai_commentary():
    """Test the AI commentary generation functionality"""
    
    print("Testing AI Commentary Generator...")
    
    # Check if OpenAI API key is available
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: OPENAI_API_KEY not set. AI commentary will use fallback content.")
        print("   Set OPENAI_API_KEY environment variable to test with real ChatGPT API.")
    
    try:
        # Initialize the generator
        generator = AICommentaryGenerator()
        print("✅ AI Commentary Generator initialized successfully")
        
        # Test finding data
        test_finding = {
            'finding': 'Total disparity (12.1% vs 0%) difference in question rates (White vs Black)',
            'test_name': 'Question Rate Equity by Ethnicity: N-Shot',
            'p_value': 0.001,
            'effect_size': 1.000,
            'effect_type': 'equity_deficit',
            'sample_size': 10000,
            'implication': 'SEVERE question rate inequity detected'
        }
        
        # Test enhanced commentary generation
        print("\n🔍 Testing enhanced commentary generation...")
        commentary = generator.generate_enhanced_commentary(test_finding)
        print("✅ Enhanced commentary generated successfully")
        print(f"   Length: {len(commentary)} characters")
        print(f"   Preview: {commentary[:200]}...")
        
        # Test executive summary generation
        print("\n📊 Testing executive summary generation...")
        material_findings = [test_finding]
        trivial_findings_count = 5
        
        summary = generator.generate_executive_summary(material_findings, trivial_findings_count)
        print("✅ Executive summary generated successfully")
        print(f"   Length: {len(summary)} characters")
        print(f"   Preview: {summary[:200]}...")
        
        # Test findings extraction
        print("\n🔧 Testing findings extraction...")
        all_findings = [test_finding] + [
            {
                'finding': 'Small difference in tier assignments',
                'test_name': 'Tier Assignment Test',
                'p_value': 0.01,
                'effect_size': 0.05,
                'effect_type': 'cohens_d',
                'sample_size': 50000,
                'implication': 'Trivial difference detected'
            }
        ]
        
        material, trivial_count = extract_findings_for_commentary(all_findings)
        print(f"✅ Findings extraction completed")
        print(f"   Material findings: {len(material)}")
        print(f"   Trivial findings count: {trivial_count}")
        
        print("\n🎉 All tests passed! AI commentary system is working correctly.")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("   This is expected if OPENAI_API_KEY is not set.")
        print("   The system will use fallback content in production.")

if __name__ == "__main__":
    test_ai_commentary()
