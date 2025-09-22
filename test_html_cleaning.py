#!/usr/bin/env python3
"""
Test script to verify that markdown code blocks are properly removed from AI commentary
"""

from ai_commentary_generator import AICommentaryGenerator

def test_html_cleaning():
    """Test that markdown code blocks are properly removed"""
    
    print("Testing HTML cleaning functionality...")
    
    generator = AICommentaryGenerator()
    
    # Test the HTML cleaning function directly
    test_content_with_markdown = '''```html
<div class="enhanced-commentary">
    <h4>Financial Services Impact</h4>
    <p>This is a test paragraph.</p>
</div>
```'''
    
    print("Original content with markdown:")
    print(repr(test_content_with_markdown))
    
    cleaned_content = generator._clean_html_content(test_content_with_markdown)
    
    print("\nCleaned content:")
    print(repr(cleaned_content))
    
    # Check that markdown artifacts are removed
    if '```' not in cleaned_content:
        print("✅ SUCCESS: Markdown code blocks removed successfully!")
    else:
        print("❌ FAILED: Markdown code blocks still present")
    
    # Test with a real AI-generated response
    print("\n🔍 Testing with real AI-generated content...")
    
    test_finding = {
        'finding': 'Test finding for HTML cleaning',
        'test_name': 'Test Analysis',
        'p_value': 0.001,
        'effect_size': 0.5,
        'effect_type': 'cohens_d',
        'sample_size': 1000,
        'implication': 'Test implication'
    }
    
    try:
        commentary = generator.generate_enhanced_commentary(test_finding)
        
        if '```' not in commentary:
            print("✅ SUCCESS: Real AI commentary has no markdown artifacts!")
        else:
            print("❌ FAILED: Real AI commentary still contains markdown artifacts")
            print("Found markdown artifacts in:")
            print(commentary[:500] + "..." if len(commentary) > 500 else commentary)
            
    except Exception as e:
        print(f"⚠️  Could not test real AI generation: {e}")
        print("   This is expected if OPENAI_API_KEY is not set")

if __name__ == "__main__":
    test_html_cleaning()
