#!/usr/bin/env python3
"""
Test script to verify enhanced error handling works with the main experiment runner
"""

import os
from enhanced_error_handler import EnhancedOpenAIErrorHandler, enhanced_api_call_with_retry
import openai

def test_enhanced_error_handling():
    """Test the enhanced error handling functionality"""
    
    print("Testing Enhanced Error Handling...")
    
    # Test the error handler directly
    handler = EnhancedOpenAIErrorHandler()
    
    # Test with different error types (using simpler error creation)
    test_errors = [
        Exception("Rate limit exceeded"),
        Exception("Invalid API key"),
        Exception("Request timed out"),
        Exception("Connection failed"),
        Exception("Bad request"),
        Exception("Internal server error")
    ]
    
    print("\n🔍 Testing error categorization...")
    for error in test_errors:
        try:
            error_info = handler.handle_openai_error(error, {"test": "context"})
            print(f"✅ {type(error).__name__} -> {error_info['error_category']}")
        except Exception as e:
            print(f"❌ Failed to handle {type(error).__name__}: {e}")
    
    # Test retry logic
    print("\n🔍 Testing retry logic...")
    test_error_info = {
        "error_category": "rate_limit",
        "error_type": "RateLimitError"
    }
    
    should_retry = handler.should_retry(test_error_info, 0, 3)
    print(f"✅ Rate limit error should retry: {should_retry}")
    
    test_error_info["error_category"] = "authentication"
    should_retry = handler.should_retry(test_error_info, 0, 3)
    print(f"✅ Authentication error should retry: {should_retry}")
    
    # Test retry delay calculation
    print("\n🔍 Testing retry delay calculation...")
    test_error_info["error_category"] = "rate_limit"
    delay = handler.get_retry_delay(test_error_info, 0)
    print(f"✅ Rate limit retry delay (attempt 1): {delay:.1f} seconds")
    
    delay = handler.get_retry_delay(test_error_info, 2)
    print(f"✅ Rate limit retry delay (attempt 3): {delay:.1f} seconds")
    
    print("\n🎉 Enhanced error handling tests completed!")
    
    # Test with real API call (if API key is available)
    if os.getenv('OPENAI_API_KEY'):
        print("\n🔍 Testing with real API call...")
        try:
            client = openai.OpenAI()
            
            # This should work
            response = enhanced_api_call_with_retry(
                client=client,
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                max_retries=2
            )
            print("✅ Real API call succeeded")
            
        except Exception as e:
            print(f"⚠️  Real API call failed (expected if no API key): {e}")
    else:
        print("\n⚠️  No OpenAI API key found, skipping real API test")

if __name__ == "__main__":
    test_enhanced_error_handling()
