#!/usr/bin/env python3
"""
Enhanced error handler for OpenAI API calls to provide detailed error information
"""

import openai
import time
from typing import Dict, Any, Optional
import traceback

class EnhancedOpenAIErrorHandler:
    """Enhanced error handler for OpenAI API calls with detailed error reporting"""
    
    @staticmethod
    def handle_openai_error(e: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle OpenAI API errors with detailed error information
        
        Args:
            e: The exception that occurred
            context: Additional context about the API call
            
        Returns:
            Dictionary with detailed error information
        """
        error_info = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "context": context or {},
            "timestamp": time.time()
        }
        
        # Handle specific OpenAI error types
        if isinstance(e, openai.RateLimitError):
            error_info.update({
                "error_category": "rate_limit",
                "suggestion": "Rate limit exceeded. Consider reducing request frequency or upgrading your plan.",
                "retry_after": getattr(e, 'retry_after', None)
            })
            
        elif isinstance(e, openai.APITimeoutError):
            error_info.update({
                "error_category": "timeout",
                "suggestion": "Request timed out. Consider increasing timeout or reducing request complexity.",
                "timeout_duration": getattr(e, 'timeout', None)
            })
            
        elif isinstance(e, openai.APIConnectionError):
            error_info.update({
                "error_category": "connection",
                "suggestion": "Network connection issue. Check internet connection and API endpoint.",
                "connection_details": str(e)
            })
            
        elif isinstance(e, openai.AuthenticationError):
            error_info.update({
                "error_category": "authentication",
                "suggestion": "API key authentication failed. Check your OpenAI API key.",
                "api_key_status": "invalid_or_missing"
            })
            
        elif isinstance(e, openai.PermissionDeniedError):
            error_info.update({
                "error_category": "permission",
                "suggestion": "Insufficient permissions. Check your API key permissions and model access.",
                "permission_details": str(e)
            })
            
        elif isinstance(e, openai.BadRequestError):
            error_info.update({
                "error_category": "bad_request",
                "suggestion": "Invalid request parameters. Check your prompt and model parameters.",
                "request_details": str(e)
            })
            
        elif isinstance(e, openai.InternalServerError):
            error_info.update({
                "error_category": "server_error",
                "suggestion": "OpenAI server error. This is likely temporary, retry after a delay.",
                "server_status": "internal_error"
            })
            
        else:
            error_info.update({
                "error_category": "unknown",
                "suggestion": "Unknown error type. Check the full error message for details.",
                "full_traceback": traceback.format_exc()
            })
        
        return error_info
    
    @staticmethod
    def print_detailed_error(error_info: Dict[str, Any]):
        """Print detailed error information in a user-friendly format"""
        print(f"\n🚨 OpenAI API Error Details:")
        print(f"   Error Type: {error_info['error_type']}")
        print(f"   Category: {error_info['error_category']}")
        print(f"   Message: {error_info['error_message']}")
        print(f"   Suggestion: {error_info['suggestion']}")
        
        if error_info.get('context'):
            print(f"   Context: {error_info['context']}")
        
        if error_info.get('retry_after'):
            print(f"   Retry After: {error_info['retry_after']} seconds")
        
        if error_info.get('full_traceback'):
            print(f"   Full Traceback:\n{error_info['full_traceback']}")
        
        print("=" * 60)
    
    @staticmethod
    def should_retry(error_info: Dict[str, Any], attempt: int, max_retries: int) -> bool:
        """Determine if the error should be retried"""
        if attempt >= max_retries:
            return False
        
        # Don't retry authentication or permission errors
        if error_info['error_category'] in ['authentication', 'permission']:
            return False
        
        # Don't retry bad request errors (they won't succeed on retry)
        if error_info['error_category'] == 'bad_request':
            return False
        
        # Retry rate limits, timeouts, connection issues, and server errors
        return error_info['error_category'] in [
            'rate_limit', 'timeout', 'connection', 'server_error', 'unknown'
        ]
    
    @staticmethod
    def get_retry_delay(error_info: Dict[str, Any], attempt: int) -> float:
        """Calculate retry delay based on error type and attempt number"""
        base_delay = 1.0
        
        if error_info['error_category'] == 'rate_limit':
            # Use the retry_after value if available, otherwise exponential backoff
            if error_info.get('retry_after'):
                return float(error_info['retry_after'])
            else:
                return base_delay * (2 ** attempt)
        
        elif error_info['error_category'] == 'timeout':
            # Shorter delay for timeouts
            return base_delay * (1.5 ** attempt)
        
        elif error_info['error_category'] == 'connection':
            # Longer delay for connection issues
            return base_delay * (3 ** attempt)
        
        else:
            # Standard exponential backoff
            return base_delay * (2 ** attempt)


def enhanced_api_call_with_retry(client, model: str, messages: list, 
                                max_retries: int = 3, **kwargs) -> Any:
    """
    Enhanced API call with detailed error handling and retry logic
    
    Args:
        client: OpenAI client instance
        model: Model name to use
        messages: List of messages for the API call
        max_retries: Maximum number of retry attempts
        **kwargs: Additional parameters for the API call
        
    Returns:
        API response or raises the last error
    """
    handler = EnhancedOpenAIErrorHandler()
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            return response
            
        except Exception as e:
            context = {
                "model": model,
                "attempt": attempt + 1,
                "max_retries": max_retries,
                "message_count": len(messages),
                "total_message_length": sum(len(str(msg)) for msg in messages)
            }
            
            error_info = handler.handle_openai_error(e, context)
            
            if attempt == 0:  # Only print detailed error on first attempt
                handler.print_detailed_error(error_info)
            
            if not handler.should_retry(error_info, attempt, max_retries):
                print(f"❌ Not retrying {error_info['error_category']} error")
                raise e
            
            if attempt < max_retries - 1:
                delay = handler.get_retry_delay(error_info, attempt)
                print(f"⏳ Retrying in {delay:.1f} seconds (attempt {attempt + 2}/{max_retries})...")
                time.sleep(delay)
            else:
                print(f"❌ Max retries ({max_retries}) exceeded")
                raise e


if __name__ == "__main__":
    # Test the error handler
    import openai
    
    # Test with a mock error
    try:
        raise openai.APIError("Test API error")
    except Exception as e:
        handler = EnhancedOpenAIErrorHandler()
        error_info = handler.handle_openai_error(e, {"test": "context"})
        handler.print_detailed_error(error_info)
