#!/usr/bin/env python3
"""
Logging configuration to reduce verbosity of OpenAI API calls
"""

import logging
import os

def configure_openai_logging():
    """Configure logging to reduce OpenAI API verbosity"""
    
    # Set httpx logging to WARNING level to suppress HTTP request logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Set openai logging to WARNING level
    logging.getLogger("openai").setLevel(logging.WARNING)
    
    # Set urllib3 logging to WARNING level (used by httpx)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Set requests logging to WARNING level (if used)
    logging.getLogger("requests").setLevel(logging.WARNING)
    
    # Set instructor logging to WARNING level
    logging.getLogger("instructor").setLevel(logging.WARNING)
    
    # Set httpx.connection to WARNING (more specific httpx logger)
    logging.getLogger("httpx.connection").setLevel(logging.WARNING)
    
    # Set httpx.transport to WARNING (more specific httpx logger)
    logging.getLogger("httpx.transport").setLevel(logging.WARNING)
    
    # Set httpcore logging to WARNING (httpx dependency)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    # Set httpcore.connection to WARNING
    logging.getLogger("httpcore.connection").setLevel(logging.WARNING)
    
    # Set httpcore.http11 to WARNING
    logging.getLogger("httpcore.http11").setLevel(logging.WARNING)
    
    # Keep our own logging at INFO level for important messages
    logging.getLogger(__name__).setLevel(logging.INFO)
    
    # Optionally set root logger to WARNING to reduce overall verbosity
    # Uncomment the next line if you want to reduce all logging verbosity
    # logging.getLogger().setLevel(logging.WARNING)

def configure_experiment_logging():
    """Configure logging specifically for experiment runs"""
    
    # Configure OpenAI API logging
    configure_openai_logging()
    
    # Set up a simple formatter for our own messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add a console handler with our formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Set the root logger level
    root_logger.setLevel(logging.INFO)

if __name__ == "__main__":
    # Test the logging configuration
    configure_openai_logging()
    
    print("✅ Logging configuration applied")
    print("   - httpx logging set to WARNING level")
    print("   - openai logging set to WARNING level")
    print("   - urllib3 logging set to WARNING level")
    print("   - requests logging set to WARNING level")
    print("   - instructor logging set to WARNING level")
    
    # Test that verbose logging is suppressed
    import logging
    httpx_logger = logging.getLogger("httpx")
    print(f"   - httpx logger level: {httpx_logger.level} ({logging.getLevelName(httpx_logger.level)})")
