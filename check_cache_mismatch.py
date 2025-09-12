#!/usr/bin/env python3
import json
import hashlib
from pathlib import Path

# Check one cache file
cache_file = Path('data_cache/anthropic/0315082c35731d65.json')
if cache_file.exists():
    with open(cache_file, 'r') as f:
        data = json.load(f)
    
    print("Cache file contents:")
    print(f"  Provider: {data['provider']}")
    print(f"  Model: {data['model_id']}")
    print(f"  System: {data['system'][:100]}...")
    print(f"  User: {data['user'][:100]}...")
    
    # Generate cache key using current method
    content = f"{data['provider']}:{data['model_id']}:{data['system']}:{data['user']}"
    current_key = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    
    print(f"\nCache key analysis:")
    print(f"  File name: {cache_file.name}")
    print(f"  Current key: {current_key}")
    print(f"  Match: {cache_file.stem == current_key}")
    
    # Check if there's a file with the current key
    expected_file = cache_file.parent / f"{current_key}.json"
    print(f"  Expected file: {expected_file}")
    print(f"  Expected file exists: {expected_file.exists()}")
else:
    print("Cache file not found")

