#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append('.')
from complaints_llm_fairness_harness import build_clients

# Build clients and check their cache stats
clients = build_clients(['claude-3.5'], 'data_cache')
claude_client = clients['claude-3.5']

print("=== Claude Cache Analysis ===")
stats = claude_client.get_stats()
print(f"API calls: {stats['api_calls']}")
print(f"Memory cache hits: {stats['memory_cache_hits']}")
print(f"Disk cache hits: {stats['disk_cache_hits']}")
print(f"Total cache hits: {stats['memory_cache_hits'] + stats['disk_cache_hits']}")
print(f"Memory hit rate: {stats['memory_hit_rate']}")
print(f"Disk hit rate: {stats['disk_hit_rate']}")
print(f"Memory cache size: {stats['memory_cache_size']}")

# Check cache directory structure
cache_dir = Path('data_cache/anthropic')
if cache_dir.exists():
    cache_files = list(cache_dir.glob('*.json'))
    print(f"\nCache files in {cache_dir}: {len(cache_files)}")
    
    # Check a few cache files to see if they're valid
    for i, cache_file in enumerate(cache_files[:3]):
        try:
            import json
            with open(cache_file, 'r') as f:
                data = json.load(f)
            print(f"  {cache_file.name}: Valid JSON, {len(str(data))} chars")
        except Exception as e:
            print(f"  {cache_file.name}: ERROR - {e}")
else:
    print(f"\nCache directory {cache_dir} does not exist")

# Test a simple cache operation
print(f"\n=== Testing Cache Operation ===")
test_system = "You are a helpful assistant."
test_user = "Hello, this is a test message."

# Try to get from cache
cached_result = claude_client._load_from_cache(claude_client._get_cache_key(test_system, test_user))
print(f"Cache lookup result: {'Found' if cached_result else 'Not found'}")

# Check cache stats after lookup
stats_after = claude_client.get_stats()
print(f"Cache stats after test lookup:")
print(f"  Memory hits: {stats_after['memory_cache_hits']} (was {stats['memory_cache_hits']})")
print(f"  Disk hits: {stats_after['disk_cache_hits']} (was {stats['disk_cache_hits']})")
print(f"  Memory cache size: {stats_after['memory_cache_size']} (was {stats['memory_cache_size']})")

# Let's check what cache keys are actually in the files
print(f"\n=== Checking Cache Keys ===")
cache_key = claude_client._get_cache_key(test_system, test_user)
print(f"Generated cache key: {cache_key}")
print(f"Expected cache file: {claude_client._get_cache_path(cache_key)}")

# Check if this key exists
cache_path = claude_client._get_cache_path(cache_key)
if cache_path.exists():
    print(f"Cache file exists: {cache_path}")
else:
    print(f"Cache file does not exist: {cache_path}")

# Let's look at a few existing cache files to see their keys
print(f"\n=== Sample Cache Files ===")
for i, cache_file in enumerate(cache_files[:5]):
    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)
        cache_key_in_file = data.get('cache_key', 'NO_KEY')
        print(f"  {cache_file.name}: {cache_key_in_file[:16]}...")
        
        # Check if this matches our generated key format
        if cache_key_in_file != 'NO_KEY':
            print(f"    File key: {cache_key_in_file}")
            print(f"    Generated key: {cache_key}")
            print(f"    Match: {cache_key_in_file == cache_key}")
    except Exception as e:
        print(f"  {cache_file.name}: ERROR - {e}")

# Let's also check if there's a mismatch in the cache key generation
print(f"\n=== Cache Key Generation Test ===")
test_cases = [
    ("You are a helpful assistant.", "Hello, this is a test message."),
    ("You are a helpful assistant.", "Hello, this is a test message."),  # Same as above
    ("You are a different assistant.", "Hello, this is a test message."),  # Different system
]

for i, (system, user) in enumerate(test_cases):
    key = claude_client._get_cache_key(system, user)
    print(f"  Test {i+1}: {key}")
    print(f"    System: {system[:30]}...")
    print(f"    User: {user[:30]}...")
