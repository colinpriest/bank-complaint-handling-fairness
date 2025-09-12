#!/usr/bin/env python3
"""
Cache Repair Script

This script identifies and repairs corrupted JSON cache files.
"""

import json
import os
from pathlib import Path
import shutil

def repair_cache_file(file_path):
    """Try to repair a corrupted JSON cache file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to parse as-is first
        try:
            json.loads(content)
            return True, "Valid JSON"
        except json.JSONDecodeError as e:
            # Try to fix common corruption patterns
            lines = content.split('\n')
            
            # Look for the closing brace and remove anything after it
            fixed_lines = []
            found_closing = False
            
            for line in lines:
                if not found_closing:
                    fixed_lines.append(line)
                    if line.strip() == '}':
                        found_closing = True
                        break
                        
            if found_closing:
                fixed_content = '\n'.join(fixed_lines)
                
                # Test if the fix works
                try:
                    json.loads(fixed_content)
                    
                    # Backup original and write fixed version
                    backup_path = file_path + '.backup'
                    shutil.copy2(file_path, backup_path)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                        
                    return True, f"Repaired (backup: {backup_path})"
                    
                except json.JSONDecodeError:
                    return False, f"Could not repair: {str(e)}"
            else:
                return False, f"No closing brace found: {str(e)}"
                
    except Exception as e:
        return False, f"Error reading file: {str(e)}"

def main():
    cache_dir = Path("data_cache")
    
    if not cache_dir.exists():
        print("No data_cache directory found")
        return
        
    print(f"Scanning cache directory: {cache_dir}")
    
    total_files = 0
    corrupted_files = 0
    repaired_files = 0
    
    for cache_file in cache_dir.glob("*.json"):
        total_files += 1
        
        success, message = repair_cache_file(cache_file)
        
        if "Repaired" in message:
            repaired_files += 1
            print(f"[REPAIRED] {cache_file.name} - {message}")
        elif "Valid JSON" in message:
            print(f"[OK] {cache_file.name}")
        else:
            corrupted_files += 1
            print(f"[CORRUPTED] {cache_file.name} - {message}")
            
            # Delete irreparably corrupted files
            try:
                cache_file.unlink()
                print(f"[DELETED] {cache_file.name}")
            except Exception as e:
                print(f"[ERROR] Could not delete {cache_file.name}: {e}")
    
    print(f"\nSUMMARY:")
    print(f"Total files: {total_files}")
    print(f"Repaired: {repaired_files}")
    print(f"Corrupted (deleted): {corrupted_files}")
    print(f"Valid: {total_files - corrupted_files - repaired_files}")

if __name__ == "__main__":
    main()