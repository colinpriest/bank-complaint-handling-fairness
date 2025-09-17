#!/usr/bin/env python3
"""Debug file loading logic"""

from pathlib import Path

results_dir = Path("nshot_v2_results")
local_runs_file = results_dir / "runs.jsonl"
out_dir = Path("out")

print(f"Results directory: {results_dir}")
print(f"Results dir exists: {results_dir.exists()}")
print(f"Local runs file: {local_runs_file}")
print(f"Local runs file exists: {local_runs_file.exists()}")
print(f"Out dir: {out_dir}")
print(f"Out dir exists: {out_dir.exists()}")
print(f"Out runs file exists: {(out_dir / 'runs.jsonl').exists()}")

if local_runs_file.exists():
    print(f"Local file size: {local_runs_file.stat().st_size}")

# Test the mapping logic
data_file_mapping = {
    "raw_results": "runs.jsonl",
}

if local_runs_file.exists():
    data_file_mapping["raw_results"] = str(local_runs_file)
    print(f"Using local file: {data_file_mapping['raw_results']}")
elif out_dir.exists() and (out_dir / "runs.jsonl").exists():
    data_file_mapping["raw_results"] = str(out_dir / "runs.jsonl")
    print(f"Using out file: {data_file_mapping['raw_results']}")

# Now test the path logic
for data_type, filename in data_file_mapping.items():
    print(f"\nProcessing {data_type}: {filename}")

    # Handle both relative and absolute paths
    if Path(filename).is_absolute() or str(filename).startswith(('out/', 'advanced_results/', 'out\\', 'advanced_results\\')):
        file_path = Path(filename)
        print(f"  Using absolute path: {file_path}")
    else:
        file_path = results_dir / filename
        print(f"  Using relative path: {file_path}")

    print(f"  Final path exists: {file_path.exists()}")
    if file_path.exists():
        print(f"  File size: {file_path.stat().st_size}")