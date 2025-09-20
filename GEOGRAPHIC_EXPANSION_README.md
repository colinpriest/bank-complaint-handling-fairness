# Geographic Persona Expansion - Sampling Approach

This directory contains scripts to expand your geographic persona categories from 3 to 9 categories using a practical sampling-based experimental design.

## Overview

The expansion adds these new geographic categories:
- `urban_upper_middle` - Upper-middle class urban residents
- `urban_working` - Working class urban residents  
- `suburban_upper_middle` - Upper-middle class suburban residents
- `suburban_working` - Working class suburban residents
- `suburban_poor` - Lower-income suburban residents
- `rural_upper_middle` - Upper-middle class rural residents
- `rural_working` - Working class rural residents
- `rural_poor` - Lower-income rural residents

## Sampling Strategy

Instead of creating every possible combination (which would be massive), this approach uses **random sampling** to:
- Cover all combinations over time
- Keep the number of experiments manageable
- Allow for incremental analysis
- Provide statistical coverage of the experimental space

## Scripts

### 1. `add_geographic_persona_sampling.py`
**Purpose**: Sets up the sampling-based experimental design

**What it does**:
- Creates new persona records for the expanded geographic categories
- Creates an initial batch of randomly sampled experiments (default: 1000)
- Uses random sampling to avoid creating every possible combination
- Marks new experiments as needing LLM analysis (llm_simplified_tier = -999)

**Usage**:
```bash
python add_geographic_persona_sampling.py
```

### 2. `generate_experiment_batch.py`
**Purpose**: Generates additional batches of randomly sampled experiments

**What it does**:
- Creates additional batches of experiments to increase coverage
- Uses random sampling to avoid duplicates
- Allows you to increase experimental coverage over time

**Usage**:
```bash
python generate_experiment_batch.py [batch_size]
python generate_experiment_batch.py 500  # Create 500 new experiments
```

### 3. `run_new_experiments_analysis_sampling.py`
**Purpose**: Runs LLM analysis on newly added experiments

**What it does**:
- Finds experiments with llm_simplified_tier = -999 (pending analysis)
- Runs the LLM analysis for these experiments
- Updates the database with results
- Processes experiments in batches to avoid memory issues

**Usage**:
```bash
python run_new_experiments_analysis_sampling.py
python run_new_experiments_analysis_sampling.py --batch-size 100 --max-batches 5
```

### 4. `check_geographic_expansion_status.py`
**Purpose**: Checks the status of the geographic persona expansion

**What it does**:
- Shows how many personas exist for each geography
- Shows how many experiments need LLM analysis
- Shows progress towards the full 9-category expansion
- Shows sampling coverage statistics

**Usage**:
```bash
python check_geographic_expansion_status.py
```

## Process

1. **Initial Setup**:
   ```bash
   python add_geographic_persona_sampling.py
   ```
   This creates new personas and an initial batch of 1000 experiments.

2. **Check the status**:
   ```bash
   python check_geographic_expansion_status.py
   ```
   This shows you the current coverage and what needs analysis.

3. **Run the analysis**:
   ```bash
   python run_new_experiments_analysis_sampling.py
   ```
   This processes the new experiments through your LLM analysis pipeline.

4. **Generate more experiments** (optional):
   ```bash
   python generate_experiment_batch.py 1000
   ```
   This creates additional batches to increase coverage.

5. **Repeat as needed**:
   You can run the analysis and batch generation scripts multiple times to increase coverage over time.

## Sampling Benefits

- **Manageable scale**: Creates thousands instead of millions of experiments
- **Statistical coverage**: Random sampling ensures all combinations are eventually covered
- **Incremental**: You can add more experiments over time as needed
- **Efficient**: Avoids duplicate experiments
- **Practical**: Realistic for actual research workflows

## Database Changes

The scripts will:
- Add new records to the `personas` table
- Add new records to the `experiments` table with random sampling
- Preserve all existing data
- Only new experiments will have `llm_simplified_tier = -999`

## Notes

- The sampling approach creates a manageable number of experiments (thousands, not millions)
- New experiments are marked with `llm_simplified_tier = -999` to indicate they need analysis
- You can generate additional batches over time to increase coverage
- The analysis script processes experiments in batches to avoid memory issues
- Coverage statistics help you understand how much of the experimental space you've covered

## Expected Results

After running the initial setup, you should have:
- **Personas**: 89 total (including new geographic categories)
- **Experiments**: ~194,000 total (including sampled new experiments)
- **Coverage**: ~0.62% of possible combinations (which is reasonable for sampling)

## Geographic Category Definitions

### Geographic Types:
- **Urban:** Major metropolitan areas, city centers, high population density
- **Suburban:** Metropolitan outskirts, planned communities, medium density  
- **Rural:** Small towns, farming communities, low population density

### Economic Classes:
- **Upper Middle:** Professional careers, college-educated, financial security
- **Working:** Hourly/skilled trades, high school/some college, paycheck-to-paycheck
- **Poor:** Service/minimum wage jobs, financial instability, government assistance

## Safety Features

- **No Data Loss:** All existing experiments are preserved
- **Duplicate Prevention:** Scripts check for existing records before creating new ones
- **Batch Processing:** LLM analysis runs in small batches to avoid rate limiting
- **Error Handling:** Comprehensive error handling and rollback on failures
- **Progress Tracking:** Detailed progress reports and statistics

## Troubleshooting

### If scripts fail:
- Check that your PostgreSQL database is accessible
- Ensure you have the necessary environment variables set (DB_HOST, DB_USER, etc.)
- Check the error messages for specific issues

### If you need to start over:
- The scripts are designed to be safe to re-run
- They will skip existing records and only process what's needed
- No existing data will be lost

## Integration with Existing Analysis

Once the expansion is complete, your existing analysis code will automatically:
- Include the new geographic categories in bias analysis
- Generate more granular geographic bias reports
- Provide insights into how geography and socioeconomic status intersect with bias
- Maintain all existing functionality while adding new capabilities