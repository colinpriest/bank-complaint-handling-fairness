# Stochastic Experiment Creation Algorithm

This document describes the new stochastic approach for creating experiments in the Bank Complaint Handling Fairness Analysis system, as specified in `documentation/experiment_sampling_specification.md`.

## Overview

The stochastic experiment creation algorithm replaces the previous factorial combination approach with a more efficient sampling method. Instead of creating all possible combinations of personas, ground truth cases, and mitigation strategies, it uses controlled random sampling to create a representative subset of experiments.

## Algorithm Steps

### 1. Ground Truth Examples
- Start with ground truth examples from the database
- These serve as the foundation for all experiments

### 2. Baseline Experiments
- For each ground truth example, create 2 baseline experiments:
  - One zero-shot experiment (no examples provided)
  - One n-shot experiment (examples provided using DPP+k-NN selection)
- Only add if no matching experiment already exists

### 3. Persona-Injected Experiments
- For each baseline experiment:
  - Get its experiment number
  - Set random seed to the experiment number (for reproducibility)
  - Randomly select 10 personas from the available personas
  - Create persona-injected experiments for each selected persona
- Only add if no matching experiment already exists

### 4. Bias-Mitigation Experiments
- For each persona-injected experiment:
  - Get its experiment number
  - For each bias mitigation strategy, create a bias-mitigation experiment
- Only add if no matching experiment already exists

## Key Benefits

### Efficiency
- **Massive reduction in experiment count**: 94.2% reduction (from 14,260 to 820 experiments for 10 ground truth cases)
- **Faster execution**: Dramatically reduced computational requirements
- **Manageable dataset**: More reasonable dataset size for analysis

### Reproducibility
- **Seeded random sampling**: Uses experiment number as random seed
- **Deterministic results**: Same inputs always produce same persona selections
- **Traceable sampling**: Can reproduce exact persona selections for any baseline experiment

### Statistical Validity
- **Representative sampling**: Covers diverse persona combinations
- **Good coverage**: 11.2% of personas sampled per baseline (with 10 personas per baseline)
- **Balanced representation**: Random sampling ensures fair representation across demographics

## Files and Usage

### Core Implementation
- **`stochastic_experiment_creator.py`**: Main implementation of the stochastic algorithm
- **`experiment_creation_comparison.py`**: Compares factorial vs stochastic approaches
- **`migrate_to_stochastic_experiments.py`**: Migration management and analysis tools

### Usage Examples

#### Create Experiments Using Stochastic Approach
```bash
# Create experiments for all ground truth cases
python stochastic_experiment_creator.py

# Create experiments for limited ground truth cases
python stochastic_experiment_creator.py --ground-truth-limit 50

# Show current statistics before creating
python stochastic_experiment_creator.py --show-stats
```

#### Compare Approaches
```bash
# Compare factorial vs stochastic approaches
python experiment_creation_comparison.py

# Compare with custom parameters
python experiment_creation_comparison.py --ground-truth-count 20 --personas-per-baseline 8
```

#### Migration Management
```bash
# Show comprehensive migration plan
python migrate_to_stochastic_experiments.py --plan

# Analyze existing experiments
python migrate_to_stochastic_experiments.py --analyze

# Create new experiments (dry run)
python migrate_to_stochastic_experiments.py --create --dry-run

# Create new experiments
python migrate_to_stochastic_experiments.py --create --ground-truth-limit 10
```

## Configuration

### Key Parameters
- **`personas_per_baseline`**: Number of personas to sample per baseline experiment (default: 10)
- **`ground_truth_limit`**: Maximum number of ground truth cases to process (default: all)
- **`llm_model`**: LLM model to use (default: 'gpt-4o-mini')

### Customization
You can modify the `StochasticExperimentCreator` class to adjust:
- Number of personas sampled per baseline
- Random sampling strategy
- Experiment creation logic
- Database interaction patterns

## Comparison with Factorial Approach

| Aspect | Factorial Approach | Stochastic Approach |
|--------|-------------------|-------------------|
| **Total Experiments** | 14,260 (10 ground truth) | 820 (10 ground truth) |
| **Reduction** | - | 94.2% |
| **Coverage** | 100% of combinations | 11.2% of personas |
| **Execution Time** | Very long | Much faster |
| **Resource Usage** | High | Low |
| **Reproducibility** | Deterministic | Seeded random |
| **Scalability** | Poor | Excellent |

## Statistical Considerations

### Coverage Analysis
- With 89 total personas and 10 sampled per baseline, we achieve 11.2% coverage
- This provides good statistical representation while maintaining efficiency
- Random sampling ensures unbiased selection across demographic groups

### Validation
- The stochastic approach maintains statistical validity through:
  - Random sampling from the full persona population
  - Reproducible results via seeded random number generation
  - Representative coverage across all demographic dimensions

### Trade-offs
- **Pros**: Massive efficiency gains, manageable dataset, good statistical coverage
- **Cons**: Does not test every possible combination, may miss some edge cases

## Migration Strategy

### For New Projects
1. Use the stochastic approach from the beginning
2. Start with a small batch (10-20 ground truth cases) to validate
3. Scale up based on results and requirements

### For Existing Projects
1. Analyze current experiment coverage
2. Identify gaps that stochastic approach can fill efficiently
3. Use stochastic approach for new experiments
4. Consider archiving or analyzing existing factorial experiments

## Database Schema

The stochastic approach uses the same database schema as the factorial approach:

```sql
CREATE TABLE experiments (
    experiment_id SERIAL PRIMARY KEY,
    case_id INTEGER NOT NULL,
    decision_method VARCHAR(20) NOT NULL,  -- 'zero-shot' or 'n-shot'
    llm_model VARCHAR(50) DEFAULT 'gpt-4o-mini',
    llm_simplified_tier INTEGER DEFAULT -999,
    persona VARCHAR(50),  -- e.g., 'asian_male_rural'
    gender VARCHAR(20),
    ethnicity VARCHAR(50),
    geography VARCHAR(50),
    risk_mitigation_strategy VARCHAR(100),
    system_prompt TEXT,
    user_prompt TEXT,
    system_response TEXT,
    cache_id INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (case_id) REFERENCES ground_truth(case_id)
);
```

## Error Handling

The implementation includes comprehensive error handling:
- Database connection failures
- Missing data (personas, ground truth cases, mitigation strategies)
- Duplicate experiment detection
- Transaction rollback on errors
- Detailed logging and progress reporting

## Performance Monitoring

The system provides detailed statistics and monitoring:
- Experiment creation counts by type
- Coverage analysis
- Efficiency metrics
- Progress reporting during creation
- Error tracking and reporting

## Future Enhancements

Potential improvements to the stochastic approach:
1. **Adaptive sampling**: Adjust persona count based on statistical significance
2. **Stratified sampling**: Ensure balanced representation across demographic groups
3. **Dynamic scaling**: Automatically adjust experiment count based on results
4. **Quality metrics**: Track and optimize experiment quality over time
5. **Integration with analysis**: Direct integration with fairness analysis tools

## Conclusion

The stochastic experiment creation algorithm provides a much more efficient and scalable approach to creating experiments for fairness analysis. While it trades some comprehensive coverage for massive efficiency gains, it maintains statistical validity and provides excellent coverage for most analysis needs.

The approach is particularly valuable for:
- Large-scale fairness analysis projects
- Rapid prototyping and iteration
- Resource-constrained environments
- Projects requiring quick turnaround times

For projects requiring exhaustive coverage of all combinations, the factorial approach may still be appropriate, but for most practical applications, the stochastic approach provides the best balance of efficiency, coverage, and statistical validity.
