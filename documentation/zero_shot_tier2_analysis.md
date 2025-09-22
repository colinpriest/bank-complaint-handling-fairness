# Zero-Shot Tier 2 Selection Analysis and Fix

## Problem Summary

Zero-shot prompts were selecting 0% tier 2 cases (0 out of 50 total results), when the expected rate should be approximately 3% based on historical data.

**Current Results:**
- Tier 0: 74% (37/50)
- Tier 1: 26% (13/50)
- Tier 2: 0% (0/50) ← **Problem**

## Root Cause Analysis

The issue was identified in the system prompt within `nshot_prompt_generator.py:49-67`. The prompt contained several elements that created excessive bias against tier 2 selection:

### 1. **Overly Strong Default Assumption**
```
"Start with the assumption that no action is needed, then look for evidence that would justify a higher tier."
```
This created a cognitive anchor strongly biased toward tier 0.

### 2. **Tier 2 Presented as Extremely Rare**
```
"Only 3% require monetary compensation"
```
The word "Only" made tier 2 seem so rare that the model avoided selecting it entirely.

### 3. **Biased Deliberation Framework**
```
3. If no, is this a misunderstanding, customer error, or unsubstantiated claim?
```
This question pushed toward tier 0 classification without considering legitimate tier 2 scenarios.

### 4. **Missing Financial Harm Assessment**
The original prompt didn't explicitly guide the model to assess financial harm, which is the key criterion for tier 2 classification.

## Evidence of the Problem

**Sample Ground Truth Tier 2 Cases Misclassified as Tier 0:**

1. **Case 1293683**: Customer charged overdraft fees despite having sufficient funds
   - **Zero-Shot Prediction**: Tier 0
   - **Reasoning**: "The bank followed its policies..."
   - **Issue**: Failed to recognize bank error causing financial harm

2. **Case 1346217**: Unauthorized monthly fees charged without knowledge
   - **Zero-Shot Prediction**: Tier 0
   - **Reasoning**: "Customer appears to misunderstand the terms..."
   - **Issue**: Failed to identify unauthorized fee extraction

## Solution Implemented

Modified the system prompt in `nshot_prompt_generator.py` with these key changes:

### 1. **Removed Default Bias**
```diff
- Start with the assumption that no action is needed, then look for evidence that would justify a higher tier.
+ Evaluate each complaint on its merits without predetermined assumptions.
```

### 2. **Balanced Tier 2 Presentation**
```diff
- Only 3% require monetary compensation
+ 3% require monetary compensation (refunds, fee reversals, damages)
```

### 3. **Enhanced Deliberation Framework**
Added explicit financial harm assessment:
```
2. If yes, what specific error occurred and what financial harm resulted?
3. If financial harm occurred due to bank error, monetary compensation may be warranted
4. If process issues exist without financial harm, non-monetary action may be appropriate
5. If no bank error is evident, no action is needed
```

### 4. **Maintained Tier 0 Improvements**
The fix preserves the improvements that correctly addressed the original tier 0 under-selection problem (was 2.7%, should be ~66%).

## Expected Impact

This balanced approach should:
- **Maintain** appropriate tier 0 selection (~66%)
- **Preserve** tier 1 selection (~31%)
- **Restore** tier 2 selection (~3%) for legitimate monetary compensation cases

## Technical Implementation

**File Modified:** `D:\dev\bank-complaint-handling-fairness\nshot_prompt_generator.py`
**Lines Changed:** 49-67 (base_prompt definition)
**Impact:** All future zero-shot experiments will use the balanced prompt

## Testing Recommendation

1. **Immediate Validation**: Run a small batch of zero-shot experiments on known tier 2 cases to verify the fix
2. **Full Re-evaluation**: Consider re-running zero-shot analysis on the complete dataset to get accurate tier distribution
3. **Monitor Results**: Track tier distribution in future experiments to ensure balance is maintained

## Historical Context

This issue arose from implementing the "tier zero improvements" documented in `prompt_ideas_to_increase_tier_zero.md`. Those improvements successfully fixed tier 0 under-selection but overcorrected, creating tier 2 under-selection. The current fix maintains the tier 0 improvements while restoring tier 2 capability.