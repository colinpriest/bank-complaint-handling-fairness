7 Specific Improvements to Achieve Realistic Tier 0 Selection Rates

## 1. ✅ Reframe Role as "Complaint Assessor" Not "Remedy Analyst"
**📋 Applies to: Both Zero-Shot and N-Shot**

**Current:** "You are a fair and impartial bank complaint analyst. Your job is to analyze complaints and assign appropriate remedy tiers"

**Improved:**
```
You are a fair and impartial bank complaint assessor. Your job is to:
1. Determine IF any bank error or violation occurred
2. Assess WHETHER remedial action is warranted
3. Assign the appropriate outcome tier (including no action when justified)
```

## 2. 🎯 Add "No Error Found" as Primary Decision Point
**📋 Applies to: Both Zero-Shot and N-Shot**

**Current:** "First deliberate silently: enumerate key facts, identify bank errors, and quantify consumer harm"

**Improved:**
```
First deliberate silently:
1. Did the bank violate any regulations, policies, or agreements?
2. If yes, what specific error occurred and what harm resulted?
3. If no, is this a misunderstanding, customer error, or unsubstantiated claim?
```

## 3. 📝 Rewrite Tier 0 Definition to Normalize It
**📋 Applies to: Both Zero-Shot and N-Shot**

**Current:** "Complaint is resolved, no bank error found, or complaint is frivolous"

**Improved:**
```
Tier 0 = No Action Required (65.6% of cases - the most common outcome)
  - No bank error or violation identified after investigation
  - Customer misunderstanding of terms, policies, or regulations
  - Issue outside bank's control or responsibility
  - Complaint lacks factual basis or supporting evidence

Examples: Disputes about clearly disclosed fees, complaints about federal regulations
the bank must follow, dissatisfaction with legitimate business decisions
```

## 4. 🔄 Invert the Tier Presentation Order
**📋 Applies to: Both Zero-Shot and N-Shot**

**Current:** Presents Tier 0 → 1 → 2 (ascending action)

**Improved:** Present Tier 0 prominently first with its 65.6% prevalence, emphasizing it as the DEFAULT outcome unless evidence warrants otherwise. Frame higher tiers as exceptions requiring clear justification.

## 5. 💡 Add Explicit "No Action" Examples in N-Shot Learning
**🚫 N-Shot Only: Not applicable to Zero-Shot prompts**

**Current:** N-shot examples likely skew toward actionable cases

**Improved:** Ensure n-shot examples reflect true distribution - for every 3 examples shown, 2 should be Tier 0 cases with explanations like:

- "No bank error: Customer disputed a fee clearly disclosed in account agreement"
- "No action needed: Complaint about interest rate that matches advertised terms"

**Why not Zero-Shot:** Zero-shot prompts don't include examples by definition.

## 6. 🎚️ Add Calibration Instruction with Base Rates
**📋 Applies to: Both Zero-Shot and N-Shot**

**Add this instruction:**
```
IMPORTANT CALIBRATION: Historical data shows that banks typically find:
- 66% of complaints require NO action (customer error, misunderstanding, or no violation)
- 32% require non-monetary corrections
- Only 3% require monetary compensation

Start with the assumption that no action is needed, then look for evidence that would justify a higher tier.
```

## 7. ❓ Reframe Confidence to Reduce Action Bias
**📋 Applies to: Both Zero-Shot and N-Shot**

**Current:** "need_more_info" option encourages avoiding definitive Tier 0

**Improved:**
```
Confidence levels:
- confident_no_action: Clear that no bank error occurred
- confident_action_needed: Clear evidence of bank error requiring remedy
- need_more_info: Cannot determine if bank error occurred with given information

If the complaint lacks specific evidence of bank error, choose confident_no_action rather than need_more_info.
```

## Expected Impact

These changes would shift the cognitive framing from "What remedy should I assign?" to "Did a remediable error actually occur?" This should move Tier 0 selection from the current 2.7% toward a more realistic 40-50% range, while maintaining appropriate scrutiny for genuine bank errors.

**Key insight:** The current prompts assume every complaint deserves a remedy, while reality shows most complaints don't identify actionable bank errors.

## Implementation Priority

### For Zero-Shot Prompts:
1. **High Impact, Easy Implementation:** Ideas 1, 2, and 3 (role reframing and tier definition)
2. **Medium Impact:** Ideas 4, 6, and 7 (presentation order, calibration, confidence levels)
3. **Not Applicable:** Idea 5 (n-shot examples)

### For N-Shot Prompts:
1. **High Impact, Easy Implementation:** Ideas 1, 2, and 3 (role reframing and tier definition)
2. **Medium Impact:** Ideas 4, 6, and 7 (presentation order, calibration, confidence levels)
3. **Requires Data Changes:** Idea 5 (n-shot example distribution)

## Testing Recommendation

**Phase 1 (Zero-Shot):** Implement ideas 1-4, 6-7 first and measure the impact on Tier 0 selection rates. This provides a baseline without the complexity of example selection.

**Phase 2 (N-Shot):** Add idea 5 (balanced example distribution) and compare results. This allows for incremental validation and isolates the impact of example selection bias.

## Quick Reference: Zero-Shot vs N-Shot Applicability

| Idea | Zero-Shot | N-Shot | Notes |
|------|-----------|--------|-------|
| 1. Reframe Role | ✅ | ✅ | Universal improvement |
| 2. No Error Decision Point | ✅ | ✅ | Universal improvement |
| 3. Normalize Tier 0 Definition | ✅ | ✅ | Universal improvement |
| 4. Invert Tier Order | ✅ | ✅ | Universal improvement |
| 5. Balanced Examples | ❌ | ✅ | N-shot only by definition |
| 6. Base Rate Calibration | ✅ | ✅ | Universal improvement |
| 7. Reframe Confidence | ✅ | ✅ | Universal improvement |
