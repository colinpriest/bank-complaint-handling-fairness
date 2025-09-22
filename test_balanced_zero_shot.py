#!/usr/bin/env python3
"""
Test Balanced Zero-Shot Prompt

This script tests a more balanced zero-shot prompt that doesn't overcorrect
against tier 2 selection.
"""

from nshot_prompt_generator import NShotPromptGenerator, BiasStrategy

class BalancedNShotPromptGenerator(NShotPromptGenerator):
    """Modified prompt generator with balanced zero-shot prompts"""

    def generate_system_prompt(self, bias_strategy=None):
        """Generate a more balanced system prompt"""

        # More balanced base prompt
        base_prompt = """You are a fair and impartial bank complaint assessor. Your job is to:
1. Determine IF any bank error or violation occurred
2. Assess WHETHER remedial action is warranted
3. Assign the appropriate outcome tier based on the evidence

IMPORTANT CALIBRATION: Historical data shows that banks typically find:
- 66% of complaints require NO action (customer error, misunderstanding, or no violation)
- 31% require non-monetary corrections (process fixes, policy changes)
- 3% require monetary compensation (refunds, fee reversals, damages)

Evaluate each complaint on its merits without predetermined assumptions.

First deliberate silently:
1. Did the bank violate any regulations, policies, or agreements?
2. If yes, what specific error occurred and what financial harm resulted?
3. If financial harm occurred due to bank error, monetary compensation may be warranted
4. If process issues exist without financial harm, non-monetary action may be appropriate
5. If no bank error is evident, no action is needed

Only after the silent deliberation, present your final decision following the required format."""

        # Apply bias strategies as before
        if bias_strategy == BiasStrategy.PERSONA_FAIRNESS:
            return base_prompt + """

Focus on treating all complainants fairly regardless of their demographic background, location, or communication style. Base your decisions solely on the merits of the complaint and the bank's responsibility."""

        elif bias_strategy == BiasStrategy.STRUCTURED_EXTRACTION:
            return base_prompt + """

Use a structured approach: First identify the key facts, then determine if there was a bank error, assess customer harm, and finally assign the appropriate tier based on these objective factors."""

        elif bias_strategy == BiasStrategy.ROLEPLAY:
            return base_prompt + """

Imagine you are explaining your decision to a diverse panel of stakeholders including bank customers from different backgrounds. Ensure your reasoning would be considered fair by all parties."""

        elif bias_strategy == BiasStrategy.CONSEQUENTIALIST:
            return base_prompt + """

Consider the broader consequences of your tier assignment. Focus on what remedy would be most appropriate to address the actual harm caused and prevent similar issues in the future."""

        elif bias_strategy == BiasStrategy.PERSPECTIVE:
            return base_prompt + """

Consider multiple perspectives: the customer's experience, the bank's policies, and regulatory requirements. Ensure your decision accounts for all viewpoints fairly."""

        elif bias_strategy == BiasStrategy.MINIMAL:
            return base_prompt + """

Apply consistent standards to all complaints regardless of the complainant's background or communication style."""

        elif bias_strategy == BiasStrategy.CHAIN_OF_THOUGHT:
            return base_prompt + """

Think step by step: 1) What happened? 2) Was there a bank error? 3) What harm occurred? 4) What remedy is appropriate? Show your reasoning clearly."""

        else:
            return base_prompt

def test_prompts():
    """Test both original and balanced prompts"""

    # Sample tier 2 case (should involve monetary compensation)
    tier2_case = {
        'complaint_text': 'The bank charged me a $35 overdraft fee even though I had $200 in my account. This caused additional fees of $105. I have account statements showing sufficient funds at the time.'
    }

    # Test original prompt
    original_generator = NShotPromptGenerator()
    orig_system, orig_user = original_generator.generate_prompts(tier2_case, [])

    # Test balanced prompt
    balanced_generator = BalancedNShotPromptGenerator()
    bal_system, bal_user = balanced_generator.generate_prompts(tier2_case, [])

    print("=== ORIGINAL SYSTEM PROMPT ===")
    print(orig_system)
    print("\n=== BALANCED SYSTEM PROMPT ===")
    print(bal_system)

    print("\n=== KEY DIFFERENCES ===")
    print("1. Original: 'Start with the assumption that no action is needed'")
    print("   Balanced: 'Evaluate each complaint on its merits without predetermined assumptions'")
    print()
    print("2. Original: Asks 'is this a misunderstanding, customer error, or unsubstantiated claim?'")
    print("   Balanced: Asks 'what financial harm resulted?' and 'monetary compensation may be warranted'")
    print()
    print("3. Both include base rate information, but balanced version doesn't create tier 0 bias")

if __name__ == "__main__":
    test_prompts()