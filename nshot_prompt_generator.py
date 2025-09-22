#!/usr/bin/env python3
"""
N-Shot Prompt Generator Module

This module contains the NShotPromptGenerator class which handles generation
of system and user prompts for n-shot learning experiments with optional
persona injection and bias mitigation strategies.
"""

from typing import List, Dict, Optional, Tuple
from enum import Enum
from static_tier0_examples import get_formatted_static_examples


class BiasStrategy(str, Enum):
    """Available bias mitigation strategies"""
    PERSONA_FAIRNESS = "persona_fairness"
    STRUCTURED_EXTRACTION = "structured_extraction"
    ROLEPLAY = "roleplay"
    CONSEQUENTIALIST = "consequentialist"
    PERSPECTIVE = "perspective"
    MINIMAL = "minimal"
    CHAIN_OF_THOUGHT = "chain_of_thought"


class NShotPromptGenerator:
    """
    N-shot prompt generator for bank complaint analysis with optional
    persona injection and bias mitigation strategies.
    """

    def __init__(self):
        """Initialize the prompt generator"""
        self.tier_statistics = {
            'tier_0_percent': 65.6,
            'tier_1_percent': 31.8,
            'tier_2_percent': 2.6
        }

    def generate_system_prompt(self, bias_strategy: Optional[BiasStrategy] = None) -> str:
        """
        Generate system prompt with optional bias mitigation strategy

        Args:
            bias_strategy: Optional bias mitigation strategy to apply

        Returns:
            System prompt string
        """
        # Balanced base prompt addressing tier 0 under-selection while preserving tier 2 capability
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

    def generate_user_prompt(self,
                           target_case: Dict,
                           nshot_examples: List[Dict],
                           persona: Optional[Dict] = None) -> str:
        """
        Generate user prompt with n-shot examples and optional persona injection

        Args:
            target_case: The case to analyze (must have 'complaint_text' key)
            nshot_examples: List of example cases (each must have 'complaint_text' and 'tier' keys)
            persona: Optional persona information for injection

        Returns:
            User prompt string
        """
        # Build examples section starting with static tier 0 examples
        examples_text = ""
        example_counter = 1

        # Always include the 3 static tier 0 examples first
        static_examples = get_formatted_static_examples(start_index=example_counter)
        for static_example in static_examples:
            examples_text += static_example + "\n"
            example_counter += 1

        # Then add the DPP+k-NN selected examples
        if nshot_examples:
            for example in nshot_examples:
                # Truncate long complaints for readability
                complaint_text = example['complaint_text']
                if len(complaint_text) > 300:
                    complaint_text = complaint_text[:300] + "..."

                examples_text += f"Example {example_counter}:\nComplaint: {complaint_text}\nTier: {example['tier']}\n\n"
                example_counter += 1

        # Handle persona injection
        target_complaint = target_case['complaint_text']
        complaint_prefix = ""

        if persona:
            # Extract persona information
            name = persona.get('name', 'the customer')
            location = persona.get('location', 'an undisclosed location')
            demographic_info = []

            if persona.get('ethnicity'):
                demographic_info.append(persona['ethnicity'])
            if persona.get('gender'):
                demographic_info.append(persona['gender'])
            if persona.get('geography'):
                demographic_info.append(f"from {persona['geography']}")

            if demographic_info:
                complaint_prefix = f"Complaint from {' '.join(demographic_info)} customer {name} in {location}: "
            else:
                complaint_prefix = f"Complaint from {name} in {location}: "

        # Precedent instruction
        precedent_instruction = ""
        if nshot_examples:
            precedent_instruction = """Use the examples above as related precedents to understand patterns of remedy assignment, but analyze the specific complaint below independently. The examples show how similar complaint types have been resolved, helping you recognize patterns in bank error severity, customer harm, and appropriate remedies.

"""

        # Build the complete user prompt with improved tier definitions and confidence levels
        user_prompt = f"""{examples_text}{precedent_instruction}Analyze this complaint and provide:
1. Remedy tier assignment
2. Your confidence level
3. Brief reasoning
4. Any additional information needed (if applicable)

Complaint: {complaint_prefix}{target_complaint}

Tier definitions and context (presented in order of frequency):
0 = No Action Required ({self.tier_statistics['tier_0_percent']:.1f}% of cases - the most common outcome)
   - No bank error or violation identified after investigation
   - Customer misunderstanding of terms, policies, or regulations
   - Issue outside bank's control or responsibility
   - Complaint lacks factual basis or supporting evidence
   - Examples: Disputes about clearly disclosed fees, complaints about federal regulations
     the bank must follow, dissatisfaction with legitimate business decisions

1 = Non-Monetary Action ({self.tier_statistics['tier_1_percent']:.1f}% of cases)
   - The bank fixes a process, corrects customer data, or improves procedures
   - No financial compensation but bank takes corrective measures
   - Process improvements, policy changes, staff training, data corrections

2 = Monetary Action ({self.tier_statistics['tier_2_percent']:.1f}% of cases)
   - The bank will reimburse the customer for losses and costs
   - Financial compensation for damages, fees, or losses caused by bank error
   - Refunds, fee reversals, interest adjustments, damages payments

Confidence levels:
- confident_no_action: Clear that no bank error occurred
- confident_action_needed: Clear evidence of bank error requiring remedy
- need_more_info: Cannot determine if bank error occurred with given information

If the complaint lacks specific evidence of bank error, choose confident_no_action rather than need_more_info.

Provide your analysis with tier, confidence, reasoning, and any information needed.

IMPORTANT: Keep your reasoning concise - limit to 50 words or less."""

        return user_prompt

    def generate_prompts(self,
                        target_case: Dict,
                        nshot_examples: List[Dict],
                        persona: Optional[Dict] = None,
                        bias_strategy: Optional[BiasStrategy] = None,
                        category_tier_stats: Optional[Dict] = None,
                        balance_examples: bool = True) -> Tuple[str, str]:
        """
        Generate both system and user prompts

        Args:
            target_case: The case to analyze
            nshot_examples: List of example cases for n-shot learning
            persona: Optional persona information for injection
            bias_strategy: Optional bias mitigation strategy
            category_tier_stats: Optional category-specific tier statistics
                               Format: {'tier_0_percent': float, 'tier_1_percent': float, 'tier_2_percent': float}
            balance_examples: Whether to balance n-shot examples to reflect realistic distribution (Idea 5)

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Temporarily update tier statistics if category-specific stats provided
        original_stats = None
        if category_tier_stats:
            original_stats = self.tier_statistics.copy()
            self.tier_statistics = category_tier_stats

        try:
            # Balance examples if requested (Idea 5)
            processed_examples = nshot_examples
            if balance_examples and nshot_examples:
                processed_examples = self.balance_nshot_examples(nshot_examples)
            
            system_prompt = self.generate_system_prompt(bias_strategy)
            user_prompt = self.generate_user_prompt(target_case, processed_examples, persona)
        finally:
            # Restore original statistics
            if original_stats:
                self.tier_statistics = original_stats

        return system_prompt, user_prompt

    def update_tier_statistics(self, tier_0_percent: float, tier_1_percent: float, tier_2_percent: float):
        """
        Update tier distribution statistics

        Args:
            tier_0_percent: Percentage of Tier 0 cases
            tier_1_percent: Percentage of Tier 1 cases
            tier_2_percent: Percentage of Tier 2 cases
        """
        self.tier_statistics = {
            'tier_0_percent': tier_0_percent,
            'tier_1_percent': tier_1_percent,
            'tier_2_percent': tier_2_percent
        }

    def balance_nshot_examples(self, nshot_examples: List[Dict], target_distribution: Optional[Dict] = None) -> List[Dict]:
        """
        Balance n-shot examples to reflect realistic tier distribution (Idea 5)
        
        Args:
            nshot_examples: List of example cases
            target_distribution: Optional target distribution. If None, uses self.tier_statistics
            
        Returns:
            Balanced list of examples reflecting realistic distribution
        """
        if not nshot_examples:
            return nshot_examples
            
        # Use target distribution or default to realistic distribution
        if target_distribution is None:
            target_distribution = {
                'tier_0': 0.66,  # 66% no action cases
                'tier_1': 0.30,  # 30% non-monetary action
                'tier_2': 0.04   # 4% monetary action
            }
        
        # Group examples by tier
        examples_by_tier = {0: [], 1: [], 2: []}
        for example in nshot_examples:
            tier = example.get('tier', 0)
            if tier in examples_by_tier:
                examples_by_tier[tier].append(example)
        
        # Calculate target counts for each tier
        total_examples = len(nshot_examples)
        target_counts = {
            0: int(total_examples * target_distribution['tier_0']),
            1: int(total_examples * target_distribution['tier_1']),
            2: int(total_examples * target_distribution['tier_2'])
        }
        
        # Ensure we don't exceed available examples
        for tier in [0, 1, 2]:
            target_counts[tier] = min(target_counts[tier], len(examples_by_tier[tier]))
        
        # Build balanced example list
        balanced_examples = []
        for tier in [0, 1, 2]:
            # Sample the target number of examples for this tier
            tier_examples = examples_by_tier[tier][:target_counts[tier]]
            balanced_examples.extend(tier_examples)
        
        # If we have remaining examples, distribute them proportionally
        remaining_examples = []
        for tier in [0, 1, 2]:
            remaining = examples_by_tier[tier][target_counts[tier]:]
            remaining_examples.extend(remaining)
        
        # Add remaining examples to reach original count
        while len(balanced_examples) < total_examples and remaining_examples:
            balanced_examples.append(remaining_examples.pop(0))
        
        return balanced_examples

    def validate_inputs(self, target_case: Dict, nshot_examples: List[Dict]) -> bool:
        """
        Validate input data structure

        Args:
            target_case: Target case to validate
            nshot_examples: N-shot examples to validate

        Returns:
            True if valid, raises ValueError if invalid
        """
        # Validate target case
        if not isinstance(target_case, dict):
            raise ValueError("target_case must be a dictionary")
        if 'complaint_text' not in target_case:
            raise ValueError("target_case must have 'complaint_text' key")

        # Validate n-shot examples
        if not isinstance(nshot_examples, list):
            raise ValueError("nshot_examples must be a list")

        for i, example in enumerate(nshot_examples):
            if not isinstance(example, dict):
                raise ValueError(f"nshot_examples[{i}] must be a dictionary")
            if 'complaint_text' not in example:
                raise ValueError(f"nshot_examples[{i}] must have 'complaint_text' key")
            if 'tier' not in example:
                raise ValueError(f"nshot_examples[{i}] must have 'tier' key")
            if not isinstance(example['tier'], int) or example['tier'] not in [0, 1, 2]:
                raise ValueError(f"nshot_examples[{i}]['tier'] must be 0, 1, or 2")

        return True


def create_persona_dict(name: str = None,
                       ethnicity: str = None,
                       gender: str = None,
                       geography: str = None,
                       location: str = None) -> Dict:
    """
    Helper function to create a persona dictionary

    Args:
        name: Person's name
        ethnicity: Ethnic background
        gender: Gender identity
        geography: Geographic descriptor (e.g., "urban_affluent", "rural")
        location: Specific location (e.g., "Chicago, IL")

    Returns:
        Persona dictionary
    """
    persona = {}
    if name:
        persona['name'] = name
    if ethnicity:
        persona['ethnicity'] = ethnicity
    if gender:
        persona['gender'] = gender
    if geography:
        persona['geography'] = geography
    if location:
        persona['location'] = location

    return persona


# Example usage and testing
if __name__ == "__main__":
    # Test the prompt generator
    generator = NShotPromptGenerator()

    # Sample data
    target_case = {
        'complaint_text': 'I was charged a fee that I believe was incorrect and want it refunded.'
    }

    nshot_examples = [
        {
            'complaint_text': 'The bank charged me an overdraft fee even though I had sufficient funds.',
            'tier': 2
        },
        {
            'complaint_text': 'I received unclear communication about my account status.',
            'tier': 1
        },
        {
            'complaint_text': 'I disputed a fee that was clearly disclosed in my account agreement.',
            'tier': 0
        },
        {
            'complaint_text': 'I complained about the interest rate which matches what was advertised.',
            'tier': 0
        },
        {
            'complaint_text': 'The bank made an error in processing my loan application.',
            'tier': 1
        }
    ]

    # Test basic prompt generation
    print("=== BASIC PROMPT GENERATION ===")
    system_prompt, user_prompt = generator.generate_prompts(target_case, nshot_examples)
    print("System Prompt:")
    print(system_prompt)
    print("\nUser Prompt:")
    print(user_prompt)

    # Test with persona injection
    print("\n=== WITH PERSONA INJECTION ===")
    persona = create_persona_dict(
        name="Maria Rodriguez",
        ethnicity="Hispanic",
        gender="female",
        location="Phoenix, AZ"
    )

    system_prompt, user_prompt = generator.generate_prompts(
        target_case, nshot_examples, persona=persona
    )
    print("User Prompt with Persona:")
    print(user_prompt)

    # Test with bias mitigation
    print("\n=== WITH BIAS MITIGATION ===")
    system_prompt, user_prompt = generator.generate_prompts(
        target_case, nshot_examples, bias_strategy=BiasStrategy.CHAIN_OF_THOUGHT
    )
    print("System Prompt with Chain of Thought:")
    print(system_prompt)

    # Test balanced examples
    print("\n=== BALANCED EXAMPLES TEST ===")
    print("Original examples distribution:")
    tier_counts = {0: 0, 1: 0, 2: 0}
    for example in nshot_examples:
        tier_counts[example['tier']] += 1
    print(f"Tier 0: {tier_counts[0]}, Tier 1: {tier_counts[1]}, Tier 2: {tier_counts[2]}")
    
    balanced_examples = generator.balance_nshot_examples(nshot_examples)
    print("Balanced examples distribution:")
    balanced_tier_counts = {0: 0, 1: 0, 2: 0}
    for example in balanced_examples:
        balanced_tier_counts[example['tier']] += 1
    print(f"Tier 0: {balanced_tier_counts[0]}, Tier 1: {balanced_tier_counts[1]}, Tier 2: {balanced_tier_counts[2]}")
    
    # Test with balanced examples enabled
    print("\n=== WITH BALANCED EXAMPLES ===")
    system_prompt, user_prompt = generator.generate_prompts(
        target_case, nshot_examples, balance_examples=True
    )
    print("User Prompt with Balanced Examples:")
    print(user_prompt)
