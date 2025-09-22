#!/usr/bin/env python3
"""
Static Tier 0 examples that demonstrate clear decision boundaries.
These examples teach the LLM when NOT to take action.
"""

# Static examples based on the misclassification analysis
STATIC_TIER_0_EXAMPLES = [
    {
        'complaint_text': """I went on line with Equifax Credit Bureau to dispute a incorrect item on my credit report and instead of starting an investigation Equifax sent me a letter stating that they were putting a "fraud Alert" on my credit profile and notified the other credit reporting bureaus to do the same. I did not request Equifax to put a "Fraud Alert" on my credit profile nor did I give them permission. I want Equifax to remove the "Fraud Alert" immediately.""",
        'tier': 0,
        'reasoning': "Third-party issue - Equifax is a credit bureau, not us. We have no control over their processes.",
        'category': 'third_party',
        'product': 'Credit reporting',
        'issue': "Credit reporting company's investigation"
    },
    {
        'complaint_text': """I signed up for the 30 day free trial period with your credit monitoring service, and when I signed up, you charged me three separate charges instead of the free trial as advertised. When I spoke with customer service, they explained that the free trial only applies to new customers and I had an account 5 years ago. This was not clearly stated in the advertisement.""",
        'tier': 0,
        'reasoning': "Customer misunderstood the policy - free trials are only for new customers as stated in terms and conditions.",
        'category': 'customer_misunderstanding',
        'product': 'Credit monitoring',
        'issue': 'Account terms and changes'
    },
    {
        'complaint_text': """I have been in bankruptcy proceedings since 2016 and Bank of America is sending foreclosure notices despite my bankruptcy attorney handling all communications. I have provided all documentation to my attorney who filed motions with the bankruptcy court. The court has scheduled a hearing next month to address the bank's claims about missed payments.""",
        'tier': 0,
        'reasoning': "Already being handled through legal channels - bankruptcy court and attorneys are managing this matter.",
        'category': 'legal_proceedings',
        'product': 'Mortgage',
        'issue': 'Loan modification,collection,foreclosure'
    }
]

def get_static_tier0_examples():
    """
    Returns the static Tier 0 examples with their reasoning.

    Returns:
        List of dictionaries containing static Tier 0 examples
    """
    return STATIC_TIER_0_EXAMPLES

def format_static_example_for_prompt(example, index):
    """
    Format a static example for inclusion in the n-shot prompt.

    Args:
        example: Dictionary containing example data
        index: Example number for display

    Returns:
        Formatted string for the prompt
    """
    return f"""Example {index}:
Complaint: {example['complaint_text']}
Tier: {example['tier']}
Reasoning: {example['reasoning']}
"""

def get_formatted_static_examples(start_index=1):
    """
    Get all static examples formatted for prompt inclusion.

    Args:
        start_index: Starting index for example numbering

    Returns:
        List of formatted example strings
    """
    formatted = []
    for i, example in enumerate(STATIC_TIER_0_EXAMPLES, start=start_index):
        formatted.append(format_static_example_for_prompt(example, i))
    return formatted

def explain_tier0_boundaries():
    """
    Returns an explanation of Tier 0 boundaries for system prompts.

    Returns:
        String explaining when Tier 0 is appropriate
    """
    return """IMPORTANT: Tier 0 (No Action Required) applies when:
1. The complaint is about a third-party company (like Equifax, Experian, TransUnion) that we don't control
2. The customer has misunderstood a clearly stated policy or made an error
3. The matter is already being handled through proper legal channels (courts, attorneys, bankruptcy)
4. The complaint lacks merit or the bank followed proper procedures
5. The issue has already been resolved or addressed

Remember: Just because a customer is upset doesn't mean the bank needs to take action. Evaluate whether the bank has actual responsibility and ability to address the issue."""