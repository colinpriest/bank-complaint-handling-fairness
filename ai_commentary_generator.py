#!/usr/bin/env python3
"""
AI Commentary Generator for LLM Fairness Dashboard

This module provides ChatGPT-powered commentary generation for statistical findings
in the bank complaint handling fairness analysis dashboard.
"""

import openai
import os
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AICommentaryGenerator:
    """
    Generates enhanced commentary for fairness analysis findings using ChatGPT.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo-preview"):
        """
        Initialize the AI commentary generator.
        
        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            model: OpenAI model to use for commentary generation
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
        
        # System prompt for consistent commentary generation
        self.system_prompt = """You are a senior financial services compliance and AI ethics expert analyzing LLM bias in bank complaint handling systems. Your role is to interpret statistical findings for executive audiences, focusing on regulatory risk, customer impact, and actionable mitigation strategies.

Context: A major bank is using LLMs to process customer complaints and determine appropriate responses (no action, non-monetary remedy, or monetary compensation). The system has been tested for fairness across demographic groups and different prompting methods.

Guidelines:
- Focus on practical business and regulatory implications
- Assume readers understand banking but may not be AI/statistics experts
- Prioritize actionable recommendations over technical details
- Consider CFPB enforcement priorities and Fair Lending Act requirements
- Address both immediate risks and long-term AI governance needs
- Use professional, data-driven tone
- Provide specific, prioritized recommendations"""

    def generate_enhanced_commentary(self, finding_data: Dict[str, Any]) -> str:
        """
        Generate enhanced commentary for a specific finding.
        
        Args:
            finding_data: Dictionary containing finding information
            
        Returns:
            HTML-formatted enhanced commentary
        """
        try:
            prompt = self._build_finding_prompt(finding_data)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent output
                max_tokens=1000
            )
            
            commentary = response.choices[0].message.content.strip()
            
            # Ensure the commentary is properly formatted as HTML
            if not commentary.startswith('<div'):
                commentary = f'<div class="enhanced-commentary">{commentary}</div>'
            
            return commentary
            
        except Exception as e:
            logger.error(f"Error generating enhanced commentary: {e}")
            return self._get_fallback_commentary(finding_data)
    
    def generate_executive_summary(self, material_findings: List[Dict], trivial_findings_count: int) -> str:
        """
        Generate executive summary for all findings.
        
        Args:
            material_findings: List of material findings
            trivial_findings_count: Number of trivial findings
            
        Returns:
            HTML-formatted executive summary
        """
        try:
            prompt = self._build_executive_summary_prompt(material_findings, trivial_findings_count)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1200
            )
            
            summary = response.choices[0].message.content.strip()
            
            # Ensure proper HTML formatting
            if not summary.startswith('<div'):
                summary = f'<div class="executive-summary">{summary}</div>'
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return self._get_fallback_executive_summary(material_findings, trivial_findings_count)
    
    def _build_finding_prompt(self, finding_data: Dict[str, Any]) -> str:
        """Build the prompt for finding-specific commentary."""
        return f"""Analyze this LLM bias finding for a bank complaint handling system:

Finding: {finding_data.get('finding', 'N/A')}
Test: {finding_data.get('test_name', 'N/A')}
Statistical Significance: p < {finding_data.get('p_value', 'N/A')}
Effect Size: {finding_data.get('effect_type', 'N/A')} = {finding_data.get('effect_size', 'N/A')}
Sample Size: n = {finding_data.get('sample_size', 'N/A')}

Current basic commentary: "{finding_data.get('implication', 'N/A')}"

Generate enhanced commentary covering:
1. What this bias means for customer treatment and regulatory compliance
2. Specific risks for CFPB enforcement and Fair Lending Act violations
3. Customer experience implications for affected demographic groups
4. Concrete mitigation strategies with implementation priorities
5. Ongoing monitoring and testing recommendations

Format as HTML with clear headings and actionable bullet points. Use the following structure:
<div class="enhanced-commentary">
    <h4>Financial Services Impact</h4>
    <p>[Detailed explanation of regulatory and operational risks]</p>
    
    <h4>Customer Experience Implications</h4>
    <p>[Analysis of how this bias affects different customer groups]</p>
    
    <h4>Recommended Actions</h4>
    <ul>
        <li>[Specific, prioritized recommendations]</li>
    </ul>
    
    <h4>Monitoring Strategy</h4>
    <p>[Ongoing testing and validation approaches]</p>
</div>"""
    
    def _build_executive_summary_prompt(self, material_findings: List[Dict], trivial_findings_count: int) -> str:
        """Build the prompt for executive summary generation."""
        findings_summary = "\n".join([
            f"- {finding.get('finding', 'N/A')} (Effect: {finding.get('effect_size', 'N/A')})"
            for finding in material_findings[:10]  # Limit to top 10 for prompt length
        ])
        
        return f"""Create a 3-paragraph executive summary for bank leadership analyzing {len(material_findings)} material bias findings and {trivial_findings_count} trivial findings from LLM fairness testing in complaint handling.

Key findings include:
{findings_summary}

Focus on:
Paragraph 1: Most important and novel bias patterns (emphasize severity-dependent bias, geographic disparities, method inconsistencies)
Paragraph 2: Financial services regulatory and operational implications
Paragraph 3: Strategic action plan with prioritized recommendations

Audience: Bank executives, compliance officers, and AI governance committee
Tone: Professional, data-driven, actionable
Length: ~150-200 words per paragraph

Format as HTML with clear paragraph structure:
<div class="executive-summary">
    <h4>Key Findings Overview</h4>
    <p>[Paragraph 1 content]</p>
    
    <h4>Financial Services Industry Implications</h4>
    <p>[Paragraph 2 content]</p>
    
    <h4>Strategic Recommendations</h4>
    <p>[Paragraph 3 content]</p>
</div>"""
    
    def _get_fallback_commentary(self, finding_data: Dict[str, Any]) -> str:
        """Provide fallback commentary when AI generation fails."""
        return f'''<div class="enhanced-commentary">
    <h4>Financial Services Impact</h4>
    <p>This finding indicates potential bias in the complaint handling system that could affect regulatory compliance and customer treatment.</p>
    
    <h4>Customer Experience Implications</h4>
    <p>Different demographic groups may receive inconsistent treatment, potentially impacting customer trust and satisfaction.</p>
    
    <h4>Recommended Actions</h4>
    <ul>
        <li>Review and update prompt engineering for this specific test case</li>
        <li>Implement additional monitoring for the affected demographic groups</li>
        <li>Conduct follow-up testing to validate mitigation efforts</li>
    </ul>
    
    <h4>Monitoring Strategy</h4>
    <p>Establish ongoing monitoring protocols to track this bias pattern and measure improvement over time.</p>
</div>'''
    
    def _get_fallback_executive_summary(self, material_findings: List[Dict], trivial_findings_count: int) -> str:
        """Provide fallback executive summary when AI generation fails."""
        return f'''<div class="executive-summary">
    <h4>Key Findings Overview</h4>
    <p>Analysis of the LLM fairness testing revealed {len(material_findings)} material bias findings and {trivial_findings_count} statistically significant but trivial findings. The most concerning patterns include demographic disparities in question rates and severity-dependent bias effects.</p>
    
    <h4>Financial Services Industry Implications</h4>
    <p>These findings indicate potential regulatory compliance risks, particularly around Fair Lending Act requirements and CFPB enforcement priorities. The bias patterns could impact customer trust and operational risk management.</p>
    
    <h4>Strategic Recommendations</h4>
    <p>Immediate action is required to address material findings through prompt engineering improvements and bias mitigation strategies. Long-term governance should include ongoing monitoring, regular fairness testing, and integration of AI ethics into operational procedures.</p>
</div>'''


def extract_findings_for_commentary(collector_results: List[Dict]) -> tuple[List[Dict], int]:
    """
    Extract and categorize findings for commentary generation.
    
    Args:
        collector_results: Results from StatisticalResultCollector
        
    Returns:
        Tuple of (material_findings, trivial_findings_count)
    """
    material_findings = []
    trivial_findings_count = 0
    
    for result in collector_results:
        # Determine if finding is material based on effect size and significance
        effect_size = result.get('effect_size', 0)
        p_value = result.get('p_value', 1)
        
        # Consider findings material if they have significant effect sizes
        # This is a simplified heuristic - could be made more sophisticated
        is_material = (
            p_value < 0.05 and (
                (result.get('effect_type') == 'equity_deficit' and effect_size > 0.2) or
                (result.get('effect_type') == 'cohens_d' and abs(effect_size) > 0.2) or
                (result.get('effect_type') == 'eta_squared' and effect_size > 0.01) or
                (result.get('effect_type') == 'disparity_ratio' and effect_size > 0.2)
            )
        )
        
        if is_material:
            material_findings.append(result)
        else:
            trivial_findings_count += 1
    
    return material_findings, trivial_findings_count


if __name__ == "__main__":
    # Test the commentary generator
    generator = AICommentaryGenerator()
    
    # Test finding
    test_finding = {
        'finding': 'Total disparity (12.1% vs 0%) difference in question rates (White vs Black)',
        'test_name': 'Question Rate Equity by Ethnicity: N-Shot',
        'p_value': 0.001,
        'effect_size': 1.000,
        'effect_type': 'equity_deficit',
        'sample_size': 10000,
        'implication': 'SEVERE question rate inequity detected'
    }
    
    print("Testing enhanced commentary generation...")
    commentary = generator.generate_enhanced_commentary(test_finding)
    print(commentary)
    
    print("\nTesting executive summary generation...")
    summary = generator.generate_executive_summary([test_finding], 5)
    print(summary)
