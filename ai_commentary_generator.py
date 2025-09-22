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

# Configure OpenAI API logging to reduce verbosity
def configure_openai_logging():
    """Configure logging to reduce OpenAI API verbosity"""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("instructor").setLevel(logging.WARNING)
    logging.getLogger("httpx.connection").setLevel(logging.WARNING)
    logging.getLogger("httpx.transport").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpcore.connection").setLevel(logging.WARNING)
    logging.getLogger("httpcore.http11").setLevel(logging.WARNING)

# Apply logging configuration
configure_openai_logging()

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
            
            # Clean up any malformed HTML and ensure proper formatting
            commentary = self._clean_html_content(commentary)
            
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
            
            # Clean up any malformed HTML and ensure proper formatting
            summary = self._clean_html_content(summary)
            
            # Ensure proper HTML formatting
            if not summary.startswith('<div'):
                summary = f'<div class="executive-summary">{summary}</div>'
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return self._get_fallback_executive_summary(material_findings, trivial_findings_count)
    
    def _clean_html_content(self, content: str) -> str:
        """Clean and format HTML content to ensure proper rendering"""
        import re
        
        # Remove any markdown code blocks that might be wrapping the HTML
        content = re.sub(r'```html\s*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'```\s*$', '', content, flags=re.IGNORECASE)
        content = re.sub(r'^```\s*', '', content, flags=re.IGNORECASE)
        
        # Remove any remaining markdown artifacts
        content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
        
        # Clean up any extra whitespace
        content = content.strip()
        
        # Ensure proper HTML structure
        # If the content doesn't have proper HTML tags, wrap it appropriately
        if not re.search(r'<h[1-6]>', content):
            # If there are no headings, try to create them from the structure
            lines = content.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if this looks like a heading
                if any(keyword in line.lower() for keyword in ['financial services impact', 'customer experience implications', 'recommended actions', 'monitoring strategy']):
                    # This is likely a heading that got converted to plain text
                    if not line.startswith('<h4>'):
                        line = f'<h4>{line}</h4>'
                elif line.startswith('- ') or line.startswith('• '):
                    # This is likely a list item
                    if not line.startswith('<li>'):
                        line = f'<li>{line[2:]}</li>'
                elif not line.startswith('<'):
                    # This is likely a paragraph
                    if not line.startswith('<p>'):
                        line = f'<p>{line}</p>'
                
                cleaned_lines.append(line)
            
            content = '\n'.join(cleaned_lines)
        
        # Ensure list items are properly wrapped in ul tags
        if '<li>' in content and '<ul>' not in content:
            # Find the first <li> and wrap the list properly
            content = re.sub(r'(<li>.*?</li>)', r'<ul>\1</ul>', content, flags=re.DOTALL)
        
        return content
    
    def _build_finding_prompt(self, finding_data: Dict[str, Any]) -> str:
        """Build the prompt for finding-specific commentary."""
        return f"""Analyze this LLM bias finding for a bank complaint handling system:

Finding: {finding_data.get('finding', 'N/A')}
Test: {finding_data.get('test_name', 'N/A')}
Statistical Significance: p < {finding_data.get('p_value', 'N/A')}
Effect Size: {finding_data.get('effect_type', 'N/A')} = {finding_data.get('effect_size', 'N/A')}
Sample Size: n = {finding_data.get('sample_size', 'N/A')}

Current basic commentary: "{finding_data.get('implication', 'N/A')}"

CRITICAL: Avoid generic recommendations like "implement fairness audits" or "establish monitoring protocols" - focus on SPECIFIC actions related to this exact finding.

Generate enhanced commentary covering:
1. What this SPECIFIC bias pattern means for customer treatment and regulatory compliance
2. Specific risks for CFPB enforcement and Fair Lending Act violations based on this finding
3. Customer experience implications for the SPECIFIC demographic groups affected
4. Concrete mitigation strategies SPECIFIC to this bias pattern and test type
5. Targeted monitoring approaches for this specific finding

Focus on:
- SPECIFIC demographic groups mentioned in the finding
- SPECIFIC test methodology (zero-shot vs n-shot, question rates vs tier assignments, etc.)
- SPECIFIC effect sizes and their practical meaning
- SPECIFIC regulatory risks based on the bias pattern
- SPECIFIC mitigation strategies for this exact type of bias

IMPORTANT: Format your response as clean HTML without any markdown code blocks or extra formatting. Use the following exact structure:

<div class="enhanced-commentary">
    <h4>Financial Services Impact</h4>
    <p>[Detailed explanation of regulatory and operational risks SPECIFIC to this finding]</p>
    
    <h4>Customer Experience Implications</h4>
    <p>[Analysis of how this SPECIFIC bias affects the SPECIFIC demographic groups mentioned]</p>
    
    <h4>Recommended Actions</h4>
    <ul>
        <li>[SPECIFIC, prioritized recommendations for this exact bias pattern]</li>
    </ul>
    
    <h4>Monitoring Strategy</h4>
    <p>[SPECIFIC testing and validation approaches for this finding type]</p>
</div>

Do not wrap your response in markdown code blocks or add any extra formatting. Return only the HTML content."""
    
    def _build_executive_summary_prompt(self, material_findings: List[Dict], trivial_findings_count: int) -> str:
        """Build the prompt for executive summary generation."""
        findings_summary = "\n".join([
            f"- {finding.get('finding', 'N/A')} (Effect: {finding.get('effect_size', 'N/A')})"
            for finding in material_findings[:10]  # Limit to top 10 for prompt length
        ])
        
        # Analyze findings to identify specific patterns for targeted recommendations
        severity_bias_findings = [f for f in material_findings if 'severity' in f.get('finding', '').lower()]
        geographic_findings = [f for f in material_findings if any(geo in f.get('finding', '').lower() for geo in ['geographic', 'suburban', 'urban', 'rural'])]
        method_inconsistency_findings = [f for f in material_findings if any(method in f.get('finding', '').lower() for method in ['zero-shot', 'n-shot', 'method'])]
        process_bias_findings = [f for f in material_findings if 'question' in f.get('finding', '').lower()]
        
        return f"""Create a 3-paragraph executive summary for bank leadership analyzing {len(material_findings)} material bias findings and {trivial_findings_count} trivial findings from LLM fairness testing in complaint handling.

Key findings include:
{findings_summary}

CRITICAL: Avoid generic recommendations like "implement fairness audits" or "establish AI ethics committees" - these are likely already in place. Instead, provide SPECIFIC strategic recommendations based on the actual research findings.

Focus on:
Paragraph 1: Most important and novel bias patterns (emphasize severity-dependent bias, geographic disparities, method inconsistencies)
Paragraph 2: Financial services regulatory and operational implications
Paragraph 3: SPECIFIC strategic recommendations based on the research findings, such as:
- Prioritize governance of high-stakes decisions (where severity-dependent bias was found)
- Address process bias in addition to outcome bias (where question rate disparities were found)
- Expand bias testing beyond traditional demographics (where geographic/socioeconomic patterns emerged)
- Use effect size filtering to prioritize bias risks (given {trivial_findings_count} trivial findings)
- Address method-dependent bias inconsistencies (where zero-shot vs n-shot differences were found)

Audience: Bank executives, compliance officers, and AI governance committee
Tone: Professional, data-driven, actionable, SPECIFIC to the research findings
Length: ~150-200 words per paragraph

IMPORTANT: Format your response as clean HTML without any markdown code blocks or extra formatting. Use the following exact structure:

<div class="executive-summary">
    <h4>Key Findings Overview</h4>
    <p>[Paragraph 1 content]</p>
    
    <h4>Financial Services Industry Implications</h4>
    <p>[Paragraph 2 content]</p>
    
    <h4>Strategic Recommendations</h4>
    <p>[Paragraph 3 content - SPECIFIC to the research findings, not generic governance advice]</p>
</div>

Do not wrap your response in markdown code blocks or add any extra formatting. Return only the HTML content."""
    
    def _get_fallback_commentary(self, finding_data: Dict[str, Any]) -> str:
        """Provide fallback commentary when AI generation fails."""
        finding = finding_data.get('finding', 'N/A')
        effect_size = finding_data.get('effect_size', 0)
        effect_type = finding_data.get('effect_type', 'N/A')
        
        # Extract specific information from the finding
        if 'severity' in finding.lower():
            specific_context = "This severity-dependent bias pattern suggests the AI system treats high-stakes complaints differently, which could amplify regulatory risk for the most important cases."
        elif 'geographic' in finding.lower() or 'suburban' in finding.lower() or 'urban' in finding.lower():
            specific_context = "This geographic bias pattern indicates the AI system may be influenced by location-based factors that correlate with socioeconomic status, creating indirect discrimination risks."
        elif 'zero-shot' in finding.lower() or 'n-shot' in finding.lower():
            specific_context = "This method-dependent bias suggests the prompting approach significantly affects fairness outcomes, requiring careful prompt engineering and method selection."
        elif 'question' in finding.lower():
            specific_context = "This process bias in questioning behavior indicates the AI system's decision-making process itself is biased, not just the final outcomes."
        else:
            specific_context = f"This bias pattern with effect size {effect_size} ({effect_type}) indicates systematic differences in treatment that require targeted intervention."
        
        return f'''<div class="enhanced-commentary">
    <h4>Financial Services Impact</h4>
    <p>{specific_context} The effect size of {effect_size} ({effect_type}) suggests {self._get_effect_interpretation(effect_size, effect_type)} that could impact regulatory compliance and customer treatment.</p>
    
    <h4>Customer Experience Implications</h4>
    <p>The specific demographic groups and conditions identified in this finding may receive inconsistent treatment, potentially impacting customer trust and satisfaction in ways that correlate with protected characteristics.</p>
    
    <h4>Recommended Actions</h4>
    <ul>
        <li>Investigate the specific prompt engineering or model behavior causing this bias pattern</li>
        <li>Implement targeted monitoring for the specific demographic groups and conditions identified</li>
        <li>Conduct follow-up testing to validate mitigation efforts for this specific bias type</li>
        <li>Consider whether this finding indicates broader systemic issues requiring model retraining</li>
    </ul>
    
    <h4>Monitoring Strategy</h4>
    <p>Establish targeted monitoring protocols specifically for this bias pattern, including regular testing of the identified demographic groups and conditions to measure improvement over time.</p>
</div>'''
    
    def _get_fallback_executive_summary(self, material_findings: List[Dict], trivial_findings_count: int) -> str:
        """Provide fallback executive summary when AI generation fails."""
        # Analyze findings to provide specific insights
        severity_findings = [f for f in material_findings if 'severity' in f.get('finding', '').lower()]
        geographic_findings = [f for f in material_findings if any(geo in f.get('finding', '').lower() for geo in ['geographic', 'suburban', 'urban', 'rural'])]
        method_findings = [f for f in material_findings if any(method in f.get('finding', '').lower() for method in ['zero-shot', 'n-shot', 'method'])]
        process_findings = [f for f in material_findings if 'question' in f.get('finding', '').lower()]
        
        # Build specific insights
        specific_insights = []
        if severity_findings:
            specific_insights.append(f"severity-dependent bias affecting {len(severity_findings)} high-stakes decision patterns")
        if geographic_findings:
            specific_insights.append(f"geographic/socioeconomic bias patterns in {len(geographic_findings)} findings")
        if method_findings:
            specific_insights.append(f"method-dependent bias inconsistencies across {len(method_findings)} prompting approaches")
        if process_findings:
            specific_insights.append(f"process bias in questioning behavior affecting {len(process_findings)} decision pathways")
        
        insights_text = ", ".join(specific_insights) if specific_insights else "systematic bias patterns across multiple demographic and methodological dimensions"
        
        return f'''<div class="executive-summary">
    <h4>Key Findings Overview</h4>
    <p>Analysis of the LLM fairness testing revealed {len(material_findings)} material bias findings and {trivial_findings_count} statistically significant but trivial findings. The most concerning patterns include {insights_text}. Notably, the presence of {trivial_findings_count} trivial findings highlights the importance of effect size filtering to prioritize actionable bias risks.</p>
    
    <h4>Financial Services Industry Implications</h4>
    <p>These findings indicate specific regulatory compliance risks: severity-dependent bias amplifies risk for high-stakes decisions, geographic patterns create indirect discrimination exposure, and method inconsistencies suggest prompt engineering vulnerabilities. The combination creates compound regulatory risk that exceeds individual finding impacts.</p>
    
    <h4>Strategic Recommendations</h4>
    <p>Prioritize governance of high-stakes decisions where severity-dependent bias was found, address process bias in addition to outcome bias for question rate disparities, expand bias testing beyond traditional demographics to include geographic/socioeconomic factors, and implement effect size filtering to distinguish material risks from statistical noise. Focus on method-dependent bias inconsistencies by standardizing prompting approaches and testing both zero-shot and n-shot methods systematically.</p>
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
