# Bank Complaint Handling Fairness Analysis

A comprehensive framework for evaluating and mitigating bias in Large Language Models (LLMs) when processing consumer financial complaints. This project implements advanced fairness testing methodologies using real CFPB (Consumer Financial Protection Bureau) complaint data.

## 🎯 Overview

This project addresses a critical question in AI ethics: **Do LLMs exhibit demographic bias when making decisions about consumer complaints?** Using real-world financial complaint data, we test various models and mitigation strategies to understand and reduce algorithmic bias in regulatory decision-making.

## 🚀 Key Features

- **Real Data**: Uses actual CFPB consumer complaint narratives
- **Multi-Model Testing**: Supports OpenAI, Anthropic, and Google models
- **Advanced Bias Detection**: Implements granular inter-group bias analysis
- **Mitigation Strategies**: Tests 6+ different fairness-enhancing prompting approaches
- **Statistical Rigor**: Comprehensive statistical analysis with proper significance testing
- **Scalable Architecture**: Multi-threaded processing with proper resource management
- **Interactive Dashboard**: Streamlit-based visualization and analysis tools

## 📊 Research Contributions

### 1. Granular Bias Analysis
- Compares bias across multiple demographic personas
- Quantifies differential treatment between groups
- Provides actionable insights for bias mitigation

### 2. Competitive Mitigation Testing
- Tests 6 distinct fairness-enhancing strategies:
  - De-biasing instructions
  - Role-playing approaches
  - Consequentialist warnings
  - Perspective-taking prompts
  - Simplified ignore commands
  - Chain-of-thought reasoning

### 3. Process vs. Outcome Fairness
- Measures both decision outcomes and process fairness
- Analyzes confidence scores and questioning patterns
- Identifies subtle forms of algorithmic bias

### 4. Scaling Laws of Fairness
- Tests whether larger models exhibit more or less bias
- Provides insights into model capability vs. fairness trade-offs

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- API keys for LLM providers (OpenAI, Anthropic, Google)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd bank-complaint-handling-fairness

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
# OPENAI_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here
# GOOGLE_API_KEY=your_key_here
```

## 🏃‍♂️ Quick Start

### Basic Fairness Testing
```bash
# Run a simple fairness experiment
python fairness_harness.py --run-experiment --models gpt-4o-mini --sample-size 50

# Run advanced analysis with multiple models
python advanced_fairness_analysis.py --run-experiment --models gpt-4o-mini claude-3.5 --sample-size 100
```

### Full Pipeline
```bash
# 1. Ingest CFPB data
python complaints_llm_fairness_harness.py ingest --total 1000

# 2. Prepare demographic pairs
python complaints_llm_fairness_harness.py prepare --personas "white_male_affluent,black_female_urban,hispanic_male_working"

# 3. Run experiments
python complaints_llm_fairness_harness.py run --models gpt-4o-mini,claude-3.5 --repeats 3

# 4. Analyze results
python complaints_llm_fairness_harness.py analyse

# 5. Launch dashboard
python dashboard.py
```

## 📁 Project Structure

```
bank-complaint-handling-fairness/
├── complaints_llm_fairness_harness.py    # Core harness and data processing
├── advanced_fairness_analysis.py         # Advanced bias analysis and mitigation
├── fairness_harness.py                   # Interactive experiment management
├── dashboard.py                          # Streamlit visualization dashboard
├── requirements.txt                      # Python dependencies
├── advanced_analysis_specifications.md   # Research methodology details
├── out/                                 # Experiment outputs
│   ├── runs.jsonl                       # Raw experiment results
│   ├── cost_summary.json               # Cost tracking
│   └── analysis.json                   # Statistical analysis results
├── advanced_results/                    # Advanced analysis outputs
│   ├── plots/                          # Generated visualizations
│   ├── granular_bias_analysis.json     # Detailed bias metrics
│   └── research_summary.md             # Research findings
├── nshot_results/                       # N-shot prompting experiments
├── data_cache/                          # LLM response caching
└── analysis_cache/                      # Analysis result caching
```

## 🔬 Research Methodology

### Data Sources
- **CFPB Consumer Complaint Database**: Real consumer financial complaints
- **Demographic Personas**: Carefully constructed demographic profiles
- **Stratified Sampling**: Ensures representative complaint distribution

### Bias Measurement
- **Outcome Fairness**: Remedy tier assignments and monetary relief decisions
- **Process Fairness**: Confidence scores and questioning patterns
- **Statistical Tests**: Wilcoxon signed-rank, McNemar, and bootstrap tests

### Mitigation Strategies
1. **Structured Extraction**: Fact-based decision making
2. **Roleplay**: Impartial auditor perspective
3. **Consequentialist**: Regulatory audit warnings
4. **Perspective-Taking**: Multi-stakeholder consideration
5. **Minimal**: Simple bias avoidance instructions
6. **Chain-of-Thought**: Step-by-step reasoning

## 📈 Key Findings

### Model Performance
- **Accuracy**: ~51% exact match with ground truth decisions
- **Bias Detection**: Significant demographic bias across all tested models
- **Mitigation Effectiveness**: 4 out of 6 strategies show statistically significant bias reduction

### Persona-Specific Results
- **White Male Affluent**: +0.099 tier shift (most favorable treatment)
- **Asian Female Professional**: +0.039 tier shift
- **White Female Senior**: +0.058 tier shift
- **White Male Rural**: +0.072 tier shift

### Statistical Significance
- Persona injection significantly affects decisions (p = 0.0015)
- Multiple mitigation strategies show significant bias reduction
- Process fairness varies significantly across demographic groups

## 🎛️ Configuration

### Model Support
- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-5
- **Anthropic**: Claude-3.5, Claude-Opus-4.1
- **Google**: Gemini-2.5

### Threading Configuration
- **Default**: 5 threads per model
- **Configurable**: Adjust via `--threads-per-model` parameter
- **Resource Management**: Automatic thread pool cleanup

### Caching
- **LLM Responses**: Cached to reduce API costs
- **Analysis Results**: Cached for faster re-analysis
- **Cost Tracking**: Detailed cost breakdowns per model

## 📊 Visualization

### Interactive Dashboard
```bash
python dashboard.py
```

Features:
- Real-time experiment monitoring
- Bias visualization across personas
- Statistical significance testing
- Cost tracking and optimization

### Generated Plots
- Bias distribution heatmaps
- Remedy tier distributions
- Persona-specific bias patterns
- Mitigation strategy effectiveness
- Process fairness analysis

## 🔧 Advanced Usage

### Custom Personas
```python
# Define custom demographic personas
CUSTOM_PERSONAS = {
    "custom_persona": {
        "demographics": "Custom demographic description",
        "background": "Relevant background information"
    }
}
```

### Custom Mitigation Strategies
```python
# Implement custom fairness strategies
def custom_fairness_prompt():
    return "Your custom fairness instruction here"
```

### Batch Processing
```bash
# Process multiple model combinations
python advanced_fairness_analysis.py --full --models gpt-4o-mini claude-3.5 gemini-2.5
```

## 📚 Research Applications

### Academic Research
- Algorithmic bias in financial services
- LLM fairness evaluation methodologies
- Mitigation strategy effectiveness
- Scaling laws of fairness

### Industry Applications
- Regulatory compliance testing
- AI system auditing
- Bias mitigation implementation
- Fairness monitoring systems

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style and standards
- Testing requirements
- Documentation updates
- Research methodology improvements

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **CFPB**: For providing the consumer complaint database
- **Research Community**: For foundational work in AI fairness
- **Open Source**: For the excellent tools and libraries used

## 📞 Support

For questions, issues, or collaboration opportunities:
- Open an issue on GitHub
- Contact the research team
- Check the documentation in `advanced_analysis_specifications.md`

---

**Note**: This research uses de-identified complaint data published under CFPB narrative policy. All decisions measured are simulated and for research purposes only.
