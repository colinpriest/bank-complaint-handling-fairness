# Headline Results Tab Specifications

## Overview

The Headline Results tab provides an executive summary of all statistical findings from the analysis, categorized by both statistical significance and practical importance. This tab automatically aggregates results as they are calculated across all other tabs, providing a clear distinction between findings that are both significant and material versus those that are statistically significant but practically trivial.

## Tab Structure

### Main Tab: "Headline Results"
Position: First tab in the dashboard (highest visibility)

### Sub-Tabs:
1. **"Statistically Significant and Material"** - Results that matter in practice
2. **"Statistically Significant but Trivial"** - Results that are artifacts of large sample size

## Data Collection and Aggregation

### Result Collection System

Each statistical test throughout the dashboard must register its results with a central collector:

```python
class StatisticalResultCollector:
    def __init__(self):
        self.results = {
            'material': [],      # Significant and material results
            'trivial': [],       # Significant but trivial results
            'non_significant': [] # Not significant (not displayed in headline)
        }

    def add_result(self, result_data):
        """
        Add a statistical test result to the collector

        Args:
            result_data: dict containing:
                - source_tab: str (e.g., "Bias Analysis", "Accuracy Analysis")
                - source_subtab: str (e.g., "Persona Comparison", "Strategy Effectiveness")
                - test_name: str (descriptive name of what was tested)
                - test_type: str (e.g., "paired_t_test", "chi_squared")
                - p_value: float
                - effect_size: float
                - effect_type: str (e.g., "cohens_d", "cramers_v")
                - sample_size: int
                - finding: str (human-readable description)
                - implication: str (what this means for fairness)
                - timestamp: datetime (when calculated)
        """
        category = self._categorize_result(result_data)
        self.results[category].append(result_data)

    def _categorize_result(self, result):
        """Categorize result based on p-value and effect size"""
        if result['p_value'] >= 0.05:
            return 'non_significant'

        # Determine materiality based on effect size type
        if result['effect_type'] == 'cohens_d':
            is_material = abs(result['effect_size']) >= 0.2
        elif result['effect_type'] == 'cramers_v':
            is_material = result['effect_size'] >= 0.1
        elif result['effect_type'] == 'eta_squared':
            is_material = result['effect_size'] >= 0.01
        else:
            # Conservative: treat unknown effect sizes as material
            is_material = True

        return 'material' if is_material else 'trivial'
```

### Integration Points

Every location in `html_dashboard.py` that performs a statistical test must be updated:

1. **Line ~1689**: Paired t-test for baseline vs persona-injected
2. **Line ~1805**: Chi-squared test for tier distributions
3. **Line ~1908**: Chi-squared test for geographic patterns
4. **Lines 2229-2397**: Chi-squared tests for tier analysis
5. **Lines 2902-4535**: Generic statistical tests from analysis results

Each integration follows this pattern:

```python
# After calculating p_value and effect_size
result_data = {
    'source_tab': 'Bias Analysis',
    'source_subtab': 'Baseline vs Persona',
    'test_name': 'Mean Tier Difference: Baseline vs African American Urban Male',
    'test_type': 'paired_t_test',
    'p_value': p_value,
    'effect_size': cohens_d,
    'effect_type': 'cohens_d',
    'sample_size': len(data),
    'finding': f"African American Urban Male personas receive {abs(mean_diff):.2f} higher tier assignments",
    'implication': "Model shows bias favoring this demographic in complaint resolution",
    'timestamp': datetime.now()
}
collector.add_result(result_data)
```

## UI/UX Design Specifications

### Tab Navigation HTML Structure

```html
<div class="tab">
    <button class="tablinks active" onclick="openTab(event, 'HeadlineResults')">Headline Results</button>
    <button class="tablinks" onclick="openTab(event, 'BiasAnalysis')">Bias Analysis</button>
    <!-- Other existing tabs -->
</div>

<div id="HeadlineResults" class="tabcontent" style="display:block;">
    <div class="sub-tabs">
        <button class="sub-tablinks active" onclick="openSubTab(event, 'MaterialFindings', 'HeadlineResults')">
            Statistically Significant and Material
        </button>
        <button class="sub-tablinks" onclick="openSubTab(event, 'TrivialFindings', 'HeadlineResults')">
            Statistically Significant but Trivial
        </button>
    </div>

    <div id="MaterialFindings" class="sub-tab-content active">
        <!-- Material findings content -->
    </div>

    <div id="TrivialFindings" class="sub-tab-content">
        <!-- Trivial findings content -->
    </div>
</div>
```

### Sub-Tab 1: Statistically Significant and Material

#### Header Section
```html
<div class="headline-header">
    <h2>Key Findings with Practical Importance</h2>
    <p class="summary-stats">
        <span class="stat-highlight">{count}</span> findings that are both statistically significant and practically important
    </p>
    <p class="explanation">
        These results represent real, meaningful differences that impact fairness in complaint handling.
    </p>
</div>
```

#### Results Display Format

Each result is displayed as a card with clear visual hierarchy:

```html
<div class="headline-result-card material">
    <div class="result-header">
        <span class="result-number">#{index}</span>
        <span class="source-badge">{source_tab} → {source_subtab}</span>
        <span class="effect-badge large-effect">Large Effect</span>
    </div>

    <div class="result-content">
        <h3 class="finding">{finding}</h3>

        <div class="statistics-row">
            <div class="stat-item">
                <span class="stat-label">Test:</span>
                <span class="stat-value">{test_name}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">p-value:</span>
                <span class="stat-value significant">{p_value:.4f}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Effect Size:</span>
                <span class="stat-value">{effect_type} = {effect_size:.3f}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Sample:</span>
                <span class="stat-value">n = {sample_size}</span>
            </div>
        </div>

        <div class="implication-box">
            <strong>What this means:</strong>
            <p>{implication}</p>
        </div>

        <div class="action-links">
            <a href="#{source_tab_id}" onclick="navigateToSource('{source_tab_id}', '{source_subtab_id}')">
                View Details →
            </a>
        </div>
    </div>
</div>
```

#### Sorting and Filtering

```html
<div class="headline-controls">
    <div class="sort-controls">
        <label>Sort by:</label>
        <select id="material-sort">
            <option value="effect_size_desc">Effect Size (Largest First)</option>
            <option value="effect_size_asc">Effect Size (Smallest First)</option>
            <option value="p_value_asc">P-Value (Most Significant First)</option>
            <option value="recent">Most Recent First</option>
            <option value="source">By Source Tab</option>
        </select>
    </div>

    <div class="filter-controls">
        <label>Filter by source:</label>
        <select id="material-filter">
            <option value="all">All Sources</option>
            <option value="bias">Bias Analysis</option>
            <option value="accuracy">Accuracy Analysis</option>
            <option value="severity">Severity Analysis</option>
            <option value="geographic">Geographic Analysis</option>
        </select>
    </div>
</div>
```

### Sub-Tab 2: Statistically Significant but Trivial

#### Header Section
```html
<div class="headline-header">
    <h2>Statistically Significant but Practically Trivial Findings</h2>
    <p class="summary-stats">
        <span class="stat-highlight">{count}</span> findings that are statistically significant but have negligible practical impact
    </p>
    <p class="explanation warning">
        <strong>⚠️ Interpretation Warning:</strong> These results likely reflect large sample sizes detecting tiny differences that don't meaningfully impact fairness. They should generally not drive decision-making.
    </p>
</div>
```

#### Results Display Format

Similar card structure but with visual de-emphasis:

```html
<div class="headline-result-card trivial">
    <div class="result-header">
        <span class="result-number">#{index}</span>
        <span class="source-badge">{source_tab} → {source_subtab}</span>
        <span class="effect-badge trivial-effect">Negligible Effect</span>
    </div>

    <div class="result-content collapsed">
        <h3 class="finding">{finding}</h3>

        <div class="trivial-warning">
            <span class="warning-icon">⚠️</span>
            <span>Small effect size ({effect_size:.3f}) suggests minimal practical importance</span>
        </div>

        <div class="statistics-row">
            <!-- Same statistics as material results -->
        </div>

        <div class="explanation-box">
            <strong>Why this is likely trivial:</strong>
            <p>With n = {sample_size}, even tiny differences become statistically significant.
               The effect size of {effect_size:.3f} indicates this difference is too small to matter in practice.</p>
        </div>
    </div>
</div>
```

## CSS Styling Specifications

```css
/* Headline Results Tab Styling */
.headline-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 8px;
    margin-bottom: 2rem;
}

.stat-highlight {
    font-size: 2.5rem;
    font-weight: bold;
    display: inline-block;
    margin-right: 0.5rem;
}

/* Result Cards */
.headline-result-card {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    margin-bottom: 1.5rem;
    overflow: hidden;
    transition: box-shadow 0.3s;
}

.headline-result-card.material {
    border-left: 4px solid #4CAF50;
}

.headline-result-card.trivial {
    border-left: 4px solid #FFC107;
    opacity: 0.85;
}

.headline-result-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

/* Effect Badges */
.effect-badge {
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.85rem;
    font-weight: 600;
}

.effect-badge.large-effect {
    background: #4CAF50;
    color: white;
}

.effect-badge.medium-effect {
    background: #2196F3;
    color: white;
}

.effect-badge.trivial-effect {
    background: #FFC107;
    color: #333;
}

/* Statistics Row */
.statistics-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    padding: 1rem;
    background: #f5f5f5;
    border-radius: 4px;
    margin: 1rem 0;
}

.stat-item {
    display: flex;
    flex-direction: column;
}

.stat-label {
    font-size: 0.85rem;
    color: #666;
    margin-bottom: 4px;
}

.stat-value {
    font-weight: 600;
    color: #333;
}

.stat-value.significant {
    color: #4CAF50;
}

/* Implication Box */
.implication-box {
    background: #E8F5E9;
    padding: 1rem;
    border-radius: 4px;
    margin: 1rem 0;
    border-left: 3px solid #4CAF50;
}

/* Trivial Warning */
.trivial-warning {
    background: #FFF3E0;
    padding: 0.75rem;
    border-radius: 4px;
    margin: 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    border: 1px solid #FFB74D;
}

/* Controls */
.headline-controls {
    display: flex;
    gap: 2rem;
    margin-bottom: 2rem;
    padding: 1rem;
    background: #f8f8f8;
    border-radius: 4px;
}

/* Responsive Design */
@media (max-width: 768px) {
    .statistics-row {
        grid-template-columns: 1fr;
    }

    .headline-controls {
        flex-direction: column;
        gap: 1rem;
    }
}
```

## JavaScript Functionality

### Navigation to Source

```javascript
function navigateToSource(tabId, subTabId) {
    // Close headline results
    document.getElementById('HeadlineResults').style.display = 'none';

    // Open target tab
    const targetTab = document.getElementById(tabId);
    if (targetTab) {
        targetTab.style.display = 'block';

        // Open target sub-tab if specified
        if (subTabId) {
            const subTab = document.getElementById(subTabId);
            if (subTab) {
                // Remove active class from all sub-tabs in target
                targetTab.querySelectorAll('.sub-tab-content').forEach(content => {
                    content.classList.remove('active');
                });

                // Add active class to target sub-tab
                subTab.classList.add('active');

                // Scroll to the specific result if possible
                setTimeout(() => {
                    subTab.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }, 100);
            }
        }
    }

    // Update tab button states
    document.querySelectorAll('.tablinks').forEach(button => {
        button.classList.remove('active');
        if (button.getAttribute('onclick').includes(tabId)) {
            button.classList.add('active');
        }
    });
}
```

### Sorting and Filtering

```javascript
document.getElementById('material-sort').addEventListener('change', function() {
    const sortBy = this.value;
    const container = document.querySelector('#MaterialFindings .results-container');
    const cards = Array.from(container.querySelectorAll('.headline-result-card'));

    cards.sort((a, b) => {
        const aData = JSON.parse(a.dataset.resultData);
        const bData = JSON.parse(b.dataset.resultData);

        switch(sortBy) {
            case 'effect_size_desc':
                return Math.abs(bData.effect_size) - Math.abs(aData.effect_size);
            case 'effect_size_asc':
                return Math.abs(aData.effect_size) - Math.abs(bData.effect_size);
            case 'p_value_asc':
                return aData.p_value - bData.p_value;
            case 'recent':
                return new Date(bData.timestamp) - new Date(aData.timestamp);
            case 'source':
                return aData.source_tab.localeCompare(bData.source_tab);
            default:
                return 0;
        }
    });

    // Re-append sorted cards
    cards.forEach(card => container.appendChild(card));
});
```

## Implementation in html_dashboard.py

### Add to __init__ method:
```python
def __init__(self, analysis_results, output_path="fairness_report.html"):
    self.analysis_results = analysis_results
    self.output_path = output_path
    self.collector = StatisticalResultCollector()  # Initialize collector
```

### Add Headline Results Tab Generation:

```python
def _generate_headline_results_tab(self) -> str:
    """Generate the Headline Results tab with material and trivial findings"""

    # Generate material findings sub-tab
    material_html = self._generate_material_findings_subtab()

    # Generate trivial findings sub-tab
    trivial_html = self._generate_trivial_findings_subtab()

    return f'''
    <div id="HeadlineResults" class="tabcontent" style="display:block;">
        <div class="sub-tabs">
            <button class="sub-tablinks active"
                    onclick="openSubTab(event, 'MaterialFindings', 'HeadlineResults')">
                Statistically Significant and Material ({len(self.collector.results['material'])})
            </button>
            <button class="sub-tablinks"
                    onclick="openSubTab(event, 'TrivialFindings', 'HeadlineResults')">
                Statistically Significant but Trivial ({len(self.collector.results['trivial'])})
            </button>
        </div>

        <div id="MaterialFindings" class="sub-tab-content active">
            {material_html}
        </div>

        <div id="TrivialFindings" class="sub-tab-content">
            {trivial_html}
        </div>
    </div>
    '''

def _generate_material_findings_subtab(self) -> str:
    """Generate the material findings sub-tab content"""
    material_results = sorted(
        self.collector.results['material'],
        key=lambda x: abs(x['effect_size']),
        reverse=True
    )

    if not material_results:
        return '''
        <div class="headline-header">
            <h2>No Significant and Material Findings</h2>
            <p>No statistical tests yielded results that were both significant and practically important.</p>
        </div>
        '''

    # Build header
    html = f'''
    <div class="headline-header">
        <h2>Key Findings with Practical Importance</h2>
        <p class="summary-stats">
            <span class="stat-highlight">{len(material_results)}</span>
            findings that are both statistically significant and practically important
        </p>
        <p class="explanation">
            These results represent real, meaningful differences that impact fairness in complaint handling.
        </p>
    </div>
    '''

    # Add controls
    html += '''
    <div class="headline-controls">
        <div class="sort-controls">
            <label>Sort by:</label>
            <select id="material-sort">
                <option value="effect_size_desc">Effect Size (Largest First)</option>
                <option value="effect_size_asc">Effect Size (Smallest First)</option>
                <option value="p_value_asc">P-Value (Most Significant First)</option>
                <option value="recent">Most Recent First</option>
                <option value="source">By Source Tab</option>
            </select>
        </div>
    </div>
    '''

    # Add results container
    html += '<div class="results-container">'

    for i, result in enumerate(material_results, 1):
        html += self._generate_result_card(i, result, 'material')

    html += '</div>'
    return html

def _generate_result_card(self, index: int, result: Dict, category: str) -> str:
    """Generate a single result card"""

    # Determine effect magnitude badge
    effect_badge_class = ''
    effect_badge_text = ''

    if result['effect_type'] == 'cohens_d':
        if abs(result['effect_size']) >= 0.8:
            effect_badge_class = 'large-effect'
            effect_badge_text = 'Large Effect'
        elif abs(result['effect_size']) >= 0.5:
            effect_badge_class = 'medium-effect'
            effect_badge_text = 'Medium Effect'
        else:
            effect_badge_class = 'small-effect'
            effect_badge_text = 'Small Effect'
    elif result['effect_type'] == 'cramers_v':
        if result['effect_size'] >= 0.3:
            effect_badge_class = 'large-effect'
            effect_badge_text = 'Large Effect'
        elif result['effect_size'] >= 0.1:
            effect_badge_class = 'medium-effect'
            effect_badge_text = 'Medium Effect'
        else:
            effect_badge_class = 'small-effect'
            effect_badge_text = 'Small Effect'

    # Format p-value display
    p_value_display = f"{result['p_value']:.4f}" if result['p_value'] >= 0.0001 else "< 0.0001"

    # Generate card HTML
    card_html = f'''
    <div class="headline-result-card {category}"
         data-result-data='{json.dumps(result)}'>
        <div class="result-header">
            <span class="result-number">#{index}</span>
            <span class="source-badge">{result['source_tab']} → {result['source_subtab']}</span>
            <span class="effect-badge {effect_badge_class}">{effect_badge_text}</span>
        </div>

        <div class="result-content">
            <h3 class="finding">{result['finding']}</h3>

            <div class="statistics-row">
                <div class="stat-item">
                    <span class="stat-label">Test:</span>
                    <span class="stat-value">{result['test_name']}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">p-value:</span>
                    <span class="stat-value significant">{p_value_display}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Effect Size:</span>
                    <span class="stat-value">{result['effect_type']} = {result['effect_size']:.3f}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Sample:</span>
                    <span class="stat-value">n = {result['sample_size']}</span>
                </div>
            </div>
    '''

    if category == 'material':
        card_html += f'''
            <div class="implication-box">
                <strong>What this means:</strong>
                <p>{result['implication']}</p>
            </div>
        '''
    else:  # trivial
        card_html += f'''
            <div class="trivial-warning">
                <span class="warning-icon">⚠️</span>
                <span>Small effect size ({result['effect_size']:.3f}) suggests minimal practical importance</span>
            </div>

            <div class="explanation-box">
                <strong>Why this is likely trivial:</strong>
                <p>With n = {result['sample_size']}, even tiny differences become statistically significant.
                   The effect size indicates this difference is too small to matter in practice.</p>
            </div>
        '''

    card_html += '''
        </div>
    </div>
    '''

    return card_html
```

## Integration Timeline

### Phase 1: Infrastructure (Immediate)
1. Add `StatisticalResultCollector` class to `html_dashboard.py`
2. Initialize collector in `__init__` method
3. Add headline results tab generation methods

### Phase 2: Integration (Next)
1. Update all statistical test locations to register results
2. Add effect size calculations per `effect_size_specifications.md`
3. Ensure all results include required metadata

### Phase 3: UI Polish (Final)
1. Add CSS styling for headline results
2. Implement JavaScript sorting/filtering
3. Add navigation functionality between tabs
4. Test responsive design

## Testing Requirements

1. **Unit Tests:**
   - Test result categorization logic
   - Verify effect size thresholds
   - Test sorting algorithms

2. **Integration Tests:**
   - Verify all statistical tests register results
   - Check result aggregation across tabs
   - Test navigation between headline and source tabs

3. **UI Tests:**
   - Verify correct counts in sub-tab labels
   - Test sorting and filtering functionality
   - Check responsive design on mobile

## Success Criteria

1. All statistical tests throughout the dashboard register their results
2. Results correctly categorized as material vs trivial based on effect sizes
3. Headline Results tab appears as the first/default tab
4. Clear visual distinction between material and trivial findings
5. Users can navigate from headline results to source details
6. Sorting and filtering work correctly
7. Page load time remains under 2 seconds with full results

## Future Enhancements

1. **Export Functionality:** Add ability to export headline results to PDF/CSV
2. **Trend Analysis:** Track how findings change over time/experiments
3. **Recommendation Engine:** Suggest actions based on material findings
4. **Interactive Visualizations:** Add charts showing effect size distributions
5. **Custom Thresholds:** Allow users to adjust material/trivial thresholds