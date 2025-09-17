import json
from fairness_analysis.statistical_analyzer import StatisticalAnalyzer
raw = []
with open('out/runs.jsonl','r',encoding='utf-8') as f:
    for line in f:
        line=line.strip()
        if line:
            raw.append(json.loads(line))
sa = StatisticalAnalyzer()
res = sa.analyze_severity_bias_variation(raw)
print('interpretation:', res.get('interpretation'))
print('finding:', res.get('finding'))
print('tiers_analyzed:', res.get('tiers_analyzed'))
print('tier_stats count:', len(res.get('tier_stats',{})))
