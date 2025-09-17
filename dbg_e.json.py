import json
from fairness_analysis.statistical_analyzer import StatisticalAnalyzer
raw=[]
with open('out/runs.jsonl','r',encoding='utf-8') as f:
    for line in f:
        line=line.strip()
        if line:
            raw.append(json.loads(line))
sa=StatisticalAnalyzer()
re=sa.analyze_ethnicity_effects(raw)
print(re)
