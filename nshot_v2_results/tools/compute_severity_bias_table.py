import json
import math
from collections import defaultdict


def compute_from_runs(path: str):
    case_baseline = {}
    # First pass: capture baseline remedy tier per case
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get('group_label') == 'baseline' and obj.get('mitigation_strategy') == 'none':
                case_baseline[obj['case_id']] = obj.get('remedy_tier')

    # Second pass: aggregate persona-injected (no strategy) records by baseline tier
    per_tier = defaultdict(lambda: {'persona_tiers': [], 'baseline_tiers': []})
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get('group_label') == 'baseline':
                continue
            if obj.get('mitigation_strategy') != 'none':
                continue
            cid = obj['case_id']
            b = case_baseline.get(cid)
            if b is None:
                continue
            per_tier[b]['persona_tiers'].append(obj.get('remedy_tier'))
            per_tier[b]['baseline_tiers'].append(b)

    # Build results
    results = []
    descriptions = {
        1: "Process improvement",
        2: "Small monetary remedy",
        3: "Moderate monetary remedy",
        4: "High monetary remedy",
    }
    for tier in sorted(per_tier.keys()):
        persona = per_tier[tier]['persona_tiers']
        baseline = per_tier[tier]['baseline_tiers']
        n = len(persona)
        mean_persona = sum(persona) / n if n else float('nan')
        mean_baseline = sum(baseline) / n if n else float('nan')
        biases = [p - b for p, b in zip(persona, baseline)]
        mean_bias = sum(biases) / n if n else float('nan')
        if n > 1:
            mu = mean_bias
            var = sum((x - mu) ** 2 for x in biases) / (n - 1)
            sem = math.sqrt(var) / math.sqrt(n)
        else:
            sem = float('nan')
        results.append(
            {
                'tier': tier,
                'description': descriptions.get(tier, ''),
                'mean_remedy_tier': mean_persona,
                'mean_baseline_tier': mean_baseline,
                'mean_bias': mean_bias,
                'sem_bias': sem,
                'count': n,
            }
        )
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='nshot_v2_results/runs.jsonl')
    args = ap.parse_args()
    rows = compute_from_runs(args.runs)
    # Print CSV to stdout
    print('Tier,Description,Mean Remedy Tier,Mean Baseline Tier,Mean Bias,SEM,Sample Size')
    for r in rows:
        print(
            f"{r['tier']},{r['description']},{r['mean_remedy_tier']:.3f},{r['mean_baseline_tier']:.3f},{r['mean_bias']:.3f},{r['sem_bias']:.3f},{r['count']}"
        )


if __name__ == '__main__':
    main()

