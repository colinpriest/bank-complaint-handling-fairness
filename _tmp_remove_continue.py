from pathlib import Path
text = Path("nshot_dpp_knn_experiment.py").read_text()
old = "                    if len(selected_examples) >= total_required:\n                        continue"
new = "                    if len(selected_examples) >= total_required:\n                        continue"
if old not in text:
    raise SystemExit('expected double continue not found')
text = text.replace(old, new, 1)
Path("nshot_dpp_knn_experiment.py").write_text(text)
