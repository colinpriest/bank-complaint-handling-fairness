from pathlib import Path
text = Path("nshot_dpp_knn_experiment.py").read_text()
old = ["        if n_dpp > 0 and len(candidates) < n_dpp:","            return []","","        selected_examples = []","        used_indices = set()","","        # First, select DPP examples for diversity","        if n_dpp > 0:","            dpp_indices = self.select_dpp_examples(candidates, n_dpp)","            for idx in dpp_indices:","                if idx < len(candidates):","                    selected_examples.append(candidates[idx])","                    used_indices.add(idx)","","        # Then, select k-NN examples for relevance (avoiding duplicates)","        if k_nn > 0:","            # Filter out already selected candidates","            remaining_candidates = [candidates[i] for i in range(len(candidates)) if i not in used_indices]","","            if remaining_candidates:","                knn_indices = self.select_knn_examples(query_embedding, remaining_candidates, k_nn)","","                # Map back to original indices","                remaining_map = {new_idx: orig_idx for new_idx, orig_idx in enumerate(range(len(candidates))) if orig_idx not in used_indices}","","                for new_idx in knn_indices:","                    if new_idx in remaining_map:","                        orig_idx = remaining_map[new_idx]","                        selected_examples.append(candidates[orig_idx])","                        used_indices.add(orig_idx)","","        return selected_examples"]
replacement = """        max_needed = total_required if total_required is not None else (n_dpp + k_nn)
        if max_needed <= 0:
            return []

        selected_examples: List[Dict] = []
        used_case_ids = set()

        if n_dpp > 0:
            selector = HybridDPP(
                np.array([ex['embedding'] for ex in candidates if ex.get('embedding') is not None], dtype=np.float32),
                random_state=42
            )
            indices = selector.select(min(n_dpp, selector.num_items if selector.num_items else 0))
            for idx in indices:
                if idx >= len(candidates):
                    continue
                ex = candidates[idx]
                selected_examples.append(ex)
                used_case_ids.add(ex.get('case_id'))

        if k_nn > 0:
            remaining_candidates = [
                ex for ex in candidates
                if ex.get('case_id') not in used_case_ids and ex.get('embedding') is not None
            ]
            if remaining_candidates:
                candidate_embeddings = np.array([ex['embedding'] for ex in remaining_candidates], dtype=np.float32)
                knn_indices = self.select_knn_examples(query_embedding, remaining_candidates, k_nn)
                for idx in knn_indices:
                    if idx >= len(remaining_candidates):
                        continue
                    ex = remaining_candidates[idx]
                    selected_examples.append(ex)
                    used_case_ids.add(ex.get('case_id'))
                    if len(selected_examples) >= max_needed:
                        break

        return selected_examples[:max_needed]
"""
if '        if n_dpp > 0 and len(candidates) < n_dpp:' not in text:
    raise SystemExit('Old selection block not found in experiment combine function')
for line in old:
    text = text.replace(line + '\n', '', 1)
text = text.replace('        return selected_examples\n\n', replacement + '\n\n', 1)
Path("nshot_dpp_knn_experiment.py").write_text(text)
