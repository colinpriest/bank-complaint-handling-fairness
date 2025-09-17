import numpy as np
from typing import List, Optional, Dict


class HybridDPP:
    """Hybrid DPP selector that mixes exact k-DPP sampling with greedy refinement."""

    def __init__(
        self,
        embeddings: np.ndarray,
        similarity: str = "cosine",
        kernel_regularization: float = 1e-6,
        random_state: Optional[int] = None,
    ) -> None:
        self.embeddings = embeddings if embeddings is not None else np.empty((0, 0))
        self.num_items = self.embeddings.shape[0]
        self.random_state = np.random.default_rng(random_state)
        self.selection_cache: Dict[int, List[int]] = {}

        if self.num_items == 0:
            self.similarity_matrix = np.empty((0, 0), dtype=float)
            self.kernel_matrix = np.empty((0, 0), dtype=float)
            self.eigvals = None
            self.eigvecs = None
            return

        self.similarity_matrix = self._compute_similarity_matrix(self.embeddings, similarity)
        self.kernel_matrix = self.similarity_matrix + kernel_regularization * np.eye(self.num_items)
        self.eigvals: Optional[np.ndarray] = None
        self.eigvecs: Optional[np.ndarray] = None

    def select(self, n: int) -> List[int]:
        if n <= 0 or self.num_items == 0:
            return []

        if n >= self.num_items:
            return list(range(self.num_items))

        if n in self.selection_cache:
            return list(self.selection_cache[n])

        if n == 1:
            selected = [self._select_central_point()]
        elif 2 <= n <= 6:
            selected = self._exact_k_dpp_sample(n)
        else:
            base_size = min(6, self.num_items)
            if base_size == 0:
                selected = []
            else:
                base_selection = self._exact_k_dpp_sample(base_size)
                selected = self._greedy_extend(base_selection, n)

        self.selection_cache[n] = list(selected)
        return selected

    def _compute_similarity_matrix(self, embeddings: np.ndarray, similarity: str) -> np.ndarray:
        if similarity != "cosine":
            raise ValueError(f"Unsupported similarity metric: {similarity}")

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = embeddings / norms
        sim = normalized @ normalized.T
        sim = np.clip(sim, -1.0, 1.0)
        return sim

    def _select_central_point(self) -> int:
        if self.num_items == 1:
            return 0

        sim = self.similarity_matrix
        sum_sim = sim.sum(axis=1) - np.diag(sim)
        quality = sum_sim / max(self.num_items - 1, 1)
        return int(np.argmax(quality))

    def _ensure_eigendecomposition(self) -> None:
        if self.eigvals is not None and self.eigvecs is not None:
            return

        eigvals, eigvecs = np.linalg.eigh(self.kernel_matrix)
        eigvals = np.clip(eigvals, 0.0, None)
        self.eigvals = eigvals
        self.eigvecs = eigvecs

    def _exact_k_dpp_sample(self, k: int) -> List[int]:
        k = min(k, self.num_items)
        if k <= 0:
            return []
        if k == 1:
            return [self._select_central_point()]

        self._ensure_eigendecomposition()
        assert self.eigvals is not None and self.eigvecs is not None

        selected_eig = self._sample_k_eigenvectors(k)
        if not selected_eig:
            return [self._select_central_point()]

        V = self.eigvecs[:, selected_eig]
        return self._sample_from_projection(V)

    def _sample_k_eigenvectors(self, k: int) -> List[int]:
        eigvals = self.eigvals
        assert eigvals is not None
        N = len(eigvals)
        E = np.zeros((N + 1, k + 1))
        E[:, 0] = 1.0

        for i in range(1, N + 1):
            lam = eigvals[i - 1]
            for l in range(1, min(i, k) + 1):
                E[i, l] = E[i - 1, l] + lam * E[i - 1, l - 1]

        selected = []
        l = k
        for i in range(N, 0, -1):
            if l == 0:
                break
            if E[i, l] == 0:
                continue
            lam = eigvals[i - 1]
            prob = 0.0
            if lam > 0 and E[i - 1, l - 1] > 0:
                prob = lam * E[i - 1, l - 1] / E[i, l]
            if self.random_state.uniform() <= prob:
                selected.append(i - 1)
                l -= 1

        if len(selected) < k:
            remaining = [i for i in range(N) if i not in selected and eigvals[i] > 0]
            while len(selected) < k and remaining:
                selected.append(remaining.pop())

        return selected

    def _sample_from_projection(self, V: np.ndarray) -> List[int]:
        selected_indices: List[int] = []
        V_current = V.copy()

        while V_current.shape[1] > 0:
            probs = np.sum(V_current ** 2, axis=1)
            total = probs.sum()
            if total <= 0:
                break
            probs /= total
            idx = int(self.random_state.choice(self.num_items, p=probs))
            selected_indices.append(idx)

            row = V_current[idx, :].copy()
            non_zero = np.nonzero(row)[0]
            if len(non_zero) == 0:
                break
            j = non_zero[0]
            row = row / row[j]
            V_current = V_current - np.outer(V_current[:, j], row)
            V_current = np.delete(V_current, j, axis=1)

        return list(dict.fromkeys(selected_indices))

    def _greedy_extend(self, base_selection: List[int], target_size: int) -> List[int]:
        selected = list(base_selection)
        if len(selected) >= target_size or len(selected) >= self.num_items:
            return selected[:target_size]

        L = self.kernel_matrix
        selected_set = set(selected)

        inv_selected = None
        if selected:
            submatrix = L[np.ix_(selected, selected)]
            inv_selected = np.linalg.pinv(submatrix)

        while len(selected) < min(target_size, self.num_items):
            best_idx = None
            best_gain = -np.inf

            candidates = [i for i in range(self.num_items) if i not in selected_set]
            if not candidates:
                break

            for idx in candidates:
                gain = L[idx, idx]
                if inv_selected is not None and selected:
                    v = L[np.ix_(selected, [idx])].reshape(-1)
                    gain -= v @ inv_selected @ v
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx

            if best_idx is None:
                break

            selected.append(best_idx)
            selected_set.add(best_idx)

            if inv_selected is None:
                diag = L[best_idx, best_idx]
                inv_selected = np.array([[1.0 / diag]]) if diag > 0 else np.array([[1.0]])
            else:
                v = L[np.ix_(selected[:-1], [best_idx])].reshape(-1)
                alpha = L[best_idx, best_idx] - v @ inv_selected @ v
                if alpha <= 0:
                    alpha = 1e-6
                inv_v = inv_selected @ v
                top_left = inv_selected + np.outer(inv_v, inv_v) / alpha
                top_right = -inv_v[:, None] / alpha
                bottom_left = (-inv_v / alpha)[None, :]
                bottom_right = np.array([[1.0 / alpha]])
                inv_selected = np.block([[top_left, top_right], [bottom_left, bottom_right]])

        return selected[:target_size]
