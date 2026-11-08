from __future__ import annotations


class ReciprocalRankFusion:
    DEFAULT_K = 60

    @classmethod
    def fuse(
        cls,
        dense_results: list[tuple[str, float]],
        sparse_results: list[tuple[str, float]],
        k: int = DEFAULT_K,
        top_n: int = 10,
    ) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}

        for rank, (doc_id, _) in enumerate(dense_results):
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + (rank + 1)))

        for rank, (doc_id, _) in enumerate(sparse_results):
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + (rank + 1)))

        sorted_fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_fused[:top_n]
