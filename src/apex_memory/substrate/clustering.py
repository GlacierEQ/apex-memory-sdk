from __future__ import annotations

import math
from typing import Any


class VectorClusterer:
    @staticmethod
    def euclidean_distance(v1: list[float], v2: list[float]) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

    @classmethod
    def compute_centroid(cls, vectors: list[list[float]]) -> list[float]:
        if not vectors:
            return []
        dims = len(vectors[0])
        centroid = [0.0] * dims
        for vec in vectors:
            for i, val in enumerate(vec):
                centroid[i] += val
        n = len(vectors)
        return [c / n for c in centroid]

    @classmethod
    def cluster_items(
        cls,
        items: list[dict[str, Any]],
        num_clusters: int = 3,
        max_iterations: int = 10,
    ) -> dict[int, list[dict[str, Any]]]:
        if not items:
            return {}
        if len(items) <= num_clusters:
            return {i: [item] for i, item in enumerate(items)}

        step = max(1, len(items) // num_clusters)
        centroids = [items[i * step]["embedding"] for i in range(num_clusters)]
        clusters: dict[int, list[dict[str, Any]]] = {i: [] for i in range(num_clusters)}

        for _ in range(max_iterations):
            new_clusters: dict[int, list[dict[str, Any]]] = {i: [] for i in range(num_clusters)}
            for item in items:
                emb = item["embedding"]
                dists = [cls.euclidean_distance(emb, c) for c in centroids]
                closest = dists.index(min(dists))
                new_clusters[closest].append(item)

            # Recompute centroids
            for i in range(num_clusters):
                if new_clusters[i]:
                    centroids[i] = cls.compute_centroid([it["embedding"] for it in new_clusters[i]])

            clusters = new_clusters

        return clusters
