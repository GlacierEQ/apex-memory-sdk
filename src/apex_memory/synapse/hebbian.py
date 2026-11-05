from __future__ import annotations

import datetime as dt
from typing import Any

class SynapticEdge:
    def __init__(self, source: str, target: str, weight: float = 0.1, co_activations: int = 1) -> None:
        self.source = source
        self.target = target
        self.weight = min(max(weight, 0.0), 1.0)
        self.co_activations = co_activations
        self.last_activated = dt.datetime.now(dt.timezone.utc).isoformat()

    def reinforce(self, delta: float = 0.15) -> None:
        self.weight = min(self.weight + delta, 1.0)
        self.co_activations += 1
        self.last_activated = dt.datetime.now(dt.timezone.utc).isoformat()

    def decay(self, factor: float = 0.98) -> None:
        self.weight = max(self.weight * factor, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "weight": round(self.weight, 4),
            "co_activations": self.co_activations,
            "last_activated": self.last_activated,
        }

class HebbianNetwork:
    def __init__(self) -> None:
        self.edges: dict[tuple[str, str], SynapticEdge] = {}

    def _key(self, a: str, b: str) -> tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    def fire(self, node_a: str, node_b: str, delta: float = 0.15) -> SynapticEdge:
        if node_a == node_b:
            raise ValueError("Cannot connect node to itself in Hebbian pair.")
        k = self._key(node_a, node_b)
        if k in self.edges:
            edge = self.edges[k]
            edge.reinforce(delta)
        else:
            edge = SynapticEdge(source=k[0], target=k[1], weight=delta, co_activations=1)
            self.edges[k] = edge
        return edge

    def fire_cluster(self, nodes: list[str], delta: float = 0.1) -> None:
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                self.fire(nodes[i], nodes[j], delta=delta)

    def apply_decay(self, factor: float = 0.95) -> None:
        to_prune = []
        for k, edge in self.edges.items():
            edge.decay(factor)
            if edge.weight < 0.01:
                to_prune.append(k)
        for k in to_prune:
            del self.edges[k]

    def get_associations(self, node: str, min_weight: float = 0.05) -> list[dict[str, Any]]:
        results = []
        for (a, b), edge in self.edges.items():
            if edge.weight < min_weight:
                continue
            if a == node:
                results.append({"node": b, "weight": edge.weight, "co_activations": edge.co_activations})
            elif b == node:
                results.append({"node": a, "weight": edge.weight, "co_activations": edge.co_activations})
        results.sort(key=lambda x: x["weight"], reverse=True)
        return results

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_count": len(self.edges),
            "edges": [e.to_dict() for e in self.edges.values()],
        }
