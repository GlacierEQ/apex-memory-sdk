from __future__ import annotations

from collections import deque
from typing import Any

from apex_memory.graph.knowledge_graph import KnowledgeGraph


class GraphTraversal:
    @staticmethod
    def shortest_path(graph: KnowledgeGraph, start_id: str, end_id: str) -> list[str] | None:
        if start_id not in graph.entities or end_id not in graph.entities:
            return None
        if start_id == end_id:
            return [start_id]

        queue: deque[list[str]] = deque([[start_id]])
        visited = {start_id}

        while queue:
            path = queue.popleft()
            current = path[-1]

            for rel in graph.get_outgoing(current):
                nxt = rel.target_id
                if nxt == end_id:
                    return path + [nxt]
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(path + [nxt])

        return None

    @staticmethod
    def k_hop_subgraph(graph: KnowledgeGraph, start_id: str, max_hops: int = 2) -> dict[str, Any]:
        if start_id not in graph.entities:
            return {"entities": [], "relations": []}

        visited_entities = {start_id}
        queue: deque[tuple[str, int]] = deque([(start_id, 0)])
        collected_relations = []

        while queue:
            curr, depth = queue.popleft()
            if depth >= max_hops:
                continue

            for rel in graph.get_outgoing(curr):
                collected_relations.append(rel)
                if rel.target_id not in visited_entities:
                    visited_entities.add(rel.target_id)
                    queue.append((rel.target_id, depth + 1))

        return {
            "root": start_id,
            "max_hops": max_hops,
            "entities": [graph.entities[eid].to_dict() for eid in visited_entities if eid in graph.entities],
            "relations": [r.to_dict() for r in collected_relations],
        }
