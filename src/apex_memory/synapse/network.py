from __future__ import annotations

from typing import Any

from apex_memory.substrate.vector_store import VectorStore
from apex_memory.synapse.hebbian import HebbianNetwork


class AssociativeRecallEngine:
    def __init__(self, store: VectorStore, network: HebbianNetwork | None = None) -> None:
        self.store = store
        self.network = network or HebbianNetwork()

    def remember(self, item_id: str, content: str, related_ids: list[str] | None = None) -> None:
        self.store.put(item_id, content)
        if related_ids:
            for rel in related_ids:
                if self.store.get(rel):
                    self.network.fire(item_id, rel, delta=0.25)

    def recall(
        self, query: str, top_k: int = 3, expand_synaptic: bool = True
    ) -> list[dict[str, Any]]:
        direct_matches = self.store.search_hybrid(query, top_k=top_k)
        if not expand_synaptic or not direct_matches:
            return direct_matches

        matched_ids = [m["item_id"] for m in direct_matches]
        if len(matched_ids) > 1:
            self.network.fire_cluster(matched_ids, delta=0.1)

        synaptic_extensions: dict[str, float] = {}
        for m in direct_matches:
            assocs = self.network.get_associations(m["item_id"])
            for a in assocs:
                nid = a["node"]
                if nid not in matched_ids:
                    synaptic_extensions[nid] = max(synaptic_extensions.get(nid, 0.0), a["weight"])

        results = list(direct_matches)
        for nid, weight in sorted(synaptic_extensions.items(), key=lambda x: x[1], reverse=True):
            it = self.store.get(nid)
            if it:
                d = it.to_dict()
                d["synaptic_boost"] = round(weight, 4)
                d["source"] = "SYNAPTIC_ASSOCIATION"
                results.append(d)

        return results
