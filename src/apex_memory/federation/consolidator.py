from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

from apex_memory.graph.knowledge_graph import KnowledgeGraph
from apex_memory.substrate.vector_store import VectorStore
from apex_memory.synapse.hebbian import HebbianNetwork


class MemoryConsolidator:
    def __init__(
        self,
        vector_store: VectorStore,
        hebbian_net: HebbianNetwork | None = None,
        graph: KnowledgeGraph | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.hebbian_net = hebbian_net or HebbianNetwork()
        self.graph = graph or KnowledgeGraph()

    def consolidate(self, shard_id: str, label: str = "Omniversal Consolidation") -> dict[str, Any]:
        timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
        memories = [m.to_dict() for m in self.vector_store.list_all()]
        synapses = self.hebbian_net.to_dict()
        kg_data = self.graph.to_dict()

        payload: dict[str, Any] = {
            "shard_id": shard_id,
            "label": label,
            "timestamp": timestamp,
            "memory_count": len(memories),
            "memories": memories,
            "synapses": synapses,
            "knowledge_graph": kg_data,
        }

        canonical = json.dumps(payload, sort_keys=True).encode("utf-8")
        shard_hash = hashlib.sha256(canonical).hexdigest()
        payload["shard_hash"] = shard_hash

        return payload

    def save_shard(self, output_path: Path | str, shard_id: str, label: str = "Episodic Shard") -> Path:
        data = self.consolidate(shard_id, label)
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return p

    def load_shard(self, shard_data: dict[str, Any]) -> int:
        memories = shard_data.get("memories", [])
        count = 0
        for m in memories:
            self.vector_store.put(
                item_id=m["item_id"],
                content=m["content"],
                metadata=m.get("metadata"),
                tags=m.get("tags"),
            )
            count += 1

        syn_edges = shard_data.get("synapses", {}).get("edges", [])
        for e in syn_edges:
            edge = self.hebbian_net.fire(e["source"], e["target"], delta=0.0)
            edge.weight = e["weight"]
            edge.co_activations = e["co_activations"]

        kg_entities = shard_data.get("knowledge_graph", {}).get("entities", [])
        for ent in kg_entities:
            self.graph.add_entity(ent["entity_id"], ent["label"], ent.get("properties"))

        kg_relations = shard_data.get("knowledge_graph", {}).get("relations", [])
        for rel in kg_relations:
            self.graph.add_relation(rel["source_id"], rel["relation"], rel["target_id"], rel.get("properties"))

        return count
