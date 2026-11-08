from __future__ import annotations

import datetime as dt
from typing import Any

from apex_memory.exceptions import NodeNotFoundError


class Entity:
    def __init__(
        self, entity_id: str, label: str, properties: dict[str, Any] | None = None
    ) -> None:
        self.entity_id = entity_id
        self.label = label
        self.properties = properties or {}
        self.created_at = dt.datetime.now(dt.timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "label": self.label,
            "properties": self.properties,
            "created_at": self.created_at,
        }


class Relation:
    def __init__(
        self,
        source_id: str,
        relation: str,
        target_id: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        self.source_id = source_id
        self.relation = relation
        self.target_id = target_id
        self.properties = properties or {}
        self.created_at = dt.datetime.now(dt.timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "relation": self.relation,
            "target_id": self.target_id,
            "properties": self.properties,
            "created_at": self.created_at,
        }


class KnowledgeGraph:
    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self.relations: list[Relation] = []

    def add_entity(
        self, entity_id: str, label: str, properties: dict[str, Any] | None = None
    ) -> Entity:
        ent = Entity(entity_id=entity_id, label=label, properties=properties)
        self.entities[entity_id] = ent
        return ent

    def add_relation(
        self,
        source_id: str,
        relation: str,
        target_id: str,
        properties: dict[str, Any] | None = None,
    ) -> Relation:
        if source_id not in self.entities:
            raise NodeNotFoundError(f"Source entity '{source_id}' does not exist in graph.")
        if target_id not in self.entities:
            raise NodeNotFoundError(f"Target entity '{target_id}' does not exist in graph.")
        rel = Relation(
            source_id=source_id, relation=relation, target_id=target_id, properties=properties
        )
        self.relations.append(rel)
        return rel

    def get_entity(self, entity_id: str) -> Entity | None:
        return self.entities.get(entity_id)

    def get_outgoing(self, source_id: str) -> list[Relation]:
        return [r for r in self.relations if r.source_id == source_id]

    def get_incoming(self, target_id: str) -> list[Relation]:
        return [r for r in self.relations if r.target_id == target_id]

    def query(
        self, subject: str | None = None, predicate: str | None = None, obj: str | None = None
    ) -> list[Relation]:
        results = self.relations
        if subject:
            results = [r for r in results if r.source_id == subject]
        if predicate:
            results = [r for r in results if r.relation.lower() == predicate.lower()]
        if obj:
            results = [r for r in results if r.target_id == obj]
        return results

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
            "entities": [e.to_dict() for e in self.entities.values()],
            "relations": [r.to_dict() for r in self.relations],
        }
