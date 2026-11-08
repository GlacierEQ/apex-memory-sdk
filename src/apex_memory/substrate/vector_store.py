from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Any

from apex_memory.substrate.bm25 import BM25Indexer
from apex_memory.substrate.embeddings import SubwordEmbeddingEngine


class MemoryItem:
    def __init__(
        self,
        item_id: str,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        created_at: str | None = None,
    ) -> None:
        self.item_id = item_id
        self.content = content
        self.embedding = embedding
        self.metadata = metadata or {}
        self.tags = tags or []
        self.created_at = created_at or dt.datetime.now(dt.timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "tags": self.tags,
            "created_at": self.created_at,
        }


class VectorStore:
    def __init__(self, db_path: Path | str = ":memory:", dimensions: int = 128) -> None:
        self.db_path = str(db_path)
        self.dimensions = dimensions
        self.embedder = SubwordEmbeddingEngine(dimensions=dimensions)
        self.bm25 = BM25Indexer()
        self._conn = sqlite3.connect(self.db_path)
        self._init_db()

    def _init_db(self) -> None:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                item_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL,
                metadata TEXT NOT NULL,
                tags TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()
        self._sync_bm25()

    def _sync_bm25(self) -> None:
        cursor = self._conn.cursor()
        cursor.execute("SELECT item_id, content FROM memories")
        for item_id, content in cursor.fetchall():
            self.bm25.add_document(item_id, content)

    def put(
        self,
        item_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> MemoryItem:
        embedding = self.embedder.embed(content)
        created_at = dt.datetime.now(dt.timezone.utc).isoformat()
        item = MemoryItem(
            item_id=item_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            tags=tags or [],
            created_at=created_at,
        )

        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO memories (item_id, content, embedding, metadata, tags, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                item.item_id,
                item.content,
                json.dumps(item.embedding),
                json.dumps(item.metadata),
                json.dumps(item.tags),
                item.created_at,
            ),
        )
        self._conn.commit()

        self.bm25.add_document(item.item_id, item.content)
        return item

    def get(self, item_id: str) -> MemoryItem | None:
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT item_id, content, embedding, metadata, tags, created_at FROM memories WHERE item_id = ?",
            (item_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return MemoryItem(
            item_id=row[0],
            content=row[1],
            embedding=json.loads(row[2]),
            metadata=json.loads(row[3]),
            tags=json.loads(row[4]),
            created_at=row[5],
        )

    def delete(self, item_id: str) -> bool:
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM memories WHERE item_id = ?", (item_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def list_all(self) -> list[MemoryItem]:
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT item_id, content, embedding, metadata, tags, created_at FROM memories"
        )
        items = []
        for row in cursor.fetchall():
            items.append(
                MemoryItem(
                    item_id=row[0],
                    content=row[1],
                    embedding=json.loads(row[2]),
                    metadata=json.loads(row[3]),
                    tags=json.loads(row[4]),
                    created_at=row[5],
                )
            )
        return items

    def search_semantic(self, query: str, top_k: int = 5) -> list[tuple[MemoryItem, float]]:
        q_vec = self.embedder.embed(query)
        items = self.list_all()
        scored = []
        for it in items:
            sim = self.embedder.cosine_similarity(q_vec, it.embedding)
            scored.append((it, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_hybrid(self, query: str, top_k: int = 5, rrf_k: int = 60) -> list[dict[str, Any]]:
        semantic_results = self.search_semantic(query, top_k=top_k * 2)
        bm25_ranked = self.bm25.score(query)

        rrf_scores: dict[str, float] = {}

        for rank, (it, _) in enumerate(semantic_results):
            rrf_scores[it.item_id] = rrf_scores.get(it.item_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, (doc_id, _) in enumerate(bm25_ranked):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        output = []
        for item_id, score in sorted_ids:
            it = self.get(item_id)
            if it:
                d = it.to_dict()
                d["rrf_score"] = round(score, 6)
                output.append(d)
        return output

    def close(self) -> None:
        self._conn.close()
