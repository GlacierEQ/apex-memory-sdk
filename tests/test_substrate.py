from __future__ import annotations

from apex_memory.substrate.bm25 import BM25Indexer
from apex_memory.substrate.embeddings import SubwordEmbeddingEngine
from apex_memory.substrate.vector_store import VectorStore


def test_subword_embedding_engine():
    engine = SubwordEmbeddingEngine(dimensions=64)
    v1 = engine.embed("artificial intelligence autonomous agents")
    v2 = engine.embed("autonomous artificial intelligence systems")
    v3 = engine.embed("completely unrelated baking recipe strawberry pie")

    assert len(v1) == 64
    assert len(v2) == 64

    sim_related = engine.cosine_similarity(v1, v2)
    sim_unrelated = engine.cosine_similarity(v1, v3)
    assert sim_related > sim_unrelated

def test_bm25_indexer():
    indexer = BM25Indexer()
    indexer.add_document("doc1", "federal court evidentiary docket transcript")
    indexer.add_document("doc2", "civil rico pattern of racketeering predicate acts")
    indexer.add_document("doc3", "unrelated garden flowers rose tulip")

    results = indexer.score("racketeering predicate")
    assert len(results) > 0
    top_doc, score = results[0]
    assert top_doc == "doc2"
    assert score > 0.0

def test_vector_store_hybrid_search():
    store = VectorStore(db_path=":memory:", dimensions=64)
    store.put("M1", "Federal court docket 1FDV-23-0001009 minute orders", tags=["legal", "court"])
    store.put("M2", "Civil RICO wire fraud 18 USC 1343 violations", tags=["rico", "fraud"])
    store.put("M3", "Section 1983 state actor civil rights deprivation", tags=["1983", "civil_rights"])

    item = store.get("M1")
    assert item is not None
    assert item.item_id == "M1"

    hybrid_hits = store.search_hybrid("wire fraud violations", top_k=2)
    assert len(hybrid_hits) >= 1
    assert hybrid_hits[0]["item_id"] == "M2"
