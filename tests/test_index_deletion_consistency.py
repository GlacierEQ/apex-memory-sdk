"""Deletion must remove memories from both SQLite and lexical search."""
from apex_memory.substrate.bm25 import BM25Indexer
from apex_memory.substrate.vector_store import VectorStore


def test_bm25_remove_document_is_idempotent_and_recalculates_statistics():
    index = BM25Indexer()
    index.add_document("old", "zebraquery zebraquery obsolete")
    index.add_document("keep", "zebraquery valid")

    assert index.corpus_size == 2
    assert index.doc_freqs["zebraquery"] == 2
    assert index.remove_document("old") is True
    assert index.remove_document("old") is False
    assert index.corpus_size == 1
    assert index.doc_freqs["zebraquery"] == 1
    assert "obsolete" not in index.doc_freqs
    assert index.avg_doc_len == 2.0
    assert [doc_id for doc_id, _ in index.score("zebraquery")] == ["keep"]
    assert index.remove_document("keep") is True
    assert index.corpus_size == 0
    assert index.doc_freqs == {}
    assert index.avg_doc_len == 0.0
    assert index.score("zebraquery") == []


def test_vector_store_delete_removes_lexical_rank_and_keeps_survivors():
    store = VectorStore(":memory:", dimensions=64)
    try:
        store.put("old", "zebraquery zebraquery obsolete")
        store.put("keep", "zebraquery valid")
        assert store.delete("old") is True
        assert store.get("old") is None
        assert store.delete("old") is False
        assert "old" not in store.bm25.term_freqs
        assert [doc_id for doc_id, _ in store.bm25.score("zebraquery")] == ["keep"]
        assert [hit["item_id"] for hit in store.search_hybrid("zebraquery", top_k=1)] == ["keep"]
    finally:
        store.close()


def test_deleted_document_stays_deleted_after_reopen(tmp_path):
    db_path = tmp_path / "memories.sqlite3"
    store = VectorStore(db_path, dimensions=64)
    store.put("old", "abolishedkeyword")
    store.put("keep", "unrelated surviving")
    assert store.delete("old") is True
    store.close()

    reopened = VectorStore(db_path, dimensions=64)
    try:
        assert reopened.get("old") is None
        assert reopened.bm25.score("abolishedkeyword") == []
        assert [item.item_id for item in reopened.list_all()] == ["keep"]
    finally:
        reopened.close()
