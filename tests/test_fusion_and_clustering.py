from apex_memory import (
    ReciprocalRankFusion,
    VectorClusterer,
)


def test_reciprocal_rank_fusion_ranking():
    # Dense results: doc1, doc2, doc3
    dense = [("doc1", 0.95), ("doc2", 0.85), ("doc3", 0.75)]
    # Sparse results: doc2, doc1, doc4
    sparse = [("doc2", 12.5), ("doc1", 8.2), ("doc4", 5.1)]

    # doc1: rank 1 (dense) + rank 2 (sparse) -> 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
    # doc2: rank 2 (dense) + rank 1 (sparse) -> 1/(60+2) + 1/(60+1) = 0.03252
    # doc3: rank 3 (dense) -> 1/(60+3) = 0.01587
    # doc4: rank 3 (sparse) -> 1/(60+3) = 0.01587

    fused = ReciprocalRankFusion.fuse(dense, sparse, k=60, top_n=3)
    assert len(fused) == 3
    top_ids = [doc_id for doc_id, _ in fused]
    assert "doc1" in top_ids[:2]
    assert "doc2" in top_ids[:2]


def test_vector_clustering_basic():
    # 4 items with distinct clusters (2 near [1, 1], 2 near [10, 10])
    items = [
        {"item_id": "a1", "embedding": [1.0, 1.1]},
        {"item_id": "a2", "embedding": [1.1, 1.0]},
        {"item_id": "b1", "embedding": [10.0, 10.1]},
        {"item_id": "b2", "embedding": [10.2, 9.9]},
    ]
    clusters = VectorClusterer.cluster_items(items, num_clusters=2, max_iterations=5)
    assert len(clusters) == 2
    # Check that a1 and a2 are grouped together
    c0_ids = {it["item_id"] for it in clusters[0]}
    c1_ids = {it["item_id"] for it in clusters[1]}
    assert c0_ids == {"a1", "a2"} or c1_ids == {"a1", "a2"}
