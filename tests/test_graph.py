from __future__ import annotations

import pytest
from apex_memory.exceptions import NodeNotFoundError
from apex_memory.graph.knowledge_graph import KnowledgeGraph
from apex_memory.graph.traversal import GraphTraversal

def test_knowledge_graph_and_traversal():
    graph = KnowledgeGraph()
    graph.add_entity("CaseyBarton", "Person")
    graph.add_entity("AspenGrove", "Entity")
    graph.add_entity("Case1FDV", "LegalMatter")
    graph.add_entity("JudgeDowd", "Person")

    graph.add_relation("CaseyBarton", "OPERATES", "AspenGrove")
    graph.add_relation("AspenGrove", "PROSECUTES", "Case1FDV")
    graph.add_relation("Case1FDV", "PRESIDED_BY", "JudgeDowd")

    with pytest.raises(NodeNotFoundError):
        graph.add_relation("CaseyBarton", "KNOWS", "NonExistentPerson")

    path = GraphTraversal.shortest_path(graph, "CaseyBarton", "JudgeDowd")
    assert path == ["CaseyBarton", "AspenGrove", "Case1FDV", "JudgeDowd"]

    subgraph = GraphTraversal.k_hop_subgraph(graph, "CaseyBarton", max_hops=1)
    assert len(subgraph["entities"]) == 2
    assert len(subgraph["relations"]) == 1

    queried = graph.query(predicate="PROSECUTES")
    assert len(queried) == 1
    assert queried[0].source_id == "AspenGrove"
