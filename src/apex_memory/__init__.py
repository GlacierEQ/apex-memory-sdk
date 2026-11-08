from __future__ import annotations

from apex_memory.exceptions import (
    ConsolidationError,
    GraphCycleError,
    MemorySDKError,
    NodeNotFoundError,
    VectorDimensionMismatchError,
)
from apex_memory.federation.consolidator import MemoryConsolidator
from apex_memory.graph.knowledge_graph import Entity, KnowledgeGraph, Relation
from apex_memory.graph.traversal import GraphTraversal
from apex_memory.mcp.server import MemoryMCPServer
from apex_memory.substrate.bm25 import BM25Indexer
from apex_memory.substrate.clustering import VectorClusterer
from apex_memory.substrate.embeddings import SubwordEmbeddingEngine
from apex_memory.substrate.fusion import ReciprocalRankFusion
from apex_memory.substrate.vector_store import MemoryItem, VectorStore
from apex_memory.synapse.hebbian import HebbianNetwork, SynapticEdge
from apex_memory.synapse.network import AssociativeRecallEngine

__version__ = "1.1.0"

__all__ = [
    "AssociativeRecallEngine",
    "BM25Indexer",
    "ConsolidationError",
    "Entity",
    "GraphCycleError",
    "GraphTraversal",
    "HebbianNetwork",
    "KnowledgeGraph",
    "MemoryConsolidator",
    "MemoryItem",
    "MemoryMCPServer",
    "MemorySDKError",
    "NodeNotFoundError",
    "ReciprocalRankFusion",
    "Relation",
    "SubwordEmbeddingEngine",
    "SynapticEdge",
    "VectorClusterer",
    "VectorDimensionMismatchError",
    "VectorStore",
    "__version__",
]
