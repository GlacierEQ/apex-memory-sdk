"""
APEX Memory SDK — Basic Usage Examples

Run with: python3 examples/basic_usage.py
"""
from pathlib import Path
from apex_memory.substrate.vector_store import VectorStore
from apex_memory.substrate.embeddings import SubwordEmbeddingEngine
from apex_memory.synapse.hebbian import HebbianNetwork
from apex_memory.synapse.network import AssociativeRecallEngine
from apex_memory.graph.knowledge_graph import KnowledgeGraph
from apex_memory.federation.consolidator import MemoryConsolidator
import os, tempfile

db_path = tempfile.mktemp(suffix=".sqlite3")
os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True) if os.path.dirname(db_path) else None

store = VectorStore(db_path=db_path)
hebbian = HebbianNetwork()
engine = AssociativeRecallEngine(store, hebbian)
graph = KnowledgeGraph()
consolidator = MemoryConsolidator(store, hebbian, graph)

# 1. Remember items
items = [
    ("alpha", "The quick brown fox jumps over the lazy dog", ["animal", "phrase"]),
    ("beta", "Python is a high-level programming language", ["language", "programming"]),
    ("gamma", "Machine learning models learn from data", ["ml", "ai"]),
    ("delta", "The brown fox is fast and agile", ["animal", "phrase"]),
    ("epsilon", "Django is a Python web framework", ["language", "framework"]),
]
for item_id, content, tags in items:
    item = store.put(item_id, content, tags=tags)
    print(f"✅ Stored: {item.item_id} — {content[:40]}...")

# 2. Associate items (Hebbian synaptic reinforcement)
hebbian.fire("alpha", "delta", delta=0.3)  # both mention brown fox
hebbian.fire("beta", "epsilon", delta=0.3)  # both Python-related
hebbian.fire("gamma", "beta", delta=0.2)  # ML built on Python
print("\n🔗 Synaptic connections reinforced")

# 3. Recall (hybrid search)
print("\n🔍 Recall results for 'brown fox':")
results = engine.recall("brown fox", top_k=3)
for r in results:
    print(f"   [{r.get('score', 0):.3f}] {r.get('text', '')[:60]}")

print("\n🔍 Recall results for 'python programming':")
results = engine.recall("python programming", top_k=3)
for r in results:
    print(f"   [{r.get('score', 0):.3f}] {r.get('text', '')[:60]}")

# 4. Knowledge graph
graph.add_entity("Python", "Language")
graph.add_entity("Django", "Framework")
graph.add_entity("FastAPI", "Framework")
graph.add_relation("Django", "built_with", "Python")
graph.add_relation("FastAPI", "built_with", "Python")
print("\n📊 Knowledge graph relations:")
for r in graph.query(predicate="built_with"):
    print(f"   {r.source} —{r.relation}→ {r.target}")

# 5. Consolidate shard
shard = consolidator.consolidate("shard-1", "Basic usage example")
print(f"\n📦 Consolidated shard: {shard['shard_id']}")
print(f"   Hash: {shard['shard_hash']}")
print(f"   Memories: {shard['memory_count']}")
print(f"   Synapses: {shard['synapse_count']}")

print("\n✨ All examples completed successfully!")
