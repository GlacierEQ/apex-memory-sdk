# Changelog

## [1.0.0] — 2026-11-21

### Added
- ANE-accelerated embeddings via CoreML (Apple Neural Engine)
- Hybrid vector + BM25 search with RRF fusion
- Hebbian synaptic network (fire/decay/prune)
- Directed knowledge graph with transitive traversal (BFS)
- Cryptographic shard consolidation (SHA-256 digest)
- MCP JSON-RPC 2.0 stdio server
- CLI: remember, recall, associate, consolidate, mcp, serve, stop, status, query, seed, info
- Markdown seeding from files/directories
- HTTP server mode (/health, /query, /add endpoints)
- Module-level ANE model cache (0ms init on cache hit)
- Structured error handling + logging
- Apache-2.0 license
- SKILL.md agent skill manifest
- PyPI publish workflow on tag push

### Changed
- License: Proprietary → Apache-2.0
- CLI: Added lifecycle commands transcribed from zer0dex gem
- pyproject.toml: Added classifiers and project URLs

### Performance
- ANE embedding latency: 0.71ms per embedding
- Semantic similarity (related concepts): 0.997
- Test suite: 20 tests, <5s execution
- Cached init: 0ms vs 4s cold start
