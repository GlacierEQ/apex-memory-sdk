from __future__ import annotations


class MemorySDKError(Exception):
    pass

class NodeNotFoundError(MemorySDKError):
    pass

class VectorDimensionMismatchError(MemorySDKError):
    pass

class GraphCycleError(MemorySDKError):
    pass

class ConsolidationError(MemorySDKError):
    pass
