from __future__ import annotations

from apex_memory.substrate.vector_store import VectorStore
from apex_memory.synapse.hebbian import HebbianNetwork
from apex_memory.synapse.network import AssociativeRecallEngine


def test_hebbian_network_reinforcement_and_decay():
    net = HebbianNetwork()
    edge = net.fire("nodeA", "nodeB", delta=0.2)
    assert edge.weight == 0.2
    assert edge.co_activations == 1

    edge2 = net.fire("nodeA", "nodeB", delta=0.3)
    assert round(edge2.weight, 2) == 0.5
    assert edge2.co_activations == 2

    assocs = net.get_associations("nodeA")
    assert len(assocs) == 1
    assert assocs[0]["node"] == "nodeB"

    net.apply_decay(factor=0.5)
    assocs_decayed = net.get_associations("nodeA")
    assert round(assocs_decayed[0]["weight"], 2) == 0.25


def test_associative_recall_engine():
    store = VectorStore(db_path=":memory:", dimensions=64, use_ane=False)
    net = HebbianNetwork()
    engine = AssociativeRecallEngine(store, net)
