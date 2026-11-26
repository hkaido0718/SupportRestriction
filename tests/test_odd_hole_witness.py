"""Tests for label-preserving odd-hole witnesses and their verification.

The verifier, ``native_inputs`` and the adapter's input checks are pure Python and
run in every installation.  Tests marked ``requires_cpp`` call the compiled search
and are skipped when the optional extension ``_supportrestriction_graph`` is not
installed (install with ``SUPPORTRESTRICTION_BUILD_CPP=1``).

Ported from ``tests/test_odd_hole_witness.py`` of the SupportRestriction2026
package at commit ``c73eae9``.
"""

import importlib.util
import types

import networkx as nx
import numpy as np
import pytest

import odd_hole
from graph_analysis_utils import GraphAnalyzer

requires_cpp = pytest.mark.skipif(
    importlib.util.find_spec("_supportrestriction_graph") is None,
    reason="C++ extension _supportrestriction_graph is not installed "
           "(install with SUPPORTRESTRICTION_BUILD_CPP=1)",
)


def _fake_native(result):
    """Stand-in for the extension whose search returns ``result``."""
    return types.SimpleNamespace(find_odd_hole=lambda adj_list, A: result)


def _interference_graph():
    nodes = [(y, d) for d in range(4) for y in (0, 1)]

    def violates_nospillover(u, v):
        y1, d1 = u
        y2, d2 = v
        return (
            ((d1 == 1 and d2 == 0) or (d1 == 0 and d2 == 1))
            and y1 != y2
        )

    return GraphAnalyzer.build_graph_pairwise(
        nodes, violates_nospillover, lambda node: node[1]
    )


_ADAPTERS = [
    odd_hole.native_inputs,
    odd_hole.find_odd_hole_witness,
    odd_hole.find_odd_hole_or_antihole,
]

_PERFECT_GRAPHS = [
    pytest.param(nx.cycle_graph(4), id="C4"),
    pytest.param(nx.cycle_graph(6), id="C6"),
    pytest.param(nx.path_graph(4), id="P4"),
    pytest.param(nx.complete_graph(3), id="K3"),
    pytest.param(nx.complete_bipartite_graph(3, 3), id="K3-3"),
    pytest.param(_interference_graph(), id="interference"),
]


@requires_cpp
@pytest.mark.parametrize("n", [5, 7], ids=["C5", "C7"])
@pytest.mark.parametrize("reverse_nodes", [False, True], ids=["default", "reversed"])
def test_tuple_labelled_cycle_witness(n, reverse_nodes):
    labels = [((i,), "z") for i in range(n)]
    G = nx.relabel_nodes(nx.cycle_graph(n), dict(enumerate(labels)))

    if reverse_nodes:
        witness = odd_hole.find_odd_hole_witness(G, nodes=labels[::-1])
    else:
        witness = odd_hole.find_odd_hole_witness(G)

    assert isinstance(witness, list)
    assert len(witness) == n
    assert all(isinstance(vertex, tuple) for vertex in witness)
    assert set(witness) == set(labels)
    odd_hole.verify_odd_hole(G, witness)


@requires_cpp
@pytest.mark.parametrize(
    "G",
    _PERFECT_GRAPHS
    + [pytest.param(nx.complement(nx.cycle_graph(7)), id="complement-C7")],
)
def test_no_odd_hole_witness(G):
    assert odd_hole.find_odd_hole_witness(G) == []


@requires_cpp
@pytest.mark.parametrize(
    "G, expected_kind, expected_length",
    [
        pytest.param(nx.cycle_graph(5), "odd_hole", 5, id="C5"),
        pytest.param(
            nx.complement(nx.cycle_graph(7)), "odd_antihole", 7,
            id="complement-C7",
        ),
    ],
)
def test_find_odd_hole_or_antihole(G, expected_kind, expected_length):
    kind, witness = odd_hole.find_odd_hole_or_antihole(G)
    assert kind == expected_kind
    assert len(witness) == expected_length
    assert set(witness) == set(G)
    target = G if expected_kind == "odd_hole" else nx.complement(G)
    odd_hole.verify_odd_hole(target, witness)


@requires_cpp
@pytest.mark.parametrize("G", _PERFECT_GRAPHS)
def test_perfect_graph_has_neither_witness(G):
    assert odd_hole.find_odd_hole_or_antihole(G) == (None, [])


@pytest.mark.parametrize(
    "G, cycle",
    [
        pytest.param(nx.complete_graph(3), [0, 1, 2], id="triangle"),
        pytest.param(nx.cycle_graph(4), [0, 1, 2, 3], id="C4"),
        pytest.param(nx.cycle_graph(6), list(range(6)), id="even-C6"),
        pytest.param(
            nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]),
            [0, 1, 2, 3, 4],
            id="chord",
        ),
        pytest.param(
            nx.cycle_graph(6), [0, 1, 2, 3, 4], id="missing-closing-edge"
        ),
        pytest.param(nx.cycle_graph(5), [0, 1, 2, 1, 0], id="repeated-vertex"),
        pytest.param(nx.cycle_graph(5), [0, 1, 2, 3, 9], id="unknown-vertex"),
        pytest.param(nx.cycle_graph(5), [0, 2, 4, 1, 3], id="wrong-order"),
    ],
)
def test_verifier_rejects_invalid_cycles(G, cycle):
    with pytest.raises(ValueError):
        odd_hole.verify_odd_hole(G, cycle)
    assert odd_hole.is_odd_hole(G, cycle) is False


@pytest.mark.parametrize(
    "cycle",
    [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [2, 3, 4, 0, 1]],
    ids=["forward", "reversed", "rotated"],
)
def test_verifier_accepts_cycle_orientations(cycle):
    G = nx.cycle_graph(5)
    odd_hole.verify_odd_hole(G, cycle)
    assert odd_hole.is_odd_hole(G, cycle) is True


@pytest.mark.parametrize("adapter", _ADAPTERS)
@pytest.mark.parametrize(
    "G, exception",
    [
        pytest.param(nx.DiGraph([(0, 1)]), TypeError, id="directed"),
        pytest.param(nx.MultiGraph([(0, 1)]), TypeError, id="multigraph"),
        pytest.param(nx.Graph([(0, 0)]), ValueError, id="self-loop"),
    ],
)
def test_adapter_rejects_invalid_graphs(adapter, G, exception):
    with pytest.raises(exception):
        adapter(G)


@pytest.mark.parametrize("adapter", _ADAPTERS)
@pytest.mark.parametrize(
    "nodes",
    [
        pytest.param([0, 1, 2, 3], id="missing"),
        pytest.param([0, 1, 2, 3, 3], id="duplicated"),
        pytest.param([0, 1, 2, 3, 4, 5], id="extra"),
        pytest.param([0, 1, 2, 3, 5], id="foreign-node"),
    ],
)
def test_adapter_rejects_invalid_node_orders(adapter, nodes):
    with pytest.raises(ValueError):
        adapter(nx.cycle_graph(5), nodes=nodes)


def test_native_inputs_respect_custom_label_order():
    labels = {i: f"v{i}" for i in range(5)}
    G = nx.relabel_nodes(nx.cycle_graph(5), labels)
    order = ["v3", "v0", "v4", "v1", "v2"]
    nodes, adj_list, A = odd_hole.native_inputs(G, nodes=order)

    assert nodes == order
    assert len(adj_list) == 5
    assert A.dtype == np.dtype(np.intc)
    assert A.shape == (5, 5)
    assert np.array_equal(A, A.T)
    assert np.all(np.diag(A) == 0)
    for i, u in enumerate(nodes):
        expected_neighbors = {j for j, v in enumerate(nodes) if G.has_edge(u, v)}
        assert set(adj_list[i]) == expected_neighbors
        for j, v in enumerate(nodes):
            assert A[i, j] == int(G.has_edge(u, v))
            assert (A[i, j] == 1) == (j in adj_list[i])


def test_native_inputs_ignore_edge_weights():
    G = nx.cycle_graph(7)
    nx.set_edge_attributes(G, 2, "weight")
    nodes, adj_list, A = odd_hole.native_inputs(G)

    assert set(np.unique(A)) <= {0, 1}
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            assert A[i, j] == int(G.has_edge(u, v))
            assert (A[i, j] == 1) == (j in adj_list[i])


@requires_cpp
def test_find_witness_without_verification():
    G = nx.cycle_graph(5)
    witness = odd_hole.find_odd_hole_witness(G, verify=False)
    assert isinstance(witness, list)
    assert len(witness) == 5
    assert set(witness) == set(G)
    odd_hole.verify_odd_hole(G, witness)


@pytest.mark.parametrize(
    "fake_result",
    [
        pytest.param([0, 1, 2], id="fails-verification"),
        pytest.param([-1, 0, 1, 2, 3], id="negative-index"),
        pytest.param([0, 1, 2, 3, 5], id="index-out-of-range"),
    ],
)
def test_bad_native_result_is_a_failure_not_an_empty_list(monkeypatch, fake_result):
    # The adapter must never turn a broken native result into [] or a label list.
    monkeypatch.setattr(odd_hole, "_require_cpp", lambda: _fake_native(fake_result))
    with pytest.raises(RuntimeError):
        odd_hole.find_odd_hole_witness(nx.cycle_graph(5))


def test_verification_can_be_skipped_for_timing(monkeypatch):
    monkeypatch.setattr(odd_hole, "_require_cpp", lambda: _fake_native([0, 1, 2]))
    assert odd_hole.find_odd_hole_witness(nx.cycle_graph(5), verify=False) == [0, 1, 2]
