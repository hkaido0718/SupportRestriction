"""``GraphAnalyzer`` and the optional C++ backend.

``GraphAnalyzer.is_perfect(backend="cpp")`` and
``GraphAnalyzer.find_imperfectness_witness()`` delegate to ``odd_hole.py``, which loads
the compiled extension ``_supportrestriction_graph`` only at the moment a search is
requested.  The first part of this file runs in every installation: it checks that
nothing loads the extension on import, that the Python backend is the default and works
without the extension, and that a missing extension or a failed search is an exception,
never a perfectness result.  Tests marked ``requires_cpp`` run the compiled search and
are skipped when the extension is not installed (install with
``SUPPORTRESTRICTION_BUILD_CPP=1``).
"""

import importlib.util
import inspect
import os
import subprocess
import sys

import networkx as nx
import numpy as np
import pytest

import odd_hole
from graph_analysis_utils import GraphAnalyzer
from iv_model import IVModel
from mediation_model import MediationModel
from partial_monotonicity_model import PartialMonotonicityModel

EXTENSION = "_supportrestriction_graph"

requires_cpp = pytest.mark.skipif(
    importlib.util.find_spec(EXTENSION) is None,
    reason="C++ extension _supportrestriction_graph is not installed "
           "(install with SUPPORTRESTRICTION_BUILD_CPP=1)",
)


def _analyzer(G):
    return GraphAnalyzer(G, lambda node: 0)


def _iv(K, **kwargs):
    return IVModel((0, 1), (0, 1), tuple(range(K)), **kwargs)


@pytest.fixture
def no_extension(monkeypatch):
    """Make ``import _supportrestriction_graph`` fail, whether or not it is installed."""
    monkeypatch.setitem(sys.modules, EXTENSION, None)


class _FakeNative:
    """Stand-in for the extension: records its inputs and replays ``results``."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def find_odd_hole(self, adj_list, A):
        self.calls.append((adj_list, A))
        result = self.results[len(self.calls) - 1]
        if isinstance(result, BaseException):
            raise result
        return result


def _use_fake_native(monkeypatch, *results):
    fake = _FakeNative(*results)
    monkeypatch.setattr(odd_hole, "_require_cpp", lambda: fake)
    return fake


# ---------------------------------------------------------------------------
# Every installation: optional loading
# ---------------------------------------------------------------------------

def test_imports_and_python_backend_do_not_load_the_extension(tmp_path):
    # A fresh interpreter, started outside the repository, with the module search
    # path of this test run.
    code = "\n".join([
        "import sys",
        "import support_validation, iv_model, partial_monotonicity_model, mediation_model",
        "loaded = set(sys.modules)",
        "assert 'graph_analysis_utils' not in loaded, 'a model imported the analyzer'",
        "assert 'odd_hole' not in loaded, 'a model imported the adapter'",
        "import graph_analysis_utils, odd_hole",
        f"assert {EXTENSION!r} not in sys.modules, 'an import loaded the extension'",
        "model = iv_model.IVModel((0, 1), (0, 1), (0, 1))",
        "analyzer = graph_analysis_utils.GraphAnalyzer(model.build_graph(), model.group_fn)",
        "assert analyzer.is_perfect() is True",
        "assert analyzer.is_perfect(backend='python') is True",
        "odd_hole.native_inputs(analyzer.G)",
        "odd_hole.verify_odd_hole(__import__('networkx').cycle_graph(5), [0, 1, 2, 3, 4])",
        f"assert {EXTENSION!r} not in sys.modules, 'the Python backend loaded the extension'",
        "print('ok')",
    ])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env,
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_backend_defaults():
    assert inspect.signature(GraphAnalyzer.is_perfect).parameters["backend"].default == "python"
    witness_signature = inspect.signature(GraphAnalyzer.find_imperfectness_witness)
    assert witness_signature.parameters["backend"].default == "cpp"


@pytest.mark.parametrize(
    "G, perfect",
    [
        pytest.param(nx.cycle_graph(5), False, id="C5"),
        pytest.param(nx.cycle_graph(6), True, id="C6"),
        pytest.param(nx.complement(nx.cycle_graph(7)), False, id="complement-C7"),
        pytest.param(_iv(2).build_graph(), True, id="IV-K2"),
        pytest.param(_iv(4).build_graph(), False, id="IV-K4"),
    ],
)
def test_python_backend_works_without_the_extension(no_extension, G, perfect):
    analyzer = _analyzer(G)
    assert analyzer.is_perfect() is perfect
    assert analyzer.is_perfect(backend="python") is perfect


@pytest.mark.parametrize(
    "G",
    [
        pytest.param(nx.cycle_graph(5), id="not-perfect"),
        pytest.param(nx.cycle_graph(6), id="perfect"),
    ],
)
def test_cpp_request_without_the_extension_is_an_import_error(no_extension, G):
    analyzer = _analyzer(G)
    requests = [
        lambda: analyzer.is_perfect(backend="cpp"),
        lambda: analyzer.find_imperfectness_witness(),
        lambda: analyzer.find_imperfectness_witness(backend="cpp"),
        lambda: odd_hole.find_odd_hole_witness(G),
        lambda: odd_hole.find_odd_hole_or_antihole(G),
    ]
    for request in requests:
        with pytest.raises(ImportError, match="SUPPORTRESTRICTION_BUILD_CPP=1") as excinfo:
            request()
        assert isinstance(excinfo.value.__cause__, ImportError)


@pytest.mark.parametrize("backend", ["c++", "CPP", "pybind11", "", None])
def test_unknown_backend_is_a_value_error(no_extension, backend):
    analyzer = _analyzer(nx.cycle_graph(6))
    with pytest.raises(ValueError, match="backend must be 'python' or 'cpp'"):
        analyzer.is_perfect(backend=backend)
    with pytest.raises(ValueError, match="supports backend='cpp' only"):
        analyzer.find_imperfectness_witness(backend=backend)


def test_witness_retrieval_has_no_python_backend(no_extension):
    with pytest.raises(ValueError, match="supports backend='cpp' only"):
        _analyzer(nx.cycle_graph(5)).find_imperfectness_witness(backend="python")


@pytest.mark.parametrize(
    "G, exception",
    [
        pytest.param(nx.DiGraph([(0, 1)]), TypeError, id="directed"),
        pytest.param(nx.MultiGraph([(0, 1)]), TypeError, id="multigraph"),
        pytest.param(nx.Graph([(0, 0)]), ValueError, id="self-loop"),
    ],
)
def test_invalid_graph_is_reported_before_the_extension_is_needed(no_extension, G, exception):
    analyzer = _analyzer(G)
    with pytest.raises(exception):
        analyzer.find_imperfectness_witness()
    with pytest.raises(exception):
        analyzer.is_perfect(backend="cpp")


# ---------------------------------------------------------------------------
# Every installation: delegation to the adapter, with a stand-in for the extension
# ---------------------------------------------------------------------------

def test_is_perfect_cpp_runs_the_adapter_search_once(monkeypatch):
    G = nx.cycle_graph(6)
    calls = []

    def fake_search(graph, nodes=None, verify=True):
        calls.append((graph, nodes, verify))
        return None, []

    monkeypatch.setattr(odd_hole, "find_odd_hole_or_antihole", fake_search)
    assert _analyzer(G).is_perfect(backend="cpp") is True
    assert len(calls) == 1
    graph, nodes, verify = calls[0]
    assert graph is G and nodes == list(G.nodes()) and verify is True


@pytest.mark.parametrize("kind", ["odd_hole", "odd_antihole"])
def test_is_perfect_cpp_is_false_for_either_kind_of_witness(monkeypatch, kind):
    monkeypatch.setattr(
        odd_hole, "find_odd_hole_or_antihole",
        lambda graph, nodes=None, verify=True: (kind, [0, 1, 2, 3, 4]),
    )
    assert _analyzer(nx.cycle_graph(5)).is_perfect(backend="cpp") is False


def test_both_searches_use_one_node_order_and_binary_matrices(monkeypatch):
    # Labels in a non-sorted insertion order, with edge weights.
    order = ["c", "a", "e", "b", "d", "f"]
    G = nx.Graph()
    G.add_nodes_from(order)
    G.add_weighted_edges_from(
        [("a", "b", 2.5), ("b", "c", 7), ("c", "d", 0.5), ("d", "e", 3), ("e", "f", 2), ("f", "a", 9)]
    )
    fake = _use_fake_native(monkeypatch, [], [])

    assert _analyzer(G).find_imperfectness_witness() == (None, [])
    assert len(fake.calls) == 2  # the graph, then its complement

    (adj_graph, A_graph), (adj_complement, A_complement) = fake.calls
    n = len(order)
    for i, u in enumerate(order):
        assert sorted(adj_graph[i]) == sorted(order.index(v) for v in G.neighbors(u))
        for j, v in enumerate(order):
            assert A_graph[i, j] == int(G.has_edge(u, v))  # 0/1, weights ignored
            assert (A_graph[i, j] == 1) == (j in adj_graph[i])
            assert (A_complement[i, j] == 1) == (j in adj_complement[i])
    # Same order in both calls: the two matrices are complementary off the diagonal.
    assert np.array_equal(A_graph + A_complement + np.eye(n, dtype=A_graph.dtype),
                          np.ones((n, n), dtype=A_graph.dtype))


def test_node_order_given_as_an_iterator_serves_both_searches(monkeypatch):
    # The source adapter passed the iterator on to the complement search, which
    # then found it exhausted and raised ValueError.
    G = nx.cycle_graph(6)
    order = [3, 0, 4, 1, 5, 2]
    fake = _use_fake_native(monkeypatch, [], [])

    assert odd_hole.find_odd_hole_or_antihole(G, nodes=iter(order)) == (None, [])
    assert len(fake.calls) == 2
    (adj_graph, _), (adj_complement, _) = fake.calls
    for i, u in enumerate(order):
        assert sorted(adj_graph[i]) == sorted(order.index(v) for v in G.neighbors(u))
        assert sorted(adj_complement[i]) == sorted(
            order.index(v) for v in G if v != u and not G.has_edge(u, v)
        )


@requires_cpp
def test_node_order_given_as_an_iterator_with_the_compiled_search():
    C6, co_C7 = nx.cycle_graph(6), nx.complement(nx.cycle_graph(7))
    assert odd_hole.find_odd_hole_or_antihole(C6, nodes=iter(C6)) == (None, [])
    kind, witness = odd_hole.find_odd_hole_or_antihole(co_C7, nodes=reversed(list(co_C7)))
    assert kind == "odd_antihole" and sorted(witness) == list(range(7))


def test_complement_is_not_searched_after_an_odd_hole(monkeypatch):
    labels = ["v0", "v1", "v2", "v3", "v4"]
    G = nx.relabel_nodes(nx.cycle_graph(5), dict(enumerate(labels)))
    fake = _use_fake_native(monkeypatch, [2, 3, 4, 0, 1])

    kind, witness = _analyzer(G).find_imperfectness_witness()
    assert kind == "odd_hole"
    assert witness == ["v2", "v3", "v4", "v0", "v1"]  # indices mapped back to labels
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    "bad_result",
    [
        pytest.param([0, 1, 2], id="not-a-cycle"),
        pytest.param([0, 1, 2, 3], id="even-cycle"),
        pytest.param([0, 2, 4, 1, 3], id="not-induced-order"),
        pytest.param([0, 1, 2, 3, 7], id="index-out-of-range"),
        pytest.param([-1, 0, 1, 2, 3], id="negative-index"),
    ],
)
def test_unverifiable_native_result_is_an_error_not_a_verdict(monkeypatch, bad_result):
    analyzer = _analyzer(nx.cycle_graph(5))
    _use_fake_native(monkeypatch, bad_result)
    with pytest.raises(RuntimeError):
        analyzer.find_imperfectness_witness()
    _use_fake_native(monkeypatch, bad_result)
    with pytest.raises(RuntimeError):
        analyzer.is_perfect(backend="cpp")


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(RuntimeError("search failed"), id="runtime-error"),
        pytest.param(ValueError("invalid native input"), id="value-error"),
        pytest.param(MemoryError(), id="memory-error"),
        pytest.param(KeyboardInterrupt(), id="keyboard-interrupt"),
    ],
)
@pytest.mark.parametrize("fails_on", ["graph", "complement"])
def test_search_failure_or_interruption_propagates(monkeypatch, failure, fails_on):
    # C6 is perfect: a swallowed failure would look like "no witness".
    analyzer = _analyzer(nx.cycle_graph(6))
    results = [failure] if fails_on == "graph" else [[], failure]
    for request in (analyzer.find_imperfectness_witness,
                    lambda: analyzer.is_perfect(backend="cpp")):
        _use_fake_native(monkeypatch, *results)
        with pytest.raises(type(failure)):
            request()


# ---------------------------------------------------------------------------
# With the extension: the integrated search
# ---------------------------------------------------------------------------

# Graphs small enough for the exhaustive Python backend.
_SMALL_GRAPHS = [
    pytest.param(nx.empty_graph(0), id="empty"),
    pytest.param(nx.empty_graph(1), id="single-vertex"),
    pytest.param(nx.path_graph(4), id="P4"),
    pytest.param(nx.complete_graph(4), id="K4"),
    pytest.param(nx.cycle_graph(4), id="C4"),
    pytest.param(nx.cycle_graph(5), id="C5"),
    pytest.param(nx.cycle_graph(6), id="C6"),
    pytest.param(nx.cycle_graph(7), id="C7"),
    pytest.param(nx.cycle_graph(8), id="C8"),
    pytest.param(nx.cycle_graph(9), id="C9"),
    pytest.param(nx.complement(nx.cycle_graph(7)), id="complement-C7"),
    pytest.param(nx.complement(nx.cycle_graph(9)), id="complement-C9"),
    pytest.param(nx.complete_bipartite_graph(3, 3), id="K3-3"),
    pytest.param(nx.petersen_graph(), id="Petersen"),
    pytest.param(_iv(2).build_graph(), id="IV-K2"),
    pytest.param(_iv(3).build_graph(), id="IV-K3"),
    pytest.param(_iv(3, d_monotonicity=True).build_graph(), id="IV-K3-monotone"),
    pytest.param(
        _iv(3, exclusion=False, d_monotonicity=True).build_graph(),
        id="IV-K3-monotone-no-exclusion",
    ),
    pytest.param(_iv(4).build_graph(), id="IV-K4"),
    pytest.param(
        PartialMonotonicityModel((0, 1), (0, 1), ((0, 1), (0, 1)), exclusion=True).build_graph(),
        id="PM-J2-exclusion",
    ),
    pytest.param(MediationModel((0, 1), ((0, 1),), (0, 1)).build_graph(), id="Mediation-1"),
]


@requires_cpp
@pytest.mark.parametrize("G", _SMALL_GRAPHS)
def test_backends_agree(G):
    analyzer = _analyzer(G)
    perfect = analyzer.is_perfect()
    assert analyzer.is_perfect(backend="cpp") is perfect

    kind, witness = analyzer.find_imperfectness_witness()
    assert (kind is None) is perfect
    if kind is None:
        assert witness == []
    else:
        assert kind in ("odd_hole", "odd_antihole")
        target = G if kind == "odd_hole" else nx.complement(G)
        odd_hole.verify_odd_hole(target, witness)


# 16-node model graphs that the Python backend needs several seconds for (it
# returned True for each of them on 2026-11-25); checked here with C++ only.
@requires_cpp
@pytest.mark.parametrize(
    "model",
    [
        pytest.param(
            PartialMonotonicityModel((0, 1), (0, 1), ((0, 1), (0, 1)), exclusion=False),
            id="PM-J2-no-exclusion",
        ),
        pytest.param(MediationModel((0, 1), ((0, 1), (0, 1)), (0, 1)), id="Mediation-2"),
        pytest.param(
            MediationModel((0, 1), ((0, 1), (0, 1)), (0, 1), active_mediators=(0,)),
            id="Mediation-2-first-only",
        ),
    ],
)
def test_perfect_model_graphs_have_no_witness(model):
    analyzer = GraphAnalyzer(model.build_graph(), model.group_fn)
    assert analyzer.find_imperfectness_witness() == (None, [])
    assert analyzer.is_perfect(backend="cpp") is True


@requires_cpp
def test_model_graph_witness_is_a_verified_odd_hole_in_original_labels():
    # The example of the README: a binary IV model with a four-valued instrument,
    # exclusion and no monotonicity.
    model = IVModel((0, 1), (0, 1), (0, 1, 2, 3), exclusion=True, d_monotonicity=False)
    G = model.build_graph()
    analyzer = GraphAnalyzer(G, group_fn=model.group_fn)

    assert analyzer.is_perfect(backend="cpp") is False
    kind, witness = analyzer.find_imperfectness_witness(backend="cpp")

    assert kind == "odd_hole"
    assert len(witness) == 5
    assert all(node in G for node in witness)  # (y, d, z) tuples, not indices
    assert all(isinstance(node, tuple) and len(node) == 3 for node in witness)
    odd_hole.verify_odd_hole(G, witness)
    assert witness == [(0, 1, 0), (1, 0, 2), (1, 1, 3), (1, 1, 2), (0, 0, 3)]
    assert analyzer.is_perfect() is False  # the Python backend agrees


@requires_cpp
def test_antihole_witness_is_a_cycle_of_the_complement():
    labels = [f"v{i}" for i in range(7)]
    G = nx.relabel_nodes(nx.complement(nx.cycle_graph(7)), dict(enumerate(labels)))
    kind, witness = _analyzer(G).find_imperfectness_witness()

    assert kind == "odd_antihole"
    assert sorted(witness) == labels
    odd_hole.verify_odd_hole(nx.complement(G), witness)
    assert odd_hole.is_odd_hole(G, witness) is False
    k = len(witness)
    for i in range(k):
        assert not G.has_edge(witness[i], witness[(i + 1) % k])  # consecutive: non-adjacent in G
    assert _analyzer(G).is_perfect(backend="cpp") is False


@requires_cpp
def test_edge_weights_are_ignored():
    G = nx.cycle_graph(7)
    nx.set_edge_attributes(G, 2, "weight")
    kind, witness = _analyzer(G).find_imperfectness_witness()
    assert kind == "odd_hole" and sorted(witness) == list(range(7))
    assert _analyzer(G).is_perfect(backend="cpp") is False


@requires_cpp
def test_witness_does_not_depend_on_label_type_or_insertion_order():
    labels = [("y", i) for i in (3, 0, 4, 1, 2)]
    G = nx.Graph()
    G.add_nodes_from(labels)  # insertion order differs from the cycle order
    cycle = sorted(labels)
    G.add_edges_from(zip(cycle, cycle[1:] + cycle[:1]))

    kind, witness = _analyzer(G).find_imperfectness_witness()
    assert kind == "odd_hole" and set(witness) == set(labels)
    odd_hole.verify_odd_hole(G, witness)
    assert list(G.nodes()) == labels  # the analyzer did not change the graph


@requires_cpp
def test_extension_is_loaded_only_by_a_cpp_request(tmp_path):
    code = "\n".join([
        "import sys, networkx as nx",
        "from graph_analysis_utils import GraphAnalyzer",
        "analyzer = GraphAnalyzer(nx.cycle_graph(5), lambda node: 0)",
        "analyzer.is_perfect()",
        f"assert {EXTENSION!r} not in sys.modules",
        "assert analyzer.is_perfect(backend='cpp') is False",
        f"assert {EXTENSION!r} in sys.modules",
        "print('ok')",
    ])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env,
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
