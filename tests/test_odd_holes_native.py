"""Reproducible checks for the compiled odd-hole search and its witness.

These tests need the optional C++ extension ``_supportrestriction_graph``
(install with ``SUPPORTRESTRICTION_BUILD_CPP=1``); the module is skipped when
the extension is not installed.  Run:
    pytest tests/test_odd_holes_native.py -v -rA

The native functions validate their inputs before searching (matrix shape,
neighbour index range, self-loops, duplicate neighbours, symmetry, and a binary
matrix that agrees with adj_list), so malformed inputs are tested here and must
raise ValueError.  The matrix entries are checked in the caller's own dtype,
before the conversion to C int: a fractional value or an out-of-range integer
is rejected, never truncated or wrapped into a valid entry (see "Matrix values
are checked before the dtype conversion" below).

Ported from ``tests/test_graph_utils_inspection.py`` of the SupportRestriction2026
package at commit ``c73eae9``.
"""
import collections
import importlib.util
import random
from itertools import combinations

import networkx as nx
import numpy as np
import numpy.ma as ma
import pytest

from graph_analysis_utils import GraphAnalyzer

if importlib.util.find_spec("_supportrestriction_graph") is None:
    pytest.skip(
        "C++ extension _supportrestriction_graph is not installed "
        "(install with SUPPORTRESTRICTION_BUILD_CPP=1)",
        allow_module_level=True,
    )

import _supportrestriction_graph as native  # noqa: E402

NATIVE_FUNCTIONS = [native.has_odd_holes_fast, native.find_odd_hole]
NATIVE_IDS = ["has_odd_holes_fast", "find_odd_hole"]


def native_inputs(G, nodes=None):
    """adj_list and binary matrix in one shared vertex order (weights ignored).

    Deliberately independent of odd_hole.native_inputs so that the native
    functions are tested without the adapter under test.
    """
    nodes = list(G.nodes()) if nodes is None else list(nodes)
    index = {v: i for i, v in enumerate(nodes)}
    adj_list = [[index[u] for u in G.neighbors(v)] for v in nodes]
    A = nx.to_numpy_array(G, nodelist=nodes, dtype=np.intc, weight=None)
    return adj_list, A


def has_odd_hole_native(G, nodes=None):
    return native.has_odd_holes_fast(*native_inputs(G, nodes))


def find_odd_hole_native(G, nodes=None):
    return native.find_odd_hole(*native_inputs(G, nodes))


def is_perfect_native(G):
    """Perfectness via the native search, with label-safe, weight-free inputs."""
    return not has_odd_hole_native(G) and not has_odd_hole_native(nx.complement(G))


def is_perfect_wrapper(G):
    return GraphAnalyzer(G, lambda node: 0).is_perfect(backend="cpp")


def is_induced_odd_cycle(G, cycle):
    """Pure-Python check: `cycle` (node labels, closing edge implied) is an odd hole of G."""
    k = len(cycle)
    if k < 5 or k % 2 == 0 or len(set(cycle)) != k:
        return False
    cycle_edges = {frozenset((cycle[i], cycle[(i + 1) % k])) for i in range(k)}
    return all(
        G.has_edge(a, b) == (frozenset((a, b)) in cycle_edges)
        for a, b in combinations(cycle, 2)
    )


def witness_labels(G, witness, nodes=None):
    """Map native indices back to labels in the order native_inputs used."""
    nodes = list(G.nodes()) if nodes is None else list(nodes)
    assert all(isinstance(i, int) and 0 <= i < len(nodes) for i in witness), witness
    return [nodes[i] for i in witness]


def has_odd_hole_bruteforce(G):
    """Exhaustive reference: some odd set of >= 5 vertices induces a cycle."""
    for k in range(5, G.number_of_nodes() + 1, 2):
        for nodes in combinations(G.nodes, k):
            H = G.subgraph(nodes)
            if (nx.is_connected(H) and H.number_of_edges() == k
                    and all(deg == 2 for _, deg in H.degree())):
                return True
    return False


# name, graph, G has odd hole, complement(G) has odd hole, G is perfect
CASES = [
    ("single_vertex", nx.empty_graph(1), False, False, True),
    ("triangle", nx.complete_graph(3), False, False, True),
    ("path_P4", nx.path_graph(4), False, False, True),
    ("C4", nx.cycle_graph(4), False, False, True),
    ("C5", nx.cycle_graph(5), True, True, False),
    ("C6", nx.cycle_graph(6), False, False, True),
    ("C7", nx.cycle_graph(7), True, False, False),
    ("complement_C7", nx.complement(nx.cycle_graph(7)), False, True, False),
    ("bipartite_K33", nx.complete_bipartite_graph(3, 3), False, False, True),
]
IDS = [c[0] for c in CASES]
PARAMS = pytest.mark.parametrize("name,G,odd_hole,co_odd_hole,perfect", CASES, ids=IDS)

# The only odd hole of an odd cycle graph is the whole cycle.
WITNESS_LENGTHS = {
    ("C5", "G"): 5, ("C5", "complement"): 5,
    ("C7", "G"): 7, ("complement_C7", "complement"): 7,
}


@PARAMS
def test_direct_odd_hole_result(name, G, odd_hole, co_odd_hole, perfect):
    assert has_odd_hole_native(G) is odd_hole


@PARAMS
def test_complement_odd_hole_result(name, G, odd_hole, co_odd_hole, perfect):
    # complement of C7: no odd hole itself, but its complement (C7) has one
    assert has_odd_hole_native(nx.complement(G)) is co_odd_hole


@PARAMS
def test_perfectness_decision(name, G, odd_hole, co_odd_hole, perfect):
    assert perfect is (not odd_hole and not co_odd_hole)
    assert is_perfect_native(G) is perfect
    assert is_perfect_wrapper(G) is perfect


@PARAMS
def test_agrees_with_pure_python_reference(name, G, odd_hole, co_odd_hole, perfect):
    assert GraphAnalyzer(G, lambda node: 0).is_perfect() is perfect


@PARAMS
def test_witness_consistent_with_boolean(name, G, odd_hole, co_odd_hole, perfect):
    for H, expected in ((G, odd_hole), (nx.complement(G), co_odd_hole)):
        witness = find_odd_hole_native(H)
        assert isinstance(witness, list)
        assert bool(witness) is expected
        assert bool(witness) is has_odd_hole_native(H)


@PARAMS
def test_witness_is_verified_odd_hole(name, G, odd_hole, co_odd_hole, perfect):
    for side, H, expected in (("G", G, odd_hole),
                              ("complement", nx.complement(G), co_odd_hole)):
        witness = find_odd_hole_native(H)
        if not expected:
            assert witness == []
            continue
        assert len(witness) == WITNESS_LENGTHS[(name, side)]
        assert len(witness) % 2 == 1 and len(witness) >= 5
        assert is_induced_odd_cycle(H, witness_labels(H, witness))


def test_c5_witness_is_the_whole_cycle_in_cyclic_order():
    # The memo's example from wedge [0, 1, 2] is [4, 0, 1, 2, 3]; the full
    # search may return another rotation or orientation, which is allowed.
    witness = find_odd_hole_native(nx.cycle_graph(5))
    assert sorted(witness) == [0, 1, 2, 3, 4]
    assert all((witness[i] - witness[(i + 1) % 5]) % 5 in (1, 4) for i in range(5))


def test_empty_graph_has_no_witness():
    A = np.zeros((0, 0), dtype=np.intc)
    assert native.find_odd_hole([], A) == []
    assert native.has_odd_holes_fast([], A) is False


def test_random_graphs_agree_with_pure_python_reference():
    rng = random.Random(20261028)
    checked = 0
    for n in range(5, 9):
        for p in (0.3, 0.5, 0.7):
            for _ in range(10):
                G = nx.gnp_random_graph(n, p, seed=rng.randrange(10**9))
                analyzer = GraphAnalyzer(G, lambda node: 0)
                assert analyzer.is_perfect(backend="cpp") is analyzer.is_perfect()
                assert is_perfect_native(G) is analyzer.is_perfect()
                checked += 1
    assert checked == 120


def test_random_graphs_witness_matches_bruteforce_odd_hole():
    rng = random.Random(20261107)
    checked = 0
    for n in range(5, 10):
        for p in (0.3, 0.5, 0.7):
            for _ in range(8):
                G = nx.gnp_random_graph(n, p, seed=rng.randrange(10**9))
                for H in (G, nx.complement(G)):
                    witness = find_odd_hole_native(H)
                    assert bool(witness) is has_odd_hole_native(H)
                    assert bool(witness) is has_odd_hole_bruteforce(H)
                    if witness:
                        assert is_induced_odd_cycle(H, witness_labels(H, witness))
                checked += 1
    assert checked == 120


def test_non_integer_labels_direct_call_with_explicit_nodelist():
    # C5 on string labels: the caller owns the order and can map results back.
    labels = ["a", "b", "c", "d", "e"]
    G = nx.relabel_nodes(nx.cycle_graph(5), dict(enumerate(labels)))
    assert has_odd_hole_native(G, nodes=labels) is True
    assert has_odd_hole_native(G, nodes=labels[::-1]) is True  # any shared order works
    assert is_perfect_wrapper(G) is False      # wrapper relabels internally
    for order in (labels, labels[::-1]):
        witness = find_odd_hole_native(G, nodes=order)
        mapped = witness_labels(G, witness, nodes=order)
        assert sorted(mapped) == labels
        assert is_induced_odd_cycle(G, mapped)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _invalid_input_cases():
    adj, A = native_inputs(nx.cycle_graph(5))

    def adj_with_row0(row):
        bad = [list(r) for r in adj]
        bad[0] = row
        return bad, A

    def matrix(edit, dtype=None):
        M = A.copy() if dtype is None else A.astype(dtype)
        edit(M)
        return adj, M

    def set_sym(i, j, value):
        def edit(M):
            M[i, j] = value
            M[j, i] = value
        return edit

    def set_one(i, j, value):
        def edit(M):
            M[i, j] = value
        return edit

    return {
        "matrix_1d": (adj, A.ravel()),
        "matrix_3d": (adj, A.reshape(5, 5, 1)),
        "matrix_wrong_shape": (adj, A[:4, :4]),
        "matrix_non_square": (adj, A[:4, :]),
        "index_too_large": adj_with_row0([1, 5]),
        "negative_index": adj_with_row0([1, -1]),
        "self_loop": adj_with_row0([1, 4, 0]),
        "duplicate_neighbour": adj_with_row0([1, 4, 1]),
        "asymmetric_adj_list": adj_with_row0([1, 4, 2]),
        "matrix_entry_2": matrix(set_sym(0, 1, 2)),
        "matrix_entry_negative": matrix(set_sym(0, 1, -1)),
        "matrix_extra_edge": matrix(set_sym(0, 2, 1)),
        "matrix_missing_edge": matrix(set_sym(0, 1, 0)),
        "matrix_nonzero_diagonal": matrix(set_one(0, 0, 1)),
        "matrix_asymmetric": matrix(set_one(0, 2, 1)),
        "int64_entry_3": matrix(set_sym(0, 1, 3), dtype=np.int64),
        "float_half_on_edge": matrix(set_sym(0, 1, 0.5), dtype=np.float64),
    }


INVALID_INPUTS = _invalid_input_cases()


@pytest.mark.parametrize("fn", NATIVE_FUNCTIONS, ids=NATIVE_IDS)
@pytest.mark.parametrize("case", list(INVALID_INPUTS), ids=list(INVALID_INPUTS))
def test_invalid_inputs_raise_value_error(fn, case):
    adj, A = INVALID_INPUTS[case]
    with pytest.raises(ValueError):
        fn(adj, A)


@pytest.mark.parametrize("fn", NATIVE_FUNCTIONS, ids=NATIVE_IDS)
def test_wrong_argument_types_raise_type_error(fn):
    adj, A = native_inputs(nx.cycle_graph(5))
    with pytest.raises(TypeError):
        fn([[1.0, 4]] + adj[1:], A)          # float inside adj_list
    with pytest.raises(TypeError):
        fn(adj, "not a matrix")


@pytest.mark.parametrize("dtype", [np.bool_, np.int8, np.uint8, np.int64, np.float64])
def test_accepted_matrix_dtypes_give_the_same_result(dtype):
    for G, expected in ((nx.cycle_graph(5), True), (nx.cycle_graph(6), False)):
        adj, A = native_inputs(G)
        M = A.astype(dtype)
        assert native.has_odd_holes_fast(adj, M) is expected
        assert bool(native.find_odd_hole(adj, M)) is expected
        assert bool(native.find_odd_hole(adj, np.asfortranarray(M))) is expected


# ---------------------------------------------------------------------------
# Matrix values are checked before the dtype conversion
# ---------------------------------------------------------------------------
# Regression tests for the matrix-input limitation of the source build (c73eae9).
# There the bindings received the matrix already cast to C int (forcecast), so the
# 0/1 check saw truncated or wrapped values.  On C5 (0-1 is an edge, 0-2 is not),
# every case below marked "accepted at c73eae9" was taken for the adjacency matrix
# of C5 and returned its odd hole.  The only fractional case in the source tests,
# float_half_on_edge above, did raise, but only because the truncated matrix
# disagreed with adj_list.

def _c5_matrix(dtype, i=None, j=None, value=None):
    """Adjacency matrix of C5 in ``dtype``, optionally with the pair (i, j) set to ``value``."""
    _, A = native_inputs(nx.cycle_graph(5))
    M = A.astype(dtype)
    if i is not None:
        M[i, j] = value
        M[j, i] = value
    return M


def _c5_with_masked_pair(dtype, i, j, hidden_value):
    """C5 matrix as a masked array whose pair (i, j) is masked over ``hidden_value``."""
    M = ma.masked_array(_c5_matrix(dtype, i, j, hidden_value))
    M[i, j] = ma.masked  # the data under the mask is unchanged
    M[j, i] = ma.masked
    assert M.data[i, j] == hidden_value
    return M


def _c5_unaligned():
    """C5 matrix of C ints at an odd byte offset: C-contiguous but not aligned."""
    A = _c5_matrix(np.intc)
    buffer = bytearray(A.nbytes + 1)
    M = np.ndarray(A.shape, dtype=np.intc, buffer=buffer, offset=1)
    M[...] = A
    return M


class _ArrayConversionFails:
    def __array__(self, *args, **kwargs):
        raise RuntimeError("no array for you")


class _ArrayProtocol:
    """Array-like, but not an array: ``__array__`` returns the given array."""

    def __init__(self, array):
        self._array = array

    def __array__(self, *args, **kwargs):
        return self._array


_LONG_DOUBLE_ABOVE_ONE = np.longdouble(1) + np.finfo(np.longdouble).eps

# name -> matrix whose entry at (0, 1) or (0, 2) is not exactly 0 or 1.
NON_BINARY_VALUES = {
    # fractions: accepted at c73eae9 (truncated to 0 on the non-edge, to 1 on the edge)
    "float64_half_on_non_edge": _c5_matrix(np.float64, 0, 2, 0.5),
    "float64_minus_half_on_non_edge": _c5_matrix(np.float64, 0, 2, -0.5),
    "float64_0.999_on_non_edge": _c5_matrix(np.float64, 0, 2, 0.999),
    "float64_1.9_on_edge": _c5_matrix(np.float64, 0, 1, 1.9),
    "float64_1.5_on_edge": _c5_matrix(np.float64, 0, 1, 1.5),
    "float32_1.25_on_edge": _c5_matrix(np.float32, 0, 1, 1.25),
    "float16_half_on_non_edge": _c5_matrix(np.float16, 0, 2, 0.5),
    "longdouble_just_above_1_on_edge": _c5_matrix(np.longdouble, 0, 1, _LONG_DOUBLE_ABOVE_ONE),
    # integers outside the C int range: accepted at c73eae9 (wrapped to 1 or to 0)
    "int64_2**32+1_on_edge": _c5_matrix(np.int64, 0, 1, 2**32 + 1),
    "int64_2**32_on_non_edge": _c5_matrix(np.int64, 0, 2, 2**32),
    "uint64_2**32+1_on_edge": _c5_matrix(np.uint64, 0, 1, 2**32 + 1),
    # non-finite: rejected at c73eae9 only because the cast happened to give INT_MIN
    "float64_nan_on_non_edge": _c5_matrix(np.float64, 0, 2, np.nan),
    "float64_inf_on_edge": _c5_matrix(np.float64, 0, 1, np.inf),
    # the fractional case of the source tests, now rejected for the entry itself
    "float64_half_on_edge": _c5_matrix(np.float64, 0, 1, 0.5),
}

# name -> numpy array whose dtype is not boolean, integer or real floating.  The
# first five were accepted at c73eae9 (a complex entry lost its imaginary part;
# object and string entries went through int()).
NON_REAL_DTYPES = {
    "complex_1+2j_on_edge": _c5_matrix(np.complex128, 0, 1, 1 + 2j),
    "complex_with_exact_values": _c5_matrix(np.complex128),
    "object_half_on_non_edge": _c5_matrix(object, 0, 2, 0.5),
    "object_with_exact_values": _c5_matrix(object),
    "string_digits": _c5_matrix(str),
    "bytes": _c5_matrix("S1"),
    "datetime": _c5_matrix("datetime64[s]"),
    "structured": np.zeros((5, 5), dtype=[("a", "i4")]),
}


def _not_arrays():
    """Inputs that are not numpy arrays.

    The bindings do not convert them.  numpy's conversion of nested containers can
    change what the caller wrote without notice: at c73eae9 the nested list with 0.5
    and the list of masked rows were both accepted as the matrix of C5, and a
    conversion in the elements' own dtype would read the value hidden under the
    masked bool element.
    """
    valid = _c5_matrix(np.intc)
    masked_rows = _c5_with_masked_pair(np.intc, 0, 1, 1)
    with_masked_bool = valid.astype(bool).tolist()
    with_masked_bool[0][1] = with_masked_bool[1][0] = ma.array(True, mask=True)
    with_masked_constant = valid.tolist()
    with_masked_constant[0][2] = with_masked_constant[2][0] = ma.masked
    return {
        "nested_list_of_valid_entries": valid.tolist(),
        "nested_list_half_on_non_edge": _c5_matrix(np.float64, 0, 2, 0.5).tolist(),
        "tuple_of_tuples": tuple(map(tuple, valid.tolist())),
        "ragged_nested_list": [[0, 1], [1, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0, 1], [1]],
        "list_of_masked_rows": list(masked_rows),
        "deque_of_masked_rows": collections.deque(masked_rows),
        "nested_list_with_masked_bool_element": with_masked_bool,
        "nested_list_with_masked_constant": with_masked_constant,
        "array_protocol_object": _ArrayProtocol(valid),
        "array_protocol_returning_masked": _ArrayProtocol(masked_rows),
        "array_conversion_raises": _ArrayConversionFails(),
        "python_int": 1,
        "numpy_scalar": np.int32(1),
        "none": None,
        "not_a_matrix": "not a matrix",
    }


NOT_ARRAYS = _not_arrays()

_BOTH_MODES = pytest.mark.parametrize("diagnostics", [False, True], ids=["normal", "diagnostics"])


@_BOTH_MODES
@pytest.mark.parametrize("fn", NATIVE_FUNCTIONS, ids=NATIVE_IDS)
@pytest.mark.parametrize("case", list(NON_BINARY_VALUES), ids=list(NON_BINARY_VALUES))
def test_non_binary_values_cannot_become_valid_adjacency(fn, case, diagnostics):
    adj, _ = native_inputs(nx.cycle_graph(5))
    # The message names the entry: the value is rejected as it was passed, not
    # because a converted matrix happens to disagree with adj_list.
    with pytest.raises(ValueError, match="entries must be 0 or 1"):
        fn(adj, NON_BINARY_VALUES[case], diagnostics=diagnostics)


def test_rejected_value_is_reported_as_passed():
    adj, _ = native_inputs(nx.cycle_graph(5))
    with pytest.raises(ValueError, match=r"A_matrix\[0, 2\] = 0\.5; entries must be 0 or 1"):
        native.find_odd_hole(adj, NON_BINARY_VALUES["float64_half_on_non_edge"])
    with pytest.raises(ValueError, match=r"A_matrix\[0, 1\] = 4294967297; entries must be 0 or 1"):
        native.find_odd_hole(adj, NON_BINARY_VALUES["int64_2**32+1_on_edge"])


# A masked entry is a missing value.  Viewing a masked array as a plain array
# drops the mask, so the data hidden under it would otherwise be read as an entry:
# the true 1 or 0 of C5 in the first two cases (accepted at c73eae9, like 0.5).
MASKED_MATRICES = {
    "masked_edge_over_1": _c5_with_masked_pair(np.intc, 0, 1, 1),
    "masked_non_edge_over_0": _c5_with_masked_pair(np.intc, 0, 2, 0),
    "masked_non_edge_over_half": _c5_with_masked_pair(np.float64, 0, 2, 0.5),
    "masked_bool_edge": _c5_with_masked_pair(np.bool_, 0, 1, True),
}


@_BOTH_MODES
@pytest.mark.parametrize("fn", NATIVE_FUNCTIONS, ids=NATIVE_IDS)
@pytest.mark.parametrize("case", list(MASKED_MATRICES), ids=list(MASKED_MATRICES))
def test_masked_entries_are_rejected(fn, case, diagnostics):
    adj, _ = native_inputs(nx.cycle_graph(5))
    with pytest.raises(ValueError, match="masked entries"):
        fn(adj, MASKED_MATRICES[case], diagnostics=diagnostics)


@_BOTH_MODES
@pytest.mark.parametrize("fn", NATIVE_FUNCTIONS, ids=NATIVE_IDS)
@pytest.mark.parametrize("case", list(NON_REAL_DTYPES), ids=list(NON_REAL_DTYPES))
def test_non_real_dtypes_raise_type_error(fn, case, diagnostics):
    adj, _ = native_inputs(nx.cycle_graph(5))
    with pytest.raises(TypeError, match="boolean, integer or real floating dtype"):
        fn(adj, NON_REAL_DTYPES[case], diagnostics=diagnostics)


@_BOTH_MODES
@pytest.mark.parametrize("fn", NATIVE_FUNCTIONS, ids=NATIVE_IDS)
@pytest.mark.parametrize("case", list(NOT_ARRAYS), ids=list(NOT_ARRAYS))
def test_matrix_must_be_a_numpy_array(fn, case, diagnostics):
    adj, _ = native_inputs(nx.cycle_graph(5))
    # Rejected as it is: no conversion is attempted (``__array__`` is not called).
    with pytest.raises(TypeError, match="must be a numpy.ndarray"):
        fn(adj, NOT_ARRAYS[case], diagnostics=diagnostics)


def test_converting_a_nested_list_is_the_callers_step():
    adj, A = native_inputs(nx.cycle_graph(5))
    as_list = NOT_ARRAYS["nested_list_of_valid_entries"]
    assert native.find_odd_hole(adj, np.array(as_list)) == native.find_odd_hole(adj, A)


@pytest.mark.parametrize("fn", NATIVE_FUNCTIONS, ids=NATIVE_IDS)
def test_zero_dimensional_array_raises_value_error(fn):
    adj, _ = native_inputs(nx.cycle_graph(5))
    with pytest.raises(ValueError, match="2D"):
        fn(adj, np.array(1))


def _valid_c5_representations():
    A = _c5_matrix(np.intc)
    read_only = A.copy()
    read_only.setflags(write=False)
    return {
        "float16_exact": A.astype(np.float16),
        "float32_exact": A.astype(np.float32),
        "longdouble_exact": A.astype(np.longdouble),
        "uint64": A.astype(np.uint64),
        "big_endian_int32": A.astype(">i4"),
        "big_endian_float64": A.astype(">f8"),
        "read_only": read_only,
        "non_contiguous_view": np.kron(A, np.ones((2, 2), dtype=np.intc))[::2, ::2],
        "unaligned_int_buffer": _c5_unaligned(),
        "masked_array_without_masked_entries": ma.masked_array(A),
    }


VALID_C5_REPRESENTATIONS = _valid_c5_representations()


@pytest.mark.parametrize(
    "case", list(VALID_C5_REPRESENTATIONS), ids=list(VALID_C5_REPRESENTATIONS)
)
def test_other_exact_representations_are_still_accepted(case):
    adj, A = native_inputs(nx.cycle_graph(5))
    M = VALID_C5_REPRESENTATIONS[case]
    expected = native.find_odd_hole(adj, A)
    assert len(expected) == 5
    assert native.find_odd_hole(adj, M) == expected
    assert native.has_odd_holes_fast(adj, M) is True
    witness, stats = native.find_odd_hole(adj, M, diagnostics=True)
    assert witness == expected and stats["witness_length"] == 5


def test_unaligned_case_is_really_unaligned():
    # The search reads the matrix without bounds or alignment checks, so the
    # binding must hand it an aligned copy of a buffer like this one.
    M = VALID_C5_REPRESENTATIONS["unaligned_int_buffer"]
    assert M.flags.c_contiguous and not M.flags.aligned


def test_module_exports_witness_function():
    path = native.__file__
    print(f"compiled module in use: {path}")   # visible with pytest -rA / -s
    assert path.endswith((".so", ".pyd"))
    assert callable(native.find_odd_hole)
    assert callable(native.has_odd_holes_fast)
    assert "ValueError" in native.find_odd_hole.__doc__


# ---------------------------------------------------------------------------
# Edge weights
# ---------------------------------------------------------------------------

def weighted_c7():
    # C7 (odd hole in G, none in its complement). nx.complement drops edge
    # attributes, so only the G-side call sees the weights.
    G = nx.cycle_graph(7)
    nx.set_edge_attributes(G, 2, "weight")
    return G


def test_direct_call_ignores_edge_weights_when_matrix_is_binary():
    assert has_odd_hole_native(weighted_c7()) is True
    assert is_perfect_native(weighted_c7()) is False


def test_wrapper_handles_edge_weights():
    # Previously a strict xfail: the wrapper built A with weight='weight', so
    # entries equal to 2 never satisfied A(u, w) == 1. It now passes weight=None.
    assert is_perfect_wrapper(weighted_c7()) is False


@pytest.mark.parametrize("fn", NATIVE_FUNCTIONS, ids=NATIVE_IDS)
def test_weighted_matrix_is_rejected_by_native_validation(fn):
    G = weighted_c7()
    nodes = list(G.nodes())
    adj = [[nodes.index(u) for u in G.neighbors(v)] for v in nodes]
    A_weighted = nx.to_numpy_array(G, nodelist=nodes, dtype=np.intc)  # weight='weight'
    assert A_weighted.max() == 2
    with pytest.raises(ValueError):
        fn(adj, A_weighted)
