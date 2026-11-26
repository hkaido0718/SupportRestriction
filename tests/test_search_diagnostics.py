"""Tests for the optional search diagnostics of the compiled odd-hole search.

``has_odd_holes_fast`` and ``find_odd_hole`` accept the keyword-only argument
``diagnostics`` (default ``False``).  With ``diagnostics=True`` they return
``(result, stats)``; the search and its order are the same in both modes.

The search explores one initial wedge at a time: a wedge is generated and
explored together with its extensions before the next wedge is generated.  The
build before that change, called the baseline here, stored every wedge before it
examined the first path.  Both examine the same paths in the same order, so they
return the same witness and agree on four counters; the other three counters
differ.  The baseline is not part of this repository; it survives here only as
the pure-Python copy ``_baseline_search``, which fixes the search order.

The expected counters come from two pure-Python copies of the loops,
``_reference_search`` (this build) and ``_baseline_search`` (the baseline), and
from the table ``EXACT_CASES`` recorded when the one-wedge-at-a-time search was
written.

These tests need the optional C++ extension ``_supportrestriction_graph``
(install with ``SUPPORTRESTRICTION_BUILD_CPP=1``); the module is skipped when
the extension is not installed.

Ported from ``tests/test_search_diagnostics.py`` of the SupportRestriction2026
package at commit ``c73eae9``.  There the model graphs came from
``benchmarks/benchmark_models.py``, a verbatim copy of the graph construction of
``IVModel`` and ``PartialMonotonicityModel`` at ``bed4ee7``; here they come from
the classes themselves, so the node and neighbour order, on which the exact
counters and the recorded witnesses depend, is the same.
"""

import importlib.util
import random
import time

import networkx as nx
import pytest

import odd_hole
from iv_model import IVModel
from partial_monotonicity_model import PartialMonotonicityModel

if importlib.util.find_spec("_supportrestriction_graph") is None:
    pytest.skip(
        "C++ extension _supportrestriction_graph is not installed "
        "(install with SUPPORTRESTRICTION_BUILD_CPP=1)",
        allow_module_level=True,
    )

import _supportrestriction_graph as native  # noqa: E402

COUNTERS = (
    "initial_wedges_generated",
    "wedges_before_first_search",
    "paths_explored",
    "extensions_created",
    "max_pending_paths",
    "max_path_vertices",
    "witness_length",
)
# Counters fixed by the sequence of examined paths: equal in both builds.
ORDER_COUNTERS = (
    "paths_explored",
    "extensions_created",
    "max_path_vertices",
    "witness_length",
)
TIME_KEY = "wedge_generation_seconds"


def _examine(stack, adj_list, adj_sets, stats):
    """One pass of the expansion loop: examine the path on top of the stack.

    Returns the witness when two candidate endpoints are adjacent; otherwise
    pushes the extensions and returns ``None``.
    """
    S, N_V = stack.pop()
    if stats["paths_explored"] == 0:
        stats["wedges_before_first_search"] = stats["initial_wedges_generated"]
    stats["paths_explored"] += 1
    stats["max_path_vertices"] = max(stats["max_path_vertices"], len(S))

    first, last = S[0], S[-1]
    N1 = [u for u in adj_list[first] if u not in N_V and u not in adj_sets[last]]
    N2 = [w for w in adj_list[last] if w not in N_V and w not in adj_sets[first]]
    if not N1 or not N2:
        return None

    N_V_tilde = N_V | adj_sets[first] | adj_sets[last]
    for u in N1:
        for w in N2:
            if w in adj_sets[u]:
                stats["witness_length"] = len(S) + 2
                return [u] + S + [w]
            stack.append(([u] + S + [w], N_V_tilde))
            stats["extensions_created"] += 1
            stats["max_pending_paths"] = max(stats["max_pending_paths"], len(stack))
    return None


def _reference_search(adj_list):
    """Pure-Python copy of the compiled search with the same counters.

    Returns ``(witness, counters)``.  Loops and stack order follow
    ``find_odd_hole_impl``: the wedge loops count downward and each wedge is
    explored, with its extensions, before the next one is generated.
    """
    n = len(adj_list)
    adj_sets = [set(neighbours) for neighbours in adj_list]
    stats = dict.fromkeys(COUNTERS, 0)
    stack = []

    for v in range(n - 1, -1, -1):
        Nv = adj_list[v]
        d = len(Nv)
        for i in range(d - 2, -1, -1):
            u = Nv[i]
            for j in range(d - 1, i, -1):
                w = Nv[j]
                if w in adj_sets[u]:
                    continue
                stack.append(([u, v, w], set(Nv)))
                stats["initial_wedges_generated"] += 1
                stats["max_pending_paths"] = max(stats["max_pending_paths"], len(stack))

                while stack:
                    witness = _examine(stack, adj_list, adj_sets, stats)
                    if witness is not None:
                        return witness, stats

    return [], stats


def _baseline_search(adj_list):
    """Pure-Python copy of the baseline search: every wedge is stored first.

    Returns ``(witness, counters, wedges_examined)``; the last value is the
    number of initial wedges the search took from the stack.
    """
    n = len(adj_list)
    adj_sets = [set(neighbours) for neighbours in adj_list]
    stats = dict.fromkeys(COUNTERS, 0)
    stack = []

    for v in range(n):
        Nv = adj_list[v]
        d = len(Nv)
        for i in range(d - 1):
            u = Nv[i]
            for j in range(i + 1, d):
                w = Nv[j]
                if w in adj_sets[u]:
                    continue
                stack.append(([u, v, w], set(Nv)))
                stats["initial_wedges_generated"] += 1
                stats["max_pending_paths"] = max(stats["max_pending_paths"], len(stack))

    wedges_examined = 0
    while stack:
        wedges_examined += len(stack[-1][0]) == 3
        witness = _examine(stack, adj_list, adj_sets, stats)
        if witness is not None:
            return witness, stats, wedges_examined

    return [], stats, wedges_examined


def _inputs(G):
    _, adj_list, A = odd_hole.native_inputs(G)
    return adj_list, A


def _counters(stats):
    return tuple(stats[name] for name in COUNTERS)


def _wedge_formula(G):
    return sum(d * (d - 1) // 2 for _, d in G.degree()) - sum(nx.triangles(G).values())


def _iv(K):
    return IVModel(
        (0, 1), (0, 1), tuple(range(K)), exclusion=True, d_monotonicity=False
    ).build_graph()


def _pm(J, exclusion):
    return PartialMonotonicityModel(
        (0, 1), (0, 1), ((0, 1),) * J, exclusion=exclusion
    ).build_graph()


# (name, graph, counters of this build, counters of the baseline), both in the
# order of COUNTERS.
EXACT_CASES = [
    ("C4", lambda: nx.cycle_graph(4), (4, 1, 4, 0, 1, 3, 0), (4, 4, 4, 0, 4, 3, 0)),
    ("C5", lambda: nx.cycle_graph(5), (1, 1, 1, 0, 1, 3, 5), (5, 5, 1, 0, 5, 3, 5)),
    ("C6", lambda: nx.cycle_graph(6), (6, 1, 12, 6, 1, 5, 0), (6, 6, 12, 6, 6, 5, 0)),
    ("C7", lambda: nx.cycle_graph(7), (1, 1, 2, 1, 1, 5, 7), (7, 7, 2, 1, 7, 5, 7)),
    ("C9", lambda: nx.cycle_graph(9), (1, 1, 3, 2, 1, 7, 9), (9, 9, 3, 2, 9, 7, 9)),
    ("K4", lambda: nx.complete_graph(4), (0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0)),
    ("isolated5", lambda: nx.empty_graph(5), (0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0)),
    ("empty", lambda: nx.empty_graph(0), (0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0)),
    ("P4", lambda: nx.path_graph(4), (2, 1, 2, 0, 1, 3, 0), (2, 2, 2, 0, 2, 3, 0)),
    (
        "K3-3",
        lambda: nx.complete_bipartite_graph(3, 3),
        (18, 1, 18, 0, 1, 3, 0),
        (18, 18, 18, 0, 18, 3, 0),
    ),
    ("Petersen", nx.petersen_graph, (1, 1, 1, 0, 1, 3, 5), (30, 30, 1, 0, 30, 3, 5)),
    (
        "co-C7",
        lambda: nx.complement(nx.cycle_graph(7)),
        (21, 1, 21, 0, 1, 3, 0),
        (21, 21, 21, 0, 21, 3, 0),
    ),
    ("IV-K2", lambda: _iv(2), (24, 1, 48, 24, 1, 5, 0), (24, 24, 48, 24, 24, 5, 0)),
    (
        "IV-K2-complement",
        lambda: nx.complement(_iv(2)),
        (24, 1, 24, 0, 1, 3, 0),
        (24, 24, 24, 0, 24, 3, 0),
    ),
    ("IV-K4", lambda: _iv(4), (1, 1, 1, 0, 1, 3, 5), (240, 240, 1, 0, 240, 3, 5)),
    ("IV-K8", lambda: _iv(8), (1, 1, 1, 0, 1, 3, 5), (2016, 2016, 1, 0, 2016, 3, 5)),
    ("IV-K16", lambda: _iv(16), (1, 1, 1, 0, 1, 3, 5), (16320, 16320, 1, 0, 16320, 3, 5)),
    ("PM-J2-on", lambda: _pm(2, True), (9, 1, 10, 3, 2, 5, 5), (156, 156, 10, 3, 156, 5, 5)),
    (
        "PM-J4-on",
        lambda: _pm(4, True),
        (75, 1, 76, 6, 5, 5, 5),
        (13884, 13884, 76, 6, 13884, 5, 5),
    ),
    (
        "PM-J3-off",
        lambda: _pm(3, False),
        (1828, 1, 1828, 0, 1, 3, 0),
        (1828, 1828, 1828, 0, 1828, 3, 0),
    ),
    (
        "PM-J3-off-complement",
        lambda: nx.complement(_pm(3, False)),
        (784, 1, 3856, 3072, 19, 7, 0),
        (784, 784, 3856, 3072, 789, 7, 0),
    ),
    (
        "PM-J4-off",
        lambda: _pm(4, False),
        (12396, 1, 12396, 0, 1, 3, 0),
        (12396, 12396, 12396, 0, 12396, 3, 0),
    ),
    (
        "PM-J4-off-complement",
        lambda: nx.complement(_pm(4, False)),
        (4352, 1, 224064, 219712, 99, 9, 0),
        (4352, 4352, 224064, 219712, 4413, 9, 0),
    ),
]
EXACT_IDS = [name for name, _, _, _ in EXACT_CASES]


def _random_graphs():
    rng = random.Random(20261114)
    graphs = []
    for index in range(120):
        n = rng.randint(1, 12)
        p = rng.choice([0.15, 0.3, 0.5, 0.7])
        graphs.append(nx.gnp_random_graph(n, p, seed=rng.randint(0, 10**6)))
    return graphs


RANDOM_GRAPHS = _random_graphs()


def _check_consistency(G, witness, stats):
    counters = dict(zip(COUNTERS, _counters(stats)))
    initial = counters["initial_wedges_generated"]
    explored = counters["paths_explored"]
    created = counters["extensions_created"]
    wedges = _wedge_formula(G)

    # One wedge at a time: the first path is examined when one wedge exists,
    # and the stack holds at most the current wedge and its extensions.
    assert counters["wedges_before_first_search"] == min(initial, 1)
    assert min(initial, 1) <= counters["max_pending_paths"] <= 1 + created
    if explored:
        assert counters["max_path_vertices"] >= 3
        assert counters["max_path_vertices"] % 2 == 1
    else:
        assert counters["max_path_vertices"] == 0
        assert initial == 0

    if witness:
        assert counters["witness_length"] == len(witness)
        assert len(witness) >= 5 and len(witness) % 2 == 1
        assert len(witness) <= counters["max_path_vertices"] + 2
        # The search stops inside the wedge loops: wedges after the one that
        # leads to the hole are never generated.
        assert 1 <= initial <= wedges
        assert 1 <= explored <= initial + created
    else:
        assert counters["witness_length"] == 0
        assert initial == wedges
        assert explored == initial + created


def _check_same_order_as_baseline(adj_list, witness, stats):
    baseline_witness, baseline, wedges_examined = _baseline_search(adj_list)

    # Same sequence of examined paths: same cycle, rotation and orientation.
    assert witness == baseline_witness
    for name in ORDER_COUNTERS:
        assert stats[name] == baseline[name], name
    # This build generates exactly the wedges the baseline examined.
    assert stats["initial_wedges_generated"] == wedges_examined
    assert stats["initial_wedges_generated"] <= baseline["initial_wedges_generated"]
    assert stats["max_pending_paths"] <= baseline["max_pending_paths"]


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("G", [nx.cycle_graph(5), nx.cycle_graph(6)], ids=["C5", "C6"])
def test_normal_mode_keeps_return_types(G):
    adj_list, A = _inputs(G)

    for result in (
        native.has_odd_holes_fast(adj_list, A),
        native.has_odd_holes_fast(adj_list, A, diagnostics=False),
    ):
        assert type(result) is bool

    for result in (
        native.find_odd_hole(adj_list, A),
        native.find_odd_hole(adj_list, A, diagnostics=False),
    ):
        assert type(result) is list
        assert all(type(vertex) is int for vertex in result)

    assert native.has_odd_holes_fast(adj_list, A) is (
        native.has_odd_holes_fast(adj_list, A, diagnostics=False)
    )
    assert native.find_odd_hole(adj_list, A) == (
        native.find_odd_hole(adj_list, A, diagnostics=False)
    )


@pytest.mark.parametrize("G", [nx.cycle_graph(5), nx.cycle_graph(6)], ids=["C5", "C6"])
def test_diagnostic_mode_returns_result_and_stats(G):
    adj_list, A = _inputs(G)

    for function, result_type in (
        (native.has_odd_holes_fast, bool),
        (native.find_odd_hole, list),
    ):
        returned = function(adj_list, A, diagnostics=True)
        assert type(returned) is tuple and len(returned) == 2
        result, stats = returned
        assert type(result) is result_type
        assert result == function(adj_list, A)
        assert type(stats) is dict
        assert set(stats) == set(COUNTERS) | {TIME_KEY}
        for name in COUNTERS:
            assert type(stats[name]) is int and stats[name] >= 0
        assert type(stats[TIME_KEY]) is float and stats[TIME_KEY] >= 0.0


# ---------------------------------------------------------------------------
# Exact counters, both completion paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, build, expected, baseline", EXACT_CASES, ids=EXACT_IDS)
def test_exact_counters(name, build, expected, baseline):
    G = build()
    adj_list, A = _inputs(G)

    witness, stats = native.find_odd_hole(adj_list, A, diagnostics=True)
    found, stats_boolean = native.has_odd_holes_fast(adj_list, A, diagnostics=True)

    assert _counters(stats) == expected
    assert _counters(stats_boolean) == expected
    assert found is bool(witness)
    assert bool(witness) is (expected[-1] > 0)
    # Same search order in both modes: the same cycle, rotation and orientation.
    assert witness == native.find_odd_hole(adj_list, A)
    assert found is native.has_odd_holes_fast(adj_list, A)
    _check_consistency(G, witness, stats)
    _check_same_order_as_baseline(adj_list, witness, stats)


def test_reference_searches_reproduce_the_plan_table():
    for name, build, expected, baseline in EXACT_CASES:
        adj_list, _ = _inputs(build())
        witness, reference = _reference_search(adj_list)
        baseline_witness, baseline_counters, _ = _baseline_search(adj_list)
        assert _counters(reference) == expected, name
        assert _counters(baseline_counters) == baseline, name
        assert witness == baseline_witness, name


def test_random_graphs_match_reference_search():
    outcomes = set()
    for G in RANDOM_GRAPHS:
        adj_list, A = _inputs(G)
        normal = native.find_odd_hole(adj_list, A)
        witness, stats = native.find_odd_hole(adj_list, A, diagnostics=True)
        found, stats_boolean = native.has_odd_holes_fast(
            adj_list, A, diagnostics=True
        )
        reference_witness, reference = _reference_search(adj_list)

        assert witness == normal == reference_witness
        assert found is bool(normal) is native.has_odd_holes_fast(adj_list, A)
        assert _counters(stats) == _counters(reference)
        assert _counters(stats_boolean) == _counters(reference)
        _check_consistency(G, witness, stats)
        _check_same_order_as_baseline(adj_list, witness, stats)
        if witness:
            odd_hole.verify_odd_hole(G, witness)
        outcomes.add(bool(witness))

    # The sample must exercise both completion paths.
    assert outcomes == {True, False}


# ---------------------------------------------------------------------------
# Counters reset between calls
# ---------------------------------------------------------------------------

def test_counters_reset_between_calls():
    large = _inputs(_pm(4, False))            # unsuccessful, 12396 wedges
    hole = _inputs(nx.cycle_graph(7))         # successful
    perfect = _inputs(nx.cycle_graph(6))      # unsuccessful
    empty = _inputs(nx.complete_graph(4))     # no path examined

    def counters(inputs, function=native.find_odd_hole):
        return _counters(function(*inputs, diagnostics=True)[1])

    fresh = {name: counters(inputs) for name, inputs in
             [("large", large), ("hole", hole), ("perfect", perfect), ("empty", empty)]}
    assert fresh["hole"] == (1, 1, 2, 1, 1, 5, 7)
    assert fresh["perfect"] == (6, 1, 12, 6, 1, 5, 0)
    assert fresh["empty"] == (0, 0, 0, 0, 0, 0, 0)

    # Repeats, interleaved successes and failures, and both functions.
    sequence = ["large", "hole", "hole", "perfect", "large", "empty", "perfect",
                "hole", "empty", "large"]
    by_name = {"large": large, "hole": hole, "perfect": perfect, "empty": empty}
    for function in (native.find_odd_hole, native.has_odd_holes_fast):
        for name in sequence:
            assert counters(by_name[name], function) == fresh[name], name

    # Normal calls in between leave the next diagnostic call untouched.
    native.find_odd_hole(*large)
    native.has_odd_holes_fast(*hole)
    assert counters(perfect) == fresh["perfect"]


# ---------------------------------------------------------------------------
# Wedge-generation time: stored on both exits of the search
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "build, found, wedges",
    [
        (lambda: _pm(4, False), False, 12396),   # final return, every wedge generated
        (lambda: _pm(4, True), True, 75),        # return from inside the wedge loops
        (lambda: _pm(6, True), True, 699),
    ],
    ids=["PM-J4-off", "PM-J4-on", "PM-J6-on"],
)
def test_wedge_generation_time_is_stored_on_both_exits(build, found, wedges):
    adj_list, A = _inputs(build())
    for function in (native.find_odd_hole, native.has_odd_holes_fast):
        started = time.perf_counter()
        result, stats = function(adj_list, A, diagnostics=True)
        elapsed = time.perf_counter() - started

        assert bool(result) is found
        assert stats["initial_wedges_generated"] == wedges
        # A return that does not store the value would report 0.0.
        assert 0.0 < stats[TIME_KEY] <= elapsed


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

def test_diagnostics_is_keyword_only():
    adj_list, A = _inputs(nx.cycle_graph(5))
    for function in (native.has_odd_holes_fast, native.find_odd_hole):
        with pytest.raises(TypeError):
            function(adj_list, A, True)


def test_invalid_input_raises_in_diagnostic_mode():
    adj_list, A = _inputs(nx.cycle_graph(5))
    bad = A.copy()
    bad[0, 1] = 0  # disagrees with adj_list
    for function in (native.has_odd_holes_fast, native.find_odd_hole):
        with pytest.raises(ValueError):
            function(adj_list, bad, diagnostics=True)
        with pytest.raises(ValueError):
            function(adj_list, bad)


def test_python_helpers_are_unchanged():
    G = nx.cycle_graph(7)
    assert len(odd_hole.find_odd_hole_witness(G)) == 7
    assert odd_hole.find_odd_hole_or_antihole(nx.cycle_graph(6)) == (None, [])


# ---------------------------------------------------------------------------
# Same witnesses as the recorded benchmark run
# ---------------------------------------------------------------------------

# Vertex indices of the odd hole returned for every odd-hole case of the benchmark
# run 20261107_2238 of the SupportRestriction2026 package (build d701657, before
# the one-wedge-at-a-time change): the "new_witness" series, "hole" entries, of
# benchmarks/results/20261107_2238/witnesses.json.  Copied here because the
# benchmarks are not part of this repository.
RECORDED_WITNESSES = {
    "iv_excl_on__iv4": [4, 10, 15, 14, 3],
    "iv_excl_on__iv6": [6, 16, 23, 22, 5],
    "iv_excl_on__iv8": [8, 22, 31, 30, 7],
    "iv_excl_on__iv12": [12, 34, 47, 46, 11],
    "iv_excl_on__iv16": [16, 46, 63, 62, 15],
    "iv_excl_on__iv24": [24, 70, 95, 94, 23],
    "iv_excl_on__iv32": [32, 94, 127, 126, 31],
    "iv_excl_on__iv48": [48, 142, 191, 190, 47],
    "iv_excl_on__iv64": [64, 190, 255, 254, 63],
    "pm_excl_on__pm2": [5, 2, 15, 9, 6],
    "pm_excl_on__pm3": [9, 6, 31, 21, 10],
    "pm_excl_on__pm4": [17, 14, 63, 45, 18],
    "pm_excl_on__pm5": [33, 30, 127, 93, 34],
    "pm_excl_on__pm6": [65, 62, 255, 189, 66],
}


def test_witnesses_match_recorded_run():
    recorded = RECORDED_WITNESSES

    # Every odd-hole case of the run. The larger ones generate a handful of
    # wedges in this build (the baseline stored up to a million of them).
    builders = {
        f"iv_excl_on__iv{K}": (lambda K=K: _iv(K)) for K in (4, 6, 8, 12, 16, 24, 32, 48, 64)
    }
    builders.update({
        f"pm_excl_on__pm{J}": (lambda J=J: _pm(J, True)) for J in (2, 3, 4, 5, 6)
    })
    assert set(builders) == set(recorded)
    for case_id, build in builders.items():
        adj_list, A = _inputs(build())
        assert native.find_odd_hole(adj_list, A) == recorded[case_id], case_id
        witness, stats = native.find_odd_hole(adj_list, A, diagnostics=True)
        assert witness == recorded[case_id], case_id
        assert stats["witness_length"] == len(recorded[case_id])
