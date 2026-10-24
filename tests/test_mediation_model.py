"""Tests for ``mediation_model.MediationModel``.

Groups: (1) one binary mediator; (2) two binary mediators, all versus selected;
(3) nonbinary supports and three-valued treatment; (4) three binary mediators;
(5) invariance to support order and coordinate relabelling; (6) validation and
the remaining interface checks.

Expected edges are hand-derived with reasons or obtained by enumerating every
potential-response vector (Method 2). The enumeration never calls model methods.

Node convention: (y, m, d), with m a tuple even for one mediator. Treatment values
d define the groups; same-d nodes are never adjacent.
"""

import ast
import inspect
from itertools import combinations, product

import networkx as nx
import numpy as np
import pytest

import mediation_model as mm_module
from mediation_model import MediationModel
from iv_model import IVModel

# --------------------------------------------------------------------------- helpers

ONE_BINARY = dict(y_support=(0, 1), m_supports=((0, 1),), d_support=(0, 1))
TWO_BINARY = dict(y_support=(0, 1), m_supports=((0, 1), (0, 1)), d_support=(0, 1))
NONBINARY = dict(y_support=(0, 1, 2), m_supports=((0, 1), (10, 20, 30)), d_support=(0, 1, 2))
THREE_BINARY = dict(y_support=(0, 1), m_supports=((0, 1), (0, 1), (0, 1)), d_support=(0, 1))


def edge_set(G):
    """Edges of G as unordered pairs, independent of insertion orientation."""
    return {frozenset(edge) for edge in G.edges()}


def as_edges(pairs):
    return {frozenset(pair) for pair in pairs}


def cross_group_pairs(nodes):
    return as_edges((u, v) for u, v in combinations(nodes, 2) if u[2] != v[2])


def method2_edges(y_support, m_supports, d_support, active):
    """Enumerate |M|**|D| mediator maps and |Y|**|profiles| outcome functions.

    Each M(.) assigns a full mediator vector to every treatment. Each g assigns
    an outcome to every selected profile, including profiles unused by M(.).
    The generated events are (g(M(d)_S), M(d), d); their cross-group pairs are
    feasible by construction. No MediationModel method is called.
    """
    Y, D = tuple(sorted(y_support)), tuple(sorted(d_support))
    M = tuple(product(*(tuple(sorted(s)) for s in m_supports)))
    profiles = tuple(sorted({tuple(m[j] for j in active) for m in M}))
    edges = set()
    for mvec in product(M, repeat=len(D)):
        m_of = dict(zip(D, mvec))
        for yvec in product(Y, repeat=len(profiles)):
            g = dict(zip(profiles, yvec))
            generated = [
                (g[tuple(m_of[d][j] for j in active)], m_of[d], d) for d in D
            ]
            for u, v in combinations(generated, 2):
                if u[2] != v[2]:
                    edges.add(frozenset((u, v)))
    return edges


# --------------------------------------------------------------------------- (1) one binary mediator

ONE_BINARY_NODES = (
    (0, (0,), 0), (0, (0,), 1), (0, (1,), 0), (0, (1,), 1),
    (1, (0,), 0), (1, (0,), 1), (1, (1,), 0), (1, (1,), 1),
)
# The same mediator value cannot fix g(m) to two different outcomes:
# 2 mediator values x 2 unequal outcome orientations = 4 forbidden pairs.
ONE_BINARY_REMOVED = as_edges(
    ((y, (m,), 0), (1 - y, (m,), 1)) for m in (0, 1) for y in (0, 1)
)
ONE_BINARY_CROSS = cross_group_pairs(ONE_BINARY_NODES)
ONE_BINARY_EDGES = ONE_BINARY_CROSS - ONE_BINARY_REMOVED


def test_one_binary_nodes_and_edges_by_hand():
    """Two groups of four nodes give 16 cross pairs; removing 4 leaves 12."""
    model = MediationModel(**ONE_BINARY)
    assert model.nodes == model._make_nodes() == ONE_BINARY_NODES
    assert model.m_support == ((0,), (1,))
    assert model.m_supports == ((0, 1),) and model.n_mediators == 1
    assert all(isinstance(m, tuple) and len(m) == 1 for _, m, _ in model.nodes)
    G = model.build_graph()
    assert tuple(G.nodes()) == ONE_BINARY_NODES
    assert G.number_of_nodes() == 8
    assert len(ONE_BINARY_CROSS) == 16 and len(ONE_BINARY_REMOVED) == 4
    assert edge_set(G) == ONE_BINARY_EDGES
    assert G.number_of_edges() == 12
    assert all(u[2] != v[2] for u, v in G.edges())
    assert nx.is_bipartite(G)


def test_one_binary_agrees_with_method2():
    G = MediationModel(**ONE_BINARY).build_graph()
    assert edge_set(G) == method2_edges(**ONE_BINARY, active=(0,))


@pytest.mark.parametrize("y_support, m_support, d_support", [
    ((0, 1), (0, 1), (0, 1)),
    ((0, 1, 2), (0, 2, 5), (0, 1, 2)),
])
def test_one_mediator_matches_ivmodel(y_support, m_support, d_support):
    """One mediator takes the IV treatment role; D takes the instrument role."""
    model = MediationModel(y_support, (m_support,), d_support)
    scalar = IVModel(y_support, m_support, d_support, exclusion=True)
    assert tuple((y, m[0], d) for y, m, d in model.nodes) == scalar.nodes
    relabelled = {
        frozenset((y, m[0], d) for y, m, d in edge)
        for edge in edge_set(model.build_graph())
    }
    assert relabelled == edge_set(scalar.build_graph())


# --------------------------------------------------------------------------- (2) two binary mediators

M2 = ((0, 0), (0, 1), (1, 0), (1, 1))
TWO_BINARY_NODES = tuple((y, m, d) for y in (0, 1) for m in M2 for d in (0, 1))
TWO_BINARY_CROSS = cross_group_pairs(TWO_BINARY_NODES)
# All selected: 4 full profiles x 2 unequal outcome orientations = 8 contradictions.
TWO_BINARY_REMOVED_ALL = as_edges(
    ((y, m, 0), (1 - y, m, 1)) for m in M2 for y in (0, 1)
)
# Only M1 selected: each of its 2 values has 2 choices of M2 at each treatment.
# These 2 x 2 x 2 x 2 = 16 pairs demand two outcomes from the same g(M1).
TWO_BINARY_REMOVED_FIRST = as_edges(
    ((y, m, 0), (1 - y, mp, 1))
    for m in M2 for mp in M2 if m[0] == mp[0] for y in (0, 1)
)
TWO_BINARY_EDGES = {
    "all": TWO_BINARY_CROSS - TWO_BINARY_REMOVED_ALL,
    (0,): TWO_BINARY_CROSS - TWO_BINARY_REMOVED_FIRST,
}


@pytest.mark.parametrize("active, count", [("all", 56), ((0,), 48)])
def test_two_binary_nodes_and_edges_by_hand(active, count):
    """Two groups of eight give 64 cross pairs; remove 8 or 16, leaving 56 or 48."""
    model = MediationModel(**TWO_BINARY, active_mediators=active)
    assert model.nodes == model._make_nodes() == TWO_BINARY_NODES
    assert model.m_support == M2 and model.n_mediators == 2
    G = model.build_graph()
    assert tuple(G.nodes()) == TWO_BINARY_NODES
    assert G.number_of_nodes() == 16 and len(TWO_BINARY_CROSS) == 64
    assert len(TWO_BINARY_REMOVED_ALL) == 8
    assert len(TWO_BINARY_REMOVED_FIRST) == 16
    assert edge_set(G) == TWO_BINARY_EDGES[active]
    assert G.number_of_edges() == count
    assert all(u[2] != v[2] for u, v in G.edges())
    assert nx.is_bipartite(G)


@pytest.mark.parametrize("active, selected", [("all", (0, 1)), ((0,), (0,))])
def test_two_binary_agrees_with_method2(active, selected):
    G = MediationModel(**TWO_BINARY, active_mediators=active).build_graph()
    assert edge_set(G) == method2_edges(**TWO_BINARY, active=selected)


@pytest.mark.parametrize("v, all_edge, first_edge", [
    # Equal full profile, unequal outcomes: g would have two values.
    ((1, (0, 0), 1), False, False),
    # Only M2 differs: it may affect the outcome only when selected.
    ((1, (0, 1), 1), True, False),
    # M1 differs: the outcomes belong to different arguments of g.
    ((1, (1, 0), 1), True, True),
    # Equal outcomes never contradict g, even with equal selected profiles.
    ((0, (0, 1), 1), True, True),
])
def test_two_binary_pair_table_both_orders(v, all_edge, first_edge):
    u = (0, (0, 0), 0)
    for active, expected in (("all", all_edge), ((0,), first_edge)):
        model = MediationModel(**TWO_BINARY, active_mediators=active)
        G = model.build_graph()
        for a, b in ((u, v), (v, u)):
            assert G.has_edge(a, b) is expected
            assert model.violates_mediation(a, b) is (not expected)
            assert model.violate_pairwise_fn(a, b) is (not expected)


def test_selecting_only_first_removes_exactly_eight_more_pairs():
    """Equal M1, different M2 and different Y account for the extra 8 exclusions."""
    all_edges = edge_set(MediationModel(**TWO_BINARY).build_graph())
    first_edges = edge_set(MediationModel(**TWO_BINARY, active_mediators=(0,)).build_graph())
    # 2 shared M1 values x 2 orientations of M2 x 2 orientations of Y = 8.
    difference = as_edges(
        ((y, (m1, m2), 0), (1 - y, (m1, 1 - m2), 1))
        for m1, m2, y in product((0, 1), repeat=3)
    )
    assert first_edges < all_edges
    assert all_edges - first_edges == difference
    assert len(difference) == 8


@pytest.mark.parametrize("selection", [(0, 1), [1, 0]])
def test_explicit_full_selection_equals_all(selection):
    reference = MediationModel(**TWO_BINARY)
    model = MediationModel(**TWO_BINARY, active_mediators=selection)
    assert model.active_mediators == reference.active_mediators == (0, 1)
    assert model.nodes == reference.nodes
    assert repr(model) == repr(reference)
    assert edge_set(model.build_graph()) == TWO_BINARY_EDGES["all"]


def test_second_coordinate_is_mirror_of_first():
    first = MediationModel(**TWO_BINARY, active_mediators=(0,))
    second = MediationModel(**TWO_BINARY, active_mediators=(1,))
    relabel = lambda node: (node[0], (node[1][1], node[1][0]), node[2])
    G = nx.relabel_nodes(second.build_graph(), relabel)
    assert set(G.nodes()) == set(first.nodes)
    assert edge_set(G) == edge_set(first.build_graph())


def test_projection_to_selected_coordinate_preserves_adjacency():
    """Each selected profile has two lifts; projection preserves every adjacency."""
    model = MediationModel(**TWO_BINARY, active_mediators=(0,))
    G = model.build_graph()
    projected = MediationModel(**ONE_BINARY).build_graph()
    project_node = lambda node: (node[0], (node[1][0],), node[2])
    for u, v in product(model.nodes, repeat=2):
        assert G.has_edge(u, v) == projected.has_edge(project_node(u), project_node(v))


# --------------------------------------------------------------------------- (3) nonbinary supports and three-valued treatment

M_NONBINARY = ((0, 10), (0, 20), (0, 30), (1, 10), (1, 20), (1, 30))


@pytest.mark.parametrize("active, removed, count", [("all", 108, 864), ((1,), 216, 756)])
def test_nonbinary_counts(active, removed, count):
    """54 nodes; 3 x 18**2 = 972 cross pairs. Remove 3 x 6 x 1 x 6 = 108
    with all selected, or 3 x 3 x 4 x 6 = 216 with only M2 selected."""
    model = MediationModel(**NONBINARY, active_mediators=active)
    assert model.m_support == M_NONBINARY
    expected_nodes = tuple(
        (y, m, d) for y in (0, 1, 2) for m in M_NONBINARY for d in (0, 1, 2)
    )
    assert model.nodes == expected_nodes
    G = model.build_graph()
    assert tuple(G.nodes()) == expected_nodes and G.number_of_nodes() == 54
    cross = cross_group_pairs(expected_nodes)
    assert len(cross) == 972 and len(cross - edge_set(G)) == removed
    assert G.number_of_edges() == count
    assert edge_set(G) <= cross


@pytest.mark.parametrize("active, selected", [("all", (0, 1)), ((1,), (1,))])
def test_nonbinary_agrees_with_method2(active, selected):
    G = MediationModel(**NONBINARY, active_mediators=active).build_graph()
    assert edge_set(G) == method2_edges(**NONBINARY, active=selected)


@pytest.mark.parametrize("u, v, all_edge, second_edge", [
    # Equal full profile, unequal outcomes: impossible under either selection.
    ((2, (0, 10), 0), (1, (0, 10), 1), False, False),
    # Equal outcomes are consistent even though M1 differs.
    ((2, (0, 10), 0), (2, (1, 10), 2), True, True),
    # Equal M2 = 10 and unequal outcomes: possible only if M1 can affect g.
    ((2, (0, 10), 0), (1, (1, 10), 2), True, False),
    # Equal outcomes are consistent at different M2 values.
    ((0, (1, 20), 1), (0, (1, 30), 2), True, True),
    # Both selected profiles differ, so both outcomes can be assigned.
    ((0, (1, 20), 1), (1, (0, 30), 2), True, True),
    # Equal full profile again demands two values from one argument of g.
    ((1, (0, 30), 0), (2, (0, 30), 2), False, False),
])
def test_nonbinary_hand_checked_pairs_both_orders(u, v, all_edge, second_edge):
    for active, expected in (("all", all_edge), ((1,), second_edge)):
        model = MediationModel(**NONBINARY, active_mediators=active)
        G = model.build_graph()
        for a, b in ((u, v), (v, u)):
            assert G.has_edge(a, b) is expected
            assert model.violates_mediation(a, b) is (not expected)
            assert model.violate_pairwise_fn(a, b) is (not expected)


def test_three_valued_treatment_table4_and_ivmodel():
    """Three groups give 3 x 12 = 36 edges; Table 4 has 28 maximal cliques
    and 15 maximal independent sets, split into 3 single-group and 12 spanning."""
    model = MediationModel((0, 1), ((0, 1),), (0, 1, 2))
    G = model.build_graph()
    assert G.number_of_nodes() == 12 and G.number_of_edges() == 36
    assert edge_set(G) == method2_edges((0, 1), ((0, 1),), (0, 1, 2), (0,))
    scalar = IVModel((0, 1), (0, 1), (0, 1, 2), exclusion=True).build_graph()
    relabelled = nx.relabel_nodes(G, lambda node: (node[0], node[1][0], node[2]))
    assert set(relabelled.nodes()) == set(scalar.nodes())
    assert edge_set(relabelled) == edge_set(scalar)
    assert len(list(nx.find_cliques(G))) == 28
    independent_sets = list(nx.find_cliques(nx.complement(G)))
    single_group = [s for s in independent_sets if len({node[2] for node in s}) == 1]
    spanning = [s for s in independent_sets if len({node[2] for node in s}) > 1]
    assert len(independent_sets) == 15
    assert len(single_group) == 3 and len(spanning) == 12


def test_three_valued_treatment_contains_odd_hole():
    """An induced five-cycle needs three colors but has clique number two."""
    model = MediationModel((0, 1), ((0, 1),), (0, 1, 2))
    G = model.build_graph()
    for subset in combinations(model.nodes, 5):
        H = G.subgraph(subset)
        if H.number_of_edges() == 5 and all(degree == 2 for _, degree in H.degree()):
            if nx.is_connected(H):
                break
    else:
        pytest.fail("Expected an induced five-cycle, witnessing non-perfectness.")
    assert H.number_of_nodes() == 5 and H.number_of_edges() == 5
    assert nx.is_connected(H) and all(degree == 2 for _, degree in H.degree())


# --------------------------------------------------------------------------- (4) three binary mediators

def test_three_binary_counts_and_subset():
    """32 nodes; 256 cross pairs minus 4 profiles x 4 lifts x 2 outcomes = 224.
    Selecting all coordinates removes only 8 x 2 = 16 pairs, leaving 240."""
    model = MediationModel(**THREE_BINARY, active_mediators=(0, 2))
    G = model.build_graph()
    assert model.n_mediators == 3 and model.active_mediators == (0, 2)
    assert G.number_of_nodes() == len(model.nodes) == 32
    cross = cross_group_pairs(model.nodes)
    assert len(cross) == 256 and len(cross - edge_set(G)) == 32
    assert G.number_of_edges() == 224
    all_edges = edge_set(MediationModel(**THREE_BINARY).build_graph())
    assert len(all_edges) == 240
    assert edge_set(G) < all_edges


def test_three_binary_agrees_with_method2():
    G = MediationModel(**THREE_BINARY, active_mediators=(0, 2)).build_graph()
    assert edge_set(G) == method2_edges(**THREE_BINARY, active=(0, 2))


@pytest.mark.parametrize("v, expected", [
    # Only unselected M2 differs: the same g(M1, M3) cannot give unequal outcomes.
    ((1, (0, 1, 0), 1), False),
    # M3 differs, so g has two different arguments.
    ((1, (0, 0, 1), 1), True),
    # M1 differs, likewise allowing distinct outcomes.
    ((1, (1, 0, 0), 1), True),
    # Equal outcomes agree at the shared selected profile.
    ((0, (0, 1, 0), 1), True),
])
def test_three_binary_selected_coordinates_both_orders(v, expected):
    u = (0, (0, 0, 0), 0)
    model = MediationModel(**THREE_BINARY, active_mediators=(0, 2))
    G = model.build_graph()
    for a, b in ((u, v), (v, u)):
        assert G.has_edge(a, b) is expected
        assert model.violates_mediation(a, b) is (not expected)
        assert model.violate_pairwise_fn(a, b) is (not expected)


# --------------------------------------------------------------------------- (5) invariance

@pytest.mark.parametrize("active", ["all", (1,)])
@pytest.mark.parametrize("y_support, m_supports, d_support", [
    ([2, 0, 1], [[1, 0], (30, 10, 20)], (2, 0, 1)),
    (np.array([1, 2, 0]), (range(1, -1, -1), np.array([20, 30, 10])), range(2, -1, -1)),
])
def test_reordering_support_values_leaves_graph_unchanged(active, y_support, m_supports, d_support):
    reference = MediationModel(**NONBINARY, active_mediators=active)
    permuted = MediationModel(y_support, m_supports, d_support, active)
    assert permuted.y_support == reference.y_support == (0, 1, 2)
    assert permuted.d_support == reference.d_support == (0, 1, 2)
    assert permuted.m_supports == reference.m_supports == ((0, 1), (10, 20, 30))
    assert permuted.m_support == reference.m_support == M_NONBINARY
    assert permuted.nodes == reference.nodes
    assert edge_set(permuted.build_graph()) == edge_set(reference.build_graph())


def test_numpy_arrays_and_ranges_are_converted():
    model = MediationModel(
        np.array([1, 0]), (np.array([0.5, 0.0]), [np.int64(3), np.int64(1)]), range(2)
    )
    assert model.y_support == (0, 1) and model.d_support == (0, 1)
    assert model.m_supports == ((0.0, 0.5), (1, 3))
    assert model.m_support == ((0.0, 1), (0.0, 3), (0.5, 1), (0.5, 3))
    for support in (model.y_support, model.d_support, *model.m_supports):
        assert all(type(value) in (int, float) for value in support)
    # Renaming mediator values preserves 64 cross pairs minus 4 x 2 contradictions.
    assert model.build_graph().number_of_edges() == 56
    relabel = lambda node: (node[0], (node[1][0] / 2, 1 + 2 * node[1][1]), node[2])
    expected = {frozenset(map(relabel, edge)) for edge in TWO_BINARY_EDGES["all"]}
    assert edge_set(model.build_graph()) == expected


def test_swapping_coordinates_relabels_nodes_and_selection():
    original = MediationModel((0, 1), ((0, 1), (10, 20, 30)), (0, 1), (1,))
    swapped = MediationModel((0, 1), ((10, 20, 30), (0, 1)), (0, 1), (0,))
    assert swapped.m_supports == ((10, 20, 30), (0, 1))
    assert swapped.m_support == ((10, 0), (10, 1), (20, 0), (20, 1), (30, 0), (30, 1))
    relabel = lambda node: (node[0], (node[1][1], node[1][0]), node[2])
    assert set(map(relabel, swapped.nodes)) == set(original.nodes)
    relabelled = {frozenset(map(relabel, edge)) for edge in edge_set(swapped.build_graph())}
    assert relabelled == edge_set(original.build_graph())


# --------------------------------------------------------------------------- (6) validation and interface

@pytest.mark.parametrize("bad, exc, match", [
    ((0, 1), TypeError, r"m_supports\[0\].*\(\(0, 1\),\)"),  # flat support, with hint
    ("01", TypeError, "m_supports"),
    (b"01", TypeError, "m_supports"),
    (5, TypeError, "m_supports"),
    (None, TypeError, "m_supports"),
    ((), ValueError, "m_supports"),                         # no coordinates
    (((),), ValueError, r"m_supports\[0\]"),
    (((0, 1), ()), ValueError, r"m_supports\[1\]"),
    (((0, 1), (1, 1)), ValueError, r"m_supports\[1\]"),
    (((0, 1), "ab"), TypeError, r"m_supports\[1\]"),
    (((0, 1), 5), TypeError, r"m_supports\[1\]"),
    ((("a", "b"),), TypeError, r"m_supports\[0\]"),
    (((0, 1j),), TypeError, r"m_supports\[0\]"),
    (((0, float("nan")),), ValueError, r"m_supports\[0\]"),
    (((0, float("inf")),), ValueError, r"m_supports\[0\]"),
    (((0, float("-inf")),), ValueError, r"m_supports\[0\]"),
    (((0, True),), TypeError, r"m_supports\[0\]"),
    (((0, np.bool_(True)),), TypeError, r"m_supports\[0\]"),
])
def test_invalid_m_supports(bad, exc, match):
    with pytest.raises(exc, match=match):
        MediationModel((0, 1), bad, (0, 1))


@pytest.mark.parametrize("bad, exc", [
    ("ALL", ValueError), ("", ValueError), ("none", ValueError),
    (None, TypeError), (True, TypeError), (False, TypeError),
    (0, TypeError), (0.5, TypeError), (b"0", TypeError), (bytearray(b"0"), TypeError),
    ((True,), TypeError), ((0.0,), TypeError), (("0",), TypeError),
    ((np.bool_(True),), TypeError),
    ((), ValueError), ((0, 0), ValueError), ((-1,), ValueError),
    ((2,), ValueError), ((0, 2), ValueError),
])
def test_invalid_active_mediators(bad, exc):
    with pytest.raises(exc, match="active_mediators"):
        MediationModel(**TWO_BINARY, active_mediators=bad)


def test_bare_active_index_message_gives_tuple_hint():
    with pytest.raises(TypeError, match=r"active_mediators=\(0,\)"):
        MediationModel(**TWO_BINARY, active_mediators=0)


@pytest.mark.parametrize("make_selection", [
    lambda: [1, 0],
    lambda: np.array([1, 0], dtype=np.int64),
    lambda: {1, 0},
    lambda: (j for j in (1, 0)),
])
def test_active_selection_iterables_are_sorted_plain_ints(make_selection):
    model = MediationModel(**TWO_BINARY, active_mediators=make_selection())
    assert model.active_mediators == (0, 1)
    assert all(type(index) is int for index in model.active_mediators)
    assert edge_set(model.build_graph()) == TWO_BINARY_EDGES["all"]


@pytest.mark.parametrize("name", ["y_support", "d_support"])
@pytest.mark.parametrize("bad, exc", [
    ("01", TypeError), (b"01", TypeError), (3, TypeError), (None, TypeError),
    ((), ValueError), ((1, 1), ValueError),
    ((0, "a"), TypeError), ((0, 1j), TypeError),
    ((0, True), TypeError), ((0, np.bool_(True)), TypeError),
    ((0, float("nan")), ValueError), ((0, float("inf")), ValueError),
    ((0, float("-inf")), ValueError),
])
def test_invalid_outcome_and_treatment_supports(name, bad, exc):
    kwargs = dict(TWO_BINARY)
    kwargs[name] = bad
    with pytest.raises(exc, match=name):
        MediationModel(**kwargs)


def test_default_selects_all_coordinates():
    model = MediationModel(**TWO_BINARY)
    assert model.active_mediators == (0, 1)
    assert repr(model) == repr(MediationModel(**TWO_BINARY, active_mediators="all"))
    assert edge_set(model.build_graph()) == TWO_BINARY_EDGES["all"]


def test_positional_arguments_in_listed_order():
    model = MediationModel((0, 1), ((0, 1), (0, 1)), (0, 1), (0,))
    assert model.y_support == model.d_support == (0, 1)
    assert model.m_supports == ((0, 1), (0, 1)) and model.active_mediators == (0,)
    assert edge_set(model.build_graph()) == TWO_BINARY_EDGES[(0,)]


@pytest.mark.parametrize("active, resolved", [("all", (0, 1)), ((1,), (1,))])
def test_repr_round_trips_with_resolved_selection(active, resolved):
    model = MediationModel(**NONBINARY, active_mediators=active)
    assert repr(model) == (
        "MediationModel(y_support=(0, 1, 2), m_supports=((0, 1), (10, 20, 30)), "
        f"d_support=(0, 1, 2), active_mediators={resolved!r})"
    )
    rebuilt = eval(repr(model), {"MediationModel": MediationModel})
    assert rebuilt.nodes == model.nodes
    assert rebuilt.active_mediators == resolved
    assert repr(rebuilt) == repr(model)
    assert edge_set(rebuilt.build_graph()) == edge_set(model.build_graph())


def test_settings_are_fixed_after_construction():
    model = MediationModel(**TWO_BINARY)
    original = vars(model).copy()
    for name in (*original, "new", "build_graph"):
        with pytest.raises(AttributeError, match="fixed after construction"):
            setattr(model, name, None)
        with pytest.raises(AttributeError, match="fixed after construction"):
            delattr(model, name)
    assert vars(model) == original
    assert edge_set(model.build_graph()) == TWO_BINARY_EDGES["all"]


@pytest.mark.parametrize("active", ["all", (0,), (1,)])
def test_predicates_return_bool_are_symmetric_and_agree(active):
    model = MediationModel(**TWO_BINARY, active_mediators=active)
    for u, v in product(model.nodes, repeat=2):
        for fn in (model.violates_mediation, model.violate_pairwise_fn):
            assert type(fn(u, v)) is bool
            assert fn(u, v) is fn(v, u)
        assert model.violate_pairwise_fn(u, v) is model.violates_mediation(u, v)


def test_build_graph_returns_a_new_graph_each_time():
    model = MediationModel(**TWO_BINARY)
    G1, G2 = model.build_graph(), model.build_graph()
    assert G1 is not G2 and isinstance(G1, nx.Graph) and isinstance(G2, nx.Graph)
    assert edge_set(G1) == edge_set(G2) == TWO_BINARY_EDGES["all"]
    G1.remove_node((0, (0, 0), 0))
    assert tuple(G2.nodes()) == model.nodes == TWO_BINARY_NODES
    assert edge_set(G2) == TWO_BINARY_EDGES["all"]
    assert edge_set(model.build_graph()) == TWO_BINARY_EDGES["all"]


def test_group_fn_returns_treatment():
    model = MediationModel(**NONBINARY)
    assert all(model.group_fn(node) == node[2] for node in model.nodes)
    assert set(map(model.group_fn, model.nodes)) == {0, 1, 2}


def test_summary_describes_all_three_specifications():
    text = MediationModel((0, 1), ((0, 1), (10, 20, 30)), (0, 1), (0,)).summary()
    for fragment in (
        "Outcome support Y:    (0, 1)", "Treatment support D:  (0, 1)",
        "M1: (0, 1)", "M2: (10, 20, 30)", "6 vectors",
        "Selected mediators:   (0,) -> M1", "Nodes (y, m, d):      24",
        "Mediation: imposed (always)", "Unselected coordinates (M2)",
        "No monotonicity", "independent of D",
    ):
        assert fragment in text
    all_selected = MediationModel((0, 1), ((0, 1), (10, 20, 30)), (0, 1), "all").summary()
    assert "Every mediator coordinate is selected" in all_selected
    assert "Unselected coordinates" not in all_selected
    one = MediationModel(**ONE_BINARY).summary()
    assert "Mediator coordinates: 1, M = (M1,)" in one


def test_module_does_not_import_graph_analyzer():
    """The class only builds the graph; the analyzer is imported separately by users."""
    tree = ast.parse(inspect.getsource(mm_module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("graph_analysis_utils" in name or "GraphAnalyzer" in name for name in imported)
    assert "networkx" in imported
    assert "support_validation" in imported
    assert not hasattr(mm_module, "GraphAnalyzer")


def test_shared_validator_sorts_supports():
    validator = mm_module.validate_support
    assert validator is IVModel._validate_support.__globals__["validate_support"]
    assert validator((2, 1), "x") == IVModel._validate_support((2, 1), "x") == (1, 2)
    model = MediationModel((2, 1), ((2, 1),), (2, 1))
    assert model.y_support == model.m_supports[0] == model.d_support == (1, 2)
    with pytest.raises(ValueError, match="x must not be empty"):
        validator([], "x")


def test_single_treatment_keeps_all_nodes_without_edges():
    """One treatment group has eight nodes and no cross-group pairs."""
    model = MediationModel((0, 1), ((0, 1), (0, 1)), (3,))
    G = model.build_graph()
    assert tuple(G.nodes()) == model.nodes and G.number_of_nodes() == 8
    assert G.number_of_edges() == 0 and edge_set(G) == cross_group_pairs(model.nodes)


def test_single_outcome_allows_every_cross_pair():
    """Three groups of two nodes give 3 x 4 = 12 edges; g cannot disagree."""
    model = MediationModel((1,), ((0, 1),), (0, 1, 2))
    G = model.build_graph()
    assert G.number_of_nodes() == 6 and G.number_of_edges() == 12
    assert edge_set(G) == cross_group_pairs(model.nodes)


def test_single_selected_profile_requires_equal_outcomes():
    """One selected profile permits 2 outcomes x 2 x 2 mediator lifts = 8 edges."""
    model = MediationModel((0, 1), ((0, 1), (5,)), (0, 1), (1,))
    # g(5) has a single value; M1 is unrestricted, but both outcomes must agree.
    expected = as_edges(
        ((y, (m1, 5), 0), (y, (mp1, 5), 1))
        for y, m1, mp1 in product((0, 1), repeat=3)
    )
    G = model.build_graph()
    assert G.number_of_nodes() == 8 and G.number_of_edges() == 8
    assert edge_set(G) == expected
    assert expected == method2_edges((0, 1), ((0, 1), (5,)), (0, 1), (1,))


def test_project_mediators_selects_first_and_third():
    model = MediationModel(**THREE_BINARY, active_mediators=(0, 2))
    assert model._project_mediators((10, 20, 30)) == (10, 30)


@pytest.mark.parametrize("m", [(), (0,), (0, 1), (0, 1, 0, 1)])
def test_project_mediators_rejects_wrong_length(m):
    model = MediationModel(**THREE_BINARY, active_mediators=(0, 2))
    with pytest.raises(ValueError, match="coordinate"):
        model._project_mediators(m)
