"""Tests for ``partial_monotonicity_model.PartialMonotonicityModel``.

Groups: (1) binary outcome, binary treatment and two binary instruments; (2) the
exclusion switch; (3) one instrument against ``IVModel``; (4) small nonbinary supports;
(5) three binary instruments; (6) invariance to the order of values and relabelling of
coordinates; (7) input validation and the remaining interface checks.

Every expected edge is either written out by hand, with the reason in a comment, or
derived by enumerating potential-response vectors (Method 2 of Kaido and Ponomarev,
2025) on the Cartesian-product instrument support. None is produced by the violation
functions under test. The hand-coded rule from the repository's
``Partial_Monotonicity.ipynb`` is reproduced verbatim as a further independent check.

Node convention: ``(y, d, z)`` with ``z`` a tuple, one entry per instrument coordinate.
"""

import ast
import inspect
from itertools import combinations, product

import networkx as nx
import numpy as np
import pytest

import partial_monotonicity_model as pm_module
from iv_model import IVModel
from partial_monotonicity_model import PartialMonotonicityModel

# --------------------------------------------------------------------------- helpers

TWO_BINARY = dict(y_support=(0, 1), d_support=(0, 1), z_supports=((0, 1), (0, 1)))
NONBINARY = dict(y_support=(0, 1, 2), d_support=(0, 2, 5), z_supports=((0, 1, 2), (10, 20)))
THREE_BINARY = dict(y_support=(0, 1), d_support=(0, 1), z_supports=((0, 1), (0, 1), (0, 1)))


def edge_set(G):
    """Edges of ``G`` as unordered pairs, independent of insertion orientation."""
    return {frozenset(edge) for edge in G.edges()}


def as_edges(pairs):
    return {frozenset(pair) for pair in pairs}


def componentwise_leq(z, zp):
    return all(a <= b for a, b in zip(z, zp))


def method2_edges(y_support, d_support, z_supports, exclusion):
    """Edges from enumerating every admissible potential-response vector (Method 2).

    The instrument support is the Cartesian product of the sorted coordinate supports.
    A treatment map assigns D(z) to every instrument vector; it is kept when D(z) <=
    D(z') for every componentwise-ordered pair z <= z', which is the partial
    monotonicity assumption on a product support. Under exclusion every outcome map
    Y(d) is enumerated and the vector generates the node (Y(D(z)), D(z), z) at every z.
    Without exclusion the outcomes Y(d, z) are free cell by cell, so a treatment map
    that generates the events (d, z) and (d', z') makes every outcome pair compatible:
    all (y, d, z), (y', d', z') are edges. This construction never calls the pairwise
    rules of ``PartialMonotonicityModel``.
    """
    Y, D = tuple(sorted(y_support)), tuple(sorted(d_support))
    Z = list(product(*[tuple(sorted(s)) for s in z_supports]))
    edges = set()
    for dvec in product(D, repeat=len(Z)):
        dmap = dict(zip(Z, dvec))
        if any(componentwise_leq(a, b) and dmap[a] > dmap[b] for a in Z for b in Z):
            continue
        if exclusion:
            for ymap in product(Y, repeat=len(D)):
                y_of = dict(zip(D, ymap))
                generated = [(y_of[dmap[z]], dmap[z], z) for z in Z]
                for u, v in combinations(generated, 2):
                    edges.add(frozenset((u, v)))
        else:
            generated = [(dmap[z], z) for z in Z]
            for (d, z), (dp, zp) in combinations(generated, 2):
                for y in Y:
                    for yp in Y:
                        edges.add(frozenset(((y, d, z), (yp, dp, zp))))
    return edges


def notebook_violates_partial_monotonicity(u, v):
    """The hand-coded rule of ``Partial_Monotonicity.ipynb`` for nodes ``(d, (z1, z2))``."""
    d1, (z1_0, z1_1) = u
    d2, (z2_0, z2_1) = v
    if z1_1 == z2_1:
        if z1_0 > z2_0 and d1 < d2:
            return True
        if z1_0 < z2_0 and d1 > d2:
            return True
    if z1_0 == z2_0:
        if z1_1 > z2_1 and d1 < d2:
            return True
        if z1_1 < z2_1 and d1 > d2:
            return True
    if (z1_0 > z2_0 and z1_1 > z2_1 and d1 < d2) or (z1_0 < z2_0 and z1_1 < z2_1 and d1 > d2):
        return True
    return False


def cross_group_pairs(nodes):
    return as_edges((u, v) for u, v in combinations(nodes, 2) if u[2] != v[2])


# --------------------------------------------------------------------------- (1) two binary instruments

Z2 = ((0, 0), (0, 1), (1, 0), (1, 1))
TWO_BINARY_NODES = tuple((y, d, z) for y in (0, 1) for d in (0, 1) for z in Z2)

# The six pairs of instrument vectors. Five are comparable componentwise; (0, 1) and
# (1, 0) are incomparable, so partial monotonicity never restricts treatments there.
COMPARABLE_Z2 = [((0, 0), (0, 1)), ((0, 0), (1, 0)), ((0, 0), (1, 1)),
                 ((0, 1), (1, 1)), ((1, 0), (1, 1))]  # (lower, upper)
INCOMPARABLE_Z2 = [((0, 1), (1, 0))]

# Partial monotonicity removes, for every comparable pair, the events with D = 1 at
# the lower vector and D = 0 at the upper one, whatever the outcomes: D(lower) = 1 > 0
# = D(upper) contradicts D(lower) <= D(upper). 5 pairs x 4 outcome combinations = 20.
TWO_BINARY_REMOVED_BY_MONOTONICITY = as_edges(
    ((y, 1, lower), (yp, 0, upper))
    for lower, upper in COMPARABLE_Z2
    for y in (0, 1)
    for yp in (0, 1)
)
# Exclusion removes, for every pair of instrument vectors (comparable or not), the
# events with equal treatment and unequal outcome: both would fix Y(d) to two values.
# 6 pairs x 2 treatments x 2 orientations of (y, y') = 24.
TWO_BINARY_REMOVED_BY_EXCLUSION = as_edges(
    ((y, d, z), (1 - y, d, zp))
    for z, zp in COMPARABLE_Z2 + INCOMPARABLE_Z2
    for d in (0, 1)
    for y in (0, 1)
)
assert len(TWO_BINARY_REMOVED_BY_MONOTONICITY) == 20
assert len(TWO_BINARY_REMOVED_BY_EXCLUSION) == 24
# The two removed sets are disjoint: one needs unequal treatments, the other equal ones.
assert not (TWO_BINARY_REMOVED_BY_MONOTONICITY & TWO_BINARY_REMOVED_BY_EXCLUSION)

TWO_BINARY_ALL_CROSS = cross_group_pairs(TWO_BINARY_NODES)  # 6 x 16 = 96
TWO_BINARY_EDGES = {
    False: TWO_BINARY_ALL_CROSS - TWO_BINARY_REMOVED_BY_MONOTONICITY,                 # 76
    True: TWO_BINARY_ALL_CROSS - TWO_BINARY_REMOVED_BY_MONOTONICITY
          - TWO_BINARY_REMOVED_BY_EXCLUSION,                                          # 52
}
assert len(TWO_BINARY_ALL_CROSS) == 96
assert {k: len(v) for k, v in TWO_BINARY_EDGES.items()} == {False: 76, True: 52}


@pytest.mark.parametrize("exclusion", [True, False])
def test_two_binary_nodes(exclusion):
    model = PartialMonotonicityModel(**TWO_BINARY, exclusion=exclusion)
    assert model.nodes == TWO_BINARY_NODES
    assert len(model.nodes) == 16
    assert model.z_support == Z2
    assert model.n_instruments == 2
    G = model.build_graph()
    assert set(G.nodes()) == set(TWO_BINARY_NODES)


@pytest.mark.parametrize("exclusion", [True, False])
def test_two_binary_edge_sets_by_hand(exclusion):
    G = PartialMonotonicityModel(**TWO_BINARY, exclusion=exclusion).build_graph()
    assert edge_set(G) == TWO_BINARY_EDGES[exclusion]
    assert G.number_of_edges() == (52 if exclusion else 76)


@pytest.mark.parametrize("exclusion", [True, False])
def test_two_binary_agrees_with_method2(exclusion):
    G = PartialMonotonicityModel(**TWO_BINARY, exclusion=exclusion).build_graph()
    assert edge_set(G) == method2_edges(**TWO_BINARY, exclusion=exclusion)


@pytest.mark.parametrize("exclusion", [True, False])
@pytest.mark.parametrize("u, v, expected", [
    # (0, 0) <= (1, 1) componentwise and D falls from 1 to 0: impossible.
    ((0, 1, (0, 0)), (0, 0, (1, 1)), False),
    # (0, 0) <= (1, 1) and D rises from 0 to 1: a complier type, allowed; the outcomes
    # 1 and 0 belong to different treatments so exclusion says nothing.
    ((1, 0, (0, 0)), (0, 1, (1, 1)), True),
    # (1, 0) and (0, 1) are incomparable, so the differing treatments are unrestricted.
    ((1, 1, (1, 0)), (0, 0, (0, 1)), True),
    # One coordinate moves: (0, 0) <= (0, 1), D falls: impossible.
    ((0, 1, (0, 0)), (1, 0, (0, 1)), False),
    # One coordinate moves: (1, 0) <= (1, 1), D rises: allowed.
    ((0, 0, (1, 0)), (1, 1, (1, 1)), True),
])
def test_two_binary_named_pairs_both_orders(exclusion, u, v, expected):
    model = PartialMonotonicityModel(**TWO_BINARY, exclusion=exclusion)
    G = model.build_graph()
    assert G.has_edge(u, v) is expected
    assert G.has_edge(v, u) is expected
    assert model.violates_partial_monotonicity(u, v) is (not expected)
    assert model.violates_partial_monotonicity(v, u) is (not expected)


def test_two_binary_no_same_instrument_edges():
    G = PartialMonotonicityModel(**TWO_BINARY).build_graph()
    assert all(u[2] != v[2] for u, v in G.edges())


@pytest.mark.parametrize("exclusion", [True, False])
def test_two_binary_projection_matches_notebook_rule(exclusion):
    """Dropping the outcome recovers the graph of ``Partial_Monotonicity.ipynb``.

    With a single outcome value exclusion is vacuous, so under either setting the node
    (0, d, z) corresponds to the notebook's (d, z) and the edge sets must agree.
    """
    nb_nodes = [(d, z) for d in (0, 1) for z in Z2]
    expected = as_edges(
        (u, v) for u, v in combinations(nb_nodes, 2)
        if u[1] != v[1] and not notebook_violates_partial_monotonicity(u, v)
    )
    assert len(expected) == 19
    G = PartialMonotonicityModel((0,), (0, 1), ((0, 1), (0, 1)), exclusion=exclusion).build_graph()
    projected = {frozenset((d, z) for (y, d, z) in edge) for edge in edge_set(G)}
    assert projected == expected


# --------------------------------------------------------------------------- (2) exclusion switch

def test_exclusion_switch_on_incomparable_pair():
    """Incomparability does not override exclusion: equal D, unequal Y is rejected."""
    u, v = (0, 0, (1, 0)), (1, 0, (0, 1))
    with_exclusion = PartialMonotonicityModel(**TWO_BINARY, exclusion=True)
    without = PartialMonotonicityModel(**TWO_BINARY, exclusion=False)
    assert not with_exclusion.build_graph().has_edge(u, v)
    assert without.build_graph().has_edge(u, v)
    for model in (with_exclusion, without):
        assert model.violates_exclusion(u, v) is True
        assert model.violates_exclusion(v, u) is True
        assert model.violates_partial_monotonicity(u, v) is False
        assert model.violates_partial_monotonicity(v, u) is False
    assert with_exclusion.violate_pairwise_fn(u, v) is True
    assert without.violate_pairwise_fn(u, v) is False


@pytest.mark.parametrize("exclusion", [True, False])
@pytest.mark.parametrize("z, zp", COMPARABLE_Z2 + INCOMPARABLE_Z2)
@pytest.mark.parametrize("y, d", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_equal_outcome_and_treatment_compatible_at_different_instruments(exclusion, z, zp, y, d):
    model = PartialMonotonicityModel(**TWO_BINARY, exclusion=exclusion)
    assert model.violate_pairwise_fn((y, d, z), (y, d, zp)) is False
    assert model.build_graph().has_edge((y, d, z), (y, d, zp))


def test_exclusion_only_changes_equal_treatment_pairs():
    with_exclusion = edge_set(PartialMonotonicityModel(**TWO_BINARY, exclusion=True).build_graph())
    without = edge_set(PartialMonotonicityModel(**TWO_BINARY, exclusion=False).build_graph())
    assert with_exclusion < without
    assert without - with_exclusion == TWO_BINARY_REMOVED_BY_EXCLUSION


# --------------------------------------------------------------------------- (3) one instrument

ONE_Z_SUPPORTS = [
    ((0, 1), (0, 1), (0, 1)),
    ((0, 1), (0, 2, 5), (-1, 3, 10)),
    ((0, 1, 2), (0, 2, 5), (-1, 3, 10)),
]


@pytest.mark.parametrize("exclusion", [True, False])
@pytest.mark.parametrize("y_support, d_support, z_support", ONE_Z_SUPPORTS)
def test_one_instrument_matches_ivmodel_nondecreasing(y_support, d_support, z_support, exclusion):
    """With one coordinate, z = (z1,) and componentwise order is the scalar order."""
    vector = PartialMonotonicityModel(y_support, d_support, (z_support,), exclusion=exclusion)
    scalar = IVModel(y_support, d_support, z_support, exclusion=exclusion, d_monotonicity=True)
    assert vector.n_instruments == 1
    assert vector.z_support == tuple((z,) for z in sorted(z_support))
    assert tuple((y, d, z[0]) for (y, d, z) in vector.nodes) == scalar.nodes
    projected = {frozenset((y, d, z[0]) for (y, d, z) in e) for e in edge_set(vector.build_graph())}
    assert projected == edge_set(scalar.build_graph())


def test_one_instrument_nodes_keep_tuple_instrument():
    model = PartialMonotonicityModel((0, 1), (0, 1), ((0, 1),))
    assert model.nodes[0] == (0, 0, (0,))
    assert all(isinstance(z, tuple) and len(z) == 1 for (_, _, z) in model.nodes)
    assert model.group_fn(model.nodes[1]) == (1,)


# --------------------------------------------------------------------------- (4) small nonbinary supports

Z_NONBINARY = tuple(product((0, 1, 2), (10, 20)))  # six vectors
# Of the 15 pairs of vectors, three are incomparable: a larger first coordinate with a
# smaller second one.
INCOMPARABLE_NONBINARY = as_edges([((0, 20), (1, 10)), ((0, 20), (2, 10)), ((1, 20), (2, 10))])


@pytest.mark.parametrize("exclusion", [True, False])
def test_nonbinary_counts(exclusion):
    """54 nodes; 15 x 81 = 1215 cross-group pairs. Partial monotonicity removes, on
    each of the 12 comparable vector pairs, the 3 falling treatment pairs (5 > 2, 5 > 0,
    2 > 0) times 9 outcome combinations: 324. Exclusion removes, on all 15 vector
    pairs, 3 equal treatments times 6 unequal outcome pairs: 270. The sets are
    disjoint, so 891 edges without exclusion and 621 with it."""
    model = PartialMonotonicityModel(**NONBINARY, exclusion=exclusion)
    G = model.build_graph()
    assert len(model.nodes) == 54
    assert model.z_support == Z_NONBINARY
    assert G.number_of_edges() == (621 if exclusion else 891)


@pytest.mark.parametrize("exclusion", [True, False])
def test_nonbinary_agrees_with_method2(exclusion):
    G = PartialMonotonicityModel(**NONBINARY, exclusion=exclusion).build_graph()
    assert edge_set(G) == method2_edges(**NONBINARY, exclusion=exclusion)


@pytest.mark.parametrize("u, v, with_exclusion, without_exclusion", [
    # (0, 10) <= (2, 20), D falls 5 -> 0: impossible either way; the third outcome
    # value 2 plays no role.
    ((2, 5, (0, 10)), (0, 0, (2, 20)), False, False),
    # (0, 10) <= (2, 20), D rises 0 -> 5: allowed; outcomes 2 and 0 are at different d.
    ((2, 0, (0, 10)), (0, 5, (2, 20)), True, True),
    # (0, 20) and (1, 10) incomparable, equal D = 2, outcomes 2 != 1: only exclusion
    # can reject, and it does.
    ((2, 2, (0, 20)), (1, 2, (1, 10)), False, True),
    # (0, 20) and (2, 10) incomparable, D = 5 and D = 0 differ: allowed either way.
    ((2, 5, (0, 20)), (0, 0, (2, 10)), True, True),
    # (1, 10) <= (1, 20) with equal D = 5 and equal Y = 2: allowed either way.
    ((2, 5, (1, 10)), (2, 5, (1, 20)), True, True),
    # (2, 10) <= (2, 20) with D rising 2 -> 5: allowed either way.
    ((1, 2, (2, 10)), (0, 5, (2, 20)), True, True),
    # (2, 10) <= (2, 20) with D falling 5 -> 2: impossible either way.
    ((0, 5, (2, 10)), (2, 2, (2, 20)), False, False),
    # (1, 20) and (2, 10) incomparable, equal D = 0 and equal Y = 2: allowed either way.
    ((2, 0, (1, 20)), (2, 0, (2, 10)), True, True),
    # Only the second coordinate moves, (1, 10) <= (1, 20), D falls 2 -> 0: impossible.
    ((1, 2, (1, 10)), (1, 0, (1, 20)), False, False),
])
def test_nonbinary_hand_checked_pairs_both_orders(u, v, with_exclusion, without_exclusion):
    for exclusion, expected in ((True, with_exclusion), (False, without_exclusion)):
        G = PartialMonotonicityModel(**NONBINARY, exclusion=exclusion).build_graph()
        assert G.has_edge(u, v) is expected
        assert G.has_edge(v, u) is expected


# --------------------------------------------------------------------------- (5) three binary instruments

@pytest.mark.parametrize("exclusion", [True, False])
def test_three_binary_counts(exclusion):
    """32 nodes; 28 x 16 = 448 cross-group pairs. In the three-dimensional Boolean
    lattice 19 of the 28 vector pairs are comparable and 9 are incomparable (three
    pairs of weight-one vectors, three of weight-two vectors, and three weight-one /
    weight-two pairs without containment). Partial monotonicity removes 4 falling
    pairs on each comparable vector pair: 76. Exclusion removes 4 on every vector
    pair: 112. Hence 372 edges without exclusion and 260 with it."""
    model = PartialMonotonicityModel(**THREE_BINARY, exclusion=exclusion)
    G = model.build_graph()
    assert len(model.nodes) == 32
    assert model.n_instruments == 3
    assert G.number_of_edges() == (260 if exclusion else 372)


@pytest.mark.parametrize("exclusion", [True, False])
def test_three_binary_agrees_with_method2(exclusion):
    G = PartialMonotonicityModel(**THREE_BINARY, exclusion=exclusion).build_graph()
    assert edge_set(G) == method2_edges(**THREE_BINARY, exclusion=exclusion)


@pytest.mark.parametrize("exclusion", [True, False])
@pytest.mark.parametrize("u, v, expected", [
    # Equal first two coordinates; the third decides: (1, 1, 0) <= (1, 1, 1), D falls.
    ((0, 1, (1, 1, 0)), (0, 0, (1, 1, 1)), False),
    # Same vectors, D rises: allowed.
    ((0, 0, (1, 1, 0)), (1, 1, (1, 1, 1)), True),
    # Comparable in the first two coordinates but the third breaks it:
    # (0, 0, 1) vs (1, 1, 0) is incomparable, so differing D is allowed.
    ((0, 1, (0, 0, 1)), (0, 0, (1, 1, 0)), True),
    # (0, 1, 1) vs (1, 0, 1): incomparable in the first two coordinates, equal third.
    ((1, 1, (0, 1, 1)), (0, 0, (1, 0, 1)), True),
    # Full comparison (0, 0, 0) <= (1, 1, 1) with D falling: impossible.
    ((0, 1, (0, 0, 0)), (0, 0, (1, 1, 1)), False),
    # (0, 1, 0) <= (0, 1, 1) differ only in the third coordinate, D falls: impossible.
    ((1, 1, (0, 1, 0)), (1, 0, (0, 1, 1)), False),
])
def test_three_binary_third_coordinate_decides(exclusion, u, v, expected):
    model = PartialMonotonicityModel(**THREE_BINARY, exclusion=exclusion)
    G = model.build_graph()
    assert G.has_edge(u, v) is expected
    assert G.has_edge(v, u) is expected
    assert model.violates_partial_monotonicity(u, v) is (not expected)
    assert model.violates_partial_monotonicity(v, u) is (not expected)


def test_three_binary_exclusion_on_incomparable_pair():
    u, v = (0, 1, (0, 0, 1)), (1, 1, (1, 1, 0))  # incomparable, equal D, unequal Y
    assert not PartialMonotonicityModel(**THREE_BINARY, exclusion=True).build_graph().has_edge(u, v)
    assert PartialMonotonicityModel(**THREE_BINARY, exclusion=False).build_graph().has_edge(u, v)


# --------------------------------------------------------------------------- (6) invariance

@pytest.mark.parametrize("exclusion", [True, False])
def test_reordering_values_within_supports_leaves_graph_unchanged(exclusion):
    reference = PartialMonotonicityModel(**NONBINARY, exclusion=exclusion)
    permuted = PartialMonotonicityModel(
        y_support=[2, 0, 1],
        d_support=(5, 0, 2),
        z_supports=[[2, 0, 1], (20, 10)],
        exclusion=exclusion,
    )
    assert permuted.y_support == reference.y_support == (0, 1, 2)
    assert permuted.d_support == reference.d_support == (0, 2, 5)
    assert permuted.z_supports == reference.z_supports == ((0, 1, 2), (10, 20))
    assert permuted.z_support == reference.z_support
    assert permuted.nodes == reference.nodes
    assert edge_set(permuted.build_graph()) == edge_set(reference.build_graph())


def test_numpy_arrays_and_ranges_accepted_and_converted():
    model = PartialMonotonicityModel(
        np.array([1, 0]), range(2), (np.array([0.5, 0.0]), [np.int64(3), np.int64(1)])
    )
    assert model.y_support == (0, 1)
    assert model.d_support == (0, 1)
    assert model.z_supports == ((0.0, 0.5), (1, 3))
    assert all(type(v) in (int, float) for support in model.z_supports for v in support)
    # Two binary coordinates with binary Y and D: the values play no role beyond their
    # order, so this is the 52-edge graph of the two-binary-instrument example.
    assert model.build_graph().number_of_edges() == 52


def test_componentwise_comparison_rejects_mismatched_lengths():
    model = PartialMonotonicityModel(**TWO_BINARY)
    with pytest.raises(ValueError, match="same number of coordinates"):
        model.violates_partial_monotonicity((0, 1, (0, 0)), (0, 0, (1,)))
    with pytest.raises(ValueError, match="same number of coordinates"):
        PartialMonotonicityModel._leq((0, 0), (0, 0, 1))


@pytest.mark.parametrize("exclusion", [True, False])
def test_swapping_coordinates_relabels_nodes(exclusion):
    """Coordinate order is preserved, so swapping the two coordinate supports gives
    the same labelled graph after swapping the entries of every instrument vector."""
    original = PartialMonotonicityModel((0, 1), (0, 2, 5), ((0, 1, 2), (10, 20)), exclusion=exclusion)
    swapped = PartialMonotonicityModel((0, 1), (0, 2, 5), ((10, 20), (0, 1, 2)), exclusion=exclusion)
    assert swapped.z_supports == ((10, 20), (0, 1, 2))
    relabel = lambda node: (node[0], node[1], (node[2][1], node[2][0]))
    assert set(map(relabel, swapped.nodes)) == set(original.nodes)
    relabelled = {frozenset(map(relabel, edge)) for edge in edge_set(swapped.build_graph())}
    assert relabelled == edge_set(original.build_graph())


# --------------------------------------------------------------------------- (7) validation and interface

@pytest.mark.parametrize("bad, exc", [
    ((0, 1), TypeError),            # flat sequence of numbers: coordinates must be iterables
    ("01", TypeError),              # string
    (5, TypeError),                 # not iterable
    ((), ValueError),               # no coordinates
    (((), ), ValueError),           # empty coordinate support
    (((0, 1), (1, 1)), ValueError), # duplicate inside a coordinate
    (((0, 1), "ab"), TypeError),    # string coordinate
    ((("a", "b"),), TypeError),     # non-numeric value
    (((0, float("nan")),), ValueError),
    (((0, float("inf")),), ValueError),
    (((0, True),), TypeError),      # booleans are not numbers
])
def test_invalid_z_supports(bad, exc):
    with pytest.raises(exc):
        PartialMonotonicityModel((0, 1), (0, 1), bad)


def test_invalid_z_supports_messages_name_the_coordinate():
    with pytest.raises(ValueError, match=r"z_supports\[1\]"):
        PartialMonotonicityModel((0, 1), (0, 1), ((0, 1), (2, 2)))
    with pytest.raises(TypeError, match=r"z_supports\[0\].*\(\(0, 1\),\)"):
        PartialMonotonicityModel((0, 1), (0, 1), (0, 1))


@pytest.mark.parametrize("name, bad, exc", [
    ("y_support", (), ValueError),
    ("y_support", (1, 1), ValueError),
    ("y_support", "01", TypeError),
    ("d_support", (0, float("nan")), ValueError),
    ("d_support", 3, TypeError),
])
def test_invalid_outcome_and_treatment_supports(name, bad, exc):
    kwargs = dict(TWO_BINARY)
    kwargs[name] = bad
    with pytest.raises(exc, match=name):
        PartialMonotonicityModel(**kwargs)


@pytest.mark.parametrize("bad_flag", [0, 1, "True", None, np.bool_(True)])
def test_non_boolean_exclusion_rejected(bad_flag):
    with pytest.raises(TypeError, match="exclusion"):
        PartialMonotonicityModel(**TWO_BINARY, exclusion=bad_flag)


def test_exclusion_defaults_to_true():
    model = PartialMonotonicityModel(**TWO_BINARY)
    assert model.exclusion is True
    assert edge_set(model.build_graph()) == TWO_BINARY_EDGES[True]


def test_positional_arguments_in_listed_order():
    model = PartialMonotonicityModel((0, 1), (0, 1), ((0, 1), (0, 1)), False)
    assert model.exclusion is False
    assert model.build_graph().number_of_edges() == 76


def test_settings_are_fixed_after_construction():
    model = PartialMonotonicityModel(**TWO_BINARY)
    for name, value in [("exclusion", False), ("z_supports", ((0,),)), ("nodes", ()), ("new", 1)]:
        with pytest.raises(AttributeError, match="fixed after construction"):
            setattr(model, name, value)
    with pytest.raises(AttributeError):
        del model.exclusion
    assert model.exclusion is True


@pytest.mark.parametrize("exclusion", [True, False])
def test_repr_round_trips(exclusion):
    model = PartialMonotonicityModel(**NONBINARY, exclusion=exclusion)
    rebuilt = eval(repr(model), {"PartialMonotonicityModel": PartialMonotonicityModel})
    assert rebuilt.nodes == model.nodes
    assert rebuilt.exclusion == model.exclusion
    assert repr(rebuilt) == repr(model)
    assert "z_supports=((0, 1, 2), (10, 20))" in repr(model)


@pytest.mark.parametrize("exclusion", [True, False])
def test_predicates_return_bool(exclusion):
    model = PartialMonotonicityModel(**TWO_BINARY, exclusion=exclusion)
    for u, v in combinations(model.nodes, 2):
        for fn in (model.violates_exclusion, model.violates_partial_monotonicity, model.violate_pairwise_fn):
            assert type(fn(u, v)) is bool


def test_build_graph_returns_a_new_graph_each_time():
    model = PartialMonotonicityModel(**TWO_BINARY)
    G1, G2 = model.build_graph(), model.build_graph()
    assert G1 is not G2
    assert isinstance(G1, nx.Graph)
    G1.remove_node((0, 0, (0, 0)))
    assert G2.number_of_nodes() == 16


def test_group_fn_returns_instrument_vector():
    model = PartialMonotonicityModel(**NONBINARY)
    assert all(model.group_fn(node) == node[2] for node in model.nodes)
    assert set(map(model.group_fn, model.nodes)) == set(Z_NONBINARY)


def test_summary_describes_the_specification():
    text = PartialMonotonicityModel(**NONBINARY, exclusion=True).summary()
    assert "Outcome support Y:    (0, 1, 2)" in text
    assert "Treatment support D:  (0, 2, 5)" in text
    assert "Z1: (0, 1, 2)" in text and "Z2: (10, 20)" in text
    assert "6 vectors" in text and "54" in text
    assert "Partial monotonicity: imposed (always)" in text
    assert "Exclusion: imposed." in text
    assert "No outcome monotonicity" in text
    assert "independent of Z" in text
    without = PartialMonotonicityModel(**NONBINARY, exclusion=False).summary()
    assert "Exclusion: not imposed." in without
    one = PartialMonotonicityModel((0, 1), (0, 1), ((0, 1),)).summary()
    assert "Instrument coordinates: 1, Z = (Z1,)" in one


def test_module_does_not_import_graph_analyzer():
    """The class only builds the graph; the analyzer is imported separately by users."""
    tree = ast.parse(inspect.getsource(pm_module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("graph_analysis_utils" in name or "GraphAnalyzer" in name for name in imported)
    assert "networkx" in imported
    assert "support_validation" in imported
    assert not hasattr(pm_module, "GraphAnalyzer")


def test_shared_validator_is_used_by_both_classes():
    import support_validation
    assert IVModel._validate_support((2, 1), "x") == support_validation.validate_support((2, 1), "x") == (1, 2)
    with pytest.raises(ValueError, match="x must not be empty"):
        support_validation.validate_support([], "x")


@pytest.mark.parametrize("exclusion", [True, False])
def test_single_instrument_vector_gives_no_edges(exclusion):
    model = PartialMonotonicityModel((0, 1), (0, 1), ((3,), (7,)), exclusion=exclusion)
    assert model.z_support == ((3, 7),)
    G = model.build_graph()
    assert G.number_of_nodes() == 4 and G.number_of_edges() == 0


def test_single_valued_outcome_and_treatment():
    model = PartialMonotonicityModel((1,), (0,), ((0, 1), (0, 1)))
    G = model.build_graph()
    assert G.number_of_nodes() == 4 and G.number_of_edges() == 6  # every cross pair


def test_componentwise_order_is_not_lexicographic():
    """(0, 20) < (1, 10) lexicographically, yet the vectors are incomparable, so a
    falling treatment between them is allowed."""
    model = PartialMonotonicityModel(**NONBINARY)
    assert (0, 20) < (1, 10)
    assert model.violates_partial_monotonicity((0, 5, (0, 20)), (0, 0, (1, 10))) is False
    assert model.violates_partial_monotonicity((0, 0, (0, 20)), (0, 5, (1, 10))) is False
    assert model.build_graph().has_edge((0, 5, (0, 20)), (0, 0, (1, 10)))
