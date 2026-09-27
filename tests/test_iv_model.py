"""Tests for ``iv_model.IVModel``.

The first four groups cover the binary supports, the edge table, larger supports, and
input validation; the groups after them are additional checks. Every expected edge is either written out by
hand, with the mathematical reason in the docstring or a comment, or derived by
enumerating potential-response vectors (Method 2 of Kaido and Ponomarev, 2025). None is
produced by the violation functions under test.

Node convention: ``(y, d, z)``. With binary supports each instrument group holds four
nodes, so there are 16 cross-group pairs; same-instrument pairs are never edges.
"""

import ast
import inspect
from itertools import combinations, product

import networkx as nx
import pytest

import iv_model as iv_model_module
from iv_model import IVModel

# --------------------------------------------------------------------------- helpers

BINARY = dict(y_support=(0, 1), d_support=(0, 1), z_support=(0, 1))
LARGER = dict(y_support=(0, 1), d_support=(0, 2, 5), z_support=(-1, 3, 10))
THREE_Z = dict(y_support=(0, 1), d_support=(0, 1), z_support=(0, 1, 2))

SPECS = {
    "exclusion": dict(exclusion=True, d_monotonicity=False),
    "monotonicity": dict(exclusion=False, d_monotonicity=True),
    "both": dict(exclusion=True, d_monotonicity=True),
}


def edge_set(G):
    """Edges of ``G`` as unordered pairs, independent of insertion orientation."""
    return {frozenset(edge) for edge in G.edges()}


def as_edges(pairs):
    return {frozenset(pair) for pair in pairs}


def method2_edges(y_support, d_support, z_support, exclusion, d_monotonicity):
    """Edges from enumerating every potential-response vector (Method 2).

    A vector consists of D(z) for each z and Y(d, z) for each (d, z); under exclusion
    Y(d, z) is the same for all z. A vector generates the node (Y(D(z), z), D(z), z)
    at every z, and every pair of nodes it generates is an edge. This construction
    never calls the pairwise rules of ``IVModel``.
    """
    Z, D, Y = tuple(sorted(z_support)), tuple(sorted(d_support)), tuple(sorted(y_support))
    edges = set()
    for dvec in product(D, repeat=len(Z)):  # dvec[k] = D(Z[k]); Z is sorted
        if d_monotonicity and any(
            dvec[i] > dvec[j] for i in range(len(Z)) for j in range(i + 1, len(Z))
        ):
            continue
        if exclusion:
            outcome_maps = (
                {(d, z): ymap[i] for i, d in enumerate(D) for z in Z}
                for ymap in product(Y, repeat=len(D))
            )
        else:
            cells = [(d, z) for d in D for z in Z]
            outcome_maps = (dict(zip(cells, yvec)) for yvec in product(Y, repeat=len(cells)))
        for ymap in outcome_maps:
            generated = [(ymap[(dvec[k], z)], dvec[k], z) for k, z in enumerate(Z)]
            for u, v in combinations(generated, 2):
                edges.add(frozenset((u, v)))
    return edges


# --------------------------------------------------------------------------- binary supports

BINARY_NODES = (
    (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
    (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1),
)

# Exclusion only. Of the 16 cross-group pairs, the four with equal treatment and unequal
# outcome are removed: both events would fix Y(d) to two different values. The other 12
# are edges: same d with the same y is consistent with one value of Y(d), and different
# d places no restriction because Y(0) and Y(1) are free and D(0), D(1) are free.
BINARY_EDGES_EXCLUSION = as_edges([
    ((0, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 0, 1)),   # d = 0 at both z, same y
    ((0, 1, 0), (0, 1, 1)), ((1, 1, 0), (1, 1, 1)),   # d = 1 at both z, same y
    ((0, 0, 0), (0, 1, 1)), ((0, 0, 0), (1, 1, 1)),   # d = 0 at z = 0, d = 1 at z = 1
    ((1, 0, 0), (0, 1, 1)), ((1, 0, 0), (1, 1, 1)),
    ((0, 1, 0), (0, 0, 1)), ((0, 1, 0), (1, 0, 1)),   # d = 1 at z = 0, d = 0 at z = 1
    ((1, 1, 0), (0, 0, 1)), ((1, 1, 0), (1, 0, 1)),
])

# D-monotonicity only. The four pairs with treatment 1 at z = 0 and treatment 0 at
# z = 1 are removed: D(0) = 1 > 0 = D(1) contradicts D(0) <= D(1). Outcomes are
# unrestricted, so every other cross-group pair is an edge.
BINARY_EDGES_MONOTONICITY = as_edges([
    ((0, 0, 0), (0, 0, 1)), ((0, 0, 0), (1, 0, 1)),   # D(0) = 0, D(1) = 0
    ((1, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 0, 1)),
    ((0, 0, 0), (0, 1, 1)), ((0, 0, 0), (1, 1, 1)),   # D(0) = 0, D(1) = 1 (complier)
    ((1, 0, 0), (0, 1, 1)), ((1, 0, 0), (1, 1, 1)),
    ((0, 1, 0), (0, 1, 1)), ((0, 1, 0), (1, 1, 1)),   # D(0) = 1, D(1) = 1
    ((1, 1, 0), (0, 1, 1)), ((1, 1, 0), (1, 1, 1)),
])

# Both. The two removed sets are disjoint (one needs equal treatments, the other
# needs treatment 1 at z = 0 and 0 at z = 1), so 16 - 4 - 4 = 8 edges remain.
BINARY_EDGES_BOTH = as_edges([
    ((0, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 0, 1)),   # never-taker types, Y(0) fixed
    ((0, 1, 0), (0, 1, 1)), ((1, 1, 0), (1, 1, 1)),   # always-taker types, Y(1) fixed
    ((0, 0, 0), (0, 1, 1)), ((0, 0, 0), (1, 1, 1)),   # complier types, Y(0), Y(1) free
    ((1, 0, 0), (0, 1, 1)), ((1, 0, 0), (1, 1, 1)),
])

BINARY_EDGES = {
    "exclusion": BINARY_EDGES_EXCLUSION,
    "monotonicity": BINARY_EDGES_MONOTONICITY,
    "both": BINARY_EDGES_BOTH,
}
BINARY_EDGE_COUNTS = {"exclusion": 12, "monotonicity": 12, "both": 8}

assert {k: len(v) for k, v in BINARY_EDGES.items()} == BINARY_EDGE_COUNTS


@pytest.mark.parametrize("spec", list(SPECS))
def test_binary_node_and_edge_counts(spec):
    """8 nodes = 2 * 2 * 2; edges 12 / 12 / 8 as derived above."""
    model = IVModel(**BINARY, **SPECS[spec])
    G = model.build_graph()
    assert G.number_of_nodes() == 8
    assert G.number_of_edges() == BINARY_EDGE_COUNTS[spec]
    assert set(G.nodes) == set(model.nodes)


@pytest.mark.parametrize("spec", list(SPECS))
def test_binary_node_labels(spec):
    """The nodes are exactly the eight (y, d, z) tuples, in y-outermost order."""
    model = IVModel(**BINARY, **SPECS[spec])
    G = model.build_graph()
    assert model.nodes == BINARY_NODES
    assert tuple(G.nodes) == BINARY_NODES
    assert set(G.nodes) == set(BINARY_NODES)


@pytest.mark.parametrize("spec", list(SPECS))
def test_binary_edge_sets(spec):
    """Counts alone do not establish that the right edges were built: compare the
    full edge set with the hand-derived lists."""
    G = IVModel(**BINARY, **SPECS[spec]).build_graph()
    assert edge_set(G) == BINARY_EDGES[spec]


def test_binary_both_counts():
    """Node and edge counts for the binary model with both assumptions."""
    model = IVModel(
        y_support=(0, 1), d_support=(0, 1), z_support=(0, 1),
        exclusion=True, d_monotonicity=True,
    )
    G = model.build_graph()
    assert G.number_of_nodes() == 8
    assert G.number_of_edges() == 8
    assert set(G.nodes) == set(model.nodes)


def test_binary_omitting_both_flags_gives_exclusion_only():
    """The default is exclusion=True, d_monotonicity=False, hence the 12 exclusion edges."""
    model = IVModel(**BINARY)
    assert model.exclusion is True
    assert model.d_monotonicity is False
    G = model.build_graph()
    assert G.number_of_edges() == 12
    assert edge_set(G) == BINARY_EDGES_EXCLUSION


def test_binary_no_same_instrument_edges():
    """D = D(Z) takes one value at a given Z, so two nodes with the same z never share a type."""
    for spec in SPECS:
        G = IVModel(**BINARY, **SPECS[spec]).build_graph()
        for u, v in G.edges():
            assert u[2] != v[2]


# --------------------------------------------------------------------------- edge table

# (u, v, edge under exclusion only, under D-monotonicity only, under both)
EDGE_TABLE = [
    # same treatment, different outcome: exclusion violated; D(0) = 0 <= 0 = D(1) is fine
    ((0, 0, 0), (1, 0, 1), False, True, False),
    # treatment falls from z = 0 to z = 1: monotonicity violated; treatments differ, so
    # exclusion says nothing
    ((0, 1, 0), (0, 0, 1), True, False, False),
    # treatment rises and the outcome falls: outcome monotonicity is not imposed
    ((1, 0, 0), (0, 1, 1), True, True, True),
    # same treatment and same outcome: Y(1) = 1 and D(0) = D(1) = 1 satisfy everything
    ((1, 1, 0), (1, 1, 1), True, True, True),
    # same instrument value: never adjacent, handled by build_graph
    ((0, 0, 0), (1, 1, 0), False, False, False),
]

# (u, v, violates_exclusion, violates_d_monotonicity) for the cross-group rows
HELPERS_62 = [
    ((0, 0, 0), (1, 0, 1), True, False),
    ((0, 1, 0), (0, 0, 1), False, True),
    ((1, 0, 0), (0, 1, 1), False, False),
    ((1, 1, 0), (1, 1, 1), False, False),
]


@pytest.mark.parametrize("u, v, excl, mono, both", EDGE_TABLE)
def test_edge_table(u, v, excl, mono, both):
    """``G.has_edge(u, v)`` against the hand-derived table, for all three specifications."""
    expected = {"exclusion": excl, "monotonicity": mono, "both": both}
    for spec, kwargs in SPECS.items():
        G = IVModel(**BINARY, **kwargs).build_graph()
        assert G.has_edge(u, v) is expected[spec], (spec, u, v)
        assert G.has_edge(v, u) is expected[spec], (spec, v, u)


@pytest.mark.parametrize("u, v, excl_violated, mono_violated", HELPERS_62)
@pytest.mark.parametrize("spec", list(SPECS))
def test_helper_predicates_regardless_of_flags(spec, u, v, excl_violated, mono_violated):
    """Each helper evaluates its own rule whatever the model's flags say."""
    model = IVModel(**BINARY, **SPECS[spec])
    assert model.violates_exclusion(u, v) is excl_violated
    assert model.violates_exclusion(v, u) is excl_violated
    assert model.violates_d_monotonicity(u, v) is mono_violated
    assert model.violates_d_monotonicity(v, u) is mono_violated


@pytest.mark.parametrize("u, v, excl, mono, both", EDGE_TABLE[:4])
def test_combined_predicate_is_opposite_of_edge(u, v, excl, mono, both):
    """For cross-group pairs the combined predicate is the negation of the edge entry."""
    expected = {"exclusion": excl, "monotonicity": mono, "both": both}
    for spec, kwargs in SPECS.items():
        model = IVModel(**BINARY, **kwargs)
        assert model.violate_pairwise_fn(u, v) is (not expected[spec]), (spec, u, v)
        assert model.violate_pairwise_fn(v, u) is (not expected[spec]), (spec, v, u)


# --------------------------------------------------------------------------- larger supports


@pytest.mark.parametrize("spec", list(SPECS))
def test_larger_supports_node_count(spec):
    """18 nodes = 2 outcomes * 3 treatments * 3 instrument values."""
    model = IVModel(**LARGER, **SPECS[spec])
    G = model.build_graph()
    assert len(model.nodes) == 18
    assert G.number_of_nodes() == 18
    assert set(G.nodes) == set(product((0, 1), (0, 2, 5), (-1, 3, 10)))


def test_larger_supports_examples_under_both():
    """Treatment rising from 0 to 5 with different outcomes is
    allowed (different treatments, so exclusion is silent; 0 <= 5 satisfies
    monotonicity); treatment falling from 5 to 2 as z rises violates monotonicity;
    equal treatment 2 with outcomes 0 and 1 violates exclusion."""
    G = IVModel(**LARGER, **SPECS["both"]).build_graph()
    assert G.has_edge((1, 0, -1), (0, 5, 10))
    assert not G.has_edge((0, 5, 3), (0, 2, 10))
    assert not G.has_edge((0, 2, -1), (1, 2, 10))


@pytest.mark.parametrize("spec", list(SPECS))
def test_input_order_does_not_matter(spec):
    """Supports are sorted numerically, so a permuted input gives the same model."""
    reference = IVModel(**LARGER, **SPECS[spec])
    permuted = IVModel(
        y_support=(0, 1), d_support=(5, 0, 2), z_support=(10, -1, 3), **SPECS[spec]
    )
    assert permuted.y_support == reference.y_support == (0, 1)
    assert permuted.d_support == reference.d_support == (0, 2, 5)
    assert permuted.z_support == reference.z_support == (-1, 3, 10)
    assert permuted.nodes == reference.nodes
    assert edge_set(permuted.build_graph()) == edge_set(reference.build_graph())


def test_input_order_lists_and_all_three_permuted():
    """Lists are accepted, and permuting every support gives the same model."""
    reference = IVModel(**LARGER, **SPECS["both"])
    permuted = IVModel(
        y_support=[1, 0], d_support=[2, 5, 0], z_support=[3, 10, -1], **SPECS["both"]
    )
    assert (permuted.y_support, permuted.d_support, permuted.z_support) == (
        (0, 1), (0, 2, 5), (-1, 3, 10)
    )
    assert permuted.nodes == reference.nodes
    assert edge_set(permuted.build_graph()) == edge_set(reference.build_graph())


# --------------------------------------------------------------------------- input validation

SUPPORT_NAMES = ["y_support", "d_support", "z_support"]

INVALID_SUPPORTS = [
    ((), ValueError, "empty"),
    ((0, 1, 1), ValueError, "duplicate"),
    ((0, 1.0, 1), ValueError, "duplicate by numeric equality"),
    ("01", TypeError, "string"),
    ((0, "1"), TypeError, "string element"),
    ((0, float("nan")), ValueError, "nan"),
    ((0, float("inf")), ValueError, "inf"),
    ((0, float("-inf")), ValueError, "-inf"),
]


@pytest.mark.parametrize("name", SUPPORT_NAMES)
@pytest.mark.parametrize("bad, exc, label", INVALID_SUPPORTS)
def test_invalid_supports(name, bad, exc, label):
    """Each invalid support is rejected with the documented exception type, whichever of
    the three supports it is supplied for, and the message names that support."""
    kwargs = dict(BINARY)
    kwargs[name] = bad
    with pytest.raises(exc) as info:
        IVModel(**kwargs)
    assert name in str(info.value)


@pytest.mark.parametrize("flag_name", ["exclusion", "d_monotonicity"])
@pytest.mark.parametrize("bad_flag", [1, 0, "True", None, 1.0, [True]])
def test_non_boolean_flags(flag_name, bad_flag):
    """Flags must be actual booleans."""
    kwargs = dict(BINARY, exclusion=True, d_monotonicity=True)
    kwargs[flag_name] = bad_flag
    with pytest.raises(TypeError) as info:
        IVModel(**kwargs)
    assert flag_name in str(info.value)


def test_both_flags_false_rejected():
    """A model with no restriction is not one of the three specifications."""
    with pytest.raises(ValueError):
        IVModel(**BINARY, exclusion=False, d_monotonicity=False)


# --------------------------------------------------------------------------- additional


@pytest.mark.parametrize("spec", list(SPECS))
@pytest.mark.parametrize("supports", [BINARY, THREE_Z, LARGER], ids=["binary", "three_z", "larger"])
def test_method1_agrees_with_method2(supports, spec):
    """Method 1 (pairwise rules) and Method 2 (enumerating all potential-response
    vectors) give the same graph. This is the content of the "why or is correct"
    question: a pair rejected by no enabled rule is generated by some full vector."""
    G = IVModel(**supports, **SPECS[spec]).build_graph()
    assert edge_set(G) == method2_edges(**supports, **SPECS[spec])


def test_three_valued_instrument_counts_match_paper_table_4():
    """Kaido and Ponomarev (2025, Table 4): with |Y| = |D| = 2 and |Z| = 3 the
    exclusion-only graph has 12 vertices and 36 edges; each of the 3 * 4 * 4 = 48
    cross pairs is removed exactly when treatments agree and outcomes differ, which
    happens for 12 of them."""
    G = IVModel(**THREE_Z, **SPECS["exclusion"]).build_graph()
    assert G.number_of_nodes() == 12
    assert G.number_of_edges() == 36


@pytest.mark.parametrize("spec", list(SPECS))
def test_predicates_return_bool(spec):
    model = IVModel(**BINARY, **SPECS[spec])
    for u, v in combinations(model.nodes, 2):
        for fn in (model.violates_exclusion, model.violates_d_monotonicity, model.violate_pairwise_fn):
            result = fn(u, v)
            assert isinstance(result, bool), (fn.__name__, u, v, result)


def test_build_graph_returns_a_new_graph_each_time():
    model = IVModel(**BINARY)
    G1, G2 = model.build_graph(), model.build_graph()
    assert G1 is not G2
    assert isinstance(G1, nx.Graph) and not isinstance(G1, nx.DiGraph)
    assert nx.utils.graphs_equal(G1, G2)
    G1.remove_edges_from(list(G1.edges()))
    assert model.build_graph().number_of_edges() == 12


def test_settings_are_fixed_after_construction():
    """The model is fixed after construction; assignment or deletion
    of an attribute raises, and the stored supports and nodes are tuples."""
    model = IVModel(**BINARY)
    for name, value in [
        ("exclusion", False), ("d_monotonicity", True), ("y_support", (0, 1, 2)),
        ("nodes", ()), ("new_attribute", 1),
    ]:
        with pytest.raises(AttributeError):
            setattr(model, name, value)
    with pytest.raises(AttributeError):
        del model.exclusion
    assert model.exclusion is True and model.d_monotonicity is False
    assert isinstance(model.y_support, tuple) and isinstance(model.nodes, tuple)
    assert model.build_graph().number_of_edges() == 12


def test_summary_describes_the_specification():
    for spec, kwargs in SPECS.items():
        text = IVModel(**LARGER, **kwargs).summary()
        assert isinstance(text, str)
        for value in ["(0, 1)", "(0, 2, 5)", "(-1, 3, 10)", "18"]:
            assert value in text
        assert "independent of Z" in text
        assert ("Exclusion: imposed" in text) is kwargs["exclusion"]
        assert ("D-monotonicity: imposed" in text) is kwargs["d_monotonicity"]
        assert "outcome monotonicity" in text.lower()
        if not kwargs["exclusion"]:
            assert "Y(d)" not in text  # no Y(d) shorthand without exclusion


def test_group_fn_returns_instrument_value():
    model = IVModel(**LARGER)
    for node in model.nodes:
        assert model.group_fn(node) == node[2]
    assert sorted({model.group_fn(n) for n in model.nodes}) == [-1, 3, 10]


def test_module_does_not_import_graph_analyzer():
    """The class only builds the graph; the analyzer is imported separately by users."""
    tree = ast.parse(inspect.getsource(iv_model_module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("graph_analysis_utils" in name or "GraphAnalyzer" in name for name in imported)
    assert "networkx" in imported
    assert not hasattr(iv_model_module, "GraphAnalyzer")


@pytest.mark.parametrize("spec", list(SPECS))
def test_single_valued_instrument_gives_no_edges(spec):
    """One instrument value means one part: no cross-group pairs, hence no edges."""
    model = IVModel(y_support=(0, 1), d_support=(0, 1), z_support=(7,), **SPECS[spec])
    G = model.build_graph()
    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 0


def test_single_valued_outcome_and_treatment():
    """With one outcome and one treatment value nothing can be violated, so the two
    nodes (one per instrument value) are joined."""
    G = IVModel(y_support=(3,), d_support=(1,), z_support=(0, 1), **SPECS["both"]).build_graph()
    assert set(G.nodes) == {(3, 1, 0), (3, 1, 1)}
    assert edge_set(G) == as_edges([((3, 1, 0), (3, 1, 1))])


def test_positional_arguments_in_listed_order():
    model = IVModel((0, 1), (0, 1), (0, 1), False, True)
    assert model.exclusion is False and model.d_monotonicity is True
    assert model.build_graph().number_of_edges() == 12


def test_repr_round_trips():
    model = IVModel(**LARGER, **SPECS["both"])
    rebuilt = eval(repr(model), {"IVModel": IVModel})
    assert rebuilt.nodes == model.nodes
    assert (rebuilt.exclusion, rebuilt.d_monotonicity) == (True, True)


@pytest.mark.parametrize("name", SUPPORT_NAMES)
@pytest.mark.parametrize("bad", [(0, True), (False, 1), (0, 1, 2 + 0j), 5, None])
def test_bool_complex_and_non_iterable_supports_rejected(name, bad):
    """Booleans and complex numbers are not support values; a scalar is not a support."""
    kwargs = dict(BINARY)
    kwargs[name] = bad
    with pytest.raises(TypeError):
        IVModel(**kwargs)


def test_decimal_rejected_fraction_accepted():
    from decimal import Decimal
    from fractions import Fraction

    with pytest.raises(TypeError):
        IVModel(y_support=(Decimal("0"), Decimal("1")), d_support=(0, 1), z_support=(0, 1))
    model = IVModel(y_support=(Fraction(1, 2), 0), d_support=(0, 1), z_support=(0, 1))
    assert model.y_support == (0, Fraction(1, 2))


def test_range_and_float_supports():
    model = IVModel(y_support=range(2), d_support=[0.5, 0.0], z_support=(1, -1))
    assert model.y_support == (0, 1)
    assert model.d_support == (0.0, 0.5)
    assert model.z_support == (-1, 1)
    assert model.nodes[0] == (0, 0.0, -1)


def test_numpy_scalars_are_converted_to_python_numbers():
    np = pytest.importorskip("numpy")
    model = IVModel(
        y_support=np.array([1, 0]), d_support=np.array([0.0, 2.5]), z_support=(np.int64(3), 1)
    )
    assert model.y_support == (0, 1)
    assert all(type(v) is int for v in model.y_support)
    assert model.d_support == (0.0, 2.5)
    assert all(type(v) is float for v in model.d_support)
    assert model.z_support == (1, 3)
    assert all(type(v) is int for v in model.z_support)
    with pytest.raises(TypeError):
        IVModel(**dict(BINARY, y_support=(np.bool_(True), 0)))
    with pytest.raises(TypeError):
        IVModel(**BINARY, exclusion=np.bool_(True))
