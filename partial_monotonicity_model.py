"""PartialMonotonicityModel: a discrete IV model with a vector instrument and partial
treatment monotonicity, for the potential-response graph framework of Kaido and
Ponomarev (2025), "Testing Exclusion and Shape Restrictions in Potential Outcomes
Models" (arXiv:2512.20851).

The model
---------
The researcher observes an outcome Y and a treatment D, each with a finite support of
numeric values, and an instrument vector Z = (Z_1, ..., Z_m) with m >= 1 coordinates.
Each coordinate Z_j has its own finite support, and the support of Z is the Cartesian
product of the coordinate supports. Potential treatments D(z) and potential outcomes
Y(d, z) generate the observed responses through

    D = D(Z),    Y = Y(D(Z), Z).

Maintained assumption (documented here; it is not a pairwise rule): the full vector of
potential responses ((Y(d, z))_{d, z}, (D(z))_z) is independent of Z.

Restrictions:

* Partial monotonicity (always imposed). D(z) is nondecreasing in each coordinate of z,
  holding the other coordinates fixed (Mogstad, Torgovitsky and Walters, 2021). On the
  Cartesian-product support this is the same as

      D(z) <= D(z')    whenever z[j] <= z'[j] for every coordinate j,

  because any componentwise increase can be reached by raising one coordinate at a
  time. Instrument vectors that are incomparable in the componentwise order impose no
  ordering on their treatment values. No outcome monotonicity is imposed.
* Exclusion (optional, default True). Y(d, z) = Y(d, z') for every treatment value d
  and every pair of instrument vectors z, z'. When it is off, outcomes may vary freely
  across instrument vectors and the shorthand Y(d) is never used.

The potential response graph (Method 1)
---------------------------------------
Each node is an observable event (y, d, z), where z is a tuple with one entry per
instrument coordinate, even when m = 1. Two nodes with different instrument vectors are
joined by an edge exactly when some potential-response vector satisfying the imposed
restrictions generates both events; nodes with the same instrument vector are never
joined, because D = D(Z) takes a single value at a given Z. Method 1 of Kaido and
Ponomarev (2025, Section 5.1) decides each edge from the pair alone:

* partial monotonicity forbids the pair u = (y, d, z), v = (y', d', z') when z <= z'
  componentwise but d > d', or z' <= z componentwise but d' > d;
* exclusion forbids the pair when the treatments agree but the outcomes differ,
  since the two events would fix Y(d) to two different values.

Why the pairwise rules are exact. Suppose neither enabled rule rejects u and v. Let
d_min be the smallest treatment value and define, on the whole instrument support,

    D(t) = max({d : z <= t} U {d' : z' <= t} U {d_min})     (componentwise <=).

The set inside the max grows with t in the componentwise order, so D(.) is
componentwise nondecreasing, and it takes the value d at z: the other value d' enters
the max at t = z only when z' <= z, in which case the partial-monotonicity rule gives
d' <= d. The same holds at z'. Under exclusion, set Y(d) = y and Y(d') = y', which is
consistent because the exclusion rule gives y = y' whenever d = d'. Without exclusion,
set Y(d, z) = y and Y(d', z') = y' on the two cells, which are distinct because
z != z'. The treatment and outcome completions act on disjoint components of the
potential-response vector, so they can be carried out together, and every other
component is free. This is why ``violate_pairwise_fn`` combines the rules with ``or``.

The same construction with D(t) = max({d_i : z_i <= t} U {d_min}) extends any set of
pairwise compatible nodes with distinct instrument vectors to a full vector (D(z_i) =
d_i because z_j <= z_i implies d_j <= d_i for compatible pairs), so the model satisfies
the pairwise incompatibility criterion (Condition 2 of the paper) and every maximal
clique of the graph is a support point. Perfectness is a separate matter and is checked
numerically by ``GraphAnalyzer``.

``build_graph`` returns a ``networkx.Graph``. Maximal independent sets, plots, and the
regularity checks are the job of ``GraphAnalyzer`` in ``graph_analysis_utils.py``,
which this module deliberately does not import. ``format_inequalities`` writes the
maximal independent sets found by the analyzer as the inequalities in conditional
probabilities that they imply, since the model knows which variable values each node
carries.
"""

import numbers
from itertools import combinations, product

import networkx as nx

from support_validation import validate_support

__all__ = ["PartialMonotonicityModel"]


class PartialMonotonicityModel:
    """A finite discrete IV model with a vector instrument, partial treatment
    monotonicity (always imposed) and optional exclusion.

    Parameters
    ----------
    y_support, d_support : iterable of real numbers
        Supports of the outcome and the treatment. Each must be a nonempty, non-string
        iterable of distinct finite real numbers. They are stored sorted.
    z_supports : iterable of iterables of real numbers
        One support per instrument coordinate, in coordinate order; at least one
        coordinate is required. Each coordinate support is validated like
        ``y_support`` and stored sorted. A flat sequence of numbers is rejected: for a
        single instrument write ``z_supports=((0, 1),)``.
    exclusion : bool, default True
        Impose Y(d, z) = Y(d, z') for all d and all instrument vectors z, z'.

    Settings are fixed after construction: create a new model to change supports or
    the exclusion flag.

    Attributes
    ----------
    y_support, d_support : tuple
        Validated, sorted supports.
    z_supports : tuple of tuples
        Validated, sorted coordinate supports, in the order supplied.
    z_support : tuple of tuples
        The instrument support: the Cartesian product of the coordinate supports, in
        product order (the first coordinate outermost).
    n_instruments : int
        The number of instrument coordinates, ``len(z_supports)``.
    exclusion : bool
        Whether exclusion is imposed. Partial monotonicity is always imposed.
    nodes : tuple of (y, d, z)
        All observable events, with y outermost, then d, then z in product order. The
        entry z is a tuple with one value per coordinate, even when there is one
        instrument.
    """

    _frozen = False

    def __init__(self, y_support, d_support, z_supports, exclusion=True):
        if not isinstance(exclusion, bool):
            raise TypeError(
                f"exclusion must be a bool (True or False), got {exclusion!r} of type "
                f"{type(exclusion).__name__}"
            )
        self.y_support = validate_support(y_support, "y_support")
        self.d_support = validate_support(d_support, "d_support")
        self.z_supports = self._validate_z_supports(z_supports)
        self.z_support = tuple(product(*self.z_supports))
        self.n_instruments = len(self.z_supports)
        self.exclusion = exclusion
        self.nodes = self._make_nodes()
        self._frozen = True

    # ------------------------------------------------------------------ immutability
    def __setattr__(self, name, value):
        if self._frozen:
            raise AttributeError(
                "PartialMonotonicityModel settings are fixed after construction; cannot "
                f"set {name!r}. Create a new PartialMonotonicityModel instead."
            )
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if self._frozen:
            raise AttributeError(
                "PartialMonotonicityModel settings are fixed after construction; cannot "
                f"delete {name!r}."
            )
        super().__delattr__(name)

    def __repr__(self):
        return (
            f"PartialMonotonicityModel(y_support={self.y_support!r}, "
            f"d_support={self.d_support!r}, z_supports={self.z_supports!r}, "
            f"exclusion={self.exclusion!r})"
        )

    # ------------------------------------------------------------------ validation
    @staticmethod
    def _validate_z_supports(z_supports):
        """Return ``z_supports`` as a tuple of sorted coordinate supports.

        Raises TypeError for a string or a non-iterable, for a flat sequence of numbers
        (each coordinate support must itself be an iterable of numbers), and for any
        element of a coordinate support that is not a real number. Raises ValueError
        when there are no coordinates or when a coordinate support is empty, has a
        duplicate, or contains nan or inf. Error messages name the coordinate as
        ``z_supports[j]``.
        """
        if isinstance(z_supports, (str, bytes, bytearray)):
            raise TypeError(
                f"z_supports must be a sequence of coordinate supports, not a string: "
                f"{z_supports!r}"
            )
        try:
            coordinates = list(z_supports)
        except TypeError:
            raise TypeError(
                f"z_supports must be an iterable of coordinate supports, got "
                f"{z_supports!r} of type {type(z_supports).__name__}"
            ) from None
        if not coordinates:
            raise ValueError("z_supports must contain at least one instrument coordinate")
        validated = []
        for j, coordinate in enumerate(coordinates):
            if isinstance(coordinate, (str, bytes, bytearray)) or not hasattr(
                coordinate, "__iter__"
            ):
                raise TypeError(
                    f"z_supports[{j}] must be an iterable of numbers (one support per "
                    f"instrument coordinate), got {coordinate!r} of type "
                    f"{type(coordinate).__name__}; for a single instrument write "
                    f"z_supports=((0, 1),)"
                )
            validated.append(validate_support(coordinate, f"z_supports[{j}]"))
        return tuple(validated)

    # ------------------------------------------------------------------ nodes and groups
    def _make_nodes(self):
        """All (y, d, z) tuples, y outermost, then d, then z in product order."""
        return tuple(
            (y, d, z) for y in self.y_support for d in self.d_support for z in self.z_support
        )

    def group_fn(self, node):
        """The instrument vector of a node; nodes with equal z form one part of the graph."""
        return node[2]

    # ------------------------------------------------------------------ pairwise rules
    @staticmethod
    def _leq(z, zp):
        """True if ``z[j] <= zp[j]`` for every coordinate j (componentwise order).

        This is the economic order on instrument vectors. Python's tuple comparison is
        lexicographic and is never used for that purpose. Vectors of different lengths
        are rejected rather than silently compared on their common prefix.
        """
        if len(z) != len(zp):
            raise ValueError(
                f"instrument vectors must have the same number of coordinates, got "
                f"{z!r} and {zp!r}"
            )
        return all(a <= b for a, b in zip(z, zp))

    def violates_exclusion(self, u, v):
        """True if u and v cannot both occur under exclusion: same d, different y.

        Evaluated regardless of the ``exclusion`` flag. Meant for nodes with different
        instrument vectors; for a same-z pair it simply evaluates the same formula.
        """
        y, d, z = u
        yp, dp, zp = v
        return d == dp and y != yp

    def violates_partial_monotonicity(self, u, v):
        """True if u and v cannot both occur under partial monotonicity.

        The pair is rejected when the instrument vectors are comparable componentwise
        and the treatment is larger at the smaller instrument vector. Incomparable
        instrument vectors never reject. Transitive implications need no extra code,
        because the componentwise order on a product support is already transitive.
        """
        y, d, z = u
        yp, dp, zp = v
        return bool((self._leq(z, zp) and d > dp) or (self._leq(zp, z) and dp > d))

    def violate_pairwise_fn(self, u, v):
        """True if an imposed assumption rules out observing both u and v.

        Partial monotonicity is always applied; exclusion only when enabled. The rules
        are combined with ``or`` (see the module docstring for why this is exact).
        """
        return self.violates_partial_monotonicity(u, v) or (
            self.exclusion and self.violates_exclusion(u, v)
        )

    # ------------------------------------------------------------------ graph
    def build_graph(self):
        """Construct the potential response graph by Method 1 and return it.

        Every node in ``self.nodes`` is added, including any node that ends up with no
        edges. Each unordered pair of distinct nodes is visited once; pairs with the
        same instrument vector are skipped, and an edge is added exactly when
        ``violate_pairwise_fn`` returns False. A new ``networkx.Graph`` is returned on
        every call.
        """
        G = nx.Graph()
        G.add_nodes_from(self.nodes)

        for u, v in combinations(self.nodes, 2):
            if self.group_fn(u) == self.group_fn(v):
                continue
            if self.violate_pairwise_fn(u, v):
                continue
            G.add_edge(u, v)

        return G

    # ------------------------------------------------------------------ inequalities
    def _check_independent_set(self, independent_set):
        """Return the nodes of ``independent_set`` as a list of this model's own node
        tuples, in the order given, after checking that they form an independent set
        of the potential response graph.

        Each element must be a tuple equal to a node in ``self.nodes``. Values are
        matched by numeric equality, so ``1.0``, ``numpy.int64(1)`` or ``True`` select
        the node whose value is ``1``, and the returned tuple carries the support value,
        which is what the inequality prints. Raises TypeError for a string, a
        non-iterable, an element that is not a tuple, or a tuple with an unhashable
        entry, and ValueError for an empty set, a tuple that is not a node of this
        model, a duplicate node, or two nodes with different instrument
        vectors that ``violate_pairwise_fn`` does not reject: such a pair is an edge of
        the graph, so some admissible potential-response vector generates both events
        and the sum of their probabilities is not bounded by one.
        """
        if isinstance(independent_set, (str, bytes, bytearray)):
            raise TypeError(
                f"an independent set must be an iterable of nodes (y, d, z), not a "
                f"string: {independent_set!r}"
            )
        try:
            nodes = list(independent_set)
        except TypeError:
            raise TypeError(
                f"an independent set must be an iterable of nodes (y, d, z), got "
                f"{independent_set!r} of type {type(independent_set).__name__}"
            ) from None
        if not nodes:
            raise ValueError("an independent set must contain at least one node")
        known = {node: node for node in self.nodes}
        canonical = []
        for node in nodes:
            if not isinstance(node, tuple):
                raise TypeError(
                    f"nodes must be hashable tuples (y, d, z), got {node!r} of type "
                    f"{type(node).__name__}"
                )
            try:
                canonical.append(known[node])
            except TypeError:
                raise TypeError(
                    f"nodes must be hashable tuples (y, d, z), got {node!r} with an "
                    f"unhashable entry"
                ) from None
            except KeyError:
                raise ValueError(
                    f"{node!r} is not a node (y, d, z) of this model"
                ) from None
        nodes = canonical
        if len(set(nodes)) != len(nodes):
            raise ValueError(
                f"an independent set must not contain duplicate nodes: {nodes!r}"
            )
        for u, v in combinations(nodes, 2):
            if self.group_fn(u) != self.group_fn(v) and not self.violate_pairwise_fn(u, v):
                raise ValueError(
                    f"{u!r} and {v!r} are adjacent in the potential response graph, so "
                    f"the set is not independent and its events are not mutually "
                    f"exclusive"
                )
        return nodes

    def _label_instrument(self, z):
        """The instrument vector ``z`` written coordinate by coordinate, ``"Z1=0, Z2=1"``,
        for the conditioning event of a probability."""
        return ", ".join(f"Z{j + 1}={z[j]}" for j in range(self.n_instruments))

    def format_inequality(self, independent_set):
        """Write one independent set of the potential response graph as the inequality
        it implies, for example
        ``"P(D=1 | Z1=0, Z2=0) + P(D=0 | Z1=1, Z2=1) <= 1"``.

        The events of an independent set are mutually exclusive under the model, so
        their conditional probabilities sum to at most one (Theorem 1 of Kaido and
        Ponomarev, 2025). Each node (y, d, z) contributes
        ``P(Y=y, D=d | Z1=z[0], ..., Zm=z[m-1])``, the instrument vector written
        coordinate by coordinate. When the set contains every outcome value for a
        treatment value d at an instrument vector z, those terms are written as
        ``P(D=d | Z1=z[0], ..., Zm=z[m-1])``. Terms are ordered by instrument vector
        (product order), then treatment value, then outcome value, each in support
        order. The set is validated by ``_check_independent_set``; a set that lies
        within a single instrument vector is accepted, but its inequality holds
        trivially.
        """
        nodes = self._check_independent_set(independent_set)
        outcomes = {}
        for y, d, z in nodes:
            outcomes.setdefault(z, {}).setdefault(d, set()).add(y)
        all_outcomes = set(self.y_support)
        terms = []
        for z in self.z_support:
            given = self._label_instrument(z)
            for d in self.d_support:
                ys = outcomes.get(z, {}).get(d)
                if ys is None:
                    continue
                if ys == all_outcomes:
                    terms.append(f"P(D={d} | {given})")
                else:
                    terms.extend(f"P(Y={y}, D={d} | {given})" for y in sorted(ys))
        return " + ".join(terms) + " <= 1"

    def format_inequalities(self, independent_sets):
        """Write each independent set in ``independent_sets`` as an inequality and
        return the strings as a list, in the order given.

        Meant for the maximal independent sets returned by
        ``GraphAnalyzer.find_mis_excluding_single_group()`` on the graph of this
        model, so that the testable implications of the model are listed as

            inequalities = model.format_inequalities(analyzer.find_mis_excluding_single_group())

        Each set is validated and formatted by ``format_inequality``. Passing a single
        independent set instead of a sequence of them raises TypeError: its first
        element is then a node, a tuple that starts with a number, rather than a
        collection of nodes.
        """
        if isinstance(independent_sets, (str, bytes, bytearray)):
            raise TypeError(
                f"independent_sets must be an iterable of independent sets, not a "
                f"string: {independent_sets!r}"
            )
        try:
            sets = list(independent_sets)
        except TypeError:
            raise TypeError(
                f"independent_sets must be an iterable of independent sets, got "
                f"{independent_sets!r} of type {type(independent_sets).__name__}"
            ) from None
        first = sets[0] if sets else None
        if isinstance(first, tuple) and first and isinstance(first[0], numbers.Real):
            raise TypeError(
                f"format_inequalities expects an iterable of independent sets but "
                f"received node-like tuples such as {first!r}; to format one "
                f"independent set call format_inequality"
            )
        return [self.format_inequality(independent_set) for independent_set in sets]

    # ------------------------------------------------------------------ description
    def summary(self):
        """A readable description of the supports, the assumptions, and the maintained
        independence assumption, for ``print(model.summary())``."""
        if self.exclusion:
            exclusion_line = (
                "Exclusion: imposed. Y(d, z) = Y(d, z') for every treatment value d "
                "and every pair of instrument vectors z, z'."
            )
        else:
            exclusion_line = (
                "Exclusion: not imposed. Y(d, z) may vary with z at every treatment "
                "value d."
            )
        m = self.n_instruments
        names = ", ".join(f"Z{j + 1}" for j in range(m))
        coordinate_lines = [
            f"    Z{j + 1}: {support}" for j, support in enumerate(self.z_supports)
        ]
        lines = [
            "PartialMonotonicityModel: discrete IV model with a vector instrument "
            "(Kaido and Ponomarev, 2025)",
            f"  Outcome support Y:    {self.y_support}",
            f"  Treatment support D:  {self.d_support}",
            f"  Instrument coordinates: {m}, Z = ({names}{',' if m == 1 else ''})",
            *coordinate_lines,
            f"  Instrument support Z: Cartesian product, {len(self.z_support)} vectors",
            f"  Nodes (y, d, z):      {len(self.nodes)}",
            "  Potential responses: D(z) and Y(d, z); observed D = D(Z), Y = Y(D(Z), Z).",
            "  Assumptions:",
            "    - Partial monotonicity: imposed (always). D(z) is nondecreasing in each",
            "      instrument coordinate holding the others fixed, so D(z) <= D(z')",
            "      whenever z <= z' componentwise; incomparable instrument vectors impose",
            "      no ordering on D.",
            f"    - {exclusion_line}",
            "    - No outcome monotonicity is imposed.",
            "  Maintained (not a pairwise rule): the full potential-response vector",
            "    ((Y(d, z))_{d, z}, (D(z))_z) is independent of Z.",
            "  Graph: build_graph() applies Method 1 (pairwise compatibility) and returns",
            "    a networkx.Graph for GraphAnalyzer.",
        ]
        return "\n".join(lines)
