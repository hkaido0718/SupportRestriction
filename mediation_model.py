"""MediationModel: a discrete mediation model with a vector of mediators and no direct
treatment effect, for the potential-response graph framework of Kaido and Ponomarev
(2025), "Testing Exclusion and Shape Restrictions in Potential Outcomes Models"
(arXiv:2512.20851).

The model
---------
The researcher observes an outcome Y and an assigned treatment D, each with a finite
support of numeric values, and a mediator vector M = (M_1, ..., M_J) with J >= 1
coordinates. Each coordinate M_j has its own finite support, and the support of M is the
Cartesian product of the coordinate supports; this first version does not accept an
arbitrary joint mediator support. Potential mediators M(d) and potential outcomes
Y(m, d) generate the observed responses through

    M = M(D),    Y = Y(M(D), D).

Maintained assumption (documented here; it is not a pairwise rule): the full vector of
potential responses ((M(d))_d, (Y(m, d))_{m, d}) is independent of D.

Restriction (always imposed). For the nonempty set S of selected mediator coordinates,
``active_mediators``,

    Y(m, d) = g(m_S)    for every mediator vector m and treatment value d,

where m_S is the subvector of m on S and g is an arbitrary function. By default S
contains every coordinate: the outcome responds to the whole mediator vector and there
is no direct effect of D on Y. Selecting only some coordinates additionally excludes the
unselected coordinates from the outcome response; it still excludes a direct treatment
effect. Selection allows the selected coordinates to affect Y; it does not require any
of them to have a nonzero effect. The potential mediators M(d) are otherwise
unrestricted, so an unselected mediator may still respond to D. No monotonicity of any
kind is imposed.

The potential response graph (Method 1)
---------------------------------------
Each node is an observable event (y, m, d), where m is a tuple with one entry per
mediator coordinate, even when J = 1. Two nodes with different treatment values are
joined by an edge exactly when some potential-response vector satisfying the restriction
generates both events; nodes with the same treatment value are never joined, because
M = M(D) takes a single value at a given D. Method 1 of Kaido and Ponomarev (2025,
Section 5.1) decides each edge from the pair alone: the pair u = (y, m, d),
v = (y', m', d') is forbidden when the selected profiles agree, m_S = m'_S, but the
outcomes differ, y != y', since the two events would fix g(m_S) to two different
values. Different selected profiles impose nothing on the two outcomes. The rule is
symmetric in u and v and involves no ordering of any support.

Why the pairwise rule is exact. Suppose the rule does not reject u and v, with d != d'.
Set M(d) = m and M(d') = m': these are distinct components of the vector because
d != d', and M(.) is unrestricted. Set g(m_S) = y and g(m'_S) = y': when m_S != m'_S
these are two different arguments of g, and when m_S = m'_S the rule gives y = y'.
Every other component is free, so an admissible vector generates both events.

The same construction extends any set of pairwise compatible nodes (y_i, m_i, d_i)
with distinct treatment values (automatic in a clique, since same-d nodes are never
adjacent): M(d_i) = m_i and g((m_i)_S) = y_i, which is well defined because two nodes
with equal selected profiles have equal outcomes by pairwise compatibility. So the
model satisfies the pairwise incompatibility criterion (Condition 2 of the paper) and
every maximal clique of the graph is a support point. Perfectness is a separate matter
and is checked numerically by ``GraphAnalyzer``: with two treatment values the graph is
bipartite, hence perfect, and the maximal-independent-set inequalities are sharp; with
three or more treatment values, at least two outcome values and at least two selected
profiles, the graph contains an odd hole (one binary mediator, binary Y and D in
{0, 1, 2} give the exclusion-only IV graph of Table 4 of the paper), so the inequalities
are then necessary but not sharp.

Relation to ``IVModel``. With every coordinate selected, the rule has the form of the
exclusion rule of ``IVModel`` (same treatment, different outcome) with the mediator in
the role of the treatment and the treatment in the role of the instrument, so the graph
coincides with the exclusion-only IV graph after relabelling (y, (m,), d) as (y, m, d).
With a proper subset S, the graph is the all-coordinates graph of the model on the
selected coordinates alone, lifted through the projection m -> m_S.

``build_graph`` returns a ``networkx.Graph``. Maximal independent sets, plots, and the
regularity checks are the job of ``GraphAnalyzer`` in ``graph_analysis_utils.py``,
which this module deliberately does not import.
"""

import numbers
from itertools import combinations, product

import networkx as nx

from support_validation import validate_support

__all__ = ["MediationModel"]


class MediationModel:
    """A finite discrete mediation model with a vector of mediators, full mediation
    through the selected coordinates, and no direct treatment effect.

    Parameters
    ----------
    y_support, d_support : iterable of real numbers
        Supports of the outcome and the assigned treatment. Each must be a nonempty,
        non-string iterable of distinct finite real numbers. They are stored sorted.
    m_supports : iterable of iterables of real numbers
        One support per mediator coordinate, in coordinate order; at least one
        coordinate is required. Each coordinate support is validated like
        ``y_support`` and stored sorted. A flat sequence of numbers is rejected: for a
        single mediator write ``m_supports=((0, 1),)``.
    active_mediators : "all" or iterable of int, default "all"
        The mediator coordinates that may affect the outcome, as zero-based indices
        into ``m_supports``. ``"all"`` selects every coordinate. A selection must be a
        nonempty, non-string iterable of distinct integers in ``range(n_mediators)``;
        ``None``, booleans, floats, other strings and a bare integer are rejected (for
        the first mediator alone write ``(0,)``). The selection is stored sorted, so
        ``(1, 0)`` and ``(0, 1)`` give the same model.

    Settings are fixed after construction: create a new model to change supports or
    the selection.

    Attributes
    ----------
    y_support, d_support : tuple
        Validated, sorted supports.
    m_supports : tuple of tuples
        Validated, sorted coordinate supports, in the order supplied.
    m_support : tuple of tuples
        The mediator support: the Cartesian product of the coordinate supports, in
        product order (the first coordinate outermost).
    n_mediators : int
        The number of mediator coordinates, ``len(m_supports)``.
    active_mediators : tuple of int
        The selected coordinate indices in increasing order; ``"all"`` is resolved to
        ``tuple(range(n_mediators))``.
    nodes : tuple of (y, m, d)
        All observable events, with y outermost, then m in product order, then d. The
        entry m is a tuple with one value per coordinate, even when there is one
        mediator.
    """

    _frozen = False

    def __init__(self, y_support, m_supports, d_support, active_mediators="all"):
        self.y_support = validate_support(y_support, "y_support")
        self.m_supports = self._validate_m_supports(m_supports)
        self.d_support = validate_support(d_support, "d_support")
        self.m_support = tuple(product(*self.m_supports))
        self.n_mediators = len(self.m_supports)
        self.active_mediators = self._validate_active_mediators(
            active_mediators, self.n_mediators
        )
        self.nodes = self._make_nodes()
        self._frozen = True

    # ------------------------------------------------------------------ immutability
    def __setattr__(self, name, value):
        if self._frozen:
            raise AttributeError(
                f"MediationModel settings are fixed after construction; cannot set "
                f"{name!r}. Create a new MediationModel instead."
            )
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if self._frozen:
            raise AttributeError(
                f"MediationModel settings are fixed after construction; cannot delete "
                f"{name!r}."
            )
        super().__delattr__(name)

    def __repr__(self):
        return (
            f"MediationModel(y_support={self.y_support!r}, "
            f"m_supports={self.m_supports!r}, "
            f"d_support={self.d_support!r}, "
            f"active_mediators={self.active_mediators!r})"
        )

    # ------------------------------------------------------------------ validation
    @staticmethod
    def _validate_m_supports(m_supports):
        """Return ``m_supports`` as a tuple of sorted coordinate supports.

        Raises TypeError for a string or a non-iterable, for a flat sequence of numbers
        (each coordinate support must itself be an iterable of numbers), and for any
        element of a coordinate support that is not a real number. Raises ValueError
        when there are no coordinates or when a coordinate support is empty, has a
        duplicate, or contains nan or inf. Error messages name the coordinate as
        ``m_supports[j]``.
        """
        if isinstance(m_supports, (str, bytes, bytearray)):
            raise TypeError(
                f"m_supports must be a sequence of coordinate supports, not a string: "
                f"{m_supports!r}"
            )
        try:
            coordinates = list(m_supports)
        except TypeError:
            raise TypeError(
                f"m_supports must be an iterable of coordinate supports, got "
                f"{m_supports!r} of type {type(m_supports).__name__}"
            ) from None
        if not coordinates:
            raise ValueError("m_supports must contain at least one mediator coordinate")
        validated = []
        for j, coordinate in enumerate(coordinates):
            if isinstance(coordinate, (str, bytes, bytearray)) or not hasattr(
                coordinate, "__iter__"
            ):
                raise TypeError(
                    f"m_supports[{j}] must be an iterable of numbers (one support per "
                    f"mediator coordinate), got {coordinate!r} of type "
                    f"{type(coordinate).__name__}; for a single mediator write "
                    f"m_supports=((0, 1),)"
                )
            validated.append(validate_support(coordinate, f"m_supports[{j}]"))
        return tuple(validated)

    @staticmethod
    def _validate_active_mediators(active_mediators, n_mediators):
        """Return the selected coordinate indices as a sorted tuple of ``int``.

        ``"all"`` resolves to ``tuple(range(n_mediators))``. Otherwise the selection
        must be a nonempty, non-string iterable of distinct integers in
        ``range(n_mediators)``. Raises ValueError for any other string, for an empty
        selection, a duplicate index, or an index outside that range (negative indices
        included). Raises TypeError for ``None``, a bool, a float, a bare integer (for
        the first mediator alone write ``(0,)``), any other non-iterable, and for an
        element that is not an integer (booleans included). NumPy integers are
        converted to ``int``.
        """
        if isinstance(active_mediators, str):
            if active_mediators == "all":
                return tuple(range(n_mediators))
            raise ValueError(
                f"active_mediators must be 'all' or a sequence of mediator coordinate "
                f"indices; got the string {active_mediators!r}"
            )
        if isinstance(active_mediators, bool):
            raise TypeError(
                f"active_mediators must be 'all' or a sequence of mediator coordinate "
                f"indices, not a bool: {active_mediators!r}"
            )
        if isinstance(active_mediators, numbers.Integral):
            raise TypeError(
                f"active_mediators must be 'all' or a sequence of mediator coordinate "
                f"indices, got the integer {active_mediators!r}; for a single "
                f"coordinate write active_mediators=({active_mediators!r},)"
            )
        if isinstance(active_mediators, (bytes, bytearray)):
            raise TypeError(
                f"active_mediators must be 'all' or a sequence of mediator coordinate "
                f"indices, not bytes: {active_mediators!r}"
            )
        try:
            items = list(active_mediators)
        except TypeError:
            raise TypeError(
                f"active_mediators must be 'all' or an iterable of mediator coordinate "
                f"indices, got {active_mediators!r} of type "
                f"{type(active_mediators).__name__}"
            ) from None
        if not items:
            raise ValueError("active_mediators must select at least one mediator coordinate")
        indices = []
        for value in items:
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise TypeError(
                    f"active_mediators must contain integer coordinate indices, got "
                    f"{value!r} of type {type(value).__name__}"
                )
            index = int(value)  # numpy integer -> plain int
            if not 0 <= index < n_mediators:
                raise ValueError(
                    f"active_mediators indices must lie in range({n_mediators}) "
                    f"(zero-based coordinate indices), got {value!r}"
                )
            indices.append(index)
        if len(set(indices)) != len(indices):
            raise ValueError(
                f"active_mediators must not contain duplicate indices: {tuple(items)!r}"
            )
        return tuple(sorted(indices))

    # ------------------------------------------------------------------ nodes and groups
    def _make_nodes(self):
        """All (y, m, d) tuples, y outermost, then m in product order, then d."""
        return tuple(
            (y, m, d) for y in self.y_support for m in self.m_support for d in self.d_support
        )

    def _project_mediators(self, m):
        """The coordinates of the mediator vector ``m`` selected by ``active_mediators``,
        as a tuple in coordinate order.

        A vector with the wrong number of coordinates is rejected rather than silently
        indexed.
        """
        if len(m) != self.n_mediators:
            raise ValueError(
                f"mediator vectors must have {self.n_mediators} coordinate(s), got {m!r}"
            )
        return tuple(m[j] for j in self.active_mediators)

    def group_fn(self, node):
        """The assigned treatment value of a node; nodes with equal d form one part of
        the graph."""
        return node[2]

    # ------------------------------------------------------------------ pairwise rule
    def violates_mediation(self, u, v):
        """True if u and v cannot both occur under the mediation restriction: equal
        selected mediator profiles but different outcomes.

        Different selected profiles never reject, whatever the outcomes. Symmetric in
        its arguments. Meant for nodes with different treatment values; for a same-d
        pair it simply evaluates the same formula.
        """
        y, m, d = u
        yp, mp, dp = v
        return (
            self._project_mediators(m) == self._project_mediators(mp)
            and y != yp
        )

    def violate_pairwise_fn(self, u, v):
        """True if the mediation restriction rules out observing both u and v.

        Mediation is the only restriction of this class, so this delegates to
        ``violates_mediation``; the name is the interface shared with the other model
        classes and with ``GraphAnalyzer.enumerate_cliques_and_check``.
        """
        return self.violates_mediation(u, v)

    # ------------------------------------------------------------------ graph
    def build_graph(self):
        """Construct the potential response graph by Method 1 and return it.

        Every node in ``self.nodes`` is added, including any node that ends up with no
        edges. Each unordered pair of distinct nodes is visited once; pairs with the
        same treatment value are skipped, and an edge is added exactly when
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

    # ------------------------------------------------------------------ description
    def summary(self):
        """A readable description of the supports, the mediation restriction, and the
        maintained independence assumption, for ``print(model.summary())``."""
        J = self.n_mediators
        names = ", ".join(f"M{j + 1}" for j in range(J))
        selected = ", ".join(f"M{j + 1}" for j in self.active_mediators)
        unselected = ", ".join(
            f"M{j + 1}" for j in range(J) if j not in self.active_mediators
        )
        coordinate_lines = [
            f"    M{j + 1}: {support}" for j, support in enumerate(self.m_supports)
        ]
        if unselected:
            mediation_lines = [
                f"    - Mediation: imposed (always). Y(m, d) = g(m_S) with S = ({selected}"
                f"{',' if len(self.active_mediators) == 1 else ''}): the outcome",
                "      responds to the selected mediator coordinates only, so there is no",
                "      direct effect of D on Y.",
                f"    - Unselected coordinates ({unselected}) do not enter the outcome response;",
                "      they may still respond to D.",
            ]
        else:
            mediation_lines = [
                "    - Mediation: imposed (always). Y(m, d) = g(m): the outcome responds to",
                "      the whole mediator vector, so there is no direct effect of D on Y.",
                "    - Every mediator coordinate is selected; none is excluded from the",
                "      outcome response.",
            ]
        lines = [
            "MediationModel: discrete mediation model with a mediator vector "
            "(Kaido and Ponomarev, 2025)",
            f"  Outcome support Y:    {self.y_support}",
            f"  Treatment support D:  {self.d_support}",
            f"  Mediator coordinates: {J}, M = ({names}{',' if J == 1 else ''})",
            *coordinate_lines,
            f"  Mediator support M:   Cartesian product, {len(self.m_support)} vectors",
            f"  Selected mediators:   {self.active_mediators} -> {selected}",
            f"  Nodes (y, m, d):      {len(self.nodes)}",
            "  Potential responses: M(d) and Y(m, d); observed M = M(D), Y = Y(M(D), D).",
            "  Assumptions:",
            *mediation_lines,
            "    - No monotonicity is imposed; the potential mediators M(d) are",
            "      unrestricted.",
            "  Maintained (not a pairwise rule): the full potential-response vector",
            "    ((M(d))_d, (Y(m, d))_{m, d}) is independent of D.",
            "  Graph: build_graph() applies Method 1 (pairwise compatibility) and returns",
            "    a networkx.Graph for GraphAnalyzer.",
        ]
        return "\n".join(lines)
