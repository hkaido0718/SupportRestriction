"""IVModel: a discrete instrumental-variable model for the potential-response graph
framework of Kaido and Ponomarev (2025), "Testing Exclusion and Shape Restrictions in
Potential Outcomes Models" (arXiv:2512.20851).

The model
---------
The researcher observes an outcome Y, a treatment D, and an instrument Z, each with a
finite support of numeric values. Potential treatments D(z) and potential outcomes
Y(d, z) generate the observed responses through

    D = D(Z),    Y = Y(D(Z), Z).

Maintained assumption (documented here; it is not a pairwise rule): the full vector of
potential responses ((Y(d, z))_{d, z}, (D(z))_z) is independent of Z.

Optional support restrictions, selected by the two assumption flags:

* ``exclusion``       Y(d, z) = Y(d, z') for every treatment value d and every pair of
                      instrument values z, z'.
* ``d_monotonicity``  D(z) <= D(z') whenever z < z': treatment is nondecreasing in the
                      instrument. No outcome monotonicity is imposed.

At least one flag must be True. When exclusion is off, nothing is assumed about how
Y(d, z) varies with z, so the shorthand Y(d) is never used in that case.

The potential response graph (Method 1)
---------------------------------------
Each node is an observable event (y, d, z). Two nodes with different instrument values
are joined by an edge exactly when some potential-response vector satisfying the
enabled restrictions generates both events; nodes with the same instrument value are
never joined, because D = D(Z) takes a single value at a given Z. Method 1 of Kaido
and Ponomarev (2025, Section 5.1) decides each edge from the pair alone:

* exclusion forbids the pair when the treatments agree but the outcomes differ,
  since the two events would fix Y(d) to two different values;
* D-monotonicity forbids the pair when the treatment falls as the instrument rises.

A pair rejected by neither enabled rule is compatible. The two rules constrain
different components of the vector: the two treatment values extend to a
nondecreasing D(.) over the whole instrument support (carry each value forward to the
next observed instrument value, and use the smaller one below), and the outcome
values are assigned per treatment value under exclusion, or per (d, z) cell without
it, with every other component free. This is why ``violate_pairwise_fn`` combines the
enabled rules with ``or``. The same construction extends any pairwise compatible set
of nodes with distinct instrument values to a full vector, so the model satisfies the
pairwise incompatibility criterion (Condition 2 of the paper) and every maximal clique
of the graph is a support point.

``build_graph`` returns a ``networkx.Graph``. Maximal independent sets, plots, and the
regularity checks are the job of ``GraphAnalyzer`` in ``graph_analysis_utils.py``,
which this module deliberately does not import.
"""

import math
import numbers
from itertools import combinations

import networkx as nx

__all__ = ["IVModel"]


class IVModel:
    """A finite discrete IV model with optional exclusion and D-monotonicity.

    Parameters
    ----------
    y_support, d_support, z_support : iterable of real numbers
        Supports of the outcome, the treatment, and the instrument. Each must be a
        nonempty, non-string iterable of distinct finite real numbers. They are stored
        sorted, so comparisons follow the numeric values, not the order supplied.
    exclusion : bool, default True
        Impose Y(d, z) = Y(d, z') for all d and all z, z'.
    d_monotonicity : bool, default False
        Impose D(z) <= D(z') whenever z < z'.

    At least one flag must be True. To request D-monotonicity alone, pass
    ``exclusion=False`` explicitly. Settings are fixed after construction: create a new
    ``IVModel`` to change supports or assumptions.

    Attributes
    ----------
    y_support, d_support, z_support : tuple
        Validated, sorted supports.
    exclusion, d_monotonicity : bool
        The selected assumptions.
    nodes : tuple of (y, d, z)
        All observable events, with y outermost, then d, then z.
    """

    _frozen = False

    def __init__(self, y_support, d_support, z_support, exclusion=True, d_monotonicity=False):
        for name, flag in (("exclusion", exclusion), ("d_monotonicity", d_monotonicity)):
            if not isinstance(flag, bool):
                raise TypeError(
                    f"{name} must be a bool (True or False), got {flag!r} of type "
                    f"{type(flag).__name__}"
                )
        if not (exclusion or d_monotonicity):
            raise ValueError(
                "at least one of exclusion and d_monotonicity must be True; to impose "
                "D-monotonicity alone, pass exclusion=False and d_monotonicity=True"
            )
        self.y_support = self._validate_support(y_support, "y_support")
        self.d_support = self._validate_support(d_support, "d_support")
        self.z_support = self._validate_support(z_support, "z_support")
        self.exclusion = exclusion
        self.d_monotonicity = d_monotonicity
        self.nodes = self._make_nodes()
        self._frozen = True

    # ------------------------------------------------------------------ immutability
    def __setattr__(self, name, value):
        if self._frozen:
            raise AttributeError(
                f"IVModel settings are fixed after construction; cannot set {name!r}. "
                "Create a new IVModel instead."
            )
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if self._frozen:
            raise AttributeError(
                f"IVModel settings are fixed after construction; cannot delete {name!r}."
            )
        super().__delattr__(name)

    def __repr__(self):
        return (
            f"IVModel(y_support={self.y_support!r}, d_support={self.d_support!r}, "
            f"z_support={self.z_support!r}, exclusion={self.exclusion!r}, "
            f"d_monotonicity={self.d_monotonicity!r})"
        )

    # ------------------------------------------------------------------ validation
    @staticmethod
    def _validate_support(values, name):
        """Return ``values`` as a sorted tuple of distinct finite real numbers.

        Raises TypeError for a string or a non-iterable, and for an element that is not
        a real number (booleans count as not real here, so ``(0, True)`` is rejected
        rather than treated as a duplicate of ``(0, 1)``). Raises ValueError for an
        empty support, a duplicate value, or a value that is nan or infinite. NumPy
        scalars are converted to plain Python numbers; int and float are kept as given.
        """
        if isinstance(values, (str, bytes, bytearray)):
            raise TypeError(f"{name} must be a sequence of numbers, not a string: {values!r}")
        try:
            items = list(values)
        except TypeError:
            raise TypeError(
                f"{name} must be an iterable of numbers, got {values!r} of type "
                f"{type(values).__name__}"
            ) from None
        if not items:
            raise ValueError(f"{name} must not be empty")
        cleaned = []
        for value in items:
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(
                    f"{name} values must be real numbers, got {value!r} of type "
                    f"{type(value).__name__}"
                )
            if not math.isfinite(value):
                raise ValueError(f"{name} values must be finite, got {value!r}")
            if type(value).__module__ == "numpy":  # numpy scalar -> plain int or float
                value = value.item()
            cleaned.append(value)
        if len(set(cleaned)) != len(cleaned):
            raise ValueError(f"{name} must not contain duplicate values: {tuple(items)!r}")
        return tuple(sorted(cleaned))

    # ------------------------------------------------------------------ nodes and groups
    def _make_nodes(self):
        """All (y, d, z) tuples, y outermost, then d, then z, each in sorted order."""
        return tuple(
            (y, d, z)
            for y in self.y_support
            for d in self.d_support
            for z in self.z_support
        )

    def group_fn(self, node):
        """The instrument value of a node; nodes with equal z form one part of the graph."""
        return node[2]

    # ------------------------------------------------------------------ pairwise rules
    def violates_exclusion(self, u, v):
        """True if u and v cannot both occur under exclusion: same d, different y.

        Evaluated regardless of the ``exclusion`` flag. Meant for nodes with different
        instrument values; for a same-z pair it simply evaluates the same formula.
        """
        y, d, z = u
        yp, dp, zp = v
        return d == dp and y != yp

    def violates_d_monotonicity(self, u, v):
        """True if u and v cannot both occur under D(z) <= D(z') for z < z'.

        That is, the treatment is larger at the smaller instrument value. Evaluated
        regardless of the ``d_monotonicity`` flag.
        """
        y, d, z = u
        yp, dp, zp = v
        return (z < zp and d > dp) or (zp < z and dp > d)

    def violate_pairwise_fn(self, u, v):
        """True if an enabled assumption rules out observing both u and v.

        The enabled rules are combined with ``or``: a pair is compatible only if no
        enabled rule rejects it. This is exact because the rules constrain different
        components of the potential-response vector (see the module docstring).
        """
        return (
            self.exclusion and self.violates_exclusion(u, v)
        ) or (
            self.d_monotonicity and self.violates_d_monotonicity(u, v)
        )

    # ------------------------------------------------------------------ graph
    def build_graph(self):
        """Construct the potential response graph by Method 1 and return it.

        Every node in ``self.nodes`` is added, including any node that ends up with no
        edges. Each unordered pair of distinct nodes is visited once; pairs with the
        same instrument value are skipped, and an edge is added exactly when
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
        """A readable description of the supports, the assumptions, and the maintained
        independence assumption, for ``print(model.summary())``."""
        if self.exclusion:
            exclusion_line = (
                "Exclusion: imposed. Y(d, z) = Y(d, z') for every treatment value d "
                "and every pair of instrument values z, z'."
            )
        else:
            exclusion_line = (
                "Exclusion: not imposed. Y(d, z) may vary with z at every treatment "
                "value d."
            )
        if self.d_monotonicity:
            monotonicity_line = (
                "D-monotonicity: imposed. D(z) <= D(z') whenever z < z' (treatment is "
                "nondecreasing in the instrument)."
            )
        else:
            monotonicity_line = (
                "D-monotonicity: not imposed. The potential treatments D(z) are "
                "unrestricted across z."
            )
        lines = [
            "IVModel: discrete instrumental-variable model (Kaido and Ponomarev, 2025)",
            f"  Outcome support Y:    {self.y_support}",
            f"  Treatment support D:  {self.d_support}",
            f"  Instrument support Z: {self.z_support}",
            f"  Nodes (y, d, z):      {len(self.nodes)}",
            "  Potential responses: D(z) and Y(d, z); observed D = D(Z), Y = Y(D(Z), Z).",
            "  Assumptions:",
            f"    - {exclusion_line}",
            f"    - {monotonicity_line}",
            "    - No outcome monotonicity is imposed.",
            "  Maintained (not a pairwise rule): the full potential-response vector",
            "    ((Y(d, z))_{d, z}, (D(z))_z) is independent of Z.",
            "  Graph: build_graph() applies Method 1 (pairwise compatibility) and returns",
            "    a networkx.Graph for GraphAnalyzer.",
        ]
        return "\n".join(lines)
