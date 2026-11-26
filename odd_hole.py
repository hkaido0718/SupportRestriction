"""Label-preserving adapter and independent verifier for the compiled odd-hole search.

The compiled function ``_supportrestriction_graph.find_odd_hole`` (source:
``cpp/odd_holes.cpp``) works on integer vertex indices.  The helpers here fix one
explicit vertex order, build both native inputs from it, map a returned witness
back to the original node labels, and check that witness with pure Python before
handing it to the caller.

The extension is optional: it is built only when the package is installed with
``SUPPORTRESTRICTION_BUILD_CPP=1``.  Importing this module never loads it; it is
loaded by :func:`_require_cpp` at the moment a search is requested, and a missing
extension is then an ``ImportError``, never a search result.  ``native_inputs``,
``verify_odd_hole`` and ``is_odd_hole`` are pure Python and work without it.

Ported from ``src/SupportRestriction/odd_hole.py`` of the SupportRestriction2026
package at commit ``c73eae9``.
"""
import networkx as nx
import numpy as np

__all__ = [
    "native_inputs",
    "find_odd_hole_witness",
    "verify_odd_hole",
    "is_odd_hole",
    "find_odd_hole_or_antihole",
]


def _require_cpp():
    try:
        import _supportrestriction_graph
    except ImportError as exc:
        raise ImportError(
            "The C++ backend could not be loaded. Reinstall from source "
            "with SUPPORTRESTRICTION_BUILD_CPP=1 and a C++17 compiler."
        ) from exc
    return _supportrestriction_graph


def native_inputs(G, nodes=None):
    """Return ``(nodes, adj_list, A)`` for the compiled search.

    ``nodes`` fixes the vertex order shared by ``adj_list`` and ``A``; by
    default it is ``list(G.nodes())``.  ``A`` is the binary adjacency matrix
    (``dtype=numpy.intc``, edge attributes ignored).

    Raises ``TypeError`` for directed graphs or multigraphs and ``ValueError``
    for self-loops or a ``nodes`` list that is not a permutation of ``G.nodes()``.
    """
    if G.is_directed() or G.is_multigraph():
        raise TypeError("G must be an undirected simple networkx.Graph")
    if nx.number_of_selfloops(G) > 0:
        raise ValueError("G must not contain self-loops")

    nodes = list(G.nodes()) if nodes is None else list(nodes)
    if len(nodes) != G.number_of_nodes() or len(set(nodes)) != len(nodes) \
            or any(v not in G for v in nodes):
        raise ValueError("nodes must be a permutation of G.nodes()")

    index = {v: i for i, v in enumerate(nodes)}
    adj_list = [[index[u] for u in G.neighbors(v)] for v in nodes]
    A = nx.to_numpy_array(G, nodelist=nodes, dtype=np.intc, weight=None)
    return nodes, adj_list, A


def find_odd_hole_witness(G, nodes=None, verify=True):
    """Return one odd hole of ``G`` as an ordered list of original node labels.

    The list is empty when the compiled search finishes without finding an odd
    hole.  Consecutive entries are adjacent and the last entry is adjacent to
    the first; the starting vertex is not repeated.

    With ``verify=True`` the witness is checked with :func:`verify_odd_hole`
    before it is returned.  A witness that fails that check, or indices outside
    ``range(len(nodes))``, raise ``RuntimeError`` (a failure is never reported
    as an empty list).  ``verify=False`` skips the check so callers can time it
    separately.

    Raises ``ImportError`` when the C++ extension is not installed.
    """
    nodes, adj_list, A = native_inputs(G, nodes)
    native = _require_cpp()
    raw = native.find_odd_hole(adj_list, A)

    n = len(nodes)
    bad = [i for i in raw if not (isinstance(i, int) and 0 <= i < n)]
    if bad:
        raise RuntimeError(
            f"compiled search returned vertex indices {bad} outside range({n})"
        )

    witness = [nodes[i] for i in raw]
    if verify and witness:
        try:
            verify_odd_hole(G, witness)
        except ValueError as exc:
            raise RuntimeError(
                f"compiled witness {witness!r} failed independent verification: {exc}"
            ) from exc
    return witness


def verify_odd_hole(G, cycle):
    """Check with pure Python that ``cycle`` is an odd hole of ``G``.

    ``cycle`` is an ordered sequence of node labels; the closing edge from the
    last entry back to the first is implied.  The check uses only
    ``G.has_edge``: odd length at least 5, distinct vertices of ``G``,
    consecutive vertices adjacent, and no chord between non-consecutive ones.
    Raises ``ValueError`` naming the first failed condition.
    """
    cycle = list(cycle)
    k = len(cycle)
    if k < 5:
        raise ValueError(f"cycle has {k} vertices; an odd hole needs at least 5")
    if k % 2 == 0:
        raise ValueError(f"cycle has even length {k}")
    if len(set(cycle)) != k:
        raise ValueError("cycle repeats a vertex")
    missing = [v for v in cycle if v not in G]
    if missing:
        raise ValueError(f"vertices {missing!r} are not in G")
    for i in range(k):
        u, v = cycle[i], cycle[(i + 1) % k]
        if not G.has_edge(u, v):
            raise ValueError(f"consecutive vertices {u!r} and {v!r} are not adjacent")
    for i in range(k):
        for j in range(i + 2, k):
            if i == 0 and j == k - 1:
                continue  # the closing edge, already required above
            if G.has_edge(cycle[i], cycle[j]):
                raise ValueError(f"chord between {cycle[i]!r} and {cycle[j]!r}")


def is_odd_hole(G, cycle):
    """Boolean form of :func:`verify_odd_hole`."""
    try:
        verify_odd_hole(G, cycle)
    except ValueError:
        return False
    return True


def find_odd_hole_or_antihole(G, nodes=None, verify=True):
    """Return ``(kind, witness)`` explaining why ``G`` is not perfect, if it is not.

    ``kind`` is ``"odd_hole"`` with a witness in ``G``, ``"odd_antihole"`` with
    a witness that is an odd hole of ``nx.complement(G)`` (verified there), or
    ``None`` with an empty list when both searches finish without a witness.
    The complement is not searched when ``G`` itself yields an odd hole.

    An ``"odd_antihole"`` witness is ordered as a cycle of the complement:
    consecutive vertices (and the last and the first) are NOT adjacent in ``G``,
    and every other pair of its vertices is adjacent in ``G``.

    Both searches use the same vertex order ``nodes``.  ``(None, [])`` is returned
    only after both have finished; a missing extension (``ImportError``), invalid
    input or a witness that fails verification raises instead.
    """
    if nodes is not None:
        nodes = list(nodes)  # used by both searches; an iterator would be spent after the first
    witness = find_odd_hole_witness(G, nodes, verify)
    if witness:
        return "odd_hole", witness
    witness = find_odd_hole_witness(nx.complement(G), nodes, verify)
    if witness:
        return "odd_antihole", witness
    return None, []
