#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   // automatic conversion for STL containers
#include <pybind11/numpy.h> // numpy array access

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

// Source of the extension module _supportrestriction_graph (built only when
// SUPPORTRESTRICTION_BUILD_CPP=1, see setup.py). Ported from
// src/SupportRestriction/test_odd_holes_pybind11.cpp of the SupportRestriction2026
// package at commit c73eae9; the search is unchanged, the module is renamed and
// the adjacency matrix is now checked before its dtype conversion.
//
// Detects whether an undirected graph contains an induced odd cycle (odd hole)
// and returns one such cycle as a witness.
//
// Two Python-visible functions share one search implementation:
// - find_odd_hole(adj_list, A_matrix)     -> list of vertex indices (odd hole) or []
// - has_odd_holes_fast(adj_list, A_matrix) -> bool
//
// Both accept the keyword-only argument diagnostics (default False). With
// diagnostics=True they return (result, stats), where result is what the normal
// call returns and stats is a dict of search counters (see SearchStats). The
// search and its order are the same in both modes; with diagnostics off no
// counter, clock or dictionary is touched.
//
// Input validation (performed before any search; failures raise ValueError):
// - A_matrix is 2D with shape (n, n), n = len(adj_list);
// - every neighbour index lies in [0, n), no self-loops, no duplicate neighbours;
// - adj_list is symmetric (v in adj_list[u] iff u in adj_list[v]);
// - every A_matrix entry is 0 or 1 and A[u, v] == 1 iff v in adj_list[u].
// The 0/1 check reads the caller's original values, before the matrix is
// converted to a C-contiguous int array (see binary_matrix_from): a fractional
// entry, a NaN or an integer outside the C int range raises ValueError instead
// of being truncated or wrapped into a valid entry. A_matrix must be a
// numpy.ndarray with a boolean, integer or real floating dtype (TypeError
// otherwise; nested lists and other array-likes are not converted); floats
// that are exactly 0.0 or 1.0 are accepted. A masked array with masked entries
// raises ValueError.
//
// Notes:
// - Initial wedges are explored one at a time: a wedge is generated, pushed and
//   explored together with its extensions before the next wedge is generated.
//   The wedge loops count downward, so paths are examined in the same order as
//   in the earlier version, which stored every wedge before searching.
// - The wedge-listing step is kept generic (no part structure V_k).
// - The key fix (vs. the original code) is in the path-expansion step:
//   when extending a line subgraph, the OLD endpoints become internal vertices, so
//   we must update N(V) by adding N(old_first) and N(old_last), not N(new_endpoints).
// - The witness is built at the detection point as [u] + path.S + [w]; the
//   particular cycle, rotation and orientation are not part of the contract.

using AdjList = std::vector<std::vector<int>>;
using IntMatrix = py::array_t<int, py::array::c_style | py::array::forcecast>;

namespace {

// Counters of one search, collected only when diagnostics are requested. Each
// call owns a fresh instance; nothing accumulates across calls.
//
// Wedges are explored one at a time, so: initial_wedges_generated counts every
// wedge of the graph only when no hole is found (a successful search returns
// before the remaining wedges are generated); wedges_before_first_search is 1
// whenever a wedge exists; max_pending_paths covers the current wedge and its
// extensions only.
struct SearchStats {
    std::uint64_t initial_wedges_generated = 0;   // valid initial wedges pushed
    std::uint64_t wedges_before_first_search = 0; // wedges generated when the first path was examined
    std::uint64_t paths_explored = 0;             // paths removed from the stack for examination
    std::uint64_t extensions_created = 0;         // longer paths pushed
    std::size_t max_pending_paths = 0;            // largest stack size after an insertion
    std::size_t max_path_vertices = 0;            // largest path.S.size() examined
    std::size_t witness_length = 0;               // cycle length on success, 0 otherwise
    double wedge_generation_seconds = 0.0;        // wedge loops, outside the expansion loop
};

// Returns the int matrix the search reads, after checking the caller's ORIGINAL
// values. The dtype conversion at the end (forcecast) truncates fractions and
// wraps wide integers, so it is only applied once every entry has been shown to
// equal 0 or 1 in the caller's own dtype; for such a matrix it is exact.
IntMatrix binary_matrix_from(const py::object& A_input) {
    // Only an actual numpy array is accepted. Turning any other object (a
    // nested list, a deque, an object with __array__) into an array goes
    // through numpy's coercion rules, which can change what the caller wrote
    // without notice, e.g. read the data hidden under a masked element. That
    // conversion is left to the caller, who can inspect its result.
    if (!py::isinstance<py::array>(A_input)) {
        throw py::type_error(
            "A_matrix must be a numpy.ndarray; got " +
            py::str(py::type::of(A_input).attr("__name__")).cast<std::string>() + ".");
    }

    // A masked entry stands for a missing value: the data hidden under the mask
    // must not be read as a 0 or a 1.
    if (py::module_::import("numpy.ma").attr("is_masked")(A_input).cast<bool>()) {
        throw std::invalid_argument("A_matrix has masked entries.");
    }

    // The same data as a base-class ndarray (a view, still in the caller's dtype).
    py::array original = py::array::ensure(A_input);
    if (!original) {
        throw py::type_error("A_matrix could not be viewed as a numpy.ndarray.");
    }

    py::module_ np = py::module_::import("numpy");
    const char kind = original.dtype().kind();
    if (kind != 'b' && kind != 'i' && kind != 'u' && kind != 'f') {
        throw py::type_error(
            "A_matrix must have a boolean, integer or real floating dtype; got dtype " +
            py::str(original.dtype()).cast<std::string>() + ".");
    }
    if (original.ndim() != 2) {
        throw std::invalid_argument("A_matrix must be a 2D numpy array.");
    }

    // Entries that are neither 0 nor 1, compared in the original dtype (NaN
    // compares unequal to both and is therefore rejected).
    py::object not_binary = np.attr("logical_and")(
        np.attr("not_equal")(original, 0), np.attr("not_equal")(original, 1));
    if (np.attr("any")(not_binary).cast<bool>()) {
        py::object first = np.attr("argwhere")(not_binary)[py::int_(0)];
        py::object value = original[py::tuple(first)];
        throw std::invalid_argument(
            "A_matrix[" + py::str(first[py::int_(0)]).cast<std::string>() + ", " +
            py::str(first[py::int_(1)]).cast<std::string>() + "] = " +
            py::str(value).cast<std::string>() + "; entries must be 0 or 1.");
    }

    // Native C int, C-contiguous and aligned ("C", "A"): what the unchecked
    // element access of the search requires. numpy.require copies only when the
    // caller's array does not already have these properties.
    IntMatrix converted = IntMatrix::ensure(
        np.attr("require")(original, py::dtype::of<int>(), py::make_tuple("C", "A")));
    if (!converted) {
        throw py::type_error("A_matrix could not be converted to an integer matrix.");
    }
    return converted;
}

// Validates both inputs and returns the adjacency sets the search uses.
std::vector<std::unordered_set<int>> validate_inputs(
    const AdjList& adj_list,
    const IntMatrix& A_matrix
) {
    if (adj_list.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("adj_list has more vertices than the search can index with int.");
    }
    const int n = static_cast<int>(adj_list.size());

    if (A_matrix.ndim() != 2) {
        throw std::invalid_argument("A_matrix must be a 2D numpy array.");
    }
    if (A_matrix.shape(0) != n || A_matrix.shape(1) != n) {
        throw std::invalid_argument("A_matrix must have shape (n, n) matching adj_list.size().");
    }

    // Build adjacency sets for O(1) membership checks in the wedge step and
    // when computing exclusive neighborhoods; reject bad indices, self-loops
    // and duplicate neighbours on the way.
    std::vector<std::unordered_set<int>> adj_sets(n);
    for (int v = 0; v < n; ++v) {
        adj_sets[v].reserve(adj_list[v].size() * 2);
        for (int nb : adj_list[v]) {
            if (nb < 0 || nb >= n) {
                throw std::invalid_argument(
                    "adj_list[" + std::to_string(v) + "] contains vertex " + std::to_string(nb) +
                    ", outside the range [0, " + std::to_string(n) + ").");
            }
            if (nb == v) {
                throw std::invalid_argument(
                    "adj_list[" + std::to_string(v) + "] contains a self-loop.");
            }
            if (!adj_sets[v].insert(nb).second) {
                throw std::invalid_argument(
                    "adj_list[" + std::to_string(v) + "] lists vertex " + std::to_string(nb) +
                    " more than once.");
            }
        }
    }

    for (int v = 0; v < n; ++v) {
        for (int nb : adj_list[v]) {
            if (adj_sets[nb].count(v) == 0) {
                throw std::invalid_argument(
                    "adj_list is not symmetric: " + std::to_string(nb) + " is in adj_list[" +
                    std::to_string(v) + "] but " + std::to_string(v) + " is not in adj_list[" +
                    std::to_string(nb) + "].");
            }
        }
    }

    auto A = A_matrix.unchecked<2>();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const int a = A(i, j);
            if (a != 0 && a != 1) {
                throw std::invalid_argument(
                    "A_matrix[" + std::to_string(i) + ", " + std::to_string(j) + "] = " +
                    std::to_string(a) + "; entries must be 0 or 1.");
            }
            const bool adjacent = adj_sets[i].count(j) != 0;
            if ((a == 1) != adjacent) {
                throw std::invalid_argument(
                    "A_matrix[" + std::to_string(i) + ", " + std::to_string(j) + "] = " +
                    std::to_string(a) + " disagrees with adj_list.");
            }
        }
    }

    return adj_sets;
}

// Shared search. Returns the odd hole found as [u] + path.S + [w], or an empty
// vector when the whole search finishes without finding one. stats is nullptr
// in normal mode; every diagnostic update below is guarded by if (stats) and
// none of them changes the search or its order.
std::vector<int> find_odd_hole_impl(
    const AdjList& adj_list,
    const IntMatrix& A_matrix,
    SearchStats* stats
) {
    std::vector<std::unordered_set<int>> adj_sets = validate_inputs(adj_list, A_matrix);

    const int n = static_cast<int>(adj_list.size()); // validated to fit in int

    auto A = A_matrix.unchecked<2>(); // fast direct access (no bounds checks)

    struct Path {
        std::vector<int> S;          // full path: [endpoint, ..., endpoint]
        std::unordered_set<int> N_V; // N(V) where V is the INTERNAL set (S without endpoints)
    };

    std::vector<Path> stack;
    stack.reserve(200000);

    // ------------------------------------------------------------
    // 1) Initial wedges, one at a time: induced 3-vertex paths (u - v - w)
    // ------------------------------------------------------------
    // Each wedge is pushed and explored together with its extensions (block 2)
    // before the next wedge is generated, so the stack only holds paths
    // descended from the current wedge. The three loops count downward: that
    // is the order in which a stack holding every wedge would release them
    // (last generated, first examined), so paths are examined in the same
    // order as when all wedges were stored before the search began.
    //
    // Diagnostics: wedge_generation_seconds is the time spent in these loops
    // outside the expansion loop.
    std::chrono::steady_clock::time_point wedges_started;
    std::chrono::steady_clock::duration expansion_total{};
    if (stats) {
        wedges_started = std::chrono::steady_clock::now();
    }

    for (int v = n - 1; v >= 0; --v) {
        const auto& Nv = adj_list[v];
        const int d = static_cast<int>(Nv.size());
        if (d < 2) continue;

        for (int i = d - 2; i >= 0; --i) {
            const int u = Nv[i];
            for (int j = d - 1; j > i; --j) {
                const int w = Nv[j];

                // Induced: u and w must NOT be adjacent.
                if (adj_sets[u].count(w) != 0) continue;

                Path p;
                p.S = {u, v, w};

                // For a wedge (u, V={v}, w), we need N(V)=N(v).
                p.N_V.reserve(Nv.size() * 2);
                for (int nb : Nv) {
                    p.N_V.insert(nb);
                }

                stack.push_back(std::move(p));
                if (stats) {
                    ++stats->initial_wedges_generated;
                    stats->max_pending_paths =
                        std::max(stats->max_pending_paths, stack.size());
                }

                std::chrono::steady_clock::time_point expansion_started;
                if (stats) {
                    expansion_started = std::chrono::steady_clock::now();
                }

                // ------------------------------------------------------------
                // 2) DFS-like expansion loop (stack)
                // ------------------------------------------------------------
                while (!stack.empty()) {
                    Path path = std::move(stack.back());
                    stack.pop_back();

                    if (stats) {
                        if (stats->paths_explored == 0) {
                            stats->wedges_before_first_search = stats->initial_wedges_generated;
                        }
                        ++stats->paths_explored;
                        stats->max_path_vertices =
                            std::max(stats->max_path_vertices, path.S.size());
                    }

                    const int first = path.S.front();
                    const int last  = path.S.back();

                    const auto& N_first = adj_list[first];
                    const auto& N_last  = adj_list[last];

                    // Compute exclusive neighborhoods:
                    // N1 = N(first) \ N(last) \ N(V)
                    // N2 = N(last)  \ N(first) \ N(V)
                    std::vector<int> N1;
                    N1.reserve(N_first.size());
                    for (int u : N_first) {
                        if (path.N_V.count(u) == 0 && adj_sets[last].count(u) == 0) {
                            N1.push_back(u);
                        }
                    }

                    std::vector<int> N2;
                    N2.reserve(N_last.size());
                    for (int w : N_last) {
                        if (path.N_V.count(w) == 0 && adj_sets[first].count(w) == 0) {
                            N2.push_back(w);
                        }
                    }

                    if (N1.empty() || N2.empty()) {
                        continue;
                    }

                    // --------------------------
                    // N(V) update
                    // --------------------------
                    // If we extend (first, V, last) to (u, V~, w), then
                    // V~ = V ∪ {first, last}, so
                    // N(V~) = N(V) ∪ N(first) ∪ N(last).
                    std::unordered_set<int> N_V_tilde = path.N_V;
                    N_V_tilde.reserve(N_V_tilde.size() + N_first.size() + N_last.size());
                    for (int x : N_first) N_V_tilde.insert(x);
                    for (int x : N_last)  N_V_tilde.insert(x);

                    const std::size_t old_len = path.S.size();

                    for (int u : N1) {
                        for (int w : N2) {
                            // If the two candidate endpoints are adjacent, we found an odd hole:
                            // the induced odd cycle u - path.S - w - u.
                            if (A(u, w) == 1) {
                                if (stats) {
                                    stats->witness_length = old_len + 2; // path.S plus u and w
                                    // Wedge loops up to the start of this expansion,
                                    // without the expansions of earlier wedges.
                                    stats->wedge_generation_seconds =
                                        std::chrono::duration<double>(
                                            (expansion_started - wedges_started)
                                            - expansion_total).count();
                                }
                                std::vector<int> witness;
                                witness.reserve(old_len + 2);
                                witness.push_back(u);
                                witness.insert(witness.end(), path.S.begin(), path.S.end());
                                witness.push_back(w);
                                return witness;
                            }

                            Path new_path;
                            new_path.S.reserve(old_len + 2);
                            new_path.S.push_back(u);
                            new_path.S.insert(new_path.S.end(), path.S.begin(), path.S.end());
                            new_path.S.push_back(w);

                            new_path.N_V = N_V_tilde;

                            stack.push_back(std::move(new_path));
                            if (stats) {
                                ++stats->extensions_created;
                                stats->max_pending_paths =
                                    std::max(stats->max_pending_paths, stack.size());
                            }
                        }
                    }
                }

                if (stats) {
                    expansion_total += std::chrono::steady_clock::now() - expansion_started;
                }
            }
        }
    }

    if (stats) {
        stats->wedge_generation_seconds = std::chrono::duration<double>(
            (std::chrono::steady_clock::now() - wedges_started) - expansion_total).count();
    }

    return {};
}

py::dict stats_to_dict(const SearchStats& stats) {
    py::dict d;
    d["initial_wedges_generated"] = stats.initial_wedges_generated;
    d["wedges_before_first_search"] = stats.wedges_before_first_search;
    d["paths_explored"] = stats.paths_explored;
    d["extensions_created"] = stats.extensions_created;
    d["max_pending_paths"] = stats.max_pending_paths;
    d["max_path_vertices"] = stats.max_path_vertices;
    d["witness_length"] = stats.witness_length;
    d["wedge_generation_seconds"] = stats.wedge_generation_seconds;
    return d;
}

} // namespace

// Both bindings receive the matrix as the caller passed it (py::object) and
// convert it with binary_matrix_from, so no value is cast before it is checked.

// Normal mode returns the vertex list; diagnostics=True returns (vertex_list, stats).
py::object find_odd_hole(const AdjList& adj_list, const py::object& A_input, bool diagnostics) {
    const IntMatrix A_matrix = binary_matrix_from(A_input);
    if (!diagnostics) {
        return py::cast(find_odd_hole_impl(adj_list, A_matrix, nullptr));
    }
    SearchStats stats;
    std::vector<int> witness = find_odd_hole_impl(adj_list, A_matrix, &stats);
    return py::make_tuple(py::cast(witness), stats_to_dict(stats));
}

// Normal mode returns the bool; diagnostics=True returns (found, stats).
py::object has_odd_holes_fast(const AdjList& adj_list, const py::object& A_input, bool diagnostics) {
    const IntMatrix A_matrix = binary_matrix_from(A_input);
    if (!diagnostics) {
        return py::bool_(!find_odd_hole_impl(adj_list, A_matrix, nullptr).empty());
    }
    SearchStats stats;
    const bool found = !find_odd_hole_impl(adj_list, A_matrix, &stats).empty();
    return py::make_tuple(py::bool_(found), stats_to_dict(stats));
}

#define SEARCH_DIAGNOSTICS_DOC \
    "\n\nWith the keyword-only argument diagnostics=True the return value is (result, stats):\n" \
    "result is what the normal call returns and stats is a dict with the counters\n" \
    "initial_wedges_generated, wedges_before_first_search, paths_explored,\n" \
    "extensions_created, max_pending_paths, max_path_vertices, witness_length (int) and\n" \
    "wedge_generation_seconds (float). The search and its order are the same in both\n" \
    "modes; with diagnostics=False (the default) nothing is collected."

PYBIND11_MODULE(_supportrestriction_graph, m) {
    m.def("has_odd_holes_fast", &has_odd_holes_fast,
          py::arg("adj_list"), py::arg("A_matrix"),
          py::kw_only(), py::arg("diagnostics") = false,
          "Detects whether an undirected graph contains an induced odd cycle (odd hole).\n\n"
          "adj_list: list of neighbour lists over vertices 0..n-1; A_matrix: (n, n) binary\n"
          "adjacency matrix in the same vertex order. Raises ValueError for invalid input\n"
          "and TypeError for a matrix that is not a numpy.ndarray of boolean, integer or\n"
          "real floating dtype."
          SEARCH_DIAGNOSTICS_DOC);
    m.def("find_odd_hole", &find_odd_hole,
          py::arg("adj_list"), py::arg("A_matrix"),
          py::kw_only(), py::arg("diagnostics") = false,
          "Returns one odd hole (induced odd cycle of length >= 5) as an ordered list of\n"
          "vertex indices: consecutive entries are adjacent, the last is adjacent to the\n"
          "first, and the starting vertex is not repeated. Returns [] when the whole search\n"
          "finishes without finding an odd hole. Raises ValueError for invalid input\n"
          "(shape, indices, self-loops, duplicate neighbours, asymmetry, or a matrix that is\n"
          "not binary or disagrees with adj_list). The matrix entries are checked in their\n"
          "original dtype, so a fractional or out-of-range value is rejected, not rounded.\n"
          "Which cycle, rotation and orientation is returned is not specified. Same inputs\n"
          "as has_odd_holes_fast."
          SEARCH_DIAGNOSTICS_DOC);
}
