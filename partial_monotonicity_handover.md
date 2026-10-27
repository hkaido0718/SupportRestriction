# Notes on the `PartialMonotonicityModel` class

A discrete IV model with a vector instrument and partial treatment monotonicity, for the
potential-response graph framework of Kaido and Ponomarev (2025). It follows the design
of `IVModel` (see `iv_model_handover.md`) and was specified in Section 5 of the
supervisor's notes of October 1, 2026.

## Files

| File | Content |
|---|---|
| `partial_monotonicity_model.py` | The `PartialMonotonicityModel` class (networkx and `support_validation` are its only dependencies) |
| `support_validation.py` | `validate_support(values, name)`, the support validator shared by `IVModel` and `PartialMonotonicityModel` |
| `tests/test_partial_monotonicity_model.py` | pytest suite: two binary instruments, the exclusion switch, one instrument against `IVModel`, nonbinary supports, three binary instruments, invariance, validation and interface, inequalities |
| `PartialMonotonicityModel.ipynb` | The examples of the specification through `build_graph()` and `GraphAnalyzer`, with the MISs written as inequalities by `format_inequalities` |

`IVModel` is unchanged apart from `_validate_support`, which now delegates to the shared
`validate_support` (same behaviour, same error messages). The supervisor's
`Partial_Monotonicity.ipynb`, which hand-codes the two-binary-instrument case for the
treatment alone, is untouched and is used as an independent check in the tests.

## Interface

```python
model = PartialMonotonicityModel(
    y_support=(0, 1, 2),
    d_support=(0, 1, 2),
    z_supports=((0, 1), (0, 1, 2), (10, 20)),   # one support per instrument coordinate
    exclusion=True,
)
G = model.build_graph()
analyzer = GraphAnalyzer(G, group_fn=model.group_fn)
inequalities = model.format_inequalities(analyzer.find_mis_excluding_single_group())
```

- Partial monotonicity is always imposed: `D(z)` is nondecreasing in each coordinate
  holding the others fixed. Exclusion is optional and defaults to `True`. No outcome
  monotonicity is imposed. The nonincreasing direction is not implemented in this class;
  reversing the order of a coordinate's support (negate its values) is equivalent.
- Attributes: `y_support`, `d_support` (sorted tuples); `z_supports` (tuple of sorted
  tuples, coordinate order preserved); `z_support` (the Cartesian product of the
  coordinate supports as a tuple of tuples, first coordinate outermost);
  `n_instruments` (`len(z_supports)`); `exclusion` (bool); `nodes` (all `(y, d, z)`, y
  outermost, then d, then z in product order, with `z` a tuple even for one
  instrument). Settings are fixed after construction: assigning or deleting an
  attribute raises `AttributeError`.
- Methods: `_validate_z_supports(z_supports)`, `_make_nodes()`, `group_fn(node)`
  (returns `node[2]`, the full instrument vector), `_leq(z, zp)` (componentwise `<=`; vectors of different lengths raise `ValueError`),
  `violates_exclusion(u, v)`, `violates_partial_monotonicity(u, v)`,
  `violate_pairwise_fn(u, v)`, `build_graph()`, `_check_independent_set(independent_set)`,
  `_label_instrument(z)` (`"Z1=0, Z2=1"`), `format_inequality(independent_set)`,
  `format_inequalities(independent_sets)`, `summary()`, and `__repr__`
  (`eval(repr(model))` rebuilds the model).
- Validation. `z_supports` must be an iterable of at least one coordinate support;
  each coordinate support is validated by `validate_support` (nonempty, distinct,
  finite real numbers; booleans rejected; NumPy scalars converted). A flat sequence of
  numbers such as `(0, 1)` is a `TypeError` whose message says to write
  `((0, 1),)`; a string, a non-iterable, or a string coordinate is a `TypeError`;
  no coordinates, an empty coordinate, a duplicate, `nan` or `inf` is a `ValueError`.
  Messages name the coordinate as `z_supports[j]`. `exclusion` must be a `bool`.
- Inequalities (added 2026-10-27, replacing the `inequality` helper that the notebook
  used to define). `format_inequality(independent_set)` writes one independent set of
  the graph as the inequality it implies, as a string such as
  `P(D=1 | Z1=0, Z2=0) + P(D=0 | Z1=1, Z2=1) <= 1`: each node `(y, d, z)` contributes
  `P(Y=y, D=d | Z1=z[0], ..., Zm=z[m-1])`, the instrument vector spelled coordinate by
  coordinate as in `summary()` (the notebook helper wrote `Z=(0, 0)`), and when the set
  contains every outcome value for a treatment value `d` at an instrument vector `z`
  those terms merge into `P(D=d | Z1=..., Zm=...)`. Terms are ordered by instrument
  vector (product order), then `d`, then `y`. `format_inequalities(independent_sets)`
  maps it over a sequence of sets and returns a list of strings in the same order; it
  is meant for the output of `GraphAnalyzer.find_mis_excluding_single_group()`. Every
  set is validated by `_check_independent_set`: `TypeError` for a string, a
  non-iterable, an element that is not a tuple, or a tuple with an unhashable entry;
  `ValueError` for an empty set, a tuple that is not a node of the model, a duplicate
  node, or two cross-group nodes that `violate_pairwise_fn` does not reject (an edge of
  the graph, so the events are not mutually exclusive). Nodes are matched to `nodes` by
  numeric equality (`1.0`, `numpy.int64(1)` and `True` all select the node whose value
  is `1`) and the string shows the support values. A set inside a single instrument
  vector is accepted; its inequality holds trivially. Passing one set to
  `format_inequalities` instead of a sequence of sets is a `TypeError` that points to
  `format_inequality` (its first element is then a tuple that starts with a number, not
  a collection of nodes). With one instrument coordinate the
  strings equal those of `IVModel` with nondecreasing D-monotonicity after replacing
  `Z=` by `Z1=`; the tests check this.
- `summary()` lists each coordinate support, the number of instrument vectors, and the
  assumptions; it says partial monotonicity is always imposed and whether exclusion is.

## The pairwise rules and why `or` combines them correctly

For cross-group nodes `u = (y, d, z)` and `v = (y', d', z')`:

- `violates_partial_monotonicity`: `(z <= z' componentwise and d > d') or
  (z' <= z componentwise and d' > d)`. Incomparable vectors never reject. The
  componentwise order on a product support is already transitive, so implications
  obtained by chaining one-coordinate moves need no extra code; Python's lexicographic
  tuple order is never used (the test `test_componentwise_order_is_not_lexicographic`
  checks a pair ordered lexicographically but incomparable componentwise).
- `violates_exclusion`: `d == d' and y != y'`, applied whether or not the vectors are
  comparable.
- `violate_pairwise_fn`: `violates_partial_monotonicity(u, v) or (exclusion and
  violates_exclusion(u, v))`.

*Exactness.* If neither enabled rule rejects the pair, an admissible potential-response
vector generates both events. Let `d_min` be the smallest treatment value and define on
the whole instrument support

```
D(t) = max({d : z <= t} ∪ {d' : z' <= t} ∪ {d_min})      (componentwise <=).
```

The set inside the `max` grows with `t`, so `D(.)` is componentwise nondecreasing. At
`t = z` the value `d'` enters only when `z' <= z`, and then the rule gives `d' <= d`, so
`D(z) = d`; symmetrically `D(z') = d'`. This covers `z < z'`, `z' < z` and the
incomparable case. Under exclusion set `Y(d) = y` and `Y(d') = y'`, consistent because
`d = d'` forces `y = y'`; without exclusion set `Y(d, z) = y` and `Y(d', z') = y'` on
two distinct cells. Treatments and outcomes are disjoint components of the vector, so
the two completions are made simultaneously and all other components are free.

*Beyond pairs.* For pairwise compatible nodes `(y_i, d_i, z_i)` with distinct
instrument vectors (automatic in a clique, since same-z nodes are never adjacent),
`D(t) = max({d_i : z_i <= t} ∪ {d_min})` gives `D(z_i) = d_i`, because `z_j <= z_i`
implies `d_j <= d_i` for compatible pairs, and the outcomes are assigned as above. So
every maximal clique is a support point (Condition 2 of the paper), and Method 1
applies. This argument was cross-checked independently by two external reviewers on
2026-10-05.

*Perfectness* is a separate matter. With two binary instruments and binary outcome and
treatment, `GraphAnalyzer.is_perfect()` returns `True` without exclusion (5 MISs, the
five inequalities of `Partial_Monotonicity.ipynb`) and `False` with exclusion (45 MISs),
so the MIS inequalities are sharp in the first case and only necessary in the second.
The notebook reports both; the check is numerical.

## Hand-derived counts used in the tests and the notebook

| Example | Nodes | Cross-group pairs | Removed by monotonicity | Removed by exclusion | Edges without / with exclusion |
|---|---|---|---|---|---|
| Y, D binary, Z in {0,1}^2 | 16 | 96 | 5 comparable vector pairs x 4 = 20 | 6 x 4 = 24 | 76 / 52 |
| Y in {0,1,2}, D in {0,2,5}, Z in {0,1,2} x {10,20} | 54 | 1215 | 12 x 27 = 324 | 15 x 18 = 270 | 891 / 621 |
| Y, D binary, Z in {0,1}^3 | 32 | 448 | 19 x 4 = 76 | 28 x 4 = 112 | 372 / 260 |

The two removed sets are always disjoint (one needs unequal treatments, the other equal
ones). With one instrument the graph equals the `IVModel` graph with nondecreasing
D-monotonicity after dropping the tuple around `z`.

## Tests

Expected edges are written out by hand with the reason in the test file, derived by
enumerating potential-response vectors on the product support (`method2_edges`, which
never calls the pairwise rules), or compared with `IVModel` and with the hand-coded rule
of `Partial_Monotonicity.ipynb`. Run from the repository root:

```
uv run --python 3.12 --with pytest --with networkx --with numpy python -m pytest tests -q
```

Result on 2026-10-27 with Python 3.12.13, networkx 3.7, NumPy 2.5.4 and pytest 9.1.1:
657 passed (232 for `IVModel`, 219 for `PartialMonotonicityModel`, 206 for
`MediationModel`); the inequality tests of 2026-10-27 added 69 to this file. The same
657 tests pass with networkx 3.4.2 and NumPy 2.2.6, the versions of the local Jupyter
kernel.

Contents, in order:

1. Two binary instruments: nodes and the product support; the 76- and 52-edge sets from
   the hand-listed removed pairs; agreement with Method 2; the named pairs in both
   argument orders; no same-instrument edges; projection onto `(d, z)` equals the
   notebook's graph (19 edges).
2. Exclusion switch: the incomparable pair with equal treatment and unequal outcome is
   an edge only without exclusion; equal outcome and treatment at different vectors are
   compatible under both settings; exclusion removes exactly the 24 equal-treatment pairs.
3. One instrument: agreement with `IVModel` on three support sets, exclusion on and off;
   the instrument stays a 1-tuple in the nodes.
4. Nonbinary supports: the counts, agreement with Method 2, nine hand-checked pairs
   involving the third outcome value and comparable and incomparable vectors.
5. Three binary instruments: the counts, agreement with Method 2, pairs decided by the
   third coordinate, exclusion on an incomparable pair.
6. Invariance: permuted values within supports (tuples, lists, NumPy arrays, `range`)
   leave supports, nodes and edges unchanged; swapping two coordinates relabels the
   graph.
7. Validation and interface: invalid `z_supports` (flat sequence, string, non-iterable,
   no coordinates, empty coordinate, duplicate, `nan`, `inf`, booleans) with messages
   naming the coordinate; invalid outcome and treatment supports; non-Boolean
   `exclusion`; the default; positional arguments; immutability; `repr` round trip;
   Boolean predicates; a fresh graph on each call; `group_fn`; `summary()` content; no
   analyzer import (AST check); the shared validator; single-vector and single-valued
   supports; the lexicographic-order trap.
8. Inequalities: the five hand-checked monotonicity inequalities, a mixed set under
   exclusion, single-group and one-node sets; `_label_instrument` on one, two and
   three coordinates with negative and float values; term order (product order of the
   instrument vectors, then `d`, then `y`) for list, tuple, set, frozenset and
   iterator inputs; an incomparable pair under exclusion; merging only with the
   complete outcome support; every validation error through the three entry points;
   order preservation, the empty list, invalid outer inputs and a single set passed to
   `format_inequalities`; the term count of every multi-group MIS (5 and 45) against
   the number of complete outcome pairs; equality with the `IVModel` strings for one
   instrument coordinate after relabelling; numeric aliases of node values (float,
   NumPy scalar, bool) matched to the model's nodes and printed as support values;
   non-tuple nodes; and node-like tuples or unhashable nodes passed to
   `format_inequalities`.

## Notebook

`PartialMonotonicityModel.ipynb` builds the two-binary-instrument model with and without
exclusion, prints `summary()`, checks the named pairs, plots both graphs, lists the
MISs as the inequalities written by `format_inequalities` (the five inequalities of
`Partial_Monotonicity.ipynb` without exclusion, with the instrument vector spelled
coordinate by coordinate), and runs the perfectness and maximal-clique checks. It then
shows agreement with `IVModel` for one instrument, the 54-node nonbinary example with a
table of
hand-checked pairs, and the three-binary-instrument example with pairs decided by the
third coordinate. The perfectness check is not run on the 54- and 32-node graphs
because `is_perfect()` enumerates vertex subsets.

The setup cell clones the repository and checks out `REPO_REF` only when
`partial_monotonicity_model.py` is absent from the working directory (Colab);
`REPO_REF` is the branch under review, `atsuki-update_functions_documents`, to be
changed to `main` after the merge together with the Colab badge link. The notebook was
last executed in place with nbconvert on 2026-10-27 under Python 3.12.13 with
matplotlib 3.11.2, networkx 3.7, pandas 3.1.0 and NumPy 2.5.4.

## Not implemented

Per-coordinate monotonicity directions (a nonincreasing coordinate is equivalent to
negating that coordinate's support), outcome monotonicity, and restrictions across
incomparable vectors such as the Imbens-Angrist ordering `D(1, 0) >= D(0, 1)` of the
second part of `Partial_Monotonicity.ipynb`.
