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
| `tests/test_partial_monotonicity_model.py` | pytest suite: two binary instruments, the exclusion switch, one instrument against `IVModel`, nonbinary supports, three binary instruments, invariance, validation and interface |
| `PartialMonotonicityModel.ipynb` | The examples of the specification through `build_graph()` and `GraphAnalyzer` |

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
  `violate_pairwise_fn(u, v)`, `build_graph()`, `summary()`, and `__repr__`
  (`eval(repr(model))` rebuilds the model).
- Validation. `z_supports` must be an iterable of at least one coordinate support;
  each coordinate support is validated by `validate_support` (nonempty, distinct,
  finite real numbers; booleans rejected; NumPy scalars converted). A flat sequence of
  numbers such as `(0, 1)` is a `TypeError` whose message says to write
  `((0, 1),)`; a string, a non-iterable, or a string coordinate is a `TypeError`;
  no coordinates, an empty coordinate, a duplicate, `nan` or `inf` is a `ValueError`.
  Messages name the coordinate as `z_supports[j]`. `exclusion` must be a `bool`.
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

Result on 2026-10-05 with Python 3.12.13, networkx 3.7, NumPy 2.5.3 and pytest 9.1.1: 321 passed (171 for
`IVModel`, 150 for `PartialMonotonicityModel`).

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

## Notebook

`PartialMonotonicityModel.ipynb` builds the two-binary-instrument model with and without
exclusion, prints `summary()`, checks the named pairs, plots both graphs, lists the
MISs as inequalities (the five inequalities of `Partial_Monotonicity.ipynb` without
exclusion), and runs the perfectness and maximal-clique checks. It then shows agreement
with `IVModel` for one instrument, the 54-node nonbinary example with a table of
hand-checked pairs, and the three-binary-instrument example with pairs decided by the
third coordinate. The perfectness check is not run on the 54- and 32-node graphs
because `is_perfect()` enumerates vertex subsets.

The setup cell clones the repository and checks out `REPO_REF` only when
`partial_monotonicity_model.py` is absent from the working directory (Colab);
`REPO_REF` is the branch under review, `atsuki-iv_partial_monotonicity_class`, to be
changed to `main` after the merge together with the Colab badge link.

## Not implemented

Per-coordinate monotonicity directions (a nonincreasing coordinate is equivalent to
negating that coordinate's support), outcome monotonicity, and restrictions across
incomparable vectors such as the Imbens-Angrist ordering `D(1, 0) >= D(0, 1)` of the
second part of `Partial_Monotonicity.ipynb`.
