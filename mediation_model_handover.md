# Notes on the `MediationModel` class

A discrete mediation model with a vector of mediators and no direct treatment effect,
for the potential-response graph framework of Kaido and Ponomarev (2025). It follows the
design of `IVModel` and `PartialMonotonicityModel` (see `iv_model_handover.md` and
`partial_monotonicity_handover.md`) and was specified in the supervisor's assignment note
"RA assignment: a discrete MediationModel" (October 2026).

## Files

| File | Content |
|---|---|
| `mediation_model.py` | The `MediationModel` class (networkx and `support_validation` are its only dependencies) |
| `support_validation.py` | `validate_support(values, name)`, the support validator shared by the three model classes; unchanged |
| `tests/test_mediation_model.py` | pytest suite: one binary mediator, two binary mediators (all selected versus `M1` only), nonbinary supports and a three-valued treatment, three binary mediators, invariance, validation and interface, inequalities |
| `MediationModel.ipynb` | The examples of the specification through `build_graph()` and `GraphAnalyzer`, with the MISs written as inequalities by `format_inequalities` |
| `README.md` | File list extended by the three new files |

The original delivery (merged as PR #5) changed no existing module or test; the update
of 2026-10-27 added `format_inequalities` to all three model classes, with tests, and
removed the `inequality` helper the notebook used to define. The notebook imports the
model and the analyzer separately:

```python
from mediation_model import MediationModel
from graph_analysis_utils import GraphAnalyzer
```

## The model

Observed outcome `Y`, assigned treatment `D` and mediator vector `M = (M_1, ..., M_J)`,
`J >= 1`, all with finite numeric supports; the support of `M` is the Cartesian product
of the coordinate supports. Potential responses `M(d)` and `Y(m, d)` generate

```
M = M(D),    Y = Y(M(D), D).
```

Maintained assumption, documented in the module and in `summary()` but not a pairwise
rule: the full vector `((M(d))_d, (Y(m, d))_{m, d})` is independent of `D`.

Restriction, always imposed: for the nonempty set `S` of selected mediator coordinates
(`active_mediators`),

```
Y(m, d) = g(m_S)    for every m and d,
```

with `g` unrestricted. By default `S` contains every coordinate: the outcome responds to
the whole mediator vector and there is no direct effect of `D` on `Y`. Selecting only
some coordinates additionally excludes the unselected coordinates from the outcome
response; it still excludes a direct treatment effect. Selection allows the selected
coordinates to affect `Y` without requiring any of them to have a nonzero effect. The
potential mediators `M(d)` are unrestricted, so an unselected mediator may still respond
to `D`; no monotonicity of any kind is imposed.

## Interface

```python
model = MediationModel(
    y_support=(0, 1),
    m_supports=((0, 1), (0, 1)),   # one support per mediator coordinate
    d_support=(0, 1),
    active_mediators="all",        # or zero-based coordinate indices, e.g. (0,)
)
graph = model.build_graph()
analyzer = GraphAnalyzer(graph, group_fn=model.group_fn)
independent_sets = analyzer.find_mis_excluding_single_group()
inequalities = model.format_inequalities(independent_sets)
```

- Attributes: `y_support`, `d_support` (sorted tuples); `m_supports` (tuple of sorted
  tuples, coordinate order preserved); `m_support` (the Cartesian product
  `tuple(product(*m_supports))`, first coordinate outermost); `n_mediators`
  (`len(m_supports)`); `active_mediators` (the resolved selection as a tuple of `int`
  in increasing order; `"all"` becomes `tuple(range(n_mediators))`); `nodes` (all
  `(y, m, d)`, y outermost, then m in product order, then d, with `m` a tuple even for
  one mediator). Settings are fixed after construction: assigning or deleting an
  attribute raises `AttributeError`.
- Methods: `_validate_m_supports(m_supports)`,
  `_validate_active_mediators(active_mediators, n_mediators)`, `_make_nodes()`,
  `_project_mediators(m)` (the selected coordinates of `m` as a tuple; a vector with
  the wrong number of coordinates raises `ValueError`), `group_fn(node)` (returns
  `node[2]`, the assigned treatment), `violates_mediation(u, v)`,
  `violate_pairwise_fn(u, v)` (delegates to `violates_mediation`; the name is the
  interface shared with the other classes and with `GraphAnalyzer`), `build_graph()`,
  `_check_independent_set(independent_set)`, `_label_mediators(m)` (`"M1=0, M2=1"`),
  `_label_selected(s)` (the selected profile with the names of the selected
  coordinates, `"M1=0, M3=1"` for `active_mediators == (0, 2)`),
  `format_inequality(independent_set)`, `format_inequalities(independent_sets)`,
  `summary()`, and `__repr__` (shows the resolved indices even when `"all"` was
  supplied; `eval(repr(model))` rebuilds the model).
- Validation of supports. `y_support`, `d_support` and every coordinate of
  `m_supports` go through `validate_support` (nonempty, distinct, finite real numbers;
  booleans rejected; NumPy scalars converted). `m_supports` must be an iterable of at
  least one coordinate support; a flat sequence of numbers such as `(0, 1)` is a
  `TypeError` whose message says to write `((0, 1),)`; a string, a non-iterable, or a
  string coordinate is a `TypeError`; no coordinates, an empty coordinate, a duplicate,
  `nan` or `inf` is a `ValueError`. Messages name the coordinate as `m_supports[j]`.
- Validation of `active_mediators`, following the error conventions of the other
  classes (wrong type: `TypeError`; right type, wrong value: `ValueError`):

  | Input | Result |
  |---|---|
  | `"all"` | every coordinate, `tuple(range(n_mediators))` |
  | any other string (`"ALL"`, `""`, `"none"`) | `ValueError` |
  | `None`, a bool, a float, bytes, any other non-iterable | `TypeError` |
  | a bare integer such as `0` | `TypeError`; the message says to write `(0,)` |
  | an element that is not an integer (`True`, `0.0`, `"0"`, `numpy.bool_`) | `TypeError` |
  | empty selection, duplicate index, index outside `range(n_mediators)` (negative indices included) | `ValueError` |
  | a list, tuple, set, generator or NumPy integer array of valid indices | accepted; NumPy integers become `int`; stored sorted, so `(1, 0)` and `(0, 1)` give the same model and the same `repr` |

- Inequalities (added 2026-10-27, replacing the `inequality(mis, model)` helper that the
  notebook used to define). `format_inequality(independent_set)` writes one
  independent set of the graph as the inequality it implies, as a string such as
  `P(Y=0, M1=0 | D=0) + P(Y=1, M1=0 | D=1) <= 1`: each node `(y, m, d)` contributes
  `P(Y=y, M1=m[0], ..., MJ=m[J-1] | D=d)`. Two merges shorten the expression. When,
  for an outcome `y` and a selected profile `s`, the set contains every mediator
  vector with that profile, the unselected coordinates are summed out and the term is
  written with the selected coordinates only (`P(Y=1, M1=0 | D=0)` under
  `active_mediators=(0,)`); with every coordinate selected this is the full label.
  When, in addition, every outcome value is present for that profile, the outcome is
  summed out too, `P(M1=0 | D=0)`. A set that covers only some vectors of a profile is
  written vector by vector for that outcome. Terms are ordered by `d`, then selected
  profile, then `y`, then `m`, in support order. `format_inequalities(independent_sets)`
  maps it over a sequence of sets and returns a list of strings in the same order; it
  is meant for the output of `GraphAnalyzer.find_mis_excluding_single_group()`. Every
  set is validated by `_check_independent_set`: `TypeError` for a string, a
  non-iterable, an element that is not a tuple, or a tuple with an unhashable entry;
  `ValueError` for an empty set, a tuple that is not a node of the model, a duplicate
  node, or two cross-treatment nodes that `violates_mediation` does not reject (an edge
  of the graph, so the events are not mutually exclusive). Nodes are matched to `nodes`
  by numeric equality (`1.0`, `numpy.int64(1)` and `True` all select the node whose
  value is `1`) and the string shows the support values. A set inside a single
  treatment value is accepted; its inequality holds trivially. Passing one set to
  `format_inequalities` instead of a sequence of sets is a `TypeError` that points to
  `format_inequality` (its first element is then a tuple that starts with a number, not
  a collection of nodes). With one mediator the strings equal
  those of the exclusion-only `IVModel` after renaming `D` to `M1` and `Z` to `D`; the
  tests check this.
- `summary()` lists the supports coordinate by coordinate, the size of the product
  support, the selected coordinates, the restriction in the form imposed (whole vector,
  or selected coordinates with the unselected ones named), that no monotonicity is
  imposed, and the maintained independence assumption.

## The pairwise rule and why it is exact

For cross-group nodes `u = (y, m, d)` and `v = (y', m', d')`, `d != d'`:

- `violates_mediation`: `m_S == m'_S and y != y'`, where `m_S` is
  `_project_mediators(m)`. Equal selected profiles must produce equal outcomes;
  different selected profiles impose nothing on the outcomes. The rule is symmetric
  and involves no ordering of any support.
- `violate_pairwise_fn`: `violates_mediation(u, v)`. Mediation is the only
  restriction, so there is nothing to combine.

*Exactness.* If the rule does not reject the pair, an admissible vector generates both
events: set `M(d) = m` and `M(d') = m'`, which are distinct components because
`d != d'` and `M(.)` is unrestricted; set `g(m_S) = y` and `g(m'_S) = y'`, which are
two different arguments of `g` when `m_S != m'_S` and the same value when `m_S = m'_S`,
since then the rule gives `y = y'`. All other components are free.

*Beyond pairs.* A clique has at most one node per treatment value, because same-`d`
nodes are never adjacent. For pairwise compatible nodes `(y_i, m_i, d_i)` with distinct
`d_i`, set `M(d_i) = m_i` and `g((m_i)_S) = y_i`; `g` is well defined because two nodes
with equal selected profiles have equal outcomes by pairwise compatibility. So every
maximal clique is a support point (Condition 2 of the paper) and Method 1 applies. The
argument was cross-checked independently by three external reviewers on 2026-10-16.

*Relation to `IVModel`.* With every coordinate selected, the rule has the form of the
exclusion rule of `IVModel` (same treatment, different outcome) with the mediator in the
role of the treatment and the treatment in the role of the instrument. The graph of
`MediationModel(Y, (M,), D)` equals the graph of `IVModel(Y, M, D, exclusion=True)`
after relabelling `(y, (m,), d)` as `(y, m, d)`; the tests check this for two support
sets. With a proper subset `S`, the graph is the all-coordinates graph of the model on
the selected coordinates alone, lifted through the projection `m -> m_S`: the tests
check that adjacency under `active_mediators=(0,)` is the adjacency of the one-mediator
graph after projecting each node.

*Perfectness* is a separate matter, checked numerically by `GraphAnalyzer`. With two
treatment values the graph is bipartite, hence perfect, and the MIS inequalities are
sharp; the notebook confirms this for all three examples. With three or more treatment
values, at least two outcome values and at least two selected profiles, the graph
contains an odd hole: one binary mediator, binary `Y` and `D` in `{0, 1, 2}` give the
exclusion-only IV graph of Table 4 of the paper (12 nodes, 36 edges, 28 maximal cliques,
15 maximal independent sets including the three single-group ones), which is an induced
subgraph of every such case, so the MIS inequalities are then necessary but not sharp.
The test suite exhibits an induced five-cycle in that graph. If `Y` is single-valued or
only one selected profile exists, the graph is a disjoint union of complete multipartite
graphs and is perfect.

## Hand-derived counts used in the tests and the notebook

With `nY = |Y|`, `nM = |m_support|`, `nD = |D|`, `nP` selected profiles and
`mult = nM / nP` mediator vectors per profile,

```
edges = C(nD, 2) * [ (nY * nM)^2 - nP * mult^2 * nY * (nY - 1) ].
```

| Example | Selected | Nodes | Cross-group pairs | Removed | Edges |
|---|---|---|---|---|---|
| Y, D binary; M in {0,1} | all | 8 | 16 | 2 x 1 x 2 = 4 | 12 |
| Y, D binary; M in {0,1}^2 | all | 16 | 64 | 4 x 1 x 2 = 8 | 56 |
| same | (0,) | 16 | 64 | 2 x 4 x 2 = 16 | 48 |
| Y in {0,1,2}; M in {0,1} x {10,20,30}; D in {0,1,2} | all | 54 | 972 | 3 x 6 x 1 x 6 = 108 | 864 |
| same | (1,) | 54 | 972 | 3 x 3 x 4 x 6 = 216 | 756 |
| Y, D binary; M in {0,1}^3 | (0, 2) | 32 | 256 | 4 x 4 x 2 = 32 | 224 |

The four cross-treatment pairs of the specification behave as required:
`(0, (0, 0), 0)` with `(1, (0, 0), 1)` is incompatible under both selections; with
`(1, (0, 1), 1)` compatible when all mediators are selected and incompatible under
`(0,)`; with `(1, (1, 0), 1)` and with `(0, (0, 1), 1)` compatible under both.

## Tests

Expected edges are written out by hand with the reason in the test file, derived by
enumerating potential-response vectors (`method2_edges`, which enumerates every map
`M(.)` and every outcome function `g` on the selected profiles and never calls the
model's rules), or compared with `IVModel`. Run from the repository root:

```
uv run --python 3.12 --with pytest --with networkx --with numpy python -m pytest tests -q
```

Result on 2026-10-27 with Python 3.12.13, networkx 3.7, NumPy 2.5.4 and pytest 9.1.1:
657 passed (232 for `IVModel`, 219 for `PartialMonotonicityModel`, 206 for
`MediationModel`); the inequality tests of 2026-10-27 added 76 to this file and 61 and
69 to the other two. The same 657 tests pass with networkx 3.4.2 and NumPy 2.2.6, the
versions of the local Jupyter kernel.

Contents, in order:

1. One binary mediator: the eight nodes in order, `m_support == ((0,), (1,))`, the
   12-edge set from the four hand-listed removed pairs, agreement with Method 2,
   equality with the exclusion graph of `IVModel` after relabelling (binary supports
   and `Y in {0,1,2}`, `M in {0,2,5}`, `D in {0,1,2}`), no same-treatment edges, `m`
   kept as a 1-tuple, bipartiteness.
2. Two binary mediators, all selected versus `(0,)`: the 56- and 48-edge sets from the
   hand-listed removed pairs (8 and 16), Method 2, the specification's pair table in
   both argument orders through `has_edge`, `violates_mediation` and
   `violate_pairwise_fn`, the subset relation and the eight-pair difference,
   `"all" == (0, 1) == [1, 0]`, `(1,)` as the mirror image of `(0,)` under a
   coordinate swap, the projection identity, bipartiteness.
3. Nonbinary supports and a three-valued treatment: the 54-node counts (864 and 756
   edges), Method 2, six hand-checked pairs in both orders, the 12-node graph's
   agreement with `IVModel` and with the Table 4 counts, and an induced five-cycle.
4. Three binary mediators with `(0, 2)`: 224 edges (240 with all selected), Method 2,
   and pairs that differ only in the unselected middle coordinate (rejected when the
   outcomes differ, compatible when they agree) or in a selected coordinate
   (compatible).
5. Invariance: permuted values inside supports (lists, tuples, NumPy arrays, `range`)
   leave supports, nodes and edges unchanged; NumPy and `range` inputs are converted to
   plain numbers; swapping two coordinates relabels the graph when the selection is
   remapped.
6. Validation and interface: every invalid `m_supports` case with messages naming the
   coordinate; every invalid `active_mediators` case of the table above; invalid
   outcome and treatment supports; the default `"all"`; positional arguments; `repr`
   round trip with the resolved selection; immutability; Boolean, symmetric predicates
   with `violate_pairwise_fn` equal to `violates_mediation` on every pair; a fresh
   graph on each call; `group_fn`; `summary()` wording for the three cases; no analyzer
   import (AST check); the shared validator; single-valued treatment, single-valued
   outcome and single-profile supports; `_project_mediators` on three coordinates and
   on vectors of the wrong length.
7. Inequalities: hand-checked strings for one binary mediator, for two binary
   mediators with all coordinates selected and with `M1` only (full and partial
   coverage of `M2`), for the `(0, 2)` selection of three mediators, and for
   single-group sets; `_label_mediators` and `_label_selected` (selected profiles keep
   the original coordinate names); term order (`d`, then selected profile, then `y`,
   then `m`) for list, tuple, set, frozenset and iterator inputs; summing out `Y` only
   with the complete outcome support; every validation error through the three entry
   points; order preservation, the empty list, invalid outer inputs and a single set
   passed to `format_inequalities`; the term count of every multi-group MIS of the
   three binary graphs; equality with the exclusion-only `IVModel` strings for one
   mediator after renaming; numeric aliases of node values (float, NumPy scalar, bool)
   matched to the model's nodes and printed as support values; non-tuple nodes; and
   node-like tuples or unhashable nodes passed to `format_inequalities`.

## Notebook

`MediationModel.ipynb` builds the one-binary-mediator model, prints `repr` and
`summary()`, plots the graph, lists the four MISs as the inequalities written by
`format_inequalities` (Pearl's instrumental inequalities with the mediator in the role
of the treatment), and runs the
perfectness and maximal-clique checks. It then builds the two-binary-mediator model with
all mediators selected and with `M1` only, prints the counts and the summary that names
the unselected coordinate, checks the four pairs of the specification in both argument
orders, plots both graphs, lists the eight and four MIS inequalities (the latter merge
the two values of `M2` and each implies two of the former, so excluding a mediator from
the outcome response strengthens the testable implications), and runs the regularity
checks. `GraphAnalyzer` is constructed separately from the graph in every case.

The setup cell clones the repository and checks out `REPO_REF` only when
`mediation_model.py` is absent from the working directory (Colab); `REPO_REF` is the
branch under review, `atsuki-update_functions_documents`, to be changed to `main`
after the merge together with the Colab badge link. The notebook was last executed in
place from a clean session with nbconvert on 2026-10-27 under Python 3.12.13 with
matplotlib 3.11.2, networkx 3.7, pandas 3.1.0 and NumPy 2.5.4.

## Not implemented

An arbitrary joint mediator support (only the Cartesian product of coordinate supports
is accepted); any direct effect of `D` on `Y`; restrictions on how unselected mediators
respond to `D`; outcome or mediator monotonicity; outcome restrictions other than the
selection of coordinates; sharp inequalities when `D` has three or more values while
`Y` and the selected profiles both take at least two values (the graph is then not
perfect, and the higher-level inequalities of the paper would be needed).
