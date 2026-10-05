# Notes on the `IVModel` class

A simple discrete IV model class for the potential-response graph framework of
Kaido and Ponomarev (2025).

## Files

| File | Content |
|---|---|
| `iv_model.py` | The `IVModel` class (networkx is its only dependency) |
| `tests/test_iv_model.py` | pytest suite: binary supports, edge table, larger supports, input validation, the nonincreasing direction, then additional checks |
| `tests/conftest.py` | Puts the repository root on `sys.path`, so the tests can `import iv_model` from any working directory, as the notebook does |
| `IVModel_Exclusion_Monotonicity.ipynb` | The three specifications through `build_graph()` and the existing `GraphAnalyzer` (plot, MISs, regularity), the nonincreasing direction of D-monotonicity, and a three-valued instrument |

`GraphAnalyzer` and the rest of the library are unchanged apart from the plotting fix
described at the end, and `iv_model.py` does not import the analyzer; the notebook
imports the two separately:

```python
from iv_model import IVModel
from graph_analysis_utils import GraphAnalyzer
```

Inside a local checkout both modules are importable from the working directory. The
notebook's setup cell clones the repository, checks out the branch or commit named by
`REPO_REF` (currently `atsuki-iv_model`), and adds it to `sys.path` only when
`iv_model.py` is not found in the working directory (for example in Google Colab). The
cell after it prints where the two modules were imported from and the branch and commit
of that checkout.

## Interface

```python
model = IVModel(y_support=(0, 1), d_support=(0, 1), z_support=(0, 1),
                exclusion=True, d_monotonicity=False,
                d_monotonicity_direction="nondecreasing")
G = model.build_graph()
analyzer = GraphAnalyzer(G, group_fn=model.group_fn)
```

- Attributes: `y_support`, `d_support`, `z_support` (validated, sorted tuples),
  `exclusion`, `d_monotonicity` (bools), `d_monotonicity_direction` (`"nondecreasing"`
  or `"nonincreasing"`), `nodes` (all `(y, d, z)`, y outermost, then d, then z).
  Settings are fixed after construction: assigning or deleting an attribute raises
  `AttributeError`; build a new `IVModel` to change anything.
- Methods: `_validate_support(values, name)`, `_make_nodes()`, `group_fn(node)`,
  `violates_exclusion(u, v)`, `violates_d_monotonicity(u, v)`,
  `violate_pairwise_fn(u, v)`, `build_graph()`, `summary()`, and `__repr__`.
- Direction of D-monotonicity. `d_monotonicity_direction` is the last constructor
  argument and defaults to `"nondecreasing"`, which imposes `D(z) <= D(z')` whenever
  `z < z'` as before; `"nonincreasing"` imposes `D(z) >= D(z')` whenever `z < z'`. Equal
  treatment values are allowed in either direction. The value is validated and stored
  whether or not `d_monotonicity` is `True`, so a misspelt direction is rejected even
  when monotonicity is off, but it changes the graph only when monotonicity is imposed.
  `summary()` states the direction when monotonicity is imposed and says that the
  setting has no effect otherwise; `__repr__` includes it, so `eval(repr(model))`
  rebuilds the same model.
- Validation. `TypeError`: a string or non-iterable support, a support value that is
  not a real number (booleans and `Decimal` included), a flag that is not a `bool`
  (so `0`, `1` and `numpy.bool_` are rejected), and a direction that is not a `str`.
  `ValueError`: an empty support, a duplicate value (by numeric equality, so `1` and
  `1.0` clash), `nan` or `inf`, both flags `False`, and a direction string other than
  the two allowed values (the match is exact, so `"Nondecreasing"` and
  `"non-decreasing"` are rejected). Messages name the offending support, flag or
  direction. NumPy scalars are converted to plain Python numbers; int and float are
  otherwise kept as given.
- Supports may be given in any order and as any non-string iterable (tuple, list,
  range, NumPy array); comparisons use the numeric values. A single-valued support is
  allowed (one instrument value gives a graph with no edges).
- Not implemented: outcome monotonicity, the `build_pairwise_graph` refactor.

## The pairwise rules and why `or` combines them correctly

A node is an observable event `(y, d, z)`, and an edge between two nodes with
different instrument values means that some potential-response vector
`((Y(d, z))_{d, z}, (D(z))_z)` satisfying the enabled assumptions generates both events,
that is, `D(z) = d, Y(d, z) = y` and `D(z') = d', Y(d', z') = y'`. Method 1 decides this
from the pair alone.

*Exclusion.* With `Y(d, z) = Y(d)` for all z, two events with the same treatment fix
`Y(d)` twice, so they are compatible exactly when `y = y'`. Events with different
treatments fix different components `Y(d)` and `Y(d')`, which are free. Hence
`violates_exclusion` is `d == d' and y != y'`.

*D-monotonicity.* The two events fix `D(z) = d` and `D(z') = d'`. Under the default
nondecreasing direction, if `z < z'` and `d > d'` (or the mirror image), no
nondecreasing `D(.)` can do this. Otherwise, for `z < z'` and `d <= d'`, set `D(t) = d`
for `t < z'` and `D(t) = d'` for `t >= z'` on the whole instrument support: it is
nondecreasing and takes the required values. Hence `violates_d_monotonicity` is
`(z < z' and d > d') or (z' < z and d' > d)`. Under the nonincreasing direction the
argument is the same with the inequality reversed: the pair is impossible when `z < z'`
and `d < d'` (or the mirror image), and otherwise the same carry-forward construction
is nonincreasing, so the rule is `(z < z' and d < d') or (z' < z and d' < d)`. Equal
treatment values pass both rules.

*Why `or`.* `violate_pairwise_fn` rejects a pair when some enabled rule rejects it. This
is necessary, because each rule alone already shows that no admissible vector generates
both events. It is sufficient because the two completions act on different components
of the vector and can be carried out simultaneously: monotonicity is completed on the
treatments `(D(t))_t`, in the imposed direction, and never looks at outcomes, and
exclusion is completed on the outcomes `(Y(d))_d`, setting `Y(d) = y` and `Y(d') = y'`
when `d != d'` and the common value when `d = d'`, and never looks at treatments at
other instrument values. Without exclusion, `Y(d, z) = y` and `Y(d', z') = y'` are set
separately at the two cells `(d, z) != (d', z')`, which never conflicts. Every remaining
component is free. The constructed vector satisfies every enabled assumption and
generates both events. The test `test_method1_agrees_with_method2` confirms this by
enumerating all vectors for binary supports, for a three-valued instrument, and for the
larger supports (Y in {0, 1}, D in {0, 2, 5}, Z in {-1, 3, 10}), and
`test_method1_agrees_with_method2_nonincreasing` does the same for the nonincreasing
direction.

*Beyond pairs.* The same construction extends any set of pairwise compatible nodes, which
has at most one node per instrument value because same-z nodes are never adjacent: order
the nodes by z, carry each treatment forward to the next observed instrument value (and
the first one backward), which is monotone in the imposed direction exactly because the
pairs satisfy monotonicity, and define `Y(d)` on the treatments that appear, which is
consistent exactly because the pairs satisfy exclusion. So every clique extends to the
set of events generated by an admissible vector, and every maximal clique is a support
point. This is the pairwise incompatibility criterion (Condition 2 and Proposition 5 of
the paper; Proposition 6 proves it for exclusion), which the paper cites as the reason
Method 1 applies to this model and which is condition (ii) of its regularity
Assumption 3.

*Reflection.* Reversing the order of the instrument values swaps the two directions:
the nonincreasing model on a support `Z` has the same graph as the nondecreasing model
on `-Z = {-z : z in Z}` after relabelling each node `(y, d, z)` as `(y, d, -z)`, because
both rules depend on the instrument only through the order of its values
(`test_reflecting_instrument_support_swaps_directions`). Every graph property
established for the nondecreasing direction therefore carries over to the nonincreasing
one, including the perfectness result below.

*What this does not give.* Perfectness, condition (i) of Assumption 3, is a separate
matter. With a binary instrument the graph is bipartite and every model is regular.
With three or more instrument values (and at least two outcome and two treatment
values, so that exclusion restricts something), exclusion alone gives an imperfect
graph, the paper's leading non-regular example (Section 3.3 and Table 4): the binary
three-instrument graph contains an odd hole, and it is an induced subgraph of every
larger case, so the MIS inequalities are then necessary but not sharp. Adding D-monotonicity, in either direction, gives a
comparability graph, which is perfect (Appendix B.1), and the MIS inequalities are sharp
again. The notebook checks both numerically.

## Tests

Expected edges are written out by hand with the reason in the test file, or derived by
enumerating potential-response vectors; none come from the violation functions under
test. Run from the repository root:

```
uv run --python 3.12 --with pytest --with networkx --with numpy python -m pytest tests -q
```

Result on 2026-10-02 with Python 3.12.13, networkx 3.7, and NumPy 2.5.3:

```
........................................................................ [ 42%]
........................................................................ [ 84%]
...........................                                              [100%]
171 passed in 0.31s
```

The same 171 tests pass with networkx 3.4.2 and NumPy 2.2.6, the versions of the local
Jupyter kernel.

Contents, in order:

1. Binary supports: 8 nodes and 12 / 12 / 8 edges for the three specifications, node
   labels against the eight expected tuples, full edge sets against hand-derived lists,
   `test_binary_both_counts`, the default flags giving exclusion only,
   and no same-instrument edges.
2. Edge table: the `has_edge` table, the helper predicates on the four cross-group rows
   under every flag combination, and the combined predicate as the negation of the edge
   entry.
3. Larger supports: 18 nodes, the three example pairs under both assumptions, and equality of
   supports, nodes and edge set under permuted inputs (tuples and lists).
4. Input validation: empty, duplicate, string, `nan`, `inf` for each support; non-Boolean
   flags; both flags `False`.
5. Nonincreasing direction: the default direction reproduces the original binary edge
   sets; with exclusion off, a rising-treatment pair is allowed only under nondecreasing
   and a falling-treatment pair only under nonincreasing monotonicity, checked through
   `violates_d_monotonicity` in both argument orders; equal-treatment pairs are
   unaffected by the direction while exclusion still rejects unequal outcomes;
   hand-derived binary edge sets for the nonincreasing direction (12 edges with
   monotonicity alone and 8 with exclusion, differing from the nondecreasing sets by
   the four complier and the four defier pairs); the larger-support pairs
   `(1, 5, -1)`, `(0, 0, 10)` (allowed only under nonincreasing, with and without
   exclusion) and `(1, 0, -1)`, `(0, 5, 10)` (allowed only under nondecreasing); Method 1
   versus Method 2 under the nonincreasing direction on the three support sets with
   exclusion on and off; invalid directions rejected with the monotonicity flag on and
   off; the exclusion-only graph unchanged by the direction; `repr` round trip,
   `summary()` wording, immutability, the positional sixth argument, Boolean
   predicates, and the reflection identity.
6. Additional: Method 1 versus Method 2 on three support sets, the Table 4 counts for a
   three-valued instrument, predicates return `bool`, `build_graph` returns a fresh
   graph, immutability, `summary()` content, `group_fn`, no analyzer import in the module
   (checked on the AST), single-valued supports, positional arguments, `repr` round trip,
   rejection of booleans, complex numbers, scalars and `Decimal`, acceptance of
   `Fraction`, `range`, lists and NumPy arrays.

## Notebook

`IVModel_Exclusion_Monotonicity.ipynb` builds the three binary models, prints
`summary()`, plots each graph with `plot_grouped_on_circle`, lists the MISs and
translates them into inequalities (Pearl's instrumental inequalities for exclusion,
the propensity-score inequality for monotonicity, the Balke-Pearl / Kitagawa
inequalities plus one redundant monotonicity inequality for both), and runs
`is_perfect()` and the maximal-clique check. A section on the nonincreasing direction
repeats the counts, plots, MISs and regularity check for
`d_monotonicity_direction="nonincreasing"` with and without exclusion; the inequalities
are the mirror images of the nondecreasing ones, with the two instrument values
exchanged. A last section uses a three-valued instrument to show that exclusion alone
matches Table 4 of the paper (12 nodes, 36 edges, 28 maximal cliques, 15 MISs including
the three parts) and is not perfect, while adding D-monotonicity in either direction
restores perfectness.

The notebook runs from inside the repository (the setup cell finds `iv_model.py` and
does not clone) and in Google Colab, where the setup cell clones the repository and
checks out `REPO_REF`, set to the branch under review, `atsuki-iv_model`; change it to
`main` after the merge, together with the Colab badge link. It was executed in place
with nbconvert on 2026-10-02 under Python 3.12.13 with matplotlib 3.11.2, networkx 3.7,
pandas 3.0.6, and NumPy 2.5.3. The call to `matplotlib.cm.get_cmap` in
`plot_grouped_on_circle`, deprecated since matplotlib 3.7 and removed in 3.11, was
replaced by `matplotlib.pyplot.get_cmap(cmap_name, K)` (commit `c7e1cae`), so the plots
run on matplotlib 3.10 and 3.11; the second argument keeps one colour per instrument
group.
