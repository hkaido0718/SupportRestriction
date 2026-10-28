# PartialMonotonicityModel

`PartialMonotonicityModel` describes a finite instrumental-variable model with an outcome Y, a treatment D, and a vector instrument Z, imposing partial treatment
monotonicity and optional exclusion. It constructs a **potential response graph** whose nodes are observable events and whose edges link events that an admissible
potential-response vector can generate together. An **independent set** contains no linked pair; it is **maximal** if no further node can be added while preserving that
property. The workflow builds this graph, obtains its maximal independent sets, and writes the inequalities in conditional probabilities they imply: the model's testable
implications. The reference is Kaido and Ponomarev (2025), [“Testing Exclusion and Shape Restrictions in Potential Outcomes Models,”
arXiv:2512.20851](https://arxiv.org/abs/2512.20851).

## Requirements and setup

The model requires Python 3 and `networkx`. `GraphAnalyzer` also imports `pandas`, `matplotlib`, and NumPy, including when you only use its graph-analysis methods. Keep
`partial_monotonicity_model.py`, `support_validation.py`, and `graph_analysis_utils.py` importable together. Run Python from the repository root, or add that directory to
`sys.path` when working elsewhere.

```python
from partial_monotonicity_model import PartialMonotonicityModel
from graph_analysis_utils import GraphAnalyzer
```

The examples below use these imports. The quick start also repeats them so it can be run on its own. No observed dataset is needed: supports and assumptions determine the
graph.

## The model

The observed outcome Y and treatment D are scalar variables with finite numeric supports, specified by `y_support` and `d_support`. The instrument is Z = (Z1, ..., Zm),
with m >= 1. `z_supports` supplies one finite numeric support per coordinate; their Cartesian product, meaning every combination of coordinate values, is the instrument
support. For one instrument, write `z_supports=((0, 1),)`. Every instrument value in a node is a tuple, including `(0,)` or `(1,)` in this case.

The potential treatment D(z) is the treatment an individual would receive at instrument vector z. The potential outcome Y(d, z) is the outcome at treatment d and
instrument z. They generate observed data through D = D(Z) and Y = Y(D(Z), Z). Potential treatments take values in `d_support`; potential outcomes take values in
`y_support`.

The maintained independence assumption is that the full vector ((Y(d, z)) over d and z, (D(z)) over z) is independent of Z. It applies to the joint vector, not merely to
its individual components. There is no constructor switch for this assumption, and graph construction does not check it against data.

- **Partial treatment monotonicity is always imposed.** Holding all other instrument coordinates fixed, raising any coordinate cannot lower D(z). This is the partial
  monotonicity restriction of Mogstad, Torgovitsky and Walters (2021). On the product support it is equivalent to D(z) <= D(z') whenever z[j] <= z'[j] for every j: a
  componentwise increase can be reached by raising one coordinate at a time. The numeric values supplied in `d_support` and `z_supports` determine the order; there is no
  monotonicity flag or direction argument.
- **Exclusion is controlled by `exclusion`, default `True`.** It imposes Y(d, z) = Y(d, z') for every d and every pair z, z'. The instrument then affects the outcome only
  through treatment, so Y(d) is valid shorthand. With `exclusion=False`, Y(d, z) may depend freely on z.

The instrument order is **componentwise**. Python regards `(0, 1) < (1, 0)` as true because it compares tuples lexicographically, starting with the first unequal entry.
These vectors are incomparable componentwise: the first coordinate rises and the second falls. Partial monotonicity imposes no order between D(0, 1) and D(1, 0).
Exclusion can still restrict outcomes at such vectors. No outcome monotonicity is imposed.

Graph construction uses **Method 1**, the paper's procedure that decides edges by checking compatibility of pairs of observable events, without enumerating all potential
responses. The justification for this procedure in this model is given below.

## Quick start

Take binary Y and D and two binary instruments. Set `exclusion=False` explicitly to study partial monotonicity alone.

```python
from partial_monotonicity_model import PartialMonotonicityModel
from graph_analysis_utils import GraphAnalyzer

model = PartialMonotonicityModel(
    y_support=(0, 1),
    d_support=(0, 1),
    z_supports=((0, 1), (0, 1)),
    exclusion=False,
)
print(model.summary())
G = model.build_graph()
analyzer = GraphAnalyzer(G, group_fn=model.group_fn)
mis = analyzer.find_mis_excluding_single_group()
inequalities = model.format_inequalities(mis)
print(f"Nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
print(f"Maximal independent sets spanning multiple groups: {len(mis)}")
for inequality in sorted(inequalities):
    print(inequality)
print("Perfect:", analyzer.is_perfect())
```
```text
PartialMonotonicityModel: discrete IV model with a vector instrument (Kaido and Ponomarev, 2025)
  Outcome support Y:    (0, 1)
  Treatment support D:  (0, 1)
  Instrument coordinates: 2, Z = (Z1, Z2)
    Z1: (0, 1)
    Z2: (0, 1)
  Instrument support Z: Cartesian product, 4 vectors
  Nodes (y, d, z):      16
  Potential responses: D(z) and Y(d, z); observed D = D(Z), Y = Y(D(Z), Z).
  Assumptions:
    - Partial monotonicity: imposed (always). D(z) is nondecreasing in each
      instrument coordinate holding the others fixed, so D(z) <= D(z')
      whenever z <= z' componentwise; incomparable instrument vectors impose
      no ordering on D.
    - Exclusion: not imposed. Y(d, z) may vary with z at every treatment value d.
    - No outcome monotonicity is imposed.
  Maintained (not a pairwise rule): the full potential-response vector
    ((Y(d, z))_{d, z}, (D(z))_z) is independent of Z.
  Graph: build_graph() applies Method 1 (pairwise compatibility) and returns
    a networkx.Graph for GraphAnalyzer.
Nodes: 16, edges: 76
Maximal independent sets spanning multiple groups: 5
P(D=1 | Z1=0, Z2=0) + P(D=0 | Z1=0, Z2=1) <= 1
P(D=1 | Z1=0, Z2=0) + P(D=0 | Z1=1, Z2=0) <= 1
P(D=1 | Z1=0, Z2=0) + P(D=0 | Z1=1, Z2=1) <= 1
P(D=1 | Z1=0, Z2=1) + P(D=0 | Z1=1, Z2=1) <= 1
P(D=1 | Z1=1, Z2=0) + P(D=0 | Z1=1, Z2=1) <= 1
Perfect: True
```

Each group consists of nodes with the same full instrument vector. `find_mis_excluding_single_group()` returns a list of lists of nodes, retaining only maximal
independent sets that span at least two groups. It does not remove any group from the graph or require a set to meet every group. Single-group sets imply only ordinary
probability bounds and are omitted from this list.

Nodes within each returned set are sorted. The order of the sets follows NetworkX's enumeration and has no economic meaning; the example sorts the formatted strings for
display. There are five retained sets, hence five inequalities.

## Constructor and attributes

Signature: `PartialMonotonicityModel(y_support, d_support, z_supports, exclusion=True)`.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `y_support` | Iterable of real numbers | Required | Nonempty, distinct, finite outcome values. Stored sorted. |
| `d_support` | Iterable of real numbers | Required | Nonempty, distinct, finite treatment values; their numeric order defines treatment monotonicity. Stored sorted. |
| `z_supports` | Iterable of iterables of real numbers | Required | One nonempty, distinct, finite support per instrument coordinate. At least one coordinate is required. Values are sorted within each coordinate; coordinate order is preserved. |
| `exclusion` | `bool` | `True` | Whether to impose Y(d, z) = Y(d, z') for every d, z, z'. |

Lists, tuples, ranges, NumPy arrays, and other suitable iterables can supply supports. Booleans are rejected as support values. NumPy real scalars are converted to
ordinary Python numbers; Python `int` and `float` values are kept as supplied.

| Attribute | Type | Meaning |
|---|---|---|
| `y_support` | `tuple` | Validated outcome support, sorted numerically. |
| `d_support` | `tuple` | Validated treatment support, sorted numerically. |
| `z_supports` | `tuple` of `tuple`s | Validated coordinate supports, preserving the supplied coordinate order. |
| `z_support` | `tuple` of `tuple`s | All instrument vectors in Cartesian-product order, with the first coordinate outermost. |
| `n_instruments` | `int` | Number of instrument coordinates, equal to `len(z_supports)`. |
| `exclusion` | `bool` | Whether exclusion is imposed. |
| `nodes` | `tuple` of `(y, d, z)` tuples | All observable events: y outermost, then d, then z in product order. Each z is a tuple of length `n_instruments`, even for one coordinate. |

There are `len(y_support) * len(d_support) * len(z_support)` nodes. For the quick start, `z_support` is `((0, 0), (0, 1), (1, 0), (1, 1))` and `nodes[0]` is `(0, 0, (0,
0))`. Product order controls storage and display; componentwise comparisons control monotonicity.

Settings are fixed after construction. Assigning or deleting an attribute raises `AttributeError`; construct a new model to change supports or exclusion. `repr(model)` is
a constructor expression. For the quick-start model, a round trip is:

```python
rebuilt = eval(repr(model), {"PartialMonotonicityModel": PartialMonotonicityModel})
print(repr(model))
print(rebuilt.nodes == model.nodes, rebuilt.exclusion == model.exclusion)
```
```text
PartialMonotonicityModel(y_support=(0, 1), d_support=(0, 1), z_supports=((0, 1), (0, 1)), exclusion=False)
True True
```

The reconstructed object has the same specification; it is a new instance.

## Methods

Signatures below are for calls on a model instance; `self` is omitted. For pair predicates, supply nodes u = (y, d, z) and v = (y', d', z') of the model.

- **`group_fn(node) -> tuple`** returns `node[2]`, the full instrument vector. Pass this bound method as `GraphAnalyzer`'s `group_fn` to keep groups consistent with the
  model.
- **`violates_exclusion(u, v) -> bool`** returns whether d == d' and y != y'. Use it to diagnose exclusion for a pair. It evaluates this formula even when
  `exclusion=False`, and even for a same-group pair.
- **`violates_partial_monotonicity(u, v) -> bool`** returns whether comparable instrument vectors have treatments ordered against monotonicity. Incomparable vectors
  return `False`. Use it to diagnose the treatment restriction.
- **`violate_pairwise_fn(u, v) -> bool`** combines partial monotonicity with exclusion when enabled. `True` means an imposed rule rejects the pair. Pass it to
  `GraphAnalyzer.enumerate_cliques_and_check` for compatibility checks. It does not perform the separate same-group exclusion used in graph construction, so `False` alone
  does not establish an edge for arbitrary same-group inputs.
- **`build_graph() -> networkx.Graph`** constructs a fresh undirected graph by Method 1. It includes every node, including isolated ones. Call it after choosing the
  supports and assumptions. The returned graph is mutable; editing it does not alter the model or graphs returned by later calls. Use the unmodified graph for the model's
  compatibility and sharpness statements.
- **`format_inequality(independent_set) -> str`** validates a nonempty independent set and writes its inequality. It accepts iterables such as lists, tuples, sets,
  frozensets, and iterators of node tuples. Maximality is not required, and single-group sets are accepted. Graph construction need not be called first.
- **`format_inequalities(independent_sets) -> list[str]`** validates and formats each supplied set, preserving the outer order, including repeated sets. Use it on the
  analyzer's maximal independent sets. An empty outer iterable returns `[]`; passing one set instead of an iterable of sets raises `TypeError`.
- **`summary() -> str`** describes supports, assumptions, node count, and the maintained independence assumption. Use `print(model.summary())` to inspect a specification;
  the method itself does not print.

For raw node listings, `analyzer.print_mis_excluding_single_group()` prints the retained count and numbered sets and returns `None`; it does not format probability
inequalities. `analyzer.plot_grouped_on_circle(title="Graph", node_size=1000, figsize=(6, 6), cmap_name="Set2", group_gap=np.pi/15)` draws and labels nodes around a
circle, colors them by group, calls `plt.show()`, and returns `None`. For example, using the quick-start analyzer:

```python
analyzer.plot_grouped_on_circle(
    title="Two binary instruments, partial monotonicity only",
    node_size=900, figsize=(7, 7),
)
```

## Reading the inequalities

A node `(y, d, (z1, ..., zm))` contributes the term `P(Y=y, D=d | Z1=z1, ..., Zm=zm)`. These are conditional population probabilities of observed outcomes and treatment,
at the indicated instrument values. The model formats symbolic strings; it does not estimate or evaluate these probabilities.

For an independent set, the corresponding events in the potential responses are mutually exclusive. Independence from Z identifies their probabilities with the observed
conditional probabilities, so their sum is at most one. This is the `<= 1` restriction in [Theorem 1 of the paper](https://arxiv.org/html/2512.20851). An event here
describes a response at a specified instrument value; the argument is about incompatibility of potential responses across those values.

The formatter merges terms only when the set includes **every** outcome value in `y_support` for a fixed pair (d, z). That sum becomes `P(D=d | Z1=..., ..., Zm=...)`. It
does not merge across treatment values or instrument vectors. Terms appear in instrument product order, then treatment support order, then outcome support order,
regardless of the input node order. Instrument coordinates are labelled `Z1`, `Z2`, and so on, including `Z1` for a single coordinate.

**Complete outcome support.** With binary Y, both outcome values at each selected (d, z) disappear into treatment marginals. Monotonicity rules out D(0, 0) = 1 together
with D(1, 1) = 0:

```python
monotone = PartialMonotonicityModel((0, 1), (0, 1), ((0, 1), (0, 1)), exclusion=False)
events = [(y, 1, (0, 0)) for y in (0, 1)] + [(y, 0, (1, 1)) for y in (0, 1)]
print(monotone.format_inequality(events))
```
```text
P(D=1 | Z1=0, Z2=0) + P(D=0 | Z1=1, Z2=1) <= 1
```

Since D is binary, this also says P(D=1 | Z=(0, 0)) <= P(D=1 | Z=(1, 1)).

**Joint and marginal terms under exclusion.** In the first set below, only D=0 at Z=(1, 1) includes every outcome. Monotonicity excludes its coexistence with the first
node, and exclusion separates the two D=1 nodes with different outcomes. The second set uses incomparable vectors: exclusion alone separates its events.

```python
excluded = PartialMonotonicityModel((0, 1), (0, 1), ((0, 1), (0, 1)))
events = [(1, 1, (0, 0)), (0, 0, (1, 1)), (1, 0, (1, 1)), (0, 1, (1, 1))]
print(excluded.format_inequality(events))
incomparable = [(0, 0, (1, 0)), (1, 0, (0, 1))]
print(excluded.format_inequality(incomparable))
```
```text
P(Y=1, D=1 | Z1=0, Z2=0) + P(D=0 | Z1=1, Z2=1) + P(Y=0, D=1 | Z1=1, Z2=1) <= 1
P(Y=1, D=0 | Z1=0, Z2=1) + P(Y=0, D=0 | Z1=1, Z2=0) <= 1
```

**Incomplete outcome support.** When Y has three possible values, including only Y=0 and Y=1 does not permit a marginal treatment term. Adding Y=2 does. Both sets lie
within one instrument group, so these are ordinary probability bounds:

```python
three_outcomes = PartialMonotonicityModel((0, 1, 2), (0, 1), ((0, 1), (0, 1)))
events = [(0, 0, (0, 0)), (1, 0, (0, 0))]
print(three_outcomes.format_inequality(events))
print(three_outcomes.format_inequality(events + [(2, 0, (0, 0))]))
```
```text
P(Y=0, D=0 | Z1=0, Z2=0) + P(Y=1, D=0 | Z1=0, Z2=0) <= 1
P(D=0 | Z1=0, Z2=0) <= 1
```

## How the graph is built

There is one node for each (y, d, z). All nodes with the same full z form a group. Two distinct nodes within a group have no edge: one potential-response vector produces
a single (Y, D) response at that instrument vector. Across groups, an edge means that some admissible potential-response vector produces both responses.

For u = (y, d, z) and v = (y', d', z') with z != z', the rejection rules are:

| Rule | Rejected pair | Reason |
|---|---|---|
| Partial monotonicity, always | (z <= z' componentwise and d > d') or (z' <= z componentwise and d' > d) | Treatment falls along a componentwise increase in the instrument. |
| Exclusion, if enabled | d == d' and y != y' | The same Y(d) would have to take two different values. This applies to comparable and incomparable vectors. |

`build_graph()` visits each unordered pair once, skips same-group pairs, and adds an edge exactly when neither enabled rule rejects it. The rules are symmetric in u and
v. Componentwise order is transitive, so chains of coordinate increases already enter the comparison. There is no additional transitive-closure step.

To see why these rules are exact, suppose a cross-group pair passes them. Let d_min be the smallest treatment support value, and define, for every instrument vector t,

D(t) = max({d : z <= t} union {d' : z' <= t} union {d_min}),

where all vector comparisons are componentwise. Each conditional singleton contributes its treatment value only when its condition holds. As t increases, this set can
only grow, so D(t) is nondecreasing. At t=z, d is included; d' can be included only if z' <= z, in which case compatibility gives d' <= d. Thus D(z)=d, and likewise
D(z')=d'. Every assigned value remains in the treatment support.

Under exclusion, assign Y(d)=y and Y(d')=y'; the exclusion rule ensures consistency when d=d'. Without exclusion, assign Y(d, z)=y and Y(d', z')=y' in distinct cells. All
remaining outcome cells may take any outcome support value. Treatment and outcome assignments specify different components of the potential-response vector, so both can
be completed simultaneously. This proves that accepting a pair really does imply compatibility, and explains the `or` in `violate_pairwise_fn`.

A **clique** is a set of nodes with an edge between every pair; it is **maximal** if no node can be added while retaining that property. A **support point** is one
possible full potential-response vector satisfying the model's restrictions, also called an admissible response type. A clique contains at most one node per group. For
any clique with nodes (y_i, d_i, z_i), use the same construction with D(t) = max({d_i : z_i <= t} union {d_min}). Pairwise compatibility ensures D(z_i)=d_i for every i,
and the outcome assignments remain consistent. Thus every clique extends to a support point. Every maximal clique already gives the resulting responses at all instrument
vectors: otherwise an additional node from that completion could be added. It represents at least one support point; unobserved potential outcome cells can make that
representation nonunique.

This extension property is the pairwise incompatibility criterion, Condition 2 of the paper. It justifies Method 1 for this class. Independent sets have the opposite role
to cliques: their nodes are mutually incompatible, giving the probability sums above. Maximal means that a set cannot be enlarged; it does not mean that it has the
greatest cardinality among all independent sets or cliques.

## When the inequalities are sharp

**Necessary** inequalities hold for every observable distribution generated by the model. They are **sharp** if they are also sufficient: every valid conditional
probability distribution satisfying them can be generated by a distribution of admissible potential responses independent of Z. Valid probabilities must also be
nonnegative and sum to one over (y, d) separately at each instrument vector.

A **perfect graph** has no induced odd cycle of length at least five, either in the graph or its complement. An induced subgraph retains all edges among selected nodes;
the complement links exactly the distinct pairs missing an edge in the original graph. Combined with the support-point property established above, perfectness guarantees
sharp maximal independent set inequalities by [Theorem 3 of the paper](https://arxiv.org/html/2512.20851).

`analyzer.is_perfect()` returns a Boolean. It enumerates odd-sized vertex subsets of size at least five, checking for induced cycles in the graph and its complement. It
returns `False` upon finding one, and `True` if neither search finds one.

`analyzer.enumerate_cliques_and_check(model.violate_pairwise_fn)` returns a **list of dictionaries**, each with `index` (one-based enumeration index), `clique` (sorted
list of nodes), and `satisfies_condition` (Boolean). It enumerates maximal cliques and rechecks the supplied predicate on every pair within each clique. Passing these
checks confirms pairwise consistency. Joint realizability follows from the extension argument above; the method does not construct potential responses or check
perfectness.

For binary Y and D and two binary instruments, the graph is perfect without exclusion: the five inequalities are sharp. With exclusion, the graph is not perfect and there
are 45 inequalities spanning multiple groups. These are necessary; the perfect-graph sharpness guarantee does not apply. Do not use this check to claim that satisfying
all 45 suffices. The following prints the counts and three of the 45 inequalities:

```python
for exclusion in (False, True):
    checked = PartialMonotonicityModel((0, 1), (0, 1), ((0, 1), (0, 1)), exclusion=exclusion)
    graph = checked.build_graph()
    analysis = GraphAnalyzer(graph, group_fn=checked.group_fn)
    sets = analysis.find_mis_excluding_single_group()
    rows = analysis.enumerate_cliques_and_check(checked.violate_pairwise_fn)
    print(f"exclusion={exclusion}: nodes={len(graph)}, edges={graph.number_of_edges()}, "
          f"inequalities={len(sets)}, perfect={analysis.is_perfect()}")
    print(f"Maximal cliques: {len(rows)}; "
          f"all pass pairwise checks: {all(row['satisfies_condition'] for row in rows)}")
    if exclusion:
        print("First three inequalities in sorted order:")
        for inequality in sorted(checked.format_inequalities(sets))[:3]:
            print(inequality)
```
```text
exclusion=False: nodes=16, edges=76, inequalities=5, perfect=True
Maximal cliques: 96; all pass pairwise checks: True
exclusion=True: nodes=16, edges=52, inequalities=45, perfect=False
Maximal cliques: 20; all pass pairwise checks: True
First three inequalities in sorted order:
P(D=1 | Z1=0, Z2=0) + P(D=0 | Z1=0, Z2=1) <= 1
P(D=1 | Z1=0, Z2=0) + P(D=0 | Z1=1, Z2=0) <= 1
P(D=1 | Z1=0, Z2=0) + P(D=0 | Z1=1, Z2=1) <= 1
```

The number of nodes grows as the product of all support sizes. For the two larger specifications below, `is_perfect()` is impractical because it enumerates vertex
subsets, so it is deliberately not called. `enumerate_cliques_and_check()` remains usable for these examples, as the following checks show; clique enumeration can also
become expensive for larger graphs. These checks alone do not certify sharpness.

```python
specifications = [
    ("nonbinary", (0, 1, 2), (0, 2, 5), ((0, 1, 2), (10, 20))),
    ("three instruments", (0, 1), (0, 1), ((0, 1), (0, 1), (0, 1))),
]
for label, ys, ds, zs in specifications:
    for exclusion in (False, True):
        large = PartialMonotonicityModel(ys, ds, zs, exclusion=exclusion)
        graph = large.build_graph()
        analysis = GraphAnalyzer(graph, group_fn=large.group_fn)
        rows = analysis.enumerate_cliques_and_check(large.violate_pairwise_fn)
        print(f"{label}, exclusion={exclusion}: nodes={len(graph)}, "
              f"edges={graph.number_of_edges()}, "
              f"clique checks={all(row['satisfies_condition'] for row in rows)}")
```
```text
nonbinary, exclusion=False: nodes=54, edges=891, clique checks=True
nonbinary, exclusion=True: nodes=54, edges=621, clique checks=True
three instruments, exclusion=False: nodes=32, edges=372, clique checks=True
three instruments, exclusion=True: nodes=32, edges=260, clique checks=True
```

## Input validation and errors

The constructor validates supports and `exclusion`. The formatters validate node membership, uniqueness, and independence. Error messages below are representative
messages from the implementation; coordinate indices in `z_supports[j]` are zero-based.

| Argument | Exception | Invalid input | Representative message |
|---|---|---|---|
| `y_support` | `TypeError` | String or non-iterable; a non-real value, including a Boolean | `y_support values must be real numbers, got True of type bool` |
| `y_support` | `ValueError` | Empty, duplicate, NaN, or infinite values | `y_support must not be empty` |
| `d_support` | `TypeError` | String or non-iterable; a non-real value, including a Boolean | `d_support must be an iterable of numbers, got 3 of type int` |
| `d_support` | `ValueError` | Empty, duplicate, NaN, or infinite values | `d_support values must be finite, got nan` |
| Outer `z_supports` | `TypeError` | String or non-iterable | `z_supports must be a sequence of coordinate supports, not a string: '01'` |
| Outer `z_supports` | `ValueError` | No instrument coordinates | `z_supports must contain at least one instrument coordinate` |
| `z_supports[j]` | `TypeError` | A string or non-iterable coordinate, including the numbers in a flat `(0, 1)` | `z_supports[0] must be an iterable of numbers (one support per instrument coordinate), got 0 of type int; for a single instrument write z_supports=((0, 1),)` |
| Values in `z_supports[j]` | `TypeError` | A non-real value, including a Boolean | `z_supports[0] values must be real numbers, got True of type bool` |
| `z_supports[j]` | `ValueError` | Empty, duplicate, NaN, or infinite values | `z_supports[1] must not contain duplicate values: (2, 2)` |
| `exclusion` | `TypeError` | Anything other than a Python `bool`, including `0`, `1`, or a NumPy Boolean | `exclusion must be a bool (True or False), got 1 of type int` |
| Predicate arguments u, v | `ValueError` | Instrument vectors of unequal lengths in a partial-monotonicity comparison | `instrument vectors must have the same number of coordinates, got (0, 0) and (1,)` |
| `independent_set` | `TypeError` | String or non-iterable | `an independent set must be an iterable of nodes (y, d, z), not a string: 'abc'` |
| A node in `independent_set` | `TypeError` | Not a tuple, or a tuple containing an unhashable entry such as a list | `nodes must be hashable tuples (y, d, z), got (0, 0, [0, 0]) with an unhashable entry` |
| `independent_set` | `ValueError` | Empty | `an independent set must contain at least one node` |
| A node in `independent_set` | `ValueError` | Tuple absent from the model's nodes | `(2, 0, (0, 0)) is not a node (y, d, z) of this model` |
| `independent_set` | `ValueError` | Duplicate nodes | `an independent set must not contain duplicate nodes: [(0, 0, (0, 0)), (0, 0, (0, 0))]` |
| `independent_set` | `ValueError` | A compatible cross-group pair, hence an edge | `(0, 0, (0, 0)) and (0, 1, (1, 1)) are adjacent in the potential response graph, so the set is not independent and its events are not mutually exclusive` |
| Outer `independent_sets` | `TypeError` | String or non-iterable | `independent_sets must be an iterable of independent sets, not a string: 'abc'` |
| Outer `independent_sets` | `TypeError` | One independent set supplied where a collection of sets is required | `format_inequalities expects an iterable of independent sets but received node-like tuples such as (0, 0, (0, 0)); to format one independent set call format_inequality` |

Each inner set passed to `format_inequalities` is subject to the same validation as `format_inequality`. Nodes match by numeric equality: `1.0`, `numpy.int64(1)`, and
`True` can select a node whose stored value is `1`, even though Boolean constructor support values are rejected. Output uses the stored support values, and numeric
aliases of the same node count as duplicates. Predicate methods expect well-formed nodes; they do not perform the formatters' full node-membership validation.

Attempting assignment or deletion after construction raises `AttributeError`, for example: `PartialMonotonicityModel settings are fixed after construction; cannot set
'exclusion'. Create a new PartialMonotonicityModel instead.`

## Relation to the other model classes

All three classes construct potential response graphs and format independent sets as inequalities, with graph analysis delegated to `GraphAnalyzer`.

| Class and documentation | Model and assumptions | Nodes and groups |
|---|---|---|
| [`IVModel`](iv_model.md) | Scalar instrument; optional exclusion and treatment monotonicity, with at least one enabled. Monotonicity can be nondecreasing or nonincreasing. | `(y, d, z)`, with scalar z; groups are instrument values. |
| [`PartialMonotonicityModel`](partial_monotonicity_model.md) | Vector instrument; componentwise nondecreasing treatment monotonicity always imposed, exclusion optional. | `(y, d, z)`, with tuple z; groups are instrument vectors. |
| [`MediationModel`](mediation_model.md) | Assigned treatment D, mediator vector M, and outcome Y; Y(m, d) = g(m_S) for selected mediator coordinates S, excluding a direct treatment effect. Potential responses are independent of D; no monotonicity is imposed. | `(y, m, d)`, with tuple m; groups are assigned treatment values. |

With one instrument coordinate, this class reproduces `IVModel` with nondecreasing treatment monotonicity. Drop the tuple around z to compare graphs, and replace `Z=` by
`Z1=` in the scalar model's strings to compare inequalities. This runnable check uses binary supports and both exclusion settings; `iv_model.py` must also be importable:

```python
from iv_model import IVModel

for exclusion in (False, True):
    vector = PartialMonotonicityModel((0, 1), (0, 1), ((0, 1),), exclusion=exclusion)
    scalar = IVModel((0, 1), (0, 1), (0, 1), exclusion=exclusion,
                     d_monotonicity=True, d_monotonicity_direction="nondecreasing")
    vg, sg = vector.build_graph(), scalar.build_graph()
    projected_edges = {frozenset((y, d, z[0]) for y, d, z in edge) for edge in vg.edges()}
    scalar_edges = {frozenset(edge) for edge in sg.edges()}
    vector_sets = GraphAnalyzer(vg, vector.group_fn).find_mis_excluding_single_group()
    scalar_sets = GraphAnalyzer(sg, scalar.group_fn).find_mis_excluding_single_group()
    vector_strings = sorted(vector.format_inequalities(vector_sets))
    scalar_strings = sorted(s.replace("Z=", "Z1=") for s in scalar.format_inequalities(scalar_sets))
    print(f"exclusion={exclusion}: same edges={projected_edges == scalar_edges}, "
          f"same inequalities={vector_strings == scalar_strings}")
```
```text
exclusion=False: same edges=True, same inequalities=True
exclusion=True: same edges=True, same inequalities=True
```

## Limitations

- The supports are finite and numeric. Instrument supports are Cartesian products; arbitrary joint supports with missing combinations are outside this interface.
- Partial treatment monotonicity is always nondecreasing in every coordinate. There is no per-coordinate direction option. To model a coordinate with a nonincreasing
  effect, recode it as its negative, including its support and the instrument values used to interpret the probabilities. Reversing the order of a supplied list has no
  effect because supports are sorted numerically.
- Outcomes have no monotonicity restrictions. The class does not impose additional treatment orderings across incomparable vectors, such as D(1, 0) >= D(0, 1).
- The maintained independence assumption is part of the model interpretation; there is no data-based independence check or covariate-conditioning interface.
- The output consists of graphs and symbolic necessary inequalities, with a sharpness guarantee when the stated conditions hold. The class does not estimate conditional
  probabilities, compute test statistics or p-values, or generate the additional inequalities that may be needed when the graph is not perfect.
- The graph visits all unordered node pairs. The numbers of maximal independent sets and maximal cliques can be large, and the analyzer materializes their lists;
  `is_perfect()` additionally searches vertex subsets exhaustively.

## Further examples

[`PartialMonotonicityModel.ipynb`](../PartialMonotonicityModel.ipynb) works through the two-binary-instrument model with and without exclusion, diagnoses particular
pairs, plots the graphs, prints inequalities, and checks perfectness and clique compatibility. It also compares a single instrument with `IVModel` and illustrates
nonbinary supports and three binary instrument coordinates.

Run the test suite from the repository root with `uv run --python 3.12 --with pytest --with networkx --with numpy python -m pytest tests -q`.
