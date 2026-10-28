# MediationModel

`MediationModel` describes a finite discrete mediation model in which an assigned treatment affects an outcome only through selected coordinates of a mediator vector. The workflow constructs a **potential response graph**, whose nodes represent observable events and whose edges connect events that one admissible potential-response vector can generate. It then finds maximal independent sets—an independent set has no edges between its nodes, and a maximal one cannot be enlarged—and writes the inequalities in conditional probabilities they imply, which are the model's testable implications. The framework is from Kaido and Ponomarev (2025), [“Testing Exclusion and Shape Restrictions in Potential Outcomes Models,” arXiv:2512.20851](https://arxiv.org/abs/2512.20851).

## Requirements and setup

Use Python 3 and `networkx` for the model. `GraphAnalyzer` also imports `pandas`, `matplotlib`, and NumPy. Make `mediation_model.py`, `support_validation.py`, and `graph_analysis_utils.py` importable. Run the examples from the repository root, or add that directory to `sys.path` before importing:

```python
from mediation_model import MediationModel
from graph_analysis_utils import GraphAnalyzer
```

The model and analyzer are separate classes. The model builds the graph and formats inequalities; the analyzer finds sets of nodes, checks graph properties, and plots. The examples below can be run in order in one Python session.

## The model

The observed variables are an outcome `Y`, an assigned treatment `D`, and a mediator vector `M = (M1, ..., MJ)`, with `J >= 1`. A support is the finite set of allowed values. `y_support` and `d_support` specify the scalar supports; `m_supports` supplies one support per mediator coordinate. The support of `M` is their Cartesian product: every combination of coordinate values is included. All support values must be distinct finite real numbers; booleans are rejected.

The potential mediators `M(d)` describe the mediator vector under treatment `d`. The potential outcome `Y(m, d)` describes the outcome under mediator vector `m` and treatment `d`. They generate observations through `M = M(D)` and `Y = Y(M(D), D)`.

The maintained independence assumption is `((M(d))_d, (Y(m, d))_{m,d}) independent of D`: the entire vector of potential responses is independent of assigned treatment. This is always assumed, with no constructor switch. It is recorded in `summary()` but is neither checked from data nor implemented as an edge rejection rule. It does not assert independence between the potential mediators and potential outcomes.

For the nonempty coordinate selection `S = active_mediators`, the outcome restriction is `Y(m, d) = g(m_S)` for every `m` and `d`, where `m_S` is the selected subvector. The unrestricted function `g` can differ across individuals; this is not a claim that the observed outcome is a common deterministic function of observed mediators.

- **No direct treatment effect:** `Y(m, d) = Y(m, d')` for all `m, d, d'`. This is always imposed; no argument turns it off.
- **Only selected coordinates enter the outcome response:** if `m_S = m'_S`, then `Y(m, d) = Y(m', d')`. With `active_mediators="all"`, every coordinate may enter `g`. A subset such as `(0,)` additionally excludes all but the first coordinate. Selection permits an effect; it does not require a nonzero effect.
- **Unrestricted mediator responses:** `M(d)` may be any vector in `m_support` for each `d`. Unselected mediators may still respond to treatment. There is no monotonicity restriction on either mediator or outcome responses and no constructor argument for one.

`build_graph()` uses **Method 1**: it decides compatibility one pair of observable events at a time, without enumerating all potential-response vectors. The construction is justified in [How the graph is built](#how-the-graph-is-built).

## Quick start

With binary `Y`, binary `D`, and one binary mediator, construct the model, describe it, build its graph, and format the maximal independent sets that span more than one treatment value. Sorting the strings makes the displayed order independent of the analyzer's enumeration order.

```python
from mediation_model import MediationModel
from graph_analysis_utils import GraphAnalyzer

model = MediationModel(
    y_support=(0, 1), m_supports=((0, 1),), d_support=(0, 1)
)
print(model.summary())
graph = model.build_graph()
print(f"Nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")
analyzer = GraphAnalyzer(graph, group_fn=model.group_fn)
independent_sets = analyzer.find_mis_excluding_single_group()
inequalities = model.format_inequalities(independent_sets)
print(f"Inequalities: {len(inequalities)}")
print(*sorted(inequalities), sep="\n")
print(f"Perfect: {analyzer.is_perfect()}")
```
```text
MediationModel: discrete mediation model with a mediator vector (Kaido and Ponomarev, 2025)
  Outcome support Y:    (0, 1)
  Treatment support D:  (0, 1)
  Mediator coordinates: 1, M = (M1,)
    M1: (0, 1)
  Mediator support M:   Cartesian product, 2 vectors
  Selected mediators:   (0,) -> M1
  Nodes (y, m, d):      8
  Potential responses: M(d) and Y(m, d); observed M = M(D), Y = Y(M(D), D).
  Assumptions:
    - Mediation: imposed (always). Y(m, d) = g(m): the outcome responds to
      the whole mediator vector, so there is no direct effect of D on Y.
    - Every mediator coordinate is selected; none is excluded from the
      outcome response.
    - No monotonicity is imposed; the potential mediators M(d) are
      unrestricted.
  Maintained (not a pairwise rule): the full potential-response vector
    ((M(d))_d, (Y(m, d))_{m, d}) is independent of D.
  Graph: build_graph() applies Method 1 (pairwise compatibility) and returns
    a networkx.Graph for GraphAnalyzer.
Nodes: 8, edges: 12
Inequalities: 4
P(Y=0, M1=0 | D=0) + P(Y=1, M1=0 | D=1) <= 1
P(Y=0, M1=1 | D=0) + P(Y=1, M1=1 | D=1) <= 1
P(Y=1, M1=0 | D=0) + P(Y=0, M1=0 | D=1) <= 1
P(Y=1, M1=1 | D=0) + P(Y=0, M1=1 | D=1) <= 1
Perfect: True
```

These are the four Pearl instrumental inequalities, with the mediator in the role of the IV treatment and assigned treatment in the role of the instrument. Here they give a complete characterization of compatible conditional distributions, together with nonnegativity and probabilities summing to one within each treatment group. See [When the inequalities are sharp](#when-the-inequalities-are-sharp) for the general boundary of this guarantee.

## Constructor and attributes

The signature is `MediationModel(y_support, m_supports, d_support, active_mediators="all")`. The mediator supports come **before** the treatment support.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `y_support` | Iterable of real numbers | Required | Nonempty, distinct, finite outcome values; stored sorted. |
| `m_supports` | Iterable of iterables of real numbers | Required | One nonempty numeric support per mediator coordinate. Coordinate order is preserved; values within each support are sorted. For one mediator, use `((0, 1),)`, not `(0, 1)`. |
| `d_support` | Iterable of real numbers | Required | Nonempty, distinct, finite assigned-treatment values; stored sorted. |
| `active_mediators` | `"all"` or iterable of integers | `"all"` | Nonempty set of distinct zero-based coordinate indices. For the first mediator alone use `(0,)`. The resolved indices are stored sorted. |

Lists, tuples, sets, generators, ranges, and suitable NumPy arrays can supply supports or index selections. Coordinate order matters, so use an ordered outer iterable for `m_supports`. NumPy numeric support scalars become Python scalars, and NumPy integer indices become Python `int` values.

| Attribute | Type | Meaning |
|---|---|---|
| `y_support` | `tuple` | Sorted outcome support. |
| `m_supports` | `tuple` of `tuple` | Sorted coordinate supports in the supplied coordinate order. |
| `d_support` | `tuple` | Sorted treatment support. |
| `m_support` | `tuple` of `tuple` | Cartesian-product mediator vectors, with the first coordinate outermost. |
| `n_mediators` | `int` | Number of mediator coordinates, `len(m_supports)`. |
| `active_mediators` | `tuple` of `int` | Increasing selected indices; `"all"` resolves to `tuple(range(n_mediators))`. |
| `nodes` | `tuple` of `(y, m, d)` tuples | Length `len(y_support) * len(m_support) * len(d_support)`. Order: `y` outermost, then `m` in product order, then `d`. Every `m` is a tuple of length `n_mediators`, even with one mediator. |

Settings are fixed after construction. Assigning or deleting attributes raises `AttributeError`; construct another model to change supports or selected coordinates. `repr(model)` includes the resolved indices and can reconstruct the same specification with `MediationModel` in scope. Continuing the quick start:

```python
rebuilt = eval(repr(model), {"MediationModel": MediationModel})
print(repr(model))
print(rebuilt.nodes == model.nodes)
print(repr(rebuilt) == repr(model))
```
```text
MediationModel(y_support=(0, 1), m_supports=((0, 1),), d_support=(0, 1), active_mediators=(0,))
True
True
```

## Methods

Signatures below omit `self`. Node arguments use the model's `(y, m, d)` convention. A clique, used by the analyzer's checks, is a set of nodes with an edge between every pair.

| Method | Returns | When to call it |
|---|---|---|
| `group_fn(node)` | The treatment value `node[2]` | Pass to `GraphAnalyzer` to group nodes by assigned treatment. |
| `violates_mediation(u, v)` | `bool`: equal selected profiles and unequal outcomes | Inspect why a pair is forbidden by mediation. This is the class's only `violates_*` predicate. |
| `violate_pairwise_fn(u, v)` | The same `bool` as `violates_mediation(u, v)` | Supply the common pairwise-rule interface to the analyzer's clique check. |
| `build_graph()` | A fresh undirected `networkx.Graph` | Construct the potential response graph, including nodes with no edges. Mutating the returned graph does not change the model or later graph builds. |
| `format_inequality(independent_set)` | One inequality as a `str` | Format a nonempty iterable of distinct node tuples forming an independent set. Maximality is not required. |
| `format_inequalities(independent_sets)` | `list[str]` in the input order | Format an iterable of independent sets, normally the analyzer's output. An empty outer iterable returns `[]`; pass a single set to the singular method. |
| `summary()` | A multiline `str` | Read the supports, selected coordinates, assumptions, node count, and graph workflow; use `print(model.summary())` to display it. |

The two pairwise predicates are symmetric. They are intended for nodes with different treatment values, but they evaluate the same profile/outcome formula for same-treatment pairs. A `False` result alone does not establish an edge: `build_graph()` separately excludes every same-treatment pair. These predicates assume valid nodes rather than validating full node membership; a mediator vector of the wrong length raises `ValueError`.

Both formatters validate membership, duplicate nodes, and independence. They accept list, tuple, set, frozenset, or iterator inputs, provided the nodes themselves are hashable tuples. A set confined to one treatment group is accepted and gives a trivial probability bound. Matching uses numeric equality: for example, `1.0`, `numpy.int64(1)`, or `True` can match a node value `1`, even though Boolean support values are rejected at construction. Output uses the model's stored support values.

For the companion analyzer:

- `find_mis_excluding_single_group()` returns a list of sorted node lists. It enumerates maximal independent sets, then retains those containing **more than one distinct group label**. A retained set need not meet every treatment group. “Maximal” means no node can be added while preserving independence, not that the set has the largest cardinality.
- `print_mis_excluding_single_group()` prints their count and numbered node lists and returns `None`; it does not print probability inequalities. Use the model's formatter for those. The analyzer does not explicitly sort the outer list of sets.
- `plot_grouped_on_circle(title="Graph", node_size=1000, figsize=(6, 6), cmap_name="Set2", group_gap=np.pi/15)` displays a labeled circular plot, grouping and coloring nodes by treatment. It calls Matplotlib's `show()` and returns `None`, not a figure.

Continuing the quick start, plot the graph with:

```python
analyzer.plot_grouped_on_circle(title="One binary mediator", node_size=1600)
```

The analyzer's `is_perfect()` and `enumerate_cliques_and_check(...)` methods are described under [When the inequalities are sharp](#when-the-inequalities-are-sharp).

## Reading the inequalities

A node `(y, m, d)` contributes the conditional probability `P(Y=y, M1=m[0], ..., MJ=m[J-1] | D=d)`. The condition is the assigned treatment value, not a mediator value. The corresponding potential-response event is `M(d)=m` and `g(m_S)=y`. Independence from `D` identifies its probability with the displayed conditional probability.

For an independent set, these potential-response events are mutually exclusive under the model, even when their displayed probabilities condition on different treatment values. Their probabilities therefore sum to at most one: `sum P(Y=y, M=m | D=d) <= 1`. This is the implication in [Theorem 1 of Kaido and Ponomarev](https://arxiv.org/html/2512.20851v1). These are necessary restrictions: every distribution compatible with the model must satisfy them.

The formatter applies two merges separately within each treatment value and selected profile:

1. For a fixed outcome, if every mediator vector with that selected profile is present, it sums out unselected coordinates and prints only the selected labels. With all coordinates selected, this is already the full mediator label.
2. If that full coverage holds for every value in `y_support`, it also sums out `Y` and prints `P(M_S=s | D=d)`.

Incomplete coverage is written vector by vector for that outcome; even the presence of every outcome at a single vector does not trigger outcome merging if other vectors of the profile are missing. Terms are ordered by `d`, selected profile, `y`, then full mediator vector, all in sorted support order. Selected labels retain their original coordinate numbers.

**Two binary mediators: all selected versus only the first.** The same supports give eight inequalities when both coordinates may affect the outcome, and four when only `M1` may do so:

```python
two_supports = dict(
    y_support=(0, 1), m_supports=((0, 1), (0, 1)), d_support=(0, 1)
)
for selection in ("all", (0,)):
    two = MediationModel(**two_supports, active_mediators=selection)
    two_graph = two.build_graph()
    two_analyzer = GraphAnalyzer(two_graph, group_fn=two.group_fn)
    texts = two.format_inequalities(two_analyzer.find_mis_excluding_single_group())
    print(f"Selected {two.active_mediators}: {len(texts)} inequalities, "
          f"{two_graph.number_of_edges()} edges")
    print(*sorted(texts), sep="\n")
```
```text
Selected (0, 1): 8 inequalities, 56 edges
P(Y=0, M1=0, M2=0 | D=0) + P(Y=1, M1=0, M2=0 | D=1) <= 1
P(Y=0, M1=0, M2=1 | D=0) + P(Y=1, M1=0, M2=1 | D=1) <= 1
P(Y=0, M1=1, M2=0 | D=0) + P(Y=1, M1=1, M2=0 | D=1) <= 1
P(Y=0, M1=1, M2=1 | D=0) + P(Y=1, M1=1, M2=1 | D=1) <= 1
P(Y=1, M1=0, M2=0 | D=0) + P(Y=0, M1=0, M2=0 | D=1) <= 1
P(Y=1, M1=0, M2=1 | D=0) + P(Y=0, M1=0, M2=1 | D=1) <= 1
P(Y=1, M1=1, M2=0 | D=0) + P(Y=0, M1=1, M2=0 | D=1) <= 1
P(Y=1, M1=1, M2=1 | D=0) + P(Y=0, M1=1, M2=1 | D=1) <= 1
Selected (0,): 4 inequalities, 48 edges
P(Y=0, M1=0 | D=0) + P(Y=1, M1=0 | D=1) <= 1
P(Y=0, M1=1 | D=0) + P(Y=1, M1=1 | D=1) <= 1
P(Y=1, M1=0 | D=0) + P(Y=0, M1=0 | D=1) <= 1
P(Y=1, M1=1 | D=0) + P(Y=0, M1=1 | D=1) <= 1
```

In the second selection, `M2` is summed out. Each of its four inequalities implies two of the eight all-selected inequalities: fixing either value of `M2` gives smaller or equal probability terms. For example, the bound involving `Y=0, M1=0` at `D=0` and `Y=1, M1=0` at `D=1` bounds both corresponding expressions with `M2=0` and with `M2=1`. Excluding a mediator from the outcome response thus strengthens the testable implications, despite producing fewer inequalities.

**Partial coverage of a selected profile.** This hand-built independent set covers both `M2` values for `Y=0, D=0`, but only `M2=0` for `Y=1, D=1`. The second term must retain the full vector:

```python
first_only = MediationModel(**two_supports, active_mediators=(0,))
partial_set = [
    (0, (0, 0), 0),
    (0, (0, 1), 0),
    (1, (0, 0), 1),
]
print(first_only.format_inequality(partial_set))
```
```text
P(Y=0, M1=0 | D=0) + P(Y=1, M1=0, M2=0 | D=1) <= 1
```

This set is independent but not maximal; the formatter accepts it. Including `(1, (0, 1), 1)` would complete the second profile and permit summing out `M2` there too.

**Three mediators with the first and third selected.** The first call sums out the unselected middle coordinate. The second also includes every outcome value within one treatment group, so it sums out `Y` and gives a trivial bound:

```python
three = MediationModel(
    (0, 1), ((0, 1), (0, 1), (0, 1)), (0, 1), active_mediators=(0, 2)
)
across_treatments = [
    (y, (0, m2, 1), d) for y, d in ((0, 0), (1, 1)) for m2 in (0, 1)
]
within_treatment = [
    (y, (0, m2, 1), 0) for y in (0, 1) for m2 in (0, 1)
]
print(three.format_inequality(across_treatments))
print(three.format_inequality(within_treatment))
```
```text
P(Y=0, M1=0, M3=1 | D=0) + P(Y=1, M1=0, M3=1 | D=1) <= 1
P(M1=0, M3=1 | D=0) <= 1
```

## How the graph is built

Every `(y, m, d)` combination is a node, and nodes with the same treatment value form a group. There are no edges within a group: a potential-response vector produces only one mediator/outcome pair at a given treatment. Across groups, an edge means that the two events can be generated together by an admissible potential-response vector.

For `u = (y, m, d)` and `v = (y', m', d')` with `d != d'`, mediation forbids the pair precisely when `m_S = m'_S` and `y != y'`. Equivalently, an edge exists precisely when `d != d'` and either `m_S != m'_S` or `y = y'`. Equal selected profiles must have the same outcome; different selected profiles impose no relation on their outcomes. No ordering of support values enters this rule.

The pairwise rule is exact. For a pair it permits, set `M(d)=m` and `M(d')=m'`. These assignments concern distinct, unrestricted components. Set `g(m_S)=y` and `g(m'_S)=y'`. Different profiles give different arguments of `g`; equal profiles have equal outcomes by the rule. Complete the other potential responses arbitrarily within their supports. This constructs an admissible vector generating both events. A forbidden pair cannot be generated, because it would assign two different values to one argument of `g`.

A clique has an edge between every pair of its nodes; a maximal clique cannot be enlarged. A **support point** is an admissible realization of the full vector of potential responses. It determines one observable response at each treatment value; distinct support points can generate the same pattern of nodes. Every clique has at most one node per treatment group. For all its nodes `(y_i, m_i, d_i)`, jointly set `M(d_i)=m_i` and `g((m_i)_S)=y_i`. Pairwise compatibility makes `g` well defined, and the remaining responses can be completed. A maximal clique must already cover every treatment group, since a completion supplies a compatible node for any missing group. Thus every maximal clique represents at least one support point. This verifies Condition 2, the paper's pairwise incompatibility criterion, for this class.

By contrast, an independent set consists of pairwise unconnected nodes, so no response pattern can realize two of its events. Extending it to a maximal independent set yields an inequality at least as strong, since each added probability is nonnegative. Single-group maximal independent sets give the ordinary within-treatment probability bound; the analyzer excludes them from its inequality workflow.

For counts, let `nY = len(y_support)`, `nM = len(m_support)`, `nD = len(d_support)`, and `nP` be the number of selected profiles. Cartesian-product supports give the same number `mult = nM / nP` of full vectors per selected profile. With `C(nD, 2) = nD * (nD - 1) / 2`,

`edges = C(nD, 2) * [(nY * nM)^2 - nP * mult^2 * nY * (nY - 1)]`.

Each pair of groups starts with `(nY * nM)^2` pairs. The subtracted term counts equal-profile mediator pairs with unequal outcomes. Here is a check for one binary mediator and for a nonbinary two-mediator model with only the second selected:

```python
from math import comb

count_models = [
    MediationModel((0, 1), ((0, 1),), (0, 1)),
    MediationModel((0, 1, 2), ((0, 1), (10, 20, 30)), (0, 1, 2), (1,)),
]
for counted in count_models:
    nY, nM, nD = len(counted.y_support), len(counted.m_support), len(counted.d_support)
    nP = len({tuple(m[j] for j in counted.active_mediators) for m in counted.m_support})
    mult = nM // nP
    predicted = comb(nD, 2) * ((nY * nM)**2 - nP * mult**2 * nY * (nY - 1))
    actual = counted.build_graph().number_of_edges()
    print(f"Nodes: {len(counted.nodes)}, formula: {predicted}, graph: {actual}")
    assert predicted == actual
```
```text
Nodes: 8, formula: 12, graph: 12
Nodes: 54, formula: 756, graph: 756
```

## When the inequalities are sharp

A **perfect graph** is one for which every induced subgraph—a subset of nodes retaining all edges between them—can be colored with as many colors as the size of its largest clique, with adjacent nodes receiving different colors. Together with the support-point property proved above, perfectness makes the maximal independent set inequalities **sharp**: along with nonnegativity and within-treatment normalization, they are necessary and sufficient for a conditional distribution to come from this model. See [Theorem 3 of Kaido and Ponomarev](https://arxiv.org/html/2512.20851v1).

`GraphAnalyzer.is_perfect()` returns a Boolean. It searches both the graph and its complement (which joins distinct nodes exactly when the original graph does not) for odd holes: induced cycles of odd length at least five. An odd hole in the complement is an odd antihole in the original graph. The check is exhaustive over odd-sized node subsets, so it can be expensive even for moderately sized graphs.

For this class, the cases are:

- With two treatment values, the graph is bipartite: its nodes split into two groups with no within-group edges. It is therefore perfect, and the inequalities are sharp. With one treatment value there are no edges and no multi-group inequalities.
- With three or more treatment values, at least two outcome values, and at least two selected profiles, the graph contains an odd hole and the inequalities are necessary but not sharp. The nodes with two outcome values, one full vector from each of two distinct selected profiles, and three treatment values induce the same 12-node graph as the binary-mediator example below, which contains an induced five-cycle.
- With only one outcome value, the graph is complete multipartite: every pair from distinct treatment groups is adjacent. With only one selected profile, it is a disjoint union of such graphs, one for each outcome. These graphs are perfect for any number of treatment values.

`enumerate_cliques_and_check(violate_pairwise_fn)` returns a **list of dictionaries**, with keys `index` (one-based enumeration number), `clique` (sorted node list), and `satisfies_condition` (Boolean). It enumerates maximal cliques and checks that no pair violates the supplied predicate. Passing `model.violate_pairwise_fn` checks consistency with the construction rule; it does not independently establish that cliques extend to full response vectors, nor does it test perfectness. The extension argument above supplies the former guarantee.

The two checks give different information, as this example shows:

```python
for treatments in ((0, 1), (0, 1, 2)):
    checked = MediationModel((0, 1), ((0, 1),), treatments)
    checked_graph = checked.build_graph()
    checked_analyzer = GraphAnalyzer(checked_graph, group_fn=checked.group_fn)
    rows = checked_analyzer.enumerate_cliques_and_check(checked.violate_pairwise_fn)
    print(f"D={treatments}: nodes={len(checked_graph)}, "
          f"edges={checked_graph.number_of_edges()}, "
          f"perfect={checked_analyzer.is_perfect()}, maximal cliques={len(rows)}, "
          f"all pairs pass={all(row['satisfies_condition'] for row in rows)}")
```
```text
D=(0, 1): nodes=8, edges=12, perfect=True, maximal cliques=12, all pairs pass=True
D=(0, 1, 2): nodes=12, edges=36, perfect=False, maximal cliques=28, all pairs pass=True
```

The three-treatment case reproduces the exclusion-only IV graph with 12 nodes and 36 edges. Its passing pairwise checks do not make its independent-set inequalities sharp.

## Input validation and errors

The shared support validator distinguishes invalid types from invalid values. The table gives representative messages from the implementation; coordinate indices in messages are zero-based. Errors inside an independent set also propagate through `format_inequalities`.

| Argument | Exception and trigger | Representative message |
|---|---|---|
| `y_support` | `TypeError`: string/bytes, non-iterable, or a non-real element, including a Boolean or complex number | `y_support values must be real numbers, got True of type bool` |
| `y_support` | `ValueError`: empty support, duplicate values, NaN, or infinity | `y_support must not be empty` |
| `d_support` | `TypeError`: the same invalid types as `y_support` | `d_support must be an iterable of numbers, got None of type NoneType` |
| `d_support` | `ValueError`: empty support, duplicates, or nonfinite values | `d_support values must be finite, got inf` |
| `m_supports` | `TypeError`: outer string/bytes or non-iterable | `m_supports must be a sequence of coordinate supports, not a string: '01'` |
| `m_supports` | `ValueError`: no mediator coordinates | `m_supports must contain at least one mediator coordinate` |
| `m_supports[j]` | `TypeError`: a flat number or string instead of a coordinate support | `m_supports[0] must be an iterable of numbers (one support per mediator coordinate), got 0 of type int; for a single mediator write m_supports=((0, 1),)` |
| `m_supports[j]` | `TypeError`: a coordinate support contains a non-real value or Boolean | `m_supports[0] values must be real numbers, got True of type bool` |
| `m_supports[j]` | `ValueError`: empty coordinate support, duplicate values, or nonfinite values | `m_supports[0] must not contain duplicate values: (0, 0)` |
| `active_mediators` | `TypeError`: `None`, Boolean, float, bytes, other non-iterable, or a bare integer | `active_mediators must be 'all' or a sequence of mediator coordinate indices, got the integer 0; for a single coordinate write active_mediators=(0,)` |
| `active_mediators` elements | `TypeError`: non-integer index, including floats and Python/NumPy booleans | `active_mediators must contain integer coordinate indices, got 0.0 of type float` |
| `active_mediators` | `ValueError`: a string other than `"all"`, empty selection, duplicate index, or index outside `range(n_mediators)`, including a negative index | `active_mediators must select at least one mediator coordinate` |
| `u`, `v` in the pairwise predicates | `ValueError`: mediator tuple has the wrong number of coordinates | `mediator vectors must have 1 coordinate(s), got (0, 1)` |
| `independent_set` | `TypeError`: string/bytes, non-iterable, a non-tuple node, or a tuple containing an unhashable entry | `nodes must be hashable tuples (y, m, d), got (0, [0], 0) with an unhashable entry` |
| `independent_set` | `ValueError`: empty set, duplicate node, node outside the model, or a pair joined by an edge | `(2, (0,), 0) is not a node (y, m, d) of this model` |
| `independent_sets` | `TypeError`: string/bytes, non-iterable, or a single independent set supplied instead of an iterable of sets | `format_inequalities expects an iterable of independent sets but received node-like tuples such as (0, (0,), 0); to format one independent set call format_inequality` |

Numeric equality also determines duplicate support values: `1` and `1.0` cannot both appear in one support. Assignment to a fixed setting raises `AttributeError`, for example: `MediationModel settings are fixed after construction; cannot set 'd_support'. Create a new MediationModel instead.`

## Relation to the other model classes

All three classes use the shared support validator, build potential response graphs, and format independent sets as probability inequalities. Their observed variables and restrictions differ:

| Class and documentation | Model and restrictions | Graph groups |
|---|---|---|
| [IVModel](iv_model.md) | Scalar instrument `Z`, treatment `D`, outcome `Y`; exclusion and/or treatment monotonicity in the instrument, with nondecreasing or nonincreasing direction. | Instrument values `z` |
| [PartialMonotonicityModel](partial_monotonicity_model.md) | Vector instrument `Z`, scalar `D` and `Y`; treatment is nondecreasing in each instrument coordinate, with optional exclusion. | Instrument vectors `z` |
| [MediationModel](mediation_model.md) | Assigned treatment `D`, mediator vector `M`, outcome `Y`; full mediation through selected coordinates and unrestricted mediator responses. | Treatment values `d` |

With one mediator, the graph equals the exclusion-only `IVModel` graph after relabeling `(y, (m,), d)` as `(y, m, d)`. The mediator takes the IV treatment role, and assigned treatment takes the IV instrument role. Inequality strings agree after renaming the IV labels `D` to `M1` and `Z` to `D`. This check additionally requires `iv_model.py` to be importable:

```python
import networkx as nx
from iv_model import IVModel

med = MediationModel((0, 1), ((0, 1),), (0, 1))
iv = IVModel((0, 1), (0, 1), (0, 1), exclusion=True, d_monotonicity=False)
med_graph, iv_graph = med.build_graph(), iv.build_graph()
def relabel(node):
    y, m, d = node
    return y, m[0], d

mapped = nx.relabel_nodes(med_graph, relabel)
same_nodes = set(mapped) == set(iv_graph)
same_edges = {frozenset(e) for e in mapped.edges()} == {frozenset(e) for e in iv_graph.edges()}
sets = GraphAnalyzer(med_graph, group_fn=med.group_fn).find_mis_excluding_single_group()
renamed = [
    iv.format_inequality([relabel(node) for node in s])
    .replace("D=", "M1=").replace("Z=", "D=") for s in sets
]
print(f"Same nodes and edges: {same_nodes and same_edges}")
print(f"Same inequalities after renaming: {med.format_inequalities(sets) == renamed}")
```
```text
Same nodes and edges: True
Same inequalities after renaming: True
```

For a proper subset `S`, projection sends `(y, m, d)` to `(y, m_S, d)`. Adjacency in the original graph is exactly adjacency in the all-coordinates-selected model on the selected supports. Each projected node is represented by all full mediator vectors with that profile; these copies have no edges among themselves. This is the sense in which the full graph is the selected-coordinate graph lifted through `m -> m_S`.

## Limitations

- Supports are finite and numeric. Mediator supports are Cartesian products of coordinate supports; arbitrary joint supports with missing combinations are outside the class's scope.
- The outcome always excludes a direct effect of `D`. Restrictions on the outcome are limited to selecting the coordinates that may enter `g`, with at least one selected coordinate.
- Mediator responses to treatment are unrestricted, including responses of unselected mediators. The class imposes no mediator or outcome monotonicity.
- The formatted inequalities are population restrictions. The class does not estimate probabilities, fit a model, calculate causal effects, or supply test statistics, standard errors, or p-values. Independence from treatment is maintained rather than empirically certified by the graph.
- In the non-perfect cases described above, the class supplies necessary inequalities but does not compute the additional inequalities needed for a sharp characterization.
- Product supports can create large graphs. `build_graph()` visits all unordered node pairs; maximal-set enumeration and the analyzer's exhaustive perfectness check can become costly. The examples do not imply a scalability guarantee.

## Further examples

[MediationModel.ipynb](../MediationModel.ipynb) shows the one-binary-mediator workflow and compares two binary mediators with all coordinates selected versus only `M1`. It prints model descriptions and graph counts, plots treatment groups, examines compatible and incompatible pairs, lists the four, eight, and four inequalities, and runs the perfectness and maximal-clique checks with a separate `GraphAnalyzer` for each graph.

Run the test suite from the repository root with `uv run --python 3.12 --with pytest --with networkx --with numpy python -m pytest tests -q`.
