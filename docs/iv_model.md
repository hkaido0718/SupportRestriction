# IVModel

`IVModel` represents a finite instrumental-variable model with optional exclusion and treatment monotonicity. The workflow produces a
**potential response graph**, whose nodes are observable events and whose edges mean that one admissible potential-response vector
can generate both events; its **independent sets**, sets of nodes with no edges between them, especially **maximal independent
sets** (MISs), which cannot be enlarged; and the inequalities in conditional probabilities those sets imply, which are testable
implications of the model. The framework is from Kaido and Ponomarev (2025), [“Testing Exclusion and Shape Restrictions in Potential Outcomes Models”, arXiv:2512.20851](https://arxiv.org/abs/2512.20851).

## Requirements and setup

Use Python 3 and `networkx` for `IVModel`. `GraphAnalyzer` also imports `pandas`, `matplotlib`, and NumPy, including when you only use its graph analysis methods. The
files `iv_model.py`, `support_validation.py`, and `graph_analysis_utils.py` must be importable. Run the examples from the repository root, or add that directory to `sys.path` before
importing. The two classes are imported separately:

```python
from iv_model import IVModel
from graph_analysis_utils import GraphAnalyzer
```

## The model

The observed variables are a scalar outcome Y, a scalar treatment D, and a scalar instrument Z, with finite numeric supports
supplied as `y_support`, `d_support`, and `z_support`. Potential treatments D(z) and potential outcomes Y(d, z) take values in the supplied treatment and
outcome supports and generate observations through D = D(Z) and Y = Y(D(Z), Z).

The maintained independence assumption is that the full vector ((Y(d, z))_{d, z}, (D(z))_z) is independent of Z. It is assumed
throughout; there is no constructor switch or data check for it. It lets the conditional distribution at each instrument value
describe responses from the same population of potential vectors.

- `exclusion=True`: the instrument has no direct effect on the outcome at a fixed treatment. Formally, Y(d, z) = Y(d, z') for every d, z, and
  z'. Then Y(d) is valid shorthand. With `exclusion=False`, Y(d, z) can vary freely with z.
- `d_monotonicity=True`: treatment moves weakly in the direction selected by `d_monotonicity_direction`. The default `"nondecreasing"` imposes D(z) <= D(z') whenever z < z'; `"nonincreasing"` imposes
  D(z) >= D(z') whenever z < z'. Equal treatment values are allowed in either direction. These restrictions apply to each
  potential treatment function, not just to average treatment rates.

At least one flag must be `True`. To impose D-monotonicity alone, explicitly pass `exclusion=False`. No monotonicity of potential outcomes is
imposed.

The direction is validated and stored even when `d_monotonicity=False`, but affects the graph only when monotonicity is enabled. Supports are sorted
numerically, so supplying Z in reverse order does not change the direction. Reversing its numeric order by relabelling z as -z
does: nonincreasing on Z is nondecreasing on -Z after relabelling each node (y, d, z) as (y, d, -z).

## Quick start

This complete binary example imposes both restrictions. `build_graph()` uses **Method 1**, the paper's construction of graph edges by checking
compatibility of each pair of events. The four inequalities with three probability terms reproduce the Balke-Pearl / Kitagawa
inequalities; the remaining propensity-score inequality is implied by them. The final check is explained in “When the inequalities
are sharp”.

```python
from iv_model import IVModel
from graph_analysis_utils import GraphAnalyzer

model = IVModel(
    y_support=(0, 1), d_support=(0, 1), z_support=(0, 1),
    exclusion=True, d_monotonicity=True,
)
print(model.summary())
G = model.build_graph()
analyzer = GraphAnalyzer(G, group_fn=model.group_fn)
mis = analyzer.find_mis_excluding_single_group()
print(f"Nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
print(f"Maximal independent sets spanning instrument values: {len(mis)}")
for inequality in sorted(model.format_inequalities(mis)):
    print(inequality)
print("Perfect:", analyzer.is_perfect())
```
```text
IVModel: discrete instrumental-variable model (Kaido and Ponomarev, 2025)
  Outcome support Y:    (0, 1)
  Treatment support D:  (0, 1)
  Instrument support Z: (0, 1)
  Nodes (y, d, z):      8
  Potential responses: D(z) and Y(d, z); observed D = D(Z), Y = Y(D(Z), Z).
  Assumptions:
    - Exclusion: imposed. Y(d, z) = Y(d, z') for every treatment value d and every pair of instrument values z, z'.
    - D-monotonicity: imposed, nondecreasing. D(z) <= D(z') whenever z < z' (treatment is nondecreasing in the instrument).
    - No outcome monotonicity is imposed.
  Maintained (not a pairwise rule): the full potential-response vector
    ((Y(d, z))_{d, z}, (D(z))_z) is independent of Z.
  Graph: build_graph() applies Method 1 (pairwise compatibility) and returns
    a networkx.Graph for GraphAnalyzer.
Nodes: 8, edges: 8
Maximal independent sets spanning instrument values: 5
P(D=1 | Z=0) + P(D=0 | Z=1) <= 1
P(Y=0, D=0 | Z=0) + P(D=1 | Z=0) + P(Y=1, D=0 | Z=1) <= 1
P(Y=0, D=1 | Z=0) + P(D=0 | Z=1) + P(Y=1, D=1 | Z=1) <= 1
P(Y=1, D=0 | Z=0) + P(D=1 | Z=0) + P(Y=0, D=0 | Z=1) <= 1
P(Y=1, D=1 | Z=0) + P(D=0 | Z=1) + P(Y=0, D=1 | Z=1) <= 1
Perfect: True
```

Sorting the strings makes the displayed order reproducible; the formatter itself preserves the order of the sets it receives.
These are population restrictions: the example supplies supports and assumptions, without a dataset.

For the nonincreasing direction, the binary inequalities are mirror images with Z=0 and Z=1 exchanged. Here are both
specifications that impose that direction:

```python
from iv_model import IVModel
from graph_analysis_utils import GraphAnalyzer

for exclusion in (False, True):
    decreasing = IVModel(
        (0, 1), (0, 1), (0, 1), exclusion=exclusion,
        d_monotonicity=True, d_monotonicity_direction="nonincreasing",
    )
    analysis = GraphAnalyzer(decreasing.build_graph(), decreasing.group_fn)
    sets = analysis.find_mis_excluding_single_group()
    print(f"Nonincreasing, exclusion={exclusion}")
    for inequality in sorted(decreasing.format_inequalities(sets)):
        print(inequality)
```
```text
Nonincreasing, exclusion=False
P(D=0 | Z=0) + P(D=1 | Z=1) <= 1
Nonincreasing, exclusion=True
P(D=0 | Z=0) + P(D=1 | Z=1) <= 1
P(D=0 | Z=0) + P(Y=0, D=1 | Z=0) + P(Y=1, D=1 | Z=1) <= 1
P(D=0 | Z=0) + P(Y=1, D=1 | Z=0) + P(Y=0, D=1 | Z=1) <= 1
P(Y=0, D=0 | Z=0) + P(Y=1, D=0 | Z=1) + P(D=1 | Z=1) <= 1
P(Y=1, D=0 | Z=0) + P(Y=0, D=0 | Z=1) + P(D=1 | Z=1) <= 1
```

## Constructor and attributes

The signature is `IVModel(y_support, d_support, z_support, exclusion=True, d_monotonicity=False, d_monotonicity_direction="nondecreasing")`.

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `y_support` | Iterable of real numbers | Required | Nonempty, distinct, finite outcome values. |
| `d_support` | Iterable of real numbers | Required | Nonempty, distinct, finite treatment values; numeric order determines treatment comparisons. |
| `z_support` | Iterable of real numbers | Required | Nonempty, distinct, finite instrument values; numeric order determines instrument comparisons. |
| `exclusion` | `bool` | `True` | Impose outcome exclusion across instrument values. |
| `d_monotonicity` | `bool` | `False` | Impose treatment monotonicity in the selected direction. |
| `d_monotonicity_direction` | `str` | `"nondecreasing"` | Exactly `"nondecreasing"` or `"nonincreasing"`, including when monotonicity is off. |

Supports accept non-string iterables such as tuples, lists, ranges, generators, and NumPy arrays, in any order. Single-valued
supports are allowed. Values must satisfy the validator's real-number check: ordinary integers, floats, and `Fraction` values are
accepted; booleans, complex numbers, and `Decimal` values are rejected. NumPy real scalars become Python numbers; ordinary integers and
floats retain their types. Duplicates are determined by numeric equality, so 1 and 1.0 cannot both appear.

| Attribute | Type | Meaning |
| --- | --- | --- |
| `y_support` | `tuple` | Validated outcome values, sorted increasingly. |
| `d_support` | `tuple` | Validated treatment values, sorted increasingly. |
| `z_support` | `tuple` | Validated instrument values, sorted increasingly. |
| `exclusion` | `bool` | Whether exclusion is imposed. |
| `d_monotonicity` | `bool` | Whether treatment monotonicity is imposed. |
| `d_monotonicity_direction` | `str` | Stored direction, even when it has no effect. |
| `nodes` | Tuple of three-element tuples `(y, d, z)` | All `len(y_support) * len(d_support) * len(z_support)` combinations. Order is y outermost, then d, then z, each in sorted support order. |

Settings are fixed after construction: assigning or deleting attributes raises `AttributeError`. Construct a new model to change supports or
assumptions. The NetworkX graph returned by `build_graph()` is a separate, mutable object.

`repr(model)` includes every constructor setting. With integer or float supports, it can reconstruct the specification when `IVModel` is in the
evaluation namespace:

```python
from iv_model import IVModel

original = IVModel([1, 0], [1, 0], [1, 0], d_monotonicity=True)
rebuilt = eval(repr(original), {"IVModel": IVModel})
print(repr(original))
print("Same nodes:", rebuilt.nodes == original.nodes)
print("Same representation:", repr(rebuilt) == repr(original))
```
```text
IVModel(y_support=(0, 1), d_support=(0, 1), z_support=(0, 1), exclusion=True, d_monotonicity=True, d_monotonicity_direction='nondecreasing')
Same nodes: True
Same representation: True
```

If support values have named representations, such as `Fraction(1, 2)`, those names must also be available to evaluate the representation.

## Methods

Signatures below omit `self`. Pass model nodes to the pairwise predicates; these methods unpack the nodes directly and do not
validate node membership.

| Method | Returns | When to use it |
| --- | --- | --- |
| `group_fn(node)` | The node's instrument value, `node[2]` | Supply it as `GraphAnalyzer`'s grouping function. |
| `violates_exclusion(u, v)` | `bool`: same treatment and different outcomes | Inspect the exclusion rule, even when `exclusion=False`. Intended for different instrument values; on a same-Z pair it evaluates the same formula. |
| `violates_d_monotonicity(u, v)` | `bool`: treatment moves against the stored direction as Z rises | Inspect monotonicity, even when `d_monotonicity=False`. Equal treatment values do not violate it; same-Z pairs also return `False`. |
| `violate_pairwise_fn(u, v)` | `bool`: at least one enabled rule rejects the pair | Check compatibility across instrument groups or supply the predicate to `enumerate_cliques_and_check`. It combines enabled rules with `or`; same-group exclusion from the graph is handled separately. |
| `build_graph()` | A fresh undirected `networkx.Graph` | Construct all nodes and compatible cross-group edges, including isolated nodes. Repeated calls return separate graphs. |
| `format_inequality(independent_set)` | One inequality as a `str` | Format any nonempty independent set of this model's graph; maximality is not required. |
| `format_inequalities(independent_sets)` | `list[str]`, in input order | Format an iterable of independent sets, usually the analyzer's output. An empty outer iterable returns `[]`; a single set must go to the singular method. |
| `summary()` | A multiline `str` | Inspect supports, node count, enabled assumptions, direction, and maintained independence. Call `print(model.summary())` to display it. |

The formatter checks that its input is an independent set of the model's own graph, as well as checking node membership and
duplicates. A set confined to one instrument value is accepted, although its inequality is just a probability bound within that
group.

A **clique** is a set of nodes with an edge between every pair; a **maximal clique** cannot be enlarged while retaining that
property. For `GraphAnalyzer(G, group_fn=model.group_fn)`, `find_mis_excluding_single_group()` returns a list of sorted node lists, one per maximal independent set spanning at least two distinct Z
values. It enumerates maximal cliques in the complement graph (the graph whose edges are precisely the missing edges between
distinct nodes of G), then filters out sets contained in one group. It does not discard a set merely because several of its nodes
share a group. The outer enumeration order is not sorted. `print_mis_excluding_single_group()` prints the count and numbered node lists and returns `None`; it does
not format probability inequalities.

`plot_grouped_on_circle(title="Graph", node_size=1000, figsize=(6, 6), cmap_name="Set2", group_gap=np.pi/15)` arranges nodes around a circle, orders and colors groups by their Z values, calls Matplotlib's `show()`, and returns `None`. Using `analyzer`
from the quick start:

```python
analyzer.plot_grouped_on_circle(title="Exclusion and D-monotonicity")
```

## Reading the inequalities

A node (y, d, z) contributes `P(Y=y, D=d | Z=z)`. Under independence, this conditional probability equals the population probability of the
potential-response event {D(z)=d, Y(d, z)=y}. For an independent set, no potential vector can generate two of these events, so
they are mutually exclusive and their probabilities sum to at most one. This is the `<= 1` bound of [Theorem 1](https://arxiv.org/html/2512.20851v1). It compares conditional
probabilities across Z values; it is not a sum of joint probabilities that include the probability of assignment to each Z.

At a fixed (d, z), terms merge into `P(D=d | Z=z)` exactly when all values in `y_support` are present. There is no further merging across treatments
or instrument values. Terms are ordered by z, then d, then y, in sorted support order, regardless of the order of input nodes.
Numeric aliases such as 1.0 or `numpy.int64(1)` match the node with value 1; the displayed values come from the model's stored supports.

**Exclusion alone.** For binary supports, there are four inequalities spanning instrument values: Pearl's instrumental
inequalities. For example, a unit cannot have Y(0)=0 at Z=0 and Y(0)=1 at Z=1.

```python
from iv_model import IVModel
from graph_analysis_utils import GraphAnalyzer

exclusion_only = IVModel((0, 1), (0, 1), (0, 1))
analysis = GraphAnalyzer(exclusion_only.build_graph(), exclusion_only.group_fn)
sets = analysis.find_mis_excluding_single_group()
print("Inequalities:", len(sets))
for inequality in sorted(exclusion_only.format_inequalities(sets)):
    print(inequality)
```
```text
Inequalities: 4
P(Y=0, D=0 | Z=0) + P(Y=1, D=0 | Z=1) <= 1
P(Y=0, D=1 | Z=0) + P(Y=1, D=1 | Z=1) <= 1
P(Y=1, D=0 | Z=0) + P(Y=0, D=0 | Z=1) <= 1
P(Y=1, D=1 | Z=0) + P(Y=0, D=1 | Z=1) <= 1
```

**D-monotonicity alone.** The independent set contains both outcome values at (D=1, Z=0) and at (D=0, Z=1), whose treatment
pattern violates D(0) <= D(1). Merging complete outcome supports gives the propensity-score inequality. Since P(D=0 | Z=1) = 1 -
P(D=1 | Z=1), it states P(D=1 | Z=0) <= P(D=1 | Z=1).

```python
from iv_model import IVModel
from graph_analysis_utils import GraphAnalyzer

monotonicity_only = IVModel(
    (0, 1), (0, 1), (0, 1), exclusion=False, d_monotonicity=True,
)
analysis = GraphAnalyzer(monotonicity_only.build_graph(), monotonicity_only.group_fn)
sets = analysis.find_mis_excluding_single_group()
for inequality in monotonicity_only.format_inequalities(sets):
    print(inequality)
```
```text
P(D=1 | Z=0) + P(D=0 | Z=1) <= 1
```

With both restrictions, the quick start's four three-term inequalities can be rewritten as P(Y=y, D=0 | Z=1) <= P(Y=y, D=0 | Z=0)
and P(Y=y, D=1 | Z=0) <= P(Y=y, D=1 | Z=1), for y=0,1. Summing the latter pair yields the propensity-score inequality, explaining
why that fifth output is redundant.

**Incomplete versus complete outcome support.** If Y has three values, two of them do not merge. These single-group examples also
show that the singular formatter accepts independent sets that are not maximal:

```python
from iv_model import IVModel

three_outcomes = IVModel((0, 1, 2), (0, 1), (0, 1))
partial = [(1, 0, 0), (0, 0, 0)]
print(three_outcomes.format_inequality(partial))
print(three_outcomes.format_inequality(partial + [(2, 0, 0)]))
```
```text
P(Y=0, D=0 | Z=0) + P(Y=1, D=0 | Z=0) <= 1
P(D=0 | Z=0) <= 1
```

## How the graph is built

There is one node for every (y, d, z) in the Cartesian product of the supports. Nodes with the same z form a group. A potential
vector generates exactly one node per group, so distinct same-group nodes have no edge. For nodes u=(y,d,z) and v=(y',d',z') with
z != z', an edge means that some admissible potential vector generates both. Method 1 decides this using the following rules:

- Exclusion rejects `d == d' and y != y'`: the pair would assign conflicting values to the same Y(d). Different treatments fix different outcome
  components.
- Nondecreasing D-monotonicity rejects `(z < z' and d > d') or (z' < z and d' > d)`. Treatment cannot fall when the instrument rises.
- Nonincreasing D-monotonicity rejects `(z < z' and d < d') or (z' < z and d' < d)`. Treatment cannot rise when the instrument rises.

Only enabled restrictions participate. If either enabled rule rejects the pair, no admissible vector can generate both nodes.
Conversely, suppose neither rejects it and order the pair so z < z'. If monotonicity is imposed, set D(t)=d for t < z' and D(t)=d'
for t >= z'. This matches both observations and is monotone in the selected direction. Without monotonicity, assign the two
treatments separately. Under exclusion, set Y(d)=y and Y(d')=y'; if d=d', compatibility guarantees y=y'. Without exclusion, assign
Y(d,z)=y and Y(d',z')=y' to distinct cells. Complete the remaining unconstrained components freely. The treatment and outcome
assignments constrain different components, so they can be carried out together. This establishes why combining the violations
with `or` gives the exact edge rule.

A clique has at most one node per instrument value. The same completion works for any clique: order its nodes by z, carry each
specified treatment forward until the next specified z, and use the first treatment below the first specified z. Compatible pairs
ensure the required monotonicity and consistent outcomes for repeated treatments under exclusion. Thus a full admissible vector
generates all nodes of the clique.

A **support point** is one admissible realization of the full potential-response vector. It generates one response at every
instrument value. A maximal clique, one that cannot be enlarged, therefore consists of all the events generated by some support
point; multiple vectors can agree on those events. This extension property is the pairwise incompatibility criterion: if joint
compatibility fails, some pair already fails. An independent set instead contains no edges, and a maximal independent set cannot
accept any additional node without creating an edge. “Maximal” means inclusion-maximal, not necessarily largest in size.

## When the inequalities are sharp

**Necessary** inequalities hold for every distribution allowed by the model. **Sharp** inequalities are also sufficient: together
with nonnegative cell probabilities summing to one within each Z group, they characterize exactly the observable distributions
allowed by the model. A **perfect graph** has, in every induced subgraph (a subset of nodes with all edges between them retained),
minimum vertex-color count equal to largest clique size, where adjacent nodes must receive different colors. For this class,
perfectness and the clique completion proved above give sharpness by [Theorem 3](https://arxiv.org/html/2512.20851v1).

`GraphAnalyzer.is_perfect()` returns a `bool`. It exhaustively looks in G and its complement for an induced odd cycle of length at least five: a cycle with no
extra edges between its nodes. Finding one returns `False`. `enumerate_cliques_and_check(violate_pairwise_fn)` returns a list of dictionaries with keys `index` (one-based integer), `clique`
(sorted node list), and `satisfies_condition` (Boolean: no pair violates the supplied predicate). It checks all maximal cliques, not all cliques or
potential vectors. For a graph built with the same predicate, passing this check follows from construction; joint extendability
relies on the argument above, and perfectness remains separate.

- With two instrument values, every specification is bipartite (nodes split into two groups with edges only across groups), hence
  perfect. A single instrument value gives an edgeless, perfect graph and no inequalities spanning groups.
- With at least three instrument values and at least two outcome and two treatment values, exclusion alone produces a non-perfect
  graph. Its binary-outcome, binary-treatment, three-instrument subgraph already contains an induced odd cycle. The maximal
  independent set inequalities are necessary but not sharp.
- With D-monotonicity, with or without exclusion, either direction gives a perfect graph for all supported finite supports.
  Orienting edges toward increasing Z gives a transitive order (reverse the treatment comparison for nonincreasing). Such graphs
  are comparability graphs: edges connect comparable elements of an order. The exclusion-plus-monotonicity result is established
  in [Appendix B.1](https://arxiv.org/html/2512.20851v1).
- The non-perfect exclusion-only case requires both outcome and treatment variation; with only one outcome or one treatment value,
  that graph is perfect.

Here Y and D are binary and Z has three values. Counting all maximal independent sets uses the complement graph directly, so the
count includes the three single-group sets that the analyzer filters out.

```python
import networkx as nx
from iv_model import IVModel
from graph_analysis_utils import GraphAnalyzer

specifications = [
    ("Exclusion only", {}),
    ("Both, nondecreasing", {"d_monotonicity": True}),
    ("Both, nonincreasing", {
        "d_monotonicity": True,
        "d_monotonicity_direction": "nonincreasing",
    }),
]
for label, options in specifications:
    current = IVModel((0, 1), (0, 1), (0, 1, 2), **options)
    graph = current.build_graph()
    analysis = GraphAnalyzer(graph, current.group_fn)
    rows = analysis.enumerate_cliques_and_check(current.violate_pairwise_fn)
    all_mis = list(nx.find_cliques(nx.complement(graph)))
    filtered = analysis.find_mis_excluding_single_group()
    print(label)
    print(f"  nodes={len(graph)}, edges={graph.number_of_edges()}, "
          f"maximal cliques={len(rows)}, all MISs={len(all_mis)}, "
          f"MISs spanning groups={len(filtered)}")
    print(f"  perfect={analysis.is_perfect()}, "
          f"all cliques pass={all(row['satisfies_condition'] for row in rows)}")
```
```text
Exclusion only
  nodes=12, edges=36, maximal cliques=28, all MISs=15, MISs spanning groups=12
  perfect=False, all cliques pass=True
Both, nondecreasing
  nodes=12, edges=24, maximal cliques=12, all MISs=26, MISs spanning groups=23
  perfect=True, all cliques pass=True
Both, nonincreasing
  nodes=12, edges=24, maximal cliques=12, all MISs=26, MISs spanning groups=23
  perfect=True, all cliques pass=True
```

## Input validation and errors

The messages below are representative literal messages; support errors identify the affected argument. Formatting errors apply to
the singular method and to each inner set supplied to the plural method.

| Argument | Exception | Rejected input | Representative message |
| --- | --- | --- | --- |
| Any support | `TypeError` | String, bytes, or bytearray | `y_support must be a sequence of numbers, not a string: '01'` |
| Any support | `TypeError` | Non-iterable | `y_support must be an iterable of numbers, got 5 of type int` |
| Any support | `TypeError` | Non-real element, including Boolean, complex, or Decimal values | `y_support values must be real numbers, got True of type bool` |
| Any support | `ValueError` | Empty iterable | `y_support must not be empty` |
| Any support | `ValueError` | Duplicate by numeric equality | `y_support must not contain duplicate values: (0, 1, 1)` |
| Any support | `ValueError` | NaN or positive/negative infinity | `y_support values must be finite, got inf` |
| `exclusion`, `d_monotonicity` | `TypeError` | Anything other than Python `bool`, including 0, 1, and `numpy.bool_` | `exclusion must be a bool (True or False), got 1 of type int` |
| Both flags | `ValueError` | Both `False` | `at least one of exclusion and d_monotonicity must be True; to impose D-monotonicity alone, pass exclusion=False and d_monotonicity=True` |
| `d_monotonicity_direction` | `TypeError` | Non-string | `d_monotonicity_direction must be a str, one of 'nondecreasing', 'nonincreasing'; got None of type NoneType` |
| `d_monotonicity_direction` | `ValueError` | Any other string; spelling and case must match exactly | `d_monotonicity_direction must be one of 'nondecreasing', 'nonincreasing'; got 'increasing'` |
| `independent_set` | `TypeError` | String-like input or non-iterable | `an independent set must be an iterable of nodes (y, d, z), not a string: 'abc'` |
| Elements of `independent_set` | `TypeError` | Non-tuple node | `nodes must be hashable tuples (y, d, z), got [0, 0, 0] of type list` |
| Elements of `independent_set` | `TypeError` | Tuple with an unhashable entry | `nodes must be hashable tuples (y, d, z), got ([0], 0, 0) with an unhashable entry` |
| `independent_set` | `ValueError` | Empty iterable | `an independent set must contain at least one node` |
| Elements of `independent_set` | `ValueError` | Tuple not in model nodes, including wrong length | `(2, 0, 0) is not a node (y, d, z) of this model` |
| `independent_set` | `ValueError` | Duplicate node | `an independent set must not contain duplicate nodes: [(0, 0, 0), (0, 0, 0)]` |
| `independent_set` | `ValueError` | A cross-group pair joined by an edge | `(0, 0, 0) and (0, 0, 1) are adjacent in the potential response graph, so the set is not independent and its events are not mutually exclusive` |
| `independent_sets` | `TypeError` | String-like input or non-iterable | `independent_sets must be an iterable of independent sets, got 3 of type int` |
| `independent_sets` | `TypeError` | One set of node tuples instead of an iterable of sets | `format_inequalities expects an iterable of independent sets but received node-like tuples such as (0, 0, 1); to format one independent set call format_inequality` |

Node lookup uses numeric equality, so Boolean entries can alias existing numeric node values during formatting, even though
Boolean support values are rejected at construction. Aliased duplicates still raise `ValueError`. Attribute assignment or deletion raises
`AttributeError`, for example: `IVModel settings are fixed after construction; cannot set 'exclusion'. Create a new IVModel instead.`

## Relation to the other model classes

The three classes share the workflow of defining supports, building a potential response graph, analyzing it with `GraphAnalyzer`, and
formatting inequalities. `IVModel`, this document, uses a scalar instrument and offers exclusion alone, treatment monotonicity alone, or both, with
either monotonicity direction.

[PartialMonotonicityModel](partial_monotonicity_model.md) uses a vector instrument on a Cartesian-product support. It always imposes treatment nondecreasing in each coordinate, holding
the others fixed, and optionally imposes exclusion. With one coordinate its graph matches nondecreasing `IVModel` with D-monotonicity
enabled and the same exclusion flag, after replacing scalar z by the one-element tuple (z,).

[MediationModel](mediation_model.md) models assigned treatment D, mediators M, and outcome Y. Outcomes depend only on selected mediators, with no direct effect
of assigned treatment; potential responses are independent of D. Its nodes are (y, m, d), grouped by assigned treatment. With a single mediator (which must
be selected), its graph is the exclusion-only IV graph after relabelling: the mediator plays IVModel's treatment role and assigned
treatment plays its instrument role. It imposes no monotonicity.

## Limitations

The class covers finite numeric supports for scalar Y, D, and Z, with common outcome and treatment supports across instrument
values. It has no interface for continuous variables, vector instruments, outcome monotonicity, covariates, or arbitrary
additional support restrictions. It does not estimate probabilities, fit a causal model, compute treatment effects, or perform
statistical tests. The inequalities concern population probabilities; sampling uncertainty needs a separate inference procedure.
Independence is maintained rather than verified.

Graph construction visits every node pair, so its cost grows quadratically with the product of support sizes. Maximal independent
set and clique enumeration can be large, and `is_perfect()` searches subsets exhaustively. These routines suit small finite examples.
Formatting does not remove redundant inequalities or generate the extra constraints needed for sharpness in the non-perfect cases.
It validates sets against the model's rules, so its guarantees apply to the graph from `build_graph()`, rather than arbitrary edits to a
returned graph.

## Further examples

[IVModel_Exclusion_Monotonicity.ipynb](../IVModel_Exclusion_Monotonicity.ipynb) shows the three binary specifications, graph plots, maximal independent sets and their inequalities, perfectness and clique
checks, the nonincreasing direction, and the three-valued instrument comparison under exclusion alone and both restrictions.

Run the test suite from the repository root with `uv run --python 3.12 --with pytest --with networkx --with numpy python -m pytest tests -q`.
