# SupportRestriction
This repository collects codes and Jupyter notebooks that illustrate how to use the framework of Kaido and Ponomarev (2025) to obtain the sharp testable implication of a potential outcome model. It contains the following files.
- graph_analysis_utils.py: A library containing Python functions to build/plot a graph, derive MISs, and check regularity; `GraphAnalyzer.is_perfect` runs in Python by default and can use the optional C++ backend (see [C++ backend for perfectness checks](#c-backend-for-perfectness-checks))
- odd_hole.py: The adapter between `GraphAnalyzer` and the C++ backend: it converts a graph to the inputs of the compiled search, maps a returned odd hole back to the original node labels, verifies it in pure Python, and searches the complement for an odd antihole
- cpp/odd_holes.cpp: C++ (pybind11) source of the optional extension module `_supportrestriction_graph`, which searches a graph for an odd hole
- setup.py, pyproject.toml, MANIFEST.in: Installation with `pip`; the C++ extension is compiled only on request (see [Installation](#installation))
- iv_model.py: An `IVModel` class that builds the potential response graph of a discrete IV model under exclusion and/or D-monotonicity (nondecreasing or nonincreasing in the instrument) and writes the maximal independent sets of the graph as inequalities in conditional probabilities (`format_inequalities`)
- partial_monotonicity_model.py: A `PartialMonotonicityModel` class for a vector instrument with any number of discrete coordinates, imposing partial monotonicity of the treatment in each coordinate and, optionally, exclusion; it also writes the maximal independent sets of its graph as inequalities (`format_inequalities`)
- mediation_model.py: A `MediationModel` class for a discrete mediation model with a scalar outcome, a scalar assigned treatment and any number of discrete mediators, imposing full mediation through a selected set of mediator coordinates (no direct treatment effect); it also writes the maximal independent sets of its graph as inequalities (`format_inequalities`)
- support_validation.py: The support validator shared by `IVModel`, `PartialMonotonicityModel` and `MediationModel`
- Partial_Monotonicity.ipynb: demonstrates an application of the library to the partial monotonicity example
- Interference.ipynb: Derives sharp identifying restrictions for spillover effects in the empirical application
- ExposureMap.ipynb: Derives sharp identifying restrictions for exposure maps in the empirical application
- CessationLength.ipynb: Derives sharp identifying restrictions for cessation length hypotheses in the empirical application
- CessationLengthSage.ipynb: Checks the perfectness of graphs using SageMath used in the cessation length example
- IVModel_Exclusion_Monotonicity.ipynb: Derives sharp identifying restrictions for the IV model under exclusion and D-monotonicity in either direction using `IVModel`
- PartialMonotonicityModel.ipynb: Derives the testable implications of partial monotonicity with a vector instrument using `PartialMonotonicityModel`, starting from the two-binary-instrument example of Partial_Monotonicity.ipynb
- MediationModel.ipynb: Derives the testable implications of the mediation model with one and two binary mediators, with all mediators selected and with one mediator excluded from the outcome response, using `MediationModel`
- docs/iv_model.md: User documentation for `IVModel` (model, usage, API reference, reading the inequalities)
- docs/partial_monotonicity_model.md: User documentation for `PartialMonotonicityModel`
- docs/mediation_model.md: User documentation for `MediationModel`
- tests/: pytest suite for `IVModel`, `PartialMonotonicityModel`, `MediationModel` and the C++ backend (run `python -m pytest tests -q` from the repository root; the tests of the compiled search are skipped when the extension is not installed)

## Installation

The modules can be used straight from a clone of the repository, as the notebooks do. Installing them with `pip` makes them importable from any directory and, on request, compiles the C++ backend. Python 3.11 or later is required.

Run these commands from the repository root, in an activated Conda environment, on Ubuntu or macOS:

```bash
# Python-only installation: no C++ compiler required.
SUPPORTRESTRICTION_BUILD_CPP=0 python -m pip install .

# Installation with C++: requires a C++17 compiler.
SUPPORTRESTRICTION_BUILD_CPP=1 python -m pip install --force-reinstall --no-deps .
python -c "import _supportrestriction_graph; print(_supportrestriction_graph.__file__)"
```

- The second command assumes that the first has installed the runtime dependencies (`numpy`, `networkx`, `pandas`, `matplotlib`).
- `pip` installs the build requirements in an isolated build environment. Even the Python-only build may download `pybind11`, but it does not compile C++.
- `SUPPORTRESTRICTION_BUILD_CPP` matters during installation only. It must be `0` or `1`; unset means `0`.
- A C++17 compiler is `g++` on Ubuntu (`sudo apt install build-essential`) and `clang++` from the Xcode command line tools on macOS (`xcode-select --install`).
- If compilation is requested and fails, the installation fails with the compiler error. It does not fall back to a Python-only installation.
- Installing leaves `build/` and `SupportRestriction.egg-info/` in the repository directory (both ignored by git). `pip` reuses object files in `build/`, so delete that directory to force a fresh compilation.

Everything except the C++ backend works after the Python-only installation.

## C++ backend for perfectness checks

`GraphAnalyzer.is_perfect()` keeps its Python implementation as the default. It enumerates odd-sized node subsets of the graph and of its complement, which is practical for small graphs only. The C++ backend searches for the same objects, an odd hole or an odd antihole, and returns the one it finds:

```python
from graph_analysis_utils import GraphAnalyzer
from iv_model import IVModel
from odd_hole import verify_odd_hole

model = IVModel((0, 1), (0, 1), (0, 1, 2, 3), exclusion=True, d_monotonicity=False)
G = model.build_graph()
analyzer = GraphAnalyzer(G, group_fn=model.group_fn)

analyzer.is_perfect()                 # Python backend (default): False
analyzer.is_perfect(backend="cpp")    # C++ backend: False

kind, witness = analyzer.find_imperfectness_witness(backend="cpp")
print(kind)                           # odd_hole
print(witness)                        # [(0, 1, 0), (1, 0, 2), (1, 1, 3), (1, 1, 2), (0, 0, 3)]
verify_odd_hole(G, witness)           # pure-Python check; raises ValueError if it fails
```

`find_imperfectness_witness` returns one of:

- `("odd_hole", nodes)`: `nodes` is an induced cycle of the graph of odd length at least five, in cyclic order (the closing edge from the last node to the first is implied).
- `("odd_antihole", nodes)`: `nodes` is an odd hole of the complement graph, **ordered as a cycle in the complement**. Consecutive nodes are therefore not adjacent in the graph itself; check it with `verify_odd_hole(nx.complement(G), nodes)`.
- `(None, [])`: the searches of the graph and of its complement both finished without a witness, so the graph is perfect.

`is_perfect(backend="cpp")` runs that search once and returns `kind is None`.

The nodes of a witness are the original node labels, edge weights are ignored, and every witness is verified in pure Python before it is returned. A failure is never reported as perfectness: without the compiled extension both C++ calls raise `ImportError` with the installation command, and invalid input or a witness that fails verification raises as well. Witness retrieval supports `backend="cpp"` only. The compiled search cannot be interrupted with Ctrl-C while it runs.
