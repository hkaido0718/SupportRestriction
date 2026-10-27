# SupportRestriction
This repository collects codes and Jupyter notebooks that illustrate how to use the framework of Kaido and Ponomarev (2025) to obtain the sharp testable implication of a potential outcome model. It contains the following files.
- graph_analysis_utils.py: A library containing Python functions to build/plot a graph, derive MISs, and check regularity
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
- iv_model_handover.md: Notes on the design of `IVModel`
- partial_monotonicity_handover.md: Notes on the design of `PartialMonotonicityModel`
- mediation_model_handover.md: Notes on the design of `MediationModel`
- tests/: pytest suite for `IVModel`, `PartialMonotonicityModel` and `MediationModel` (run `python -m pytest tests -q` from the repository root)
