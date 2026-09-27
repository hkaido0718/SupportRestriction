# SupportRestriction
This repository collects codes and Jupyter notebooks that illustrate how to use the framework of Kaido and Ponomarev (2025) to obtain the sharp testable implication of a potential outcome model. It contains the following files.
- graph_analysis_utils.py: A library containing Python functions to build/plot a graph, derive MISs, and check regularity
- iv_model.py: An `IVModel` class that builds the potential response graph of a discrete IV model under exclusion and/or D-monotonicity
- Partial_Monotonicity.ipynb: demonstrates an application of the library to the partial monotonicity example
- Interference.ipynb: Derives sharp identifying restrictions for spillover effects in the empirical application
- ExposureMap.ipynb: Derives sharp identifying restrictions for exposure maps in the empirical application
- CessationLength.ipynb: Derives sharp identifying restrictions for cessation length hypotheses in the empirical application
- CessationLengthSage.ipynb: Checks the perfectness of graphs using SageMath used in the cessation length example
- IVModel_Exclusion_Monotonicity.ipynb: Derives sharp identifying restrictions for the IV model under exclusion and D-monotonicity using `IVModel`
- iv_model_handover.md: Notes on the design of `IVModel`
- tests/: pytest suite for `IVModel` (run `python -m pytest tests -q` from the repository root)
