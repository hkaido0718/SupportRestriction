"""Build script: Python modules always, the C++ extension only on request.

``SUPPORTRESTRICTION_BUILD_CPP=1`` compiles ``cpp/odd_holes.cpp`` into the extension
module ``_supportrestriction_graph`` (needs a C++17 compiler). With ``0`` or with the
variable unset, the installation is Python-only. Project metadata is in ``pyproject.toml``.
"""
import os
from setuptools import setup

build_cpp = os.environ.get("SUPPORTRESTRICTION_BUILD_CPP", "0")
if build_cpp not in {"0", "1"}:
    raise ValueError("SUPPORTRESTRICTION_BUILD_CPP must be 0 or 1")

ext_modules = []
cmdclass = {}
if build_cpp == "1":
    from pybind11.setup_helpers import Pybind11Extension, build_ext

    ext_modules = [Pybind11Extension(
        "_supportrestriction_graph",
        ["cpp/odd_holes.cpp"],
        cxx_std=17,
    )]
    cmdclass = {"build_ext": build_ext}

setup(
    packages=[],
    py_modules=[
        "iv_model", "partial_monotonicity_model", "mediation_model",
        "support_validation", "graph_analysis_utils", "odd_hole",
    ],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
