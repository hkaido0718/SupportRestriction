"""Validation of finite numeric supports, shared by ``IVModel`` and
``PartialMonotonicityModel``.

A support is a nonempty, non-string iterable of distinct finite real numbers. It is
returned sorted, as a tuple, so that comparisons in the model classes follow the numeric
values and not the order in which the values were supplied.
"""

import math
import numbers

__all__ = ["validate_support"]


def validate_support(values, name):
    """Return ``values`` as a sorted tuple of distinct finite real numbers.

    Raises TypeError for a string or a non-iterable, and for an element that is not a
    real number (booleans count as not real here, so ``(0, True)`` is rejected rather
    than treated as a duplicate of ``(0, 1)``). Raises ValueError for an empty support,
    a duplicate value, or a value that is nan or infinite. NumPy scalars are converted
    to plain Python numbers; int and float are kept as given. ``name`` is used in the
    error messages.
    """
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of numbers, not a string: {values!r}")
    try:
        items = list(values)
    except TypeError:
        raise TypeError(
            f"{name} must be an iterable of numbers, got {values!r} of type "
            f"{type(values).__name__}"
        ) from None
    if not items:
        raise ValueError(f"{name} must not be empty")
    cleaned = []
    for value in items:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(
                f"{name} values must be real numbers, got {value!r} of type "
                f"{type(value).__name__}"
            )
        if not math.isfinite(value):
            raise ValueError(f"{name} values must be finite, got {value!r}")
        if type(value).__module__ == "numpy":  # numpy scalar -> plain int or float
            value = value.item()
        cleaned.append(value)
    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"{name} must not contain duplicate values: {tuple(items)!r}")
    return tuple(sorted(cleaned))
