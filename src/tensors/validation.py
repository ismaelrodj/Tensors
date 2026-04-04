from __future__ import annotations

from typing import Any

import numpy as np

from .tensor_types import FloatArray, TensorType

def to_float_array(data: Any) -> FloatArray:
    return np.asarray(data, dtype=np.float64)


def validate_tensor_type(tensor_type: TensorType) -> None:
    if not isinstance(tensor_type, tuple):
        raise TypeError("tensor_type must be a tuple of two integers.")

    if len(tensor_type) != 2:
        raise ValueError("tensor_type must have length 2.")

    r, s = tensor_type

    if not isinstance(r, int) or not isinstance(s, int):
        raise TypeError("tensor_type entries must be integers.")

    if r < 0 or s < 0:
        raise ValueError("tensor_type entries must be non-negative.")


def validate_rank_against_type(
    components: FloatArray,
    tensor_type: TensorType,
) -> None:
    expected_rank = sum(tensor_type)
    actual_rank = components.ndim

    if actual_rank != expected_rank:
        raise ValueError(
            f"Expected rank {expected_rank} for tensor type {tensor_type}, "
            f"but got rank {actual_rank}."
        )


def validate_vector_array(components: FloatArray) -> None:
    if components.ndim != 1:
        raise ValueError(
            f"Vector components must be a 1-dimensional array, got ndim={components.ndim}."
        )


def validate_covector_array(components: FloatArray) -> None:
    if components.ndim != 1:
        raise ValueError(
            f"Covector components must be a 1-dimensional array, got ndim={components.ndim}."
        )