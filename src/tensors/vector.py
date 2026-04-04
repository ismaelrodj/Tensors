from __future__ import annotations

from typing import Any

from .tensor import Tensor
from .tensor_types import FloatArray
from .validation import to_float_array, validate_vector_array


class Vector(Tensor):
    """
    Tensor of type (1, 0), represented by its components in a basis.
    """

    def __init__(self, components: FloatArray) -> None:
        validate_vector_array(components)
        super().__init__(components, tensor_type=(1, 0))

    @classmethod
    def from_data(cls, data: Any) -> Vector:
        components = to_float_array(data)
        return cls(components)