from __future__ import annotations

from typing import Any

from .tensor import Tensor
from .tensor_types import FloatArray
from .validation import to_float_array, validate_covector_array


class Covector(Tensor):
    """
    Tensor of type (0, 1), represented by its components in a basis.
    """

    def __init__(self, components: FloatArray) -> None:
        validate_covector_array(components)
        super().__init__(components, tensor_type=(0, 1))

    @classmethod
    def from_data(cls, data: Any) -> Covector:
        components = to_float_array(data)
        return cls(components)