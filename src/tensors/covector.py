from __future__ import annotations

from typing import Any, Self

from .tensor import Tensor
from .tensor_types import ComplexArray
from .validation import to_complex_array, validate_covector_array


class Covector(Tensor):
    """
    Tensor of type (0, 1), represented by its components in a basis.
    """

    def __init__(self, components: ComplexArray) -> None:
        validate_covector_array(components)
        super().__init__(components, tensor_type=(0, 1))

    @classmethod
    def from_data(cls, data: Any) -> Covector:
        components = to_complex_array(data)
        return cls(components)

    def _with_components(self, components: ComplexArray) -> Self:
        return self.__class__(components)
