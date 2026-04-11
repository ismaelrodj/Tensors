from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import numpy as np

from .tensor import Tensor
from .tensor_types import ComplexArray
from .validation import to_complex_array, validate_covector_array

if TYPE_CHECKING:
    from .vector import Vector


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

    def evaluate(self, vector: Vector) -> float | complex:
        if self.shape != vector.shape:
            raise ValueError(
                "Cannot evaluate a covector on a vector with a different shape."
            )
        value = complex(np.dot(self.components, vector.components))
        return self._format_scalar(value)

    def __call__(self, vector: Vector) -> float | complex:
        return self.evaluate(vector)

    def _with_components(self, components: ComplexArray) -> Self:
        return self.__class__(components)
