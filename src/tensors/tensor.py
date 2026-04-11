from __future__ import annotations

from types import NotImplementedType
from typing import Any, Self

import numpy as np

from .tensor_types import ComplexArray, Shape, TensorType
from .validation import (
    to_complex_array,
    validate_rank_against_type,
    validate_tensor_type,
)


class Tensor:
    """
    General tensor of type (r, s), represented by its components in a basis.
    """

    def __init__(self, components: ComplexArray, tensor_type: TensorType) -> None:
        validate_tensor_type(tensor_type)
        validate_rank_against_type(components, tensor_type)

        self._components = components
        self._tensor_type = tensor_type

    @classmethod
    def from_data(cls, data: Any, tensor_type: TensorType) -> Tensor:
        components = to_complex_array(data)
        return cls(components, tensor_type)

    def _with_components(self, components: ComplexArray) -> Self:
        return self.__class__(components, self.tensor_type)

    @staticmethod
    def _format_scalar(value: complex) -> float | complex:
        if np.isclose(value.imag, 0.0):
            return float(value.real)
        return value

    @classmethod
    def _format_components(cls, components: ComplexArray) -> Any:
        if components.ndim == 0:
            return cls._format_scalar(complex(components.item()))
        return [cls._format_components(subarray) for subarray in components]

    @property
    def components(self) -> ComplexArray:
        return self._components

    @property
    def tensor_type(self) -> TensorType:
        return self._tensor_type

    @property
    def shape(self) -> Shape:
        return self._components.shape

    @property
    def rank(self) -> int:
        return self._components.ndim

    def __getitem__(self, key: Any) -> Any:
        return self._components[key]

    def __mul__(self, scalar: object) -> Self | NotImplementedType:
        if not isinstance(scalar, int | float | complex):
            return NotImplemented
        return self._with_components(self.components * complex(scalar))

    def __rmul__(self, scalar: object) -> Self | NotImplementedType:
        return self.__mul__(scalar)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"tensor_type={self.tensor_type}, "
            f"shape={self.shape}, "
            f"components={self._format_components(self.components)}"
            f")"
        )
