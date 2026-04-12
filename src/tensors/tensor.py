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
        """
        Public outputs are shown as real whenever the imaginary part vanishes,
        even though internal storage always uses complex scalars.
        """
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

    def as_matrix(
        self, row_axes: tuple[int, ...], col_axes: tuple[int, ...]
    ) -> ComplexArray:
        """
        Flatten the tensor into a 2D array by grouping selected axes into rows
        and the remaining selected axes into columns.
        """
        axes = row_axes + col_axes
        expected_axes = tuple(range(self.rank))

        if tuple(sorted(axes)) != expected_axes:
            raise ValueError(
                "row_axes and col_axes must partition all tensor axes exactly once."
            )

        permuted = np.transpose(self.components, axes=axes)
        row_size = int(np.prod([self.shape[axis] for axis in row_axes], dtype=int))
        col_size = int(np.prod([self.shape[axis] for axis in col_axes], dtype=int))
        return permuted.reshape((row_size, col_size))

    def display_matrix(
        self, row_axes: tuple[int, ...], col_axes: tuple[int, ...]
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """
        Return a 2D matrix view prepared for display, showing real values
        whenever the imaginary part is numerically zero.
        """
        return np.real_if_close(self.as_matrix(row_axes=row_axes, col_axes=col_axes))

    def display_slice(self, key: Any) -> Any:
        """
        Return a component slice prepared for display, showing real values
        whenever the imaginary part is numerically zero.
        """
        result = self[key]
        if isinstance(result, np.ndarray):
            return np.real_if_close(result)
        return self._format_scalar(complex(result))

    def apply(self, other: object) -> Any:
        """
        Contract the last covariant index of this tensor with the first
        contravariant index of another tensor.
        """
        if not isinstance(other, Tensor):
            return NotImplemented
        if self.tensor_type[1] == 0:
            raise ValueError("Cannot apply a tensor with no covariant indices.")
        if other.tensor_type[0] == 0:
            raise ValueError(
                "Cannot apply a tensor to an argument with no contravariant indices."
            )
        if self.shape[-1] != other.shape[0]:
            raise ValueError(
                "Cannot contract tensors whose matching index dimensions differ."
            )

        result_components = np.tensordot(self.components, other.components, axes=([-1], [0]))
        result_type = (
            self.tensor_type[0] + other.tensor_type[0] - 1,
            self.tensor_type[1] + other.tensor_type[1] - 1,
        )

        if result_components.ndim == 0:
            return self._format_scalar(complex(result_components.item()))

        if result_type == (1, 0):
            from .vector import Vector

            return Vector(result_components)
        if result_type == (0, 1):
            from .covector import Covector

            return Covector(result_components)
        return Tensor(result_components, result_type)

    def __add__(self, other: object) -> Self | NotImplementedType:
        if not isinstance(other, Tensor):
            return NotImplemented
        if self.tensor_type != other.tensor_type:
            raise ValueError("Cannot add tensors with different tensor types.")
        if self.shape != other.shape:
            raise ValueError("Cannot add tensors with different shapes.")
        return self._with_components(self.components + other.components)

    def tensor_product(self, other: object) -> Tensor | NotImplementedType:
        if not isinstance(other, Tensor):
            return NotImplemented

        result_type = (
            self.tensor_type[0] + other.tensor_type[0],
            self.tensor_type[1] + other.tensor_type[1],
        )
        result_components = np.tensordot(self.components, other.components, axes=0)
        return Tensor(result_components, result_type)

    def __matmul__(self, other: object) -> Tensor | NotImplementedType:
        return self.tensor_product(other)

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
