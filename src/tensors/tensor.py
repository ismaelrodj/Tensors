from __future__ import annotations

from typing import Any

from .tensor_types import FloatArray, Shape, TensorType
from .validation import (
    to_float_array,
    validate_rank_against_type,
    validate_tensor_type,
)


class Tensor:
    """
    General tensor of type (r, s), represented by its components in a basis.
    """

    def __init__(self, components: FloatArray, tensor_type: TensorType) -> None:
        validate_tensor_type(tensor_type)
        validate_rank_against_type(components, tensor_type)

        self._components = components
        self._tensor_type = tensor_type

    @classmethod
    def from_data(cls, data: Any, tensor_type: TensorType) -> Tensor:
        components = to_float_array(data)
        return cls(components, tensor_type)

    @property
    def components(self) -> FloatArray:
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

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"tensor_type={self.tensor_type}, "
            f"shape={self.shape}, "
            f"components={self.components}"
            f")"
        )