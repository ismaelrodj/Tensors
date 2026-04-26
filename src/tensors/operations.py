from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .tensor import Tensor

TensorSide = Literal["left", "right"]
AxisSpec = tuple[TensorSide, int]


def tensor_sum(left: Tensor, right: Tensor) -> Tensor:
    result = left + right
    if result is NotImplemented:
        raise TypeError("tensor_sum is only defined between Tensor objects.")
    return result


def tensor_product(left: Tensor, right: Tensor) -> Tensor:
    result = left @ right
    if result is NotImplemented:
        raise TypeError("tensor_product is only defined between Tensor objects.")
    return result


def _axis_variance(tensor: Tensor, axis: int) -> str:
    if axis < tensor.tensor_type[0]:
        return "contravariant"
    return "covariant"


def _normalize_axis(axis: int, tensor: Tensor, name: str) -> int:
    normalized = axis + tensor.rank if axis < 0 else axis
    if normalized < 0 or normalized >= tensor.rank:
        raise ValueError(f"{name} must refer to a valid tensor axis.")
    return normalized


def _resolve_axis_spec(
    axis_spec: AxisSpec,
    left: Tensor,
    right: Tensor,
    name: str,
) -> tuple[TensorSide, int]:
    side, axis = axis_spec
    if side not in {"left", "right"}:
        raise ValueError(f"{name} must refer to 'left' or 'right'.")
    tensor = left if side == "left" else right
    return side, _normalize_axis(axis, tensor, name)


def _tensor_from_components(components: Any, tensor_type: tuple[int, int]) -> Any:
    if not isinstance(components, np.ndarray):
        return Tensor._format_scalar(complex(components))
    if components.ndim == 0:
        return Tensor._format_scalar(complex(components.item()))
    if tensor_type == (1, 0):
        from .vector import Vector

        return Vector(components)
    if tensor_type == (0, 1):
        from .covector import Covector

        return Covector(components)
    return Tensor(components, tensor_type)


def tensor_contract(
    left: Tensor,
    right: Tensor,
    contravariant_axis: AxisSpec | None = None,
    covariant_axis: AxisSpec | None = None,
) -> Any:
    if contravariant_axis is None or covariant_axis is None:
        raise ValueError(
            "tensor_contract requires both contravariant_axis and covariant_axis."
        )

    contravariant_side, contravariant_index = _resolve_axis_spec(
        contravariant_axis, left, right, "contravariant_axis"
    )
    covariant_side, covariant_index = _resolve_axis_spec(
        covariant_axis, left, right, "covariant_axis"
    )

    if contravariant_side == covariant_side:
        raise ValueError("Contraction axes must belong to different tensors.")

    if contravariant_side == "left":
        left_axis = contravariant_index
        right_axis = covariant_index
        if _axis_variance(left, left_axis) != "contravariant":
            raise ValueError("contravariant_axis must point to a contravariant axis.")
        if _axis_variance(right, right_axis) != "covariant":
            raise ValueError("covariant_axis must point to a covariant axis.")
    else:
        left_axis = covariant_index
        right_axis = contravariant_index
        if _axis_variance(right, right_axis) != "contravariant":
            raise ValueError("contravariant_axis must point to a contravariant axis.")
        if _axis_variance(left, left_axis) != "covariant":
            raise ValueError("covariant_axis must point to a covariant axis.")
    if left.shape[left_axis] != right.shape[right_axis]:
        raise ValueError(
            "Cannot contract tensors whose matching index dimensions differ."
        )

    result_components = np.tensordot(
        left.components,
        right.components,
        axes=([left_axis], [right_axis]),
    )
    result_type = (
        left.tensor_type[0]
        + right.tensor_type[0]
        - 1,
        left.tensor_type[1]
        + right.tensor_type[1]
        - 1,
    )
    return _tensor_from_components(result_components, result_type)
