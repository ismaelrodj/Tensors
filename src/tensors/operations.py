from __future__ import annotations

from .tensor import Tensor


def tensor_sum(left: Tensor, right: Tensor) -> Tensor:
    result = left + right
    if result is NotImplemented:
        raise TypeError("tensor_sum is only defined between Tensor objects.")
    return result


def tensor_product(left: Tensor, right: Tensor) -> Tensor:
    result = left.tensor_product(right)
    if result is NotImplemented:
        raise TypeError("tensor_product is only defined between Tensor objects.")
    return result
