from __future__ import annotations

from typing import Any

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


def tensor_contract(left: Tensor, right: Tensor) -> Any:
    result = left.contract(right)
    if result is NotImplemented:
        raise TypeError("tensor_contract is only defined between Tensor objects.")
    return result
