from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

TensorType: TypeAlias = tuple[int, int]
Shape: TypeAlias = tuple[int, ...]
ComplexArray: TypeAlias = npt.NDArray[np.complex128]
