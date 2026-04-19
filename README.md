# Tensors

A typed educational Python library for exploring vectors, covectors, and tensors with conceptual rigor.

## Project Goal

The goal of this project is not only to perform tensor-related computations, but to understand tensor algebra through a clean and explicit implementation.

This library is being developed with two main priorities:

1. **Conceptual clarity**
2. **Rigorous typing and structure**

The intention is to represent vectors, covectors, and tensors in a way that remains faithful to their mathematical meaning, while still being practical enough for examples, experiments, and future extensions.

## Current Status

At its current stage, the project includes:

- a general `Tensor` class;
- specialized `Vector` and `Covector` classes;
- basic validation for tensor types and component shapes;
- evaluation of a covector on a vector;
- tensor products between tensors;
- a modular package structure;
- and an initial test suite.

Operations between these objects will be added progressively as the conceptual design becomes more mature.

## Project Structure

```text
tensors/
  pyproject.toml
  .gitignore
  src/
    tensors/
      __init__.py
      covector.py
      tensor.py
      tensor_types.py
      validation.py
      vector.py
  tests/
    test_covector.py
    test_tensor.py
    test_vector.py
  notebooks/
```

## Installation

Create and activate a virtual environment, then install the project in editable mode:

```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install -e .[dev]
```

## Running Tests

```bash
python -m pytest
```

## Basic Usage

```python
from tensors import Tensor, Vector, Covector, tensor_contract, tensor_product, tensor_sum

v = Vector.from_data([1, 2, 3])
alpha = Covector.from_data([4, 5, 6])
T = Tensor.from_data([[1, 2], [3, 4]], tensor_type=(1, 1))
S = Tensor.from_data([[10, 20], [30, 40]], tensor_type=(1, 1))

print(v)
print(alpha)
print(T)
print(alpha(v))
print(tensor_sum(T, S))
print(tensor_product(v, alpha))
print(v @ alpha)
print(tensor_product(T, v).as_matrix(row_axes=(0, 1), col_axes=(2,)))
print(tensor_product(T, v).display_matrix(row_axes=(0, 1), col_axes=(2,)))
print(tensor_contract(T, Vector.from_data([2, 4])))
```

In this example, `alpha(v)` is displayed as a real number whenever its
imaginary part is zero, even though the library computes internally with
complex scalars.

Tensor operations are available both in Python's natural form, such as
`A + B` and `A @ B`, and as explicit public functions such as
`tensor_sum(A, B)` and `tensor_product(A, B)`.

For display and exploration, `Tensor.as_matrix(row_axes=..., col_axes=...)`
lets you regroup the tensor indices into a 2D matrix view, and
`Tensor.display_matrix(...)` prepares that view for cleaner notebook output.
For contractions, `Tensor.contract(other)` and `tensor_contract(A, B)` contract
the last covariant index of the left tensor with the first contravariant index
of the right tensor. Evaluation remains a separate notion, represented for
instance by `Covector.evaluate(vector)` and `alpha(vector)`.

## Design Principles

This project follows a few guiding principles:

- **Mathematical meaning first**  
  The implementation should reflect the conceptual role of each object.

- **Explicit over implicit**  
  The code should make structure visible rather than hiding it behind shortcuts.

- **Minimal operations at first**  
  Operations should only be introduced once their semantics are clear.

- **Educational value**  
  The library should help students understand tensor algebra, not just manipulate arrays.

## Scalar Convention

The library uses a single scalar representation internally:

- all tensor components are stored as `complex128`;
- all algebraic operations are carried out over complex scalars.

At the public interface level, however, values are shown as real whenever
possible:

- real inputs such as `1` or `2.5` are accepted naturally and converted
  internally to complex form;
- scalar outputs are returned as `float` when their imaginary part is zero;
- tensor representations display entries as real numbers whenever the
  imaginary part vanishes.

This keeps the implementation algebraically uniform while preserving a clean
and intuitive interface for real-valued examples.

## Workflow Organization

- `src/tensors/` contains the modular implementation of the library.
- `tests/` contains the automated test suite.
- `notebooks/` is reserved for exploration, examples, and experimentation.

## Planned Next Steps

Likely next steps include:

- contractions;
- and more precise algebraic operations.

## Project Status

This project is currently in an early but already structured stage of development.

Its purpose is twofold:

- to serve as a learning tool;
- and to provide the foundation for a more robust and conceptually clean tensor algebra library.
