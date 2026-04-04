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
from tensors import Tensor, Vector, Covector

v = Vector.from_data([1, 2, 3])
alpha = Covector.from_data([4, 5, 6])
T = Tensor.from_data([[1, 2], [3, 4]], tensor_type=(1, 1))

print(v)
print(alpha)
print(T)
```

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

## Workflow Organization

- `src/tensors/` contains the modular implementation of the library.
- `tests/` contains the automated test suite.
- `notebooks/` is reserved for exploration, examples, and experimentation.

## Planned Next Steps

Likely next steps include:

- scalar multiplication;
- addition of compatible tensors;
- evaluation of a covector on a vector;
- tensor products;
- and more precise algebraic operations.

## Project Status

This project is currently in an early but already structured stage of development.

Its purpose is twofold:

- to serve as a learning tool;
- and to provide the foundation for a more robust and conceptually clean tensor algebra library.