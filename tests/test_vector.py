from tensors import Vector


def test_vector_from_data_creates_vector_correctly() -> None:
    vector = Vector.from_data([1, 2, 3])

    assert vector.tensor_type == (1, 0)
    assert vector.shape == (3,)
    assert vector.rank == 1
    assert vector.components.dtype.name == "complex128"
    assert vector[2] == 3 + 0j


def test_vector_rejects_non_1d_data() -> None:
    try:
        Vector.from_data([[1, 2], [3, 4]])
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_vector_scalar_multiplication_preserves_subclass() -> None:
    vector = Vector.from_data([1, 2, 3])

    scaled = vector * 3

    assert isinstance(scaled, Vector)
    assert scaled.tensor_type == (1, 0)
    assert scaled[0] == 3 + 0j
    assert scaled[2] == 9 + 0j


def test_vector_supports_right_scalar_multiplication() -> None:
    vector = Vector.from_data([1, 2, 3])

    scaled = 2 * vector

    assert isinstance(scaled, Vector)
    assert scaled[1] == 4 + 0j


def test_vector_supports_complex_inputs_and_scalars() -> None:
    vector = Vector.from_data([1, 2 + 3j, 4])

    scaled = (1 - 1j) * vector

    assert vector[0] == 1 + 0j
    assert vector[1] == 2 + 3j
    assert scaled[1] == 5 + 1j


def test_vector_addition_preserves_subclass() -> None:
    left = Vector.from_data([1, 2, 3])
    right = Vector.from_data([4, 5, 6])

    result = left + right

    assert isinstance(result, Vector)
    assert result.tensor_type == (1, 0)
    assert result[0] == 5 + 0j
    assert result[2] == 9 + 0j
