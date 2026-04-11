from tensors import Tensor


def test_tensor_from_data_creates_tensor_correctly() -> None:
    tensor = Tensor.from_data([[1, 2], [3, 4]], tensor_type=(1, 1))

    assert tensor.tensor_type == (1, 1)
    assert tensor.shape == (2, 2)
    assert tensor.rank == 2
    assert tensor.components.dtype.name == "complex128"
    assert tensor[0, 1] == 2 + 0j


def test_tensor_rejects_wrong_rank_for_type() -> None:
    try:
        Tensor.from_data([1, 2, 3], tensor_type=(1, 1))
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_tensor_rejects_invalid_tensor_type_length() -> None:
    try:
        Tensor.from_data([1, 2, 3], tensor_type=(1, 0, 0))  # type: ignore[arg-type]
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_tensor_rejects_negative_tensor_type_entry() -> None:
    try:
        Tensor.from_data([1, 2, 3], tensor_type=(1, -1))
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_tensor_scalar_multiplication_preserves_type_and_values() -> None:
    tensor = Tensor.from_data([[1, 2], [3, 4]], tensor_type=(1, 1))

    scaled = tensor * 2

    assert isinstance(scaled, Tensor)
    assert scaled.tensor_type == (1, 1)
    assert scaled.shape == (2, 2)
    assert scaled[0, 0] == 2 + 0j
    assert scaled[1, 1] == 8 + 0j


def test_tensor_supports_right_scalar_multiplication() -> None:
    tensor = Tensor.from_data([[1, 2], [3, 4]], tensor_type=(1, 1))

    scaled = 0.5 * tensor

    assert isinstance(scaled, Tensor)
    assert scaled.tensor_type == (1, 1)
    assert scaled[0, 1] == 1 + 0j
    assert scaled[1, 0] == 1.5 + 0j


def test_tensor_supports_complex_scalar_multiplication() -> None:
    tensor = Tensor.from_data([[1, 2], [3, 4]], tensor_type=(1, 1))

    scaled = tensor * (1 + 2j)

    assert scaled[0, 0] == 1 + 2j
    assert scaled[0, 1] == 2 + 4j


def test_tensor_repr_hides_zero_imaginary_parts() -> None:
    tensor = Tensor.from_data([[1, 2 + 0j], [3, 4 + 5j]], tensor_type=(1, 1))

    representation = repr(tensor)

    assert "1.0" in representation
    assert "2.0" in representation
    assert "(4+5j)" in representation
