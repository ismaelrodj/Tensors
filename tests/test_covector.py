from tensors import Covector


def test_covector_from_data_creates_covector_correctly() -> None:
    covector = Covector.from_data([4, 5, 6])

    assert covector.tensor_type == (0, 1)
    assert covector.shape == (3,)
    assert covector.rank == 1
    assert covector.components.dtype.name == "complex128"
    assert covector[1] == 5 + 0j


def test_covector_rejects_non_1d_data() -> None:
    try:
        Covector.from_data([[1, 2], [3, 4]])
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_covector_scalar_multiplication_preserves_subclass() -> None:
    covector = Covector.from_data([4, 5, 6])

    scaled = covector * 2

    assert isinstance(scaled, Covector)
    assert scaled.tensor_type == (0, 1)
    assert scaled[0] == 8 + 0j
    assert scaled[2] == 12 + 0j


def test_covector_supports_right_scalar_multiplication() -> None:
    covector = Covector.from_data([4, 5, 6])

    scaled = 0.5 * covector

    assert isinstance(scaled, Covector)
    assert scaled[1] == 2.5 + 0j


def test_covector_supports_complex_inputs_and_repr() -> None:
    covector = Covector.from_data([4, 5j, 6])

    assert covector[0] == 4 + 0j
    assert covector[1] == 5j
    assert "4.0" in repr(covector)


def test_covector_addition_preserves_subclass() -> None:
    left = Covector.from_data([4, 5, 6])
    right = Covector.from_data([1, 2, 3])

    result = left + right

    assert isinstance(result, Covector)
    assert result.tensor_type == (0, 1)
    assert result[0] == 5 + 0j
    assert result[2] == 9 + 0j
