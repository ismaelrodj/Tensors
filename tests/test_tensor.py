from tensors import Tensor


def test_tensor_from_data_creates_tensor_correctly() -> None:
    tensor = Tensor.from_data([[1, 2], [3, 4]], tensor_type=(1, 1))

    assert tensor.tensor_type == (1, 1)
    assert tensor.shape == (2, 2)
    assert tensor.rank == 2
    assert tensor[0, 1] == 2.0


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