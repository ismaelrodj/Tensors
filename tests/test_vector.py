from tensors import Vector

def test_vector_from_data_creates_vector_correctly() -> None:
    vector = Vector.from_data([1, 2, 3])

    assert vector.tensor_type == (1, 0)
    assert vector.shape == (3,)
    assert vector.rank == 1
    assert vector[2] == 3.0


def test_vector_rejects_non_1d_data() -> None:
    try:
        Vector.from_data([[1, 2], [3, 4]])
        assert False, "Expected ValueError"
    except ValueError:
        pass