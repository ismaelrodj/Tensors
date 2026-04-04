from tensors import Covector


def test_covector_from_data_creates_covector_correctly() -> None:
    covector = Covector.from_data([4, 5, 6])

    assert covector.tensor_type == (0, 1)
    assert covector.shape == (3,)
    assert covector.rank == 1
    assert covector[1] == 5.0


def test_covector_rejects_non_1d_data() -> None:
    try:
        Covector.from_data([[1, 2], [3, 4]])
        assert False, "Expected ValueError"
    except ValueError:
        pass