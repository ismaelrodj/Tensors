from tensors import Tensor, tensor_contract, tensor_product, tensor_sum


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


def test_tensor_addition_preserves_type_and_values() -> None:
    left = Tensor.from_data([[1, 2], [3, 4]], tensor_type=(1, 1))
    right = Tensor.from_data([[10, 20], [30, 40]], tensor_type=(1, 1))

    result = left + right

    assert isinstance(result, Tensor)
    assert result.tensor_type == (1, 1)
    assert result.shape == (2, 2)
    assert result[0, 0] == 11 + 0j
    assert result[1, 1] == 44 + 0j


def test_tensor_addition_rejects_different_tensor_types() -> None:
    left = Tensor.from_data([[1, 2], [3, 4]], tensor_type=(1, 1))
    right = Tensor.from_data([[10, 20], [30, 40]], tensor_type=(2, 0))

    try:
        left + right
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_tensor_addition_rejects_different_shapes() -> None:
    left = Tensor.from_data(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        tensor_type=(2, 1),
    )
    right = Tensor.from_data(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]],
        ],
        tensor_type=(2, 1),
    )

    try:
        left + right
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_tensor_repr_hides_zero_imaginary_parts() -> None:
    tensor = Tensor.from_data([[1, 2 + 0j], [3, 4 + 5j]], tensor_type=(1, 1))

    representation = repr(tensor)

    assert "1.0" in representation
    assert "2.0" in representation
    assert "(4+5j)" in representation


def test_tensor_product_combines_components_and_tensor_type() -> None:
    left = Tensor.from_data([[1, 2], [3, 4]], tensor_type=(1, 1))
    right = Tensor.from_data([5, 6], tensor_type=(1, 0))

    result = tensor_product(left, right)

    assert isinstance(result, Tensor)
    assert result is not NotImplemented
    assert result.tensor_type == (2, 1)
    assert result.shape == (2, 2, 2)
    assert result[0, 0, 0] == 5 + 0j
    assert result[1, 1, 1] == 24 + 0j


def test_tensor_product_of_vector_and_covector_returns_rank_two_tensor() -> None:
    from tensors import Covector, Vector

    vector = Vector.from_data([1, 2])
    covector = Covector.from_data([3, 4])

    result = tensor_product(vector, covector)

    assert isinstance(result, Tensor)
    assert result.tensor_type == (1, 1)
    assert result.shape == (2, 2)
    assert result[0, 0] == 3 + 0j
    assert result[1, 0] == 6 + 0j
    assert result[1, 1] == 8 + 0j


def test_tensor_product_rejects_non_tensor() -> None:
    tensor = Tensor.from_data([1, 2], tensor_type=(1, 0))

    try:
        tensor_product(tensor, "invalid")  # type: ignore[arg-type]
        assert False, "Expected TypeError"
    except TypeError:
        pass


def test_tensor_product_function_matches_matmul_result() -> None:
    left = Tensor.from_data([1, 2], tensor_type=(1, 0))
    right = Tensor.from_data([3, 4], tensor_type=(0, 1))

    via_function = tensor_product(left, right)
    via_operator = left @ right

    assert via_operator is not NotImplemented
    assert via_function.tensor_type == (1, 1)
    assert via_function.shape == (2, 2)
    assert (via_function.components == via_operator.components).all()


def test_tensor_matmul_matches_tensor_product() -> None:
    left = Tensor.from_data([[1, 2], [3, 4]], tensor_type=(1, 1))
    right = Tensor.from_data([5, 6], tensor_type=(1, 0))

    via_operator = left @ right
    via_function = tensor_product(left, right)

    assert via_operator is not NotImplemented
    assert via_operator.tensor_type == (2, 1)
    assert via_operator.shape == (2, 2, 2)
    assert (via_operator.components == via_function.components).all()


def test_tensor_matmul_returns_not_implemented_for_non_tensor() -> None:
    tensor = Tensor.from_data([1, 2], tensor_type=(1, 0))

    result = tensor.__matmul__("invalid")

    assert result is NotImplemented


def test_tensor_sum_function_matches_operator_result() -> None:
    left = Tensor.from_data([[1, 2], [3, 4]], tensor_type=(1, 1))
    right = Tensor.from_data([[10, 20], [30, 40]], tensor_type=(1, 1))

    via_function = tensor_sum(left, right)
    via_operator = left + right

    assert via_function.tensor_type == (1, 1)
    assert via_function.shape == (2, 2)
    assert via_operator is not NotImplemented
    assert (via_function.components == via_operator.components).all()


def test_as_matrix_groups_axes_into_rows_and_columns() -> None:
    tensor = Tensor.from_data(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        tensor_type=(2, 1),
    )

    matrix = tensor.as_matrix(row_axes=(0, 1), col_axes=(2,))

    assert matrix.shape == (4, 2)
    assert matrix[0, 0] == 1 + 0j
    assert matrix[3, 1] == 8 + 0j


def test_as_matrix_rejects_missing_or_repeated_axes() -> None:
    tensor = Tensor.from_data(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        tensor_type=(2, 1),
    )

    try:
        tensor.as_matrix(row_axes=(0, 1), col_axes=(1,))
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_display_matrix_hides_zero_imaginary_parts() -> None:
    tensor = Tensor.from_data(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        tensor_type=(2, 1),
    )

    matrix = tensor.display_matrix(row_axes=(0, 1), col_axes=(2,))

    assert matrix.shape == (4, 2)
    assert matrix.dtype.kind in {"f", "c"}
    assert matrix[0, 0] == 1.0
    assert matrix[3, 1] == 8.0


def test_display_slice_hides_zero_imaginary_parts_for_array_slice() -> None:
    tensor = Tensor.from_data(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        tensor_type=(2, 1),
    )

    slice_view = tensor.display_slice(0)

    assert slice_view.shape == (2, 2)
    assert slice_view.dtype.kind in {"f", "c"}
    assert slice_view[0, 0] == 1.0
    assert slice_view[1, 1] == 4.0


def test_display_slice_formats_scalar_output() -> None:
    tensor = Tensor.from_data([[1, 2], [3, 4 + 5j]], tensor_type=(1, 1))

    assert tensor.display_slice((0, 1)) == 2.0
    assert tensor.display_slice((1, 1)) == 4 + 5j


def test_tensor_contract_matrix_on_vector_returns_vector() -> None:
    from tensors import Vector

    tensor = Tensor.from_data([[1, 3], [5, 1]], tensor_type=(1, 1))
    vector = Vector.from_data([2, 4])

    result = tensor_contract(
        tensor,
        vector,
        contravariant_axis=("right", 0),
        covariant_axis=("left", 1),
    )

    assert isinstance(result, Vector)
    assert result.tensor_type == (1, 0)
    assert result.shape == (2,)
    assert result[0] == 14 + 0j
    assert result[1] == 14 + 0j


def test_tensor_contract_covector_on_vector_returns_scalar() -> None:
    from tensors import Covector, Vector

    covector = Covector.from_data([2, 3])
    vector = Vector.from_data([5, 7])

    result = tensor_contract(
        covector,
        vector,
        contravariant_axis=("right", 0),
        covariant_axis=("left", 0),
    )

    assert result == 31.0


def test_tensor_contract_tensor_on_matrix_returns_tensor() -> None:
    tensor = Tensor.from_data([[1, 0], [0, 1]], tensor_type=(1, 1))
    matrix = Tensor.from_data([[2, 3], [4, 5]], tensor_type=(1, 1))

    result = tensor_contract(
        tensor,
        matrix,
        contravariant_axis=("right", 0),
        covariant_axis=("left", 1),
    )

    assert isinstance(result, Tensor)
    assert result.tensor_type == (1, 1)
    assert result.shape == (2, 2)
    assert result[0, 0] == 2 + 0j
    assert result[1, 1] == 5 + 0j


def test_tensor_contract_tensor_on_covector_with_explicit_axes_returns_covector() -> None:
    from tensors import Covector

    tensor = Tensor.from_data([[1, 3], [5, 1]], tensor_type=(1, 1))
    covector = Covector.from_data([2, 4])

    result = tensor_contract(
        tensor,
        covector,
        contravariant_axis=("left", 0),
        covariant_axis=("right", 0),
    )

    assert isinstance(result, Covector)
    assert result.tensor_type == (0, 1)
    assert result.shape == (2,)
    assert result[0] == 22 + 0j
    assert result[1] == 10 + 0j


def test_tensor_contract_matrix_on_vector_matches_expected_type_and_values() -> None:
    left = Tensor.from_data([[1, 3], [5, 1]], tensor_type=(1, 1))

    from tensors import Vector

    right = Vector.from_data([2, 4])

    result = tensor_contract(
        left,
        right,
        contravariant_axis=("right", 0),
        covariant_axis=("left", 1),
    )

    assert result.tensor_type == (1, 0)
    assert result.shape == (2,)
    assert result[0] == 14 + 0j
    assert result[1] == 14 + 0j


def test_tensor_contract_function_accepts_explicit_axes() -> None:
    from tensors import Covector

    left = Tensor.from_data([[1, 3], [5, 1]], tensor_type=(1, 1))
    right = Covector.from_data([2, 4])

    result = tensor_contract(
        left,
        right,
        contravariant_axis=("left", 0),
        covariant_axis=("right", 0),
    )

    assert isinstance(result, Covector)
    assert result[0] == 22 + 0j
    assert result[1] == 10 + 0j


def test_tensor_contract_requires_explicit_axes() -> None:
    from tensors import Covector

    covector = Covector.from_data([1, 2])
    vector = Covector.from_data([3, 4])

    try:
        tensor_contract(covector, vector)  # type: ignore[call-arg]
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_tensor_contract_rejects_non_contravariant_axis() -> None:
    from tensors import Covector

    tensor = Tensor.from_data([[1, 2], [3, 4]], tensor_type=(1, 1))
    covector = Covector.from_data([1, 2])

    try:
        tensor_contract(
            tensor,
            covector,
            contravariant_axis=("left", 1),
            covariant_axis=("right", 0),
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_tensor_contract_rejects_incompatible_shapes() -> None:
    from tensors import Vector

    tensor = Tensor.from_data([[1, 2], [3, 4]], tensor_type=(1, 1))
    vector = Vector.from_data([1, 2, 3])

    try:
        tensor_contract(
            tensor,
            vector,
            contravariant_axis=("right", 0),
            covariant_axis=("left", 1),
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_tensor_contract_requires_both_axis_specs() -> None:
    from tensors import Vector

    tensor = Tensor.from_data([[1, 2], [3, 4]], tensor_type=(1, 1))
    vector = Vector.from_data([1, 2])

    try:
        tensor_contract(tensor, vector, contravariant_axis=("right", 0))
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_tensor_contract_rejects_axes_from_same_tensor() -> None:
    tensor = Tensor.from_data([[1, 2], [3, 4]], tensor_type=(1, 1))
    matrix = Tensor.from_data([[5, 6], [7, 8]], tensor_type=(1, 1))

    try:
        tensor_contract(
            tensor,
            matrix,
            contravariant_axis=("left", 0),
            covariant_axis=("left", 1),
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass
