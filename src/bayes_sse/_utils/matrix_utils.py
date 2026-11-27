from collections.abc import Iterable
from typing import Literal

import numpy as np

from bayes_sse._utils.quaternion_utils import Quaternion, quaternion
from bayes_sse.typing import Float64FlatVector, Float64Matrix, Float64Vector


def generate_symmetric_sigma_points(
    center: Float64Vector,
    offsets: Float64Matrix,
    scale: float | np.float64,
) -> Float64Matrix:
    """
    Generate symmetric sigma points.

    The columns are ordered as [x, x + sΔ, x − sΔ].
    """
    if center.ndim != 2 or center.shape[1] != 1:
        raise ValueError(f"center must have shape (n, 1), got {center.shape}")

    if offsets.ndim != 2 or offsets.shape[0] != center.shape[0]:
        raise ValueError(
            "offsets must have shape (n, k), with the same n as center; "
            f"got center={center.shape}, offsets={offsets.shape}"
        )

    scaled_offsets = scale * offsets

    return np.hstack(
        (
            center,
            center + scaled_offsets,
            center - scaled_offsets,
        )
    )


def generate_third_degree_cubature_points(
    center: Float64Vector,
    offsets: Float64Matrix,
) -> Float64Matrix:
    """
    Generate 3rd degree spherical cubature points.

    The columns are ordered as [x + sΔ, x − sΔ].
    """
    if center.ndim != 2 or center.shape[1] != 1:
        raise ValueError(f"center must have shape (n, 1), got {center.shape}")

    if offsets.ndim != 2 or offsets.shape[0] != center.shape[0]:
        raise ValueError(
            "offsets must have shape (n, k), with the same n as center; "
            f"got center={center.shape}, offsets={offsets.shape}"
        )

    unit_offsets = np.hstack((+np.eye(center.shape[0]), -np.eye(center.shape[0])))
    return center + offsets @ unit_offsets


def generate_random_sqrt_mtrix(n: int) -> Float64Matrix:
    spd_matrix = generate_random_spd_matrix(n)
    return np.linalg.cholesky(spd_matrix)


def generate_random_spd_matrix(n: int) -> Float64Matrix:
    rng = np.random.default_rng()
    matrix = rng.standard_normal((n, n))

    spd_matrix = matrix @ matrix.T
    spd_matrix += n * np.eye(n)

    try:
        _ = np.linalg.cholesky(spd_matrix)
    except np.linalg.LinAlgError:
        raise RuntimeError("Verification Failed: The matrix is not positive definite.")

    return spd_matrix


def ensure_positive_semi_definite(matrix: Float64Matrix) -> Float64Matrix:
    min_eig = np.min(np.real(np.linalg.eigvals(matrix)))
    if min_eig < 0:
        matrix -= 10 * min_eig * np.eye(*matrix.shape)

    return matrix


def is_square_2d_array(matrix: Iterable) -> bool:
    if np.ndim(matrix) != 2:
        return False

    (row_size, column_size) = np.shape(matrix)
    return row_size == column_size


def shape_square_2d_array(matrix: Iterable) -> int:
    if not is_square_2d_array(matrix):
        raise Exception("matrix is not square")

    row_size, _ = np.shape(matrix)
    return row_size


def ensure_diagonal_matrix(matrix: Iterable) -> Float64Matrix:
    return np.diag(np.diag(np.asarray(matrix)))


def svd_sqrt(matrix: Float64Matrix) -> Float64Matrix:
    """
    Produce square root decomposition from standard svd decomposition via following equation
        result = 0.5*(u + v) @ sqrt(s)

    Where u, s, v defined as:
    [u, s, v_h] = svd(x) produces a diagonal matrix s, of the same dimension as x and with non negative diagonal elements in
    decreasing order, and unitary matrices U and V_h so, that X = u*s*v_h.


    :param matrix: input matrix.
    :return: square root decomposition of input matrix (0.5*(u + v) * sqrt(s), where u, v, s - results of svd decomposition, v is vh.T).
    """
    u, s, v_h = np.linalg.svd(matrix)
    v = v_h.T
    return np.atleast_2d(0.5 * (u + v) @ np.sqrt(s))


def cholesky_update(
    a: Float64Matrix,
    x: Float64Matrix,
    sign: Literal["+", "-"],
) -> Float64Matrix:
    """
    Update or downdate an upper triangular Cholesky factor.

    :param a: upper triangular factor of shape (n, n), where A = a.T @ a.
    :param x: matrix of shape (n, k), with one update vector per column.
    :param sign: '+' for an update or '-' for a downdate.
    :return: upper triangular Cholesky factor of A + x @ x.T or A - x @ x.T,
        respectively.
    :raises ValueError: if sign is invalid.
    :raises numpy.linalg.LinAlgError: if a pivot is zero or the downdated
        matrix is not positive definite.
    """
    match sign:
        case "+":
            multiplier = 1
        case "-":
            multiplier = -1
        case _:
            raise ValueError(f"Unknown sign {sign!r}")

    upper = a.copy()
    for j in range(x.shape[1]):
        remaining = x[:, j]
        for k in range(upper.shape[0]):
            diagonal = upper[k, k]
            if diagonal == 0:
                raise np.linalg.LinAlgError("Zero Cholesky pivot")

            value = remaining[0]
            s = value / diagonal
            if sign == "+":
                pivot = np.hypot(diagonal, value)
            else:
                if abs(s) >= 1:
                    raise np.linalg.LinAlgError("Downdate is not positive definite")
                pivot = abs(diagonal) * np.sqrt((1 - s) * (1 + s))

            c = pivot / diagonal
            upper[k, k] = pivot
            row = upper[k, k + 1 :]
            tail = remaining[1:]
            row[:] = (row + multiplier * s * tail) / c
            remaining = c * tail - s * row

    return upper


def cartesian_product(*arrays) -> Float64Matrix:
    """
    Calculate cartesian product of n-dimensional set, i.e. calculate all possible combinations between items of input vectors.
    :param arrays: n-dimensional set, i.e. list of arrays where an array is a n-dim vector from the set.
    :return: list of all possible combinations between different elements
    """
    dim = len(arrays)
    return np.transpose(
        np.meshgrid(*arrays, indexing="ij"), np.roll(np.arange(dim + 1), -1)
    ).reshape(-1, dim)


def get_shape_at(
    array: Float64Matrix,
    axis: int,
    default: int | None = None,
) -> int | None:
    """
    Calculate shape of an array along specified dimension. If dimension of an array less than specified axis than return default value.
    :param array: input array
    :param axis: axis along that shape needs to be calculated
    :param default: default value in case when array dimension less that requested axis
    :return:
    """
    shape = np.shape(array)
    shape_item = dict(enumerate(shape)).get(axis)
    return default if shape_item is None else shape_item


def vect_to_quaternion(vect: Float64Matrix) -> Quaternion:
    """
    Convert 3D vector to quaternion representation.
    :param vect: 3D vector (with components x, y, z)
    :return: quaternion representation of input vector
    """
    return quaternion(0, vect[0], vect[1], vect[2])


def get_locked_copy(arr: Float64Matrix) -> Float64Matrix:
    """
    Creates immutable (writeable flag is set to false) copy of input array
    :param arr: input array that should be copied and locked.
    :return: not writable copy of input array
    """
    result = np.asarray(arr).copy()
    result.flags["WRITEABLE"] = False
    return result


def divide_inv(
    a: Float64Matrix,
    b: Float64Matrix,
) -> Float64Matrix:
    """
    Solve linear equation
        x * a = b
    for x.
    :param a: np array, 1-st matrix.
    :param b: np array, 2-nd matrix.
    :return: np array, Solution of equation: 'x * a = b' for x.
    """
    x, *_ = np.linalg.lstsq(b.T, a.T, rcond=None)
    return x.T


def divide(
    a: Float64Matrix,
    b: Float64Matrix,
) -> Float64Matrix:
    """
    Solve linear equation
        a * x = b
    for x.
    :param a: np array, 1-st matrix.
    :param b: np array, 2-nd matrix.
    :return: np array, Solution of equation: 'a * x = b' for x.
    """
    x, *_ = np.linalg.lstsq(b, a, rcond=None)
    return x


def pad_1d_list_with_zero(
    array: Float64Matrix,
    length: int,
) -> Float64Matrix:
    """
    Pad initial 1D array with zeros.
    Length of initial array will be equal to length
    :param array: np array, input array.
    :param length: int, length of result array (with zeros)
    :return: np array, array with specified length filled by zero elements
    """
    zero_array = np.zeros(length - len(array), dtype=array.dtype)
    return np.concatenate((array, zero_array), axis=0)


def cum_sum_with_last_equal_to(
    array: Float64FlatVector,
    last_element: float | np.float64,
) -> Float64FlatVector:
    array_type = array.dtype

    cumulative_sum = np.cumsum(array)
    cumulative_sum[-1] = last_element

    return cumulative_sum.astype(array_type)


def cum_sum_with_last_equal_to_one(array: Float64FlatVector) -> Float64FlatVector:
    return cum_sum_with_last_equal_to(array, 1)
