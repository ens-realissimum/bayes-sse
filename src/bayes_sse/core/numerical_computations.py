from functools import lru_cache
from typing import Tuple, Union

import numpy as np
from scipy.special import gamma

from bayes_sse._utils.matrix_utils import cartesian_product, get_shape_at


@lru_cache()
def eval_fifth_degree_cubature_rule(dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate cubature points and weights for 5-th degree Cubature Rule.
    :param dim: dimension of space for which 5-th degree Cubature Rule must be calculated.
    :return: Tuple of points matrix (points) and weights vector (weights).
    """
    num = 2 * dim ** 2 + 1
    points = np.zeros((dim, num))
    weights = np.zeros(num)

    points[:, 0] = np.zeros(dim)
    weights[0] = 2 / (dim + 2)

    for i in range(dim):
        points[i, i + 1] = np.sqrt(dim / 2 + 1)
        weights[i + 1] = (4 - dim) / (2 * (dim + 2) ** 2)

        points[i, i + dim + 1] = -np.sqrt(dim / 2 + 1)
        weights[i + dim + 1] = (4 - dim) / (2 * (dim + 2) ** 2)

    count = 2 * dim
    x_1 = np.sqrt(dim / 4 + 1 / 2)
    x_2 = 1 / ((dim + 2) ** 2)

    for i in range(dim):
        for j in range(i + 1, dim):
            count += 1
            points[i, count] = x_1
            points[j, count] = x_1
            weights[count] = x_2

            count += 1
            points[i, count] = x_1
            points[j, count] = -x_1
            weights[count] = x_2

            count += 1
            points[i, count] = -x_1
            points[j, count] = x_1
            weights[count] = x_2

            count += 1
            points[i, count] = -x_1
            points[j, count] = -x_1
            weights[count] = x_2

    points = points * np.sqrt(2)

    return points, weights


@lru_cache()
def eval_laguerre_quadrature_rule(order: int, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate cubature points using Gauss-Laguerre quadratures.
    Determines the abscisas (points) and weights (weights) for the Chebyshev-Laguerre quadrature of order n > 1, on the interval [0, +infinity].
    This is due to the fact that the companion matrix (of the n'th degree Laguerre polynomial) is now constructed as a symmetrical
    matrix, guaranteeing that all the eigenvalues (roots) will be real.

    For the details see:
        https://en.wikipedia.org/wiki/Classical_orthogonal_polynomials#Chebyshev_polynomials

    :param order: nth degree Laguerre polynomial (subscript index of polynomial)
    :param alpha: parameter of Laguerre polynomial (superscript index polynomial), alpha = n / 2 - 1, where n - dimension of state space
    :return:
        points: set of generated points;
        weights: set of corresponding weights for every point.
    """
    i = np.arange(1, order + 1)
    a = (2 * i - 1) + alpha
    b = np.sqrt(i[:-1] * (np.arange(1, order) + alpha))
    cm = np.diag(a) + np.diag(b, 1) + np.diag(b, -1)

    d, v = np.linalg.eig(cm)
    points = np.sort(d, axis=0)
    ind = np.argsort(d, axis=0)
    v = v[:, ind].T
    weights = gamma(alpha + 1) * v[:, 0] ** 2

    return points, weights


def intersect_line_hyper_sphere(line: np.ndarray, sphere: np.ndarray, n: int) -> np.ndarray:
    """
    Return intersection points between a line in hyper space and a hyper sphere.
        For additional details please see https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    :param line: array of line coordinates and in hyper space, for instance [x0 y0 z0 ... n0 dx dy dz ... dn];
    :param sphere: sphere  array of center points and radius of hyper sphere, for instance [x_center y_center z_center ... n_center  R];
    :param n: dimension of hyper space.
    :return: intersection points between a unit hyper sphere and its axis
    """
    diff_centers = line[:, :n] - sphere[:, :-1]

    # equation coefficients
    a = np.sum(line[:, n:] * line[:, n:], axis=1)
    b = 2 * np.sum(diff_centers * line[:, n:], axis=1)
    c = np.sum(diff_centers * diff_centers, axis=1) - sphere[:, -1] * sphere[:, -1]

    # solve equation
    delta = b * b - 4 * a * c

    # initialize empty results
    points = np.nan * np.ones((n, 2 * delta.shape[0]))

    # process couples with two intersection points
    idx, *_ = np.nonzero(delta > np.spacing(1))

    if np.size(idx) > 0:
        # delta positive: find two roots of second order equation
        u1, *_ = np.linalg.lstsq(
            np.atleast_2d(2),
            np.atleast_2d(-b[idx] - np.sqrt(delta[idx])),
            rcond=None
        )
        u1 = np.squeeze(u1.T) / a[idx]

        u2, *_ = np.linalg.lstsq(
            np.atleast_2d(2),
            np.atleast_2d(-b[idx] + np.sqrt(delta[idx])),
            rcond=None
        )
        u2 = np.squeeze(u2.T) / a[idx]

        # convert into n-D coordinate
        points[:, idx] = line[idx, 0: n] + u1 * line[idx, n:]
        points[:, idx + np.max(delta.shape)] = line[idx, 0: n] + u2 * line[idx, n:]

    # process couples with one intersection point
    idx, *_ = np.nonzero(delta < np.spacing(1))

    if np.size(idx) > 0:
        # delta around zero: find unique root, and convert to n-D coord.
        u, *_ = np.linalg.lstsq(
            np.atleast_2d(2),
            np.atleast_2d(-b[idx]),
            rcond=None
        )
        u = np.squeeze(u.T) / a[idx]

        # convert into n-D coordinate
        pts = line[idx, :] + u * line[idx, n:]
        points[:, idx] = pts
        points[:, idx + np.max(delta.shape)] = points

    return points


@lru_cache()
def intersect_unit_vector_hyper_sphere(n: int) -> np.ndarray:
    """
    Return intersection points between a unit hyper sphere and its axis.
        For additional details please see https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    :param n: hyper space dimension.
    :return: intersection points between a unit hyper sphere and its axis.
    """
    # unit sphere. [x_center y_center z_center  R] for 3D:
    #  [
    #     [0 0 0]  [1]
    #     [0 0 0]  [1]
    #     [0 0 0]  [1]
    #  ]
    sphere = np.hstack((np.zeros((n, n)), np.ones((n, 1))))

    # unit vectors. for 3D ([x0 y0 z0] is [0 0 0]):
    #  [
    #     [x0 y0 z0]  [dx 0 0]
    #     [x0 y0 z0]  [0 dy 0]
    #     [x0 y0 z0]  [0 0 dz]
    #  ]
    line = np.hstack((np.zeros((n, n)), np.eye(n)))

    return intersect_line_hyper_sphere(line, sphere, n)


@lru_cache()
def eval_cubature_quadrature_points(dim: int, order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw cubature-quadrature points for cubature-quadrature Kalman filter
        Cubature points calculated as intersection of unit hyper-sphere (dimension equal to dim) and its axes.
        Quadrature points calculated as solution of Chebyshev-Laguerre polynoms with order n' and alpha = n / 2 - 1.
    :param dim: space dimension (state space dimension);
    :param order: order of the laguerre quadrature rule.
    :return: Tuple of
        points: matrix of cubature points;
        weights: array of corresponded weights.
    """
    alpha = dim / 2 - 1
    cubature_points = intersect_unit_vector_hyper_sphere(dim)
    quadrature_points, w = eval_laguerre_quadrature_rule(order, alpha)

    num = 2 * dim * order
    weights = np.zeros(num)
    points = np.zeros((dim, num))

    for i in range(order):
        start = 2 * dim * i
        end = 2 * dim * (i + 1)

        points[:, start: end] = np.sqrt(2 * quadrature_points[i]) * cubature_points
        weights[start: end] = w[i] / (2 * dim * gamma(dim / 2))

    return points, weights


@lru_cache()
def eval_gauss_hermite_rule(order: int, dim: Union[int, None]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate points and weights according to Gauss-Hermite quadrature rule.
    Weight function is chosen to be the standard Gaussian density with zero mean and unit variance N(0; I).
    The interval of interest is chosen to be (-infinity; +infinity).
    According to the fundamental theorem of Gauss-Hermite quadrature, the quadrature points are chosen to be the zeros of the m-th
    order Hermite polynomial.
    Since the zeros of the Hermite polynomials are distinct,
    it is noteworthy that the determinant of the coefficient matrix in is the well known Vandermonde's determinant that is nonzero.
    For an m-point quadrature scheme, the resulting quadrature rule is exact for all polynomials of degree  <= 2m - 1

    Suppose J is a symmetric tridiagonal matrix with zero diagonal elements and J(i, i+1) = sqrt(i/2), 1 <= i <= m-1
    Then the quadrature point KSI is taken to be KSI(k) = sqrt(2)*eigenvalue_J(k);
    where "l is the l-th eigenvalue of J;
    and the corresponding weight w(k) = v(k, 1)^2 where v(k, 1) is the first element of the k-th normalized eigenvector of J.

    :param order: Order of Gauss-Hermite polynomial.
    :param dim: dimension.
    :return: Tuple of
        points: matrix of generated points.
        weights: array of corresponding weights.
    """
    elements = np.sqrt(np.arange(1, order) / 2)
    eig_values, eig_vectors = np.linalg.eig(
        np.diag(elements, 1) + np.diag(elements, -1)
    )

    points = np.atleast_2d(np.sort(eig_values) * np.sqrt(2))

    weights = eig_vectors[:, np.argsort(eig_values)].T
    weights = np.sqrt(np.pi) * weights[:, 0] ** 2
    weights /= np.sum(weights)

    if dim is not None:
        points = cartesian_product(*(np.tile(points, (dim, 1))))

        weights = cartesian_product(*(np.tile(np.atleast_2d(weights), (dim, 1))))
        weights = np.prod(weights, 1)

    return points.T, weights


def is_reasonable_sghr(accuracy_level: int, dimension: int, index: np.ndarray) -> bool:
    return True if np.sum(index) <= accuracy_level + dimension - 1 else False


def generate_next_index_sparse_gauss_hermite_rule(accuracy_level: int, index: np.ndarray, old_index: np.ndarray) -> np.ndarray:
    dim = np.size(index)
    result = np.zeros((dim, dim))

    for i in range(dim):
        result[:, i] = index
        result[i, i] += 1

        if not is_reasonable_sghr(accuracy_level, dim, result[:, i]):
            result[:, i] = 0

    result = np.delete(result, np.nonzero(np.sum(np.abs(result), 1) == 0), axis=1)
    nr = get_shape_at(result, 1, default=0)
    n_old = get_shape_at(old_index, 1, default=0)

    for i in range(n_old):
        residual = np.abs(result - np.repeat(old_index[:, i], nr))
        result[:, np.sum(residual) == 0] = 0

    result = np.delete(result, np.nonzero(np.sum(np.abs(result), 1) == 0), axis=1)

    return result


def generate_index_sparse_gauss_hermite_rule(accuracy_level: int, dimension: int) -> np.ndarray:
    n_index = generate_next_index_sparse_gauss_hermite_rule(accuracy_level, np.ones(dimension), np.asarray([]))
    nn_index = np.vstack((np.ones((1, dimension)), n_index))

    while True:
        tmp_index = []
        for i in range(get_shape_at(n_index, 1, 0)):
            tmp_index = generate_next_index_sparse_gauss_hermite_rule(accuracy_level, n_index[:, i], nn_index)
            nn_index = np.hstack((nn_index, tmp_index))

        if not tmp_index:
            break

        n_index = nn_index

    return nn_index


def get_one_dim_sparse_gauss_hermite_point(idx: np.ndarray, point: int, manner: int) -> Tuple[np.ndarray, np.ndarray]:
    if point == 1:
        if not idx == 0:
            if manner == 1:
                n = idx
            elif manner == 2:
                n = 2 * idx - 1
            elif manner == 3:
                n = 2 ** idx - 1
            else:
                raise Exception(f'Not supported: manner = {manner}')

            return eval_gauss_hermite_rule(n, None)

    return np.asarray([]), np.asarray([])


def generate_sparse_gauss_hermite_point(
        accuracy_level: int,
        idx: np.ndarray,
        point_set: np.ndarray,
        manner: int
) -> Tuple[np.ndarray, np.ndarray]:
    dim = np.size(idx)
    q = np.sum(idx) - dim

    if accuracy_level - dim <= q <= accuracy_level - 1:
        points, weights = get_one_dim_sparse_gauss_hermite_point(idx[0], point_set[0], manner)
        points = points.T
        weights = np.atleast_2d(weights).T

        for i in range(1, dim):
            npt, nw = get_one_dim_sparse_gauss_hermite_point(idx[i], point_set[i], manner)

            num_npt = np.size(nw)

            points = np.tile(points, (1, num_npt))
            # pt_add = np.tile(npt, num_pt)
            pt_add = np.tile(npt, (1, get_shape_at(points, 1, 1)))
            pt_add = pt_add.flatten()  # todo (pt_add(:))'
            points = np.vstack((points, pt_add))

            weights = np.tile(weights, (1, num_npt))
            w_add = np.tile(nw, (1, get_shape_at(weights, 1, 1)))
            w_add = w_add.flatten()  # todo (w_add(:))'
            weights = np.vstack((weights, w_add))

        if weights.any():
            n = np.math.factorial(dim - 1) / (np.math.factorial(accuracy_level - 1 - q) * np.math.factorial((dim - 1) - (accuracy_level - 1 - q)))
            weights = (-1) ** (accuracy_level - 1 - q) * np.prod(weights) * n
    else:
        points = np.asarray([])
        weights = np.asarray([])

    return np.atleast_2d(points), np.atleast_1d(weights)


@lru_cache()
def generate_sparse_gauss_hermite_set(accuracy_level: int, dimension: int, manner: int) -> Tuple[np.ndarray, np.ndarray]:
    accuracy_level_sequences_set = {}
    for q in range(accuracy_level - dimension, accuracy_level - 1):
        if q <= 0:
            accuracy_level_sequences_set[q] = []
        else:
            accuracy_level_sequence = []

    index_set = generate_index_sparse_gauss_hermite_rule(accuracy_level, dimension)

    points = np.asarray([])
    weights = np.asarray([])

    for i in range(get_shape_at(index_set, 1, 0)):
        tmp_points, tmp_weights = generate_sparse_gauss_hermite_point(accuracy_level, index_set[:, i], np.ones(dimension), manner)

        if np.size(points) == 0:  # todo: looks like a first run?? check and simplify, i.e. i == 0
            points = tmp_points
            weights = tmp_weights
        else:
            index_to_del = np.asarray([])
            npt = np.size(weights)

            for j in range(np.size(tmp_weights)):
                residual = np.abs(np.squeeze(points) - np.repeat(tmp_points[:, j], npt))
                fi = np.nonzero(np.sum(residual) < 1e-6)

                if fi:
                    weights[fi] = tmp_weights[j] + weights[fi]
                    index_to_del = np.hstack((index_to_del, j))

            tmp_points = np.delete(tmp_points, index_to_del, axis=1)  # tmp_points[:, index_to_del] = []
            tmp_weights = np.delete(tmp_weights, index_to_del, axis=0)  # tmp_weights[index_to_del] = []

            points = np.hstack((points, tmp_points))
            weights = np.hstack((weights, tmp_weights))

    return points, weights
