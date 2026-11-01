import numpy as np

import bayes_sse._utils.matrix_utils as matrix_utils
from bayes_sse.bayesian_filter_type import BayesianFilterType, sqrt_sigma_point_filter, linear_kalman_family, sigma_point_family
from bayes_sse.inference.stochastic_models.covariance_type import CovarianceType, is_sqrt_like


def to_full_covariance(cov: np.ndarray, source_cov_type: CovarianceType) -> np.ndarray:
    if source_cov_type == CovarianceType.diag:
        return matrix_utils.ensure_diagonal_matrix(cov)
    elif source_cov_type == CovarianceType.sqrt_diag:
        cov_diag = matrix_utils.ensure_diagonal_matrix(cov)
        return cov_diag @ cov_diag.T.conj()
    elif source_cov_type == CovarianceType.full:
        return cov
    elif source_cov_type == CovarianceType.sqrt:
        return cov @ cov.T.conj()
    else:
        raise Exception(f"Not supported cov_type: {source_cov_type.name}")


def to_mixture_full_covariance(cov: np.ndarray, source_cov_type: CovarianceType) -> np.ndarray:
    is_mixture = cov.ndim == 3
    cov_list = cov if is_mixture else [cov]

    return np.asarray(
        list(map(lambda x: to_full_covariance(x, source_cov_type), cov_list))
    )


def to_sqrt_covariance(cov: np.ndarray, source_cov_type: CovarianceType) -> np.ndarray:
    if source_cov_type == CovarianceType.diag:
        return np.linalg.cholesky(
            matrix_utils.ensure_diagonal_matrix(cov)
        )
    elif source_cov_type == CovarianceType.sqrt_diag:
        return matrix_utils.ensure_diagonal_matrix(cov)
    elif source_cov_type == CovarianceType.full:
        return np.linalg.cholesky(cov)
    elif source_cov_type == CovarianceType.sqrt:
        return cov
    else:
        raise Exception(f"Not supported cov_type: {source_cov_type.name}")


def to_mixture_sqrt_covariance(cov: np.ndarray, source_cov_type: CovarianceType) -> np.ndarray:
    is_mixture = cov.ndim == 3
    cov_list = cov if is_mixture else [cov]

    if is_mixture:
        return np.atleast_3d(
            list(map(lambda x: to_sqrt_covariance(x, source_cov_type), cov_list))
        )

    return np.atleast_2d(
        list(map(lambda x: to_sqrt_covariance(x, source_cov_type), cov_list))
    )


def to_covariance_with_type(cov: np.ndarray, source_cov_type: CovarianceType, target_cov_type: CovarianceType) -> np.ndarray:
    target_is_sqrt_like = is_sqrt_like(target_cov_type)

    res = to_mixture_sqrt_covariance(cov, source_cov_type) if target_is_sqrt_like else to_mixture_full_covariance(cov, source_cov_type)
    is_mixture = cov.ndim == 3
    return res if is_mixture else np.atleast_2d(np.squeeze(res))


def prepare_cov_for_estimation(cov: np.ndarray, filter_type: BayesianFilterType) -> np.ndarray:
    if filter_type in sqrt_sigma_point_filter:
        return to_covariance_with_type(cov, CovarianceType.full, CovarianceType.sqrt)

    if filter_type in linear_kalman_family or filter_type in sigma_point_family:
        return cov

    raise Exception(f"not supported filter type: {filter_type.name}")
