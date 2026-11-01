import numpy as np
from numpy.random.mtrand import uniform, random

from bayes_sse._utils.matrix_utils import generate_random_spd_matrix, generate_random_sqrt_mtrix
from bayes_sse.inference.stochastic_models import stochastic_models as sm
from bayes_sse.inference.stochastic_models.covariance_type import CovarianceType


def get_shape_scale_pairs():
    return zip(uniform(low=0, high=100, size=(200,)), uniform(low=0, high=100, size=(200,)))


def build_gmm(dim: int = 2) -> sm.GaussianMixture:
    cov_sqrt_1 = generate_random_sqrt_mtrix(dim)
    cov_sqrt_2 = generate_random_spd_matrix(dim)
    cov_sqrt_3 = generate_random_sqrt_mtrix(dim)
    mean_1 = np.random.random(dim)
    mean_2 = np.random.random(dim)
    mean_3 = np.random.random(dim)

    return sm.GaussianMixture(
        mixture_size=3,
        mean=[mean_1, mean_2, mean_3],
        covariance=[cov_sqrt_1, cov_sqrt_2, cov_sqrt_3],
        covariance_type=sm.CovarianceType.sqrt
    )


def build_gamma() -> sm.Gamma:
    return sm.Gamma(shape=random(), scale=random())


def build_gauss(dim: int = 3, covariance_type: CovarianceType = CovarianceType.sqrt_diag) -> sm.Gaussian:
    return sm.Gaussian(
        mean=np.random.random(dim),
        covariance=np.random.random((dim, dim)) @ np.random.random((dim, dim)).T,
        covariance_type=covariance_type
    )
