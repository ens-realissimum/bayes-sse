from typing import Union

import bayes_sse.inference.stochastic_models.stochastic_models as sm
from bayes_sse.inference.stochastic_models.covariance_type import CovarianceType
from bayes_sse.core.covariance_utils import to_covariance_with_type

GaussNoise = Union[sm.Gaussian, sm.GaussianMixture]


def convert_to_gaussian(stochastic_model: sm.StochasticModel, target_cov_type: CovarianceType) -> GaussNoise:
    """
    Convert stochastic process (stochastic_model) to Gaussian stochastic process (single or mixture) with
    specified covariance type (target_cov_type). Target process has same mean and covariance
    (but covariance is converted to requested type) as source process.
    Always creates new instance of converted stochastic process.
    :param stochastic_model: model of source stochastic process
    :param target_cov_type: covariance type of target Gaussian process
    :return: Gaussian process (single or mixture) created based on mean and covariance of source process.
    """
    cov = to_covariance_with_type(stochastic_model.covariance, stochastic_model.covariance_type, target_cov_type)

    if isinstance(stochastic_model, sm.GaussianMixture):
        return sm.GaussianMixture(
            mixture_size=stochastic_model.n_components,
            mean=stochastic_model.mean,
            covariance=cov,
            weights=stochastic_model.weights,
            covariance_type=target_cov_type
        )

    return sm.Gaussian(mean=stochastic_model.mean, covariance=cov, covariance_type=target_cov_type)


def convert_to_gmm(stochastic_model: sm.StochasticModel, target_cov_type: CovarianceType) -> sm.GaussianMixture:
    if isinstance(stochastic_model, sm.GaussianMixture):
        return convert_to_gaussian(stochastic_model, target_cov_type)

    return sm.GaussianMixture.from_gaussian(convert_to_gaussian(stochastic_model, target_cov_type))
