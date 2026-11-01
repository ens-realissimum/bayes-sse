from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture as SklearnGaussianMixture

from bayes_sse.core.covariance_utils import to_covariance_with_type
from bayes_sse.inference.stochastic_models.covariance_type import CovarianceType
from bayes_sse.typing import (
    Float64Matrix,
    MixtureCovarianceFactors,
    MixtureCovariances,
    MixtureMeans,
    MixtureWeights,
)


class GaussianMixture:
    def __init__(
        self,
        n_components: int,
        means: MixtureMeans,
        sqrt_covariances: MixtureCovarianceFactors,
        weights: MixtureWeights,
    ) -> None:
        super().__init__()

        self._n_components = n_components
        self._means = means
        self._sqrt_covariances = sqrt_covariances
        self._weights = weights

    @property
    def n_components(self) -> int:
        return self._n_components

    @property
    def means(self) -> MixtureMeans:
        return self._means

    @property
    def sqrt_covariances(self) -> MixtureCovarianceFactors:
        return self._sqrt_covariances

    @property
    def weights(self) -> MixtureWeights:
        return self._weights

    def set_params(
        self,
        means: MixtureMeans,
        covariances: MixtureCovariances,
        weights: MixtureWeights,
        covariance_type: CovarianceType,
    ) -> None:
        self._means = means
        self._sqrt_covariances = to_covariance_with_type(
            covariances, covariance_type, CovarianceType.sqrt
        )
        self._weights = weights

    @staticmethod
    def init(n_components: int, particles: Float64Matrix) -> GaussianMixture:
        gmm = SklearnGaussianMixture(n_components)
        gmm.fit(particles)

        covariances = gmm.covariances_
        means = gmm.means_
        weights = gmm.weights_

        if covariances is None or means is None or weights is None:
            raise ValueError("Gaussian mixture must be fitted")

        if gmm.covariance_type != "full":
            raise ValueError("Only full covariance matrices are supported")

        sqrt_covariances = np.linalg.cholesky(covariances)

        return GaussianMixture(n_components, means, sqrt_covariances, weights)
