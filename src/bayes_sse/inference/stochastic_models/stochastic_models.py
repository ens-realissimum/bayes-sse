from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import Covariance, gamma, multivariate_normal

import bayes_sse._utils.matrix_utils as matrix_utils
import bayes_sse.core.covariance_utils as cov_utils

from .covariance_type import CovarianceType
from .noise_type import NoiseType


class StochasticModel(ABC):
    """
    The generated noise data structure (model), Model has the following required fields.
    Depending on the noise source type, the data structure may also have other type dependent fields.
        .update (function handle) <<optional>> function to update the internal structure of noise source;
        .covarianceType.
    """

    @property
    @abstractmethod
    def noise_type(self) -> NoiseType:
        """
        Noise type (NoiseType enum).
        :return: noise type.
        """
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """
        State space dimension (scalar number).
        :return: noise dimension.
        """
        ...

    @property
    @abstractmethod
    def covariance_type(self) -> CovarianceType:
        """
        Type of covariance (CovarianceType enum).
        :return: Type of covariance
        """
        ...

    @property
    @abstractmethod
    def covariance(self) -> np.ndarray:
        """
        Covariance matrix. None if covariance is not applicable for particular model.
        This property returns covariance in a format matching `covariance_type`
        :return: Covariance matrix
        """
        ...

    @property
    @abstractmethod
    def mean(self) -> np.ndarray:
        """
        Expectation, mathematical expectation, EV, average, mean value, mean, or first moment.
        :return: 'Expected value' (first moment)
        """
        ...

    @abstractmethod
    def sample(self, size: int) -> np.ndarray:
        """
        Generates ` noise source samples.
        :type size: int
        :return: nd-array of noise source samples with dimension equal to `dim` x `size`.
        """
        ...

    @abstractmethod
    def likelihood(self, samples: np.ndarray) -> np.ndarray:
        """
        Evaluates the likelihood of a given samples.
        :return: likelihood of a given noise source sample (scalar number).
        """
        ...

    @abstractmethod
    def update(self, **kwargs: Any) -> None:
        """
        Synchronizes any derived or dependent internal state after model parameters
        have changed.
        """
        pass

    @property
    def n_components(self) -> int:
        return 0

    @property
    def weights(self) -> np.ndarray:
        raise RuntimeError("weights aren't available.")

    def __init__(
        self,
        *,
        shape: int | float | Sequence,
        scale: int | float | Sequence,
    ):
        is_shape_list = isinstance(shape, Sequence)
        is_scale_list = isinstance(scale, Sequence)

        if is_shape_list and is_scale_list:
            if len(shape) != len(scale):
                raise Exception("dimensions mismatch of shape and scale")

        if is_shape_list and not is_scale_list or not is_shape_list and is_scale_list:
            raise Exception(
                "shape and scale both should be sequence or scalar at the same time"
            )

        self._dimension = len(shape) if is_shape_list else 1
        self._shape = shape
        self._scale = scale

    def __str__(self) -> str:
        return f"Gamma process. Shape: {self._shape}; Scale: {self._scale}"

    def likelihood(self, samples: np.ndarray) -> np.ndarray:
        return np.atleast_1d(gamma.pdf(samples.T, self._shape, scale=self._scale))

    def sample(self, size: int) -> np.ndarray:
        if size <= 0:
            raise Exception(
                "`size` is negative or equal to zero. Must be integer greater than zero"
            )

        return np.atleast_2d(gamma.rvs(self._shape, scale=self._scale, size=size))

    @property
    def mean(self) -> np.ndarray:
        return np.atleast_1d(gamma.mean(self._shape, scale=self._scale))

    @property
    def covariance(self) -> np.ndarray:
        return np.asarray(gamma.var(self._shape, scale=self._scale))

    @property
    def covariance_type(self) -> CovarianceType:
        return CovarianceType.full

    @property
    def dim(self) -> int:
        return self._dimension

    @property
    def noise_type(self) -> NoiseType:
        return NoiseType.gamma

    def update(self, **kwargs: Any) -> None:
        _ = kwargs


class Gaussian(StochasticModel):
    def __init__(
        self,
        *,
        mean: np.ndarray,
        covariance: np.ndarray,
        covariance_type: CovarianceType,
    ):
        mean_arr = np.atleast_1d(np.asarray(mean))

        if mean_arr.ndim >= 1 and mean_arr.size > 1:
            if not matrix_utils.is_square_2d_array(covariance):
                raise Exception("Covariance matrix must be square matrix")

            if len(mean_arr) != matrix_utils.shape_square_2d_array(covariance):
                raise Exception("Dimensions mismatch of mean and covariance matrix")

            self._dimension = len(mean_arr)
            self._mean = mean_arr
        else:
            self._dimension = 1
            self._mean = mean_arr

        is_diag = covariance_type in (CovarianceType.diag, CovarianceType.sqrt_diag)
        self._covariance_type = covariance_type
        self._covariance = np.asarray(
            matrix_utils.ensure_diagonal_matrix(covariance) if is_diag else covariance,
            dtype=np.float64,
        )
        self._covariance_sqrt: np.ndarray = np.asarray(
            cov_utils.to_sqrt_covariance(covariance, covariance_type), dtype=np.float64
        )
        self._covariance_full: np.ndarray = np.asarray(
            cov_utils.to_full_covariance(covariance, covariance_type), dtype=np.float64
        )

    def __str__(self) -> str:
        return f"Gaussian noise. Mean: {self.mean}; Covariance: {self._covariance_full}"

    @property
    def noise_type(self) -> NoiseType:
        return NoiseType.gaussian

    @property
    def dim(self) -> int:
        return self._dimension

    @property
    def covariance_type(self) -> CovarianceType:
        return self._covariance_type

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    def sample(self, size: int) -> np.ndarray:
        return np.atleast_2d(
            multivariate_normal.rvs(
                mean=self._mean, cov=cast(Any, self._covariance_full), size=size
            ).T
        )

    def likelihood(self, samples: np.ndarray) -> np.ndarray:
        return np.asarray(
            multivariate_normal.pdf(
                samples.T, mean=self._mean, cov=cast(Any, self._covariance_full)
            )
        )

    def update(self, **kwargs: Any) -> None:
        _ = kwargs


class ComboGaussian(StochasticModel):
    def __init__(self, *, dimension: int, sources: Sequence):
        cov_type = sources[0].covariance_type
        if not all(source.covariance_type == cov_type for source in sources):
            raise Exception("all source items must have same covariance_type")

        cov = matrix_utils.put_matrices_into_zero_matrix_one_by_one(
            dimension, list(map(lambda x: x.covariance, sources))
        )
        self._dimension = dimension
        self._covariance_type = sources[0].covariance_type
        self._mean = np.atleast_1d(
            [sub_mean for source in sources for sub_mean in source.mean]
        )
        self._covariance = np.atleast_2d(cov)
        self._covariance_full = np.atleast_2d(
            cov_utils.to_full_covariance(cov, cov_type)
        )
        self._sources = sources

        if not np.size(self._mean) == dimension:
            raise Exception("length of mean vector must be equal to dimension")

    def __str__(self) -> str:
        return f"Combo Gaussian.\nMean:\n {self.mean};\nCovariance:\n {self._covariance_full}\n\n"

    @property
    def noise_type(self) -> NoiseType:
        return NoiseType.combo_gaussian

    @property
    def dim(self) -> int:
        return self._dimension

    @property
    def covariance_type(self) -> CovarianceType:
        return self._covariance_type

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    def sample(self, size: int) -> np.ndarray:
        return np.atleast_2d(
            multivariate_normal.rvs(
                mean=self._mean, cov=cast(Any, self._covariance_full), size=size
            ).T
        )

    def likelihood(self, samples: np.ndarray) -> np.ndarray:
        return np.atleast_1d(
            multivariate_normal.pdf(
                samples.T, mean=self._mean, cov=cast(Any, self._covariance_full)
            )
        )

    def update(self, **kwargs) -> None:
        shift = 0

        for source in self._sources:
            source.mean = self._mean[shift : shift + source.dim]
            source.covariance = self._covariance[
                shift : shift + source.dim, shift : shift + source.dim
            ]

            shift += source.dim


class Combo(ComboGaussian):
    def __str__(self) -> str:
        return (
            f"Combo noise.\nMean:\n {self.mean};\nCovariance:\n {self.covariance}\n\n"
        )

    @property
    def noise_type(self) -> NoiseType:
        return NoiseType.combo

    def sample(self, size: int) -> np.ndarray:
        result = np.zeros((self.dim, size))
        shift = 0

        for source in self._sources:
            result[shift : shift + source.dim, :] = source.sample(size)
            shift += source.dim

        return np.atleast_2d(result)

    def likelihood(self, samples: np.ndarray) -> np.ndarray:
        _, size = np.shape(samples)
        llh = np.ones(size)
        shift = 0

        for source in self._sources:
            llh = llh * source.likelihood(samples[shift : shift + source.dim, :])
            shift += source.dim

        return np.atleast_1d(llh)


class GaussianMixture(StochasticModel):
    def __init__(
        self,
        *,
        mixture_size: int,
        mean: Sequence[FloatArray],
        covariance: Sequence[FloatArray],
        covariance_type: CovarianceType,
        weights: Sequence[float] | FloatArray | None = None,
    ) -> None:
        self._n_components = mixture_size
        self._mean: FloatArray = np.asarray(mean, dtype=np.float64)
        self._covariance_type = covariance_type
        self._covariance: FloatArray = np.asarray(covariance, dtype=np.float64)

        cov_full = (
            cov_utils.to_full_covariance(cov, covariance_type)
            for cov in self._covariance
        )
        self._covariance_full: FloatArray = np.asarray(list(cov_full), dtype=np.float64)

        _, self._dimension = self._mean.shape

        if weights is None:
            self._weights: FloatArray = (
                np.ones(mixture_size, dtype=np.float64) / mixture_size
            )
        else:
            weights_arr = np.asarray(weights, dtype=np.float64)
            self._weights = weights_arr / weights_arr.sum()

    def __str__(self) -> str:
        return f"Gaussian Mixture.\nWeights:\n {self.weights};\nMean:\n {self._mean};\nCovariance:\n {self._covariance_full}\n\n"

    @property
    def noise_type(self) -> NoiseType:
        return NoiseType.gaussian_mixture

    @property
    def dim(self) -> int:
        return self._dimension

    @property
    def covariance_type(self) -> CovarianceType:
        return self._covariance_type

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    @property
    def n_components(self) -> int:
        return self._n_components

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    def sample(self, size) -> np.ndarray:
        samples_component = np.random.multinomial(size, self.weights)
        return np.atleast_2d(
            np.hstack(
                [
                    np.transpose(
                        np.random.multivariate_normal(mean, covariance, int(sample))
                    )
                    for (mean, covariance, sample) in zip(
                        self.mean, self._covariance_full, samples_component
                    )
                ]
            )
        )

    def likelihood(self, samples: np.ndarray) -> np.ndarray:
        # evidence is used as likelihood, because
        # i-th entry of evidence is the total data probability for a given data vector X[i],
        # i.e. P(X[i]) = sum_over_all_j { P(X[i] | C[j]) }
        # i.e. effectively it's a likelihood
        _, _, llh, _ = self.probability(samples)
        return np.atleast_1d(llh)

    def probability(
        self, samples: np.ndarray, evidence_weights: np.ndarray = np.ndarray([])
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates any of the related (through Bayes rule) probabilities of a Gaussian Mixture Model and
        a given data set (samples). Probabilities are:
                   P(X|C) * P(C)                   likelihood * prior
        P(C|X) = -----------------   posterior =  --------------------
                       P(X)                            evidence

        where C is the component classes (Gaussian) of the GMM and X is the data.

        :param samples:
        :param evidence_weights:
        :return: Tuple of: Prior probability, Likelihood, Evidence, Posterior probability
            Prior       the prior (without seeing data) probability of a component density generating any given
                        data vector, i.e. P(C[i]). This is simply the same as the prior mixing weights
            Likelihood  matrix where the j, i-th entry is the likelihood of input column vector i (of samples)
                        conditioned on component density j, i.e. P(X[i] | C[j]);
            Evidence    matrix where the i-th entry is the total data probability for a given data vector X[i], i.e.
                        P(X[i]) = sum_over_all_j { P(X[i] | C[j]) };
            Posterior   matrix where the j, i-th entry is the posterior probability (after seeing the data) that
                        a component density j has generated a specific data vector X[i], i.e.
                        P(C[j] | X[i]) (class posterior probabilities).
        """
        prior = np.copy(self.weights)

        llh = np.column_stack(
            [
                multivariate_normal.pdf(samples.T, mean=mean, cov=cov)
                for (mean, cov) in zip(self.mean, self._covariance_full)
            ]
        )

        evidence_w = (
            np.ones(np.shape(llh)) if evidence_weights is None else evidence_weights
        )
        evidence = (llh / evidence_w) @ prior

        posterior = (
            llh / (np.reciprocal(prior)[:, np.newaxis] @ evidence[np.newaxis, :]).T
        )
        posterior = posterior / np.sum(posterior)

        return prior, llh, evidence, posterior

    def update(self, **kwargs: Any) -> None:
        _ = kwargs

    @staticmethod
    def from_gaussian(gauss_process: Gaussian) -> GaussianMixture:
        return GaussianMixture(
            mixture_size=1,
            mean=np.asarray([gauss_process.mean]),
            covariance=np.asarray([gauss_process.covariance]),
            covariance_type=gauss_process.covariance_type,
        )


def build_stochastic_process(noise_type: NoiseType, **kwargs) -> StochasticModel:
    if noise_type == NoiseType.gamma:
        return Gamma(**kwargs)
    elif noise_type == NoiseType.gaussian:
        return Gaussian(**kwargs)
    elif noise_type == NoiseType.combo_gaussian:
        return ComboGaussian(**kwargs)
    elif noise_type == NoiseType.combo:
        return Combo(**kwargs)
    elif noise_type == NoiseType.gaussian_mixture:
        return GaussianMixture(**kwargs)
    else:
        raise Exception(f"Not supported noise_type: {noise_type.name}")
