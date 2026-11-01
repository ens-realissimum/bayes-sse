import numpy as np
from numpy.random import random, uniform
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal
from pytest import raises
from scipy.stats import gamma, multivariate_normal

import bayes_sse.inference.stochastic_models.stochastic_models as sm
import tests.stochastic_models_test_utils as utils
from bayes_sse.inference.stochastic_models.covariance_type import CovarianceType
from bayes_sse.inference.stochastic_models.noise_type import NoiseType


class TestGammaStochasticModel:
    def test__likelihood__samples_specified__equal_to_gamma_pdf(self):
        shape = random()
        scale = random()
        gamma_model = sm.Gamma(shape=shape, scale=scale)
        samples = gamma_model.sample(100000)

        actual_likelihood = gamma_model.likelihood(samples)
        expected_likelihood = gamma.pdf(samples.T, shape, scale=scale)
        assert_array_almost_equal(actual_likelihood, expected_likelihood, decimal=3)

    def test__sample__size_is_zero__raise_error(self):
        gamma_model = utils.build_gamma()
        with raises(
            Exception,
            match="`size` is negative or equal to zero. Must be integer greater than zero",
        ):
            gamma_model.sample(0)

    def test__sample__size_is_less_than_zero__raise_error(self):
        gamma_model = utils.build_gamma()
        for negative_number in uniform(low=-100, high=0, size=100):
            with raises(
                Exception,
                match="`size` is negative or equal to zero. Must be integer greater than zero",
            ):
                gamma_model.sample(negative_number)

    def test__sample__size_specified__return_horizontally_stacked_random_vectors(self):
        gamma_model = utils.build_gamma()
        size = 1000
        samples = gamma_model.sample(size)
        assert samples.shape[0] == 1
        assert samples.shape[1] == size

    def test__sample__size_specified__mean_of_generated_equal_to_specified_mean(self):
        gamma_model = utils.build_gamma()
        samples = gamma_model.sample(1000000)
        assert_array_almost_equal(np.mean(samples, axis=1), gamma_model.mean, decimal=2)

    def test__sample__size_specified__variance_of_generated_equal_to_specified_variance(
        self,
    ):
        gamma_model = utils.build_gamma()
        samples = gamma_model.sample(10000)
        assert_array_almost_equal(
            np.atleast_2d(np.var(samples, axis=1)), gamma_model.covariance, decimal=2
        )

    def test__mean__shape_and_scale_specified__calculated_based_shape_and_scale(self):
        for shape, scale in utils.get_shape_scale_pairs():
            gamma_model = sm.Gamma(shape=shape, scale=scale)
            desired_mean = gamma.mean(shape, scale=scale)
            assert_array_almost_equal(gamma_model.mean, desired_mean, decimal=5)

    def test__covariance__shape_and_scale_specified__calculated_based_on_shape_and_scale(
        self,
    ):
        for shape, scale in utils.get_shape_scale_pairs():
            gamma_model = sm.Gamma(shape=shape, scale=scale)
            desired_variance = gamma.var(shape, scale=scale)
            assert_array_almost_equal(
                gamma_model.covariance, desired_variance, decimal=5
            )

    def test__covariance_type__always__full(self):
        gamma_model = utils.build_gamma()
        assert gamma_model.covariance_type is CovarianceType.full

    def test__noise_type__always__gamma(self):
        gamma_model = utils.build_gamma()
        assert gamma_model.noise_type is NoiseType.gamma


class TestGaussianStochasticModel:
    def test__likelihood__samples_specified__equal_to_gauss_pdf(self):
        mean = np.random.random(size=(3,))
        sqrt_cov = np.random.random(size=(3, 3))
        cov = sqrt_cov @ sqrt_cov.T
        gauss_model = sm.Gaussian(
            mean=mean, covariance=sqrt_cov, covariance_type=CovarianceType.sqrt
        )
        samples = gauss_model.sample(100000)

        actual_likelihood = gauss_model.likelihood(samples)
        expected_likelihood = multivariate_normal.pdf(
            samples.T, mean=mean, cov=cov.tolist()
        )
        assert_array_almost_equal(actual_likelihood, expected_likelihood, decimal=3)

    def test__samples__size_is_positive_integer__shape_must_be_dim_x_samples_count(
        self,
    ):
        dim = 3
        gauss_model = utils.build_gauss(dim)
        size = 100000
        samples = gauss_model.sample(size)

        expected_shape = (dim, size)
        assert_equal(samples.shape, expected_shape)


class TestGaussianMixtureStochasticModel:
    def test__samples__size_is_positive_integer__shape_must_be_dim_x_samples_count(
        self,
    ):
        dim = 3
        gmm = utils.build_gmm(3)

        size = 100000
        samples = gmm.sample(size)

        expected_shape = (dim, size)
        assert_equal(samples.shape, expected_shape)

    def test__from_gaussian__gaussian_process_specified__must_be_created_with_same_cov_mean_cov_type(
        self,
    ):
        gauss_model = utils.build_gauss()

        gmm = sm.GaussianMixture.from_gaussian(gauss_model)

        assert gmm.covariance_type == gauss_model.covariance_type
        assert_equal(gmm.dim, gauss_model.dim)

        # mix contains only 1 item, hence ignore dim related to  mixture size as it's always equal to 1
        assert_array_equal(gmm.covariance.squeeze(), gauss_model.covariance)
        assert_array_equal(gmm.mean.squeeze(), gauss_model.mean)
