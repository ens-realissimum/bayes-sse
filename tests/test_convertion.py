from numpy.testing import assert_array_equal, assert_equal

import tests.stochastic_models_test_utils as utils
from bayes_sse.inference.stochastic_models.convertion import convert_to_gaussian
from bayes_sse.inference.stochastic_models.covariance_type import CovarianceType


class TestConvertToGaussian:
    def test__from_gamma__cov_mean_dim_must_be_same(self):
        gamma = utils.build_gamma()
        expected_cov_type = CovarianceType.full
        gaussian = convert_to_gaussian(gamma, expected_cov_type)

        assert_equal(gaussian.dim, gamma.dim)
        assert_array_equal(gaussian.covariance, gamma.covariance)
        assert_array_equal(gaussian.mean, gamma.mean)

    def test__from_gaussian__cov_mean_dim_must_be_same(self):
        expected_cov_type = CovarianceType.sqrt_diag
        expected_gaussian = utils.build_gauss(covariance_type=CovarianceType.sqrt_diag)
        gaussian = convert_to_gaussian(expected_gaussian, expected_cov_type)

        assert_equal(gaussian.dim, expected_gaussian.dim)
        assert_array_equal(gaussian.covariance, expected_gaussian.covariance)
        assert_array_equal(gaussian.mean, expected_gaussian.mean)

    def test__from_gaussian_mixture__cov_mean_dim_must_be_same(self):
        gmm = utils.build_gmm()
        gaussian = convert_to_gaussian(gmm, gmm.covariance_type)

        assert_equal(gaussian.dim, gmm.dim)
        assert_array_equal(gaussian.covariance, gmm.covariance)
        assert_array_equal(gaussian.mean, gmm.mean)
