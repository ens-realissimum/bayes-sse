import bayes_sse.bayesian_filter_type as ft
from bayes_sse.inference.gssm import StateSpaceModel
from bayes_sse.inference.stochastic_models.convertion import convert_to_gaussian, convert_to_gmm
from bayes_sse.inference.stochastic_models.covariance_type import CovarianceType


def build_filterable_model(gssm: StateSpaceModel, filter_type: ft.BayesianFilterType) -> StateSpaceModel:
    if ft.is_sigma_point_filter(filter_type) or ft.is_linear_kalman_filter(filter_type):
        sp_cov_type = CovarianceType.sqrt if ft.is_sqrt_sigma_point_filter(filter_type) else CovarianceType.full

        return gssm.clone(convert_to_gaussian(gssm.state_noise, sp_cov_type),
                          convert_to_gaussian(gssm.observation_noise, sp_cov_type),
                          gssm.reconcile_strategy)

    if filter_type is ft.BayesianFilterType.gspf:
        return gssm.clone(convert_to_gmm(gssm.state_noise, CovarianceType.sqrt),
                          gssm.observation_noise,
                          gssm.reconcile_strategy)

    if filter_type is ft.BayesianFilterType.gmsppf:
        return gssm.clone(convert_to_gmm(gssm.state_noise, CovarianceType.sqrt),
                          convert_to_gmm(gssm.observation_noise, CovarianceType.sqrt),
                          gssm.reconcile_strategy)

    if filter_type is ft.BayesianFilterType.sppf:
        return gssm.clone(convert_to_gaussian(gssm.state_noise, CovarianceType.sqrt),
                          convert_to_gaussian(gssm.observation_noise, CovarianceType.sqrt),
                          gssm.reconcile_strategy)

    if filter_type is ft.BayesianFilterType.pf:
        return gssm.clone(gssm.state_noise, gssm.observation_noise, gssm.reconcile_strategy)

    raise Exception(f"Not supported filter type: {filter_type.name}")
