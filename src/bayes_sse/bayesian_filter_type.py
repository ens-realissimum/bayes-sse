from enum import Enum


class BayesianFilterType(Enum):
    kf = 1
    ekf = 2
    ukf = 3
    cdkf = 4
    srukf = 5
    srcdkf = 6
    ckf = 7
    srckf = 8
    fdckf = 9
    cqkf = 10
    ghqf = 11
    sghqf = 12
    pf = 13
    gspf = 14
    gmsppf = 15
    sppf = 16


sigma_point_family = {
    BayesianFilterType.ukf: BayesianFilterType.ukf,
    BayesianFilterType.cdkf: BayesianFilterType.cdkf,
    BayesianFilterType.srukf: BayesianFilterType.srukf,
    BayesianFilterType.srcdkf: BayesianFilterType.srcdkf,
    BayesianFilterType.ckf: BayesianFilterType.ckf,
    BayesianFilterType.srckf: BayesianFilterType.srckf,
    BayesianFilterType.fdckf: BayesianFilterType.fdckf,
    BayesianFilterType.cqkf: BayesianFilterType.cqkf,
    BayesianFilterType.ghqf: BayesianFilterType.ghqf,
    BayesianFilterType.sghqf: BayesianFilterType.sghqf,
}

linear_kalman_family = {
    BayesianFilterType.kf: BayesianFilterType.kf,
    BayesianFilterType.ekf: BayesianFilterType.ekf,
}

sqrt_sigma_point_filter = {
    BayesianFilterType.srukf: BayesianFilterType.srukf,
    BayesianFilterType.srcdkf: BayesianFilterType.srcdkf,
    BayesianFilterType.srckf: BayesianFilterType.srckf,
}


def is_sigma_point_filter(filter_type: BayesianFilterType) -> bool:
    return filter_type in sigma_point_family


def is_linear_kalman_filter(filter_type: BayesianFilterType) -> bool:
    return filter_type in linear_kalman_family


def is_sqrt_sigma_point_filter(filter_type: BayesianFilterType) -> bool:
    return filter_type in sqrt_sigma_point_filter


def is_sigma_point_full_filter(filter_type: BayesianFilterType) -> bool:
    return (
        filter_type in sigma_point_family and filter_type not in sqrt_sigma_point_filter
    )
