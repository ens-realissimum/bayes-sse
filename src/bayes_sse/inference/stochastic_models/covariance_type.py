from enum import Enum


class CovarianceType(Enum):
    unknown = 0
    full = 1
    diag = 2
    sqrt = 3
    sqrt_diag = 4
    svd = 5


def is_sqrt_like(cov_type: CovarianceType):
    return cov_type in (CovarianceType.sqrt, CovarianceType.sqrt_diag)


def is_diag_like(cov_type: CovarianceType):
    return cov_type in (CovarianceType.diag, CovarianceType.sqrt_diag)
