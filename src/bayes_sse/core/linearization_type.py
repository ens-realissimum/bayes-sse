from enum import Enum


# x  - state,
# v  - state noise,
# n  - observation  noise,
# n  - observation  noise,
# p  - system params,
# u1 - state control,
# u2 - observation  control


class LinearizationType(Enum):
    F = 1   # F = df / dx
    B = 2   # B = df / du1
    C = 3   # C = dh / dx
    D = 4   # D = dh / du2
    G = 5   # G = df / dv
    H = 6   # H = dh / dn
    JFW = 7  # JFW =  df / dp
    JHW = 8  # dh / dp
