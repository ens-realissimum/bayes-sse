from enum import Enum


class NoiseType(Enum):
    unknown = 0
    gaussian = 1
    gamma = 2
    combo_gaussian = 3
    combo = 4
    gaussian_mixture = 5
