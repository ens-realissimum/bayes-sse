from typing import Tuple

import numpy as np


def stat_errors(delta_list: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate mean of error, standard deviations of error and root mean squared error of given input list.
    :param delta_list: list of errors, ie list of x_predicted - x_target
    :return: tuple of mean, std and rmse
    """
    mean = np.mean(delta_list, axis=0)
    std = np.std(delta_list, axis=0)
    rmse = np.sqrt(
        np.mean(np.power(delta_list, 2), axis=0)
    )

    return mean, std, rmse


def stat_errors_between(predicted: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return stat_errors(predicted - target)
