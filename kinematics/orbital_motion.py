from typing import Tuple

import numpy as np

from kinematics.models import KinematicState


def eval_earth_non_spherical_influence(state: KinematicState) -> np.ndarray:
    """
    Evaluate Earth non-spherical gravity influence to body trajectory.
    :param state: state vector of body, KinematicState.
    :return: acceleration influence, np array (3,), [km/s^2]
    """
    earth_radius = 6378136  # [km] - Earth's equatorial radius
    mu = 398600.4418  # [km^3 / s^2] - Earth gravity const

    j2 = 0.00108262575
    j3 = -0.000002533
    j4 = -0.000001616

    r = np.linalg.norm(state.position)
    ro = earth_radius / r

    j2_axis = j2 * 3 / 2 * ro**2 * (1 - 5 * (state.r_z / r) ** 2)
    j3_axis = j3 * ro**3 * 5 / 2 * (3 - 7 * (state.r_z / r) ** 2) * state.r_z / r
    j4_axis = (
        j4 * ro**4 * 5 / 8 * (3 - 42 * (state.r_z / r) ** 2 + 63 * (state.r_z / r) ** 4)
    )

    g_x = -(mu * state.r_x / r**3) * (j2_axis + j3_axis - j4_axis)

    g_y = -(mu * state.r_y / r**3) * (j2_axis + j3_axis - j4_axis)

    j3_z = j3 * ro**3 * 5 / 2 * (3 - 7 * ro**2) * state.r_z / r
    g_z = -(mu * state.r_z / r**3) * (j2_axis + j3_z - j4_axis)

    return np.asarray([g_x, g_y, g_z])


def eval_orbital_eccentricity_anomaly(avg_anomaly: float, eccentricity: float) -> float:
    eccentricity_anomaly = avg_anomaly
    while (
        np.abs(
            avg_anomaly
            + eccentricity * np.sin(eccentricity_anomaly)
            - eccentricity_anomaly
        )
        > 1e-8
    ):
        eccentricity_anomaly = avg_anomaly + eccentricity * np.sin(eccentricity_anomaly)

    return eccentricity_anomaly


def eval_sin_and_cos_fi(avg_anomaly: float, eccentricity: float) -> Tuple[float, float]:
    eccentricity_anomaly = eval_orbital_eccentricity_anomaly(avg_anomaly, eccentricity)

    sin_fi = (np.sqrt(1 - eccentricity**2) * np.sin(eccentricity_anomaly)) / (
        1 - eccentricity * np.cos(eccentricity_anomaly)
    )
    cos_fi = (np.cos(eccentricity_anomaly) - eccentricity) / (
        1 - eccentricity * np.cos(eccentricity_anomaly)
    )

    return sin_fi, cos_fi


def eval_sun_influence(t_epoch, state: KinematicState) -> np.ndarray:
    """
    Evaluate Sun gravity influence to body trajectory.
    :param t_epoch: unix epoch time,
    :param state: state vector of body, KinematicState.
    :return: acceleration influence, np array (3,), [km/s^2]
    """
    avg_anomaly = (
        6.2400601269 + 628.3019551714 * t_epoch - 0.0000026820 * t_epoch**2
    )  # [rad] the average anomaly of the Sun
    avg_long = (
        -7.6281824375 + 0.0300101976 * t_epoch + 0.0000079741 * t_epoch**2
    )  # [rad] the average longitude of the ascending node of the Sun
    epsilon = (
        0.4090926006 - 0.0002270711 * t_epoch
    )  # [rad] the average inclination of the ecliptic to the equator

    eccentricity = 0.016719  # [] eccentricity of the Sun orbit
    sin_fi, cos_fi = eval_sin_and_cos_fi(avg_anomaly, eccentricity)

    eccentricity = cos_fi * np.cos(avg_long) - sin_fi * np.sin(avg_long)
    etta = (sin_fi * np.cos(avg_long) + cos_fi * np.sin(avg_long)) * np.cos(epsilon)
    f_sun = (sin_fi * np.cos(avg_long) + cos_fi * np.sin(avg_long)) * np.sin(epsilon)

    a_sun = 1.49598e8  # [] the semi-major axis of the Sun's orbit
    r_sun = a_sun * (
        1
        - eccentricity
        * np.cos(eval_orbital_eccentricity_anomaly(avg_anomaly, eccentricity))
    )

    x_norm = state.r_x / r_sun
    y_norm = state.r_y / r_sun
    z_norm = state.r_z / r_sun

    mu = 0.1325263e12  # [km^3/sec^2] - constant of the gravitational field of the Sun
    mu_r = mu / r_sun**2
    delta_sun = np.sqrt(
        (eccentricity - x_norm) ** 2 + (etta - y_norm) ** 2 + (f_sun - z_norm) ** 2
    )

    return np.asarray(
        [
            mu_r * ((eccentricity - x_norm) / delta_sun**3 - eccentricity),
            mu_r * ((etta - y_norm) / delta_sun**3 - etta),
            mu_r * ((f_sun - z_norm) / delta_sun**3 - f_sun),
        ]
    )


def eval_moon_influence(t_epoch, state: KinematicState) -> np.ndarray:
    """
    Evaluate Sun gravity influence to body trajectory.
    :param t_epoch: unix epoch time,
    :param state: state vector of body, KinematicState.
    :return: acceleration influence, np array (3,), [km/s^2]
    """
    omega_moon = (
        2.1824391966 - 33.7570459536 * t_epoch + 0.0000362262 * t_epoch**2
    )  # [rad] the average longitude of the ascending node of the Moon
    i_moon = 0.0898041080  # [rad] mean inclination of the Moon's orbit to the plane of the ecliptic
    etta_a = np.sin(omega_moon) * np.sin(i_moon)
    e_a = np.cos(i_moon)

    e11 = np.sin(omega_moon) * np.cos(omega_moon) * (1 - np.cos(i_moon))
    e12 = 1 - (np.sin(omega_moon)) ** 2 * (1 - np.cos(i_moon))

    epsilon = (
        0.4090926006 - 0.0002270711 * t_epoch
    )  # [rad] the average inclination of the ecliptic to the equator
    etta11 = e_a * np.cos(epsilon) - np.cos(omega_moon) * np.sin(i_moon) * np.sin(
        epsilon
    )
    etta12 = e11 * np.cos(epsilon) + etta_a * np.sin(epsilon)

    f11 = e_a * np.sin(epsilon) + np.cos(omega_moon) * np.sin(i_moon) * np.cos(epsilon)
    f12 = e11 * np.sin(epsilon) + etta_a * np.cos(epsilon)

    # Kepler's equation for Moon
    e_moon = 0.054900489  # [] eccentricity of the Moon orbit
    q_moon = 2.3555557435 + 8328.6914257190 * t_epoch + 0.0001545547 * t_epoch**2
    sin_fi, cos_fi = eval_sin_and_cos_fi(q_moon, e_moon)

    g_moon = 1.4547885346 + 71.0176852437 * t_epoch - 0.0001801481 * t_epoch**2
    e_moon = (sin_fi * np.cos(g_moon) + cos_fi * np.sin(g_moon)) * e11 + (
        cos_fi * np.cos(g_moon) - sin_fi * np.sin(g_moon)
    ) * e12
    etta_moon = (sin_fi * np.cos(g_moon) + cos_fi * np.sin(g_moon)) * etta11 + (
        cos_fi * np.cos(g_moon) - sin_fi * np.sin(g_moon)
    ) * etta12
    f_moon = (sin_fi * np.cos(g_moon) + cos_fi * np.sin(g_moon)) * f11 + (
        cos_fi * np.cos(g_moon) - sin_fi * np.sin(g_moon)
    ) * f12

    a_moon = 3.84385243e5  # [] the semi-major axis of the Moon's orbit
    r_moon = a_moon * (
        1 - e_moon * np.cos(eval_orbital_eccentricity_anomaly(q_moon, e_moon))
    )

    x_norm_moon = state.r_x / r_moon
    y_norm_moon = state.r_y / r_moon
    z_norm_moon = state.r_z / r_moon

    mu = 4902.835  # [km^3/sec^2] - constant of the gravitational field of the Moon
    mu_norm_moon = mu / r_moon**2
    delta_moon = np.sqrt(
        (e_moon - x_norm_moon) ** 2
        + (etta_moon - y_norm_moon) ** 2
        + (f_moon - z_norm_moon) ** 2
    )

    return np.asarray(
        [
            mu_norm_moon * ((e_moon - x_norm_moon) / delta_moon**3 - e_moon),
            mu_norm_moon * ((etta_moon - y_norm_moon) / delta_moon**3 - etta_moon),
            mu_norm_moon * ((f_moon - z_norm_moon) / delta_moon**3 - f_moon),
        ]
    )
