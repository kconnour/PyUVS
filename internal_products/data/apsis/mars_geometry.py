"""This module provides functions for computing geometry about Mars."""
import numpy as np
import spiceypy


def compute_solar_longitude(ephemeris_time: float) -> float:
    """Compute Mars's solar longitude at a given ephemeris time.

    Parameters
    ----------
    ephemeris_time: float
        The ephemeris time.

    Returns
    -------
    float
        Mars's solar longitude [degrees] at the input ephemeris time.

    Notes
    -----
    This requires the planetary constants kernel and the Mars kernel.

    """
    return np.degrees(spiceypy.lspcn('Mars', ephemeris_time, 'LT+S'))


def compute_subsolar_point(ephemeris_time: float) -> tuple[float, float]:
    """Compute the subsolar point at a given ephemeris time.

    Parameters
    ----------
    ephemeris_time: float
        The ephemeris time.

    Returns
    -------
    tuple[float, float]
        The (latitude [degrees N], longitude [degrees E]) of Mars's subsolar
        point at the input ephemeris time.

    Notes
    -----
    This requires the planetary constants kernel, the Mars kernel, and the
    MAVEN orbit kernels.

    """
    # This requires the MAVEN orbit kernels because 'obsrvr' is 'MAVEN'. If
    # it is 'Mars', you'll get nearly the same answer, but 'MAVEN' accounts for
    # the light travel time.
    spoint, _, _ = spiceypy.subslr('Intercept: ellipsoid', 'Mars',
        ephemeris_time, 'IAU_MARS', 'LT+S', 'MAVEN')
    _, colatitude, longitude = spiceypy.recsph(spoint)
    subsolar_latitude = 90 - np.degrees(colatitude)
    subsolar_longitude = np.degrees(longitude) % 360
    return subsolar_latitude, subsolar_longitude


def compute_mars_sun_distance(ephemeris_time: float) -> float:
    """Compute the distance between the sun and Mars at a given ephemeris time.

    Parameters
    ----------
    ephemeris_time: float
        The ephemeris time.

    Returns
    -------
    float
        The Mars-sun distance [km] at the input ephemeris time.

    Notes
    -----
    This requires the planetary constants kernel and the Mars kernel.

    """
    _, _, srfvec = spiceypy.subpnt('Intercept: ellipsoid', 'Mars',
        ephemeris_time, 'IAU_MARS', 'LT+S', 'SUN')
    return np.sqrt(np.sum(srfvec ** 2))
