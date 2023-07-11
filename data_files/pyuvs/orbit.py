"""This module is for computing properties of MAVEN's and Mars' orbit via SPICE. It assumes SPICE kernels are loaded
beforehand.

"""
import numpy as np
import spiceypy

observer = 'MAVEN'
target = 'Mars'
frame = 'IAU_Mars'
method = 'Intercept: ellipsoid'
aberration_correction = 'LT+S'


def compute_solar_longitude(et: float) -> float:
    """Compute the solar longitude for a given ephemeris time.

    Parameters
    ----------
    et: float
        The ephemeris time.

    Returns
    -------
    float
        The solar longitude [degrees] corresponding to the input ephemeris time.

    """
    return float(np.degrees(spiceypy.lspcn(target, et, aberration_correction)))


def compute_subsolar_point(et: float) -> tuple[float, float]:
    """Compute the sub-solar point for a given ephemeris time.

    Parameters
    ----------
    et: float
        The ephemeris time.

    Returns
    -------
    tuple[float, float]
        The latitude [degrees] and longitude [degrees] of the sub-solar point.

    """
    sub_point, _, _ = spiceypy.subslr(method, target, et, frame, aberration_correction, observer)
    return _rectangular_point_to_latitude_and_longitude(sub_point)


def compute_subspacecraft_point(et: float) -> tuple[float, float]:
    """Compute the sub-spacecraft point for a given ephemeris time.

    Parameters
    ----------
    et: float
        The ephemeris time.

    Returns
    -------
    tuple[float, float]
        The latitude [degrees] and longitude [degrees] of the sub-spacecraft point.

    """
    sub_point, _, _ = spiceypy.subpnt(method, target, et, frame, aberration_correction, observer)
    return _rectangular_point_to_latitude_and_longitude(sub_point)


def compute_spacecraft_altitude(et: float) -> float:
    """Compute the spacecraft altitude [km] for a given ephemeris time.

    Parameters
    ----------
    et: float
        The ephemeris time.

    Returns
    -------
    float
        The sub-spacecraft altitude.

    """
    return _compute_distance(et, observer)


def compute_mars_sun_distance(et: float) -> float:
    """Compute the Mars-sun distance [km] for a given ephemeris time.

    Parameters
    ----------
    et: float
        The ephemeris time.

    Returns
    -------
    float
        The Mars-sun distance.

    """
    return _compute_distance(et, 'SUN')


def compute_subspacecraft_local_time(et: float) -> float:
    """ Compute the sub-spacecraft local time for a given ephemeris time.

    Parameters
    ----------
    et: float
        The ephemeris time.

    Returns
    -------
    float
        The sub-spacecraft local time.

    """
    longitude = float(np.radians(compute_subspacecraft_point(et)[1]))
    # 499 is the IAU code for Mars
    hours, minutes, seconds, _, _ = spiceypy.et2lst(et, 499, longitude, 'planetocentric', timlen=256, ampmlen=256)
    return hours + minutes / 60 + seconds / 3600


def compute_intercept_point(et: float, vector: np.ndarray) -> tuple[float, float]:
    """Compute the point where a given look vector intercepts the planet for a given ephemeris time.

    Parameters
    ----------
    et: float
        The ephemeris time.
    vector
        A 1-dimensional unit vector describing IUVS' look direction.

    Returns
    -------
    tuple[float, float]
        The [latitude, longitude] of the sub-spacecraft point.

    """
    try:
        spoint, _, _ = spiceypy.sincpt('Ellipsoid', target, et, frame, aberration_correction, observer, frame, vector)
        return _rectangular_point_to_latitude_and_longitude(spoint)
    except spiceypy.utils.exceptions.NotFoundError:
        return np.nan, np.nan


def _rectangular_point_to_latitude_and_longitude(point: np.ndarray) -> tuple[float, float]:
    _, colatpoint, lonpoint = spiceypy.recsph(point)
    latitude = 90 - np.degrees(colatpoint)
    longitude = np.degrees(lonpoint) % 360
    return latitude, longitude


def _compute_distance(et: float, observr: str) -> float:
    _, _, surface_vector = spiceypy.subpnt(method, target, et, frame, aberration_correction, observr)
    return np.sqrt(np.sum(surface_vector ** 2))
