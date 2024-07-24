"""This module provides functions for computing geometry about MAVEN's orbit."""
import datetime

import numpy as np
import spiceypy

from kernels import get_datetime_of_earliest_maven_orbit_kernel, \
    get_datetime_of_latest_maven_orbit_kernel


def compute_maven_apsis_ephemeris_times(
        segment: str = 'apoapse',
        start_time: datetime.datetime = get_datetime_of_earliest_maven_orbit_kernel(),
        end_time: datetime.datetime = get_datetime_of_latest_maven_orbit_kernel(),
        step_size: float = 60) -> np.ndarray:
    """Compute the ephemeris time at MAVEN's apses.

    Parameters
    ----------
    segment : str
        The orbit point at which to calculate the ephemeris time. Must be either
        "apoapse" or "periapse".
    start_time: datetime
        The earliest datetime to include in the search.
    end_time: datetime
        The latest datetime to include in the search.
    step_size: float
        The step size [seconds] to use for the search.

    Returns
    -------
    np.ndarray
        Ephemeris times for the chosen orbit segment.

    Notes
    -----
    This requires the leap second kernel, the Mars kernel, and the
    MAVEN orbit kernels.

    """
    # Define the starting and ending ephemeris times
    et_start = spiceypy.datetime2et(start_time)
    et_end = spiceypy.datetime2et(end_time)

    # Define values for the apoapse and periapse segments
    abcorr = 'NONE'
    match segment:
        case 'apoapse':
            relate = 'LOCMAX'
            refval = 3396 + 6200
        case 'periapse':
            relate = 'LOCMIN'
            refval = 3396 + 500
        case _:
            raise ValueError('The segment must either be "apoapse" or '
                             '"periapse".')

    # Make a double precision window and insert it
    cnfine = spiceypy.utils.support_types.SPICEDOUBLE_CELL(2)
    spiceypy.wninsd(et_start, et_end, cnfine)

    # Get the window over which a constraint is met
    ninterval = round((et_end - et_start) / step_size)
    result = spiceypy.utils.support_types.SPICEDOUBLE_CELL(
        round(1.1 * (et_end - et_start) / 4.5))
    spiceypy.gfdist('Mars', abcorr, 'MAVEN', relate, refval,
                    0, step_size, ninterval, cnfine, result=result)

    # Get the number of maxima or minima that SPICE found
    n_extrema = spiceypy.wncard(result)
    apsis_ephemeris_times = np.zeros(n_extrema) * np.nan

    if n_extrema == 0:
        print('No apsis points found. Try making a larger temporal window.')
    else:
        for i in range(n_extrema):
            lr = spiceypy.wnfetd(result, i)
            left = lr[0]
            right = lr[1]
            if left == right:
                apsis_ephemeris_times[i] = left

    return apsis_ephemeris_times


def compute_subspacecraft_point(ephemeris_time: float) -> tuple[float, float]:
    """Compute MAVEN's sub-spacecraft point at a given ephemeris time.

    Parameters
    ----------
    ephemeris_time: float
        The ephemeris time.

    Returns
    -------
    tuple[float, float]
        MAVEN's sub-spacecraft point (latitude [degrees N],
        longitude [degrees E]) at the input ephemeris time.

    Notes
    -----
    This requires the planetary constants kernel, the Mars kernel, and the
    MAVEN orbit kernels.

    """
    sub_observer_point, _, _ = spiceypy.subpnt('Intercept: ellipsoid',
    'Mars', ephemeris_time, 'IAU_MARS', 'LT+S', 'MAVEN')
    _, colatitude, longitude = spiceypy.recsph(sub_observer_point)
    subspacecraft_latitude = 90 - np.degrees(colatitude)
    subspacecraft_longitude = np.degrees(longitude) % 360
    return subspacecraft_latitude, subspacecraft_longitude


def compute_spacecraft_altitude(ephemeris_time: float) -> float:
    """Compute MAVEN's altitude at a given ephemeris time.

    Parameters
    ----------
    ephemeris_time: float
        The ephemeris time.

    Returns
    -------
    float
        MAVEN's altitude [km] at the input ephemeris time.

    Notes
    -----
    This requires the planetary constants kernel, the Mars kernel, and the
    MAVEN orbit kernels.

    """
    # surface_vector is the vector from the observer to sub-observer point
    _, _, surface_vector = spiceypy.subpnt('Intercept: ellipsoid',
    'Mars', ephemeris_time, 'IAU_MARS', 'LT+S', 'MAVEN')
    return np.sqrt(np.sum(surface_vector ** 2))


def compute_subspacecraft_local_time(ephemeris_time: float) -> float:
    """Compute the local solar time of the sub-spacecraft point at a given
    ephemeris time.

    Parameters
    ----------
    ephemeris_time
        The ephemeris time.

    Returns
    -------
    float
        The local solar time of the sub-spacecraft point at the input ephemeris
        time.

    Notes
    -----
    This requires the planetary constants kernel, the Mars kernel, and the
    MAVEN orbit kernels.

    """
    _, subspacecraft_longitude = compute_subspacecraft_point(ephemeris_time)
    hour, minute, second, _, _ = spiceypy.et2lst(ephemeris_time, 499,
        subspacecraft_longitude, 'planetocentric', timlen=256, ampmlen=256)
    return hour + minute / 60 + second / 3600


def compute_mars_state(ephemeris_time: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute the position and velocity of Mars relative to MAVEN at a given
    ephemeris time.

    Parameters
    ----------
    ephemeris_time
        The ephemeris time.

    Returns
    -------
    [np.ndarray, np.ndarray]
        The position and velocity of Mars relative to MAVEN.

    Notes
    -----
    This requires the MAVEN frame kernel, the Mars kernel, and the MAVEN orbit
    kernels.

    """
    state, _ = spiceypy.spkezr('Mars', ephemeris_time,
        'MAVEN_MME_2000', 'LT+S', 'MAVEN')
    position = state[0:3]
    velocity = state[3:6]
    return position, velocity
