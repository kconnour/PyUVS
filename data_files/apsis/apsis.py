"""This module provides functions for computing MAVEN's apses.

"""
import numpy as np
import spiceypy

import pyuvs as pu


def compute_maven_apsis_et(segment='apoapse', step_size: float = 60) -> np.ndarray:
    """Compute the ephemeris time at MAVEN's apsis.

    Parameters
    ----------
    segment : str
        The orbit point at which to calculate the ephemeris time. Must be either "apoapse" or "periapse".
    step_size: float
        The step size [seconds] to use for the search.

    Returns
    -------
    np.ndarray
        Ephemeris times at the chosen apsis.

    Notes
    -----
    You must have already furnished the full mission's kernels for this to work.

    """
    et_start = spiceypy.datetime2et(pu.spice_start)
    et_end = spiceypy.datetime2et(pu.get_latest_spk_datetime())

    abcorr = 'NONE'
    match segment:
        case 'apoapse':
            relate = 'LOCMAX'
            refval = 3396 + 6200
        case 'periapse':
            relate = 'LOCMIN'
            refval = 3396 + 500

    cnfine = spiceypy.utils.support_types.SPICEDOUBLE_CELL(2)
    spiceypy.wninsd(et_start, et_end, cnfine)
    ninterval = round((et_end - et_start) / step_size)
    result = spiceypy.utils.support_types.SPICEDOUBLE_CELL(round(1.1 * (et_end - et_start) / 4.5))
    spiceypy.gfdist(pu.target, abcorr, pu.observer, relate, refval, 0, step_size, ninterval, cnfine, result=result)
    count = spiceypy.wncard(result)
    et = np.zeros(count)
    for i in range(count):
        lr = spiceypy.wnfetd(result, i)
        left = lr[0]
        right = lr[1]
        if left == right:
            et[i] = left
    return et
