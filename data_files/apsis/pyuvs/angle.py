"""This module has generic functions for computing the angle between points.
"""
import numpy as np


def haversine(grid: tuple[np.ndarray, np.ndarray], target: tuple[float, float]) -> np.ndarray:
    """Compute the angular distance between an array of latitude/longitude points and a target point.

    Parameters
    ----------
    grid
        The [latitude, longitude] of the grid. The latitude and longitude must have the same shape and are assumed to be
        in degrees.
    target
        The [latitude, longitude] of the target point. Both values are assumed to be in degrees.

    Returns
    -------
    np.ndarray
        The angular distance between the point and the grid in degrees.

    Examples
    --------
    Get the angle between points on opposite sides of the planet.

    >>> import numpy as np
    >>> import pyuvs as pu
    >>> pu.haversine((np.array([0]), np.array([0])), (0, 180))
    array([180.])

    Get the angle between any point on the equator and the pole.

    >>> pu.haversine((np.array([0]), np.array([34])), (90, 300))
    array([90.])

    """
    latitude = np.radians(grid[0])
    longitude = np.radians(grid[1])
    target_latitude = np.radians(target[0])
    target_longitude = np.radians(target[1])

    a = np.sin((latitude - target_latitude) / 2) ** 2 + \
        np.cos(target_latitude) * np.cos(latitude) * np.sin((longitude - target_longitude) / 2) ** 2
    angle = 2 * np.arcsin(np.sqrt(a))
    return np.degrees(angle)
