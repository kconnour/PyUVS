"""This module contains functions for working with IUVS orbits.
"""
import math


def make_orbit_block(orbit: int) -> str:
    """Make the orbit block corresponding to an input orbit.

    Parameters
    ----------
    orbit
        The orbit number.

    Returns
    -------
    str
        The orbit block.

    Examples
    --------
    Make the orbit block for orbit 3453

    >>> import pyuvs as pu
    >>> pu.make_orbit_block(3453)
    'orbit03400'

    """
    block = math.floor(orbit / 100) * 100
    return 'orbit' + f'{block}'.zfill(5)


def make_orbit_code(orbit: int) -> str:
    """Make the orbit code corresponding to an input orbit.

    Parameters
    ----------
    orbit
        The orbit number.

    Returns
    -------
    str
        The orbit code.

    See Also
    --------
    make_orbit_block: Make the orbit block corresponding to a given orbit.

    Examples
    --------
    Make the orbit code for orbit 3453

    >>> import pyuvs as pu
    >>> pu.make_orbit_code(3453)
    'orbit03453'

    """
    return 'orbit' + f'{orbit}'.zfill(5)
