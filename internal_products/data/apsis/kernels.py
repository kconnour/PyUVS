"""This module provides functions for loading in SPICE kernels.

It assumes the same file structure as on the NAIF."""
import datetime

import spiceypy

import pyuvs as pu


def furnish_app_kernels() -> None:
    """Furnish the reconstructed kernels of MAVEN's APP.

    Returns
    -------
    None

    """
    kernel_location = pu.spice_kernel_location / 'ck'
    app_kernels = sorted(kernel_location.glob('mvn_sc_rel*.bc'))
    for kernel in app_kernels:
        spiceypy.furnsh(str(kernel))


def furnish_iuvs_mirror_kernels() -> None:
    """Furnish the reconstructed kernels of IUVS's internal mirror.

    Returns
    -------
    None

    """
    kernel_location = pu.spice_kernel_location / 'ck'
    mirror_kernels = sorted(kernel_location.glob('mvn_iuvs_rem*.bc'))
    for kernel in mirror_kernels:
        spiceypy.furnsh(str(kernel))


def furnish_maven_orientation_kernels() -> None:
    """Furnish the reconstructed kernels of the MAVEN spacecraft's orientation.

    Returns
    -------
    None

    """
    kernel_location = pu.spice_kernel_location / 'ck'
    sc_kernels = sorted(kernel_location.glob('mvn_sc_rel*.bc'))
    for kernel in sc_kernels:
        spiceypy.furnsh(str(kernel))


def furnish_maven_frame_kernel() -> None:
    """Furnish the kernel defining reference frames for the MAVEN spacecraft
    and its instruments.

    Returns
    -------
    None

    """
    kernel = pu.spice_kernel_location / 'fk' / 'maven_v11.tf'
    spiceypy.furnsh(str(kernel))


def furnish_maven_instrument_kernels() -> None:
    """Furnish the instrumental kernels for all instruments on-board MAVEN.

    Returns
    -------
    None

    """
    kernel_location = pu.spice_kernel_location / 'ik'
    instrument_kernels = sorted(kernel_location.glob('maven*.ti'))
    for kernel in instrument_kernels:
        spiceypy.furnsh(str(kernel))


def furnish_leap_second_kernel() -> None:
    """Furnish the leap second kernel.

    Returns
    -------
    None

    """
    kernel = pu.spice_kernel_location / 'lsk' / 'naif0012.tls'
    spiceypy.furnsh(str(kernel))


def furnish_planetary_constants_kernel() -> None:
    """Furnish the planetary constants kernel.

    Returns
    -------
    None

    """
    kernel = pu.spice_kernel_location / 'pck' / 'pck00010.tpc'
    spiceypy.furnsh(str(kernel))


def furnish_maven_clock_kernels() -> None:
    """Furnish the MAVEN on-board clock kernels.

    Returns
    -------
    None

    """
    location = pu.spice_kernel_location / 'sclk'
    clock_kernels = sorted(location.glob('mvn_sclkscet*.tsc'))
    for kernel in clock_kernels:
        spiceypy.furnsh(str(kernel))


def furnish_maven_orbit_kernels() -> None:
    """Furnish reconstructed kernels of MAVEN's orbit.

    Returns
    -------
    None

    """
    location = pu.spice_kernel_location / 'spk'
    orbit_kernels = sorted(location.glob('maven_orb_rec*.bsp'))
    for kernel in orbit_kernels:
        spiceypy.furnsh(str(kernel))


def furnish_mars_kernel() -> None:
    """Furnish the Mars kernel.

    Returns
    -------
    None

    """
    kernel = pu.spice_kernel_location / 'spk' / 'mar097s.bsp'
    spiceypy.furnsh(str(kernel))


def get_datetime_of_earliest_maven_orbit_kernel() -> datetime.datetime:
    """Get the datetime associated with the most recent MAVEN orbit kernel.

    Returns
    -------
    datetime.datetime
        The datetime of the earliest orbit kernel.

    """
    location = pu.spice_kernel_location / 'spk'
    orbit_kernels = sorted(location.glob('maven_orb_rec*.bsp'))
    kernel_name = orbit_kernels[-0].name
    end_date = kernel_name.split('_')[-3]

    # The 3-hour offset accounts for the fact that the kernel doesn't start at
    # midnight on the file's date
    return (datetime.datetime.strptime(end_date, '%y%m%d') +
            datetime.timedelta(hours=3))


def get_datetime_of_latest_maven_orbit_kernel() -> datetime.datetime:
    """Get the datetime associated with the most recent MAVEN orbit kernel.

    Returns
    -------
    datetime.datetime
        The datetime of the most recent orbit kernel.

    """
    location = pu.spice_kernel_location / 'spk'
    orbit_kernels = sorted(location.glob('maven_orb_rec*.bsp'))
    kernel_name = orbit_kernels[-1].name
    end_date = kernel_name.split('_')[-2]
    return datetime.datetime.strptime(end_date, '%y%m%d')
