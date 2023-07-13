"""This module is for working with SPICE kernels produced by JPL, *not* the ones on the LASP VM.

The JPL kernels can be found here: https://naif.jpl.nasa.gov/pub/naif/pds/pds4/maven/maven_spice/spice_kernels/
SPICE info: https://naif.jpl.nasa.gov/naif/spiceconcept.html

"""
import datetime
from pathlib import Path

import spiceypy

kernel_directory = Path('/mnt/science/data_lake/mars/maven/spice')


def furnish_ck_files() -> None:
    """Furnish C-kernels.

    Returns
    -------
    None

    """
    location = kernel_directory / 'ck'
    app_kernels = sorted(location.glob('mvn_app_rel*.bc'))
    for kernel in app_kernels:
        spiceypy.furnsh(str(kernel))
    sc_kernels = sorted(location.glob('mvn_sc_rel*.bc'))
    for kernel in sc_kernels:
        spiceypy.furnsh(str(kernel))


def furnish_lsk_files() -> None:
    """Furnish leap-second kernels.

    Returns
    -------
    None

    """
    location = kernel_directory / 'lsk'
    kernels = sorted(location.glob('naif*.tls'))
    for kernel in kernels:
        spiceypy.furnsh(str(kernel))


def furnish_pck_files() -> None:
    """Furnish P-kernels.

    Returns
    -------
    None

    """
    location = kernel_directory / 'pck'
    kernels = sorted(location.glob('pck*.tpc'))
    for kernel in kernels:
        spiceypy.furnsh(str(kernel))


def furnish_sclk_files() -> None:
    """Furnish spacecraft clock kernels.

    Returns
    -------
    None

    """
    location = kernel_directory / 'sclk'
    kernels = sorted(location.glob('mvn_sclkscet_*.tsc'))
    for kernel in kernels:
        spiceypy.furnsh(str(kernel))


def furnish_spk_files() -> None:
    """Furnish the spacecraft and planetary kernels.

    Returns
    -------
    None

    """
    location = kernel_directory / 'spk'
    kernels = sorted(location.glob('maven_orb_rec*.bsp'))
    for kernel in kernels:
        spiceypy.furnsh(str(kernel))

    mars = location / 'mar097s.bsp'
    spiceypy.furnsh(str(mars))


def get_latest_spk_datetime() -> datetime.datetime:
    """Get the datetime corresponding to the latest SPK file.

    Returns
    -------
    datetime.datetime
        The latest datetime.

    """
    location = kernel_directory / 'spk'
    kernels = sorted(location.glob('maven_orb_rec*.bsp'))
    latest_kernel = kernels[-1].name
    timestamp = latest_kernel.split('_')[-2]
    return datetime.datetime.strptime(timestamp, '%y%m%d')
