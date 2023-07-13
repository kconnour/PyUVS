"""This module contains constants and conversions factors relevant to IUVS.
"""
import datetime

import numpy as np

maven_arrival = datetime.datetime(2014, 9, 22, 2, 24)

spice_start = datetime.datetime(2014, 9, 22, 3, 0)   # The datetime to start SPICE searches from

angular_slit_width: float = 10.64
"""Width of the slit [degrees]."""

spatial_slit_width: float = 0.1
"""Width of the slit [mm]."""

cmos_pixel_well_depth: int = 3400
"""Saturation level of an IUVS CMOS detector pixel [DN]."""

day_night_voltage_boundary: int = 790
"""Voltage defining the boundary between dayside and nightside settings."""

minimum_mirror_angle: float = 30.2508544921875
"""Minimum angle [degrees] the scan mirror can be."""

maximum_mirror_angle: float = 59.6502685546875
"""Maximum angle [degrees] the scan mirror can be."""

telescope_focal_length: int = 100
"""Focal length of the IUVS telescope mirror [mm]."""

pixel_size: float = 24/1024
"""Size of an IUVS detector pixel [mm]. From Fig 11, p94 of the original IUVS paper."""

pixel_omega: float = pixel_size / telescope_focal_length * spatial_slit_width / telescope_focal_length
"""Detector pixel angular dispersion along the slit [sr]."""

muv_wavelength_width: float = 0.16367098
"""Docstring."""

kR: float = 10**9 / (4 * np.pi)
"""Definition of the kilorayleigh [photons/second/m**2/steradian]."""

radius_mars: float = 3.3895 * 10**6
"""Radius of Mars [m]"""
