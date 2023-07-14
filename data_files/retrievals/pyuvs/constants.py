"""This module contains constants relevant to IUVS.
"""
from datetime import datetime
import numpy as np


orbit_insertion_date: datetime = datetime(2014, 9, 22, 2, 24, 0)
"""Date of MAVEN's orbital insertion. 

Notes
-----
This value comes from `JPL 
<https://www.jpl.nasa.gov/news/nasas-newest-mars-mission-spacecraft-enters-orbit-around-red-planet>`_.
"""

spatial_slit_thickness: float = 0.1
"""Thickness of the slit [mm].

Notes
-----
This value comes from Table 3 (p. 91) of the `IUVS paper
<https://link.springer.com/article/10.1007/s11214-014-0098-7>`_.
"""

spatial_slit_length: float = 19.8
"""Length of the slit [mm].

Notes
-----
This value comes from Table 3 (p. 91) of the `IUVS paper
<https://link.springer.com/article/10.1007/s11214-014-0098-7>`_.
"""

telescope_focal_length: float = 99.5
"""Focal length of the IUVS telescope mirror [mm].

Notes
-----
This value comes from an email Alan Hoskins where he noted that he used this
value in his ray tracing code. The IUVS paper simply notes this value is 
100 mm (see Table 3, p. 91) with only 1 significant figure. 
"""

detector_length: int = 22
"""Length of the length of the detector [mm]."""

angular_detector_width: float = spatial_slit_length / telescope_focal_length * 180 / np.pi
"""Angular width of the detector [degrees]."""

angular_slit_width: float = spatial_slit_length / telescope_focal_length * 180 / np.pi
r"""Width of the slit [degrees].

Notes
-----
This is the angular size of the slit, which covers neither keyhole.
The formula for creating this constant is

.. math::

   \theta = \frac{s}{f} * \frac{180}{\pi}

where :math:`s` is the spatial slit length and :math:`f` is the telescope
focal length.
"""

pixel_length: float = 22 / 1024
"""Length of an IUVS detector pixel [mm].

Notes
-----
This values is derived from figure 11 of the IUVS paper, where the pixel length
is simply the length of the detector divided into 1024 pixels.
"""

pixel_angular_size: float = pixel_length / telescope_focal_length * \
                            spatial_slit_thickness / telescope_focal_length
r"""Angular size of a detector pixel [sr]. 

Notes
-----
The formula for creating this constant is

.. math::

   \omega = \frac{s}{f} * \frac{l}{f}

where :math:`s` is the spatial length of a pixel, :math:`f` is the telescope
focal length, and :math:`l` is the thickness of the slit.
"""

minimum_mirror_angle: float = 30.2508544921875
"""Minimum angle [degrees] the scan mirror can obtain.

Notes
-----
This value comes from Justin Deighan.
"""

maximum_mirror_angle: float = 59.6502685546875
"""Maximum angle [degrees] the scan mirror can obtain.

Notes
-----
This value comes from Justin Deighan.
"""

apoapse_muv_day_night_voltage_boundary: int = 790
"""Voltage defining the boundary between dayside and nightside settings.

Notes
-----
This value is simply an engineering convention and has no physical basis.
"""

apoapse_muv_failsafe_voltage: float = 497.63803
"""The voltage of the failsafe mode.

Notes
-----
This value is simply an engineering convention and has no physical basis.
"""