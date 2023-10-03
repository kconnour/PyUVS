from pathlib import Path

from netCDF4 import Dataset
import numpy as np


class Ames:
    def __init__(self):
        self._base_path = Path('/mnt/science/data_lake/mars/gcm/ames/my30')
        self._simulation = Dataset(self._base_path / 'c48L36_my30.atmos_diurn.nc')

        # Grid variables
        self.ak = self._simulation.variables['pk']
        self.bk = self._simulation.variables['bk']

        self.latitude_bin_centers = self._simulation.variables['lat']
        self.longitude_bin_centers = self._simulation.variables['lon']
        self.local_time_bin_centers = self._simulation.variables['time_of_day_24']
        self.total_sol_bin_centers = self._simulation.variables['time']
        self.areo = self._simulation.variables['areo']

        # Surface variables
        self.surface_pressure = self._simulation.variables['ps']
        self.surface_temperature = self._simulation.variables['ts']

        # Atmospheric variables
        self.atmospheric_temperature = self._simulation.variables['temp']

        # Aerosols
        self.dust_vertical_profile = self._simulation.variables['dustref']
        self.dust_visible_optical_depth = self._simulation.variables['dodvis']
        self.ice_vertical_profile = self._simulation.variables['cldref']
        self.ice_visible_optical_depth = self._simulation.variables['taucloud_VIS']

    def get_solar_longitude(self) -> np.ndarray:
        """Get the solar longitude of the input grid.

        Returns
        -------
        np.ndarray
            The solar longitude of each time axis in the simulation.

        Notes
        -----
        This is the solar longitude of each seasonal and local time bin.

        """
        return self.areo[..., 0] % 360

    def get_nearest_latitude_index(self, latitude: float) -> int:
        """Get the index along the latitude axis that is closest to an input latitude.

        Parameters
        ----------
        latitude: float
            Any latitude.

        Returns
        -------
        int
            The nearest index to an input latitude.

        """
        return np.argmin(np.abs(self.latitude_bin_centers[:] - latitude))

    def get_nearest_longitude_index(self, longitude: float) -> int:
        """Get the index along the longitude axis that is closest to an input longitude.

        Parameters
        ----------
        longitude: float
            Any longitude.

        Returns
        -------
        int
            The nearest index to an input longitude.

        """
        return np.argmin(np.abs(self.longitude_bin_centers[:] - longitude))

    def get_nearest_local_time_index(self, longitude: float, local_time: float) -> int:
        """Get the index along the longitude axis that is closest to the local time at an input longitude.

        Parameters
        ----------
        longitude: float
            Any longitude.
        local_time: float
            The local time at the input longitude.

        Returns
        -------
        int
            The nearest index of the local time.

        Notes
        -----
        The GCM uses the convention that the local time in the simulation is
        at longitude 0, which is why this transformation is necessary.

        """
        return np.argmin(np.abs(self.local_time_bin_centers - (local_time - (longitude / 360 * 24)) % 24))

    def get_nearest_seasonal_index(self, sol: float) -> int:
        """Get the seasonal index corresponding to an input sol.

        Parameters
        ----------
        sol

        Returns
        -------

        Notes
        -----
        The simulation may not go from 0 to 360 degrees Ls... I have one that
        goes from 10 to 10. This algorithm works by finding the nearest sol
        bin and then shifting it by the right amount of bins by Ls.

        """
        total_sol_bin_centers = self.total_sol_bin_centers[:]
        delta = total_sol_bin_centers[1] - total_sol_bin_centers[0]
        yearly_sol_bin_centers = total_sol_bin_centers - total_sol_bin_centers[0] + delta/2
        seasonal_index = np.argmin(np.abs(yearly_sol_bin_centers - sol))
        solar_longitudes = self.get_solar_longitude()[:, 0]
        solar_longitude_diff = np.diff(solar_longitudes)
        discontinuity_idx = np.argmax(solar_longitude_diff < 0)
        return seasonal_index + (discontinuity_idx - yearly_sol_bin_centers.size)

    def get_atmospheric_pressure(self, surface_pressure: np.ndarray) -> np.ndarray:
        """Get the pressure from any arbitrary array of surface pressures;
        i.e. the surface pressure can be N-dimensional.

        Parameters
        ----------
        surface_pressure: np.ndarray
            The surface pressure.

        Returns
        -------
        np.ndarray
            The atmospheric_pressure of the array.

        """
        return np.multiply.outer(surface_pressure, self.bk[:]) + self.ak[:]
