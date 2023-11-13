import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
from pathlib import Path
from astropy.io import fits
from netCDF4 import Dataset
from swath import swath_number
from constants import *
from h5py import File

cmap = plt.get_cmap('copper')

mola = np.load('/mnt/science/data_lake/mars/maps/mola-topography.npy') / 1000   # the base unit of the array is meters, but dividing by 1000 changes it to km
mola_lat = np.linspace(90, -90, num=1440)
mola_lon = np.linspace(0, 360, num=2880)


class Radprop:
    def __init__(self):
        self._base_path = Path('/mnt/science/data_lake/mars/radiative_properties')
        self.hdul = None

    def _get_file_forward_scattering_properties(self) -> np.ndarray:
        return self.hdul['forw'].data

    def _get_file_legendre_coefficients(self) -> np.ndarray:
        return self.hdul['pmom'].data

    def _get_file_phase_function(self) -> np.ndarray:
        return self.hdul['phsfn'].data

    def _get_file_phase_function_reexpansion(self) -> np.ndarray:
        return self.hdul['expansion'].data

    def get_primary(self) -> np.ndarray:
        """Get the primary hdul.

        Returns
        -------
        The primary structure

        Notes
        -----
        This array should contain no relevant info.

        """
        return self.hdul['primary'].data

    def get_particle_sizes(self) -> np.ndarray:
        """Get the particle sizes associated with each of the radiative properties.

        Returns
        -------
        The particle sizes

        Notes
        -----
        The particle sizes are the centers of some particle size distribution.

        """
        return self.hdul['particle_sizes'].data

    def get_wavelengths(self) -> np.ndarray:
        return self.hdul['wavelengths'].data

    def get_scattering_angles(self) -> np.ndarray:
        return self.hdul['scattering_angle'].data

    def get_scattering_cross_sections(self) -> np.ndarray:
        return self._get_file_forward_scattering_properties()[..., 1]

    def get_extinction_cross_sections(self) -> np.ndarray:
        return self._get_file_forward_scattering_properties()[..., 0]

    def get_asymmetry_parameters(self) -> np.ndarray:
        return self._get_file_forward_scattering_properties()[..., 2]

    def get_phase_functions(self) -> np.ndarray:
        return np.moveaxis(self._get_file_phase_function(), 0, -1)

    def get_legendre_coefficients(self) -> np.ndarray:
        return np.moveaxis(self._get_file_legendre_coefficients(), 0, -1)

    def get_phase_function_reexpansions(self) -> np.ndarray:
        return np.moveaxis(self._get_file_phase_function_reexpansion(), 0, -1)

    def _get_header(self) -> fits.header.Header:
        return self.hdul['primary'].header

    def get_file_creation_date(self) -> str:
        header = self._get_header()
        return header['date']

    def get_history(self) -> str:
        header = self._get_header()
        history = str(header['history'])
        # The ??? currently only applies to the dust file
        return history.replace('???', '--')


class Dust(Radprop):
    def __init__(self):
        super().__init__()
        self.hdul = fits.open(self._base_path / 'dust.fits.gz')


class Ice(Radprop):
    def __init__(self):
        super().__init__()
        self.hdul = fits.open(self._base_path / 'ice.fits.gz')


for orbit in range(3453, 3454):
    apsis_file = File('/mnt/science/data_lake/mars/maven/apsis.hdf5')
    all_orbits = apsis_file['apoapse/orbit'][:]
    sol = apsis_file['apoapse/sol'][all_orbits==orbit][0]
    lt = apsis_file['apoapse/subspacecraft_local_time'][all_orbits==orbit]
    ls = apsis_file['apoapse/solar_longitude'][all_orbits==orbit][0]
    subsclon = apsis_file['apoapse/subspacecraft_longitude'][all_orbits == orbit]
    mars_year = apsis_file['apoapse/mars_year'][all_orbits==orbit][0]

    ames_lt_idx = np.argmin(np.abs(np.arange(24)+0.5 - (lt - (subsclon / 360 * 24)) % 24))
    ames_sol_idx = int(sol / 668 * 133)

    gcm_path = Path('/mnt/science/data_lake/mars/gcm/ames/my30') / 'c48L36_my30.atmos_diurn.nc'
    yearly_gcm = Dataset(gcm_path)

    dust_radprop = Dust()
    ice_radprop = Ice()

    # Load in radprop
    dust_cext = dust_radprop.get_scattering_cross_sections()
    ice_cext = ice_radprop.get_scattering_cross_sections()

    dust_wavelengths = dust_radprop.get_wavelengths()
    dust_particle_sizes = dust_radprop.get_particle_sizes()
    ice_wavelengths = ice_radprop.get_wavelengths()
    ice_particle_sizes = ice_radprop.get_particle_sizes()

    foo = dust_cext[np.argmin(np.abs(dust_particle_sizes - 1.5))]
    dustvis = np.mean(foo[(0.4 <= dust_wavelengths) & (dust_wavelengths <= 0.8)])
    dustuv = np.mean(foo[(0.2 <= dust_wavelengths) & (dust_wavelengths <= 0.3)])
    dust_scaling = dustvis / dustuv

    foo = ice_cext[np.argmin(np.abs(ice_particle_sizes - 5))]
    icevis = np.mean(foo[(0.4 <= ice_wavelengths) & (ice_wavelengths <= 0.8)])
    iceuv = np.mean(foo[(0.2 <= ice_wavelengths) & (ice_wavelengths <= 0.3)])
    ice_scaling = icevis / iceuv

    gcm_dust = yearly_gcm['dodvis'][ames_sol_idx, ames_lt_idx]
    gcm_ice = yearly_gcm['taucloud_VIS'][ames_sol_idx, ames_lt_idx]

    fig, ax = plt.subplots(2, 2)
    X = np.arange(181) * 2
    Y = np.arange(91) * 2 - 90

    gcm_dust_ax = ax[0, 0].pcolormesh(X, Y, gcm_dust * dust_scaling, vmin=0, vmax=2, cmap='cividis')
    gcm_ice_ax = ax[0, 1].pcolormesh(X, Y, gcm_ice * ice_scaling, vmin=0, vmax=1, cmap='viridis')

    ax[0, 0].contour(mola_lon, mola_lat, mola, [0, 5, 10, 15, 20], colors=[cmap(0), cmap(1/4), cmap(2/4), cmap(3/4), cmap(1)], linewidths=[0.5], linestyles=['solid'])

    plt.savefig('/home/kyle/amestopotest.png')
