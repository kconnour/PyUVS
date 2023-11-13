from pathlib import Path
import numpy as np
from netCDF4 import Dataset
from h5py import File
import mars_time as mt
import math
from astropy.io import fits
import matplotlib.pyplot as plt


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


def regrid_pcm():
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
    dust_scaling = dustuv / dustvis

    foo = ice_cext[np.argmin(np.abs(ice_particle_sizes - 5))]
    icevis = np.mean(foo[(0.4 <= ice_wavelengths) & (ice_wavelengths <= 0.8)])
    iceuv = np.mean(foo[(0.2 <= ice_wavelengths) & (ice_wavelengths <= 0.3)])
    ice_scaling = iceuv / icevis

    for mars_year in [33, 34, 35]:
        regridded_gcm_dust = np.zeros((72, 24, 49, 65)) * np.nan  # 72 5 Ls bins over 3 MYs, 24 LTs, 49 lats, 65 lons. This is 126 MB
        regridded_gcm_ice = np.zeros((72, 24, 49, 65)) * np.nan  # 72 5 Ls bins over 3 MYs, 24 LTs, 49 lats, 65 lons. This is 126 MB

        # Regrid over Ls
        for month in range(12):
            month_str = str(month + 1).zfill(2)
            pcm_path = Path(f'/mnt/science/data_lake/mars/gcm/pcm/my{mars_year}') / f'diagfi{month_str}_MY{mars_year}_OPAext.nc'
            pcm = Dataset(pcm_path)

            pcm_dust = pcm.variables['tau_dust'][:]
            pcm_ice = pcm.variables['tau_h2o_ice'][:]

            pcm_dust = np.reshape(pcm_dust, (pcm_dust.shape[0]//24, 24, pcm_dust.shape[1], pcm_dust.shape[2]))
            pcm_ice = np.reshape(pcm_ice, (pcm_ice.shape[0]//24, 24, pcm_ice.shape[1], pcm_ice.shape[2]))

            # hack: say Ls is linear within a Mars month
            monthly_sols = pcm_dust.shape[0]
            monthly_ls_bounds = [int(monthly_sols * i / 6) for i in range(6)]
            monthly_ls_bounds.append(monthly_sols)

            for binidx in range(len(monthly_ls_bounds)-1):
                regridded_gcm_dust[month * 6 + binidx] = np.mean(pcm_dust[monthly_ls_bounds[binidx]: monthly_ls_bounds[binidx+1]], axis=0)
                regridded_gcm_ice[month * 6 + binidx] = np.mean(pcm_ice[monthly_ls_bounds[binidx]: monthly_ls_bounds[binidx + 1]], axis=0)

        # Do the LT transformation
        def do_lt_transformation(regridded_arr):
            # Make a diagonalized array
            diag = np.zeros((24, 65))
            for i in range(24):
                low = int(i / 24 * 65)
                high = int((i + 1) / 24 * 65)
                diag[i, low:high] = 1

            # Roll the array by 180 degrees longitude to account for the fact that their grid is [-180, 180] so LT=0 is in the center of the array
            diag = np.roll(diag, 65//2, axis=1)

            # Roll the diagonalized array to only pick out the LTs of interest and put them the regridded array
            reshaped_arr = np.moveaxis(regridded_arr, 2, 1)  # shape: (72, 90, 24, 180)    testing:i changed regridded_arr to arr
            answer = np.zeros((72, 24, 49, 65))  # testing: i changed 72 to 133
            for i in range(24):
                roll_amount = int(-i / 24 * 65)
                rolled_diag = np.flip(np.roll(diag, roll_amount, axis=1), axis=1)
                data = reshaped_arr * rolled_diag
                answer[:, i, :, :] = np.sum(data, axis=-2)

            return answer

        dust = do_lt_transformation(regridded_gcm_dust) * dust_scaling
        ice = do_lt_transformation(regridded_gcm_ice) * ice_scaling

        np.save(f'/mnt/science/data_lake/mars/gcm/pcm/my{mars_year}/dust_lt_frame.npy', dust)
        np.save(f'/mnt/science/data_lake/mars/gcm/pcm/my{mars_year}/ice_lt_frame.npy', ice)


def regrid_ames():
    ames_path = Path(f'/mnt/science/data_lake/mars/gcm/ames/my30') / f'c48L36_my30.atmos_diurn.nc'
    ames = Dataset(ames_path)

    ames_dust = np.roll(ames['dodvis'][:], 3, axis=0)
    ames_ice = np.roll(ames['taucloud_VIS'][:], 3, axis=0)

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
    dust_scaling = dustuv / dustvis

    foo = ice_cext[np.argmin(np.abs(ice_particle_sizes - 5))]
    icevis = np.mean(foo[(0.4 <= ice_wavelengths) & (ice_wavelengths <= 0.8)])
    iceuv = np.mean(foo[(0.2 <= ice_wavelengths) & (ice_wavelengths <= 0.3)])
    ice_scaling = iceuv / icevis

    def regrid(arr: np.ndarray):
        # arr shape: (133, 24, 90, 180)
        ls_bounds = [mt.MarsTime(30, i).solar_longitude for i in np.arange(133) * 5]
        ls_bounds[0] = 0
        ls_target_grid_right_edges = np.arange(360 / 5) * 5 + 5

        # testing: i commented this all out to keep the same native grid
        ls_idx = np.array([np.argmax(np.abs(ls_bounds[i] < ls_target_grid_right_edges)) for i in range(len(ls_bounds))])  # This is the bin index each point seasonal point belongs in
        regridded_arr = np.zeros((72, 24, 90, 180)) * np.nan  # 72 Ls bins 5 degrees each, 4 LT bins 3 hours each, 90 lat, 180 lon
        for i in range(72):
            idx = ls_idx == i
            if np.sum(idx) == 1:
                regridded_arr[i] = arr[idx]
            else:
                regridded_arr[i] = np.mean(arr[idx], axis=0)

        # Make a diagonalized array
        diag = np.zeros((24, 180))
        for i in range(24):
            low = int(i / 24 * 180)
            high = int((i + 1) / 24 * 180)
            diag[i, low:high] = 1

        # Roll the diagonalized array to only pick out the LTs of interest and put them the regridded array
        reshaped_arr = np.moveaxis(regridded_arr, 2, 1)  # shape: (72, 90, 24, 180)    testing:i changed regridded_arr to arr
        answer = np.zeros((72, 24, 90, 180))  # testing: i changed 72 to 133
        for i in range(24):
            roll_amount = int(-i / 24 * 180)
            rolled_diag = np.flip(np.roll(diag, roll_amount, axis=1), axis=1)  # I have no fucking goddamn fucking clue why I need the fucking flip but it fucking works with it
            data = reshaped_arr * rolled_diag
            answer[:, i, :, :] = np.sum(data, axis=-2)

        return answer

    regridded_dust = regrid(ames_dust) * dust_scaling
    regridded_ice = regrid(ames_ice) * ice_scaling

    np.save('/mnt/science/data_lake/mars/gcm/ames/my30/dust_lt_frame.npy', regridded_dust)
    np.save('/mnt/science/data_lake/mars/gcm/ames/my30/ice_lt_frame.npy', regridded_ice)


if __name__ == '__main__':
    #regrid_ames()
    regrid_pcm()

    # NOTE: PCM latitude is +90 to -90
    # Note Ames latitude is -90 to +90

    '''import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 2)

    # Add in Ames dust and ice
    ames_X = np.arange(73) * 5
    ames_Y = np.arange(91) * 2 - 90

    ames_regridded_dust = np.load('/mnt/science/data_lake/mars/gcm/ames/my30/dust_lt_frame.npy')  # (72, 24, 90, 180)
    ames_regridded_ice = np.load('/mnt/science/data_lake/mars/gcm/ames/my30/ice_lt_frame.npy')   # (72, 24, 90, 180)

    ames_dust_to_plot = np.zeros((4, 72, 90))
    ames_ice_to_plot = np.zeros((4, 72, 90))
    for lt in range(4):
        ames_dust_to_plot[lt] = np.mean(np.mean(ames_regridded_dust[:, lt * 3 + 6: (lt + 1) * 3 + 6], axis=-1), axis=1)
        ames_ice_to_plot[lt] = np.mean(np.mean(ames_regridded_ice[:, lt * 3 + 6: (lt + 1) * 3 + 6], axis=-1), axis=1)

    ax[1, 0].pcolormesh(ames_X, ames_Y, ames_dust_to_plot[3].T, vmin=0, vmax=2, cmap='cividis')
    ax[1, 1].pcolormesh(ames_X, ames_Y, ames_ice_to_plot[3].T, vmin=0, vmax=1, cmap='viridis')

    # Add in PCM dust and ice
    pcm_X = np.arange(73) * 5
    pcm_Y = np.flip(np.arange(50) * 180/49 - 90)

    pcm_regridded_dust = np.load('/mnt/science/data_lake/mars/gcm/pcm/my33/dust_lt_frame.npy')
    pcm_regridded_ice = np.load('/mnt/science/data_lake/mars/gcm/pcm/my33/ice_lt_frame.npy')

    pcm_dust_to_plot = np.zeros((4, 72, 49))
    pcm_ice_to_plot = np.zeros((4, 72, 49))

    for lt in range(4):
        pcm_dust_to_plot[lt] = np.mean(np.mean(pcm_regridded_dust[:, lt * 3 + 6: (lt + 1) * 3 + 6], axis=-1), axis=1)
        pcm_ice_to_plot[lt] = np.mean(np.mean(pcm_regridded_ice[:, lt * 3 + 6: (lt + 1) * 3 + 6], axis=-1), axis=1)

    ax[2, 0].pcolormesh(pcm_X, pcm_Y, pcm_dust_to_plot[3].T, vmin=0, vmax=2, cmap='cividis')
    ax[2, 1].pcolormesh(pcm_X, pcm_Y, pcm_ice_to_plot[3].T, vmin=0, vmax=1, cmap='viridis')

    plt.savefig('/home/kyle/thesis/season3.png')'''
