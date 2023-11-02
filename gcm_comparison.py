import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from netCDF4 import Dataset
from swath import swath_number
from constants import *
from h5py import File
import matplotlib.ticker as ticker
import mars_time as mt
import warnings
import multiprocessing as mp
import matplotlib.gridspec as gridspec

cmap = plt.get_cmap('copper')

# This is needed for John Wilson's topography request
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


cbar_props = {
    'aspect': 10,
    'pad': 0.01
}


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

    See Also
    --------
    make_orbit_code: Make the orbit code corresponding to a given orbit.

    Examples
    --------
    Make the orbit block for orbit 3453

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

    """
    return 'orbit' + f'{orbit}'.zfill(5)


hdulist = list[fits.hdu.hdulist.HDUList]
def _get_segment_orbit_channel_fits_file_paths(iuvs_fits_file_location: Path, segment: str, orbit: int, channel: str) -> list[Path]:
    orbit_block = make_orbit_block(orbit)
    orbit_code = make_orbit_code(orbit)
    return sorted((iuvs_fits_file_location / orbit_block).glob(f'*{segment}*{orbit_code}*{channel}*.gz'))


def _remove_files_with_oulier_obs_id(hduls: hdulist, obs_id: list[int]) -> hdulist:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        median_obs_id = np.median(obs_id)
    outlier_files = [f for c, f in enumerate(hduls) if obs_id[c] != median_obs_id]
    for file in outlier_files:
        hduls.remove(file)
    return hduls

def _get_apoapse_fits_files(iuvs_fits_file_location: Path, orbit: int, segment: str) -> hdulist:
    # Test case: orbit 7857 has outbound file as file index 0 with a strange obs id
    data_file_paths = _get_segment_orbit_channel_fits_file_paths(iuvs_fits_file_location, 'apoapse', orbit, segment)
    hduls = [fits.open(f) for f in data_file_paths]
    obs_id = [f['primary'].header['obs_id'] for f in hduls]
    return _remove_files_with_oulier_obs_id(hduls, obs_id)


def get_apoapse_muv_fits_files(orbit: int, iuvs_fits_file_location: Path) -> hdulist:
    return _get_apoapse_fits_files(iuvs_fits_file_location, orbit, 'muv')


def get_apoapse_muv_failsafe_files(orbit: int, iuvs_fits_file_location: Path) -> hdulist:
    apoapse_hduls = get_apoapse_muv_fits_files(orbit, iuvs_fits_file_location)
    mcp_voltage = [f['observation'].data['mcp_volt'][0] for f in apoapse_hduls]
    failsafe = [np.isclose(f, 497.63803) for f in mcp_voltage]
    return [f for c, f in enumerate(apoapse_hduls) if failsafe[c]]


def get_apoapse_muv_dayside_files(orbit: int, iuvs_fits_file_location: Path) -> hdulist:
    apoapse_hduls = get_apoapse_muv_fits_files(orbit, iuvs_fits_file_location)
    mcp_voltage = [f['observation'].data['mcp_volt'][0] for f in apoapse_hduls]
    failsafe = [np.isclose(f, 497.63803) for f in mcp_voltage]
    nightside = [f >= 790 for f in mcp_voltage]
    return [f for c, f in enumerate(apoapse_hduls) if not failsafe[c] and not nightside[c]]


def get_apoapse_muv_nightside_files(orbit: int, iuvs_fits_file_location: Path) -> hdulist:
    apoapse_hduls = get_apoapse_muv_fits_files(orbit, iuvs_fits_file_location)
    mcp_voltage = [f['observation'].data['mcp_volt'][0] for f in apoapse_hduls]
    nightside = [f >= 790 for f in mcp_voltage]
    return [f for c, f in enumerate(apoapse_hduls) if nightside[c]]


def process_image(orbit: int):
    try:
        apsis_file = File('/mnt/science/data_lake/mars/maven/apsis.hdf5')
        all_orbits = apsis_file['apoapse/orbit'][:]
        sol = apsis_file['apoapse/sol'][all_orbits==orbit][0]
        lt = apsis_file['apoapse/subspacecraft_local_time'][all_orbits==orbit][0]
        ls = apsis_file['apoapse/solar_longitude'][all_orbits==orbit][0]
        subsclon = apsis_file['apoapse/subspacecraft_longitude'][all_orbits == orbit][0]
        mars_year = apsis_file['apoapse/mars_year'][all_orbits==orbit][0]

        ames_lt_idx = np.argmin(np.abs(np.arange(24)+0.5 - (lt - (subsclon / 360 * 24)) % 24))
        ames_sol_idx = int(sol / 668 * 133) - 4   # The -4 is because John sent me a simulation from sol 10 to 10, not 0 to 0

        # Plot parameters
        fig, ax = plt.subplots(ncols=4, nrows=4, width_ratios=[2, 3, 3, 3],
                                layout='constrained', sharex='col', figsize=(15, 6.5))
        gs = ax[1, 2].get_gridspec()
        for axs in ax[:4, 0]:
            axs.remove()
        #biglygridspec = gridspec.
        axbig0 = fig.add_subplot(gs[:2, 0])
        axbig1 = fig.add_subplot(gs[2:, 0])
        axbig1.sharex(axbig0)
        axbig0.tick_params(labelbottom=False)
        for row in range(4):
            ax[row, 2].sharey(ax[row, 1])
            ax[row, 2].tick_params(labelleft=False)
            ax[row, 3].sharey(ax[row, 1])
            ax[row, 3].tick_params(labelleft=False)

        for axis in ax.ravel():
            axis.xaxis.set_major_locator(ticker.MultipleLocator(60))
            axis.yaxis.set_major_locator(ticker.MultipleLocator(30))

        dustvmax = 2
        icevmax = 0.5
        errorvmax = 0.01
        latmin = -60
        latmax = 60
        lonmin = 0
        lonmax = 360
        dustcmap = 'cividis'
        icecmap = 'viridis'
        errorcmap = 'magma'

        dust_radprop = Dust()
        ice_radprop = Ice()

        # Load in radprop
        dust_cext = dust_radprop.get_scattering_cross_sections()
        ice_cext = ice_radprop.get_scattering_cross_sections()

        dust_wavelengths = dust_radprop.get_wavelengths()
        dust_particle_sizes = dust_radprop.get_particle_sizes()
        ice_wavelengths = ice_radprop.get_wavelengths()
        ice_particle_sizes = ice_radprop.get_particle_sizes()

        #######################
        ### Add in the IUVS data in QL form
        #######################

        orbit_code = f'orbit' + f'{orbit}'.zfill(5)
        block = math.floor(orbit / 100) * 100
        orbit_block = 'orbit' + f'{block}'.zfill(5)

        file_path = Path('/media/kyle/iuvs/data/')
        h5file = File(file_path / orbit_block / f'{orbit_code}.hdf5')
        app_flip = h5file['apoapse/instrument_geometry/app_flip'][0]


        base_path = Path(f'/mnt/science/data/mars/maven/iuvs/retrievals/{orbit_block}')  # Path(f'/mnt/science/data/mars/maven/iuvs/retrievals/{orbit_block}')
        files = sorted(base_path.glob(f'*{orbit_code}*'))
        if not files:
            return

        loadeddata = [np.load(f) for f in files]
        retrievals = np.vstack([f for f in loadeddata if f.shape[1] != 33])
        dust = retrievals[..., 0]
        ice = retrievals[..., 1]
        error = retrievals[..., 2]

        #files = sorted(Path(f'/mnt/science/data_lake/mars/maven/iuvs/production/{orbit_block}').glob(f'*apoapse*{orbit_code}*muv*.gz'))
        files = get_apoapse_muv_fits_files(orbit, Path('/mnt/science/data_lake/mars/maven/iuvs/production'))

        lat = np.vstack([f['pixelgeometry'].data['pixel_corner_lat'] for f in files if f['primary'].data.shape[1] != 33])
        lon = np.vstack([f['pixelgeometry'].data['pixel_corner_lon'] for f in files if f['primary'].data.shape[1] != 33])
        alt = np.vstack([f['pixelgeometry'].data['pixel_corner_mrh_alt'][..., 4] for f in files if f['primary'].data.shape[1] != 33])
        fov = np.concatenate([f['integration'].data['fov_deg'] for f in files if f['primary'].data.shape[1] != 33])
        sn = swath_number(fov)


        def make_swath_grid(field_of_view: np.ndarray, sn: int,
                            n_positions: int, n_integrations: int) \
                -> tuple[np.ndarray, np.ndarray]:
            """Make a swath grid of mirror angles and spatial bins.

            Parameters
            ----------
            field_of_view: np.ndarray
                The instrument's field of view.
            swath_number: int
                The swath number.
            n_positions: int
                The number of positions.
            n_integrations: int
                The number of integrations.

            Returns
            -------
            tuple[np.ndarray, np.ndarray]
                The swath grid.

            """
            slit_angles = np.linspace(angular_slit_width * sn,
                                      angular_slit_width * (sn + 1),
                                      num=n_positions + 1)
            mean_angle_difference = np.mean(np.diff(field_of_view))
            field_of_view = np.linspace(field_of_view[0] - mean_angle_difference / 2,
                                        field_of_view[-1] + mean_angle_difference / 2,
                                        num=n_integrations + 1)
            return np.meshgrid(slit_angles, field_of_view)


        for swath in np.unique(sn):
            # Do this no matter if I'm plotting primary or angles
            swath_inds = sn == swath
            n_integrations = np.sum(swath_inds)
            n_positions = dust.shape[1]
            x, y = make_swath_grid(fov[swath_inds], swath, n_positions, n_integrations)
            # APP flip
            # dax = ax[0, 0].pcolormesh(x, y, np.fliplr(dust[swath_inds]), linewidth=0, edgecolors='none', rasterized=True, vmin=0, vmax=dustvmax, cmap='cividis')
            # iax = ax[1, 0].pcolormesh(x, y, np.fliplr(ice[swath_inds]), linewidth=0, edgecolors='none', rasterized=True, vmin=0, vmax=icevmax, cmap='viridis')
            # eax = ax[2, 0].pcolormesh(x, y, np.fliplr(error[swath_inds]), linewidth=0, edgecolors='none', rasterized=True, vmin=0, vmax=errorvmax, cmap='magma')

            # No APP flip
            dusttoplot = dust[swath_inds] if not app_flip else np.fliplr(dust[swath_inds])
            icetoplot = ice[swath_inds] if not app_flip else np.fliplr(ice[swath_inds])
            axbig0.pcolormesh(x, y, dusttoplot, linewidth=0, edgecolors='none', rasterized=True, vmin=0, vmax=dustvmax, cmap=dustcmap)
            axbig1.pcolormesh(x, y, icetoplot, linewidth=0, edgecolors='none', rasterized=True, vmin=0, vmax=icevmax, cmap=icecmap)

        axbig0.set_title('Dust optical depth')
        axbig0.set_xlim(0, angular_slit_width * (sn[-1] + 1))
        axbig0.set_ylim(minimum_mirror_angle * 2, maximum_mirror_angle * 2)
        axbig0.set_xticks([])
        axbig0.set_yticks([])
        axbig0.set_facecolor('gray')
        axbig0.set_aspect('equal')

        axbig1.set_title('Ice optical depth')
        axbig1.set_xlim(0, angular_slit_width * (sn[-1] + 1))
        axbig1.set_ylim(minimum_mirror_angle * 2, maximum_mirror_angle * 2)
        axbig1.set_xticks([])
        axbig1.set_yticks([])
        axbig1.set_facecolor('gray')
        axbig1.set_aspect('equal')

        #######################
        ### Add in the IUVS data in cylindrical map form
        #######################

        iuvs_ames_dust = np.load(f'/mnt/science/data/mars/maven/iuvs/retrievals/gcm_grid/ames/{orbit_block}/{orbit_code}-dust.npy')
        iuvs_ames_ice = np.load(f'/mnt/science/data/mars/maven/iuvs/retrievals/gcm_grid/ames/{orbit_block}/{orbit_code}-ice.npy')
        iuvs_pcm_dust = np.load(f'/mnt/science/data/mars/maven/iuvs/retrievals/gcm_grid/pcm/{orbit_block}/{orbit_code}-dust.npy')
        iuvs_pcm_ice = np.load(f'/mnt/science/data/mars/maven/iuvs/retrievals/gcm_grid/pcm/{orbit_block}/{orbit_code}-ice.npy')

        #iuvs_pcm_dust = np.roll(iuvs_pcm_dust, 33, axis=1)
        #iuvs_pcm_ice = np.roll(iuvs_pcm_ice, 33, axis=1)

        ames_X = np.arange(181) * 2
        ames_Y = np.arange(91) * 2 - 90

        lon_centers = np.linspace(-180, 180, num=65)
        lon_boundaries = (lon_centers[1:] + lon_centers[:-1]) / 2
        lon_boundaries = np.where(lon_boundaries < 0, lon_boundaries + 360, lon_boundaries)
        lon_boundaries = np.roll(lon_boundaries, 32)
        pcm_X = np.concatenate(([0], lon_boundaries, [360]))

        pcm_Y = np.linspace(90+3.75/2, -90-3.75/2, num=50)

        ax01 = ax[0, 1].pcolormesh(ames_X, ames_Y, iuvs_ames_dust, vmin=0, vmax=dustvmax, cmap='cividis')
        ax11 = ax[1, 1].pcolormesh(pcm_X, pcm_Y, iuvs_pcm_dust, vmin=0, vmax=dustvmax, cmap='cividis')
        ax21 = ax[2, 1].pcolormesh(ames_X, ames_Y, iuvs_ames_ice, vmin=0, vmax=icevmax, cmap='viridis')
        ax31 = ax[3, 1].pcolormesh(pcm_X, pcm_Y, iuvs_pcm_ice, vmin=0, vmax=icevmax, cmap='viridis')
        ax[0, 1].set_aspect('equal')
        ax[1, 1].set_aspect('equal')
        ax[2, 1].set_aspect('equal')
        ax[3, 1].set_aspect('equal')

        for foobar in [0, 1]:
            ax[foobar, 1].set_title('IUVS Dust OD')
            ax[foobar, 1].set_xlim(lonmin, lonmax)
            ax[foobar, 1].set_ylim(latmin, latmax)
            ax[foobar, 1].set_facecolor('gray')

        for foobar in [2, 3]:
            ax[foobar, 1].set_title('IUVS Ice OD')
            ax[foobar, 1].set_xlim(lonmin, lonmax)
            ax[foobar, 1].set_ylim(latmin, latmax)
            ax[foobar, 1].set_facecolor('gray')

        #######################
        ### Add in the Ames GCM dust/ice climatology
        #######################
        gcm_path = Path('/mnt/science/data_lake/mars/gcm/ames/my30') / 'c48L36_my30.atmos_diurn.nc'
        yearly_gcm = Dataset(gcm_path)

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

        gcm_dust_ax = ax[0, 2].pcolormesh(ames_X, ames_Y, gcm_dust * dust_scaling, vmin=0, vmax=dustvmax, cmap=dustcmap)
        plt.colorbar(gcm_dust_ax, ax=ax[0, 2], **cbar_props)
        gcm_ice_ax = ax[2, 2].pcolormesh(ames_X, ames_Y, gcm_ice * ice_scaling, vmin=0, vmax=icevmax, cmap=icecmap)
        plt.colorbar(gcm_ice_ax, ax=ax[2, 2], **cbar_props)
        ax[0, 2].set_aspect('equal')
        ax[2, 2].set_aspect('equal')

        ax[0, 2].set_title('Ames GCM Dust OD')
        ax[0, 2].set_xlim(lonmin, lonmax)
        ax[0, 2].set_ylim(latmin, latmax)

        ax[2, 2].set_title('Ames GCM Ice OD')
        ax[2, 2].set_xlim(lonmin, lonmax)
        ax[2, 2].set_ylim(latmin, latmax)

        '''axins = inset_axes(ax[0, 2], width='5%', height='100%', loc="lower left", bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax[0, 2].transAxes, borderpad=0)
        fig.colorbar(gcm_dust_ax, cax=axins)
        axins = inset_axes(ax[2, 2], width='5%', height='100%', loc="lower left", bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax[2, 2].transAxes, borderpad=0)
        fig.colorbar(gcm_ice_ax, cax=axins)'''

        #######################
        ### Add in the PCM dust/ice climatology
        #######################
        month = str(math.floor(ls/360 * 12) + 1).zfill(2)   # this will be an int. The +1 is because they use 1 based indexing
        monthly_sol = math.floor((mt.MarsTime(mars_year, sol) - mt.MarsTime.from_solar_longitude(mars_year, int(month)*30)).sols)
        pcm_path = Path(f'/mnt/science/data_lake/mars/gcm/pcm/my{mars_year}') / f'diagfi{month}_MY{mars_year}_OPAext.nc'
        pcm = Dataset(pcm_path)

        pcm_time = pcm.variables['Time'][:]
        nearest_sol_idx = np.argmin(np.abs(monthly_sol - pcm_time)) + ames_lt_idx   # cause both of them use 1 hour time steps

        pcm_dust = np.roll(pcm.variables['tau_dust'][nearest_sol_idx], 33, axis=1)
        pcm_ice = np.roll(pcm.variables['tau_h2o_ice'][nearest_sol_idx], 33, axis=1)

        pcm_dust_ax = ax[1, 2].pcolormesh(pcm_X, pcm_Y, pcm_dust, vmin=0, vmax=dustvmax, cmap=dustcmap)
        plt.colorbar(pcm_dust_ax, ax=ax[1, 2], **cbar_props)
        pcm_ice_ax = ax[3, 2].pcolormesh(pcm_X, pcm_Y, pcm_ice, vmin=0, vmax=icevmax, cmap=icecmap)
        plt.colorbar(pcm_ice_ax, ax=ax[3, 2], **cbar_props)
        ax[1, 2].set_aspect('equal')
        ax[3, 2].set_aspect('equal')

        ax[1, 2].set_title('PCM Dust OD')
        ax[1, 2].set_xlim(lonmin, lonmax)
        ax[1, 2].set_ylim(latmin, latmax)

        ax[3, 2].set_title('PCM Ice OD')
        ax[3, 2].set_xlim(lonmin, lonmax)
        ax[3, 2].set_ylim(latmin, latmax)

        '''axins = inset_axes(ax[1, 2], width='5%', height='100%', loc="lower left", bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax[1, 2].transAxes, borderpad=0)
        fig.colorbar(pcm_dust_ax, cax=axins)
        axins = inset_axes(ax[3, 2], width='5%', height='100%', loc="lower left", bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax[3, 2].transAxes, borderpad=0)
        fig.colorbar(pcm_ice_ax, cax=axins)'''

        #######################
        ### Add in the Ames comparison
        #######################
        dust_comp = ax[0, 3].pcolormesh(ames_X, ames_Y, iuvs_ames_dust - gcm_dust * dust_scaling, vmin=-0.25, vmax=0.25, cmap='PRGn')
        plt.colorbar(dust_comp, ax=ax[0, 3], **cbar_props)
        ice_comp = ax[2, 3].pcolormesh(ames_X, ames_Y, iuvs_ames_ice - gcm_ice * ice_scaling, vmin=-0.25, vmax=0.25, cmap='PRGn')
        plt.colorbar(ice_comp, ax=ax[2, 3], **cbar_props)
        ax[0, 3].set_aspect('equal')
        ax[2, 3].set_aspect('equal')

        ax[0, 3].set_title('Dust retrieval - model')
        ax[0, 3].set_xlim(lonmin, lonmax)
        ax[0, 3].set_ylim(latmin, latmax)

        ax[2, 3].set_title('Ice retrieval - model')
        ax[2, 3].set_xlim(lonmin, lonmax)
        ax[2, 3].set_ylim(latmin, latmax)

        #######################
        ### Add in the PCM comparison
        #######################
        pcm_dust_comp = ax[1, 3].pcolormesh(pcm_X, pcm_Y, iuvs_pcm_dust - pcm_dust, vmin=-0.25, vmax=0.25, cmap='PRGn')
        plt.colorbar(pcm_dust_comp, ax=ax[1, 3], **cbar_props)
        pcm_ice_comp = ax[3, 3].pcolormesh(pcm_X, pcm_Y, iuvs_pcm_ice - pcm_ice, vmin=-0.25, vmax=0.25, cmap='PRGn')
        plt.colorbar(pcm_ice_comp, ax=ax[3, 3], **cbar_props)
        ax[1, 3].set_aspect('equal')
        ax[3, 3].set_aspect('equal')

        ax[1, 3].set_title('Dust retrieval - model')
        ax[1, 3].set_xlim(lonmin, lonmax)
        ax[1, 3].set_ylim(latmin, latmax)

        ax[3, 3].set_title('Ice retrieval - model')
        ax[3, 3].set_xlim(lonmin, lonmax)
        ax[3, 3].set_ylim(latmin, latmax)

        for foobar in range(4):
            ax[foobar, 3].set_facecolor('gray')

        #######################
        ### Add in the topographic contours
        #######################
        for foobar in range(1, 4):
            for foobaz in range(0, 4):
                ax[foobaz, foobar].contour(mola_lon, mola_lat, mola, [0, 5, 10, 15, 20], colors=[cmap(0), cmap(1/4), cmap(2/4), cmap(3/4), cmap(1)], linewidths=[0.5], linestyles=['solid'])
                ax[foobaz, foobar].contour(mola_lon, mola_lat, mola, [0, 5, 10, 15, 20], colors=[cmap(0), cmap(1 / 4), cmap(2 / 4), cmap(3 / 4), cmap(1)], linewidths=[0.5], linestyles=['solid'])

        plt.suptitle(f'Orbit {orbit}, MY={mars_year}, Ls={ls:.2f}, sol={sol:.2f}, apsis LT={lt:.1f}')
        my_str = f'MY{mars_year}'.zfill(2)
        ls_str = f'{ls:.0f}'.zfill(3)
        #filename = Path(f'/mnt/science/data/mars/maven/iuvs/retrievals/images/{orbit_block}/{orbit_code}.png')
        filename = Path(f'/mnt/science/data/mars/maven/iuvs/retrievals/images/{my_str}/{my_str}-Ls{ls_str}-{orbit_code}.png')
        filename.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename, dpi=200)
        plt.close(fig)
    except:
        try:
            plt.close(fig)
        except:
            return
        return


if __name__ == '__main__':
    n_cpus = mp.cpu_count()
    pool = mp.Pool(n_cpus - 5)
    for orb in range(3400, 3401):
        #pool.apply_async(func=process_image, args=(orb,))
        process_image(orb)

    #pool.close()
    #pool.join()
