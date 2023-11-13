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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

cmap = plt.get_cmap('copper')

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


mola = np.load('/mnt/science/data_lake/mars/maps/mola-topography.npy') / 1000   # the base unit of the array is meters, but dividing by 1000 changes it to km
mola_lat = np.linspace(90, -90, num=1440)
mola_lon = np.linspace(0, 360, num=2880)



def make_image(orbit: int):
    apsis_file = File('/mnt/science/data_lake/mars/maven/apsis.hdf5')
    all_orbits = apsis_file['apoapse/orbit'][:]
    sol = apsis_file['apoapse/sol'][all_orbits==orbit][0]
    lt = apsis_file['apoapse/subspacecraft_local_time'][all_orbits==orbit][0]
    ls = apsis_file['apoapse/solar_longitude'][all_orbits==orbit][0]
    subsclon = apsis_file['apoapse/subspacecraft_longitude'][all_orbits == orbit]
    mars_year = apsis_file['apoapse/mars_year'][all_orbits==orbit][0]

    ames_lt_idx = np.argmin(np.abs(np.arange(24)+0.5 - (lt - (subsclon / 360 * 24)) % 24))
    ames_sol_idx = int(sol / 668 * 133) - 4   # The -4 is because John sent me a simulation from sol 10 to 10, not 0 to 0

    # Plot parameters
    fig, ax = plt.subplots(2, 4, figsize=(15, 8))
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
    #files = sorted(list(set(sorted(base_path.glob(f'{orbit_code}*.npy'))) - set(sorted(base_path.glob(f'{orbit_code}*FF*.npy')))))
    files = sorted(base_path.glob(f'*{orbit_code}*'))
    if not files:
        return

    loadeddata = [np.load(f) for f in files]
    retrievals = np.vstack([f for f in loadeddata if f.shape[1] != 33])
    dust = retrievals[..., 0]
    ice = retrievals[..., 1]
    error = retrievals[..., 2]
    # dust_files = sorted(base_path.glob(f'{orbit_code}-*-dust.npy'))
    # ice_files = sorted(base_path.glob(f'{orbit_code}-*-ice.npy'))
    # error_files = sorted(base_path.glob(f'{orbit_code}-*-error.npy'))
    # dust = np.vstack([np.load(f) for f in dust_files])
    # ice = np.vstack([np.load(f) for f in ice_files])
    # error = np.vstack([np.load(f) for f in error_files])

    files = sorted(Path(f'/mnt/science/data_lake/mars/maven/iuvs/production/{orbit_block}').glob(f'*apoapse*{orbit_code}*muv*.gz'))
    files = [fits.open(f) for f in files]

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
        dax = ax[0, 0].pcolormesh(x, y, dust[swath_inds], linewidth=0, edgecolors='none', rasterized=True, vmin=0, vmax=dustvmax, cmap='cividis')
        iax = ax[1, 0].pcolormesh(x, y, ice[swath_inds], linewidth=0, edgecolors='none', rasterized=True, vmin=0, vmax=icevmax, cmap='viridis')
        # eax = ax[2, 0].pcolormesh(x, y, error[swath_inds], linewidth=0, edgecolors='none', rasterized=True, vmin=0, vmax=errorvmax, cmap='magma')

    ax[0, 0].set_title('Dust optical depth')
    ax[0, 0].set_xlim(0, angular_slit_width * (sn[-1] + 1))
    ax[0, 0].set_ylim(minimum_mirror_angle * 2, maximum_mirror_angle * 2)
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_facecolor('gray')

    ax[1, 0].set_title('Ice optical depth')
    ax[1, 0].set_xlim(0, angular_slit_width * (sn[-1] + 1))
    ax[1, 0].set_ylim(minimum_mirror_angle * 2, maximum_mirror_angle * 2)
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax[1, 0].set_facecolor('gray')

    '''ax[2, 0].set_title('Residual')
    ax[2, 0].set_xlim(0, angular_slit_width * (swath_number[-1] + 1))
    ax[2, 0].set_ylim(minimum_mirror_angle * 2, maximum_mirror_angle * 2)
    ax[2, 0].set_xticks([])
    ax[2, 0].set_yticks([])
    ax[2, 0].set_facecolor('gray')'''


    #######################
    ### Add in the IUVS data in cylindrical map form
    #######################

    # This will regrid the IUVS data onto the GCM 2x2 resolution
    dust_grid = np.zeros((90, 180)) * np.nan
    ice_grid = np.zeros((90, 180)) * np.nan

    for lat_idx, latitude in enumerate(np.arange(90) * 2 - 90):
        for lon_idx, longitude in enumerate(np.arange(180) * 2):
            print(lat_idx, lon_idx)
            lat_mask = np.logical_and(latitude < lat[..., 4], lat[..., 4] < latitude + 2)
            lon_mask = np.logical_and(longitude < lon[..., 4], lon[..., 4] < longitude + 2)
            mask = np.logical_and(lat_mask, lon_mask)
            dust_grid[lat_idx, lon_idx] = np.nanmean(dust[mask])
            ice_grid[lat_idx, lon_idx] = np.nanmean(ice[mask])

    X = np.arange(181) * 2
    Y = np.arange(91) * 2 - 90

    ax01 = ax[0, 1].pcolormesh(X, Y, dust_grid, vmin=0, vmax=dustvmax, cmap='cividis')   # This was named cdax
    ax11 = ax[1, 1].pcolormesh(X, Y, ice_grid, vmin=0, vmax=icevmax, cmap='viridis')     # This was named ciax

    ax01.axes.set_aspect(3)
    ax11.axes.set_aspect(3)

    ax[0, 1].set_title('IUVS Dust (cylindrical)')
    ax[0, 1].set_xlim(lonmin, lonmax)
    ax[0, 1].set_ylim(latmin, latmax)
    ax[0, 1].set_facecolor('gray')

    ax[1, 1].set_title('IUVS Ice (cylindrical)')
    ax[1, 1].set_xlim(lonmin, lonmax)
    ax[1, 1].set_ylim(latmin, latmax)
    ax[1, 1].set_facecolor('gray')
    '''
    def latlon_meshgrid(latitude, longitude, altitude):
        # make meshgrids to hold latitude and longitude grids for pcolormesh display
        X = np.zeros((latitude.shape[0] + 1, latitude.shape[1] + 1))
        Y = np.zeros((longitude.shape[0] + 1, longitude.shape[1] + 1))
        mask = np.ones((latitude.shape[0], latitude.shape[1]))

        # loop through pixel geometry arrays
        for i in range(int(latitude.shape[0])):
            for j in range(int(latitude.shape[1])):

                # there are some pixels where some of the pixel corner longitudes are undefined
                # if we encounter one of those, set the data value to missing so it isn't displayed
                # with pcolormesh
                if np.size(np.where(np.isfinite(longitude[i, j]))) != 5:
                    mask[i, j] = np.nan

                # also mask out non-disk pixels
                if altitude[i, j] != 0:
                    mask[i, j] = np.nan

                # place the longitude and latitude values in the meshgrids
                X[i, j] = longitude[i, j, 1]
                X[i + 1, j] = longitude[i, j, 0]
                X[i, j + 1] = longitude[i, j, 3]
                X[i + 1, j + 1] = longitude[i, j, 2]
                Y[i, j] = latitude[i, j, 1]
                Y[i + 1, j] = latitude[i, j, 0]
                Y[i, j + 1] = latitude[i, j, 3]
                Y[i + 1, j + 1] = latitude[i, j, 2]

        # set any of the NaN values to zero (otherwise pcolormesh will break even if it isn't displaying the pixel).
        X[np.where(~np.isfinite(X))] = 0
        Y[np.where(~np.isfinite(Y))] = 0

        # set to domain [-180,180)
        # X[np.where(X > 180)] -= 360

        # return the coordinate arrays and the mask
        return X, Y


    for swath in np.unique(sn):
        # Plot the data from [-180, 180)
        # lon -= 360
        x, y = latlon_meshgrid(lat[swath == sn], lon[swath == sn], alt[swath == sn])
        cdax = ax[0, 1].pcolormesh(x, y, dust[swath == sn], vmin=0, vmax=dustvmax, cmap='cividis')
        ciax = ax[1, 1].pcolormesh(x, y, ice[swath == sn], vmin=0, vmax=icevmax, cmap='viridis')
        # ceax = ax[2, 1].pcolormesh(x, y, error[swath == swath_number], vmin=0, vmax=errorvmax, cmap='magma')
        '''

    '''divider = make_axes_locatable(ax[0, 1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(cdax, cax=cax, orientation='vertical')

    divider = make_axes_locatable(ax[1, 1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(ciax, cax=cax, orientation='vertical')

    divider = make_axes_locatable(ax[2, 1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(ceax, cax=cax, orientation='vertical')'''

    '''ax[2, 1].set_title('IUVS Residual')
    ax[2, 1].set_xlim(lonmin, lonmax)
    ax[2, 1].set_ylim(latmin, latmax)
    ax[2, 1].set_facecolor('gray')'''

    #######################
    ### Add in the Montabone assimilated dust
    #######################
    # UV / VIS
    '''foo = dust_cext[-8]
    dustvis = np.mean(foo[(0.4 <= dust_wavelengths) & (dust_wavelengths <= 0.8)])
    dustuv = np.mean(foo[(0.2 <= dust_wavelengths) & (dust_wavelengths <= 0.3)])
    dust_scaling = dustvis / dustuv

    assimilated_dust_dataset = Dataset('/home/kyle/iuvs/dustscenario_MY33_v2-1.nc')
    assimilated_dust = assimilated_dust_dataset['cdodtot'][:]
    #dust_lat = assimilated_dust_dataset['latitude'][:]
    #dust_lon = np.roll(assimilated_dust_dataset['longitude'][:], 60)
    #dust_lon = np.where(dust_lon < 0, dust_lon+360, dust_lon)
    assimilated_dust = np.roll(assimilated_dust, 60, axis=-1)

    # Get the lat/lon midpoints and tack on the ends
    #dust_lat = (dust_lat[:-1] + dust_lat[1:]) / 2
    #dust_lat = np.concatenate(([90], dust_lat, [-90]))
    dust_lat = np.linspace(90, -90, num=61)
    dust_lon = np.linspace(0, 360, num=121)

    assim_dust_ax = ax[0, 2].pcolormesh(dust_lon, dust_lat, assimilated_dust[int(sol), :, :] * 2.6 * dust_scaling, vmin=0, vmax=dustvmax, cmap=dustcmap)

    ax[0, 2].set_title('Assimilated Dust')
    ax[0, 2].set_xlim(lonmin, lonmax)
    ax[0, 2].set_ylim(latmin, latmax)
    ax[0, 2].set_facecolor('gray')'''

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

    gcm_dust_ax = ax[0, 2].pcolormesh(X, Y, gcm_dust * dust_scaling, vmin=0, vmax=dustvmax, cmap=dustcmap)
    gcm_ice_ax = ax[1, 2].pcolormesh(X, Y, gcm_ice * ice_scaling, vmin=0, vmax=icevmax, cmap=icecmap)

    gcm_dust_ax.axes.set_aspect(3)
    gcm_ice_ax.axes.set_aspect(3)

    '''divider = make_axes_locatable(ax[0, 3])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(gcm_dust_ax, cax=cax, orientation='vertical')

    divider = make_axes_locatable(ax[1, 3])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(gcm_ice_ax, cax=cax, orientation='vertical')'''

    ax[0, 2].set_title('Ames GCM Dust')
    ax[0, 2].set_xlim(lonmin, lonmax)
    ax[0, 2].set_ylim(latmin, latmax)

    ax[1, 2].set_title('Ames GCM Ice')
    ax[1, 2].set_xlim(lonmin, lonmax)
    ax[1, 2].set_ylim(latmin, latmax)

    '''divider = make_axes_locatable(ax[0, 2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(gcm_dust_ax, cax=cax, orientation='vertical')
    
    divider = make_axes_locatable(ax[1, 2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(gcm_ice_ax, cax=cax, orientation='vertical')'''
    axins = inset_axes(ax[0, 2], width='5%', height='100%', loc="lower left", bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax[0, 2].transAxes, borderpad=0)
    fig.colorbar(gcm_dust_ax, cax=axins)
    axins = inset_axes(ax[1, 2], width='5%', height='100%', loc="lower left", bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax[1, 2].transAxes, borderpad=0)
    fig.colorbar(gcm_ice_ax, cax=axins)


    #######################
    ### Add in the comparison graphic
    #######################
    dust_comp = ax[0, 3].pcolormesh(X, Y, dust_grid - gcm_dust * dust_scaling, vmin=-0.25, vmax=0.25, cmap='PRGn')
    ice_comp = ax[1, 3].pcolormesh(X, Y, ice_grid - gcm_ice * ice_scaling, vmin=-0.25, vmax=0.25, cmap='PRGn')

    dust_comp.axes.set_aspect(3)
    ice_comp.axes.set_aspect(3)

    ax[0, 3].set_title('Dust retrieval - model')
    ax[0, 3].set_xlim(lonmin, lonmax)
    ax[0, 3].set_ylim(latmin, latmax)

    ax[1, 3].set_title('Ice retrieval - model')
    ax[1, 3].set_xlim(lonmin, lonmax)
    ax[1, 3].set_ylim(latmin, latmax)

    # Add in topographic contours
    for foobar in range(1, 4):
        ax[0, foobar].contour(mola_lon, mola_lat, mola, [0, 5, 10, 15, 20], colors=[cmap(0), cmap(1/4), cmap(2/4), cmap(3/4), cmap(1)], linewidths=[0.5], linestyles=['solid'])
        ax[1, foobar].contour(mola_lon, mola_lat, mola, [0, 5, 10, 15, 20], colors=[cmap(0), cmap(1 / 4), cmap(2 / 4), cmap(3 / 4), cmap(1)], linewidths=[0.5], linestyles=['solid'])

    '''divider = make_axes_locatable(ax[0, 3])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(dust_comp, cax=cax, orientation='vertical')

    divider = make_axes_locatable(ax[1, 3])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(ice_comp, cax=cax, orientation='vertical')'''

    # Resource: https://matplotlib.org/stable/gallery/axes_grid1/demo_colorbar_with_inset_locator.html
    axins = inset_axes(ax[0, 3], width='5%', height='100%', loc="lower left", bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax[0, 3].transAxes, borderpad=0)
    fig.colorbar(dust_comp, cax=axins)
    axins = inset_axes(ax[1, 3], width='5%', height='100%', loc="lower left", bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax[1, 3].transAxes, borderpad=0)
    fig.colorbar(ice_comp, cax=axins)


    #######################
    ### Add in the PCM dust/ice climatology
    #######################
    '''yearly_gcm = Dataset('/media/kyle/McDataFace/pcm/run_MY33/diagfi7.nc')

    pcm_sol_idx = np.argmin(np.abs(sol - 372 - yearly_gcm['Time'][:]))

    gcm_ice = np.roll(yearly_gcm['h2o_ice'][:], 32, axis=-1)
    gcm_lat = np.broadcast_to(np.linspace(90, -90, num=50), (66, 50))
    gcm_lon = np.broadcast_to(np.linspace(0, 360, num=66), (50, 66)).T
    #gcm_dust_ax = ax[0, 4].pcolormesh(gcm_lon, gcm_lat, gcm_dust[int(sol/668*140), :, :].T * dust_scaling, vmin=0, vmax=dustvmax, cmap=dustcmap)
    ax[1, 4].pcolormesh(gcm_lon, gcm_lat, np.sum(gcm_ice[pcm_sol_idx, :, :, :], axis=0).T, vmin=0, vmax=0.001, cmap=icecmap)

    divider = make_axes_locatable(ax[0, 4])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(gcm_dust_ax, cax=cax, orientation='vertical')

    divider = make_axes_locatable(ax[1, 4])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(gcm_ice_ax, cax=cax, orientation='vertical')

    ax[0, 4].set_title('PCM Dust')
    ax[0, 4].set_xlim(lonmin, lonmax)
    ax[0, 4].set_ylim(latmin, latmax)

    ax[1, 4].set_title('PCM Ice column')
    ax[1, 4].set_xlim(lonmin, lonmax)
    ax[1, 4].set_ylim(latmin, latmax)

    # Remove unused plot ticks
    ax[2, 2].set_xticks([])
    ax[2, 2].set_yticks([])

    ax[2, 3].set_xticks([])
    ax[2, 3].set_yticks([])

    ax[2, 4].set_xticks([])
    ax[2, 4].set_yticks([])'''


    plt.suptitle(f'Orbit {orbit}, MY={mars_year}, Ls={ls:.2f}, sol={sol:.2f}, apsis LT={lt:.1f}')
    plt.subplots_adjust(wspace=0.5)
    filename = Path(f'/mnt/science/data/mars/maven/iuvs/retrievals/images/{orbit_block}/{orbit_code}.png')
    filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, dpi=200)
    plt.close(fig)

