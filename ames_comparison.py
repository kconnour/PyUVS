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


orbit = 3453


apsis_file = File('/mnt/science/data_lake/mars/maven/apsis.hdf5')
sol = apsis_file['apoapse/sol'][orbit]
mars_year = apsis_file['apoapse/mars_year'][orbit]


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


# Plot parameters
fig, ax = plt.subplots(2, 3, figsize=(15, 9))
dustvmax = 2
icevmax = 0.6
errorvmax = 0.1
latmin = -45
latmax = 45
lonmin = 180#+45
lonmax = 270#+45
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

base_path = Path(f'/mnt/science/data/mars/maven/iuvs/retrievals/{orbit_block}')
dust_files = sorted(base_path.glob(f'{orbit_code}-*-dust.npy'))
ice_files = sorted(base_path.glob(f'{orbit_code}-*-ice.npy'))
error_files = sorted(base_path.glob(f'{orbit_code}-*-error.npy'))
dust = np.vstack([np.load(f) for f in dust_files])
ice = np.vstack([np.load(f) for f in ice_files])
error = np.vstack([np.load(f) for f in error_files])

files = sorted(Path(f'/mnt/science/data_lake/mars/maven/iuvs/production/{orbit_block}').glob(f'*apoapse*{orbit_code}*muv*.gz'))
files = [fits.open(f) for f in files]

lat = np.vstack([f['pixelgeometry'].data['pixel_corner_lat'] for f in files])
lon = np.vstack([f['pixelgeometry'].data['pixel_corner_lon'] for f in files])
alt = np.vstack([f['pixelgeometry'].data['pixel_corner_mrh_alt'][..., 4] for f in files])
fov = np.concatenate([f['integration'].data['fov_deg'] for f in files])
swath_number = swath_number(fov)


def make_swath_grid(field_of_view: np.ndarray, swath_number: int,
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
    slit_angles = np.linspace(angular_slit_width * swath_number,
                              angular_slit_width * (swath_number + 1),
                              num=n_positions+1)
    mean_angle_difference = np.mean(np.diff(field_of_view))
    field_of_view = np.linspace(field_of_view[0] - mean_angle_difference / 2,
                                field_of_view[-1] + mean_angle_difference / 2,
                                num=n_integrations + 1)
    return np.meshgrid(slit_angles, field_of_view)


for swath in np.unique(swath_number):
    # Do this no matter if I'm plotting primary or angles
    swath_inds = swath_number == swath
    n_integrations = np.sum(swath_inds)
    x, y = make_swath_grid(fov[swath_inds], swath, 133, n_integrations)
    # APP flip
    #dax = ax[0, 0].pcolormesh(x, y, np.fliplr(dust[swath_inds]), linewidth=0, edgecolors='none', rasterized=True, vmin=0, vmax=dustvmax, cmap='cividis')
    #iax = ax[1, 0].pcolormesh(x, y, np.fliplr(ice[swath_inds]), linewidth=0, edgecolors='none', rasterized=True, vmin=0, vmax=icevmax, cmap='viridis')
    #eax = ax[2, 0].pcolormesh(x, y, np.fliplr(error[swath_inds]), linewidth=0, edgecolors='none', rasterized=True, vmin=0, vmax=errorvmax, cmap='magma')

    # No APP flip
    dax = ax[0, 0].pcolormesh(x, y, dust[swath_inds], linewidth=0, edgecolors='none', rasterized=True,vmin=0, vmax=dustvmax, cmap='cividis')
    iax = ax[1, 0].pcolormesh(x, y, ice[swath_inds], linewidth=0, edgecolors='none', rasterized=True, vmin=0, vmax=icevmax, cmap='viridis')
    #eax = ax[2, 0].pcolormesh(x, y, error[swath_inds], linewidth=0, edgecolors='none', rasterized=True, vmin=0, vmax=errorvmax, cmap='magma')

ax[0, 0].set_title('Dust optical depth')
ax[0, 0].set_xlim(0, angular_slit_width * (swath_number[-1] + 1))
ax[0, 0].set_ylim(minimum_mirror_angle * 2, maximum_mirror_angle * 2)
ax[0, 0].set_xticks([])
ax[0, 0].set_yticks([])
ax[0, 0].set_facecolor('gray')

ax[1, 0].set_title('Ice optical depth')
ax[1, 0].set_xlim(0, angular_slit_width * (swath_number[-1] + 1))
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
    #X[np.where(X > 180)] -= 360

    # return the coordinate arrays and the mask
    return X, Y


for swath in np.unique(swath_number):
    x, y = latlon_meshgrid(lat[swath==swath_number], lon[swath==swath_number], alt[swath==swath_number])
    cdax = ax[0, 1].pcolormesh(x, y, dust[swath==swath_number], vmin=0, vmax=dustvmax, cmap='cividis')
    ciax = ax[1, 1].pcolormesh(x, y, ice[swath==swath_number], vmin=0, vmax=icevmax, cmap='viridis')
    #ceax = ax[2, 1].pcolormesh(x, y, error[swath == swath_number], vmin=0, vmax=errorvmax, cmap='magma')

'''divider = make_axes_locatable(ax[0, 1])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(cdax, cax=cax, orientation='vertical')

divider = make_axes_locatable(ax[1, 1])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(ciax, cax=cax, orientation='vertical')

divider = make_axes_locatable(ax[2, 1])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(ceax, cax=cax, orientation='vertical')'''
divider = make_axes_locatable(ax[0, 1])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(cdax, cax=cax, orientation='vertical')

divider = make_axes_locatable(ax[1, 1])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(ciax, cax=cax, orientation='vertical')

ax[0, 1].set_title('IUVS Dust')
ax[0, 1].set_xlim(lonmin, lonmax)
ax[0, 1].set_ylim(latmin, latmax)
ax[0, 1].set_facecolor('gray')

ax[1, 1].set_title('IUVS Ice')
ax[1, 1].set_xlim(lonmin, lonmax)
ax[1, 1].set_ylim(latmin, latmax)
ax[1, 1].set_facecolor('gray')

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
gcm_path = Path('/mnt/science/data_lake/mars/gcm/ames/my_generic') / '05344.atmos_average.nc'
yearly_gcm = Dataset(gcm_path)

foo = dust_cext[np.argmin(np.abs(dust_particle_sizes - 1.5))]
dustvis = np.mean(foo[(0.4 <= dust_wavelengths) & (dust_wavelengths <= 0.8)])
dustuv = np.mean(foo[(0.2 <= dust_wavelengths) & (dust_wavelengths <= 0.3)])
dust_scaling = dustvis / dustuv

foo = ice_cext[np.argmin(np.abs(ice_particle_sizes - 5))]
icevis = np.mean(foo[(0.4 <= ice_wavelengths) & (ice_wavelengths <= 0.8)])
iceuv = np.mean(foo[(0.2 <= ice_wavelengths) & (ice_wavelengths <= 0.3)])
ice_scaling = icevis / iceuv

gcm_dust = yearly_gcm['taudust_VIS'][:]
#gcm_ice = yearly_gcm['taucloud_VIS'][:]
gcm_lat = np.broadcast_to(np.linspace(-90, 90, num=181), (361, 181))
gcm_lon = np.broadcast_to(np.linspace(0, 360, num=361), (181, 361)).T
gcm_dust_ax = ax[0, 2].pcolormesh(gcm_lon, gcm_lat, gcm_dust[int(sol/668*140), :, :].T * dust_scaling, vmin=0, vmax=dustvmax, cmap=dustcmap)
#gcm_ice_ax = ax[1, 1].pcolormesh(gcm_lon, gcm_lat, gcm_ice[int(sol/668*140), :, :].T * ice_scaling, vmin=0, vmax=icevmax, cmap=icecmap)

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
plt.suptitle(f'Orbit {orbit}, MY={mars_year}, sol={sol:.2f}')
fig.tight_layout()
plt.savefig(f'/home/kyle/iuvs/retrievals/{orbit_block}/images/{orbit_code}-update.png', dpi=200)
