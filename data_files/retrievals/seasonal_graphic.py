'''
For this, I need to read in the GCM gridded data, the subspacecraft longitude, and LT.
Then, I need to make a LT map corresponding to each point in the retrieval grid.

Finally, I need to make a (lat, lon, LT, orbit) array that I can populate
with the corresponding retrieval maps. I'll loop over LT and take a nanmean over
longitude to make the graphics that I want.


'''
import math
from h5py import File
import numpy as np
import matplotlib.pyplot as plt

# Read in the SPICE data
apsis_file = File('/mnt/science/data_lake/mars/maven/apsis.hdf5')
all_orbits = apsis_file['apoapse/orbit'][:]
sol = apsis_file['apoapse/sol'][:]
lt = apsis_file['apoapse/subspacecraft_local_time'][:]
ls = apsis_file['apoapse/solar_longitude'][:]
subsclon = apsis_file['apoapse/subspacecraft_longitude'][:]
mars_year = apsis_file['apoapse/mars_year'][:]


def make_ames_seasonal_grid(orb_start, orb_end):
    shape = (90, 180)
    dust_seasonal_grid = np.zeros(shape + (12,) + (orb_end - orb_start,)) * np.nan    # This is 47GB for 16000 orbits
    ice_seasonal_grid = np.zeros(shape + (12,) + (orb_end - orb_start,)) * np.nan
    base_lt_grid = np.broadcast_to(np.linspace(0, 24, num=180), shape)

    for orb_idx, orbit in enumerate(range(orb_start, orb_end)):
        '''This algorithm works by ascribing a LT to each point in the gridded retrievals.
        Then, I loop over LT and take a slice of the LT map. This acts as a mask, from which I can just
        copy the retrieval data.
        '''
        print(orbit)
        orbit_code = f'orbit' + f'{orbit}'.zfill(5)
        block = math.floor(orbit / 100) * 100
        orbit_block = 'orbit' + f'{block}'.zfill(5)

        # Load in the retrievals
        try:
            iuvs_ames_dust = np.load(f'/mnt/science/data/mars/maven/iuvs/retrievals/gcm_grid/ames/{orbit_block}/{orbit_code}-dust.npy')
            iuvs_ames_ice = np.load(f'/mnt/science/data/mars/maven/iuvs/retrievals/gcm_grid/ames/{orbit_block}/{orbit_code}-ice.npy')
        except FileNotFoundError:
            continue

        # Get the LT grid of the observations
        orbit_lon = subsclon[all_orbits==orbit][0]
        orbit_lt = lt[all_orbits==orbit][0]
        roll_amount = int((orbit_lt - orbit_lon / 360 * 24) / 24 * 180)
        orbit_lt_grid = np.roll(base_lt_grid, roll_amount, axis=1)

        for lt_idx, local_time in enumerate(np.arange(12) + 6):
            mask = np.logical_and(local_time <= orbit_lt_grid, orbit_lt_grid <= local_time + 1)
            mask = np.where(mask, mask, np.nan)
            dust_seasonal_grid[:, :, lt_idx, orb_idx] = iuvs_ames_dust * mask
            ice_seasonal_grid[:, :, lt_idx, orb_idx] = iuvs_ames_ice * mask

    np.save('/mnt/science/data/mars/maven/iuvs/retrievals/gcm_grid/ames/dust_seasonal_grid.npy', dust_seasonal_grid)
    np.save('/mnt/science/data/mars/maven/iuvs/retrievals/gcm_grid/ames/ice_seasonal_grid.npy', ice_seasonal_grid)


def make_ames_seaonal_graphic():
    dust_seasonal_grid = np.load('/mnt/science/data/mars/maven/iuvs/retrievals/gcm_grid/ames/dust_seasonal_grid.npy')
    ice_seasonal_grid = np.load('/mnt/science/data/mars/maven/iuvs/retrievals/gcm_grid/ames/ice_seasonal_grid.npy')

    trimmedls = np.rint(ls[3401: 3501])
    orbits_ls = np.logical_and(3400 <= all_orbits, all_orbits <= 3500)
    intls = np.rint(ls[orbits_ls])
    unique_ls = np.unique(intls)

    X = np.linspace(unique_ls[0], unique_ls[-1]+1, num=len(unique_ls)+1)
    Y = np.linspace(90, -90, num=91)

    fig, ax = plt.subplots(4, 1)

    # For 12, the ax index is lt//3, lt%3
    ls_binned_dust = np.zeros(dust_seasonal_grid.shape[:-1] + (len(unique_ls),)) * np.nan
    for c, orblt in enumerate(unique_ls):
        idx = trimmedls == orblt
        ls_binned_dust[..., c] = np.nanmean(dust_seasonal_grid[..., idx], axis=-1)

    for lt in range(4):
        dust_to_plot = np.nanmean(np.nanmean(ls_binned_dust[:, :, lt*3:lt*3+3, :], axis=1), axis=1)
        print(np.nansum(dust_to_plot))
        ax[lt].pcolormesh(X, Y, np.flipud(dust_to_plot), cmap='cividis', vmin=0, vmax=1)   # idk why I need the flipud, but it seems necessary
        ax[lt].set_xlim(unique_ls[0], unique_ls[-1])
        ax[lt].set_ylim(-60, 60)

    plt.savefig('/home/kyle/seasonaldusttestls.png')
    plt.close(fig)

    fig, ax = plt.subplots(4, 1)

    ls_binned_ice = np.zeros(ice_seasonal_grid.shape[:-1] + (len(unique_ls),)) * np.nan
    for c, orblt in enumerate(unique_ls):
        idx = trimmedls == orblt
        ls_binned_ice[..., c] = np.nanmean(ice_seasonal_grid[..., idx], axis=-1)

    for lt in range(4):
        ice_to_plot = np.nanmean(np.nanmean(ls_binned_ice[:, :, lt*3:lt*3+3, :], axis=1), axis=1)
        print(np.nansum(ice_to_plot))
        ax[lt].pcolormesh(X, Y, np.flipud(ice_to_plot), cmap='viridis', vmin=0, vmax=0.2)
        ax[lt].set_xlim(unique_ls[0], unique_ls[-1])
        ax[lt].set_ylim(-60, 60)

    plt.savefig('/home/kyle/seasonalicetestls.png')
    plt.close(fig)


if __name__ == '__main__':
    #make_ames_seasonal_grid(3400, 3500)
    make_ames_seaonal_graphic()

    '''dust_seasonal_grid = np.load('/mnt/science/data/mars/maven/iuvs/retrievals/gcm_grid/ames/dust_seasonal_grid.npy')
    ice_to_plot = np.nanmean(dust_seasonal_grid[:, :, :, 0], axis=1)
    for i in range(12):
        print(np.nanmean(ice_to_plot[..., i]))'''
    '''print(dust_seasonal_grid.shape)
    for c, i in enumerate(range(3400, 3420)):
        print(np.nanmean(dust_seasonal_grid[..., c]))'''


